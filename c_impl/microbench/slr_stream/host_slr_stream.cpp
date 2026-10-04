// Iter79 step 0a host: three source->sink pairs in different SLR combinations.
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
static const char *PAIR_DESC[4] = {"", "SLR0->SLR1", "SLR1->SLR2", "SLR0->SLR2"};
int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <xclbin> [device|bdf] [beats] [reps] [timed_reps] [freq_mhz]\n"; return 2; }
    const std::string xclbin = argv[1];
    const std::string dev_arg = argc > 2 ? argv[2] : "0";
    const uint32_t beats = argc > 3 ? (uint32_t)std::stoul(argv[3]) : (1u << 20);
    const uint32_t reps = argc > 4 ? (uint32_t)std::stoul(argv[4]) : 4u;
    const uint32_t timed_reps = argc > 5 ? (uint32_t)std::stoul(argv[5]) : 5u;
    const double freq = argc > 6 ? std::stod(argv[6]) : 200.0;
    xrt::device device = dev_arg.find(':') != std::string::npos ? xrt::device(dev_arg) : xrt::device(std::stoi(dev_arg));
    auto uuid = device.load_xclbin(xclbin);
    std::vector<uint64_t> expected(8, 0);
    for (uint32_t r = 0; r < reps; ++r)
        for (uint32_t i = 0; i < beats; ++i)
            for (int k = 0; k < 8; ++k)
                expected[k] ^= ((uint64_t)r << 48) ^ ((uint64_t)i * 8 + k) ^ 0x9E3779B97F4A7C15ULL;
    std::vector<xrt::kernel> ksrc, ksink; std::vector<xrt::bo> obo;
    for (int p = 1; p <= 3; ++p) {
        ksrc.emplace_back(device, uuid, "slr_src:{slr_src_" + std::to_string(p) + "}");
        ksink.emplace_back(device, uuid, "slr_sink:{slr_sink_" + std::to_string(p) + "}");
        obo.emplace_back(device, 16 * 8, ksink.back().group_id(1));  // arg 0 is the AXIS port
    }
    int fails = 0;
    auto run_pairs = [&](const std::vector<int> &pairs) -> double {
        std::vector<xrt::run> rs;
        auto t0 = std::chrono::steady_clock::now();
        for (int p : pairs) { auto r = xrt::run(ksink[p - 1]); r.set_arg(1, obo[p - 1]); r.set_arg(2, beats); r.set_arg(3, reps); r.start(); rs.push_back(r); }
        for (int p : pairs) { auto r = xrt::run(ksrc[p - 1]); r.set_arg(1, beats); r.set_arg(2, reps); r.start(); rs.push_back(r); }
        for (auto &r : rs) r.wait();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    };
    for (int p = 1; p <= 3; ++p) {
        std::vector<double> ts;
        for (uint32_t t = 0; t < timed_reps; ++t) ts.push_back(run_pairs({p}));
        obo[p - 1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const uint64_t *o = obo[p - 1].map<const uint64_t *>();
        bool ok = o[8] == reps && o[9] == (uint64_t)beats * reps;
        for (int k = 0; k < 8; ++k) ok = ok && o[k] == expected[k];
        if (!ok) fails++;
        std::sort(ts.begin(), ts.end());
        const double med = ts[ts.size() / 2];
        const double nb = (double)beats * reps;
        std::cout << "pair " << p << " " << PAIR_DESC[p] << " : verify " << (ok ? "PASS" : "FAIL")
                  << " lasts=" << o[8] << " total=" << o[9] << " | median " << med * 1e3 << " ms, "
                  << nb * 64 / med / 1e9 << " GB/s, " << nb / (med * freq * 1e6) << " beats/cycle at " << freq << " MHz\n";
    }
    std::vector<double> ts;
    for (uint32_t t = 0; t < timed_reps; ++t) ts.push_back(run_pairs({1, 2, 3}));
    std::sort(ts.begin(), ts.end());
    const double med = ts[ts.size() / 2]; const double nb = (double)beats * reps;
    std::cout << "all three concurrently : median " << med * 1e3 << " ms, " << nb / (med * freq * 1e6) << " beats/cycle per stream\n";
    std::cout << "verify      : " << (fails ? "FAIL" : "PASS") << " (" << 3 - fails << "/3 pairs)\n";
    return fails ? 1 : 0;
}
