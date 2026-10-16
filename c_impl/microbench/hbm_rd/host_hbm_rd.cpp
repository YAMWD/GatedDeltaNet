// XRT host for hbm_rd: selects the card by PCIe BDF, times each compute unit alone and all
// together, checks the 64-bit lane sum, and prints GB/s per port.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <xclbin> <bdf> [mib_per_port=256] [reps=16] [rounds=3] [ncu=4]\n", argv[0]); return 2; }
    const std::string xclbin = argv[1], bdf = argv[2];
    const uint64_t mib = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 256;
    const uint32_t reps = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 16;
    const int rounds = argc > 5 ? std::atoi(argv[5]) : 3;
    const int ncu = argc > 6 ? std::atoi(argv[6]) : 4;
    const uint64_t bytes = mib << 20, n_beats = bytes / 64;
    xrt::device dev(bdf);
    auto uuid = dev.load_xclbin(xclbin);
    std::vector<xrt::kernel> ks; std::vector<xrt::bo> ins, outs; std::vector<uint64_t> sum_once;
    std::vector<uint64_t> host(bytes / 8);
    for (int c = 1; c <= ncu; ++c) {
        ks.emplace_back(dev, uuid, "hbm_rd:{hbm_rd_" + std::to_string(c) + "}");
        ins.emplace_back(dev, bytes, ks.back().group_id(0));
        outs.emplace_back(dev, 64, ks.back().group_id(1));
        uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t)c + 12345, sum = 0;
        for (uint64_t i = 0; i < bytes / 8; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; host[i] = x; }
        for (uint64_t b = 0; b < n_beats; ++b) sum += host[b * 8] + host[b * 8 + 7];
        sum_once.push_back(sum);
        ins.back().write(host.data()); ins.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::printf("cu %d: bank group %d, %llu MiB uploaded\n", c, ks.back().group_id(0), (unsigned long long)mib);
    }
    auto check = [&](int c, uint32_t reps_used) {
        uint64_t got = 0; outs[c].sync(XCL_BO_SYNC_BO_FROM_DEVICE); outs[c].read(&got, 8, 0);
        const uint64_t expected = sum_once[c] * (uint64_t)reps_used;
        if (got != expected) { std::printf("cu %d: SUM MISMATCH got %016llx expected %016llx (reps %u)\n", c + 1, (unsigned long long)got, (unsigned long long)expected, reps_used); std::exit(3); }
    };
    using clk = std::chrono::steady_clock;
    for (int c = 0; c < ncu; ++c) { auto r = ks[c](ins[c], outs[c], (uint32_t)n_beats, 1u); r.wait(); check(c, 1u); }   // warm-up
    std::printf("mode single: %llu MiB x %u reps per run\n", (unsigned long long)mib, reps);
    for (int c = 0; c < ncu; ++c) {
        for (int rd = 0; rd < rounds; ++rd) {
            auto t0 = clk::now(); auto r = ks[c](ins[c], outs[c], (uint32_t)n_beats, reps); r.wait(); auto t1 = clk::now();
            const double s = std::chrono::duration<double>(t1 - t0).count();
            std::printf("single cu %d round %d: %.4f s  %.3f GB/s\n", c + 1, rd, s, bytes * (double)reps / s / 1e9);
            check(c, reps);
        }
    }
    std::printf("mode concurrent: all %d cus\n", ncu);
    for (int rd = 0; rd < rounds; ++rd) {
        std::vector<xrt::run> runs; auto t0 = clk::now();
        for (int c = 0; c < ncu; ++c) runs.push_back(ks[c](ins[c], outs[c], (uint32_t)n_beats, reps));
        std::vector<double> ends(ncu);
        for (int c = 0; c < ncu; ++c) { runs[c].wait(); ends[c] = std::chrono::duration<double>(clk::now() - t0).count(); }
        double last = 0; for (double e : ends) last = e > last ? e : last;
        std::printf("concurrent round %d: per cu", rd);
        for (int c = 0; c < ncu; ++c) std::printf("  %.3f GB/s", bytes * (double)reps / ends[c] / 1e9);
        std::printf("  | aggregate %.3f GB/s over %.4f s\n", ncu * bytes * (double)reps / last / 1e9, last);
        for (int c = 0; c < ncu; ++c) check(c, reps);
    }
    std::printf("HBM_RD_DONE\n");
    return 0;
}
