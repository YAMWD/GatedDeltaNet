// Host for the HBM read probe: fills 32 per-bank buffers with a known
// pattern, runs the kernel, checks the XOR folds, and reports GB/s per port.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int NPORTS = 32;  // Iter79: ports in the image, argv[6] (default 32)
static const size_t kSyncChunk = 8u << 20;  // 8 MiB: larger syncs at nonzero offsets return EINVAL on this XRT

static void sync_chunked(xrt::bo &bo, xclBOSyncDirection dir, size_t size) {
    for (size_t off = 0; off < size; off += kSyncChunk)
        bo.sync(dir, std::min(kSyncChunk, size - off), off);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <xclbin> [device|bdf] [beats] [reps] [timed_reps] [nports]\n";
        return 2;
    }
    const std::string xclbin = argv[1];
    const std::string dev_arg = argc > 2 ? argv[2] : "0";
    const uint32_t beats = argc > 3 ? (uint32_t)std::stoul(argv[3]) : 1366528u;
    const uint32_t reps = argc > 4 ? (uint32_t)std::stoul(argv[4]) : 4u;
    const uint32_t timed_reps = argc > 5 ? (uint32_t)std::stoul(argv[5]) : 9u;
    NPORTS = argc > 6 ? std::stoi(argv[6]) : 32;
    const size_t bytes = (size_t)beats * 64;

    xrt::device device = dev_arg.find(':') != std::string::npos
        ? xrt::device(dev_arg) : xrt::device((unsigned)std::stoul(dev_arg));
    auto uuid = device.load_xclbin(xclbin);
    xrt::kernel krnl(device, uuid, "hbm_probe:{hbm_probe_1}");

    std::vector<xrt::bo> wbo(NPORTS);
    std::vector<uint64_t> expected(NPORTS, 0);
    std::vector<uint64_t> line(8);
    for (int p = 0; p < NPORTS; ++p) {
        wbo[p] = xrt::bo(device, bytes, krnl.group_id(1 + p));
        uint64_t *m = wbo[p].map<uint64_t *>();
        uint64_t acc = 0;
        for (uint32_t i = 0; i < beats; ++i) {
            for (int k = 0; k < 8; ++k)
                line[k] = ((uint64_t)i * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)p << 56) ^ ((uint64_t)k << 48);
            std::memcpy(m + (size_t)i * 8, line.data(), 64);
            acc ^= line[0] ^ line[7] ^ (uint64_t)i;
        }
        for (uint32_t r = 1; r < reps; ++r) acc ^= acc;  // XOR of identical passes cancels in pairs
        // the kernel folds every pass: odd rep counts leave one pass, even leave zero
        expected[p] = (reps & 1) ? acc : 0;
        sync_chunked(wbo[p], XCL_BO_SYNC_BO_TO_DEVICE, bytes);
    }
    xrt::bo obo(device, 4 * 64, krnl.group_id(0));

    auto make_run = [&]() {
        xrt::run run(krnl);
        run.set_arg(0, obo);
        for (int p = 0; p < NPORTS; ++p) run.set_arg(1 + p, wbo[p]);
        run.set_arg(1 + NPORTS, beats);
        run.set_arg(2 + NPORTS, reps);
        return run;
    };
    auto run = make_run();
    run.start(); run.wait();
    obo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const uint64_t *sums = obo.map<const uint64_t *>();
    int bad = 0;
    for (int p = 0; p < NPORTS; ++p)
        if (sums[p] != expected[p]) { ++bad; std::cerr << "port " << p << " fold mismatch\n"; }
    std::cout << "verify      : " << (bad ? "FAIL" : "PASS") << " (" << NPORTS - bad << "/" << NPORTS << " ports)\n";

    std::vector<double> samples;
    for (uint32_t t = 0; t < timed_reps; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run.start(); run.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    const double med = samples[samples.size() / 2];
    const double total_bytes = (double)bytes * reps * NPORTS;
    const double gbps = total_bytes / med / 1e9;
    std::cout << "beats/port  : " << beats << " x " << reps << " reps (" << bytes * reps / 1e6 << " MB/port)\n"
              << "median time : " << med * 1e3 << " ms  (min " << samples.front() * 1e3
              << ", max " << samples.back() * 1e3 << ", n=" << samples.size() << ")\n"
              << "bandwidth   : " << gbps << " GB/s total, " << gbps / NPORTS << " GB/s/port\n"
              << "beat rate   : " << (double)beats * reps / med / 1e6 << " Mbeat/s/port"
              << " (200 MHz demand = 200.0, 150 MHz = 150.0)\n";
    return bad ? 1 : 0;
}
