// Host runner for gemv_full: 32 MM2S read channels + 32 systolic GEMV lanes +
// S2MM. Measures sustained HBM weight-read bandwidth AND compute efficiency
// (GFLOP/s), and verifies the output against a scalar golden.
//
// Args (match the kernel): w0..w31 (0..31), x (32), y (33), k_packs (34),
// rows_per_ch (35).
//
//   ./host_gemv_full.exe <xclbin> [kernel=gemv_full] [rows_per_ch=2048]
//                        [k_packs=352] [device=0] [verify=1]
//                        [frequency_mhz=150] [timed_reps=5]
//                        [weights.gdnw]

#include "gemv_full.h"
#include "../../gdn_model.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t NCH = GEMV_FULL_CHANNELS;
static constexpr size_t kSyncChunk = 16U * 1024U * 1024U;
static_assert(sizeof(Pack16) == 64, "Pack16 must be 512 bits");

static void ensure_xrt() {
    if (std::getenv("XILINX_XRT") == nullptr) setenv("XILINX_XRT", "/opt/xilinx/xrt", 0);
}
static size_t align4k(size_t n) { return (n + 4095U) & ~(size_t)4095U; }

static void sync_chunked(xrt::bo &bo, xclBOSyncDirection dir, size_t size) {
    for (size_t off = 0; off < size; off += kSyncChunk)
        bo.sync(dir, std::min(kSyncChunk, size - off), off);
}

static float frand(uint32_t s) {  // matches the csim data generator
    s = s * 2654435761u + 1013904223u;
    return (float)((s >> 8) & 0xffff) / 65536.0f - 0.5f;
}

static void load_layer0_q_proj(const std::string &path,
                               const std::vector<Pack16 *> &wmap,
                               uint32_t rows, uint32_t k_packs) {
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("cannot open weight file: " + path);

    GDNWeightHeader header = {};
    if (std::fread(&header, 1, sizeof(header), file) != sizeof(header)) {
        std::fclose(file);
        throw std::runtime_error("cannot read GDN weight header");
    }
    if (std::memcmp(header.magic, "GDNWv1", 6) != 0 || header.version != 1) {
        std::fclose(file);
        throw std::runtime_error("unsupported GDN weight file");
    }

    const uint32_t expected_rows = header.hidden_size / NCH;
    const uint32_t expected_k_packs = header.hidden_size / GEMV_TILE_LANES;
    if (header.hidden_size % NCH != 0 || header.hidden_size % GEMV_TILE_LANES != 0 ||
        rows != expected_rows || k_packs != expected_k_packs) {
        std::fclose(file);
        throw std::runtime_error("layer-0 q_proj requires rows_per_ch=" +
                                 std::to_string(expected_rows) + " and k_packs=" +
                                 std::to_string(expected_k_packs));
    }

    const uint64_t payload_float_offset =
        (uint64_t)header.vocab_size * header.hidden_size +
        header.hidden_size + 2ULL * header.num_heads;
    const uint64_t q_proj_offset = sizeof(header) + payload_float_offset * sizeof(float);
    const size_t shard_bytes = (size_t)rows * header.hidden_size * sizeof(float);

    for (uint32_t c = 0; c < NCH; ++c) {
        const uint64_t offset = q_proj_offset + (uint64_t)c * shard_bytes;
        if (fseeko(file, (off_t)offset, SEEK_SET) != 0 ||
            std::fread(wmap[c], 1, shard_bytes, file) != shard_bytes) {
            std::fclose(file);
            throw std::runtime_error("cannot read layer-0 q_proj shard " + std::to_string(c));
        }
    }
    std::fclose(file);
}

// Golden row: tree dot16 + GEMV_PARTIAL banks (mirrors the kernel's lane exactly).
static float dot16_ref(const Pack16 &w, const Pack16 *x, uint32_t pk) {
    float p[16];
    for (int i = 0; i < 16; ++i) p[i] = w.data[i] * x[pk].data[i];
    float s0 = p[0] + p[1], s1 = p[2] + p[3], s2 = p[4] + p[5], s3 = p[6] + p[7];
    float s4 = p[8] + p[9], s5 = p[10] + p[11], s6 = p[12] + p[13], s7 = p[14] + p[15];
    float a0 = s0 + s1, a1 = s2 + s3, a2 = s4 + s5, a3 = s6 + s7;
    return (a0 + a1) + (a2 + a3);
}
static float reference_row(const Pack16 *w, const Pack16 *x, uint32_t row, uint32_t k_packs) {
    float part[GEMV_TILE_PARTIAL] = {};
    uint32_t groups = k_packs / GEMV_TILE_PARTIAL;
    size_t rb = (size_t)row * k_packs;
    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t base = g * GEMV_TILE_PARTIAL;
        for (uint32_t p = 0; p < GEMV_TILE_PARTIAL; ++p)
            part[p] += dot16_ref(w[rb + base + p], x, base + p);
    }
    float s0 = part[0] + part[1], s1 = part[2] + part[3];
    float s2 = part[4] + part[5], s3 = part[6] + part[7];
    return (s0 + s1) + (s2 + s3);
}

int main(int argc, char **argv) {
    ensure_xrt();
    if (argc < 2) { std::cerr << "usage: " << argv[0]
        << " <xclbin> [kernel] [rows_per_ch] [k_packs] [device] [verify]"
           " [frequency_mhz] [timed_reps] [weights.gdnw]\n"; return 1; }
    std::string xclbin = argv[1];
    std::string kname = (argc > 2) ? argv[2] : "gemv_full";
    uint32_t rows = (argc > 3) ? (uint32_t)std::stoul(argv[3]) : 2048U;
    uint32_t k_packs = (argc > 4) ? (uint32_t)std::stoul(argv[4]) : 352U;
    uint32_t dev = (argc > 5) ? (uint32_t)std::stoul(argv[5]) : 0U;
    bool verify = (argc > 6) ? (std::stoul(argv[6]) != 0) : true;
    double frequency_mhz = (argc > 7) ? std::stod(argv[7]) : 150.0;
    uint32_t timed_reps = (argc > 8) ? (uint32_t)std::stoul(argv[8]) : 5U;
    std::string weight_file = (argc > 9) ? argv[9] : "";
    if (timed_reps == 0) throw std::runtime_error("timed_reps must be positive");

    const uint32_t opacks = rows / 16;
    const size_t w_bytes = (size_t)rows * k_packs * sizeof(Pack16);       // per channel
    const size_t x_bytes = (size_t)k_packs * sizeof(Pack16);
    const size_t y_bytes = (size_t)opacks * NCH * sizeof(Pack16);

    const double weight_read_bytes = (double)NCH * (double)w_bytes;       // read once/call
    const double macs = (double)NCH * rows * k_packs * 16.0;              // 16-wide per pack
    const double flops = 2.0 * macs;                                      // mul + add

    std::cout << "gemv_full on-card benchmark\n"
              << "  shape       : 32 ch x rows_per_ch=" << rows << " x k_packs=" << k_packs
              << "  (in_dim=" << k_packs*16 << ", out=" << NCH*rows << ")\n"
              << "  weight read : " << weight_read_bytes/1e9 << " GB\n"
              << "  weights     : " << (weight_file.empty() ? "synthetic" : weight_file) << "\n"
              << "  MACs        : " << macs/1e9 << " G  (" << flops/1e9 << " GFLOP)\n";

    xrt::device device(dev);
    auto uuid = device.load_xclbin(xclbin);
    xrt::kernel krnl(device, uuid, kname + ":{" + kname + "_1}");

    // --- allocate + fill: 32 weight BOs, x BO, y BO ---
    std::vector<xrt::bo> wbo(NCH);
    std::vector<Pack16 *> wmap(NCH);
    for (uint32_t c = 0; c < NCH; ++c) {
        wbo[c] = xrt::bo(device, align4k(w_bytes), krnl.group_id(c));
        wmap[c] = wbo[c].map<Pack16 *>();
        if (weight_file.empty()) {
            for (size_t i = 0; i < (size_t)rows * k_packs; ++i)
                for (int l = 0; l < 16; ++l)
                    wmap[c][i].data[l] = frand((uint32_t)((c + 1) * 7000000u + i * 16 + l));
        }
    }
    if (!weight_file.empty()) load_layer0_q_proj(weight_file, wmap, rows, k_packs);
    for (uint32_t c = 0; c < NCH; ++c) {
        sync_chunked(wbo[c], XCL_BO_SYNC_BO_TO_DEVICE, align4k(w_bytes));
    }
    xrt::bo xbo(device, align4k(x_bytes), krnl.group_id(32));
    Pack16 *xmap = xbo.map<Pack16 *>();
    for (uint32_t i = 0; i < k_packs; ++i)
        for (int l = 0; l < 16; ++l) xmap[i].data[l] = frand(i * 16 + l);
    sync_chunked(xbo, XCL_BO_SYNC_BO_TO_DEVICE, align4k(x_bytes));
    xrt::bo ybo(device, align4k(y_bytes), krnl.group_id(33));

    xrt::run run(krnl);
    for (uint32_t c = 0; c < NCH; ++c) run.set_arg(c, wbo[c]);
    run.set_arg(32, xbo);
    run.set_arg(33, ybo);
    run.set_arg(34, k_packs);
    run.set_arg(35, rows);

    // Warm up once, then report the median to suppress launch-time noise.
    run.start();
    run.wait();

    std::vector<double> samples;
    samples.reserve(timed_reps);
    for (uint32_t rep = 0; rep < timed_reps; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run.start();
        run.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    double sec = samples[samples.size() / 2];

    double gbps = weight_read_bytes / sec / 1e9;
    double gflops = flops / sec / 1e9;
    double kernel_ceiling_gbps = NCH * sizeof(Pack16) * frequency_mhz / 1000.0;
    std::cout << std::fixed << std::setprecision(3)
              << "  median sec  : " << sec << "  (" << timed_reps << " timed runs)\n"
              << "  bandwidth   : " << gbps << " GB/s  (" << gbps / NCH << " GB/s/ch, "
              << 100.0 * gbps / kernel_ceiling_gbps << "% of "
              << kernel_ceiling_gbps << " GB/s kernel ceiling)\n"
              << "  throughput  : " << gflops << " GFLOP/s\n"
              << "  intensity   : " << flops / weight_read_bytes << " FLOP/byte\n";

    if (verify) {
        sync_chunked(ybo, XCL_BO_SYNC_BO_FROM_DEVICE, align4k(y_bytes));
        Pack16 *ymap = ybo.map<Pack16 *>();
        double max_abs = 0.0; long bad = 0;
        for (uint32_t c = 0; c < NCH; ++c)
            for (uint32_t r = 0; r < rows; ++r) {
                float ref = reference_row(wmap[c], xmap, r, k_packs);
                float got = ymap[(size_t)(r >> 4) * NCH + c].data[r & 15];  // interleaved layout
                double e = std::fabs((double)got - (double)ref);
                if (e > 1e-2) bad++;
                if (e > max_abs) max_abs = e;
            }
        std::cout << "  verify      : " << (bad == 0 ? "PASS" : "FAIL")
                  << "  max_abs=" << max_abs << "  bad=" << bad << "\n";
        if (bad) return 1;
    }
    return 0;
}
