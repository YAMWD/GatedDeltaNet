// Native csim for gemv_full (MM2S + 32 GEMV lanes + S2MM). Verifies the real
// per-row GEMV output y against a scalar golden (tree order, matching dot16 +
// part banks). Fast correctness gate before the xo/xclbin link.
//   build+run:  make csim_full
#include "gemv_full.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" void gemv_full(
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *, const Pack16 *, const Pack16 *, const Pack16 *,
    const Pack16 *x, Pack16 *y, uint32_t k_packs, uint32_t rows_per_ch);

static float dot16_ref(const Pack16 &w, const Pack16 *x, uint32_t pk) {
    float prod[16];
    for (int i = 0; i < 16; ++i) prod[i] = w.data[i] * x[pk].data[i];
    float s0 = prod[0] + prod[1], s1 = prod[2] + prod[3];
    float s2 = prod[4] + prod[5], s3 = prod[6] + prod[7];
    float s4 = prod[8] + prod[9], s5 = prod[10] + prod[11];
    float s6 = prod[12] + prod[13], s7 = prod[14] + prod[15];
    float a0 = s0 + s1, a1 = s2 + s3, a2 = s4 + s5, a3 = s6 + s7;
    return (a0 + a1) + (a2 + a3);
}

static float reference_row(const Pack16 *w, const Pack16 *x, uint32_t row, uint32_t k_packs) {
    float part[GEMV_TILE_PARTIAL] = {};
    uint32_t groups = k_packs / GEMV_TILE_PARTIAL;
    size_t row_base = (size_t)row * (size_t)k_packs;
    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t base = g * GEMV_TILE_PARTIAL;
        for (uint32_t p = 0; p < GEMV_TILE_PARTIAL; ++p)
            part[p] += dot16_ref(w[row_base + base + p], x, base + p);
    }
    float s0 = part[0] + part[1], s1 = part[2] + part[3];
    float s2 = part[4] + part[5], s3 = part[6] + part[7];
    return (s0 + s1) + (s2 + s3);
}

static float frand(uint32_t s) {
    s = s * 2654435761u + 1013904223u;
    return (float)((s >> 8) & 0xffff) / 65536.0f - 0.5f;
}

int main(int argc, char **argv) {
    uint32_t rows = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 64U;   // rows_per_ch
    uint32_t k_packs = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 16U;

    const int NP = GEMV_FULL_CHANNELS;
    uint32_t opacks = rows / 16;
    std::vector<Pack16> x(k_packs);
    std::vector<Pack16> y((size_t)opacks * NP);
    std::vector<std::vector<Pack16>> w(NP);
    for (int c = 0; c < NP; ++c) w[c].assign((size_t)rows * k_packs, Pack16());

    for (uint32_t i = 0; i < k_packs; ++i)
        for (int l = 0; l < 16; ++l) x[i].data[l] = frand(i * 16 + l);
    for (int c = 0; c < NP; ++c)
        for (size_t i = 0; i < (size_t)rows * k_packs; ++i)
            for (int l = 0; l < 16; ++l)
                w[c][i].data[l] = frand((uint32_t)((c + 1) * 7000000u + i * 16 + l));

    gemv_full(
        w[0].data(),  w[1].data(),  w[2].data(),  w[3].data(),  w[4].data(),  w[5].data(),
        w[6].data(),  w[7].data(),  w[8].data(),  w[9].data(),  w[10].data(), w[11].data(),
        w[12].data(), w[13].data(), w[14].data(), w[15].data(), w[16].data(), w[17].data(),
        w[18].data(), w[19].data(), w[20].data(), w[21].data(), w[22].data(), w[23].data(),
        w[24].data(), w[25].data(), w[26].data(), w[27].data(), w[28].data(), w[29].data(),
        w[30].data(), w[31].data(),
        x.data(), y.data(), k_packs, rows);

    double max_abs = 0.0;
    int bad = 0;
    for (int c = 0; c < NP; ++c)
        for (uint32_t r = 0; r < rows; ++r) {
            float ref = reference_row(w[c].data(), x.data(), r, k_packs);
            // round-robin collector -> interleaved layout: channel c's pack p is at
            // y[p*NP + c]; pack p = rows [p*16 .. p*16+15].
            float got = y[(size_t)(r >> 4) * NP + c].data[r & 15];
            double err = std::fabs((double)got - (double)ref);
            if (err > 1e-3) bad++;
            if (err > max_abs) max_abs = err;
        }
    std::printf("csim_full rows_per_ch=%u k_packs=%u : %s  max_abs=%g  (%d/%u bad)\n",
                rows, k_packs, bad == 0 ? "PASS" : "FAIL", max_abs, bad, rows * NP);
    return bad == 0 ? 0 : 1;
}
