// Iter86: RTL cosim of the control kernel's GEMV region (gdn_gemv_part_ctrl) for one QKVG call of
// layer 0. Activations and the remote ys words are zero, so every Q/K/V/gate value is 0: the
// convolution's silu(0) = 0 and the gate's sigmoid(0) = 0.5 are exact in C and RTL and cosim's
// comparison of every output (attn_out, the conv/scalar links, the workspace with the written
// tails) must be bit-exact. The conv tails and the attention beats are random BF16 values of
// magnitude in [0.125, 1), so the tails writer's addresses and the norm's reduction are
// exercised with real data. What this checks: the region's process graph with the new
// gdn_attn_norm_gate and gdn_tails_writer processes (AESL deadlock detector) and the mm0 writer.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "gdn_model.h"
#define GDN_HIDDEN 2048
#define GDN_HEADS 8
#define GDN_DK 256
#define GDN_DV 256
#define GDN_CONV 4
#define GDN_INTER 5632
typedef ap_axiu<512, 0, 0, 0> GDNAxisBeat;
typedef ap_axiu<64, 0, 0, 0> GDNAxisWord;
void gdn_ctrl_region_cosim_top(
    Beat512 out[2 * GDN_INTER / 32], hls::stream<Beat512> &logits_stream, const Beat512 in[GDN_HIDDEN / 32],
    const Beat512 *w0, const Beat512 *w1,
    const Beat512 *w18, const Beat512 *w19, const Beat512 *w20, const Beat512 *w21,
    const Beat512 *w22, const Beat512 *w23,
    Beat512 *workspace,
    hls::stream<GDNAxisBeat> &xr_to_slr0, hls::stream<GDNAxisBeat> &xr_to_slr2,
    hls::stream<GDNAxisWord> &ys_from_slr0, hls::stream<GDNAxisWord> &ys_from_slr2,
    hls::stream<GDNAxisBeat> &conv_to_slr2, hls::stream<GDNAxisBeat> &attn_from_slr2,
    hls::stream<GDNAxisBeat> &scalars_to_slr2,
    Beat512 attn_out[GDN_HIDDEN / 32], const Beat512 onorm_w[GDN_DV / 32],
    const Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 32],
    Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32],
    const float a[GDN_HEADS], const float b[GDN_HEADS],
    const float layer_a_log[GDN_HEADS], const float layer_dt_bias[GDN_HEADS]);

static const uint32_t W_BEATS = 16384, WS_BEATS = 818176;
static std::mt19937 rng(20261011);
static uint16_t bf16_mag() { const uint32_t sign = (rng() & 1u) << 15, exp = 124u + rng() % 3u, mant = rng() & 0x7fu; return (uint16_t)(sign | (exp << 7) | mant); }
static void fill(Beat512 *dst, size_t beats) { for (size_t b = 0; b < beats; ++b) for (int lane = 0; lane < 32; ++lane) dst[b].range(16 * lane + 15, 16 * lane) = bf16_mag(); }
static GDNAxisBeat beat(const Beat512 &v, bool last) { GDNAxisBeat x; x.data = v; x.keep = -1; x.strb = -1; x.last = last; return x; }
static GDNAxisWord word(uint64_t v, bool last) { GDNAxisWord x; x.data = v; x.keep = -1; x.strb = -1; x.last = last; return x; }

int main() {
    std::vector<Beat512> w[8];
    for (int p = 0; p < 8; ++p) { w[p].assign(W_BEATS, 0); fill(w[p].data(), W_BEATS); }
    std::vector<Beat512> workspace(WS_BEATS, 0);
    std::vector<Beat512> in(GDN_HIDDEN / 32, 0);                 // zero activations -> zero Q/K/V/gate
    std::vector<Beat512> out(2 * GDN_INTER / 32, 0), attn_out(GDN_HIDDEN / 32, 0);
    Beat512 onorm_w[GDN_DV / 32]; fill(onorm_w, GDN_DV / 32);
    static Beat512 conv_weights[3][(GDN_HIDDEN * GDN_CONV) / 32];
    static Beat512 conv_tails[3][((GDN_CONV - 1) * GDN_HIDDEN) / 32];
    for (int k = 0; k < 3; ++k) { fill(conv_weights[k], (GDN_HIDDEN * GDN_CONV) / 32); fill(conv_tails[k], ((GDN_CONV - 1) * GDN_HIDDEN) / 32); }
    float a[GDN_HEADS], b[GDN_HEADS], a_log[GDN_HEADS], dt_bias[GDN_HEADS];
    for (int h = 0; h < GDN_HEADS; ++h) { a[h] = 0.25f * (h + 1); b[h] = 0.0f; a_log[h] = -1.0f; dt_bias[h] = 0.5f; }
    hls::stream<Beat512> logits_stream("logits");
    hls::stream<GDNAxisBeat> xr_to_slr0("xr0"), xr_to_slr2("xr2"), conv_to_slr2("conv"), attn_from_slr2("attn"), scalars_to_slr2("scal");
    hls::stream<GDNAxisWord> ys_from_slr0("ys0"), ys_from_slr2("ys2");
    const uint32_t row_groups = (4u * GDN_HIDDEN / 32u) / 8u;        // 32
    for (uint32_t i = 0; i < row_groups * 64u; ++i) ys_from_slr0.write(word(0, i + 1 == row_groups * 64u));
    for (uint32_t i = 0; i < row_groups * 32u; ++i) ys_from_slr2.write(word(0, i + 1 == row_groups * 32u));
    for (uint32_t i = 0; i < GDN_HIDDEN / 32; ++i) { Beat512 v; fill(&v, 1); attn_from_slr2.write(beat(v, i + 1 == GDN_HIDDEN / 32)); }
    gdn_ctrl_region_cosim_top(out.data(), logits_stream, in.data(),
                              w[0].data(), w[1].data(), w[2].data(), w[3].data(), w[4].data(), w[5].data(), w[6].data(), w[7].data(),
                              workspace.data(), xr_to_slr0, xr_to_slr2, ys_from_slr0, ys_from_slr2,
                              conv_to_slr2, attn_from_slr2, scalars_to_slr2,
                              attn_out.data(), onorm_w, conv_weights, conv_tails, a, b, a_log, dt_bias);
    size_t x0 = 0, x2 = 0, cv = 0, sc = 0; uint64_t cv_x = 0, at_x = 0, ws_x = 0;
    while (!xr_to_slr0.empty()) { xr_to_slr0.read(); ++x0; }
    while (!xr_to_slr2.empty()) { xr_to_slr2.read(); ++x2; }
    while (!conv_to_slr2.empty()) { GDNAxisBeat v = conv_to_slr2.read(); cv_x ^= (uint64_t)v.data.range(63, 0); ++cv; }
    while (!scalars_to_slr2.empty()) { scalars_to_slr2.read(); ++sc; }
    while (!logits_stream.empty()) logits_stream.read();
    for (uint32_t i = 0; i < GDN_HIDDEN / 32; ++i) at_x ^= (uint64_t)attn_out[i].range(63, 0);
    const size_t hb = (size_t)GDN_WS_OFF_HEAD_BUF / 16;
    size_t nonzero = 0;
    for (size_t i = hb; i < hb + 3 * 384; ++i) { ws_x ^= (uint64_t)workspace[i].range(63, 0); if (workspace[i] != 0) ++nonzero; }
    printf("CTRL_REGION_COSIM xr0=%zu xr2=%zu conv=%zu scal=%zu conv_xor=%016llx attn_xor=%016llx tails_xor=%016llx tails_nonzero=%zu\n",
           x0, x2, cv, sc, (unsigned long long)cv_x, (unsigned long long)at_x, (unsigned long long)ws_x, nonzero);
    return (x0 == 64 && x2 == 64 && cv == 192 && sc == 2 && nonzero == 2 * 3 * 64) ? 0 : 1;   // old rows 1/2 random, the new rows zero
}
