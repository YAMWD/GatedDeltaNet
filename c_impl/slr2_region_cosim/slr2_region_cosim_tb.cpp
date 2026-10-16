// Iter84: RTL cosim of the SLR2 GEMV region (gdn_gemv_part_slr2) for one QKVG call of layer 0,
// on a source copy whose weight-shard constant is scaled so a port holds 16,384 weight beats, the
// 4,096-beat state stripe and the probe tail. The scalars make decay = exp(-0 * softplus) = 1 and
// beta = sigmoid(0) = 0.5 exactly, so the islands' arithmetic is plain IEEE ops and cosim's C/RTL
// comparison of every output (ys words, attention beats, the written state) must be bit-exact;
// weights, activations and state are BF16 values of magnitude in [0.125, 1) (no subnormal path).
// What this checks: the new reader -> writer token (the deferred write-back) in RTL, with the
// AESL deadlock detector, and the queue depth that must now hold a whole layer.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "gdn_model.h"

/* the shape constants live in gdn_model.cpp, not the header */
#define GDN_HIDDEN 2048
#define GDN_HEADS 8
#define GDN_DK 256

typedef ap_axiu<512, 0, 0, 0> GDNAxisBeat;
typedef ap_axiu<64, 0, 0, 0> GDNAxisWord;
void gdn_slr2_region_cosim_top(
    const Beat512 *w24, const Beat512 *w25, const Beat512 *w26, const Beat512 *w27,
    Beat512 *w28, Beat512 *w29, Beat512 *w30, Beat512 *w31,
    hls::stream<GDNAxisBeat> &xr_in, hls::stream<GDNAxisBeat> &conv_in,
    hls::stream<GDNAxisBeat> &scalars_in,
    hls::stream<GDNAxisWord> &ys_out, hls::stream<GDNAxisBeat> &attn_out);

/* a port: the call's weights, layer 0's state stripe, and (probe builds) the record tail after
 * the full 24-layer stripe at beat 16384 + 98304 = 114688 (GDN_PROBE_BEAT_OFF of the scaled header) */
static const uint32_t W_BEATS = 16384, STATE_BEATS = 4096, STRIPE_BEATS = 98304, PORT_BEATS = 115200;
static std::mt19937 rng(20261010);
static uint16_t bf16_mag() {   // magnitude in [0.125, 1), random sign: exponent 124..126, 7 mantissa bits
    const uint32_t sign = (rng() & 1u) << 15, exp = 124u + rng() % 3u, mant = rng() & 0x7fu;
    return (uint16_t)(sign | (exp << 7) | mant);
}
static void fill(Beat512 *dst, size_t beats) {
    for (size_t b = 0; b < beats; ++b) for (int lane = 0; lane < 32; ++lane)
        dst[b].range(16 * lane + 15, 16 * lane) = bf16_mag();
}
static GDNAxisBeat beat(const Beat512 &v, bool last) { GDNAxisBeat b; b.data = v; b.keep = -1; b.strb = -1; b.last = last; return b; }
static void set_f32(Beat512 &b, int lane, float f) { uint32_t u; memcpy(&u, &f, 4); b.range(32 * lane + 31, 32 * lane) = u; }

int main() {
    std::vector<Beat512> w[8];
    for (int p = 0; p < 8; ++p) { w[p].assign(PORT_BEATS, 0); fill(w[p].data(), W_BEATS); }
    for (int p = 4; p < 8; ++p) fill(w[p].data() + W_BEATS, STATE_BEATS);   // the layer-0 state stripe
    hls::stream<GDNAxisBeat> xr_in("xr_in"), conv_in("conv_in"), scalars_in("scalars_in"), attn_out("attn_out");
    hls::stream<GDNAxisWord> ys_out("ys_out");
    const uint32_t k_packs = GDN_HIDDEN / 32;                       // 64 activation beats
    for (uint32_t i = 0; i < k_packs; ++i) { Beat512 v; fill(&v, 1); xr_in.write(beat(v, i + 1 == k_packs)); }
    const uint32_t qkv_beats = GDN_HEADS * (3 * GDN_DK / 32);        // 192 Q/K/V packs
    for (uint32_t i = 0; i < qkv_beats; ++i) { Beat512 v; fill(&v, 1); conv_in.write(beat(v, i + 1 == qkv_beats)); }
    Beat512 s0 = 0, s1 = 0;
    for (int h = 0; h < GDN_HEADS; ++h) {
        set_f32(s0, h, 0.25f * (h + 1));          // a
        set_f32(s0, GDN_HEADS + h, 0.0f);         // b -> beta = sigmoid(0) = 0.5 exactly
        uint32_t ninf = 0xff800000u; float f; memcpy(&f, &ninf, 4);
        set_f32(s1, h, f);                        // a_log = -inf -> exp = 0 -> decay = exp(-0) = 1 exactly
        set_f32(s1, GDN_HEADS + h, 0.5f);         // dt_bias
    }
    scalars_in.write(beat(s0, false)); scalars_in.write(beat(s1, true));
    gdn_slr2_region_cosim_top(w[0].data(), w[1].data(), w[2].data(), w[3].data(),
                              w[4].data(), w[5].data(), w[6].data(), w[7].data(),
                              xr_in, conv_in, scalars_in, ys_out, attn_out);
    // drain and digest the outputs (cosim compares them C vs RTL on its own; the digest is for the log)
    uint64_t ys_x = 0, at_x = 0, st_x = 0; size_t ys_n = 0, at_n = 0;
    while (!ys_out.empty()) { GDNAxisWord v = ys_out.read(); ys_x ^= (uint64_t)v.data; ++ys_n; }
    while (!attn_out.empty()) { GDNAxisBeat v = attn_out.read(); at_x ^= (uint64_t)v.data.range(63, 0); ++at_n; }
    for (int p = 4; p < 8; ++p) for (uint32_t b = 0; b < STATE_BEATS; ++b) st_x ^= (uint64_t)w[p][W_BEATS + b].range(63, 0);
    printf("SLR2_REGION_COSIM ys_words=%zu attn_beats=%zu ys_xor=%016llx attn_xor=%016llx state_xor=%016llx\n",
           ys_n, at_n, (unsigned long long)ys_x, (unsigned long long)at_x, (unsigned long long)st_x);
#ifdef GDN_CALL_PROBE
    /* the SLR2 counter's 32 records of call 0, stored by writer 31 in its probe tail:
     * event[63:56] port[55:48] tag[47:40] cycle[39:0] (the native run prints zeros: no store) */
    for (uint32_t b = 0; b < 4; ++b) {
        const Beat512 &beat = w[7][W_BEATS + STRIPE_BEATS + b];
        for (uint32_t j = 0; j < 8; ++j) {
            const uint64_t v = (uint64_t)beat.range(64 * j + 63, 64 * j);
            if (v) printf("PROBE_REC slot=%u event=%u port=%u tag=%u cycle=%llu\n", 8 * b + j,
                          (unsigned)(v >> 56), (unsigned)((v >> 48) & 0xff), (unsigned)((v >> 40) & 0xff),
                          (unsigned long long)(v & 0xffffffffffULL));
        }
    }
#endif
    return (ys_n == 1024 && at_n == 64) ? 0 : 1;
}
