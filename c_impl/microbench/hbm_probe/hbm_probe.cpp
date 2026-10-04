/* Iter78 step 0b: HBM read-bandwidth probe.  Thirty-two HLS m_axi readers,
 * one per pseudo-channel, each streaming `beats` sequential 512-bit words
 * `reps` times and XOR-folding them so the reads cannot be removed; no
 * compute, no floorplan.  It measures what the production design's readers
 * can get from HBM at a given kernel clock -- at 200 MHz a port demands
 * 12.8 GB/s, 89% of its pseudo-channel peak -- with the same adapter
 * settings as the production ports 1..31 (burst 64, PROBE_READ_OUTSTANDING
 * outstanding, default 4). */
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

typedef ap_uint<512> Beat512;

#ifndef PROBE_READ_OUTSTANDING
#define PROBE_READ_OUTSTANDING 4
#endif
#define PROBE_PRAGMA_(d) _Pragma(#d)
#define PROBE_PRAGMA(d) PROBE_PRAGMA_(d)
/* Parameter names must not collide with the pragma's own keywords: with
 * parameters named `port`/`bundle`, `port=port` expands to `w0=w0` and HLS
 * silently drops the pragma into the default `gmem` bundle (build 5254). */
#define PROBE_READ_AXI(port_name, bundle_name) \
    PROBE_PRAGMA(HLS interface m_axi port=port_name offset=slave bundle=bundle_name \
                 max_widen_bitwidth=512 max_read_burst_length=64 \
                 num_read_outstanding=PROBE_READ_OUTSTANDING)

template <int PORT>
static void probe_reader(const Beat512 *w, uint32_t beats, uint32_t reps,
                         hls::stream<ap_uint<64> > &sum) {
#pragma HLS inline off
    ap_uint<64> acc = 0;
probe_rep: for (uint32_t r = 0; r < reps; ++r) {
#pragma HLS loop_tripcount min=1 max=8
    probe_beat: for (uint32_t i = 0; i < beats; ++i) {
#pragma HLS loop_tripcount min=1366528 max=1366528
#pragma HLS pipeline II=1
            const Beat512 v = w[i];
            acc ^= (ap_uint<64>)v.range(63, 0) ^ (ap_uint<64>)v.range(511, 448)
                   ^ (ap_uint<64>)i;
        }
    }
    sum.write(acc);
}

static void probe_collect(hls::stream<ap_uint<64> > &s0, hls::stream<ap_uint<64> > &s1,
                          hls::stream<ap_uint<64> > &s2, hls::stream<ap_uint<64> > &s3,
                          hls::stream<ap_uint<64> > &s4, hls::stream<ap_uint<64> > &s5,
                          hls::stream<ap_uint<64> > &s6, hls::stream<ap_uint<64> > &s7,
                          hls::stream<ap_uint<64> > &s8, hls::stream<ap_uint<64> > &s9,
                          hls::stream<ap_uint<64> > &s10, hls::stream<ap_uint<64> > &s11,
                          hls::stream<ap_uint<64> > &s12, hls::stream<ap_uint<64> > &s13,
                          hls::stream<ap_uint<64> > &s14, hls::stream<ap_uint<64> > &s15,
                          hls::stream<ap_uint<64> > &s16, hls::stream<ap_uint<64> > &s17,
                          hls::stream<ap_uint<64> > &s18, hls::stream<ap_uint<64> > &s19,
                          hls::stream<ap_uint<64> > &s20, hls::stream<ap_uint<64> > &s21,
                          hls::stream<ap_uint<64> > &s22, hls::stream<ap_uint<64> > &s23,
                          hls::stream<ap_uint<64> > &s24, hls::stream<ap_uint<64> > &s25,
                          hls::stream<ap_uint<64> > &s26, hls::stream<ap_uint<64> > &s27,
                          hls::stream<ap_uint<64> > &s28, hls::stream<ap_uint<64> > &s29,
                          hls::stream<ap_uint<64> > &s30, hls::stream<ap_uint<64> > &s31,
                          Beat512 *out) {
#pragma HLS inline off
    Beat512 w0 = 0, w1 = 0, w2 = 0, w3 = 0;
#define PROBE_LANE(word, k, s) word.range(64 * (k) + 63, 64 * (k)) = s.read()
    PROBE_LANE(w0, 0, s0); PROBE_LANE(w0, 1, s1); PROBE_LANE(w0, 2, s2); PROBE_LANE(w0, 3, s3);
    PROBE_LANE(w0, 4, s4); PROBE_LANE(w0, 5, s5); PROBE_LANE(w0, 6, s6); PROBE_LANE(w0, 7, s7);
    PROBE_LANE(w1, 0, s8); PROBE_LANE(w1, 1, s9); PROBE_LANE(w1, 2, s10); PROBE_LANE(w1, 3, s11);
    PROBE_LANE(w1, 4, s12); PROBE_LANE(w1, 5, s13); PROBE_LANE(w1, 6, s14); PROBE_LANE(w1, 7, s15);
    PROBE_LANE(w2, 0, s16); PROBE_LANE(w2, 1, s17); PROBE_LANE(w2, 2, s18); PROBE_LANE(w2, 3, s19);
    PROBE_LANE(w2, 4, s20); PROBE_LANE(w2, 5, s21); PROBE_LANE(w2, 6, s22); PROBE_LANE(w2, 7, s23);
    PROBE_LANE(w3, 0, s24); PROBE_LANE(w3, 1, s25); PROBE_LANE(w3, 2, s26); PROBE_LANE(w3, 3, s27);
    PROBE_LANE(w3, 4, s28); PROBE_LANE(w3, 5, s29); PROBE_LANE(w3, 6, s30); PROBE_LANE(w3, 7, s31);
#undef PROBE_LANE
    out[0] = w0; out[1] = w1; out[2] = w2; out[3] = w3;
}

extern "C" void hbm_probe(
    Beat512 *out,
    const Beat512 *w0, const Beat512 *w1, const Beat512 *w2, const Beat512 *w3,
    const Beat512 *w4, const Beat512 *w5, const Beat512 *w6, const Beat512 *w7,
    const Beat512 *w8, const Beat512 *w9, const Beat512 *w10, const Beat512 *w11,
    const Beat512 *w12, const Beat512 *w13, const Beat512 *w14, const Beat512 *w15,
    const Beat512 *w16, const Beat512 *w17, const Beat512 *w18, const Beat512 *w19,
    const Beat512 *w20, const Beat512 *w21, const Beat512 *w22, const Beat512 *w23,
    const Beat512 *w24, const Beat512 *w25, const Beat512 *w26, const Beat512 *w27,
    const Beat512 *w28, const Beat512 *w29, const Beat512 *w30, const Beat512 *w31,
    uint32_t beats, uint32_t reps) {
#pragma HLS interface m_axi port=out offset=slave bundle=mem0 max_widen_bitwidth=512 max_write_burst_length=16 num_write_outstanding=4
    PROBE_READ_AXI(w0, mem0)   PROBE_READ_AXI(w1, mem1)   PROBE_READ_AXI(w2, mem2)   PROBE_READ_AXI(w3, mem3)
    PROBE_READ_AXI(w4, mem4)   PROBE_READ_AXI(w5, mem5)   PROBE_READ_AXI(w6, mem6)   PROBE_READ_AXI(w7, mem7)
    PROBE_READ_AXI(w8, mem8)   PROBE_READ_AXI(w9, mem9)   PROBE_READ_AXI(w10, mem10) PROBE_READ_AXI(w11, mem11)
    PROBE_READ_AXI(w12, mem12) PROBE_READ_AXI(w13, mem13) PROBE_READ_AXI(w14, mem14) PROBE_READ_AXI(w15, mem15)
    PROBE_READ_AXI(w16, mem16) PROBE_READ_AXI(w17, mem17) PROBE_READ_AXI(w18, mem18) PROBE_READ_AXI(w19, mem19)
    PROBE_READ_AXI(w20, mem20) PROBE_READ_AXI(w21, mem21) PROBE_READ_AXI(w22, mem22) PROBE_READ_AXI(w23, mem23)
    PROBE_READ_AXI(w24, mem24) PROBE_READ_AXI(w25, mem25) PROBE_READ_AXI(w26, mem26) PROBE_READ_AXI(w27, mem27)
    PROBE_READ_AXI(w28, mem28) PROBE_READ_AXI(w29, mem29) PROBE_READ_AXI(w30, mem30) PROBE_READ_AXI(w31, mem31)
#pragma HLS interface s_axilite port=beats
#pragma HLS interface s_axilite port=reps
#pragma HLS interface s_axilite port=return
    hls::stream<ap_uint<64> > s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
        s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31;
#pragma HLS dataflow
    probe_reader<0>(w0, beats, reps, s0);    probe_reader<1>(w1, beats, reps, s1);
    probe_reader<2>(w2, beats, reps, s2);    probe_reader<3>(w3, beats, reps, s3);
    probe_reader<4>(w4, beats, reps, s4);    probe_reader<5>(w5, beats, reps, s5);
    probe_reader<6>(w6, beats, reps, s6);    probe_reader<7>(w7, beats, reps, s7);
    probe_reader<8>(w8, beats, reps, s8);    probe_reader<9>(w9, beats, reps, s9);
    probe_reader<10>(w10, beats, reps, s10); probe_reader<11>(w11, beats, reps, s11);
    probe_reader<12>(w12, beats, reps, s12); probe_reader<13>(w13, beats, reps, s13);
    probe_reader<14>(w14, beats, reps, s14); probe_reader<15>(w15, beats, reps, s15);
    probe_reader<16>(w16, beats, reps, s16); probe_reader<17>(w17, beats, reps, s17);
    probe_reader<18>(w18, beats, reps, s18); probe_reader<19>(w19, beats, reps, s19);
    probe_reader<20>(w20, beats, reps, s20); probe_reader<21>(w21, beats, reps, s21);
    probe_reader<22>(w22, beats, reps, s22); probe_reader<23>(w23, beats, reps, s23);
    probe_reader<24>(w24, beats, reps, s24); probe_reader<25>(w25, beats, reps, s25);
    probe_reader<26>(w26, beats, reps, s26); probe_reader<27>(w27, beats, reps, s27);
    probe_reader<28>(w28, beats, reps, s28); probe_reader<29>(w29, beats, reps, s29);
    probe_reader<30>(w30, beats, reps, s30); probe_reader<31>(w31, beats, reps, s31);
    probe_collect(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
                  s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, out);
}
