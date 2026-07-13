// COMPLETE SINGLE-KERNEL MM2S + 8 x 4-channel GEMV CLUSTERS + S2MM.
//
// The full pipeline, one kernel, one dataflow region:
//   32 HBM-facing MM2S loaders: read w_k on gmem_k -> local weight streams.
//   8 GEMV clusters: each consumes 4 weight streams and shares one local x copy.
//   collector: merge the 8 cluster streams into ONE result stream.
//   s2mm writer: drain the result stream -> HBM burst-write.
//
// Master budget: 32 full-duplex masters. gmem0 reads w0 + x (read channels) AND
// writes y (write channel) -- the S2MM output rides gmem0's FREE write channel,
// so there is NO 33rd master. gmem0 is read by ONE process (loader0) and written
// by ONE process (s2mm) -> legal. gmem1..31 are read-only. Total = 32. Fits.

#include "gemv_full.h"

#include "hls_stream.h"

#include <stddef.h>

#define NCH GEMV_FULL_CHANNELS
#define NCL GEMV_FULL_CLUSTERS
#define CH_PER_CLUSTER GEMV_FULL_CHANNELS_PER_CLUSTER

#ifndef GEMV_READ_OUTSTANDING
#define GEMV_READ_OUTSTANDING 16
#endif

#define GEMV_DO_PRAGMA_(directive) _Pragma(#directive)
#define GEMV_DO_PRAGMA(directive) GEMV_DO_PRAGMA_(directive)
#define GEMV_READ_AXI(port_name, bundle_name) \
    GEMV_DO_PRAGMA(HLS interface m_axi port=port_name offset=slave \
                   bundle=bundle_name max_widen_bitwidth=512 \
                   max_read_burst_length=64 \
                   num_read_outstanding=GEMV_READ_OUTSTANDING)
#define GEMV_READ_WRITE_AXI(port_name, bundle_name) \
    GEMV_DO_PRAGMA(HLS interface m_axi port=port_name offset=slave \
                   bundle=bundle_name max_widen_bitwidth=512 \
                   max_read_burst_length=64 \
                   num_read_outstanding=GEMV_READ_OUTSTANDING \
                   max_write_burst_length=16 num_write_outstanding=4)

static float dot16_pack(const Pack16 &w, const Pack16 &xv) {
#pragma HLS inline
    float prod[GEMV_TILE_LANES];
#pragma HLS array_partition variable=prod complete
    for (int i = 0; i < GEMV_TILE_LANES; ++i) {
#pragma HLS unroll
        prod[i] = w.data[i] * xv.data[i];
#pragma HLS bind_op variable=prod op=fmul impl=maxdsp
    }
    float s0 = prod[0] + prod[1];
#pragma HLS bind_op variable=s0 op=fadd impl=fulldsp
    float s1 = prod[2] + prod[3];
#pragma HLS bind_op variable=s1 op=fadd impl=fulldsp
    float s2 = prod[4] + prod[5];
#pragma HLS bind_op variable=s2 op=fadd impl=fulldsp
    float s3 = prod[6] + prod[7];
#pragma HLS bind_op variable=s3 op=fadd impl=fulldsp
    float s4 = prod[8] + prod[9];
#pragma HLS bind_op variable=s4 op=fadd impl=fulldsp
    float s5 = prod[10] + prod[11];
#pragma HLS bind_op variable=s5 op=fadd impl=fulldsp
    float s6 = prod[12] + prod[13];
#pragma HLS bind_op variable=s6 op=fadd impl=fulldsp
    float s7 = prod[14] + prod[15];
#pragma HLS bind_op variable=s7 op=fadd impl=fulldsp
    float a0 = s0 + s1;
#pragma HLS bind_op variable=a0 op=fadd impl=fulldsp
    float a1 = s2 + s3;
#pragma HLS bind_op variable=a1 op=fadd impl=fulldsp
    float a2 = s4 + s5;
#pragma HLS bind_op variable=a2 op=fadd impl=fulldsp
    float a3 = s6 + s7;
#pragma HLS bind_op variable=a3 op=fadd impl=fulldsp
    float b0 = a0 + a1;
#pragma HLS bind_op variable=b0 op=fadd impl=fulldsp
    float b1 = a2 + a3;
#pragma HLS bind_op variable=b1 op=fadd impl=fulldsp
    return b0 + b1;
}

static float reduce_part(float part[2][GEMV_TILE_PARTIAL], uint32_t bank) {
#pragma HLS inline
    float s0 = part[bank][0] + part[bank][1];
    float s1 = part[bank][2] + part[bank][3];
    float s2 = part[bank][4] + part[bank][5];
    float s3 = part[bank][6] + part[bank][7];
    return (s0 + s1) + (s2 + s3);
}

// ---- MM2S loader (read-only channels 1..31): burst-read w_k, emit as a stream.
static void mm2s_loader(const Pack16 *w, hls::stream<Pack16> &ws, uint32_t n_packs) {
#pragma HLS inline off
mm2s: for (uint32_t i = 0; i < n_packs; ++i) {
#pragma HLS loop_tripcount min=8192 max=720896
#pragma HLS pipeline II=1
        ws.write(w[i]);
    }
}

// ---- Loader 0 (gmem0, full-duplex read side): inject x into cluster 0, then
// stream w0. Sole reader of gmem0; s2mm is the sole writer.
static void loader0_sys(const Pack16 *w0, const Pack16 *x,
                        hls::stream<Pack16> &xr0, hls::stream<Pack16> &ws0,
                        uint32_t k_packs, uint32_t n_packs) {
#pragma HLS inline off
l0_x: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        xr0.write(x[kp]);
    }
l0_w: for (uint32_t i = 0; i < n_packs; ++i) {
#pragma HLS loop_tripcount min=8192 max=720896
#pragma HLS pipeline II=1
        ws0.write(w0[i]);
    }
}

// ---- Drain: terminate the eight-cluster activation ripple.
static void drain(hls::stream<Pack16> &x_end, uint32_t k_packs) {
#pragma HLS inline off
dr: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        (void)x_end.read();
    }
}

// ---- Four-channel cluster. The channels consume four independent HBM streams
// in lockstep, but read one banked activation pack and fan it out locally. This
// replaces four URAM activation copies and four process-control trees with one.
static void cluster4_sys(hls::stream<Pack16> &ws0,
                         hls::stream<Pack16> &ws1,
                         hls::stream<Pack16> &ws2,
                         hls::stream<Pack16> &ws3,
                         hls::stream<Pack16> &x_in,
                         hls::stream<Pack16> &x_out,
                         hls::stream<Pack16> &ys,
                         uint32_t k_packs, uint32_t rows_per_ch) {
#pragma HLS inline off
    float xbuf[GEMV_TILE_IN_DIM_MAX];
#pragma HLS array_partition variable=xbuf cyclic factor=16
#pragma HLS bind_storage variable=xbuf type=ram_2p impl=bram
cl_load: for (uint32_t kp = 0; kp < k_packs; ++kp) {
#pragma HLS loop_tripcount min=128 max=352
#pragma HLS pipeline II=1
        Pack16 v = x_in.read();
        x_out.write(v);
        for (int lane_i = 0; lane_i < GEMV_TILE_LANES; ++lane_i) {
#pragma HLS unroll
            xbuf[kp * GEMV_TILE_LANES + (uint32_t)lane_i] = v.data[lane_i];
        }
    }

    float part0[2][GEMV_TILE_PARTIAL], part1[2][GEMV_TILE_PARTIAL];
    float part2[2][GEMV_TILE_PARTIAL], part3[2][GEMV_TILE_PARTIAL];
#pragma HLS array_partition variable=part0 complete dim=0
#pragma HLS array_partition variable=part1 complete dim=0
#pragma HLS array_partition variable=part2 complete dim=0
#pragma HLS array_partition variable=part3 complete dim=0
#pragma HLS bind_op variable=part0 op=fadd impl=fulldsp
#pragma HLS bind_op variable=part1 op=fadd impl=fulldsp
#pragma HLS bind_op variable=part2 op=fadd impl=fulldsp
#pragma HLS bind_op variable=part3 op=fadd impl=fulldsp
    Pack16 yp0, yp1, yp2, yp3;
#pragma HLS array_partition variable=yp0.data complete
#pragma HLS array_partition variable=yp1.data complete
#pragma HLS array_partition variable=yp2.data complete
#pragma HLS array_partition variable=yp3.data complete

    uint32_t groups_per_row = k_packs / GEMV_TILE_PARTIAL;
    uint32_t total_groups = rows_per_ch * groups_per_row;
    uint32_t row = 0, g_in_row = 0, a_base = 0, cur = 0;
    bool have_prev = false;

cl_flat: for (uint32_t g = 0; g < total_groups; ++g) {
#pragma HLS loop_tripcount min=1024 max=90112
#pragma HLS pipeline II=8
        bool row_start = (g_in_row == 0);
        bool row_end = (g_in_row == groups_per_row - 1);
    cl_p: for (int p = 0; p < GEMV_TILE_PARTIAL; ++p) {
#pragma HLS unroll
            Pack16 xv;
#pragma HLS array_partition variable=xv.data complete
            for (int i = 0; i < GEMV_TILE_LANES; ++i) {
#pragma HLS unroll
                xv.data[i] = xbuf[(a_base + (uint32_t)p) * GEMV_TILE_LANES + (uint32_t)i];
            }
            Pack16 wv0 = ws0.read();
            Pack16 wv1 = ws1.read();
            Pack16 wv2 = ws2.read();
            Pack16 wv3 = ws3.read();
            float d0 = dot16_pack(wv0, xv);
            float d1 = dot16_pack(wv1, xv);
            float d2 = dot16_pack(wv2, xv);
            float d3 = dot16_pack(wv3, xv);
            part0[cur][p] = (row_start ? 0.0f : part0[cur][p]) + d0;
            part1[cur][p] = (row_start ? 0.0f : part1[cur][p]) + d1;
            part2[cur][p] = (row_start ? 0.0f : part2[cur][p]) + d2;
            part3[cur][p] = (row_start ? 0.0f : part3[cur][p]) + d3;
        }
        if (row_end) {
            if (have_prev) {
                uint32_t er = row - 1;
                yp0.data[er & 15] = reduce_part(part0, cur ^ 1);
                yp1.data[er & 15] = reduce_part(part1, cur ^ 1);
                yp2.data[er & 15] = reduce_part(part2, cur ^ 1);
                yp3.data[er & 15] = reduce_part(part3, cur ^ 1);
                if ((er & 15) == 15) {
                    ys.write(yp0);
                    ys.write(yp1);
                    ys.write(yp2);
                    ys.write(yp3);
                }
            }
            have_prev = true;
            cur ^= 1;
            row++;
            g_in_row = 0;
            a_base = 0;
        } else {
            g_in_row++;
            a_base += GEMV_TILE_PARTIAL;
        }
    }
    if (have_prev) {
        uint32_t er = rows_per_ch - 1;
        yp0.data[er & 15] = reduce_part(part0, cur ^ 1);
        yp1.data[er & 15] = reduce_part(part1, cur ^ 1);
        yp2.data[er & 15] = reduce_part(part2, cur ^ 1);
        yp3.data[er & 15] = reduce_part(part3, cur ^ 1);
        if ((er & 15) == 15) {
            ys.write(yp0);
            ys.write(yp1);
            ys.write(yp2);
            ys.write(yp3);
        }
    }
}

// ---- SLR-local result collectors. Keeping each collector's control FSM local
// avoids coupling all eight cluster FIFOs through one cross-SLR ready network.
static void collector_slr0(hls::stream<Pack16> &ys0,
                           hls::stream<Pack16> &ys1,
                           hls::stream<Pack16> &local_result,
                           uint32_t opacks_per_ch) {
#pragma HLS inline off
slr0_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=1 max=128
    slr0_c0: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys0.read());
        }
    slr0_c1: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys1.read());
        }
    }
}

static void collector_slr1(hls::stream<Pack16> &ys2,
                           hls::stream<Pack16> &ys3,
                           hls::stream<Pack16> &ys4,
                           hls::stream<Pack16> &local_result,
                           uint32_t opacks_per_ch) {
#pragma HLS inline off
slr1_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=1 max=128
    slr1_c2: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys2.read());
        }
    slr1_c3: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys3.read());
        }
    slr1_c4: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys4.read());
        }
    }
}

static void collector_slr2(hls::stream<Pack16> &ys5,
                           hls::stream<Pack16> &ys6,
                           hls::stream<Pack16> &ys7,
                           hls::stream<Pack16> &local_result,
                           uint32_t opacks_per_ch) {
#pragma HLS inline off
slr2_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=1 max=128
    slr2_c5: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys5.read());
        }
    slr2_c6: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys6.read());
        }
    slr2_c7: for (int local = 0; local < CH_PER_CLUSTER; ++local) {
#pragma HLS pipeline II=1
            local_result.write(ys7.read());
        }
    }
}

// Reassemble the three contiguous channel groups in the original interleaved
// output order: channels 0..7, 8..19, then 20..31 for every output pack.
static void collector_final(hls::stream<Pack16> &slr0_result,
                            hls::stream<Pack16> &slr1_result,
                            hls::stream<Pack16> &slr2_result,
                            hls::stream<Pack16> &result,
                            uint32_t opacks_per_ch) {
#pragma HLS inline off
final_p: for (uint32_t p = 0; p < opacks_per_ch; ++p) {
#pragma HLS loop_tripcount min=1 max=128
    final_slr0: for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
            result.write(slr0_result.read());
        }
    final_slr1: for (int i = 0; i < 12; ++i) {
#pragma HLS pipeline II=1
            result.write(slr1_result.read());
        }
    final_slr2: for (int i = 0; i < 12; ++i) {
#pragma HLS pipeline II=1
            result.write(slr2_result.read());
        }
    }
}

// ---- S2MM writer: drain the result stream and burst-write via gmem0's otherwise
// unused write channel (the full-duplex half of loader0's master).
static void s2mm(hls::stream<Pack16> &result, Pack16 *y, uint32_t total_opacks) {
#pragma HLS inline off
s2: for (uint32_t i = 0; i < total_opacks; ++i) {
#pragma HLS loop_tripcount min=32 max=4096
#pragma HLS pipeline II=1
        y[i] = result.read();
    }
}

extern "C" {
void gemv_full(
    const Pack16 *w0,  const Pack16 *w1,  const Pack16 *w2,  const Pack16 *w3,
    const Pack16 *w4,  const Pack16 *w5,  const Pack16 *w6,  const Pack16 *w7,
    const Pack16 *w8,  const Pack16 *w9,  const Pack16 *w10, const Pack16 *w11,
    const Pack16 *w12, const Pack16 *w13, const Pack16 *w14, const Pack16 *w15,
    const Pack16 *w16, const Pack16 *w17, const Pack16 *w18, const Pack16 *w19,
    const Pack16 *w20, const Pack16 *w21, const Pack16 *w22, const Pack16 *w23,
    const Pack16 *w24, const Pack16 *w25, const Pack16 *w26, const Pack16 *w27,
    const Pack16 *w28, const Pack16 *w29, const Pack16 *w30, const Pack16 *w31,
    const Pack16 *x, Pack16 *y, uint32_t k_packs, uint32_t rows_per_ch) {
GEMV_READ_WRITE_AXI(w0, gmem0);
GEMV_READ_AXI(w1, gmem1);
GEMV_READ_AXI(w2, gmem2);
GEMV_READ_AXI(w3, gmem3);
GEMV_READ_AXI(w4, gmem4);
GEMV_READ_AXI(w5, gmem5);
GEMV_READ_AXI(w6, gmem6);
GEMV_READ_AXI(w7, gmem7);
GEMV_READ_AXI(w8, gmem8);
GEMV_READ_AXI(w9, gmem9);
GEMV_READ_AXI(w10, gmem10);
GEMV_READ_AXI(w11, gmem11);
GEMV_READ_AXI(w12, gmem12);
GEMV_READ_AXI(w13, gmem13);
GEMV_READ_AXI(w14, gmem14);
GEMV_READ_AXI(w15, gmem15);
GEMV_READ_AXI(w16, gmem16);
GEMV_READ_AXI(w17, gmem17);
GEMV_READ_AXI(w18, gmem18);
GEMV_READ_AXI(w19, gmem19);
GEMV_READ_AXI(w20, gmem20);
GEMV_READ_AXI(w21, gmem21);
GEMV_READ_AXI(w22, gmem22);
GEMV_READ_AXI(w23, gmem23);
GEMV_READ_AXI(w24, gmem24);
GEMV_READ_AXI(w25, gmem25);
GEMV_READ_AXI(w26, gmem26);
GEMV_READ_AXI(w27, gmem27);
GEMV_READ_AXI(w28, gmem28);
GEMV_READ_AXI(w29, gmem29);
GEMV_READ_AXI(w30, gmem30);
GEMV_READ_AXI(w31, gmem31);
// x rides gmem0's READ channel (co-located with w0); y rides gmem0's WRITE channel
// (the free full-duplex half). loader0 is the sole reader, s2mm the sole writer.
GEMV_READ_AXI(x, gmem0);
#pragma HLS interface m_axi port=y offset=slave bundle=gmem0 max_widen_bitwidth=512 max_write_burst_length=16 num_write_outstanding=4
#pragma HLS interface s_axilite port=w0
#pragma HLS interface s_axilite port=w1
#pragma HLS interface s_axilite port=w2
#pragma HLS interface s_axilite port=w3
#pragma HLS interface s_axilite port=w4
#pragma HLS interface s_axilite port=w5
#pragma HLS interface s_axilite port=w6
#pragma HLS interface s_axilite port=w7
#pragma HLS interface s_axilite port=w8
#pragma HLS interface s_axilite port=w9
#pragma HLS interface s_axilite port=w10
#pragma HLS interface s_axilite port=w11
#pragma HLS interface s_axilite port=w12
#pragma HLS interface s_axilite port=w13
#pragma HLS interface s_axilite port=w14
#pragma HLS interface s_axilite port=w15
#pragma HLS interface s_axilite port=w16
#pragma HLS interface s_axilite port=w17
#pragma HLS interface s_axilite port=w18
#pragma HLS interface s_axilite port=w19
#pragma HLS interface s_axilite port=w20
#pragma HLS interface s_axilite port=w21
#pragma HLS interface s_axilite port=w22
#pragma HLS interface s_axilite port=w23
#pragma HLS interface s_axilite port=w24
#pragma HLS interface s_axilite port=w25
#pragma HLS interface s_axilite port=w26
#pragma HLS interface s_axilite port=w27
#pragma HLS interface s_axilite port=w28
#pragma HLS interface s_axilite port=w29
#pragma HLS interface s_axilite port=w30
#pragma HLS interface s_axilite port=w31
#pragma HLS interface s_axilite port=x
#pragma HLS interface s_axilite port=y
#pragma HLS interface s_axilite port=k_packs
#pragma HLS interface s_axilite port=rows_per_ch
#pragma HLS interface s_axilite port=return

    hls::stream<Pack16> ws[NCH];
#pragma HLS array_partition variable=ws complete
#pragma HLS stream variable=ws depth=64
#pragma HLS bind_storage variable=ws type=fifo impl=bram
    hls::stream<Pack16> xr[NCL + 1];
#pragma HLS array_partition variable=xr complete
#pragma HLS stream variable=xr depth=64
#pragma HLS bind_storage variable=xr type=fifo impl=bram
    hls::stream<Pack16> ys[NCL];
#pragma HLS array_partition variable=ys complete
#pragma HLS stream variable=ys depth=64
#pragma HLS bind_storage variable=ys type=fifo impl=bram
    hls::stream<Pack16> slr0_result, slr1_result, slr2_result;
#pragma HLS stream variable=slr0_result depth=64
#pragma HLS stream variable=slr1_result depth=64
#pragma HLS stream variable=slr2_result depth=64
#pragma HLS bind_storage variable=slr0_result type=fifo impl=bram
#pragma HLS bind_storage variable=slr1_result type=fifo impl=bram
#pragma HLS bind_storage variable=slr2_result type=fifo impl=bram
    hls::stream<Pack16> result;
#pragma HLS stream variable=result depth=64
#pragma HLS bind_storage variable=result type=fifo impl=bram

    uint32_t n_packs = rows_per_ch * k_packs;
    uint32_t opacks_per_ch = rows_per_ch / GEMV_TILE_LANES;
    uint32_t total_opacks = opacks_per_ch * NCH;

#pragma HLS dataflow disable_start_propagation
    loader0_sys(w0, x, xr[0], ws[0], k_packs, n_packs);
    mm2s_loader(w1, ws[1], n_packs);
    mm2s_loader(w2, ws[2], n_packs);
    mm2s_loader(w3, ws[3], n_packs);
    mm2s_loader(w4, ws[4], n_packs);
    mm2s_loader(w5, ws[5], n_packs);
    mm2s_loader(w6, ws[6], n_packs);
    mm2s_loader(w7, ws[7], n_packs);
    mm2s_loader(w8, ws[8], n_packs);
    mm2s_loader(w9, ws[9], n_packs);
    mm2s_loader(w10, ws[10], n_packs);
    mm2s_loader(w11, ws[11], n_packs);
    mm2s_loader(w12, ws[12], n_packs);
    mm2s_loader(w13, ws[13], n_packs);
    mm2s_loader(w14, ws[14], n_packs);
    mm2s_loader(w15, ws[15], n_packs);
    mm2s_loader(w16, ws[16], n_packs);
    mm2s_loader(w17, ws[17], n_packs);
    mm2s_loader(w18, ws[18], n_packs);
    mm2s_loader(w19, ws[19], n_packs);
    mm2s_loader(w20, ws[20], n_packs);
    mm2s_loader(w21, ws[21], n_packs);
    mm2s_loader(w22, ws[22], n_packs);
    mm2s_loader(w23, ws[23], n_packs);
    mm2s_loader(w24, ws[24], n_packs);
    mm2s_loader(w25, ws[25], n_packs);
    mm2s_loader(w26, ws[26], n_packs);
    mm2s_loader(w27, ws[27], n_packs);
    mm2s_loader(w28, ws[28], n_packs);
    mm2s_loader(w29, ws[29], n_packs);
    mm2s_loader(w30, ws[30], n_packs);
    mm2s_loader(w31, ws[31], n_packs);

    cluster4_sys(ws[0], ws[1], ws[2], ws[3], xr[0], xr[1], ys[0], k_packs, rows_per_ch);
    cluster4_sys(ws[4], ws[5], ws[6], ws[7], xr[1], xr[2], ys[1], k_packs, rows_per_ch);
    cluster4_sys(ws[8], ws[9], ws[10], ws[11], xr[2], xr[3], ys[2], k_packs, rows_per_ch);
    cluster4_sys(ws[12], ws[13], ws[14], ws[15], xr[3], xr[4], ys[3], k_packs, rows_per_ch);
    cluster4_sys(ws[16], ws[17], ws[18], ws[19], xr[4], xr[5], ys[4], k_packs, rows_per_ch);
    cluster4_sys(ws[20], ws[21], ws[22], ws[23], xr[5], xr[6], ys[5], k_packs, rows_per_ch);
    cluster4_sys(ws[24], ws[25], ws[26], ws[27], xr[6], xr[7], ys[6], k_packs, rows_per_ch);
    cluster4_sys(ws[28], ws[29], ws[30], ws[31], xr[7], xr[8], ys[7], k_packs, rows_per_ch);
    drain(xr[NCL], k_packs);
    collector_slr0(ys[0], ys[1], slr0_result, opacks_per_ch);
    collector_slr1(ys[2], ys[3], ys[4], slr1_result, opacks_per_ch);
    collector_slr2(ys[5], ys[6], ys[7], slr2_result, opacks_per_ch);
    collector_final(slr0_result, slr1_result, slr2_result, result, opacks_per_ch);
    s2mm(result, y, total_opacks);
}
}
