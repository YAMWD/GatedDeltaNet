// hbm_rd: read-only streaming kernel for the HBM pseudo-channel ceiling (Iter88 diagnosis).
// One 512-bit read port; n_beats sequential beats, reps passes; a 64-bit sum of two lanes
// keeps the reads alive and lets the host check the data. At 300 MHz the port asks for
// 19.2 GB/s, above the channel's 14.4 GB/s peak, so the memory sets the rate.
#include <ap_int.h>
#include <stdint.h>
typedef ap_uint<512> Beat512;
extern "C" void hbm_rd(const Beat512 *in, ap_uint<64> *out, uint32_t n_beats, uint32_t reps) {
#pragma HLS interface m_axi port=in bundle=gmem_in offset=slave max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=32 latency=64
#pragma HLS interface m_axi port=out bundle=gmem_out offset=slave
#pragma HLS interface s_axilite port=in
#pragma HLS interface s_axilite port=out
#pragma HLS interface s_axilite port=n_beats
#pragma HLS interface s_axilite port=reps
#pragma HLS interface s_axilite port=return
    ap_uint<64> acc = 0;
rep_loop:
    for (uint32_t r = 0; r < reps; ++r) {
#pragma HLS loop_tripcount min=1 max=16
    rd_loop:
        for (uint32_t i = 0; i < n_beats; ++i) {
#pragma HLS loop_tripcount min=4194304 max=4194304
#pragma HLS pipeline II=1
            const Beat512 b = in[i];
            acc += ap_uint<64>(b.range(63, 0)) + ap_uint<64>(b.range(511, 448));
        }
    }
    out[0] = acc;
}
