/* Iter79 step 0a: SLR-crossing stream probe, source half.  Emits `beats`
 * 512-bit AXI-Stream words `reps` times at one beat per cycle; the pattern
 * includes the rep so XOR folding across reps does not cancel. */
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <stdint.h>
typedef ap_axiu<512, 0, 0, 0> Beat;
extern "C" void slr_src(hls::stream<Beat> &out, uint32_t beats, uint32_t reps) {
#pragma HLS interface axis port=out
#pragma HLS interface s_axilite port=beats
#pragma HLS interface s_axilite port=reps
#pragma HLS interface s_axilite port=return
src_rep: for (uint32_t r = 0; r < reps; ++r) {
#pragma HLS loop_tripcount min=1 max=8
    src_beat: for (uint32_t i = 0; i < beats; ++i) {
#pragma HLS loop_tripcount min=1048576 max=1048576
#pragma HLS pipeline II=1
            Beat b;
            for (int k = 0; k < 8; ++k) {
#pragma HLS unroll
                const ap_uint<64> lane = ((ap_uint<64>)r << 48) ^ ((ap_uint<64>)i * 8 + k) ^ (ap_uint<64>)0x9E3779B97F4A7C15ULL;
                b.data.range(64 * k + 63, 64 * k) = lane;
            }
            b.keep = -1;
            b.strb = -1;
            b.last = (i + 1 == beats) ? 1 : 0;
            out.write(b);
        }
    }
}
