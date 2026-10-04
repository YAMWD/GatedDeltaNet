/* Iter79 step 0a: SLR-crossing stream probe, sink half.  Consumes the
 * source's beats at one per cycle, XOR-folds each 64-bit lane, counts TLAST,
 * and writes 16 result words to HBM. */
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <stdint.h>
typedef ap_axiu<512, 0, 0, 0> Beat;
extern "C" void slr_sink(hls::stream<Beat> &in, ap_uint<64> *out, uint32_t beats, uint32_t reps) {
#pragma HLS interface axis port=in
#pragma HLS interface m_axi port=out offset=slave bundle=mem max_widen_bitwidth=512
#pragma HLS interface s_axilite port=beats
#pragma HLS interface s_axilite port=reps
#pragma HLS interface s_axilite port=return
    ap_uint<64> x[8];
#pragma HLS array_partition variable=x complete
    for (int k = 0; k < 8; ++k) x[k] = 0;
    uint32_t lasts = 0;
    uint32_t total = 0;
sink_rep: for (uint32_t r = 0; r < reps; ++r) {
#pragma HLS loop_tripcount min=1 max=8
    sink_beat: for (uint32_t i = 0; i < beats; ++i) {
#pragma HLS loop_tripcount min=1048576 max=1048576
#pragma HLS pipeline II=1
            const Beat b = in.read();
            for (int k = 0; k < 8; ++k) {
#pragma HLS unroll
                x[k] ^= (ap_uint<64>)b.data.range(64 * k + 63, 64 * k);
            }
            lasts += b.last ? 1u : 0u;
            total++;
        }
    }
    for (int k = 0; k < 8; ++k) out[k] = x[k];
    out[8] = lasts;
    out[9] = total;
    for (int k = 10; k < 16; ++k) out[k] = 0;
}
