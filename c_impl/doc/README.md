# Accelerator Documentation

**Current architecture:** the FPGA is a decode-only accelerator. Iter39C
forwards one token at a time with 32 HBM weight readers, 16 two-port GEMV
clusters, BRAM MM2S/FIFO decoupling, transient activations resident in local
BRAM, four-port packed recurrent state, a 32-lane head-local fused recurrent
engine, head-streamed Q/K/V convolution, and an on-chip LM-head argmax. Prefill
runs on the GPU and supplies persistent recurrent and convolution state.

The integrated U55C image routes with zero failed/unrouted nets and zero
overlaps and closes timing at the requested 100 MHz. It is the fastest
validated image, with exact 64-token parity at **43.093 ms/token / 4.309M
cycles**, 1.093x faster than Iter38 and 2.817x faster than the 121.4 ms
eight-port baseline.

## Current References

- [architecture.md](architecture.md): authoritative Iter39C top-level data
  flow, activation residency, 32-port GEMV topology, interfaces, state
  handling, HBM map, physical design, and measured result.
- [cycle_optimization_roadmap.md](cycle_optimization_roadmap.md): proposed
  cycle-first path toward a fully streamed exact-FP32 design, distinguishing
  completed Iter39C work from remaining targets.
- [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md): implementation
  history and measured progression from the original single-reader decode
  kernel through the integrated 32-port design.
- [decode_premise.md](decode_premise.md): GPU measurements motivating the
  decode-only partition.
- [recurrent_attention.md](recurrent_attention.md),
  [depthwise_conv.md](depthwise_conv.md), and [output_norm.md](output_norm.md):
  active non-GEMV compute blocks. Their older synthesis tables are historical.
- [optimization_log.md](optimization_log.md): exhaustive chronological record
  of synthesis, routing, timing, rejected experiments, and on-card results.

The earlier standalone 32-port microbenchmark, its build commands, bandwidth,
and post-route results are documented in
[`../microbench/gemv_tile/README.md`](../microbench/gemv_tile/README.md).

Retired prefill matmul implementations, the re-prefill baseline, and the
intermediate dual-mode decode design no longer have separate documents. Their
relevant measurements remain in [optimization_log.md](optimization_log.md).
New status statements must identify whether they refer to the production
Iter39C integrated kernel, a historical integrated iteration, a proposed
roadmap stage, or the standalone GEMV microbenchmark.
