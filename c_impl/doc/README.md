# Accelerator Documentation

**Current architecture:** the FPGA is a decode-only accelerator. Iter57
forwards one token at a time with 32 HBM weight readers, 16 two-port GEMV
clusters, BRAM MM2S/FIFO decoupling, transient activations resident in local
BRAM, four-port packed recurrent state, two concurrent 16-column recurrent
islands, registered GEMV collector boundaries, head-streamed Q/K/V convolution
and recurrence, and an on-chip LM-head argmax. Prefill runs on the GPU and
supplies persistent recurrent and convolution state.

The integrated Iter57 image routes with zero failed/unrouted nets and zero
overlaps, closes both kernel and fixed DMA timing at a true 100 MHz, and passes
exact 64-token parity at **42.023540 ms/token / 4.202354M cycles**. It is 2.5%
faster than Iter39C and 2.889x faster than the 121.4 ms eight-port baseline.

## Current References

- [architecture.md](architecture.md): authoritative Iter57 top-level data
  flow, activation residency, 32-port GEMV topology, interfaces, state
  handling, HBM map, physical design, and measured result.
- [cycle_optimization_roadmap.md](cycle_optimization_roadmap.md): proposed
  cycle-first path toward a fully streamed exact-FP32 design, distinguishing
  completed Iter57 work from remaining targets.
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
Iter57 integrated kernel, a historical integrated iteration, a proposed
roadmap stage, or the standalone GEMV microbenchmark.
