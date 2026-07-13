# Accelerator Documentation

**Current architecture:** the FPGA is a decode-only accelerator. It forwards
one token at a time with an eight-reader sharded GEMV engine. Prefill runs on
the GPU and supplies persistent recurrent and convolution state. The 32-port
GEMV is currently a routed standalone microbenchmark, not part of the complete
GDN kernel.

## Current References

- [architecture.md](architecture.md): authoritative top-level data flow,
  interfaces, state handling, and HBM map.
- [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md): implementation
  history and measured progression of the integrated decode GEMV.
- [decode_premise.md](decode_premise.md): GPU measurements motivating the
  decode-only partition.
- [recurrent_attention.md](recurrent_attention.md),
  [depthwise_conv.md](depthwise_conv.md), and [output_norm.md](output_norm.md):
  active non-GEMV compute blocks. Their older synthesis tables are historical.
- [optimization_log.md](optimization_log.md): chronological record, including
  the routed 32-port GEMV milestone.

The standalone 32-port implementation, build commands, on-card bandwidth, and
post-route results are documented in
[`../microbench/gemv_tile/README.md`](../microbench/gemv_tile/README.md).

Retired prefill matmul implementations, the re-prefill baseline, and the
intermediate dual-mode decode design no longer have separate documents. Their
relevant measurements remain in [optimization_log.md](optimization_log.md).
New status statements must identify whether they refer to the integrated
eight-port model or the standalone 32-port microbenchmark.
