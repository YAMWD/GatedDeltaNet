# Accelerator Documentation

**Current architecture:** the FPGA is a decode-only accelerator. **Iter66e**
forwards one token at a time with 32 HBM weight readers, 16 two-port GEMV
clusters running free-running pipelines, a native `ap_float<16,8>` BF16
multiplier feeding FP32 reduction trees, packed-BF16 weights, transient
activations resident in local BRAM, four-port packed **BF16** recurrent state
behind full-window 4,096-deep URAM queues, two concurrent 16-column recurrent
islands, registered GEMV collector boundaries, head-streamed Q/K/V convolution
and recurrence, an on-chip LM-head argmax, and a streamed full-vocabulary logit
export. Prefill runs on the GPU and supplies persistent recurrent and
convolution state.

The integrated Iter66e image routes with zero overlaps, zero unrouted nets and
zero failing timing endpoints of 2,277,369 (WNS +0.003 ns, WHS +0.009 ns) at a
true 100 MHz, and measures **26.654 ms/token wall / 25.625 ms kernel
(2.5625M cycles)** with an exact 64-token trajectory. It is **1.582x** faster
than Iter61 and **4.55x** faster than the 121.4 ms eight-port baseline.

Toolchain: **Vitis 2024.2** (the native BF16 multiplier requires its
`ap_float`). Evidence: build job 2502, on-card job 2507.

## Current References

- [architecture.md](architecture.md): authoritative Iter66e top-level data
  flow, arithmetic contract, activation residency, 32-port GEMV topology,
  interfaces, state handling, HBM map, physical design, and measured result.
- [recurrent_attention.md](recurrent_attention.md): the recurrent block — now
  the largest identifiable cycle consumer at 40.7% of the token — its BF16
  state transport, per-head schedule, and measured per-loop cycles.
- [cycle_optimization_roadmap.md](cycle_optimization_roadmap.md): remaining
  cycle targets, rebased on the Iter66e measurement, with the levers ranked by
  measured share of the token rather than by share of bytes.
- [fp32_bf16_quality_evaluation.md](fp32_bf16_quality_evaluation.md): the
  GPU-side precision study that cleared BF16, **complete**, plus the on-card
  WikiText-2 sequel — also complete: FPGA word perplexity 16.774840 against
  GPU 16.776124 on identical windows, **-0.0077%** against a 5% gate.

## The three gates, after the Iter66 milestone

| Gate | Scope | Status |
|---|---|---|
| Native csim vs cached golden | exact trajectory + every pre-argmax logit | **bit-exact, unchanged** — this is what the edit hook runs |
| Independent-GPU vector gate | on card: NRMSE, cosine, top-5, argmax | **the on-card gate**; Iter66e passes over 2,016,000 logits |
| WikiText-2 perplexity | on card, teacher-forced, 314,843 tokens | **-0.0077%** vs GPU on identical windows |
| ~~Hardware vs native bit-exact~~ | ~~on card~~ | **REMOVED** — gated on something unachievable; see `architecture.md` § *Arithmetic contract* |
- [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md): implementation
  history and measured progression from the original single-reader decode
  kernel through the integrated 32-port design. **History, not the spec.**
- [decode_premise.md](decode_premise.md): GPU measurements motivating the
  decode-only partition.
- [depthwise_conv.md](depthwise_conv.md) and [output_norm.md](output_norm.md):
  active non-GEMV compute blocks. Their embedded synthesis tables predate
  Iter66 and are historical.
- [optimization_log.md](optimization_log.md): exhaustive chronological record
  of synthesis, routing, timing, rejected experiments, and on-card results.

## Two things that are easy to get wrong

**The gate is no longer bit-exact end to end.** Native-vs-golden is still
bit-exact and is what the edit hook runs. *Hardware*-vs-native is not, by a
measured and understood cause: `expf`/`log1pf` are outside IEEE-754's
correct-rounding mandate, glibc and the AMD FPO cores disagree in the last bit
on 21.4% of the real per-head `decay` operands, and each such scalar flips the
state lanes sitting on an RNE tie — 129 of 12,582,912, each by one BF16 ULP.
The accepted on-card gate is the scale-aware CUDA vector gate. See
`architecture.md` § *Arithmetic contract and what "correct" means*.

**Bytes are no longer the binding metric.** At 2.597 GB of weights per token
the 32 ports are busy 49.5% of the time, so a lever that removes bytes no
longer buys proportional time. Rank levers by measured share of the *token*.

The earlier standalone 32-port microbenchmark, its build commands, bandwidth,
and post-route results are documented in
[`../microbench/gemv_tile/README.md`](../microbench/gemv_tile/README.md).

Retired prefill matmul implementations, the re-prefill baseline, and the
intermediate dual-mode decode design no longer have separate documents. Their
relevant measurements remain in [optimization_log.md](optimization_log.md).
New status statements must identify whether they refer to the production
Iter66e integrated kernel, a historical integrated iteration, a proposed
roadmap stage, or the standalone GEMV microbenchmark.
