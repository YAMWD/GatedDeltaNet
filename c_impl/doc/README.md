# Accelerator Documentation

**Current architecture:** the FPGA is a decode-only accelerator. The
production image — the **150 MHz design reproduced by the Iter76 `make run_hw`
flow** (Iter73b2/75b source, Iter75d/e/f physical passes; Iter67c is its
100 MHz predecessor) — forwards one token at a time with 32 HBM weight readers, 16 two-port GEMV
clusters running free-running pipelines, a native `ap_float<16,8>` BF16
multiplier feeding FP32 reduction trees, packed-BF16 weights, transient
activations resident in local BRAM, four-port packed **BF16** recurrent state
behind full-window 4,096-deep URAM queues, two concurrent 16-column recurrent
islands, a five-phase II=1 recurrent read schedule, registered GEMV collector
boundaries, head-streamed Q/K/V convolution and recurrence, an on-chip strict
LM-head argmax, and a streamed full-vocabulary logit export. Prefill runs on
the GPU and supplies persistent recurrent and convolution state.

The production image routes with zero routing errors (1,592,229 nets) and
closes a true **150 MHz** kernel clock — setup 0.000 / hold +0.002 ns, with the
fixed 250 MHz DMA clock at +0.003 and the 450 MHz HBM clock at +0.052, no
exceptions and no automatic scaling. It measures **16.255 ms/token production
TPOT / 16.131 ms kernel (2.4197M cycles)** with an exact 64-token trajectory,
or **7.53x** the 121.4 ms eight-port baseline by kernel median; paired power is
48.0 W, 0.779 J/token gross. The build is reproducible from source in one
command and was demonstrated twice (builds 3987 and 4022, checksum-identical
implementations, on-card jobs 4021 and 4023); see
[reproduce_f150.md](reproduce_f150.md).

Toolchain: **Vitis 2024.2** (the native BF16 multiplier requires its
`ap_float`). Evidence: closure jobs 3953/3955 and image 3956 (on-card 3957,
qualification 3958–3960); reproduction builds 3987/4022 with on-card
4021/4023.

## Current References

- [architecture.md](architecture.md): authoritative top-level data flow,
  arithmetic contract, activation residency, 32-port GEMV topology, interfaces,
  state handling, HBM map, physical design, and measured result of the 150 MHz
  production image (with Iter67c and Iter66e as recorded predecessors).
- [reproduce_f150.md](reproduce_f150.md): how to rebuild and validate the
  production image with `make run_hw`, what the flow checks, measured wall
  times, and its evidence boundary.
- [recurrent_attention.md](recurrent_attention.md): the recurrent block — now
  the largest identifiable cycle consumer at 40.7% of the token — its BF16
  state transport, per-head schedule, and measured per-loop cycles.
- [cycle_optimization_roadmap.md](cycle_optimization_roadmap.md): remaining
  cycle targets, rebased on the Iter67c measurement, with the levers ranked by
  measured share of the token rather than by share of bytes.
- [frequency_250mhz_roadmap.md](frequency_250mhz_roadmap.md): the record of
  the Iter68 frequency-locality redesign (closed 2026-09-05 as
  stopped/inconclusive; 250 MHz is no longer a target) and the HBM-aware
  frequency floor model. Historical, not the retained production specification.
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

**At the production 150 MHz, bytes are still not the binding metric.** At
2.799 GB of weights per token the 32 ports are busy 56.5% of the 2,419,650-cycle
token, and a busy port draws 9.6 GB/s, 66.7% of its 14.4 GB/s pseudo-channel
peak, so a lever that removes bytes does not buy proportional time at this
clock. This changes near 200 MHz (88.9% of peak per port) and is impossible at
250 MHz (111%). Use the HBM-aware frequency roadmap before projecting a
higher-clock result.

The earlier standalone 32-port microbenchmark, its build commands, bandwidth,
and post-route results are documented in
[`../microbench/gemv_tile/README.md`](../microbench/gemv_tile/README.md).

Retired prefill matmul implementations, the re-prefill baseline, and the
intermediate dual-mode decode design no longer have separate documents. Their
relevant measurements remain in [optimization_log.md](optimization_log.md).
New status statements must identify whether they refer to the production
150 MHz integrated kernel, a historical integrated iteration (Iter67c, Iter66e,
…), a proposed roadmap stage, or the standalone GEMV microbenchmark.
