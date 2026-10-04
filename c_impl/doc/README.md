# Accelerator Documentation

**Current architecture:** the FPGA is a decode-only accelerator. The
production image — the **200 MHz three-kernel image of the Iter79 campaign**,
promoted 2026-10-04 (the 150 MHz Iter76 monolith is its predecessor, Iter67c
the 100 MHz one before that) — forwards one token at a time with 32 HBM weight
readers and 16 two-port GEMV clusters split across three kernels, one per die:
`gdn_forward_p` (control, SLR1), `gdn_k_slr0` (SLR0) and `gdn_k_slr2` (SLR2,
with the recurrent islands and the state writers), joined by seven registered
AXI-Stream links that carry data only under one static schedule. The datapath
is unchanged: free-running cluster pipelines, a native `ap_float<16,8>` BF16
multiplier feeding FP32 reduction trees, packed-BF16 weights, transient
activations resident in local BRAM, four-port packed **BF16** recurrent state
behind full-window 4,096-deep URAM queues, two concurrent 16-column recurrent
islands, a five-phase II=1 recurrent read schedule, head-streamed Q/K/V
convolution and recurrence, an on-chip strict LM-head argmax, and a streamed
full-vocabulary logit export. Prefill runs on the GPU and supplies persistent
recurrent and convolution state.

The production image routes with zero routing errors (1,671,899 nets) and
closes a true **200 MHz** kernel clock — setup +0.029 / hold +0.010 ns, with
the fixed 250 MHz DMA clock at +0.003 and the 450 MHz HBM clock at +0.015, no
exceptions and no automatic scaling. It measures **12.350 ms/token production
TPOT / 12.227 ms kernel (2,445,400 cycles)** with an exact 64-token trajectory,
or **9.8x** the 121.4 ms eight-port baseline; paired power is 57.8 W,
0.717 J/token gross (−24 % time and −8 % energy per token versus the 150 MHz
image). The link reproduced itself to the picosecond (builds 6548 and 6568,
on-card jobs 6549 and 6569) and the flow is one command; see
[reproduce_f200.md](reproduce_f200.md).

Toolchain: **Vitis 2024.2** (the native BF16 multiplier requires its
`ap_float`). Evidence: builds 6548/6568 with on-card 6549/6569, the
`make run_hw` verification build 6642 with on-card 6643, qualification
jobs 6570 (drift, WikiText-2) and 6571/6572 (paired power); the hang that the
first 200 MHz link shipped with, and its one-line fix, are recorded in
`architecture.md` and `optimization_log.md`.

## Current References

- [gpu_latency_energy_evaluation.md](gpu_latency_energy_evaluation.md):
  standalone H100/A100 eager latency and energy measurement from a fresh clone,
  pinned dependencies, Slurm submission and CPU validation scope.
- [architecture.md](architecture.md): authoritative top-level data flow,
  arithmetic contract, activation residency, 32-port GEMV topology, the
  three-kernel split and its links, interfaces, state handling, HBM map,
  physical design, and measured result of the 200 MHz production image (with
  the 150 MHz Iter76 image, Iter67c and Iter66e as recorded predecessors).
- [reproduce_f200.md](reproduce_f200.md): how to rebuild and validate the
  production image with `make run_hw`, what the flow checks, measured wall
  times, and its evidence boundary. [reproduce_f150.md](reproduce_f150.md) is
  the historical 150 MHz monolith recipe.
- [recurrent_attention.md](recurrent_attention.md): the recurrent block — the
  largest identifiable cycle consumer at ~41% of the token, now hosted by the
  SLR2 kernel and fed over the stream links — its BF16 state transport,
  per-head schedule, and measured per-loop cycles.
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
