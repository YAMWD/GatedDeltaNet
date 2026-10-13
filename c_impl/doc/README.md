# Accelerator Documentation

**Current architecture:** the FPGA is a decode-only accelerator. The
production image — the **200 MHz three-kernel image of the Iter82–87 series**,
promoted 2026-10-13 (the Iter82–86 image of 2026-10-11 is its predecessor, the
Iter81 balanced-island image, the Iter79 three-kernel image, the 150 MHz Iter76
monolith and Iter67c before that) — forwards
one token at a time with 32 HBM weight
readers and 16 two-port GEMV clusters split across three kernels, one per die:
`gdn_forward_p` (control, SLR1), `gdn_k_slr0` (SLR0) and `gdn_k_slr2` (SLR2,
with the recurrent islands and the state writers), joined by seven registered
AXI-Stream links that carry data only under one static schedule. The datapath
is unchanged: free-running cluster pipelines, a native `ap_float<16,8>` BF16
multiplier feeding FP32 reduction trees, packed-BF16 weights, transient
activations resident in local BRAM, four-port packed **BF16** recurrent state
behind full-window 4,096-deep URAM queues, two concurrent 16-column recurrent
islands — since Iter81 each a three-stage head pipeline (prep → read →
update) that hides under the QKVG weight stream, with the layer's state
fetched before the call's weights and written back after them (Iter84) — a
five-phase II=1 recurrent read schedule, head-streamed Q/K/V convolution and
recurrence with the output norm/gate and the convolution-tail write done per
head inside the QKVG call (Iter86; the tail writes wait until port 0's weights
are read, Iter87), a packed BF16 aux-weight image with one
fused tiny a/b GEMV (Iter83), the residual add fused with the following RMSNorm
and the SwiGLU inside the GU store (Iter85), an on-chip strict LM-head argmax,
and a streamed full-vocabulary logit export. Prefill runs on the GPU and supplies persistent
recurrent and convolution state.

The production image routes with zero routing errors and closes a true
**200 MHz** kernel clock — setup +0.027 / hold +0.010 ns, with the fixed
250 MHz DMA clock at +0.003 ns; the shell's 450 MHz HBM AXI clock missed by
0.040 ns on its own crossing pipes and runs at 442.0 MHz under the flow's
440 MHz floor. It measures **8.484 ms/token production TPOT / 8.361 ms
kernel (1,672,120 cycles)** on acclnode01's card with an exact 64-token
trajectory and the production vector-gate figures, **−18.5 % kernel time**
against the Iter81 image on the same card (10.256 ms) with order-preserving
arithmetic, **−0.19 %** against the Iter82–86 image in a same-job A/B, or
**14.3x** the 121.4 ms eight-port baseline. Paired power is about 70 W active,
0.59–0.60 J/token gross, 3.7x faster than an A100 running eager PyTorch. The flow is one
command (`make run_hw`); see
[reproduce_f200.md](reproduce_f200.md).

Toolchain: **Vitis 2024.2** (the native BF16 multiplier requires its
`ap_float`). Evidence: build 7370 with on-card 7371 (Iter81), qualification
jobs 7401 (drift, WikiText-2) and 7437/7438 (paired power); the Iter79 image's builds 6548/6568/6642 with on-card
6549/6569/6643 and qualification 6570–6572. The two failed Iter81 links — a
state reader whose address Vivado synthesis folded 128 MiB low (now covered by
a reader address gate) and an HBM-clock refusal — and the hang the first
200 MHz link shipped with are recorded in `architecture.md` and
`optimization_log.md`.

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
- [recurrent_attention.md](recurrent_attention.md): the recurrent block —
  through Iter79 the pacing item of the QKVG call at ~41 % of the token, since
  Iter81 a three-stage head pipeline hidden under the weight stream — its
  BF16 state transport, the pipeline's per-stage cycles, timing and resources,
  and the historical serial schedule.
- [cycle_optimization_roadmap.md](cycle_optimization_roadmap.md): remaining
  cycle targets (reference 2.0694M cycles at 200 MHz since Iter81; Iter67c's
  2.4099M before), with the levers ranked by measured share of the
  token rather than by share of bytes.
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
