# Cycle-First Optimization Roadmap After Iter36

**Status:** Proposed architecture; not yet implemented.

**Reference:** Iter36 exact-FP32 production architecture, reconstructed at
5,998,198 cycles/token and measured at 59.578 ms/token at 100 MHz. The same
source is also validated at an auto-scaled 115.7 MHz and 51.844 ms/token; this
frequency result does not change the cycle baseline.

**Primary objective:** reduce exact single-token decode cycles as far as the
dense FP32 model permits, then raise frequency only after the cycle-reduced
architecture routes and closes timing at 100 MHz.

This roadmap preserves the proven Iter36 32-port/16-cluster GEMV topology as
the rollback point. It keeps MM2S readers and BRAM FIFO decoupling, transient
activation residency, on-chip LM-head argmax, external FP32 recurrent state,
and exact token parity. Planned storage formats, state-port mappings, and
streaming regions are changes to be validated; they are not part of the
current production architecture.

## 1. Measured Starting Point and Limits

The Iter36 on-card mean is within 0.404 ms of its reconstructed static
schedule, so the current kernel is schedule-bound rather than dominated by an
unexplained runtime stall.

| Component | Iter36 cycles/token |
|---|---:|
| Large GEMV | 2,785,524 |
| Recurrent attention | 1,841,112 |
| Q/K/V convolution | 624,744 |
| SwiGLU | 321,048 |
| RMSNorm | 139,405 |
| Output norm/gate | 128,712 |
| Tiny A/B GEMV | 103,584 |
| Activation handoff/copy/argmax | 54,069 |
| **Total** | **5,998,198** |

The raw dense-FP32 weight stream is approximately 2,733,056 cycles/token per
fully occupied weight port. Combining matrices does not change this byte
count. Fusion is valuable because it exposes head- and chunk-level results
early enough to overlap the remaining work with that unavoidable stream.

Each token also reads and writes 48 MiB of recurrent state, or about 1,572,864
512-bit `Pack16` transfers in total. If state traffic time-multiplexes ports
that otherwise carry the full weight stream, the conservative busiest-port
floor is:

| State-striped ports | Weight plus state floor |
|---:|---:|
| 2 | 3.519M cycles |
| 4 | 3.126M cycles |
| 8 | 2.930M cycles |
| 16 | 2.831M cycles |
| 32 | 2.782M cycles |

Four state ports are therefore the minimum sensible end architecture for a
3.3M-cycle target. Eight ports are the likely bandwidth/routing compromise for
approaching 3.0M cycles. The exact retained choice must follow measured port
occupancy and implementation results.

## 2. Target Outcome

| Target | Cycles/token | 100 MHz | 130 MHz |
|---|---:|---:|---:|
| Conservative streamed FP32 | 3.5M | 35.0 ms | 26.9 ms |
| Primary target | 3.3M | 33.0 ms | 25.4 ms |
| Stretch target | 3.0M | 30.0 ms | 23.1 ms |

The recorded GDN GPU reference is about 34.9 ms/token. The primary target can
beat it at 100 MHz, while a timing-closed 130 MHz image would provide useful
margin. The short-context 9--12 ms transformer GPU result is a different and
much harder target; dense exact FP32 weight traffic cannot reach it without
compression, sparsity, or additional hardware bandwidth.

## 3. Stage 0 — Preserve Iter36

Keep the Iter36 production source and its one-command 100 MHz build as the
rollback point:

- 32 independent HBM weight masters and MM2S readers;
- 16 two-port GEMV clusters with MAC II=4;
- BRAM weight/result FIFOs, activation ripple, and SLR-local collectors;
- one shared physical GEMV instance;
- local transient activations and on-chip LM-head argmax;
- 16-lane, head-local recurrent state fused with packed HBM transfer;
- exact 64-token parity; and
- zero failed/unrouted nets, zero overlaps, and closed setup/hold timing.

Do not overwrite or silently repurpose the successful Iter36 recipe.
Also preserve the faster exact 115.7 MHz artifact produced by
`bash c_impl/build_iter36_headlocal.sh 130`. It is a positive performance
milestone, but because Vivado auto-scaled the failed 130 MHz request, it must
not be described as 130 MHz timing closure.

## 4. Stage 1 — Unified Layout and Isolated State Proof

Prepare a versioned weight/state layout without launching a full hardware
build:

1. Combine Q, K, V, and the attention gate into one logical `QKVG` matrix.
2. Store its output rows in head-major order so one head's Q/K/V/gate values
   become available together.
3. Combine MLP gate/up weights into pair- or chunk-interleaved `GU` storage.
4. Make recurrent-state striping configurable for 2, 4, and 8 selected HBM
   ports.
5. Retain each output row's original inner-dimension and FP32 accumulation
   order. Native tools must reconstruct every original tensor and state word
   exactly from the new format.

In an isolated recurrent-state synthesis, test:

- 32 state columns per local compute cycle;
- three 256 KiB head buffers with rotating prefetch, compute, and writeback
  ownership;
- early old-state reads and deferred updated-state writes; and
- exact multi-token recurrent-state and token parity.

Three 32-bank head buffers are expected to use about 96 recurrence URAMs. The
whole kernel should remain around 112 URAMs, below the 144 URAMs of the routed
Iter35 image.

Acceptance gates:

- approximately 43.8--48K recurrent cycles/layer, down from 76.7K;
- no more than roughly 300 additional DSPs;
- bounded local control/fanout suitable for SLR2 placement; and
- no change to GEMV MM2S II=1 or cluster II=4.

## 5. Stage 2 — Iter37: Four-Port, 32-Lane Recurrent Pipeline

Move recurrent state onto four physically suitable HBM ports selected from
the routed endpoint locations. Place the new state buffers and service logic
away from the already dense SLR0 and, where the device topology permits, local
to the recurrent engine in SLR2.

The selected ports must have persistent service actors that own their
`m_axi` interfaces. Do not let independent dataflow tasks access the same
pointer. Each actor arbitrates:

1. weight traffic with a hard compute deadline;
2. early old-state prefetch; and
3. deferred updated-state writeback.

The recurrence permits long lookahead. Old state for a future layer is
independent of the current layer activation, and an updated head need only be
committed before that layer executes for the next token. Use those facts to:

- fetch layer `L` head 0 during preceding auxiliary work;
- fetch head `h+1` while head `h` computes when port service permits;
- retain an updated head in the writeback buffer while later compute proceeds;
  and
- flush all pending writes before kernel completion.

Expected result:

- recurrent attention: 1.05--1.20M cycles/token;
- whole kernel: 5.15--5.35M cycles/token; and
- 51.5--53.5 ms/token at 100 MHz.

This is the first new full 100 MHz hardware candidate. If it does not route,
apply fallbacks in this order: reduce three head buffers to two, reduce four
state ports to two, retain 32-lane arithmetic if possible, and only then return
to 16 recurrent lanes. Do not reduce GEMV port or cluster count as the first
response.

## 6. Stage 3 — Persistent Fused GEMV Service

Convert the one shared GEMV engine into a command-driven actor while retaining
its successful physical topology:

- the same 32 MM2S readers and 16 clusters;
- the same BRAM FIFO decoupling;
- the same activation ripple and 4/6/6 SLR-local result collection; and
- one physical compute engine, not one engine per call site.

A command identifies the matrix section, head or chunk, output tag, dimensions,
and store/accumulate mode. First run fused `QKVG` and `GU` commands serially to
prove that the new layout does not regress weight throughput or exact parity.

The direct saving is expected to be only 0.04--0.10M cycles/token. The purpose
of this stage is to remove call and whole-result barriers so later actors can
consume tagged partial results. Do not spend a full hardware build on source
cleanup alone; proceed after native and integrated csynth gates pass.

## 7. Stage 4 — Iter38: Head-Streamed Attention

Build a bounded attention dataflow pipeline:

```text
QKVG head
    -> Q/K/V convolution for that head
    -> recurrent attention
    -> output norm and gate
    -> output-projection partial accumulation
```

Expected steady work after the 32-lane state change is:

| Actor | Approximate cycles/head |
|---|---:|
| QKVG weight stream | 4,096 |
| Three Q/K/V convolutions | 3,250 |
| Recurrent attention | 5,474 |
| Output norm and gate | 629 |
| O-projection input-head chunk | 1,024 |

Recurrence should become the cadence limiter. Convolution, output norm/gate,
and O-projection chunks should fit behind it.

For output projection, carry partial accumulators across input-head chunks,
visit chunks in increasing input-index order, and preserve the existing
four-way FP32 accumulation and final combination order. Remove the full
attention-vector GEMV result barrier without changing numerical behavior.

Target: 3.6--4.0M cycles/token, or 36--40 ms at 100 MHz. This transition is
large enough to justify its own 100 MHz implementation and on-card measurement.

## 8. Stage 5 — Iter39: Stream the MLP

Use the interleaved `GU` layout to process bounded intermediate chunks:

1. produce a gate/up pair chunk;
2. apply SwiGLU immediately;
3. feed it into MLP-down output partial accumulators; and
4. reuse the local chunk storage.

The shared GEMV engine still reads `GU` and MLP-down weights serially, so this
does not reduce their dense weight bytes. It removes the standalone 321K-cycle
SwiGLU pass and the full 5,632-element gate/up materialization.

Expected additional saving: 0.25--0.35M cycles/token. Target after this stage:
3.3--3.7M cycles/token, or 33--37 ms at 100 MHz.

## 9. Stage 6 — Hide Remaining Auxiliary Work

Make only changes that shorten the measured critical schedule:

- load RMSNorm weights as `Pack16` instead of the current scalar 2,048-cycle
  loop;
- preload compact normalization, convolution, and tiny-GEMV weights;
- run A/B tiny GEMVs alongside QKVG when their activation and weights are
  locally available;
- accumulate the next RMSNorm sum of squares while the preceding residual
  output is produced; and
- overlap convolution-tail traffic through the persistent port services.

Expected additional saving: 0.10--0.20M cycles/token.

Do not widen convolution, output-gate, norm, or SwiGLU arithmetic simply
because resources are available. Once these actors are streamed, retain a
wider variant only when dataflow stall reports prove it limits cadence and the
integrated saving justifies its physical cost.

## 10. Stage 7 — Expand State Striping Only When Measured

The main state-parallelism change is already in Stage 2. This stage tests only
the external striping width after the four-port pipeline has real occupancy
data.

- Keep four ports if state service is hidden and the implementation is near
  the expected schedule.
- Move from four to eight ports when the projected retained saving is at least
  0.15M cycles/token.
- Avoid 16 or 32 read/write state ports unless a later measurement proves the
  diminishing bandwidth gain is worth the additional AXI control and routing.

Expected final exact-FP32 range:

- four state ports: 3.2--3.5M cycles/token;
- eight state ports: 3.0--3.3M cycles/token.

## 11. Stage 8 — Build at 100 MHz, Then Raise Frequency

For each major candidate:

1. build the native executable;
2. pass fast and full exact parity;
3. synthesize isolated changed actors and the integrated kernel;
4. reconstruct a dimension-correct token schedule;
5. implement at 100 MHz;
6. require zero failed/unrouted nets, zero node overlaps, WNS >= 0, and WHS >= 0;
7. pass the eight-token on-card smoke test; and
8. measure the exact 64-token fixture and compare it with csynth.

If measured cycles exceed the reconstructed schedule by more than 5%, profile
actor wait cycles and HBM service occupancy before changing parallelism.

Attempt 130 MHz only after the same source closes and is measured at 100 MHz.
Re-extract the new critical paths rather than blindly replaying an old timing
fix. If cluster enable/control cones remain critical, localize and replicate
them per cluster or clock region before placement. Frequency changes wall time
but do not count as cycle reduction.

## 12. Rejected or Deprioritized Directions

- **Direct AXI reads inside clusters:** previously worsened local congestion;
  preserve MM2S/FIFO decoupling.
- **All recurrent state on chip:** the persistent 48 MiB state does not fit
  alongside the accelerator.
- **Eight time-multiplexed GEMV clusters:** routability fallback only; it gives
  up simultaneous arithmetic consumption and raises GEMV cycles.
- **Source `for` loops replacing repeated calls:** maintainability only unless
  the generated hardware hierarchy changes.
- **Q/K/V merge as a bandwidth multiplier:** all 32 ports are already active;
  fusion enables streaming but does not reduce bytes.
- **One whole-layer dataflow region:** creates excessive control fanout and
  deadlock risk. Use bounded port-service, attention, and MLP regions.
- **More GEMV arithmetic clusters:** the dense engine is already bandwidth
  limited and physically dense.
- **Frequency-first optimization:** helpful for milliseconds, irrelevant to
  the cycle target and currently timing-risky.

## 13. Beyond Exact FP32

The dense FP32 weight floor prevents a large reduction below 3M cycles. Further
steps require a separate accuracy contract:

- BF16 or FP16 weights;
- calibrated INT8 weights and dequantization;
- quantized recurrent state;
- structured sparsity or low-rank projections; or
- speculative multi-token decoding.

These should follow, not replace, the exact-FP32 streaming baseline.

## 14. Iteration Discipline

Every stage and sub-variant must be appended to `optimization_log.md`, whether
it is retained, rejected, fails synthesis, fails routing, fails timing, or is
stopped. Positive results must update the authoritative architecture and the
relevant block documentation at the evidence level actually reached. Complete
the corresponding focused commits before beginning the next named iteration,
as required by the repository-level `AGENTS.md`.
