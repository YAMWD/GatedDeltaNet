# Cycle-First Optimization Roadmap After Iter38

**Status:** Iter38E unified layouts and concurrent four-port recurrent state are
implemented, routed, timing-closed at 100 MHz, and on-card validated. The
bounded overlap stages below remain proposed.

**Current reference:** Iter38E preserves the 32-port/16-cluster FP32 GEMV
topology and measures **4.708M cycles/token** (47.079335 ms at 100 MHz) with
exact 64-token parity. It closes timing with design WNS/WHS +0.003/+0.006 ns.
Iter37D remains the immediate rollback at 5.145M cycles.

**Primary objective:** minimize exact single-token decode cycles first. Perform
only the 100 MHz implementation checkpoints required to verify real cycles and
routability while the architecture is evolving. Begin deliberate frequency
recovery only after the minimum-cycle exact-FP32 architecture is stable.

## 1. Measured Baseline

Iter38 stripes the 48 MiB recurrent state over tails appended to weight ports
28--31 and advances all four ports concurrently. Compact 64-bit low/high state
pairs let the same 32 URAM banks serve both halves without a second address.

| Metric | Iter36 | Iter37 | Iter38 |
|---|---:|---:|---:|
| Clock used on card | 100 MHz | 100 MHz | 100 MHz |
| Mean cycles/token | 5.958M | 5.145M | **4.708M** |
| Mean latency | 59.578 ms | 51.451 ms | **47.079 ms** |
| Recurrent HLS cycles/layer | 76.7K | 43.9--44.1K | **27.3--27.5K** |

The dimension-correct Iter37 schedule is:

| Component | Iter37 cycles/token |
|---|---:|
| Large GEMV | 2,785,524 |
| Recurrent attention | 1,057,944 |
| Q/K/V convolution | 624,744 |
| SwiGLU | 321,048 |
| RMSNorm | 139,405 |
| Output norm/gate | 128,712 |
| Tiny A/B GEMV | 103,584 |
| Activation handoff/copy/argmax | 54,069 |
| **Reconstructed total** | **5,215,030** |

Iter38's raw HLS minimum is 2,744,183 cycles, but about 0.816M of its nominal
reduction is QKVG/GU dynamic-call accounting rather than eliminated GEMV work.
The defensible fixed-trip recurrent reduction is 0.397M cycles and the measured
reduction is 0.437M. Continue using on-card kernel cycles for roadmap decisions,
not the raw HLS top minimum alone.

## 2. Hard Floors and Target Outcome

The dense FP32 weight stream is approximately 2,733,056 cycles/token per fully
occupied weight port. Matrix merging cannot reduce these bytes; it only exposes
smaller results early enough to overlap downstream work.

Each token also reads and writes 48 MiB of recurrent state, or 1,572,864
Pack16 transfers. When state shares ports with the weight stream, the
conservative busiest-port traffic floor is:

| State-striped ports | Weight plus state floor |
|---:|---:|
| 2 | 3.519M cycles |
| 4 | 3.126M cycles |
| 8 | 2.930M cycles |
| 16 | 2.831M cycles |
| 32 | 2.782M cycles |

Four ports remain the minimum sensible exact-FP32 architecture for a roughly
3.3M-cycle target. Eight ports are a conditional stretch lever; 16 or 32 ports
have too little theoretical benefit to justify their routing cost without new
measurements.

| Outcome | Cycles/token | 100 MHz | 115 MHz | 130 MHz |
|---|---:|---:|---:|---:|
| Current Iter38 | 4.708M | 47.08 ms | 40.94 ms | 36.22 ms |
| Conservative streamed FP32 | 3.5M | 35.0 ms | 30.4 ms | 26.9 ms |
| Primary target | 3.3M | 33.0 ms | 28.7 ms | 25.4 ms |
| Stretch target | 3.0M | 30.0 ms | 26.1 ms | 23.1 ms |

The recorded GDN GPU reference is about 34.9 ms/token. A 3.3M-cycle image can
beat it at 100 MHz. The 9--12 ms short-context transformer GPU result is outside
the dense exact-FP32 traffic envelope and requires compression, sparsity, or
additional hardware bandwidth.

## 3. Completed Foundations

### Stage 0 — Iter36 rollback

Preserve both Iter36 artifacts:

- the exact, timing-closed 100 MHz image at 59.578 ms/token; and
- the exact auto-scaled 115.7 MHz image at 51.844 ms/token.

They retain the lower-density 16-lane head-local recurrence and remain the
rollback when a later streaming architecture fails native parity, synthesis,
routing, timing, or on-card performance.

### Stages 1--2 — Iter37 four-port/32-lane recurrence

Completed and retained evidence:

- contiguous `.gdnstate` ABI preserved;
- host scatters aligned state words into four 12 MiB shard tails on ports
  28--31;
- 32 recurrent columns processed per cycle;
- all four state read/write loops at II=1;
- recurrent call 43,873--44,081 cycles/layer;
- whole-kernel resources: 1,511 RAMB18, 3,627 DSP, 851,903 FF, 912,694 LUT,
  and 48 URAM;
- zero failed/unrouted nets and zero node overlaps after assigning the recurrent
  hierarchy to SLR2; and
- exact 64-token on-card result at 5.145M cycles/token.

Iter37 does not yet contain persistent state service actors or rotating
prefetch/compute/writeback head buffers. It reads the low port pair and high
port pair sequentially inside recurrence. Those mechanisms remain enabling
work for head-level overlap.

### Stage 3A — Iter38 unified layouts and concurrent state

Completed and retained evidence:

- one head-major QKVG command replaces four projection commands;
- one pair-interleaved GU command replaces two MLP commands;
- a native full-byte validator checks every sharded projection float;
- all four recurrent state ports read and write concurrently at II=1;
- compact 64-bit low/high pairs preserve the 48-URAM whole-kernel total;
- recurrent latency is 27,329--27,537 cycles/layer;
- MM2S II=1 and GEMV MAC II=4 are unchanged;
- resources are 1,527 RAMB18, 3,629 DSP, 857,343 FF, 929,941 LUT, and
  48 URAM;
- route completed with zero failed/unrouted nets and zero node overlaps;
- 100 MHz timing closed at WNS/WHS +0.003/+0.006 ns; and
- exact 64-token hardware performance is 4.708M cycles/token, a 1.093x
  speedup over Iter37D.

The layout merge is an enabling change rather than a weight-bandwidth saving:
the four QKVG tensors and two GU tensors still perform the same FP32 MACs and
stream the same bytes. Its value is exposing head/chunk order for bounded
downstream overlap.

## 4. Stage 3B — Persistent Services

The layouts and concurrent four-port state traversal are complete. The next
foundation is turning them into independently scheduled bounded services.

### Matrix layout

1. Head-major QKVG: **complete**.
2. Original inner-dimension and FP32 accumulation order: **preserved**.
3. Pair-interleaved GU: **complete**.
4. Native full-byte shard reconstruction gate: **complete**.

Merging matrices does not reduce weight traffic. Its purpose is to make one
head or MLP chunk visible before the complete projection finishes.

### Persistent GEMV actor

Convert the single shared GEMV engine into a command-driven service while
retaining:

- 32 MM2S readers and 16 two-port clusters;
- BRAM FIFO decoupling;
- activation ripple and 4/6/6 SLR-local collectors; and
- one physical GEMV graph.

A command identifies matrix section, head/chunk, dimensions, output tag, and
store/accumulate mode. First execute QKVG and GU commands serially and require
unchanged MM2S II=1, cluster II=4, exact parity, and weight byte count.

### Persistent state services

Give each selected state port one owner that arbitrates weight reads, old-state
prefetch, and updated-state writeback. Use two or three rotating 256 KiB
head-buffer slots so all four ports can transfer concurrently while the 32-lane
recurrent core processes another head. Flush pending writes before kernel exit.

Acceptance gates:

- exact multi-token state and token parity;
- no new AXI master;
- no GEMV II regression;
- no more than 0.10M added cycles/token before overlap is enabled; and
- bounded FIFOs and control fanout, with no whole-layer dataflow region.

Expected direct saving is only 0.04--0.10M cycles. The stage is retained only
when it enables tagged head/chunk consumption without regressing the current
4.708M-cycle behavior.

## 5. Stage 4 — Head-Streamed Attention

Build a bounded per-head pipeline:

```text
QKVG head
    -> Q/K/V convolution for that head
    -> recurrent attention
    -> output norm and gate
    -> output-projection partial accumulation
```

Approximate steady work after Iter38 is:

| Actor | Approximate cycles/head |
|---|---:|
| QKVG weight stream | 4,096 |
| Three Q/K/V convolutions | 3,250 |
| Recurrent attention | 3,416--3,442 |
| Output norm and gate | 629 |
| O-projection input-head chunk | 1,024 |

QKVG streaming is now the largest individual per-head actor, with convolution
and recurrence close behind. Prefetch head `h+1` while head `h` computes and
defer its writeback while later heads proceed. For output
projection, retain partial accumulators across input-head chunks, visit chunks
in increasing input-index order, and preserve the existing FP32 partial and
final-combination order.

Target: **3.6--4.0M cycles/token**. This is the first remaining change large
enough for an explicit 100 MHz implementation and exact on-card measurement.
Do not attempt 115 or 130 MHz during this stage.

## 6. Stage 5 — Chunk-Streamed MLP

Use the interleaved GU layout to process bounded intermediate chunks:

1. produce one gate/up pair chunk;
2. apply SwiGLU immediately;
3. stream the result into MLP-down partial accumulators; and
4. reuse the chunk storage.

The shared GEMV engine still reads GU and MLP-down weights serially, so dense
weight bytes do not change. The gain comes from hiding or removing the 321K
standalone SwiGLU pass and full 5,632-element materialization.

Expected additional saving: **0.25--0.35M cycles/token**. Target after this
stage: **3.3--3.7M cycles/token**. Build and measure at 100 MHz only after native
and integrated csynth prove exactness and a material schedule reduction.

## 7. Stage 6 — Hide Remaining Auxiliary Work

Retain only changes that shorten the measured critical schedule:

- load RMSNorm weights as Pack16 rather than through scalar 2,048-cycle loops;
- preload compact norm, convolution, and tiny-GEMV weights;
- execute A/B tiny GEMVs when QKVG has the required local activation;
- accumulate the next RMSNorm sum of squares during the preceding residual
  production; and
- overlap convolution-tail traffic through persistent port services.

Expected additional saving: **0.10--0.20M cycles/token**. Do not widen norm,
convolution, output-gate, or SwiGLU arithmetic without a measured actor-stall
reason. Stop retaining auxiliary variants whose integrated saving is below
0.05M cycles/token or whose resource cost threatens the major pipeline.

## 8. Stage 7 — Conditional State-Port Expansion

Use real service occupancy from the streamed four-port design:

- keep four ports when state service is hidden or the design reaches 3.2--3.5M
  cycles;
- test eight ports only when the projected retained saving is at least 0.15M
  cycles/token; and
- reject 16/32-port state service unless new measurements overturn the small
  theoretical gain.

Expected final exact-FP32 range:

- four state ports: **3.2--3.5M cycles/token**;
- eight state ports: **3.0--3.3M cycles/token**.

## 9. Cycle-Phase Validation Gates

Every sub-variant must pass the evidence level appropriate to its size:

1. native build and exact fast parity;
2. full 32-token native parity before integrated synthesis;
3. isolated actor synthesis when interfaces or buffering change;
4. integrated csynth and a dimension-correct token schedule;
5. no GEMV MM2S/cluster II regression;
6. resource comparison against Iter38; and
7. optimization-log entry whether retained or rejected.

For the major attention and MLP milestones, additionally require:

1. explicit 100 MHz implementation;
2. zero failed/unrouted nets and zero node overlaps;
3. non-negative setup and hold slack at 100 MHz;
4. exact eight-token on-card smoke parity; and
5. exact 64-token measurement with 63 non-seed kernel calls.

These 100 MHz builds validate real cycle count and buildability; they are not
frequency-optimization iterations. If on-card cycles exceed the reconstructed
schedule by more than 5%, instrument actor wait cycles and HBM service
occupancy before changing parallelism.

## 10. Final Frequency-Recovery Phase

Begin this phase only after the exact-FP32 cycle architecture reaches a stable
minimum and the last positive cycle result is committed. Do not optimize an
intermediate topology for a higher clock.

The final source must first route and close at 100 MHz. Then:

1. remove obsolete buffers, muxes, and service modes without increasing the
   measured cycle schedule;
2. expose the logical 32-lane recurrence as physically local 16-lane halves
   associated with their state-port pairs;
3. localize bounded stream actors and their buffers near relevant HBM/SLR
   endpoints;
4. replicate high-fanout enables, resets, and FIFO address/control cones per
   actor, cluster, or clock region;
5. pipeline long paths only when added fill/drain cycles are negligible and
   included in the cycle comparison; and
6. try explicit 110 MHz, then 115 MHz, and only then 130 MHz when checkpoint
   timing predicts it is plausible.

Re-extract critical paths from the final architecture. Do not blindly reuse a
fix from Iter35--37: Iter38 closes 100 MHz with only +0.003 ns design WNS, so
the final cycle-reduced netlist must be analyzed from its own routed checkpoint.

Frequency changes do not count as cycle improvements. If a timing repair adds
cycles, retain it only when the achieved frequency produces a lower measured
latency and the exact result remains stable.

## 11. Stop Conditions and Expected Endpoint

For exact FP32, stop cycle work when one of these holds:

- the measured schedule reaches the four- or eight-port traffic floor within
  5%;
- the next safe lever projects less than 0.05M cycles/token;
- a proposed gain requires changing FP32 accumulation order or exact parity;
  or
- routing/resource cost forces a larger loss elsewhere than the projected
  saving.

The realistic endpoint is **3.2--3.5M cycles** with four state ports and the
stretch endpoint is **3.0--3.3M cycles** with eight. At a recovered 115 MHz,
those ranges correspond to approximately 26.1--30.4 ms/token.

## 12. Rejected or Deprioritized Directions

- **Direct AXI reads inside GEMV clusters:** previously worsened local
  congestion; retain MM2S/FIFO decoupling.
- **All recurrent state on chip:** the persistent 48 MiB state does not fit.
- **Eight time-multiplexed GEMV clusters:** routability fallback only; it raises
  GEMV cycles.
- **Repeated-call cleanup with source loops:** maintainability only unless the
  synthesized topology changes.
- **Q/K/V merge as a bandwidth multiplier:** it enables streaming but cannot
  reduce dense weight bytes.
- **One whole-layer dataflow region:** excessive control fanout and deadlock
  risk; use bounded head/chunk regions.
- **More GEMV arithmetic clusters:** the current engine is weight-bandwidth
  limited and already physically dense.
- **Frequency-first work:** useful only after the cycle topology is final.

## 13. Beyond Exact FP32

The dense FP32 weight floor prevents a material reduction below roughly 3M
cycles. Further improvement requires a separate accuracy contract:

- BF16/FP16 or calibrated INT8 weights;
- quantized recurrent state;
- structured sparsity or low-rank projections;
- speculative multi-token decoding; or
- additional independent hardware bandwidth.

These follow the exact-FP32 streaming baseline rather than replacing its
correctness contract.

## 14. Iteration Discipline

Every stage and sub-variant must be appended to `optimization_log.md`, whether
it is retained, rejected, fails synthesis, fails routing/timing, or is stopped.
Every positive iteration must also update this roadmap: mark the completed
stage, replace estimates with evidence, rebase the measured starting point,
and revise remaining targets or dependencies. Update `architecture.md` and the
relevant block document at the evidence level actually reached, then make the
focused positive commits required by `AGENTS.md`.
