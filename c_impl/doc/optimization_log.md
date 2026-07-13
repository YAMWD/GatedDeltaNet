# GDN HLS Optimisation Log

This document is a chronological record. The v1-v7 and prefill sections retain
their original context, but they do not describe the current kernel. The active
`gdn_forward` is decode-only and uses sharded GEMV; see
[architecture.md](architecture.md). Section dates are taken from this file's
git history where available.

## Current decode architecture

*Updated: 2026-07-13.*

The current FPGA path forwards one token at a time from GPU-exported recurrent
and convolution state. Eight compact weight shards feed eight independent
512-bit HBM readers in `gdn_gemv`. All large layer projections and `lm_head`
use GEMV; A/B use `gdn_gemv_tiny`. No tiled, systolic, or weight-stationary
matmul is called by `gdn_forward`.

Documentation map:

- [README.md](README.md) -- current versus historical document index.
- [architecture.md](architecture.md) -- authoritative decode architecture.
- [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md) -- integrated
  GEMV evolution and measured decode results.
- The remaining sections in this file retain the retired tiled, systolic, and
  weight-stationary prefill results as historical measurements.

## Decode GEMV routing weakness: high-fanout dataflow

*Logged: 2026-07-04.*

The current decode GEMV experiments expose a routing weakness that C-synthesis
throughput estimates do not capture. A load/compute/store dataflow GEMV can look
clean architecturally, but it creates high-fanout activation distribution and
wide inter-process wiring: the loaded activation vector must reach many parallel
HBM reader/MAC lanes or tiles, while the store/collector side adds cross-region
control and stream paths.

This is a pain point for the current GDN design. A monolithic load -> compute
-> store GEMV dataflow region tends to concentrate broadcast, stream, and AXI
control routing around the GEMV tile array, so it can be hard to route even at
small tile counts such as `N=16`. Future GEMV designs should treat routing as a
primary constraint, preferring physically local tiles, SLR-local activation
broadcast or local activation copies, direct per-bank weight reads, and minimal
global collectors over one large fanout network.

## Routed 32-port mono-kernel GEMV milestone

*Logged: 2026-07-13.*

The isolated `c_impl/microbench/gemv_tile/gemv_full` design resolves the earlier
unroutable topology. It retains all 32 independent 512-bit HBM weight ports but
groups compute into eight four-port clusters, distributes those clusters 2/3/3
across SLR0/1/2, leaves AXI adapters at their natural HMSS placement, and merges
outputs through SLR-local collectors. Vivado completed routing with zero routing
errors and zero unrouted nets.

The 150 MHz implementation missed setup timing by 0.985 ns and was encoded at
130.6 MHz. On U55C it sustained **263.063 GB/s** and **131.531 GFLOP/s** on the
large saturation shape, or **98.353%** of the 267.469 GB/s clock-rate ceiling.
Synthetic parity and real layer-0 `q_proj` parity both passed with zero maximum
absolute error. The small real projection reached 139.964 GB/s host-visible
throughput because launch/completion overhead is significant for a 16.8 MB GEMV.

The remaining timing problem is physical, not arithmetic: SLR1 uses 90.47% of
CLBs and 96.88% of BRAM, the SLR0-SLR1 boundary uses 95.03% of available SLLs,
and the worst setup path is 97.4% routing delay. Driver replication is therefore
unlikely to be safe. A future 150 MHz attempt should split each four-port cluster
into smaller independently controlled clusters while preserving the 32-port
read rate.

## Weight-traffic optimization (on-card, hardware-measured)

*Logged: 2026-06-01; next-bottleneck note updated on 2026-06-02.*

The csynth latency above is a fixed-latency estimate that hides HBM bandwidth
stalls. The real U55C run was **memory bound on weight traffic**: the systolic
chain re-read the weights ~128x (once per 16-row token stripe) in 64-byte
non-bursted transfers (1.55 % efficient), moving **507 GB at 387 MB/s ~= 22 of
the 26 minutes**. `gdn_matmul_2d` (activation-stationary 256-row block +
bursting) cut weight re-reads 16x. Measured on hardware, same wikitext run:

| Metric | Systolic chain | Stage 1 | Stage 2 |
|--------|---------------:|--------:|--------:|
| Application runtime | 25.9 min | 6.5 min | **5.2 min** |
| Kernel time | — | 4.7 min | **3.45 min** |
| Matmul weight bandwidth | 387 MB/s | 388 MB/s | **5,405 MB/s** |
| Weight data read | 507.8 GB | 32.9 GB | 32.8 GB |
| Wikitext perplexity | 15.81 | 15.81 | **15.81** |

Stage 1 = activation-stationary blocking + bursting (16x fewer weight
re-reads). Stage 2 = 512-bit weight reads (aligned base + integer Pack16
offset) on a dedicated `mem_weights_mm` bundle — lifted the weight port from
388 MB/s to 5.4 GB/s (14x), so the 32.8 GB of weights now read in ~6 s (was
~82 s). Weight traffic is no longer the bottleneck.

Next bottleneck after Stage 2 (confirmed on-card): the 207 s kernel splits
between matmul compute (~107 s, 256 MAC/cycle FP32 @ 100 MHz) and the gmem
activation port (HBM[0] single channel: 78 GB reads + 7.5 GB of 11-byte
writes).

## Activation-memory phases A & B (on-card, hardware-measured)

*Logged: 2026-06-02.*

| Metric | Stage 2 | Phase A | Phase B |
|--------|--------:|--------:|--------:|
| Application runtime | 5.2 min | 4.4 min | **4.2 min** |
| Kernel time | 207 s | 153 s | **141 s** |
| Δ kernel vs prev | — | 1.35x | 1.09x |
| Wikitext perplexity | 15.81 | 15.81 | **15.81** |

- **Phase A** (`00e3264`): Pack16-widen the three scalar activation stages —
  `rmsnorm`, depthwise `conv`, `output_norm` read/wrote 1 float/cycle (the
  11-byte gmem writes). Rewrote to index the Pack16 base by integer offset and
  process 16 lanes/beat. gmem writes 11 B → 35 B/transfer, 183 → 642 MB/s.
- **Phase B** (`fda62c7`): split activations off the single gmem/HBM[0] master
  into 3 AXI masters (`gmem_x`, `gmem_qkv`, `gmem_mlp`) on distinct HBM channels;
  weights compressed to HBM[10:31]. The modest 1.09x and the 8–16% port
  utilization confirmed the stages are sequential/latency-bound, not
  bandwidth-bound — i.e. **the memory wall is solved; compute is the floor.**

Net A+B: kernel 207 → 141 s; end-to-end (with Stages 1/2) 25.9 min → 4.2 min
(~6.2x). HEAD is Phase B (`fda62c7`).

## Phase C (PE-grid widening) — attempted, reverted

*Logged: 2026-06-02.*

Phase C widened the 16×16 grid (256 MAC/cycle) to cut the prefill compute floor.
Both configs built were design-valid (csynth II=1, parity PASS) but failed on
infrastructure: **32×32** → BRAM 4054 > 4032 RAMB18; **32×16** → `route_design`
SLR1–2 SLL congestion at 102%. It was **reverted** (`git reset` to Phase B,
grid kept at 16×16) because the target shifted to **decode**, where grid width
is the wrong lever (decode is weight-bandwidth-bound, and the 16×16 grid is
already ~19x over-provisioned for the available weight bandwidth).

The subsequent decode pivot replaced the grid with a GEMV datapath and
multi-channel weight readers. Of all the above, only Stage 2's 512-bit weight
read transfers to decode; the rest is prefill-specific.

Historical post-v7 single-attention synthesis snapshot:

*Snapshot date: 2026-05-11.*

| Metric | v7 tiled matmul | Systolic matmul experiment |
|--------|----------------:|------------------------:|
| Top-level latency | 141.03 G cycles | 3.976 G cycles |
| Speedup | 1.0x | 35.5x vs v7 |
| Timing slack | 0.00 ns | -0.04 ns |
| BRAM_18K | 322 (7 %) | 1602 (39 %) |
| DSP | 1042 (11 %) | 4690 (51 %) |
| FF | 209.8 k (8 %) | 848.9 k (32 %) |
| LUT | 237.0 k (18 %) | 932.0 k (71 %) |

Historical pre-decode full-model synthesis snapshot:

*Snapshot date: 2026-05-11.*

| Metric | Prefill-era `gdn_forward` |
|--------|----------------------:|
| Top-level latency | 129.686 G cycles |
| Timing slack | -0.04 ns |
| BRAM_18K | 1058 (26 %) |
| DSP | 2847 (31 %) |
| FF | 508.4 k (19 %) |
| LUT | 580.3 k (44 %) |

The full-model report is a reused hardware datapath over a 24-iteration layer
loop. It does not instantiate 24 physical copies of the layer.

## Headline numbers

*Logged: 2026-05-07.*

| Metric                          | Baseline (v0) | Final (v7) | Δ |
|---------------------------------|---------------|------------|---|
| Top-level latency (cycles)      | 190.96 G      | **141.03 G** | −26 % |
| Top-level latency (ns @ 100 MHz)| 1.910 × 10¹²  | 1.410 × 10¹² | −500 ms |
| Timing slack                    | −0.46 ns      | **0.00 ns** | +0.46 ns (closes timing) |
| BRAM_18K                        | 938 (23 %)    | 322 (7 %)   | −616 |
| DSP                             | 317 (3 %)     | 1042 (11 %) | +725 |
| LUT                             | 172,549 (13 %)| 237,048 (18 %) | +65 k |
| FF                              | 96,071 (3 %)  | 209,843 (8 %) | +114 k |
| URAM                            | 0             | 0           | — |
| II violations                   | 7             | **0**       | −7 |
| Single-layer parity max abs diff| 9.5 × 10⁻⁷    | **1.2 × 10⁻⁶** | within 1 × 10⁻³ tolerance |

(All numbers from `GDN_single_attn/solution2/syn/report/csynth.rpt`, target
`xcu55c-fsvh2892-2L-e`, Vitis HLS 2022.1 csynth at a 10 ns target clock.)

The latency reduction is modest because the matmul still dominates (7 × 20.18 G
cycles ≈ 141.3 G of the 141.5 G total); the matmul's fundamental bottleneck is
the per-tile load/store overhead through the shared `gmem` AXI port. Lifting
that further requires structural changes (dataflow + streaming GEMM, tier-2
work).

What v1–v7 *did* achieve (U55C v7 vs v0 baseline):
- All compute-bound II violations are gone — every accumulator that was
  scalar-dependence-bound now runs at II=1.
- Top-level timing **closes** at 100 MHz (slack 0.00 ns vs −0.46 ns at v0).
- The conv1d phase is 86× faster (759 M → 8.75 M cycles per call).
- The output-norm phase is 40× faster (685 M → 17.24 M).
- The recurrent-attention phase is 11 % faster (175.9 M → 157.29 M).
- The matmul inner compute loop runs at II=1 (was II=2).
- BRAM_18K usage drops by ~3× (938 → 322) — HLS uses a denser per-partition
  state mapping on U55C than it did on the prior VU11P iteration runs.

## Iteration map (per pass)

*Logged: 2026-05-07.*

Each iteration was verified for parity (`gdn_attn_test`) before re-running
`vitis_hls -f test_single_GDN_attn.tcl`. Numbers below are after the change
of that iteration only.

### v1 — local-fix sweep
*Goal: clean up obvious II offenders without restructuring.*

- **`delta_out` → `delta_out` + `delta_drain`** (`gdn_recurrent_attention`):
  16 simultaneous m_axi stores on shared `gmem` forced II=16. Split into a
  pure-on-chip compute pass (II=1, P_K=16) into `out_loc[256]`, and a separate
  drain loop (II=1, sequential) writing to AXI.
  - Result: 371 → 358 cyc per call.

- **`onorm_sq` / `onorm_gate` → on-chip buffer + multi-lane partial accs**
  (`gdn_output_norm_and_gate`): The original loop read `attn_head[i]` via
  m_axi inside the same iteration that wrote it, producing a distance-1 carried
  AXI dep (II=160). Added local `attn_loc[256]`, `gate_loc[256]`, pre-loaded
  shared `weight_loc[256]` once outside the token loop.
  - Result: `onorm_gate` 40,962 → 344 cyc; `onorm_sq` 846 → 587 cyc.

- **`mm_comp_k` partial accumulators** (`gdn_matmul`): tried 8-lane partial
  sums to break the FP32 fadd carried dep on `local_out[r][c]`. HLS muxed the
  lane-indexed array into a single mux-register and the dep tracker re-detected
  it. Net effect was modest — relied on auto-flatten.
  - Result: II=2 unchanged but the auto-flattened body trip went 16×16 → 256
    iters, saving fill overhead.

- **`load_qk`, `dot_alpha` partial accumulators** — same pattern, same outcome
  (II=3→2 for `load_qk` from dropping `double` to `float`).

- **`conv_kern` unrolled, `conv_col` pipelined**: the original `conv_kern`
  inner loop ran at II=3 (scalar `sum +=`). Unrolling exposed 4 parallel m_axi
  loads of `in[]` and `weights[]`, which the single `gmem` port couldn't
  service in one cycle. `conv_col` failed to pipeline → 759 M → 1.74 G cyc.
  **Reverted in v3**.

### v2 — re-synth, conv refactor
- Re-synth confirmed the v1 changes; conv regression confirmed.
- Removed `conv_kern` unroll; rewrote conv with 4-row sliding window and
  pre-loaded weights, but with the load+shift+compute+write all fused into one
  col loop. The single fused loop touched both `gmem.read` (in) and
  `gmem.write` (out) per iter, and HLS treated this as a carried AXI dep on the
  shared `gmem` port → II=155.
  - Result: conv 759 M → 1.74 G (worse than v1).

### v3 — matmul loop swap, conv 2-phase split
- **Matmul loop nest swap**: changed pipelined dim from `mm_comp_k` (with
  unrolled `c` and a carried dep on `local_out[r][c]`) to `mm_comp_c` (with
  unrolled `k` and a fresh `dot` per iter). HLS still couldn't flatten with
  `mm_comp_r` (warning: "outer loop is not a perfect loop") so each `r` paid
  16 pipeline-fills. HLS also serialised the unrolled `dot += ...` chain
  rather than auto-tree-balancing.
  - Result: matmul 24.3 G → 38.9 G (regression, depth=55).
- **Conv 2-phase split**: separated `conv_load` (m_axi read of `in[]`,
  shift the window) from `conv_compute` (read window + weights, m_axi write
  of `out[]`). With each phase touching only one direction of the gmem port,
  both pipeline at II=1.
  - Result: conv 1.74 G → 8.76 M (200× faster than v2, 86× faster than v1).

### v4 — manual flatten + explicit fadd tree for matmul
*Goal: get matmul to a true II=1 with one combined R×C pipeline.*

- Collapsed `mm_comp_r` and `mm_comp_c` into a single `mm_comp_rc` loop with
  manual `r = rc / MM_TILE_C; c = rc % MM_TILE_C;` indexing (HLS would not
  auto-flatten because the parent tile loops have non-perfect bodies).
- Wrote the 16-input dot product as an explicit balanced 4-level paired-sum
  tree (instead of `for k: dot += ...`), so the critical path is log₂(16)
  fadd stages.
- Added `#pragma HLS dependence variable=local_out type=inter direction=RAW
  false`. The dep is genuinely false because each `rc` iter touches a unique
  `(r, c)` pair, and `dim 2 complete` partition makes each `c` an independent
  register bank.
  - Result: matmul 38.9 G → 20.18 G; per tile_c iter went 1456 → 1271 cyc;
    `mm_comp_rc` II=1, depth 19; total 272 G → 141.5 G.

### v5 — store-products + tree-reduce for `load_qk`/`dot_alpha`/`onorm_sq`
- The lane-indexed partial accumulators kept failing because HLS muxed them
  into a single register. Replaced with the same pattern that worked in
  matmul: each iteration writes a unique scratch element (no carried dep), a
  separate phase reduces.
- Initial reduction used `for (j) { #pragma HLS unroll; sum += arr[j]; }` —
  this *unrolled* but HLS emitted a 256-deep serial fadd chain rather than a
  tree. `onorm_sq_reduce` became 256 × 4 cycles = 1024 cyc per call,
  cancelling the II=1 gains.
  - Result (mixed): `dot_alpha` and `onorm_sq` hit II=1 but the linear
    reduction ate the savings; total essentially unchanged from v4.

### v6 — explicit balanced tree for 256-input reductions
- Wrote `gdn_tree_reduce_256()` as an `inline` helper with 8 explicit levels
  (256 → 128 → 64 → … → 1), each fully unrolled. Used it for `q_sq`, `k_sq`,
  `alpha`, and `sum` in onorm.
  - Result: recurrent 162.15 → **162.15** M (effectively same as v4),
    onorm 33.7 → **17.3** M (best yet, 19 % below v4).

### v7 — split q/k onto separate AXI bundles
- Last remaining II violation was `load_qk` II=2 from HLS 200-885 ("limited
  memory ports"): two simultaneous m_axi reads of `q_head[]` and `k_head[]` on
  the shared `gmem` bundle. Added `bundle=mem_q` to `q` and `bundle=mem_k` to
  `k` on the top-level `gdn_attn_forward`'s m_axi pragmas.
  - Result: `load_qk` II=2 → **II=1**, depth 76 → 4, 587 → 258 cyc per call.
  - Side effect: HLS replicated the matmul and conv into `_1` and `_2`
    instances because some calls now read from `mem_q`/`mem_k` rather than
    `gmem`. Resource cost is real (BRAM +32, DSP +101, LUT +20 k) but
    utilisation stays under 25 %.
  - Recurrent attention 162.2 → **157.29** M (U55C re-measurement),
    top-level 141.51 → **141.03** G.

## Per-loop II status, before vs after

*Logged: 2026-05-07.*

| Loop                     | v0 II | v7 II | Notes |
|--------------------------|------:|------:|-------|
| `mm_comp_k`/`mm_comp_rc` | 2     | **1** | Manual flatten + explicit tree + dep false |
| `conv_kern`/`conv_compute` | 3   | **1** | Pre-buffered weights, 4-row sliding window, 2-phase per row |
| `conv_load`              | n/a   | **1** | New phase, AXI-read only |
| `state_clr`              | 1     | 1     | (unchanged) |
| `load_qk`                | 3 (double) | **1** | Float partials → tree reduce + q/k bundle split |
| `load_v`                 | 1     | 1     | (unchanged) |
| `norm_qk`                | 1     | 1     | (unchanged) |
| `dot_alpha`              | 2     | **1** | Tree reduce |
| `init_ro`                | 1     | 1     | (unchanged) |
| `fused_rd_j_fused_rd_i`  | 1     | 1     | (unchanged) |
| `delta_out`              | 16    | **1** | On-chip out_loc + drain phase |
| `delta_drain`            | n/a   | **1** | New phase |
| `fused_wr_j_fused_wr_i`  | 1     | 1     | (unchanged) |
| `onorm_sq`               | 3 (double) | **1** | Tree reduce |
| `onorm_gate`             | 160   | **1** | On-chip attn/gate/weight buffers |

No II violations remain on U55C. Only `Cannot flatten` informational
warnings (HLS 200-960, harmless). Top-level timing slack is **0.00 ns** at
the 100 MHz target — the design closes timing with zero margin.

## Critical follow-ups after v7

*Logged: 2026-05-07; item 1 updated on 2026-05-11.*

1. **Streaming/dataflow GEMM** -- historically completed by the systolic
   experiment, then superseded by the decode-only GEMV pivot.
2. **`gdn_attn_forward` macro-stage dataflow** — wrap the body
   (matmul → conv → recurrent → onorm → matmul) in a `dataflow` region with
   `hls::stream` between stages. Eliminates the three `attn_conv_copy_*` AXI
   round-trips and overlaps the projection matmuls with the recurrent step.
3. **`a` and `b` AXI bundle split** — these are tiny (504 floats each) but read
   inside `gdn_recurrent_attention`'s scalar-gate prologue; if combined with q/k
   on a wider mux, the gate path could pipeline tighter.
4. **Higher clock target** — top-level slack is 0.00 ns at 10 ns target on
   U55C, so any clock pull-in (e.g. 9 ns / 111 MHz) needs additional pipeline
   stages on the longest fadd combinational paths. `bind_op op=fadd
   latency=8` on the tree-reduce sites would buy headroom at the cost of
   ~2 % more cycles in those pipelines.
