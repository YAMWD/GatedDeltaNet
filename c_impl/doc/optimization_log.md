# GDN HLS Optimisation Log

This document is a chronological record. The v1-v7 and prefill sections retain
their original context, but they do not describe the current kernel. The active
`gdn_forward` is decode-only and uses sharded GEMV; see
[architecture.md](architecture.md). Section dates are taken from this file's
git history where available.

## Current decode architecture

*Updated: 2026-07-27.*

The current FPGA path forwards one token at a time from GPU-exported recurrent
and convolution state. Thirty-two compact weight shards feed 32 independent
512-bit HBM readers and 16 two-port compute clusters in `gdn_gemv`. All large
layer projections and `lm_head` use GEMV; A/B use `gdn_gemv_tiny`. No tiled,
systolic, or weight-stationary matmul is called by `gdn_forward`.

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

## Integrated 32-port `gdn_forward`: exhaustive record of attempted configurations

*Logged: 2026-07-19; updated through iter26 on 2026-07-27. **Read this before
proposing any 32-port floorplan** — every row below has been built. Iter24b
produced the first integrated XCLBIN at an auto-scaled 109 MHz, and iter26
reproduced the design from source/config/Tcl while closing at the requested
130 MHz. Do not repeat an earlier experiment without a stated reason why the
outcome would differ.*

Porting the routed 32-port microbenchmark engine into the real `gdn_forward`
(`GEMV_CHANNELS=32`, `GEMV_CLUSTERS=16`, `GEMV_CHANNELS_PER_CLUSTER=2`) has not
yet produced an `.xclbin`. Twenty-two iterations were built between 2026-07-14
and 2026-07-25. **iter22 is the high-water mark and the first legally routed
integrated 32-port design: 0 failed nets, 0 unrouted nets and 0 node overlaps.**
It passed route verification and reached bitstream generation, which then
stopped on timing for the fixed 250 MHz platform clock `dma_ip_axi_aclk_1`
(WNS -0.307 ns). The 130 MHz kernel clock also remained below target
(WNS -0.955 ns). No `.xclbin` was emitted. iter21 was the prior routing best
(5,156 overlaps); iter16 was the earlier high-water mark (16,471 conflicts).
**Judge every lever by its physical distribution, not only by total resource
savings** — iter16 won by moving FIFO storage arrays out of SLR0 CLB into BRAM,
while iter22 won by moving one topology-boundary cluster out of SLR0 and steering
its two weight FIFO endpoints away from the previously overloaded SLL columns.

| # | Build / iteration | Floorplan tried | Stage reached | Measured failure |
|---|---|---|---|---|
| 1 | `gdn32.f150.o16` (07-15) | none | global placer | could not place all instances; "Exit after global placer" |
| 2 | `gdn32x2.f150.o16` (07-16) | none | placer | unplaced instances found |
| 3 | `gdn32x2aux8` iter4 (07-17) | none | **route** | congestion level 7, partially-conflicted nets |
| 4 | `gdn32x2aux8hard` iter5 (07-17) | hard pblocks | placer | SLR0 80.1% BRAM, **26,089 SLLs** of 23,040 |
| 5 | `gdn32x2aux8hardports` iter6 (07-18) | hard pblocks + ports | placer | SLR0 70.2% LUT, **28,562 SLLs** |
| 6 | `gdn32x2movers20120` iter7 (07-18) | hard pblocks + mover pinning | placer | SLR0 73.4% LUT, **28,928 SLLs** |
| 7 | `allmovers0` iter8-topology (07-18) | soft 4/6/6 | script error | pblock pattern matched 118,565 cells — glob bug, never placed |
| 8 | `allmovers0` iter8-retry (07-18) | **soft 4/6/6 + `xr_N`/`ys_N` FIFOs pinned to clusters** | placer | SLR2 84.2% LUT, **28,813 SLLs** |
| 9 | `allmovers0` iter9-soft (07-18) | **soft 4/6/6**, adapters/movers free | placer | SLR1 75.1% LUT, **31,291 SLLs** |
| 10 | `allmovers0` iter10-unconstrained (07-19) | **none** | **route** | placed legally, **setup MET** (WNS +0.003 ns, 0 of 2.75 M endpoints failing); route died at congestion level 7, 1,652,125 unrouted nets |
| 11 | `allmovers0` iter11-qor (07-19) | none + `read_qor_suggestions` replay | synthesis | **no design signal** — Vivado 2022.2 JVM deadlock (zero-byte `hs_err_pid*.log`, 0% CPU, futex wait, 8.5 h). Tool flake; killed |
| 12 | `allmovers0` iter12-c664 (07-19) | **cluster-only 6/6/4** | **route** | placement solved, SLL 81% inside cap, **congestion still 7**; `ys_*_write` conflicts. See §iter12 below |
| 13 | `gdn32x2p4` iter13-p4 (07-20) | **none** + `GEMV_PARTIAL` 8→4 | **route COMPLETED** | first 32-port route to run every phase and write a checkpoint; **203,247 nets left with resource conflicts**, congestion still 7. See §iter13 below |
| 14 | `gdn32x2p4m8` iter14-mm0 (07-21) | none + mm0 outstanding 64→8 + `RQS_CONG-9/-16` | **route REFUSED** | **REGRESSION** — back to `[Route 35-3] not routable`, zero global iterations. mm0 lever worked (BRAM −50); the QoR CONG suggestions cost +2,225 SLLs. See §iter14 below |
| 15 | `gdn32x2p4m8` iter15-mm0only (07-21) | none + mm0 outstanding 64→8 **alone** | **route REFUSED** | **mm0 lever REFUTED.** Link cfg diff-identical to iter13, so mm0 is the only variable — it alone causes the refusal. Decided by ~22 K cells of SLR0 occupancy. See §iter15 below |
| 16 | `gdn32x2p4bramfifo` iter16 (07-22) | none + FIFOs lutram→**bram** depth 64 | **route COMPLETED** | **BEST THROUGH ITER20.** Conflicts **203,247 → 16,471** (12.3× cut), 0 unrouted. Cleared `35-3`, GI-1 hit 10,983 overlaps (iter13: 297 K). Residual conflicts are SLR0 **control/PCIe/DMA**, not GEMV. See §iter16 below |
| 17 | `gdn32x2p4bramfifo772` iter17 (07-22) | iter16 + **7/7/2 hard floorplan** | **route REFUSED** | REGRESSED — `35-3`, 0 global iters, South Long congestion 25.6% (worst of series). Floorplanning exhausted (3rd orientation). See §iter17 below |
| 18 | `gdn32x2p4uram` iter18 (07-23) | iter16 + 32 `ws` FIFOs **bram→uram** | **route COMPLETED** | REGRESSED — conflicts **16,471 → 187,304** (11.4× worse), 0 unrouted. URAM columns are SLR0/SLR1-only; forcing ws there drained SLR2 80→38% and packed SLR0/1 to ~99.7%. See §iter18 below |
| 19 | `gdn32x2p4ctrl` iter19 (07-24) | iter16 + **step 4** (hardcode dims + 15 buffers→1 workspace) | **route COMPLETED** | REGRESSED — conflicts **16,471 → 133,143** (8× worse), 0 unrouted. BUT LUT −19% and conflict MOVED `control_s_axi`→FP-adder (control plane cleared). WNS −4.611 confounds (5.840 ns HLS at 150). See §iter19 below |
| 20 | `gdn32x2p4bramfifo.f130r1` iter20 (07-24) | iter16 `.xo`, frequency **150→130 MHz**, no floorplan | **route COMPLETED** | Frequency lever did not relieve congestion: level 7, **17,394 node overlaps**, 19,423 signals failed to route, 0 unrouted in final status. See §iter20 below |
| 21 | `gdn32x2p4auxshare.f130` iter21 (07-24) | auxiliary sharing/compact weights + workspace, no floorplan | **route COMPLETED** | Prior best: **5,156 node overlaps**, 7,017 signals failed to route. SLL peaks 140% (SLR1–2) and 113% (SLR0–1). See §iter21 below |
| 22 | `gdn32x2p4auxsharec8s1east.f130` iter22 (07-25) | iter21 `.xo` + cluster 8/xr8→SLR1 + ws16/ws17→SLR1 east | **ROUTE LEGAL; bitstream timing FAILED** | **BEST RESULT.** 0 failed, 0 unrouted, 0 overlaps; route verification passed. Bitstream gate failed only after routing: fixed `dma_ip_axi_aclk_1` WNS −0.307 ns; kernel WNS −0.955 ns. No `.xclbin`. See §iter22 below |

### Cluster granularity: 8×4 and 16×2 are BOTH refuted, at different stages

Row 1 (`gdn32`) is not a different floorplan — it is a different **cluster
granularity**: eight four-port clusters (`gemv32_cluster4`), which is the exact
topology of the routed `gemv_full` microbenchmark. Rows 2-12 (`gdn32x2`) are
sixteen two-port clusters (`gemv32_cluster2`); the `x2` in the build name marks
that change. Its measured failure:

```
Command: place_design -directive SSI_SpreadSLLs
WARNING: [Place 30-356] This design requires 22281 Super Long Lines (SLLs)
                        out of 23040 for the crossing of SLR#0 to SLR#1.
ERROR:   [Place 30-99]  Placer failed with error: 'Exit after global placer'
```

**8×4 saturates SLLs at 96.7% and dies in global placement, before routing.** A
four-port cluster drags four 512-bit weight streams across a boundary when
relocated (~2,084 crossing nets) versus ~1,042 for a two-port cluster, so coarser
clusters make spreading twice as expensive per unit moved. The 4→2 split was made
to fix exactly this and it succeeded — 16×2 places at 15,954-18,709 SLLs.

**Do not "revert to 8×4 because the microbenchmark routed with it."** The
microbenchmark is GEMV-only; the integrated kernel additionally carries ~200K LUT
of aux path and the HBM[0] aux master, which is what pushes 8×4 over the SLL cap.
Reverting trades a congestion failure for a placement failure.

**The two walls.** Cluster granularity only selects which wall is hit:

| Strategy | Wall |
|---|---|
| concentrate (16×2, few SLRs — iter10) | congestion level 7 |
| spread coarse (8×4 — gdn32) | SLL 96.7%, placement fails |
| spread fine (16×2, 6/6/4 — iter12) | SLL 81% OK, placement OK, **still congestion 7** |

Neither granularity escapes, because the totals are fixed: ~411K LUT of MAC
pipeline must exist somewhere, and 32×512 bits of weight stream must travel from
SLR0's HBM to wherever the multipliers are. At this stage, the apparent remaining
lever was reducing absolute density (per-port LUT), or fewer ports. iter22 later
showed that a much narrower topology-boundary move could also clear routing; see
the current-best entry below.

### Density lever: GEMV_PARTIAL 8 → 4 — measured −16% LUT at identical throughput

*csynth on Vitis HLS 2022.2, xcu55c, 6.667 ns. 2026-07-20.*

**Where the density is.** `gemv32_cl_flat` is 20,021 LUT — 88% of a cluster.
Breakdown: 47% real arithmetic (31 `fadd` @225 + 32 `fmul` @77), **31% operand
multiplexers** (134 muxes, 47 LUT each, **input size 9 = GEMV_PARTIAL + 1**), 14%
address arithmetic. The `gemv32_cl_p` loop is `unroll`ed over GEMV_PARTIAL lanes
while the pipeline runs at II=8, so HLS time-shares each adder/multiplier across
all 8 lanes and pays a 9:1 mux on every operand — 86% of all mux LUT.

**Result of halving the sharing factor** (`GEMV_PARTIAL` 8→4 *and*
`pragma HLS pipeline` II 8→4, which must move together):

| | BRAM | DSP | FF | LUT |
|---|---:|---:|---:|---:|
| `gdn_gemv` P=8/II=8 | 256 | 2692 | 659,820 | 496,086 |
| `gdn_gemv` **P=4/II=4** | 256 | **2692** | 594,108 | **417,878 (−15.8%)** |
| cluster2_10 P=8/II=8 | 16 | 168 | 31,960 | 22,771 |
| cluster2_10 **P=4/II=4** | 16 | **168** | 27,853 | **17,883 (−21.5%)** |

Operand muxes: 134 @ input-size 9 → **68 @ input-size 5**, 6,299 → 2,784 LUT.
**DSP is identical (2692), which is the proof throughput is preserved** — same
arithmetic per cycle. Estimated Fmax unchanged at 205 MHz.

**Two traps found while testing this — GEMV_PARTIAL only *looks* parameterized:**
1. `gemv32_reduce_part` (~`gdn_model.cpp:1859`) **hardcodes bank indices 0-7**.
   With PARTIAL=4 synthesis fails outright: *"Cannot apply array transformation
   pragma because of full array load/store"*. Must be narrowed to match.
2. `#pragma HLS pipeline II=8` is **hardcoded, not derived from GEMV_PARTIAL**.
   Changing PARTIAL alone leaves II=8, which HALVES throughput (1 `dot16`/cycle
   instead of 2) while showing a flattering −34% LUT / −49% DSP. **DSP count is
   the tell: if DSP drops, throughput dropped.** A first pass at this experiment
   nearly banked that as a win.
3. The `loop_tripcount min=1024 max=90112` hint on `gemv32_cl_flat` is also
   PARTIAL-dependent and stale after the change — reported latency halves
   spuriously (it is just tripcount_hint x II). Real latency is unchanged.

**Parity consequence — a risk, not a blocker.** Narrowing the tree changes the FP
summation grouping (`pack mod 8` → `pack mod 4`, and
`((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))` → `(p0+p1)+(p2+p3)`). FP addition is not
associative, so results shift ~1 ULP. **But the gate
(`check_gdn_c_parity.py`) is `exact_traj_match` — decoded token-ID equality, not
float bits.** The FPGA is already compared against a GPU golden whose reduction
order differs far more radically, and the trajectory matches over 64 tokens — so
there is demonstrable argmax margin. Verify with
`scripts/decode_correctness_check.sh`; do not assume either way.

### What the record establishes

**Floorplanning to a balanced split is refuted.** Rows 4-9 all forced spreading
and all failed *at placement* on SLL overflow, 26,089-31,291 against a 23,040
cap. Two independent 4/6/6 attempts (rows 8, 9) made it *worse* than the hard
splits. Pinning aux stages into SLR2 while their shared `mem_weights_mm0` master
sits elsewhere is the specific mistake in rows 4-8; row 8's own error message
records SLR2 at 84.2% LUT as a consequence.

**Not floorplanning is also refuted, but differently.** Row 10 is the only
configuration that ever placed legally *and* met setup timing at 150 MHz. It
failed in the router, not the placer, and for the opposite reason: the placer
minimised crossings by jamming SLR0 and SLR1 to 99.77% / 99.86% CLB while leaving
SLR2 at 58.92%.

### Phase-0 ground truth (measured on the iter10 routed-error checkpoint)

Extracted from `level0_wrapper_routed_error.dcp` on 2026-07-19. **Two Vivado
idioms return empty silently and invalidated two earlier analysis passes**:
`get_cells -of_objects <cell>` (use `-filter "NAME =~ root/*"`) and `get_slrs
-of_objects <hierarchical net>` (resolve via `get_pins -leaf` → cells → `get_slrs`).

- Cluster placement **9 / 5 / 2** across SLR0 / SLR1 / SLR2; GEMV primitives
  615,040 / 440,773 / 90,004. SLR0 carries 53.7% of the engine, SLR2 only 7.9%.
- Congestion windows are dominated by clusters 9, 10 and 8 sitting on `hmss_0` at
  the HBM south edge. The aux path is essentially absent (rmsnorm 1 appearance,
  gemv_tiny 1, conv / output-norm / SwiGLU 0).
- **Relocating a cluster costs 3-7 crossing nets of control.** The real cost is
  its two 512-bit weight streams, ~521 nets per remote AXI adapter, so ~1,042 per
  relocated cluster per boundary traversed.
- The activation ripple is **already near-optimal**: only `xr_6` and `xr_7` cross
  (~515 nets each), i.e. two boundary crossings, the minimum for a 3-SLR spread.
  Restructuring the ripple into per-SLR broadcasts would buy little — do not
  spend a build on it before the cluster split.
- Top crossers are the aux path: `mem_weights_mm0_m_axi_U` 1,721,
  `output_norm_and_gate` 1,583, `recurrent_attention` 1,031, `swiglu` 1,029,
  `pack16_add` 811, `conv_silu` 523, `rmsnorm` 519 (~7,200 total). The clique is
  split — mm0/conv/rmsnorm/gemv_tiny in SLR1, output-norm/SwiGLU/recurrent in
  SLR2 (recurrent follows its 128 URAMs, all of which are in SLR2).

**Cost model, and its known weakness.** SLL ≈ crossing nets. For iter10:
7 remote clusters × 1,042 + ~7,200 aux + ~1,038 ripple ≈ 15,530 predicted against
15,954 measured, within 3%. But applied to row 8's 4/6/6 it predicts ~20,700
against 28,813 actual — **a 39% under-prediction**. The model is calibrated on one
point and is not trustworthy for large moves; the residual is probably the aux
exile plus FIFO pinning, but that is unverified. Size any move so that a 39%
under-prediction still lands inside 23,040.

### iter12 result — cluster-only 6/6/4: placement SOLVED, congestion UNMOVED

*Built 2026-07-19, 5 h 57 m, exit 2. Logs: `diagnostics/iter12_c664/`
(`link.wrapper.log`, `impl_runme.iter12.log`); checkpoint preserved at
`diagnostics/iter12_c664/link.failed/level0_wrapper_routed_error.dcp`.*

Floorplan applied exactly as written (SLR2: clusters 0-3, SLR1: 4-9, SLR0:
10-15). Outcome, in order:

| Metric | iter10 (no floorplan) | **iter12 (cluster-only 6/6/4)** |
|---|---|---|
| `place_design` | passed | **passed** (iters 4-9 all failed here) |
| SLL, worst boundary | 15,954 SLR0↔SLR1 (69%) | **18,709 SLR1↔SLR2 (81%)** — inside cap |
| Post-place estimated WNS | +0.003 (met) | −3.528, recovered by physopt to **−1.299**, route intermediate **−1.193** |
| `route_design` | **congestion 7**, 1.65 M unrouted | **congestion 7**, max 114.9% (East 64×64) |
| Conflicted nets | `ys_*_write` result writes | **`ys_*_write` result writes** (same class) |

**The load-bearing negative result: moving 3 clusters out of SLR0 (9 → 6) did
not reduce congestion at all.** It stayed at level 7 with the same class of
conflicted net. Spreading clusters across SLRs is therefore *not* the congestion
lever, even though it is affordable in SLLs and fixes placement.

Two things were nevertheless learned and are worth keeping:
1. **A cluster-only floorplan places.** The SLL wall that killed iters 4-9 was
   caused by pinning FIFOs/adapters/aux as well, not by spreading clusters.
2. **Timing is not the blocker.** The alarming −3.528 ns post-placement estimate
   recovered to −1.193 ns during routing (≈127 MHz), comparable to the routed
   microbenchmark's 130.6 MHz.

**Model correction.** The Phase-0 cost model predicted the hot boundary would be
SLR0↔SLR1 at 83% and SLR1↔SLR2 at 62%; reality was the reverse. The error: the
aux clique is split with `mem_weights_mm0`/conv/rmsnorm in SLR1 and
output-norm/SwiGLU/recurrent in SLR2, so most of its ~7,200 crossing nets land on
**SLR1↔SLR2**, not SLR0↔SLR1. Adding clusters to SLR2 stacked on top of that.
Phase 0 had already measured this; it was mis-applied.

**iter12 congestion post-mortem — FLOORPLANNING IS CLOSED FOR 32 PORTS.**
`report_design_analysis -congestion` on the preserved checkpoint names the owner
of every congested window, and it is the same structure everywhere:

| window | dominant owner |
|---|---|
| North Global L5/L6/L7 | `gemv32_cluster2_12/gemv32_cl_flat` 34% / 49% / 29%, `hmss_0` 23% / 8% / 14% |
| South Global L5/L6 | `cluster2_3/cl_flat` 39%, `cluster2_11/cl_flat` 47% |
| East Global L5/L6 | `cluster2_14/cl_flat` 49%, `cluster2_13/cl_flat` 31% |
| South Long L5 | `cluster2_11` 28%, `cluster2_12` 27% |
| West Global L5 | `gemv32_store` 43%, `cluster2_0/cl_flat` 20% |
| West Global L6 / North Long L5-L6 | `rmsnorm_scale` 33% / 32%, `conv_silu` 22% / 17%, `pack16_add` 17% / 12% |

The congested clusters are 0, 2, 3, 11, 12, 13 and 14 — i.e. clusters in **all
three SLRs**, congesting in **all four directions**. The `gemv32_cl_flat` MAC
pipeline is intrinsically routing-dense and congests wherever it is placed.
Relocating it moves the hotspot, it does not remove it. Combined with iter10
(no floorplan, congestion 7) and iters 4-9 (spread, SLL overflow), **three
orientations have now failed and floorplanning is exhausted as a lever for the
32-port FP32 GEMV** — the same conclusion the microbenchmark reached at
`gemv_tile` level. Do not spend another link on a floorplan variant.

Note the aux path (`rmsnorm_scale`, `conv_silu`, `pack16_add`) appears in the
congestion table for the first time — it did not in iter10. Moving clusters into
SLR2 created new local contention with the aux stages already living there.

**Timing is a SEPARATE and more tractable problem.** The critical paths are not
the datapath: WNS −1.328 with **95.30% routing delay** on
`SLR0/interconnect_axilite_user → control_s_axi_U/rdata_reg`, and −1.326 on
`gemv32_load_x_and_w0 → ap_CS_fsm_reg[266]` / `ap_CS_fsm_reg[232] →
mem_weights_mm26_m_axi_U`. These are the AXI-Lite control interconnect and the
single top-level `ap_CS_fsm` register bank, which must reach all 32 AXI masters
and all 16 clusters across three SLRs. Pure distance, not density.

QoR suggestions generated (`diagnostics/iter12_c664/diagnostics/qor_suggestions.rqs`):
RQS_CONG-16 (bloat modules), RQS_CONG-9 (merge over-replicated cells),
**RQS_TIMING-3 (FORCE_MAX_FANOUT on critical nets with far-apart loads)**,
**RQS_TIMING-66 (USER_CLUSTER to link cells early in placement)**, RQS_TIMING-59
(replicate LUT-driven critical nets), RQS_NETLIST-10 (retiming). The three
TIMING ones target the control-fanout problem directly and are worth applying
*once congestion is solved by other means* — they will not fix congestion.

**Open hypothesis, NOT established.** The source-level collector topology
(`gdn_model.cpp` ~line 2251) groups `collect4(ys[0..3]) → slr0_result`,
`collect6(ys[4..9]) → slr1_result`, `collect6(ys[10..15]) → slr2_result`, i.e. it
intends clusters 0-3 in SLR0 and 10-15 in SLR2 — the **opposite** of the iter12
assignment, which was chosen to minimise movement from iter10. Collectors were
left unpinned so the placer may have compensated. Whether the inversion hurt is
unverified; iter10 conflicted on the same `ys_*` nets without any floorplan, so
the result path is implicated in both. Check against the preserved checkpoint
before assuming the orientation matters.

### iter13 result — density (`GEMV_PARTIAL` 8→4): the router ENGAGES, but 203K nets still conflict

*Built 2026-07-20, 7 h 23 m, exit 2. Logs: `build.hw.gdn32x2p4.f150.o8.v2022_2/diagnostics/iter13_p4/`
(`link.wrapper.log`, `impl_runme.iter13.log`); checkpoint preserved at
`diagnostics/iter13_p4/link.failed/level0_wrapper_routed_error.dcp`; full
post-mortem in `diagnostics/iter13_p4/diagnostics/`.*

**This entry required two attempts.** Attempt 1 (07-20 03:07) was SIGKILLed
mid-route and the node rebooted 84 min later, destroying the wrapper before it
could write markers or run the post-mortem. That kill was **environmental, not
the design**: the identical re-run's `memwatch.log` shows available RAM never
below ~474 GB with Vivado peaking at 32 GB. Do not read attempt 1 as a result.

The single changed variable vs iter10 was datapath density: `GEMV_PARTIAL` 8→4
with `II=GEMV_PARTIAL`, shrinking operand muxes 9:1 → 5:1 at identical DSP
(2,692), so throughput is unchanged. Measured effect:

| Metric | iter10 (dense) | **iter13 (`GEMV_PARTIAL`=4)** |
|---|---|---|
| kernel LUT, placed | 659,484 (57.4%) | **574,716 (50.0%)** |
| SLR0 / SLR1 LUT density | 79.40% / 76.21% | **69.77% / 68.84%** |
| SLR0 / SLR1 / SLR2 **CLB sites** | 99.77% / 99.86% / 58.92% | 99.69% / 99.02% / 51.91% |
| total SLLs | 28,235 | **25,097** (SLR0↔SLR1 65.2%) |
| `route_design` verdict | **refused**: `[Route 35-3] Design is not routable` | **ran all phases, "Routing Is Done"** |
| nets fully routed | — (1,652,125 unrouted) | **1,592,356** |
| nets with resource conflicts | — | **203,247** (11.3% of routable) |

**The load-bearing result: density is the only lever that has ever changed the
router's verdict.** iter10 was refused outright; iter13 routed 88.7% of routable
nets and produced a checkpoint. Every floorplan orientation (iters 4-9, 12)
failed to move congestion at all. But 203K conflicts is a wide gap, not a
near-miss — and **this lever is now spent**: `GEMV_PARTIAL` is a power of two
that must be ≥ the 4-cycle FP32 fadd latency (`gdn_model.cpp` ~line 1643), and
`gemv32_reduce_part` `#error`s on any value but 8 or 4.

*Caution on reading the error line.* Vivado printed `[Constraints 18-1000] ...
partially-conflicted nets (Up to first 10 of violated nets)` naming only
`control_s_axi_U/logits[*]` and `control_s_axi_U/mlp_gate[*]`. Those ten are
**not representative** — `report_route_status` on the checkpoint gives the true
total of 203,247. Do not conclude "only the control block failed" from the
printed list.

**Congestion narrowed but did not clear.** Still level 7 (North Global), now
owned by a contiguous band `gemv32_cluster2_6`…`_10`'s `cl_flat` plus `hmss_0`,
versus iter12's scatter across clusters 0/2/3/11/12/13/14 in all three SLRs.

**Worst timing path is the HBM[0] shared master, not the GEMV streams.**
WNS −8.125 ns against 6.667 ns, **89.50% routing delay**, on
`mem_weights_mm0_m_axi_U/load_unit/buff_rdata/.../mem_reg_bram_4` →
`gdn_rmsnorm_rows/rms_load_w/mem_weights_mm0_addr_read_reg_413_reg[70]`.
`RQS_TIMING-6` flags **29 cascaded BRAMs** in that path. `mem_weights_mm0` is the
one master carrying config, aux weights and all ~16 mutable buffers, and the only
one declared `num_read_outstanding=64` while the other 31 use 8.

**Stream depth is NOT the lever — measured, do not re-litigate.** The `gdn_gemv`
FIFOs cost 111,729 FF + 60,063 LUT (17.2% of its FF, 14.6% of its LUT), so they
are worth attacking in principle. But per-instance cost is **depth-independent**:
a 512-bit FIFO costs **1,552 FF at depth 16** (`ws`, results) and **1,547 FF at
depth 4** (`xr`, `ys`) — 0.3% apart for a 4× depth change, because the cost is
the 512-bit staging/control logic, not the storage. Cutting `ws` 16→4 would save
~5 FF × 32 ≈ 160 FF of 111,729. **The driver is stream width × count, i.e. port
count — not depth.**

**Next levers, in evidence order.** (1) Unload `mem_weights_mm0`: it is both GEMV
shard-0's reader *and* the aux master for ~18 buffers, at
`num_read_outstanding=64`/`num_write_outstanding=64` while the other 31 masters
use 8 — the source of the 29-BRAM cascade on the worst path. (2) The congestion
QoR suggestions `RQS_CONG-16` (bloat modules) / `RQS_CONG-9` (merge
over-replicated cells) in `diagnostics/qor_suggestions.rqs`. Hold the TIMING
suggestions until conflicts are resolved — with 203K conflicts they target the
wrong problem. (3) Narrower arithmetic, at the cost of bit-exactness.

*Note on port count.* CLAUDE.md states `GEMV_CHANNELS=16` "routes, but is blocked
on SLR0-jam timing." **That claim is unsubstantiated by any artifact in this
repo**: `doc/decode_disaggregated_gemv.md` (its cited source) never mentions 16
channels, `git log --all -S"define GEMV_CHANNELS 16"` returns nothing, and no
`build.hw.*` directory corresponds to it (the `.o16` suffix is `JOBS=16`, not
channel count). Committed HEAD is `GEMV_CHANNELS 8`. Treat 16-port routability as
untested, not as a known-good fallback.

### iter14 result — mm0 unload + CONG QoR suggestions: the SUGGESTIONS MADE IT WORSE

*Built 2026-07-21, 4 h 54 m, exit 2. Logs: `build.hw.gdn32x2p4m8.f150.o8.v2022_2/diagnostics/iter14_mm0/`;
checkpoint at `diagnostics/iter14_mm0/link.failed/level0_wrapper_routed_error.dcp`.*

Two levers shipped together (requested packaging, so a verdict cannot be assigned
to either alone — but the placement data separates them cleanly anyway):

- **Lever 2**, in the .xo: `mem_weights_mm0` outstanding 64→8 read and write,
  matching ports 1..31.
- **Lever 3**, in the link: replay iter13's `RQS_CONG-9` + `RQS_CONG-16` as an
  `opt_design` pre-hook, with the four `RQS_TIMING` suggestions deleted.

**Outcome: a regression.** The router refused to start —
`ERROR: [Route 35-3] Design is not routable as its global congestion level is 7`,
zero Global Iteration phases, route abandoned after 1 h 26 m (iter13 routed for
4 h 14 m). This is the iter10 failure mode; iter13 remains the only 32-port build
whose `route_design` ran to completion.

| Metric (placed) | iter13 | **iter14** |
|---|---|---|
| BRAM | 687 (37.8%) | **637 (35.1%)** |
| LUT | 574,716 | 575,827 (+1,111) |
| FF | 810,939 | 811,403 (+464) |
| DSP / URAM | 3,338 / 144 | 3,338 / 144 (identical) |
| total SLLs | 25,097 | **27,322 (+2,225)** |
| SLR1↔SLR2 | 43.75% | 52.09% |
| CLB SLR0/1/2 | 99.69 / 99.02 / 51.91 | 99.66 / **99.53** / **48.13** |
| `route_design` | ran all phases | **refused** |

**Lever 2 is VALIDATED — keep it.** BRAM fell by exactly 50 with DSP and URAM
untouched: the oversized mm0 read FIFO shrinking, i.e. the 29-deep cascade
RQS_TIMING-6 named. Note this was **invisible to csynth** — `mem_weights_mm0_m_axi_U`
reports byte-identical resources (4,597 FF / 13,857 LUT) at outstanding 8 and 64,
because HLS does not model outstanding depth. It only appears in Vivado synthesis.
Do not try to pre-validate this class of change with csynth.

**Lever 3 is REFUTED — do not replay QoR congestion suggestions on this design.**
`RQS_CONG-16` is "bloat modules to reduce congestion" and it did exactly that:
+1,111 LUT, +464 FF, **+2,225 SLLs**, SLR1 packed tighter and SLR2 *less* used.
It bought no congestion relief and cost the routability iter13 had earned. Vivado
generated these suggestions from iter13's own checkpoint, so this is not a case of
stale advice — the advice is simply wrong for a design already at 99% CLB in two
SLRs.

**DO NOT read `route_status.rpt` conflict counts on a refused route.** iter14's
post-mortem reports "23 nets with resource conflicts" against iter13's 203,247,
which looks like a 4-order-of-magnitude improvement and is the exact opposite of
the truth. The line that matters is **unrouted**:

| | iter13 | iter14 |
|---|---|---|
| routable | 1,795,603 | 1,796,751 |
| pre-fixed (platform) | 217,457 | 217,457 |
| fully routed | **1,592,356** | **217,489** |
| unrouted | 0 | **1,579,195** |

iter14 routed **32 nets** beyond the platform's pre-fixed routing; even shell nets
(`GLOBAL_LOGIC0/1`, the `hmss_0` APB bridge, `axi_ic_ctrl_mgmt_slr1`) are unrouted.
The conflict count is low only because the router aborted before attempting the
work. Always read `unrouted` alongside `resource conflicts`.

**Next: iter15 = lever 2 alone**, reusing this .xo unchanged and reverting only the
`TCL.PRE` hook line. That isolates mm0 against iter13 and should recover iter13's
routing behaviour plus the 50 BRAM.

*Process notes (two aborted attempts, neither a design signal).* (1) The .xo was
first built at `FREQ=150`; the artifact is over-synthed (xo 250 / link 150) and at
6.67 ns HLS shed ~104 K FF of pipeline registers, so it was discarded and rebuilt
at 250. (2) The first link ran with a QoR filter using a non-existent property
`IS_ENABLED`, which threw `[Common 17-142]` and fell back to applying all six
suggestions — including `RQS_TIMING-59` (replicate) fighting `RQS_CONG-9` (merge).
Killed and restarted. The working API is `delete_qor_suggestions` (needs an open
project); the writable flags are `ENABLED` / `IS_ACTIVE`, verified by
`list_property` on a `qor_suggestion` object.

### iter15 result — mm0 lever ALONE is what breaks the route, decided by ~22 K cells

*Built 2026-07-21, 4 h 33 m, exit 2. Logs and 897 MB checkpoint:
`build.hw.gdn32x2p4m8.f150.o8.v2022_2/diagnostics/iter15_mm0only/`.*

Same .xo as iter14 (mm0 outstanding 64→8); the QoR pre-hook removed. The link cfg
was verified **byte-identical to iter13's** once comments are stripped, so the mm0
outstanding depth is the ONLY variable against iter13.

**Result: refused.** `[Route 35-3] Design is not routable`, zero Global Iteration
phases, route abandoned after 1 h 24 m. **This refutes the iter14 conclusion that
lever 3 was to blame** — lever 2 alone is sufficient to cause the failure, and the
"lever 2 validated" claim from iter14's BRAM saving was wrong. A resource win was
mistaken for a routability win.

| | mm0 outstanding | QoR | route |
|---|---|---|---|
| iter13 | **64** | none | **completed**, 203,247 conflicts |
| iter14 | 8 | CONG | refused |
| iter15 | 8 | none | **refused** |

### Engine→SLR maps: what actually separates routing from refusal

*Measured on the preserved checkpoints with `report_engine_slr_map.tcl` (per-engine
leaf-cell counts classified by site Y against each SLR's measured range: SLR0
Y0-239, SLR1 Y240-479, SLR2 Y480-719). Outputs: `engine_slr_map.txt` in each
diagnostics dir. NOTE: HLS names the first cluster `gemv32_cluster2_U0` with NO
index — a pattern of `*_0_U0` silently drops ~53,600 cells.*

Neither iter13 nor iter15 used any floorplan (zero `create_pblock` in either impl
log), so every placement below is the placer's own choice under
`AltSpreadLogic_high`.

| | iter13 (routed) | iter15 (refused) | delta |
|---|---|---|---|
| **SLR0 total cells** | **675,133** | **697,484** | **+22,351** |
| GEMV cluster cells in SLR0 | 539,393 | 562,329 | +22,936 |
| aux cells in SLR2 | 190,292 | 179,439 | −10,853 |
| cluster fragment in SLR2 | 12,471 | 15,096 | +2,625 |
| total SLLs | 25,097 | 26,140 | +1,043 |

**The discriminator is cluster 5.** iter13 placed 60% of it in SLR1
(21,532/32,156); iter15 pulled it back to 91% in SLR0 (48,889/4,792). iter13 also
put ~11 K more aux in SLR2 (mostly `recurrent_attention`, which consumes
activations not weight streams, so it is cheap to place remotely) and pushed
*less* cluster fragment across SLR1↔SLR2.

**~22,000 cells — ~3.3% of SLR0 occupancy — is the entire margin between "routes"
and "refused."** The 32-port design sits on a knife edge, and freeing 44 BRAMs in
SLR1 was enough to perturb the placer into the wrong side of it.

**Congestion is confined to SLR0.** Every congested window in iter15's router log
lies within Y0-Y207, i.e. entirely inside SLR0 (X0-X127, several anchored at the
`GTY_L_X0Y*` column). Worst direction is SOUTH — Global 128×128 at 11.26% of
tiles, Long at 15.54% — i.e. demand draining toward the HBM edge. Owners are
consistently just two things: `hmss_0` (21-51%, 124 K cells, immovable — HBM is
bonded under SLR0) and `gemv32_cl_flat` (19-51%).

### Untried lever with direct evidence: FIFOs in BRAM, not LUTRAM

The routed 32-port microbenchmark (`microbench/gemv_tile`) differs from the
integrated kernel in three ways: no aux path, an explicit 2/3/3 floorplan, **and
its stream FIFOs are `impl=bram` at depth 64** where ours are `impl=lutram` at
depth 16/4.

Our 69 × 512-bit FIFOs cost **60,063 LUT + 111,729 FF of CLB resources**, sitting
in SLR0 — the only congested region. The microbench spends BRAM instead and holds
96.88% BRAM / 90.47% CLB in SLR1, and routes. The integrated kernel uses 637 of
1,816 BRAM tiles (35%), leaving ~1,179 free; a 512-bit FIFO needs ~8 RAMB36 for
width, so converting costs roughly 550 tiles.

Note this also corrects the earlier "depth doesn't matter" finding: depth-4 and
depth-16 LUTRAM FIFOs measured 1,547 vs 1,552 FF, so depth is irrelevant **in
LUTRAM**. In BRAM depth is nearly free (a RAMB36 is 512+ deep regardless), which
is why the microbench uses 64. The change is `impl=bram` *and* deeper, together.

Against the ~22 K-cell margin above, moving ~60 K LUT + ~112 K FF out of SLR0 CLBs
is several times the deciding quantity — the most promising untried lever, and it
is seven pragma lines at `gdn_model.cpp` ~2209-2215. Build it on **iter13's .xo
baseline**, not the mm0 one.

### iter16 result — FIFOs to BRAM: 12.3× fewer conflicts, best of the series

*Built 2026-07-21/22, xo+link back to back, link ~9 h. Logs and 1.02 GB
checkpoint: `build.hw.gdn32x2p4bramfifo.f150.o8.v2022_2/diagnostics/iter16_bramfifo/`.
mm0 reverted to 64 first (iter15), so the ONLY diff from iter13 source is the 14
FIFO pragma lines; gdn_model.h byte-identical, link cfg diff-identical to iter13.*

Change: the 7 gemv stream groups (`ws`,`xr`,`ys`,`slr{0,1,2}_result`,`result`)
from `impl=lutram` depth 16/4 to **`impl=bram` depth 64**, matching the routed
microbenchmark. Single variable vs iter13.

**Route completed** (`Routing Is Done`, not a `35-3` refusal) with the best
numbers of any 32-port build:

| route_status | iter13 | **iter16** |
|---|---|---|
| routable nets | 1,795,603 | 1,783,716 |
| fully routed | 1,592,356 | 1,767,245 |
| unrouted | 0 | 0 |
| **resource conflicts** | **203,247** | **16,471** |

**12.3× fewer conflicts, 0 unrouted.** It also cleared the `35-3` gate that
refused iter14/iter15 and cruised GI-1: overlaps 1,355,678 → 10,983 (iter13 ended
GI-1 at 297,365 — 27× worse). GI-2 then churned back to 986,447 attacking the
South/HBM hotspot, same pattern as iter13, and settled at 16,471.

**Why it worked — resource swap, measured.** The 69 512-bit FIFOs are ~14.6% of
`gdn_gemv` LUT, and CLB in SLR0 is the ONLY congested resource. csynth: FIFO row
went BRAM 0→1,035, FF 111,729→76,980, LUT 60,063→40,494 (**only the storage array
migrates; ~77 K FF of control logic stays** — so this freed ~19.6 K LUT + ~34.7 K
FF, NOT the ~60 K/112 K first predicted). Placed effect was larger than the csynth
delta because the placer then rebalanced: SLR0 CLB 99.69→98.89%, SLR1 99.02→95.28%,
**SLR2 51.91→80.44%** — SLR0 shed ~132 K CLB cells, ~6× the ~22 K margin. BRAM
412→1,447 tiles (top), 637→~1,204 placed (~66% of 1,816). DSP/URAM unchanged.

**The conflicts moved OFF the datapath.** iter13 conflicted on `gemv32_cl_flat` /
`ys_*_write` (the GEMV result path). iter16's 16,471 are `pxi_ii_core`
(PCIe/DMA host path), `interconnect_axilite_user`, `axi_gpio_null`, `GLOBAL_LOGIC0`
— **SLR0 control + platform infrastructure**, around the immovable HBM/PCIe corner.
The BRAM swap fully relieved the MAC datapath; what remains is the platform-corner
congestion BRAM cannot touch (HBM and PCIe are both bonded to SLR0).

**KEEP this change.** Depth 64 is nearly free in BRAM (a RAMB36 is 512+ deep for
free; the cost is 512-bit width, ~8 RAMB36 per FIFO). Correctness: exact
trajectory match. Open risk carried forward: an intermediate WNS −6.707 ns during
GI-1 — timing closure at 150 MHz is a SEPARATE hurdle from routability, unresolved.

**Next: iter17 = iter16 .xo + 7/7/2 floorplan.** The residual conflicts are all in
SLR0, and 7/7/2 removes 7 clusters' worth of logic from SLR0 — it targets exactly
what remains and stacks cleanly on the BRAM baseline (independent levers). Whether
the ~16 K platform-corner conflicts are reducible by making room around them, or
are a hard floor (HBM+PCIe immovable), is the open question the build answers.

### iter18 result — ws FIFOs BRAM→URAM: the URAM lever FIGHTS the SLR geometry

*Built 2026-07-22/23, link ~9 h. Logs + checkpoint:
`build.hw.gdn32x2p4uram.f150.o8.v2022_2/diagnostics/iter18_uram/`. Single variable
vs iter16: the 32 ws weight-stream FIFOs `impl=bram`→`impl=uram`. Motivated by the
iter16 congestion analysis (fatal South L7 windows 90-98% RAMB, SLR0 URAM 0%).*

**csynth verified the swap perfectly** (−480 RAMB18, +256 URAM, no II regression,
+0.2% LUT) — but the LINK exposed a placement backfire csynth cannot see:

| placed | iter16 (16,471) | iter18 |
|---|---|---|
| Total SLLs | 29,761 | **25,075** (−4,686) |
| SLR0 BRAM | 83.78% | 76.26% |
| SLR0 / SLR1 / SLR2 CLB | 98.89 / 95.28 / 80.44 | **99.63 / 99.87 / 38.23** |
| SLR0 / SLR1 / SLR2 URAM | 0 / 5 / 40% | 50 / 75 / **0%** |

**The load-bearing negative result: URAM columns live only in SLR0/SLR1 on this
device.** Forcing ws into URAM pulled the clusters AND the recurrent state (63% in
SLR2 at iter16) toward those columns — **draining SLR2 from 80%→38% CLB and packing
SLR0/SLR1 to ~99.7%**. The change relieved exactly what it targeted (SLL −4,686,
SLR0 BRAM −7.5 pts) yet worsened the metric that predicts routability (SLR0/SLR1
CLB density). csynth sees resource COUNTS, not the SLR LOCATION URAM forces — so a
clean csynth is necessary but not sufficient for a memory-relocation lever.

**Outcome: completed routing (18-1000, not a 35-3 refusal) but did not improve.**
The router's initial congestion looked comparable-to-better on the fatal South Long
(17.36% vs iter16 19.95%), which briefly looked hopeful — but GI-1 convergence was
decisively worse: floor ~281,152 overlaps vs iter16's 10,983 (~25×), WNS −8.887.
Conflicts landed at **187,304 (0 unrouted) — 11.4x WORSE than iter16's 16,471**,
back to iter13's pre-BRAM-FIFO 203,247 territory. The named conflict is again
`control_s_axi_U/weight_data_mm11[13]` — a control-plane net, same class as iter16.
The URAM swap nearly erased the entire iter16 gain.

**Lesson for the plan.** Moving memory around does not relieve the structural
`gemv(51-61%)+hmss(22-27%)` density in the SLR0 corner; the URAM variant actively
fights the SLR geometry. The remaining levers must REMOVE logic from the corner,
not relocate it: (1) strip the control plane — 1-token specialization, drop the
AXI-Lite pointer/dim registers that are literally the conflicting nets; no
URAM-geometry problem since it deletes logic; (2) 16 ports — the structural fix.
Do NOT pursue further BRAM/URAM reshuffles.

### iter19 result — step 4 control-plane consolidation: cleared control, but regressed net

*Built 2026-07-23/24, xo+link. Logs+checkpoint:
`build.hw.gdn32x2p4ctrl.f150.o8.v2022_2/diagnostics/iter19_ctrl/`. Change vs iter16:
config/max_tokens/num_tokens hardcoded (constant loop bounds) + the 15
activation/state buffers consolidated into ONE `workspace` m_axi pointer (shared
GDN_WS_OFF_* layout in gdn_model.h; kernel + csim host + on-card host all updated).
Csim bit-exact. Single variable vs iter16 (same FREQ=150).*

**csynth was the biggest logic win of the series:** LUT 1,198,064 → 971,245
(**−226,819, −19%**), FF −6,889, II violations 110 → **36** (one address channel =
fewer bus-request conflicts, NOT the dataflow serialization the aliasing risk
predicted), BRAM/URAM unchanged, DSP +565 (constant bounds let HLS use more DSP).
Downside: HLS timing estimate 3.930 → **5.840 ns** — funneling 15 buffers through
one m_axi address channel lengthened a path (fits 150 MHz's 6.667 ns by only
0.83 ns).

**Route completed (18-1000, not 35-3) but REGRESSED: 133,143 conflicts, 0 unrouted
vs iter16's 16,471 (~8x worse).** GI-1 floor 230,938 (iter16: 10,983). But the
qualitative shift is the real finding:

**Step 4 DID clear the control plane — the conflict MOVED.** iter16 conflicted on
`control_s_axi_U/*` (AXI-Lite base-address regs); iter19 conflicts on
`faddfsub_32ns_32ns_32_7_full_dsp` — the FP adder datapath. So the −227K LUT
removed the control-plane congestion exactly as designed; the residual relocated
to the FP arithmetic (consistent with DSP +565).

**Confound: WNS −4.611 ns at 150 MHz.** The 5.840 ns HLS estimate means iter19 was
fighting timing AND congestion at once; the 133k conflicts are partly
timing-driven rip-up, not purely congestion. A 130 MHz relink of the SAME xo (no
rebuild) would de-confound — 5.840 << 7.69 ns removes the timing pressure and
tests step 4's pure congestion effect. Whether that recovers below 16,471 is
untested. (Counter-evidence: the GI-1 floor 230,938 is largely a congestion
signal and is 21x iter16's, so a large recovery is not guaranteed.)

**At this point iter16 remained the high-water mark (16,471).** Later iter21
reduced the residual to 5,156 overlaps, and iter22's narrow topology-aware
floorplan eliminated it; see the entries below. The broad hard floorplan, URAM
relocation and control-consolidation results remain valid negative evidence.

### iter20 result — 150→130 MHz did not relieve congestion

*Built 2026-07-24 from the exact iter16 `.xo`; definitive retry directory:
`build.hw.gdn32x2p4bramfifo.f130r1.o8.v2022_2/diagnostics/iter20_iter16_f130/`.
Only the link target changed from 150 to 130 MHz.*

Placement and pre-route physical optimization met setup timing (WNS +0.003 ns),
but the router still reported global/short congestion level 7. Final route status
had 0 failed and 0 unrouted nets, yet route verification rejected **17,394 node
overlaps** and reported **19,423 signals failed to route**. The worst estimated
SLL columns were 139% across SLR1–2 and 136% across SLR0–1.

This is the direct evidence that lowering the kernel clock is not itself a
congestion lever. It relaxes timing analysis, but does not remove logic or SLL
demand and can change placement choices in either direction.

### iter21 result — auxiliary sharing reduced the illegal-route residue to 5,156

*Built 2026-07-24 at 130 MHz, no floorplan. Build:
`build.hw.gdn32x2p4auxshare.f130.o8.v2022_2/`. This `.xo` is also the exact
input to iter22 (SHA-256
`0b4454bfca064627d5e929ebd91721bd989082b16dd1760bae051ded5965cf73`).*

The auxiliary-sharing/compact-weight/workspace implementation substantially
improved convergence without changing the 32-port, 16-cluster GEMV topology.
The router completed all phases with 0 failed and 0 unrouted nets in its final
status, but route verification still found **5,156 node overlaps** and
**7,017 signals failed to route**. This became the prior best, roughly 3.4x
below iter20's overlap count.

The remaining problem was strongly localized by SLL column:

| Boundary | Total demand | Worst column |
|---|---:|---:|
| SLR1–2 | 8,003 / 23,040 (34.74%) | **2,013 / 1,440 (140%)** |
| SLR0–1 | 12,558 / 23,040 (54.51%) | **1,631 / 1,440 (113%)** |

Aggregate SLL capacity was therefore sufficient; a few physical Laguna columns
were oversubscribed. This motivated a narrow boundary move rather than another
full-SLR floorplan.

### iter22 result — first legal route; current best

*Built 2026-07-25, link 8 h 06 m, exit 1. Build:
`build.hw.gdn32x2p4auxsharec8s1east.f130.o8.v2022_2/`. Configuration:
`hw_iter22_cluster8_slr1_east.cfg`; pre-opt hook:
`apply_iter22_cluster8_slr1_east.tcl`. The kernel `.xo` is bit-identical to
iter21.*

iter22 changed only four hierarchy roots:

- assign `gemv32_cluster2_8_U0` and `xr_8_U` to the full SLR1;
- place the two small private stream FIFOs `ws_16_U` and `ws_17_U` in the
  underused east-side SLR1 region;
- leave the other 15 clusters, adapters, collectors, auxiliary stages and
  streams movable.

The floorplan redistributed rather than eliminated estimated congestion:

| Router metric | iter21 | iter22 |
|---|---:|---:|
| Global/short congestion level | 7 | 7 |
| Timing congestion level | 7 | 7 |
| SLR1–2 worst SLL column | 140% | **94%** |
| SLR0–1 worst SLL column | 113% | **130%** |
| SLL columns above 100% | 4 | **2** |
| SLR0–1 total SLL demand | 12,558 | 13,364 |

Despite the remaining level-7 estimate and the new 130% lower-boundary peak,
every global routing iteration converged to zero. Final route verification
reported:

| Final route status | iter22 |
|---|---:|
| Failed nets | **0** |
| Unrouted nets | **0** |
| Node overlaps | **0** |
| Route legality | **passed** |

This supersedes the earlier conclusion that all floorplanning was exhausted.
The negative result still applies to broad 4/6/6, 6/6/4 and 7/7/2 hard
floorplans. iter22 proves that a **minimal topology-boundary move plus
per-column endpoint steering** can help without overconstraining the placer.
Preserve these four assignments as the routable baseline.

The build failed later, during the bitstream timing gate:

| Clock domain | Target | WNS | TNS / failing endpoints | Interpretation |
|---|---:|---:|---:|---|
| `clk_kernel_00_unbuffered_net` | 130 MHz | -0.955 ns | -5,580.914 ns / 14,675 | GEMV clusters 12/14; routing-dominated high-fanout control paths. Current route implies about 115.6 MHz, so 110 MHz is the conservative kernel target |
| `dma_ip_axi_aclk_1` | fixed 250 MHz | **-0.307 ns** | -9.265 ns / 69 | Fatal unscalable platform clock in `hmss_0/path_12/.../srl_fifo_0`; one LUT level and about 96% routing delay |

Vitis could not auto-scale past the fixed DMA-clock failure, so
`write_bitstream` stopped and no `.xclbin` was produced. Lowering only the
kernel frequency does not relax the DMA clock's 4.000 ns requirement, although
the resulting re-placement may help it indirectly. The next build should retain
the iter22 floorplan, target about 110 MHz for the kernel, and use a
timing-oriented implementation variant to recover the remaining 0.307 ns on
the HMSS path.

### iter24 result — fanout repair improved DMA timing, exposed two new wire-delay paths

*Launched 2026-07-26 from the iter21/iter22 kernel `.xo` (SHA-256
`0b4454bfca064627d5e929ebd91721bd989082b16dd1760bae051ded5965cf73`).
Configuration: `hw_iter23_cluster8_slr1_east_dmaf64.cfg`; pre-place hook:
`apply_iter23_dma_fanout.tcl`. Builds:
`build.hw.gdn32x2p4auxsharec8s1eastdmaf64v2.f110.o8.v2022_2/` and
`build.hw.gdn32x2p4auxsharec8s1eastdmaf64v2.f130.o8.v2022_2/`.*

The only physical change relative to the iter22 baseline was a strict
pre-placement constraint on the original fixed-HMSS read-response net:

- match the single `r15...common.srl_fifo_0/asyncclear_state1_inst/Q`
  driver that failed iter22;
- apply `MAX_FANOUT_MODE=CLOCK_REGION` and `FORCE_MAX_FANOUT=64`;
- retain the four iter22 cluster/FIFO assignments and all 32 HBM ports.

The 110 MHz arm completed after 9 h 06 m with v++ exit 1. Routing was fully
legal:

| Final route status | iter24a, 110 MHz |
|---|---:|
| Failed nets | **0** |
| Unrouted nets | **0** |
| Node overlaps | **0** |
| Route verification | **passed** |
| Hold slack after router phys-opt | **+0.001 ns** |

Setup timing still blocked bitstream generation:

| Clock domain | Target | WNS | TNS / failing endpoints | Result |
|---|---:|---:|---:|---|
| `clk_kernel_00_unbuffered_net` | 110 MHz | **-0.716 ns** | -237.827 ns | Scalable kernel clock; routed path corresponds to about 102 MHz |
| `dma_ip_axi_aclk_1` | fixed 250 MHz | **-0.195 ns** | -5.853 ns / 56 | Fatal unscalable clock; no XCLBIN |

The original 524-load read-response path is no longer among the leading DMA
violations. The constraint therefore bought **0.112 ns** on the fatal system
clock versus iter22 (`-0.307` to `-0.195 ns`), but the bottleneck moved rather
than disappearing. The new dominant DMA path is driven by
`w15...common.srl_fifo_0/fifoaddr_reg[2]/Q`: its routed leaf segment has
1,159 loads, and routing accounts for 3.729 ns of its 4.015 ns data delay
(92.9%). A separate PCIe reset path fails at about `-0.191 ns`; it crosses from
the fixed reset source in SLR1 to the first SLR0 reset-fanout stage, with 97.5%
of its data delay in routing.

This is now a **timing-closure problem on top of a repeatably legal route**, not
an unrouted-net congestion failure. The next iteration targets 100 MHz only and
stacks three narrow, independently motivated changes: keep the repaired
read-response net at fanout 64, constrain the newly critical FIFO-address net
to clock-region fanout 32, place the first SLR0 reset stage in the bottom clock
region of SLR1, and enable an explicit post-route
`phys_opt_design -directive AggressiveExplore`. Broad GEMV or whole-FIFO-cone
pblocks remain deferred because they risk disturbing the legal-route baseline.

The 130 MHz arm completed after 11 h 42 m with v++ exit 0 and produced a
77,836,265-byte XCLBIN. Its route was also fully legal: zero unrouted nets,
zero final node overlaps and successful route verification. The two important
timing outcomes were:

| Clock domain | Requested/required | Routed WNS | Outcome |
|---|---:|---:|---|
| `dma_ip_axi_aclk_1` | fixed 250 MHz | **+0.003 ns** | Passed; unlike the 110 MHz placement, this random physical result closed the unscalable system clock |
| `clk_kernel_00_unbuffered_net` | requested 130 MHz | **-1.432 ns** | Did not close at 130 MHz; 24,319 failing endpoints and TNS -18,545.084 ns |

Vitis auto-frequency scaling selected **109 MHz** for
`ulp_ucs/aclk_kernel_00` (`DATA_CLK`) and encoded that frequency in the XCLBIN.
Thus iter24b is a usable 109 MHz bitstream, not a true 130 MHz implementation.
Its XCLBIN is
`build.hw.gdn32x2p4auxsharec8s1eastdmaf64v2.f130.o8.v2022_2/gdn_forward.xclbin`.
Together, the two arms demonstrate substantial placement variance on the fixed
DMA paths: the nominally easier 110 MHz kernel placement missed DMA timing by
0.195 ns, while the requested-130 MHz placement closed DMA timing by 0.003 ns
and handled kernel timing by scaling the programmable clock to 109 MHz.

#### iter24b on-card correctness and performance

*Measured 2026-07-26 on U55C device 1 (`0000:c1:00.1`) with the full
`gdn-1.3b-f32.gdnw` model, `decode_ex0.gdnstate`, and three independent
64-token decode-from-state host invocations. The zero-valued seed entry was
excluded from latency statistics, leaving 63 steady-state samples per run and
189 samples in the pooled result.*

All three runs matched the golden token trajectory exactly: 64/64 positions per
run, 192/192 total, with no first divergence. The measured performance is:

| Metric | Run 1 | Run 2 | Run 3 | Pooled / comparison |
|---|---:|---:|---:|---:|
| Wall TPOT median (ms/token) | 1650.751 | 1650.747 | 1650.808 | **1650.768** |
| Kernel median (ms/token) | 1650.622 | 1650.625 | 1650.640 | **1650.627** |
| Wall TPOT mean (ms/token) | 1650.782 | 1650.787 | 1650.828 | 1650.799 |
| Wall TPOT population standard deviation (ms) | 0.108 | 0.133 | 0.087 | 0.113 |
| Effective generation rate (token/s) | — | — | — | **0.606** |

Against the documented 150 MHz, 8-port flattened-GEMV baseline of
**121.4 ms/token** (8.237 token/s), iter24b is **13.60× slower** and increases
TPOT by 1259.8%. Frequency scaling alone would predict about 167.1 ms/token
when moving from 150 to the actual 109 MHz clock. The measured result remains
9.88× slower than that clock-scaled expectation. Wall time exceeds kernel time
by only about 0.14 ms/token, so host launch overhead does not explain the
regression.

Therefore iter24b is a correct, routable hardware image, but it is not a
performance success. The large gap is an architectural/kernel-throughput
regression in the integrated 32-port cluster design, not merely the lower
clock. Raw results and host logs are under
`diagnostics/iter24_dmaf64v2/on_card_109mhz/` as
`run{1,2,3}_device1.{decode.json,host.log}`. An earlier attempt on device 0
failed during recurrent-state BO synchronization before kernel execution and
was excluded.

#### iter24b preservation and clean reproduction

The generated iter24 lineage was removed after the successful result was
recorded. The retained, checksum-guarded reproduction command is:

```bash
make -C c_impl reproduce_iter24_success
```

This performs a clean Vitis 2022.2 compile and link from source with the exact
130 MHz compile/link request, U55C platform, 32-bank connectivity,
`apply_iter22_cluster8_slr1_east.tcl` floorplan, and
`apply_iter23_dma_fanout.tcl` placement hook used by iter24b. The command
refuses to run under the iter24 name if the kernel source, HLS Tcl, link
configuration, or either physical-implementation Tcl has drifted. The
historical reference hashes are:

- kernel `.xo`: `0b4454bfca064627d5e929ebd91721bd989082b16dd1760bae051ded5965cf73`;
- successful `.xclbin`: `3b58e7b4272b0d268fa06d74485d4c72e0a578be3eb2a6b01badc170b672fcdc`.

The retained recipe is **config/Tcl-only** and has no `.rqs` dependency. The
historical iter10 RQS referred largely to stale generated hierarchy. Its two
stable root-level `CELL_BLOAT_FACTOR=low` settings for HMSS and `gdn_forward`
are written directly in `apply_iter22_cluster8_slr1_east.tcl`; the incomplete
equivalent-driver seed list is not replaced with an unsafe broad approximation.
Consequently the command is portable and self-contained, although a new
implementation is not claimed to be bit-for-bit physically identical to the
historical routed image.

### iter25 result — cancelled during initial routing; no design verdict

*Launched 2026-07-26 15:52 +03 and cancelled by request at 20:28 +03. Build:
`build.hw.gdn32x2p4auxsharec8s1eastdmaf32resetprphys.f100.o8.v2022_2/`;
configuration: `hw_iter25_cluster8_slr1_east_dmaf32resetprphys.cfg`; source
`.xo` SHA-256:
`0b4454bfca064627d5e929ebd91721bd989082b16dd1760bae051ded5965cf73`.*

This is a single 100 MHz link; no parallel frequency variants are part of
iter25. It preserves all iter22/iter24 connectivity and the four-root legal-route
floorplan, then stacks these narrow timing fixes:

- retain the original 524-load HMSS read-response net at
  `MAX_FANOUT_MODE=CLOCK_REGION`, `FORCE_MAX_FANOUT=64`;
- apply `MAX_FANOUT_MODE=CLOCK_REGION`, `FORCE_MAX_FANOUT=32` to the newly
  critical 1,159-load `w15...srl_fifo_0/fifoaddr_reg[2]` net;
- move only the first PCIe-to-SLR0 reset-fanout stage into SLR1 clock region
  `CLOCKREGION_X6Y4`, splitting the long crossing across the two existing
  reset registers;
- enable the explicit `POST_ROUTE_PHYS_OPT_DESIGN` step with directive
  `AggressiveExplore`;
- emit post-place target-placement diagnostics and final post-phys-opt route,
  timing-summary and bus-skew reports.

Preflight passed before launch: both fanout nets and the reset hierarchy matched
exactly once in their component checkpoints; property readback was respectively
`CLOCK_REGION/64`, `CLOCK_REGION/32` and `SLR1/X6Y4`; the diagnostic scripts
executed on iter24a's preserved routed checkpoint; the launcher contained only
`--kernel_frequency 100`; and the copied `.xo` checksum matched the source.

Placement and pre-route physical optimization completed. The build was
interrupted in initial routing, during SLL assignment, before any route
optimization, post-route physical optimization, timing signoff or bitstream
generation. At interruption the transient router state still had 1,130,805
unrouted nets, 183,314 partially routed nets and 182 node overlaps. These are
startup-state counters, **not an iter25 routability verdict**.

The initial SLL estimate still showed localized pressure across SLR0-SLR1:
three columns were above capacity at 109%, 108% and 106%. SLR1-SLR2 peaked at
82%. This indicates that the timing-only hooks did not eliminate the inherited
local SLL demand, but the router was stopped too early to determine whether it
would converge.

The interrupt cancelled the enclosing implementation Tcl before its error
checkpoint and post-route hooks could execute. Therefore iter25 produced no
routed/error DCP, no post-route timing or bus-skew reports, no XCLBIN, and no
exported QoR payload. The original inherited 50,868-byte iter10 `.rqs` had
already been removed with the stale build directory; the surviving report and
applied-command evidence is preserved in
`iter24_qor_suggestions_recovery.rpt`. It is historical evidence only. The
retained iter24 recipe now uses config/Tcl inputs exclusively and does not read
or require that binary payload.

### iter26 result — clean config/Tcl reproduction closes at 130 MHz

*Launched 2026-07-26 20:58 +03 and completed 2026-07-27 07:58 +03. Build:
`build.hw.gdn32x2p4auxsharec8s1eastdmaf64v2.f130.o8.v2022_2/`;
requested kernel frequency: 130 MHz.*

This run tested the self-contained retained recipe after removing the binary-RQS
dependency. Its implementation inputs were
`hw_iter23_cluster8_slr1_east_dmaf64.cfg`,
`apply_iter22_cluster8_slr1_east.tcl` and
`apply_iter23_dma_fanout.tcl`. The iter22 hook encodes the two stable
root-level `CELL_BLOAT_FACTOR=low` properties directly in Tcl, then applies the
four-root cluster/FIFO floorplan. No `.rqs` file or QoR read/write command is
present in the dependency closure.

The clean rebuild completed with `v++` exit code 0 and produced a
76,879,681-byte XCLBIN. Unlike iter24b's random physical result, which required
automatic scaling from 130 to 109 MHz, iter26 met every timing constraint at the
full requested frequency:

| Final result | iter26 |
|---|---:|
| Failed / unrouted / partially routed nets | **0 / 0 / 0** |
| Final node overlaps | **0** |
| Overall setup WNS / TNS | **+0.003 ns / 0.000 ns** |
| Overall hold WHS / THS | **+0.007 ns / 0.000 ns** |
| Kernel clock requested / achieved | **130 / 130 MHz** |
| Kernel-clock setup WNS | **+0.004 ns** |
| Fixed `dma_ip_axi_aclk_1` setup WNS | **+0.003 ns** |
| XCLBIN emitted | **yes** |

The initial SLL assignment still warned about localized demand, peaking at 146%
in one SLR0-SLR1 column and 123% in one SLR1-SLR2 column. The detailed router
nevertheless converged to zero failed nets and `route_design completed
successfully`. This confirms that the retained minimal floorplan and DMA fanout
repair are reproducible build inputs; the final physical placement remains
seed-sensitive, but neither a binary RQS nor automatic clock scaling is required
for this successful result.

The full source-to-XCLBIN command took 39,582 seconds (10 h 59 m 42 s), including
9 h 12 m 24 s in the link. Durable status paths are:

- wrapper log: `diagnostics/iter26_config_only_rebuild/rebuild.wrapper.log`;
- build PID: `diagnostics/iter26_config_only_rebuild/rebuild.pid`;
- numeric exit marker: `diagnostics/iter26_config_only_rebuild/rebuild.exit`;
- final artifact summary: `diagnostics/iter26_config_only_rebuild/rebuild.done`;
- detailed implementation log:
  `build.hw.gdn32x2p4auxsharec8s1eastdmaf64v2.f130.o8.v2022_2/_x_temp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log`.

### iter27 — selective hardware-counter profiling

*Launched 2026-07-27 12:31 +03; requested kernel frequency: 130 MHz.
Diagnostic build:
`build.hw.profile.iter27.selective6.f130.o8.v2022_2/`.*

The production iter26 XCLBIN was first measured on card without implementation
instrumentation. A 64-token decode-from-state run was exact for all 64 tokens;
the 63 timed kernel calls averaged **1610.693 ms**, with a 1610.660–1610.997 ms
range. The result is stored under
`diagnostics/iter27_profile/baseline130/`. This is effectively identical to the
prior iter24b 109 MHz latency despite iter26 reporting requested and achieved
`DATA_CLK=130 MHz`, which is strong evidence that long dynamic stalls, rather
than clock-scaled arithmetic latency, dominate the integrated kernel.

The first profiling link reuses the exact iter26 HLS object
(`33e3a6cf504833b7058659f510b08c2bd49ddcd280012b395be3975ac70dd7c5`)
and the same connectivity, floorplan, fanout, place, physical-optimization and
route inputs. It changes only link-time instrumentation:

- AXI performance counters on `M_AXI_MEM_WEIGHTS_MM0`, `MM1`, `MM8`, `MM16`,
  `MM24`, and `MM31`;
- `:counters` mode only, which also supplies CU execution accounting;
- no `-g`, no device trace request, and no all-interface profiling.

The selected ports cover shared HBM0 and spatially distributed GEMV shards
without paying the area/routing cost of 32 monitors. Build PID, wrapper log,
exit marker and manifest are under
`diagnostics/iter27_profile/selective6_f130/`. The detailed implementation log
is
`build.hw.profile.iter27.selective6.f130.o8.v2022_2/_x_temp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log`.
Before accepting hardware measurements, the diagnostic image must still route,
meet 130 MHz, contain no trace S2MM, and remain close to the uninstrumented
1610.693 ms control latency.

The generated debug layout contains exactly one `ACCEL_MONITOR` and six
`AXI_MM_MONITOR` entries and no trace IP. A detached completion monitor (PID
stored in `on_card_monitor.pid`) will automatically run eight decode tokens on
device 0 after a successful link and preserve correctness and kernel-latency
results under `diagnostics/iter27_profile/selective6_f130/on_card/`. It records
a skip marker instead if implementation fails. The final single-call profile
summary and XRT run summary are retained separately under `on_card_single/`.

**Result:** iter27 completed successfully at 22:05 +03 after approximately
9 h 34 m. Routing finished with zero unrouted nets and zero final node
overlaps; final setup WNS/TNS was **+0.003 ns / 0.000 ns**, hold WHS/THS was
**+0.009 ns / 0.000 ns**, and the XCLBIN retained an achieved 130 MHz
`DATA_CLK`. The 78,366,732-byte artifact has SHA-256
`b455113913b56b063423f30af55bf083f0b9017e30177150f48e366135143787`.

The eight-token automatic run was exact and averaged **1610.670 ms** across
seven timed calls, only 0.023 ms below the uninstrumented 1610.693 ms control.
The counter instrumentation therefore caused no measurable runtime
perturbation. A follow-up one-call capture, saved under
`diagnostics/iter27_profile/selective6_f130/on_card_single/`, avoids aggregate
counter overflow and provides the decisive bottleneck evidence:

| Port / direction | Transfers per token | Average transfer | Transfer rate | Average latency |
|---|---:|---:|---:|---:|
| HBM1 read | 42,704 | 4.096 KB | 8,272.71 MB/s | 232.342 ns |
| HBM8 read | 42,704 | 4.096 KB | 8,282.51 MB/s | 266.541 ns |
| HBM16 read | 42,704 | 4.096 KB | 8,282.80 MB/s | 261.972 ns |
| HBM24 read | 42,704 | 4.096 KB | 8,280.74 MB/s | 259.228 ns |
| HBM31 read | 42,704 | 4.096 KB | 8,278.17 MB/s | 266.015 ns |
| **shared HBM0 write** | **13,211,489** | **0.004 KB** | **40.6776 MB/s** | **552.754 ns** |

Each monitored normal weight port reads 174,915,584 bytes per token and
sustains approximately **8.28 GB/s** while active, essentially matching the
263.063 GB/s / 32 = 8.22 GB/s per-port standalone GEMV result. Thus the
integrated GEMV weight path is healthy and consumes only about **21.1 ms** of
active transfer time per token. The bottleneck is the shared HBM0 path,
especially its **13.21 million scalar four-byte writes per token**. It writes
57.3998 MB at only 40.68 MB/s and accounts for roughly 1.41 seconds of
transfer-active time. The next architecture change should burst/vectorize the
recurrent state save path (and its matching restore path), not modify the GEMV
MM2S/cluster engine.

### iter28 — explicit Pack16 recurrent-state transfers

*Started 2026-07-27; requested kernel frequency: 130 MHz.*

Iter27 proved that the integrated GEMV datapath is healthy while the recurrent
state path issues millions of scalar AXI requests. Iter28 changes only the HBM
restore/save loops in `gdn_recurrent_attention`: both now cast the recurrent
state view to `Pack16 *`, issue one explicitly indexed 512-bit word per loop
iteration, and unpack/pack its 16 FP32 lanes into/from the existing eight-bank
dual-port URAM working state. The arithmetic, FP32 state representation,
workspace layout, 32-port GEMV, MM2S/FIFO decoupling, and physical implementation
recipe are unchanged.

Expected transaction-count change per token:

| Direction | iter27 scalar requests | iter28 Pack16 requests | Reduction |
|---|---:|---:|---:|
| State restore | 12,582,912 × 4 B | 786,432 × 64 B | 16× |
| State save | 12,582,912 × 4 B | 786,432 × 64 B | 16× |

Acceptance gates are: native decode exact match; HLS reports one wide external
request per packed iteration rather than 16 scalar requests; successful route
and timing using the retained iter24b/iter26 130 MHz configuration; exact
on-card decode; and a material reduction from the iter27 **1610.693 ms/token**
uninstrumented baseline. Final synthesis, implementation, and on-card results
will be appended here when available.

An isolated csynth probe of the exact Pack16/eight-bank-URAM transfer pattern
completed in 24 seconds and confirmed the intended AXI structure before the
full-kernel synthesis finished:

- inferred restore and save bursts are both **512 bits/beat**;
- the scheduled external restore operation is one `read i512`, and both
  `m_axi` `RDATA` and `WDATA` are 512 bits;
- one layer uses 32,768 packed beats in each direction;
- restore achieves **II=2** because each Pack16 needs two writes into each of
  the eight 1R1W URAM banks; save achieves **II=1**;
- estimated one-layer latency is 65,539 restore + 32,771 save = 98,310 cycles,
  versus approximately 524,359 + 524,359 cycles for the scalar implementation.

Thus csynth predicts a **10.7× state-copy cycle reduction** while definitively
eliminating four-byte external state transactions. The full integrated csynth
remains the gate for shared-mm0 scheduling, and hardware counters remain the
final confirmation of realized transfers and latency.

**Implementation result:** iter28 compiled and routed completely, but VPL
rejected the image on the fixed 250 MHz `dma_ip_axi_aclk_1` platform clock.
Setup WNS/TNS was **-0.205 ns / -2.749 ns** across 34 endpoints. The critical
path was wholly inside the HMSS response FIFO at
`path_12/slice0_12/inst/r15.r_multi`: `fifoaddr_reg[5]` drove 521 loads and the
path spent 3.843 ns of 4.048 ns (**94.9%**) in routing, including 3.580 ns on
that one high-fanout net. The prior iter23 constraint still ran, but it targets
the older `state[1]` reset net; generated implementation Tcl also confirmed
that explicit post-route physical optimization was disabled. Consequently no
XCLBIN was emitted and the automatic on-card test was correctly skipped.

The Pack16 kernel itself became smaller (423,115 LUTs and 541,519 registers),
but placement shifted: SLR0 reached 99.06% CLB utilization while SLR1 and SLR2
were 75.99% and 58.38%. Iter28 therefore remains a routing-complete kernel
architecture with one narrowly identified fixed-platform timing failure, not
a recurrence of the earlier unrouted-net congestion wall.

### iter29 — localize the DMA address net and run post-route phys-opt

*Started 2026-07-28 11:55 +03; requested kernel frequency: 130 MHz.*

Iter29 reuses the exact verified iter28 XO (SHA-256
`8421d5dea9ddfe0920f66013d6be44acab9bad6e8d4ee4a41a2a5d3fa9fe0823`);
there is no HLS source or topology change. A strict pre-place hook retains the
successful iter23 reset-net treatment and additionally selects only the
measured `path_12/slice0_12/r15` `fifoaddr_reg[5]` net, verifies its expected
400–600-load range, and applies clock-region-aware `FORCE_MAX_FANOUT=64`.
The implementation recipe also enables
`POST_ROUTE_PHYS_OPT_DESIGN=AggressiveExplore`, while preserving the 32 HBM
mapping, iter22 cluster-8 floorplan, placement directive, and 130 MHz kernel
clock.

This deliberately stacks two independent timing levers in one costly build:
pre-placement replication shortens the critical branches by construction, and
post-route physical optimization can repair the remaining measured wire delay
after routing. Acceptance requires zero unrouted nets, non-negative WNS on all
system clocks (especially `dma_ip_axi_aclk_1`), exact eight-token on-card
decode, and a material speedup over the iter27 1610.693 ms/token baseline.

**Result:** iter29 exited 1 at 23:45 +03 after 11 h 50 m. Routing was fully
legal with zero final node overlaps and zero unrouted nets. The new constraint
matched the intended 521-load `r15/fifoaddr_reg[5]` net and caused 260
replicas, so that path disappeared from the leading failures. It also
perturbed an already saturated placement: SLR0 remained **99.06%** occupied by
CLBs, SLR0↔SLR1 traffic rose to 16,922 SLLs (73.45%), and the kernel route
finished at WNS/TNS **-1.384 ns / -8666.314 ns**.

Explicit post-route `AggressiveExplore` ran for 2 h 23 m. It optimized 109
critical nets and recovered 0.383 ns, finishing at overall WNS/TNS
**-1.002 ns / -8246.182 ns**. The scalable kernel group accounted for
-1.002 ns / -8234.270 ns across 19,102 endpoints. The fatal unscalable
`dma_ip_axi_aclk_1` group finished at **-0.223 ns / -11.913 ns** across 127
endpoints, so VPL emitted no XCLBIN and the automatic on-card test skipped.

The worst DMA path moved to the `w15` write-response FIFO's
`asyncclear_state1_inst` → `mesg_reg_reg[250]` cone. Its 582-load `state[1]`
net consumed 3.816 ns, and routing was 3.888 ns of the 4.063 ns data path
(95.7%). This is evidence against chasing another individual fanout net:
iter29 moved the bottleneck and worsened global placement. The next experiment
must remove the iter29 constraint and create real whitespace in SLR0.

### iter30 — move boundary cluster 9 to SLR1, Pack16 at 100 MHz

*Started 2026-07-29 00:29 +03; requested kernel frequency: 100 MHz.*

Iter30 again reuses the bit-identical iter28 Pack16 XO. It removes only
iter29's `fifoaddr_reg[5]` replication, retains the successful iter23
read-response reset-net constraint, and extends the minimal iter22 floorplan by
one topology boundary:

- assign `gemv32_cluster2_9_U0` and `xr_9_U` to SLR1;
- steer private FIFOs `ws_18_U` and `ws_19_U` into the same underused east-side
  SLR1 region used successfully for `ws_16_U` and `ws_17_U`;
- leave collectors, adapters, auxiliary logic, and all other clusters movable.

This is intended to lower SLR0 occupancy from 99.06% by roughly one two-port
cluster while preserving contiguous cluster topology. The estimated additional
~1,042 boundary nets leave aggregate SLR0↔SLR1 SLL usage near 78%; east-side
endpoint steering limits the known per-column risk. At 100 MHz, the iter29
kernel path would gain 2.308 ns of period and cease competing with the fixed
250 MHz DMA group during post-route optimization. The build retains
`AltSpreadLogic_high`, pre-route and post-route `AggressiveExplore`, and
`route_design=Explore`.

Acceptance requires a legal route, WNS ≥ 0 on `dma_ip_axi_aclk_1`, an XCLBIN
at 100 MHz, exact eight-token on-card decode, 512-bit recurrent-state traffic,
and a material improvement over both the 1610.693 ms iter27 integrated control
and the 121.4 ms eight-port reference. Final results will be appended here.

**Result:** iter30 exited 1 at 05:44 +03 on 2026-07-29 after approximately
5 h 14 m. Synthesis, placement, and pre-route physical optimization completed
without errors. The full pre-route timing engine reported setup
**WNS/TNS +0.003 ns / 0 ns**, proving that 100 MHz removed the scalable kernel
timing pressure. Placement nevertheless warned that the design was highly
congested, and its estimated maxima regressed relative to iter28/iter29:
north global congestion grew from 64x64 to 128x128, west global from 16x16 to
64x64, and east/west long congestion from 8x8/16x16 to 16x16/16x16.

The cluster-9 move improved the upper SLR1-SLR2 boundary but overloaded the
already critical lower boundary:

| SLL boundary metric | iter29 | iter30 |
|---|---:|---:|
| SLR1-SLR2 total demand | 9,276 / 23,040 (40.26%) | 7,532 / 23,040 (32.69%) |
| SLR1-SLR2 peak column | 130% | 104% |
| SLR0-SLR1 total demand | 13,168 / 23,040 (57.15%) | 14,377 / 23,040 (62.40%) |
| SLR0-SLR1 peak column | 153% | **169%** |

Five SLR0-SLR1 columns were over 100% estimated demand (119%, 169%, 102%,
103%, and 124%). Initial routing assigned every net a provisional path, but
those paths overlapped on **2,113,790 routing nodes**. The final congestion
level was 7, with a 128x128 southbound hotspot spanning much of the lower
device. Route verification reported partially conflicted control nets across
many GEMV clusters, so this is a distributed congestion failure rather than
one repairable fanout cone. No XCLBIN was emitted and the automatic on-card
run correctly skipped.

**Verdict:** reject the cluster-9-to-SLR1 extension. It shifts useful density
out of SLR0 but adds more traffic to the boundary that already limits the
Pack16 design. The next run must restore the route-complete iter22/iter28
topology rather than move another cluster across SLR0-SLR1.

### iter31 — restore the route-complete iter22 topology at 100 MHz

*Started 2026-07-29 05:45 +03 after iter30 failed; requested kernel
frequency: 100 MHz. Detached build PID: 1863675; automatic on-card runner PID:
1863703.*

Iter31 is a configuration-only recovery that reuses the exact iter28 Pack16
XO (SHA-256
`8421d5dea9ddfe0920f66013d6be44acab9bad6e8d4ee4a41a2a5d3fa9fe0823`).
It removes the failed iter30 cluster-9/xr9/ws18/ws19 constraints and restores
the minimal iter22 cluster-8/xr8/ws16/ws17 floorplan that already routed the
same Pack16 netlist completely in iter28. It retains the iter23 localization
of the measured HMSS read-response `state[1]` reset net, targets 100 MHz, and
enables both pre-route and post-route `AggressiveExplore`. It deliberately
does **not** restore iter29's 260-replica `fifoaddr_reg[5]` constraint.

The rationale is evidence-based: iter28 proved this topology routable but
missed only the fixed 250 MHz DMA clock by 0.205 ns, while explicit post-route
physical optimization was disabled. Iter31 restores that legal topology,
removes scalable-kernel timing competition by targeting 100 MHz, and gives the
remaining fixed-DMA wire-delay path a post-route repair pass without
pre-perturbing its saturated placement.

Acceptance remains: zero unrouted or conflicted nets, non-negative WNS on
`dma_ip_axi_aclk_1`, a generated XCLBIN, exact eight-token on-card parity, and
measured latency materially below the iter27 scalar-state result of
1610.693 ms/token. Configuration:
`hw_iter31_pack16_iter22_postphys_f100.cfg`; launcher:
`build_iter31_pack16_iter22_f100.sh`; diagnostics:
`diagnostics/iter31_pack16state_iter22_postphys_f100/`.

**Implementation result:** iter31 completed successfully at 12:47 +03 on
2026-07-29 after 7 h 01 m. The route converged from 522,076 initial node
overlaps through
101,738/18,170/3,301/732/219/77/39/22/14/5/1 to zero. Final route status was
zero failed, unrouted, or partially routed nets and zero node overlaps;
verification completed successfully. Effective routed congestion peaked at
level 4, down from the initial global/timing estimates of level 6/7. The
restored floorplan also corrected the SLL regression seen in iter30:

| Initial SLL metric | iter30 | iter31 |
|---|---:|---:|
| SLR0-SLR1 total demand | 14,377 / 23,040 (62.40%) | 13,438 / 23,040 (58.32%) |
| SLR0-SLR1 peak column | 169% | 103% |
| SLR1-SLR2 total demand | 7,532 / 23,040 (32.69%) | 7,103 / 23,040 (30.83%) |
| SLR1-SLR2 peak column | 104% | 70% |

Routing improved setup WNS from -0.353 ns through
-0.256/-0.231/-0.138/-0.084 ns to -0.010 ns during delay cleanup. The
authoritative post-route timing report finished at setup WNS/TNS
**+0.003 ns / 0 ns** and hold WHS/THS **+0.004 ns / 0 ns**, so every system
clock, including the fixed 250 MHz DMA clock, met timing. Explicit post-route
`AggressiveExplore` ran but correctly made no netlist changes because WNS was
already non-negative. Auto-scaling estimated 100.9 MHz for the kernel and
selected the requested 100.0 MHz. The emitted 80,374,880-byte XCLBIN is:

`build.hw.iter31.pack16state.iter22.postphys.f100.o8.v2022_2/gdn_forward.xclbin`

with SHA-256
`28ab9b9d51962bb024eefe395e3027f9f815d0596518d68285abfaa209ceb627`.
The reused XO SHA-256 is
`8421d5dea9ddfe0920f66013d6be44acab9bad6e8d4ee4a41a2a5d3fa9fe0823`;
the final config SHA-256 is
`3fd8fa3ae0c088d12557d635b0821ca81653828bedcee61783a6881ca95c0316`.
The committed `build_iter31_pack16_iter22_f100.sh` launcher does not depend on
this ignored build artifact: it recompiles the exact XO from `gdn_model.cpp`
at the original 130 MHz HLS target, verifies the XO hash above, and then links
the proven 100 MHz configuration. From a clean worktree, the source-to-XCLBIN
reproduction command is:

```bash
bash c_impl/build_iter31_pack16_iter22_f100.sh
```

**On-card correctness and performance:** the automatic decode-from-state run
matched the golden trajectory exactly for all 8/8 tokens with 100% top-1
agreement. Across the seven real kernel invocations, latency was
212.900--212.968 ms/token, mean **212.919 ms/token**. This is a
**7.565x speedup** over iter27's 1610.693 ms scalar-state result, but is still
1.754x slower than the 121.4 ms eight-port reference. A separate one-call XRT
profile measured 212.948 ms, confirming the multi-token result. Its summary
reported zero AXI and accelerator monitors in this XCLBIN; therefore it cannot
provide a new per-port bandwidth measurement. A selective debug rebuild would
be required for current Pack16 port counters and could perturb routability.

The exact integrated csynth reports explain the remaining floor at 100 MHz:

| Scheduled component | Cycles/token | Time at 100 MHz | Share of measured time |
|---|---:|---:|---:|
| Recurrent attention, including state restore/save | 5,855,112 | 58.551 ms | 27.50% |
| of which external state restore/save only | 2,362,752 | 23.628 ms | 11.10% |
| All non-GEMV scheduled work, including recurrence | 11,386,885 | 113.869 ms | 53.48% |
| Eight GEMVs/layer plus LM head | 2,798,841 | 27.988 ms | 13.15% |
| Reconstructed static schedule | 14,185,726 | 141.857 ms | 66.63% |
| Measured-minus-scheduled dynamic stalls | 7,106,171 | 71.062 ms | 33.37% |

Pack16 moves 50,331,648 bytes in each direction per token but reduces the
request count from 12,582,912 scalar requests to 786,432 64-byte beats in each
direction. This accounts for the 7.565x end-to-end improvement and proves that
packing was necessary. It does not make state persistent on chip: every layer
still restores and saves its 2 MiB slice, and HBM0 still carries state,
activations, auxiliary tensors, outputs, and weight shard 0 through one master.
The remaining 71.062 ms cannot be assigned to a specific port without a
profiled image, but it is real external-memory/dataflow stall time above the
scheduled 141.857 ms.

**Verdict:** iter31 is the successful Pack16 implementation and the new
routability baseline. Do not reapply the iter29 DMA-address replication or the
iter30 cluster-9 move. Keeping all FP32 recurrent state on chip across decode
steps is not feasible in this implementation: one 2 MiB layer state maps to 96
URAMs, so 24 resident layers would require 2,304 URAMs versus 960 on the U55C,
before accounting for any other storage. The next practical performance lever
is therefore to separate packed recurrent state from the overloaded
`workspace` AXI master, give it its own burst-tuned `m_axi` bundle, and map that
bundle to an HBM bank whose weight traffic occurs in a disjoint phase.
Frequency or additional floorplanning alone does not address the measured
71 ms of external-memory/dataflow stalls.

### iter32 campaign — activation-resident 32-port decode

The iter32 campaign keeps iter31's routed 32-port/16-cluster topology and
100 MHz link target while recovering the non-GEMV efficiency sacrificed to
make that topology fit. The first performance target is a bit-exact result
below the 121.4 ms eight-port baseline; the working acceptance target is
118 ms/token. At 100 MHz this permits at most 12.14 M total cycles, of which
iter31's GEMV consumes 2.799 M.

**Frozen reference before source changes (2026-07-29):**

- repository HEAD: `4cbb464bd2df94fec661cd25a1f932f935d2d1a1`;
- `gdn_model.cpp` SHA-256:
  `e570ccf623abb0801d1aca5652f132fb39a2e97f5eec2c083fd16671b20c257a`;
- integrated static schedule: 14,185,726 cycles at 100 MHz;
- on-card mean: 212.919 ms/token, exact 8/8;
- `make -C c_impl -j8`: pass;
- `decode_correctness_check.sh --fast`: exact 6/6;
- full `decode_correctness_check.sh`: exact 32/32.

The source experiments are gated in three stages before any link: remove the
three Q/K/V copies and two residual-buffer passes; move all transient
activations into reusable BRAM-backed `Pack16` buffers without changing the
kernel ABI; then remove the logits round trip and selectively restore auxiliary
lanes. Rejected variants and their csynth deltas are recorded below as they are
measured.

**iter32A result — in-place convolution retained; HBM residual fusion
deferred.** Native fast and full decode remained exact (6/6 and 32/32).
Production Vitis HLS 2022.2 rejected the otherwise-correct direct residual
store before scheduling: `gemv32_load_x_and_w0` and `gemv32_store` would both
read `mem_weights_mm0` inside the same dataflow region. This is precisely the
shared-master conflict that local activation storage is intended to remove.
The three in-place Q/K/V convolutions are retained. Residual fusion is moved
into iter32B, where both the residual read and result write target local BRAM.
The first 2022.1 `test.tcl` attempt was also rejected because that older pragma
parser does not expand symbolic unroll factors; all QoR gates use the production
2022.2 toolchain used by iter31.

**iter32B/C source checkpoint — local activations and direct argmax
(2026-07-29).** The external `gdn_forward` ABI and workspace layout remain
unchanged. The kernel now loads the embedding once, keeps six reusable
activation arrays in explicitly BRAM-bound local storage, and writes only the
selected token back. Residual projection results accumulate into local `x`;
the LM-head store scans the existing reorder buffer in natural output order
using strict `>` tie-breaking. Native validation after the final buffer-alias
change passed `make -C c_impl -j8`, fast exact 6/6, and full exact 32/32.
Production 2022.2 integrated csynth is the next gate; no hardware build is
authorized unless the cycle, interface, and resource limits above pass.

Two HLS-only forms were rejected before scheduling:

- cyclically partitioned `float[]` storage with `Pack16 *` casts is illegal in
  Vitis HLS (`HLS 214-341`, pointer cast cannot be combined with an array
  transformation);
- direct calls from differently sized local BRAMs caused HLS to specialize
  three complete `gdn_gemv` dataflow graphs, tripling the cluster fabric even
  with an allocation limit.

The retained form uses explicit `Pack16` activation memories plus one
352-word input/output BRAM aperture shared by every large projection. Local
512-bit copy/add loops around the aperture add only about 47.5 K scheduled
cycles per token, while forcing one physical 32-reader/16-cluster GEMV. Q, K,
and V use the same maximum physical BRAM shape so convolution also remains one
shared function instance. V is logically 128 words; padding it to 352 words
does not consume another RAMB18 because each of its 16 banks still fits one
primitive. A forced `allocation` limit on the differently sized convolution
variant was rejected: it demoted the packed tail restore/save to scalar II=16
traffic. The equal-shape form restores 512-bit bursts and one convolution
instance.

**iter32B/C final csynth gate — pass (2026-07-29).** The first direct-argmax
form was rejected after csynth exposed a 288,010-cycle, II=9 scan: it reread
each URAM `Pack16` once per scalar lane. The retained implementation reads each
of the 2,000 LM-head packs once, compares its 16 lanes in parallel, and merges
the 16 lane winners with index-aware tie handling. Its pack scan and merge are
II=1 and take 2,020 and 18 cycles respectively. Strict `>` updates and the
lowest original index preserve the old first-index result. Native validation
after this correction passed build, fast exact 6/6, and full exact 32/32.

As in iter31, the integrated top-level HLS latency is not the token schedule:
the shared GEMV has run-time dimensions, so HLS substitutes its maximum
720,896-pack trip count for all 193 calls and reports 146.352 M cycles. The
archived iter31 synthesis reports the same behavior (150.626 M versus the
14.186 M dimension-correct reconstruction and 21.292 M measured cycles).
Applying the identical dimension-correct reconstruction to the final module
reports gives:

| Scheduled component | Iter31 cycles | Iter32 cycles | Delta |
|---|---:|---:|---:|
| 193 large GEMVs including LM head | 2,798,841 | 2,785,524 | -13,317 |
| RMSNorm (49 calls) | 1,015,476 | 139,405 | -876,071 |
| Tiny GEMV (48 calls) | 116,544 | 103,584 | -12,960 |
| Q/K/V convolution (72 calls) | 588,600 | 624,744 | +36,144 |
| Recurrent attention (24 calls) | 5,855,112 | 5,760,144 | -94,968 |
| Output norm/gate (24 calls) | 161,544 | 128,712 | -32,832 |
| SwiGLU (24 calls) | 1,469,976 | 321,048 | -1,148,928 |
| Activation copy/add/argmax and handoff | 2,179,633 | 54,069 | -2,125,564 |
| **Reconstructed static total** | **14,185,726** | **9,917,230** | **-4,268,496** |

At 100 MHz the retained static schedule is **99.172 ms/token**, 30.1% below
iter31 and 22.23 ms below the 121.4 ms reference before dynamic stalls. It is
also below the 10.8--11.0 M campaign stop threshold, so no auxiliary-lane
increase is retained or synthesized. This avoids spending new routing margin
after the architecture already passed the cycle gate.

Final HLS resources versus the archived iter31/Pack16 synthesis are:

| Resource | Iter31 | Iter32 | Delta | Gate |
|---|---:|---:|---:|---:|
| RAMB18 | 1,383 | 1,511 | +128 | pass (<=128) |
| DSP | 3,167 | 3,171 | +4 (+0.13%) | pass (<=1%) |
| FF | 797,137 | 784,731 | -12,406 (-1.56%) | pass |
| LUT | 866,781 | 819,688 | -47,093 (-5.43%) | pass |
| URAM | 112 | 112 | 0 | pass |

All 128 new RAMB18s are the explicitly BRAM-bound activation/aperture banks;
none is LUTRAM or URAM. The transform and RTL reports contain one `gdn_gemv`
dataflow graph, 16 `gemv32_cluster2` instances, one convolution instance, and
the original `mem_weights_mm0` through `mem_weights_mm31` masters. No AXI
master was added. Workspace activation accesses are only the initial
128-Pack16 embedding read and final one-Pack16 token write; recurrent-state
and convolution-tail transfers remain at their existing offsets. GEMV MAC II
is unchanged at 4 and HLS estimates 205.47 MHz. Final kernel-source SHA-256:
`5a4f079a508c71d476a101776cd6285970230cdb4c5be6a1e0f9a486ac5f21e9`.

**First hardware candidate launched 2026-07-29 20:27 +03.** The source gate
above authorizes the build without auxiliary-lane restoration. Launcher:
`build_iter32_activation_resident_iter22_f100.sh`; configuration:
`hw_iter32_activation_resident_iter22_postphys_f100.cfg` (SHA-256
`0531b6a1856de85425d8131589eb829988eaab37026cdb3c8a80189a40e3ed02`).
The launcher compiles HLS at 130 MHz and links only at 100 MHz with the exact
Iter31 connectivity, Iter22 cluster-8 floorplan, Iter23 DMA hook, BRAM MM2S
FIFOs, and pre/post-route `AggressiveExplore`. Detached build PID: 636486.
`run_iter32_oncard_after_build.sh` (PID 637029) is armed to run exact 8-token
smoke and 64-token decode-from-state measurement only after a successful
XCLBIN.

**Iter32 hardware and on-card result — target achieved (2026-07-30).** The
detached build completed successfully in 8 h 25 m. Vivado routed and verified
the design with **0 failed nets, 0 unrouted nets, and 0 node overlaps**. The
post-route timing report states that all constraints are met:

- overall WNS **+0.003 ns**, TNS 0, WHS **+0.009 ns**, THS 0;
- 100 MHz kernel clock WNS **+0.008 ns**, WHS **+0.009 ns**;
- fixed 250 MHz `dma_ip_axi_aclk_1` WNS **+0.003 ns**, WHS **+0.009 ns**;
- route and bitstream generation completed with zero errors and no automatic
  clock scaling.

The routed design retains high but legal local congestion (effective south
level 6 and west level 5 in SLR0). Global routing nevertheless converged from
888,916 node overlaps to zero in five rip-up/reroute iterations. Post-route
physical optimization ran with `AggressiveExplore`; because route had already
closed both setup and hold, it made no further netlist change.

The generated XCLBIN SHA-256 is
`fdf3993b15bb1b8c5d62c96a0e830f176233a6aff4833013c4011525902cd623`;
the XO SHA-256 is
`03bd931020acd61576b8c309760a92406cd82df9e3a56e81887b7a88fa1914ba`.
The automatic eight-token smoke run and 64-token decode-from-state run both
passed exact token parity. Excluding the initial seed entry, the 63 measured
kernel calls were:

| Metric | Iter32 |
|---|---:|
| Minimum | 98.635 ms/token |
| Maximum | 99.085 ms/token |
| Median | **98.650 ms/token** |
| Mean | **98.660 ms/token** |
| Speedup versus iter31 212.919 ms | **2.158x** |
| Speedup versus 121.4 ms baseline | **1.230x** |

The measured mean is within 0.52 ms of the reconstructed 99.172 ms static
schedule (within normal report-reconstruction variation), so the large iter31
dynamic-stall gap has been removed. Iter32 therefore exceeds both
campaign acceptance thresholds: it is below 118 ms and below the 121.4 ms
baseline with exact 64-token parity. The activation-resident architecture,
without auxiliary-lane restoration, is the new performance and routability
baseline.

### iter33 — selective auxiliary-parallelism recovery

*Started 2026-07-30 12:40 +03; requested kernel frequency: 100 MHz.
Detached build PID: 1606798; automatic on-card runner PID: 1608337.*

Iter33 preserves the successful Iter32 activation-resident ABI, 32 weight
masters, 16 two-port GEMV clusters, MM2S/BRAM-FIFO decoupling, Iter22
cluster-8 floorplan, Iter23 DMA hook, and 100 MHz physical recipe. It separates
the previously shared auxiliary lane constants so recurrence, normalization,
convolution, output gating, and SwiGLU can be evaluated independently. Native
validation passed `make -C c_impl -B -j8`, fast exact 6/6, and full exact
32/32.

The first production Vitis HLS 2022.2 experiment restored only recurrent-state
column parallelism from 8 to 16 lanes. The recurrent module improved from
239,598 to **141,350 cycles/call**:

| Recurrent phase | Iter32 | Recurrent x16 |
|---|---:|---:|
| State restore | 65,539 cycles, II=2 | **32,771 cycles, II=1** |
| Fused state read | 8,200 cycles | **4,105 cycles** |
| Fused state write | 8,201 cycles | **4,106 cycles** |
| State save | 32,771 cycles, II=1 | 32,771 cycles, II=1 |

Across 24 layers, the direct recurrent saving is 2,357,952 cycles/token
(23.580 ms at 100 MHz). Including small scheduling shifts in unchanged
auxiliary modules, the conservative dimension-correct reconstruction is
approximately **7.60 M cycles/token (76.0 ms)** versus Iter32's 9.917 M cycles.
GEMV remains one physical dataflow graph with 16 clusters and MAC II=4; all 32
weight masters remain present. Estimated Fmax remains 205.47 MHz.

The recurrent-only HLS resource delta is deliberately bounded:

| Resource | Iter32 production HLS | Iter33 recurrent x16 | Delta |
|---|---:|---:|---:|
| RAMB18 | 1,511 | 1,511 | 0 |
| DSP | 3,171 | 3,323 | +152 (+4.79%) |
| FF | 780,602 | 799,842 | +19,240 (+2.46%) |
| LUT | 822,219 | 833,040 | +10,821 (+1.32%) |
| URAM | 112 | 144 | +32 |

The extra URAM is the physical cost of doubling the recurrent state's column
banks. It is accepted for this explicitly requested experiment; the other
resource deltas stay small enough to justify one implementation attempt.

**Rejected small-op variants.** A combined sweep restored RMS/output-norm
lanes to 16 and fully pipelined 16-lane SwiGLU. It synthesized successfully,
but the individual payoff and resource cost were:

| Variant | Cycle saving/token | Principal module cost | Decision |
|---|---:|---:|---|
| RMSNorm 8→16 | 6,027 (0.060 ms) | +48 DSP | reject |
| Output norm/gate 4→16 | 74,304 (0.743 ms) | about +144 DSP | reject |
| SwiGLU 4→16 | 278,136 (2.781 ms) | +192 DSP, +17.7 K LUT | evaluate separately |

The full combined top reached 3,686 DSP, 828,708 FF, and 870,461 LUT, so it was
rejected without implementation. A final selective recurrent-x16 +
SwiGLU-x16 synthesis also preserved GEMV II=4 and estimated 205.47 MHz, but
used 3,515 DSP, 814,474 FF, and 850,263 LUT. Its DSP growth is 10.85% versus
Iter32 (10.99% versus Iter31), above the 8% implementation gate, for only
2.78 ms beyond the recurrent-only result. SwiGLU-x16 is therefore also
rejected. The hardware candidate retains Iter32's 8-lane norm and four-lane
convolution/output-gate/SwiGLU datapaths and changes only recurrence to 16.

The reproducible launcher is `build_iter33_recur16_iter22_f100.sh`; it pins
the kernel source SHA-256 to
`5f2cd5fd3482c00cb33a3ab2187918134e60b709af28a85d775b22b205174eec`
and reuses the bit-identical successful Iter32 physical configuration
`hw_iter32_activation_resident_iter22_postphys_f100.cfg`. HLS compiles at
130 MHz and link targets **100 MHz only**. The detached build is active in
`build.hw.iter33.recur16.iter22.postphys.f100.o8.v2022_2`; after a successful
XCLBIN, `run_iter33_oncard_after_build.sh` automatically runs exact 8-token
smoke and 64-token performance/parity tests.

**Hardware result — routed, but rejected on the fixed DMA clock.** The build
exited 1 at 2026-07-30 23:59:31 +03 after 10 h 55 m of link time. Routing
itself completed legally with zero failed or unrouted nets and zero node
overlaps. Before the explicit post-route pass, timing was WNS -0.509 ns,
TNS -98.252 ns, WHS +0.009 ns. Post-route `AggressiveExplore` spent
1 h 42 m optimizing 84 cells/nets and recovered 0.347 ns of WNS plus
93.977 ns of TNS:

| Final post-route metric | Iter33 |
|---|---:|
| Overall WNS / TNS | **-0.162 ns / -4.275 ns** |
| Setup-failing endpoints | **43 / 1,993,813** |
| Overall WHS / THS | **+0.007 ns / 0 ns** |
| 100 MHz kernel WNS / TNS | **0.000 ns / 0 ns** |
| 250 MHz `dma_ip_axi_aclk_1` WNS / TNS | **-0.162 ns / -4.275 ns** |
| Route legality | **0 failed, 0 unrouted, 0 overlaps** |

The first routed kernel failure was a one-LUT control path from
`ws_6_U/dout_vld_reg` to cluster-3 product-register clock enables. Its
1,028-fanout net contributed 9.050 ns of routing and the full path was
98.3% route delay. Post-route physical optimization relocated the FIFO-valid
register and many newly exposed kernel/state-write endpoints, ultimately
closing every 100 MHz kernel setup endpoint without reducing recurrent
parallelism. The remaining failure is entirely in the unscalable platform DMA
clock: all 43 failing setup endpoints belong to `dma_ip_axi_aclk_1`.

Vitis therefore stopped before bitstream generation with
`VPL_TCL 101-2`; no XCLBIN was emitted and the automatic on-card test was
correctly skipped. The final checkpoint is
`level0_wrapper_postroute_physopt.dcp`.

**Detailed routed-checkpoint diagnosis.** Vivado 2022.2 has no command named
`report_timing_analysis`; its detailed equivalent,
`report_design_analysis -timing -setup -show_all -full_logical_pin
-routed_vs_estimated`, was run together with top-200 setup paths, route-delay
distribution, congestion, high-fanout, bus-skew, SLR-utilization, and
hierarchical-utilization reports. All actionable reports completed. The
optional QoR-suggestion tail was stopped after nearly an hour so it would not
compete with the next implementation run.

Every residual failure is in one fixed-platform forward-path SRL FIFO:
`hmss_0/.../path_12/slice0_12/.../w15.../common.srl_fifo_0`. The only source
registers exposed by the failing-path analysis are:

| Source | Fanout | Worst slack | Worst net delay |
|---|---:|---:|---:|
| `fifoaddr_reg[4]` | 1,157 | **-0.162 ns** | 3.718 ns |
| `fifoaddr_reg[2]` | 1,159 | **-0.161 ns** | 3.816 ns |

The worst complete path is only three logic levels
(`SRLC32E -> MUXF7 -> LUT4`) and spends 3.988 ns of its 4.260 ns data delay
(93.6%) in routing. It crosses one clock-region boundary but no SLR boundary;
the routed delay is therefore a local high-fanout placement problem rather
than a kernel arithmetic or SLL-capacity problem. The same checkpoint reports
the 100 MHz kernel clock at WNS 0.000 ns with zero setup failures.

### iter34 — focused `w15` DMA address-fanout repair

*Launched 2026-07-31 00:54 +03; detached build PID: 3056058; automatic
on-card monitor PID: 3056059.*

Iter34 preserves the Iter33 source, recurrent-state x16 parallelism, 32 weight
ports, 16 GEMV clusters, MM2S/BRAM-FIFO decoupling, Iter22 cluster-8 floorplan,
Iter23 DMA-state hook, both `AggressiveExplore` physical-optimization passes,
and the 100 MHz link target. It reuses the exact Iter33 XO
`65eb09fd8e24807e774426b8648685420f6e581509e946605c4eb15089f63287`;
there is no HLS recompile and hence no functional or schedule change.

The new pre-place hook applies `MAX_FANOUT_MODE=CLOCK_REGION` and
`FORCE_MAX_FANOUT=64` only to the two diagnosed `w15` FIFO address nets. This
requests roughly 18 localized copies per source, approximately 36 total. The
limit is intentionally less aggressive than the rejected broad Iter29
replication experiment, which created 260 replicas and displaced the timing
bottleneck. The hook validates exact hierarchy, source-bit identity, and an
expected 1,000--1,300 pre-place fanout before applying either property.

The reproducible command is:

```bash
bash c_impl/build_iter34_recur16_dma_w15_fanout64_f100.sh
```

The launcher pins source, XO, configuration, Iter22, Iter23, and Iter34 hook
hashes and refuses to overwrite an existing build directory. On successful
timing closure, `run_iter34_oncard_after_build.sh` automatically executes
exact 8-token smoke and 64-token performance/parity runs, followed by a
separate low-overhead XRT profile capture so profiling cannot bias the primary
latency measurement.

**Result — rejected pre-place guard, no implementation verdict.** Iter34
exited 1 at 2026-07-31 03:10 +03 after 2 h 15 m. All 227 block-level synthesis
jobs and top-level synthesis completed, and `opt_design` completed with zero
errors. The Iter23 hook matched its expected 524-load net. The Iter34 hook then
matched the exact intended
`path_12/slice0_12/.../w15.../fifoaddr_reg[2]` source, but its safety check
observed 582 direct input loads rather than the 1,000--1,300 range copied from
the post-route timing report. It stopped before `place_design`; consequently
there is no placement, congestion, routing, or timing result and the on-card
monitor correctly skipped.

The discrepancy is a reporting-stage distinction: the final routed
high-fanout/timing analysis counted 1,159 timing loads, while the pre-place net
queried by the hook exposes 582 direct sink pins. The exact hierarchy and
source bit were correct. Iter34 therefore rejects only its overly strict
validation range, not the fanout-localization strategy.

### iter35 — corrected pre-place fanout guard

*Launched 2026-07-31 03:55 +03; detached build PID: 3128824; automatic
on-card monitor PID: 3128825.*

Iter35 is identical to Iter34 except that the exact-target direct-fanout guard
accepts 400--700 loads, covering the measured 582-load pre-place form for the
two diagnosed address bits. `MAX_FANOUT_MODE=CLOCK_REGION` and
`FORCE_MAX_FANOUT=64` remain unchanged; recurrent-state x16 and all kernel,
connectivity, floorplan, clock, placement, routing, and physical-optimization
inputs remain unchanged. The successful run reused the exact Iter33 XO. The
committed launcher instead recompiles the pinned kernel source at 130 MHz,
verifies that the generated XO has the same SHA-256 as the successful run, and
then performs the 100 MHz link. It can optionally seed the link with Iter34's
completed 1.8 GiB IP cache; cache absence only increases build time and is not
a functional dependency. No generated XO, RQS binary, or prior build directory
is required for a clean reproduction.

The reproducible command is:

```bash
bash c_impl/build_iter35_recur16_dma_w15_fanout64_f100.sh
```

**Hardware result — successful.** Iter35 completed at 2026-07-31 10:55 +03
after 7 h 0 m and produced XCLBIN SHA-256
`e7dace2c1343502e9e9d976b298d3512ebf9502d396e930e6211da4f6bfa2a66`.
Both diagnosed targets matched before placement:

| Pre-place target | Direct loads | Applied properties |
|---|---:|---|
| `w15...fifoaddr_reg[2]` | 582 | `CLOCK_REGION`, `FORCE_MAX_FANOUT=64` |
| `w15...fifoaddr_reg[4]` | 580 | `CLOCK_REGION`, `FORCE_MAX_FANOUT=64` |

Routing completed legally with zero failed, unrouted, or partially routed nets
and zero final node overlaps. Final timing met every constraint:

| Final metric | Iter35 |
|---|---:|
| Overall setup WNS / TNS | **+0.003 ns / 0 ns** |
| Overall hold WHS / THS | **+0.006 ns / 0 ns** |
| 100 MHz kernel WNS / TNS | **+0.023 ns / 0 ns** |
| 250 MHz `dma_ip_axi_aclk_1` WNS / TNS | **+0.003 ns / 0 ns** |
| Achieved `DATA_CLK` | **100 MHz** |

The post-route physical-optimization pass found no setup violation and
therefore made no netlist modification. The pre-place fanout localization
itself closed the fixed-DMA deficit while retaining recurrent x16. Placed
kernel resources were 413,001 LUTs (13,459 LUTRAM), 530,448 registers,
1,209 BRAMs, 144 URAMs, and 3,324 DSPs. Routed CLB occupancy by SLR was
54,002 / 39,354 / 27,714, or **98.26% / 72.88% / 51.32%**. Initial routing
still reported localized lower-boundary SLL demand peaks of 128%, but the
router resolved them completely.

**On-card result — exact and 75.062 ms/token.** The first automatic run
exposed a host-only XRT issue: a 16 MiB BO sync at the packed recurrent
state's nonzero workspace offset returned `EINVAL` before any kernel launch.
Reducing the host's transfer chunk to 8 MiB left the FPGA ABI and measured
kernel unchanged and made the upload reliable. The repeated eight-token smoke
test was exact. The full 64-token decode matched all 64 tokens with no
divergence:

| 63 timed kernel calls | Iter35 |
|---|---:|
| Mean | **75.061694 ms/token** |
| Median | **75.025197 ms/token** |
| Min / max | 75.016230 / 75.583847 ms |
| Speedup over Iter32 98.659598 ms | **1.314x** |
| Speedup over 121.4 ms reference | **1.617x** |

The measured latency agrees with the reconstructed recurrent-x16 static
schedule of approximately 76.0 ms, showing that the recovered recurrence
parallelism translates essentially one-for-one on card and that little
dynamic-stall gap remains at this stage.

A separate two-token XRT profile run was also exact and measured the real
kernel invocation at 75.064 ms. The production XCLBIN contains zero AXI,
accelerator, or stream monitor IP, so XRT correctly reports zero device
counters; per-port bandwidth or stall attribution cannot be recovered from
this image. That deeper measurement requires a separate selectively
instrumented link and must be compared against the uninstrumented
75.061694 ms control to detect probe perturbation.

### iter36 — head-local recurrent-state fusion

*Started 2026-07-31; first hardware target: 100 MHz.*

Iter36 preserves Iter35's 32 HBM weight readers, 16 two-port GEMV clusters,
activation-resident workspace ABI, recurrent-state FP32 representation,
Iter22 cluster-8 floorplan, Iter23 DMA repair, Iter35 focused `w15` repair,
and physical-optimization directives. It changes only recurrent-state
materialization.

Iter35 restored all eight heads (2 MiB) into a 128-URAM layer buffer, computed
one head at a time, then saved all eight heads. The restore and save were each
separate 32,771-cycle layer traversals. Iter36 retains only one 256 x 256 FP32
head in a 16-bank URAM buffer. The existing retrieval pass now reads each
old-state Pack16 directly from HBM and retains it locally; the existing update
pass writes each updated Pack16 directly back to the same external layout.
The FP32 expressions and per-head element order are unchanged.

Native validation passed a clean build, fast exact parity 6/6, and full exact
parity 32/32. Integrated Vitis HLS 2022.2 synthesis completed successfully:

| Metric | Iter35 recurrent x16 | Iter36 head-local | Delta |
|---|---:|---:|---:|
| Recurrent latency/call | 141,350 cycles | **76,713 cycles** | -64,637 |
| Fused read | 4,105 cycles, II=1 | **4,105 cycles, II=1** | unchanged |
| Fused update | 4,106 cycles, II=1 | **4,106 cycles, II=1** | unchanged |
| Standalone restore/save | 32,771 + 32,771 | **removed** | -65,542 |
| Recurrent BRAM / URAM | 20 / 128 | **20 / 16** | 0 / -112 |
| Recurrent DSP | 263 | **263** | 0 |
| Recurrent FF / LUT | 38,632 / 42,591 | **37,523 / 39,077** | -1,109 / -3,514 |

HLS infers 4,096-beat, 512-bit bursts in both fused state loops. The complete
kernel remains one GEMV dataflow graph with all 32 movers at II=1 and all 16
cluster MAC loops at II=4. Whole-kernel resources are 1,511 RAMB18, 3,323 DSP,
798,733 FF, 829,526 LUT, and **32 URAM**, versus Iter35's 144 URAM. No new AXI
master is present.

Replacing 24 recurrent calls in the dimension-correct Iter35 reconstruction
gives approximately 5,998,198 cycles/token, or **59.982 ms at 100 MHz**. The
successful 100 MHz build uses source SHA-256
`b0a380365d00a7535dd1256f62f6a21f97a3eee6158e3e4b53bb92ce2df5dafb`
and the exact Iter35 implementation config/Tcl hashes. It completed in 7 h 47 m
(7 h 25 m 51 s link), routed with zero failed/unrouted nets and zero final node
overlaps, and emitted a 77,139,984-byte XCLBIN. Post-route timing met every
constraint: overall WNS/WHS are **+0.003/+0.009 ns**, the 100 MHz kernel clock
has +0.104 ns setup slack, and `dma_ip_axi_aclk_1` has +0.003 ns setup slack.

The automatic eight-token smoke test and full 64-token decode both matched the
golden trajectory exactly. Excluding the initial seed, 63 timed kernel calls
measured:

| Metric | Iter36 100 MHz |
|---|---:|
| Mean | **59.577939 ms/token** |
| Median | **59.562983 ms/token** |
| Min / max | 59.517141 / 59.850744 ms |
| Speedup over Iter35 75.061694 ms | **1.260x** |
| Speedup over Iter32 98.659598 ms | **1.656x** |
| Speedup over 121.4 ms reference | **2.038x** |

The 59.578 ms hardware mean is within 0.404 ms of the reconstructed static
schedule, confirming that the fused head-local transfer removes the two
standalone state traversals without creating a new stall term. Reproduction is
one command: `bash c_impl/build_iter36_headlocal.sh 100`. Artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| XO | `c699dcc8c678cba970906d1a9ab0d62ab2315ad7d67c77fdcdf921dbd752171e` |
| XCLBIN | `b251ca1c89a2d56111c1c336e003db3c3c0bbaf02556207a50a184578c6f3c34` |

#### Iter36 frequency follow-up — requested 130 MHz, validated at 115.7 MHz

The gated follow-up reused the exact Iter36 source and physical recipe and
changed only the link request from 100 to 130 MHz. Source SHA-256 remained
`b0a380365d00a7535dd1256f62f6a21f97a3eee6158e3e4b53bb92ce2df5dafb`;
the config SHA-256 remained
`240855bd65ba1cf4525c88b34f9d07012bf73c90e75f524b2c9547eaa9e5922b`.
The one-line reproduction command is
`bash c_impl/build_iter36_headlocal.sh 130`.

The build completed all routing, post-route `AggressiveExplore`, DRC, and
bitstream phases in **31 h 31 m 16 s**. At the requested 130 MHz, the final
post-physopt report has overall WNS/TNS **-0.948/-10111.593 ns**, 19,904
failing setup endpoints, and WHS **+0.001 ns**. The kernel setup paths therefore
did not close at 130 MHz. Vivado auto-frequency scaling selected an achieved
**115.7 MHz** `DATA_CLK`; the fixed 250 MHz DMA same-clock setup path remained
positive. The emitted 79,292,496-byte XCLBIN is a valid 115.7 MHz artifact, not
a 130 MHz closure result.

The automatic eight-token smoke test and full 64-token decode both matched the
golden trajectory exactly with first divergence `-1`. Excluding the initial
seed, 63 kernel calls measured:

| Metric | Iter36 auto-scaled 115.7 MHz |
|---|---:|
| Mean | **51.844010 ms/token** |
| Median | **51.813378 ms/token** |
| Min / max | 51.776322 / 52.103974 ms |
| Improvement over Iter36 100 MHz | **1.149x / 12.98%** |
| Speedup over Iter35 75.061694 ms | **1.448x** |
| Speedup over Iter32 98.659598 ms | **1.903x** |
| Speedup over 121.4 ms reference | **2.342x** |

The ideal 100/115.7 clock-ratio prediction from the 59.577939 ms control is
51.493465 ms. The measured mean is only 0.68% above it, confirming that the
higher clock did not introduce a material dynamic stall. This is a retained
performance improvement even though the original 130 MHz timing objective was
not met.

| Artifact | SHA-256 |
|---|---|
| XO | `01b8bf9f19fd62c4762fe16d77235212731cad0b31e48601fbc0f37608ef7f6f` |
| XCLBIN | `7f631a7941f5614c91a1eb80246c0dd7fe3e8e83f1eceb1e171487c04498505f` |

### iter37A — four-port recurrent-state striping, first scheduling attempt (rejected)

*Tested 2026-08-02; stopped during integrated csynth before hardware link.*

This sub-variant moved the unchanged FP32 recurrent-state tensor from the
single HBM0 workspace region to four Pack16-striped tails on existing weight
ports 28--31 and widened recurrent column arithmetic from 16 to 32 lanes. The
native fast 6-token and full 32-token decode gates were both exact. HLS also
inferred the intended 1,024-beat, 512-bit bursts for all four read streams and
all four write streams; the 32 GEMV MM2S loops remained at II=1.

The first flattened low/high implementation did not meet its recurrent-loop
schedule gate:

| Loop | Target II | Achieved II |
|---|---:|---:|
| Low-half state read (`fused_rd01`) | 1 | **1** |
| High-half state read (`fused_rd23`) | 1 | **2** |
| Low-half update/write (`fused_wr01`) | 1 | **2** |
| High-half update/write (`fused_wr23`) | 1 | **3** |

The high-half read alias is caused by sharing one dynamically indexed
low/high accumulator array. The write-loop warnings are false state-array
carried dependencies: `(row,column)` is unique for every flattened loop
iteration, so no state element is revisited in that pass. Because these IIs
would miss the planned <=50K recurrent cycles/layer, synthesis was terminated
during RTL generation and no link was launched. Iter37B splits low/high
accumulator arrays and applies an explicit false inter-iteration dependency
only to the one-write state update loops. This rejected source/config is not a
commit candidate.

### iter37B — four-port recurrent-state striping, 115 MHz routing failure

*Tested 2026-08-02; implementation failed during route verification.*

Iter37B retained Iter37A's four-way Pack16-striped recurrent-state placement on
the existing weight ports 28--31 and its 32 recurrent arithmetic lanes. It
split the low/high accumulator storage and applied a false inter-iteration
dependency only to the flattened one-write update loops. Native fast 6-token
and full 32-token decode checks remained exact. Integrated Vitis HLS 2022.2
synthesis achieved II=1 on all four state read/write loops, with a top-level
minimum latency estimate of **3,957,551 cycles** versus Iter36's 4,741,679.
Each recurrent layer was estimated at 43,873--44,081 cycles. Whole-kernel HLS
resources were 1,511 RAMB18, 3,627 DSP, 851,903 FF, 912,694 LUT, and 48 URAM;
the 32 weight movers remained II=1 and the 16 GEMV clusters remained II=4.

The full implementation used the accepted source SHA-256
`88e68abdcba29f4355216571440d70d8611f0855717466e8f73891a3db58b216` and XO SHA-256
`5846289626acf27d200a098038ed934432fe16730071c8e185d9e9c5fc766626`.
It requested 115 MHz while preserving the Iter36/Iter35 physical recipe. The
placer completed, but routing reported global/short congestion level 6 and
timing congestion level 7. Rip-up/reroute reduced the first iteration from
1,140,068 to 3,784 overlapping nodes, but the next iteration did not converge.
Final verification reported **6,708 failed-to-route signals** and **5,845 node
overlaps**, then `route_design` failed with partially conflicted nets. No
XCLBIN was emitted, so there is no timing or on-card performance result.

Post-failure checkpoint analysis localized the new pressure. The recurrent
hierarchy was placed almost entirely in SLR1: 17,681 CLBs, 71,153 LUTs,
67,399 registers, 568 DSPs, 32 URAMs, and 13.5 BRAM tiles were in SLR1; only
781 CLBs and 1,921 registers spilled into SLR0, and none entered SLR2. It
accounted for 47% of a level-7 long-routing congestion window. Relative to the
successful Iter36 130 MHz-request placement, total SLL use rose from 26,417 to
31,643, SLR0--SLR1 crossings rose from 16,938 to 19,455, and SLR1--SLR2
crossings rose from 9,479 to 12,188. SLR0 was 99.54% occupied by CLBs and SLR1
was 87.11%, while SLR2 remained only 50.36% occupied. These measurements make
the next repair a recurrent-specific redistribution/reduction problem, not a
clock-frequency problem or a shortage of total device resources.

This is a **negative implementation result** and must not be committed as an
improvement. It shows that the additional recurrent-state striping/32-lane
logic exceeds the routability margin of the current 115 MHz physical recipe;
the preliminary -0.013 ns timing seen before detailed rerouting is not a final
timing verdict. A 100 MHz fallback was not launched from this status check.

### iter37C — recurrent-only SLR2 redistribution (rejected hook guard)

*Tested 2026-08-02--03 at 115 MHz; stopped before `opt_design`.*

Iter37C keeps the exact bit-exact Iter37B source and verified XO. Its only
changes are physical: retain the Iter22 cluster-8 and Iter35 DMA repairs,
assign the complete `grp_gdn_recurrent_attention_fu_*` hierarchy to the full
SLR2 without a rectangular pblock, use `SSI_SpreadSLLs` placement, and use
`AlternateCLBRouting`. Post-route `AggressiveExplore` remains enabled. The
goal is to move the 17,681 recurrent CLBs and their 568 DSP/32 URAM anchors out
of SLR1 while leaving Vivado free to spread them within the underused SLR2.

The checksum-guarded command was `make -C c_impl iter37c`. It reused
the accepted XO SHA-256
`5846289626acf27d200a098038ed934432fe16730071c8e185d9e9c5fc766626`;
the relocatable config-template SHA-256 is
`e31edd150c44f7745de470b5e95405c7e539a6f8c0ad3e07783ac2393abb0952`
and the recurrent-placement hook SHA-256 is
`3826b82f3266b59a36f8552fdfab46a67825f77ba01a9fa0cf5e3e0e5c014a5c`.
The build completed synthesis, then the pre-optimization hook rejected its own
selector: `NAME =~ */grp_gdn_recurrent_attention_fu_*` matched the hierarchy
root and all descendants because Vivado's glob `*` spans `/`, producing
204,004 matches instead of one. Placement and routing never ran, so this is a
**negative recipe result with no physical-design verdict**. No source, XO, or
frequency result is implicated, and no 100 MHz fallback was launched. Iter37D
replaces the glob with the checkpoint-validated anchored regexp
`^.*/grp_gdn_recurrent_attention_fu_[0-9]+$` and reruns the same experiment.

### iter37D — corrected recurrent-only SLR2 redistribution, routed at 100 MHz

*Tested 2026-08-03--04; 115 MHz timing failed and Vitis emitted an
automatically scaled 100 MHz XCLBIN.*

Iter37D is the corrected retry of Iter37C. It preserves the exact Iter37B
source and XO and changes only the physical recipe. The pre-optimization hook
now uses `get_cells -hierarchical -regexp` with the checkpoint-validated exact
root pattern `^.*/grp_gdn_recurrent_attention_fu_[0-9]+$`; the one selected
recurrent hierarchy is assigned to the full SLR2. Iter22's cluster-8 placement,
Iter35's DMA fanout repair, `SSI_SpreadSLLs`, `AlternateCLBRouting`, and
post-route `AggressiveExplore` remain enabled.

The reproducible command is `make -C c_impl iter37 ITER37_FREQ=115`. The source
SHA-256 remains
`88e68abdcba29f4355216571440d70d8611f0855717466e8f73891a3db58b216`,
the reused XO SHA-256 remains
`5846289626acf27d200a098038ed934432fe16730071c8e185d9e9c5fc766626`,
the corrected hook SHA-256 is
`15587403b5e904345abdee72cd84cfc0fa24f8be559f94f1e5554cdcedbd059e`,
and the relocatable config-template SHA-256 is
`998b71e3a8cb3b7f818f12cbe6581f0ffd2e04010dba5db3f20ca2ae844aa08f`.
No independent 100 MHz implementation was launched; this same run was allowed
to fall back only after its 115 MHz timing failure was established.

The corrected hook selected exactly one recurrent root and implementation
completed routing with **zero failed/unrouted nets and zero node overlaps**.
This confirms that moving the enlarged recurrent block into SLR2 resolves the
Iter37B routing failure. At the requested 115 MHz, however, routed timing was
WNS -1.775 ns/TNS -8697.074 ns before post-route physical optimization.
`AggressiveExplore` recovered 0.479 ns, finishing at **WNS -1.296 ns**, TNS
-5314.403 ns, WHS +0.001 ns, and THS 0.000 ns. The failing path group was the
kernel data clock and included recurrent state RAM/control paths as well as
GEMV-cluster datapaths; this is a broad physical-timing failure rather than the
single DMA path seen in earlier iterations.

Vitis therefore scaled `DATA_CLK` from 115 to **100 MHz**, generated the
bitstream, and exited successfully after 14 h 32 min. The resulting XCLBIN is
77,620,790 bytes with SHA-256
`5dd2c7c460f635f3bc3cbe931c365549f47d33653d679871953c161b62da3524`.
The eight-token smoke run and full 64-token decode-from-state run on U55C device
0 both matched the golden trajectory exactly, with first divergence `-1` and
100% top-1 agreement. Excluding the initial seed, the 63 full-run kernel calls
measured:

| Metric | Iter37D, auto-scaled 100 MHz |
|---|---:|
| Minimum | 51.422932 ms/token |
| Maximum | 51.784876 ms/token |
| Median | **51.437899 ms/token** |
| Mean | **51.450918 ms/token** |
| Speedup over Iter36 100 MHz, 59.577939 ms | **1.158x / 13.64%** |
| Speedup over Iter36 115.7 MHz, 51.844010 ms | **1.0076x / 0.76%** |
| Speedup over 121.4 ms reference | **2.360x** |

This is a **retained positive result**. Four-port state striping plus 32
recurrent lanes saves about 0.813 million measured cycles/token relative to
Iter36 at the same 100 MHz, and is narrowly faster than the previous best even
though that image ran at 115.7 MHz. The achieved 51.451 ms is higher than the
3.958-million-cycle integrated HLS minimum because the external state streams
still incur about 1.19 million cycles of dynamic transfer/backpressure, nearly
the same residual term as Iter36. The architecture, reproducible physical
recipe, and measured result are therefore commit candidates; the unachieved
115 MHz target remains explicitly recorded as a timing failure.

### iter38A — head-major unified Q/K/V/gate layout (csynth-positive foundation)

*Tested 2026-08-04; no hardware build launched.*

Iter38A changes the first per-layer shard section from four independent
output-striped Q, K, V, and attention-gate matrices into one head-major QKVG
section. Channel `4*head+kind` owns all 256 rows of one Q/K/V/gate head. A
single 8,192-output call to the existing one physical `gdn_gemv` graph replaces
four 2,048-output calls. Weight bytes, row order within each matrix, inner
dimension order, FP32 dot-product order, 32 MM2S readers, 16 two-port clusters,
and the external ABI are unchanged. A native-only validator compares every
copied shard float against its source tensor, including all unaffected matrix
sections and the LM head.

Validation and identities:

- source SHA-256: `a0774f91ab2ca305c605d3daa59618ffac562e21be89369ce9a94b0ed8cd68b4`;
- header SHA-256: `7bf6bf9d567241028d916a5def4924c6efb7031e416e573921c11d4b10fa2e74`;
- evaluator SHA-256: `2712848cab0e054a3e43b6d10e4ef4d2e50e5abc149dc2d6dea5ea12c17db953`;
- `make -C c_impl -j2`: PASS with only pre-existing warnings;
- `bash scripts/decode_correctness_check.sh --fast`: exact 6/6 positions,
  first divergence `-1`;
- `bash scripts/decode_correctness_check.sh`: exact 32/32 positions, first
  divergence `-1`; and
- Vitis HLS 2022.2 integrated csynth at 7.692 ns using Tcl SHA-256
  `b93fb3d3e84bdd0c4f3f8eeff42c04ea9508b65b6f63819eab6971c130f54258`:
  PASS, MM2S II=1 and cluster MAC II=4 unchanged.

| Integrated HLS metric | Iter37D | Iter38A | Delta |
|---|---:|---:|---:|
| Static minimum cycles | 3,957,551 | **3,343,751** | -613,800 |
| RAMB18 | 1,511 | 1,511 | 0 |
| DSP | 3,627 | 3,627 | 0 |
| FF | 851,903 | **851,782** | -121 |
| LUT | 912,694 | **911,889** | -805 |
| URAM | 48 | 48 | 0 |

The 0.614M static-cycle reduction is **not** accepted as a predicted on-card
gain. Because the shared GEMV retains runtime dimensions, top-level csynth
charges each call the module's 8,435-cycle minimum; removing three calls per
layer mechanically removes about 0.607M reported cycles even though the MAC
and weight-stream work is unchanged. The demonstrated result is narrower but
useful: the streaming-friendly head-major layout is exact, has no II or
resource regression, and eliminates three engine startups/activation reloads.
The variant is retained in the working tree as foundation for Iter38B and head
streaming, but is not independently committed or promoted to a hardware
performance result.

### iter38B — pair-interleaved unified MLP gate/up layout (retained foundation)

*Tested 2026-08-04; no hardware build launched.*

Iter38B keeps Iter38A QKVG and changes the two 5,632-row MLP gate/up shard
sections into one pair-interleaved GU section. Channel `2*chunk+kind` owns one
352-row gate or up block, so one 11,264-output GEMV replaces two 5,632-output
calls and exposes bounded gate/up pairs for later chunk streaming. MLP-down,
weight bytes, FP32 dot-product order, the 32-reader/16-cluster engine, and all
external interfaces remain unchanged. The exact shard validator was extended
to cover every GU chunk.

Validation and identities:

- source SHA-256: `92e65cb24e4dbed510a8c9164d4fac118e43206e4da0f3e9d91666b48846b79a`;
- header SHA-256: `7bf6bf9d567241028d916a5def4924c6efb7031e416e573921c11d4b10fa2e74`;
- evaluator SHA-256: `2712848cab0e054a3e43b6d10e4ef4d2e50e5abc149dc2d6dea5ea12c17db953`;
- `make -C c_impl -j2`: PASS with only pre-existing warnings;
- `bash scripts/decode_correctness_check.sh --fast`: exact 6/6 positions,
  first divergence `-1`, including the full-byte shard-layout gate; and
- Vitis HLS 2022.2 integrated csynth at 7.692 ns using Tcl SHA-256
  `f2df1b8633bcf718032e0c482d3abdc8e3f37df94c925211415dd9c539764f67`:
  PASS, GU unpack II=1, MM2S II=1, cluster MAC II=4, estimated Fmax
  167.98 MHz.

| Integrated HLS metric | Iter38A | Iter38B | Delta |
|---|---:|---:|---:|
| Static minimum cycles | 3,343,751 | **3,141,239** | -202,512 |
| RAMB18 | 1,511 | 1,527 | +16 |
| DSP | 3,627 | 3,629 | +2 |
| FF | 851,782 | 851,881 | +99 |
| LUT | 911,889 | **911,752** | -137 |
| URAM | 48 | 48 | 0 |

As with Iter38A, almost all of the reported cycle reduction is the shared
dynamic GEMV module's 8,435-cycle minimum being charged one fewer time per
layer; it is not an on-card prediction. The 16 RAMB18 increase is the enlarged
704-Pack16 common result aperture and the two DSPs are address arithmetic for
the 22-pack chunk mapping. This remains inside the Stage-3 acceptance bound,
does not change any throughput II, and enables tagged/chunk consumption. It is
retained as foundation in the working tree, but is not independently committed
or claimed as a demonstrated hardware-speed improvement.

### iter38C — merged four-port recurrent-state loops (rejected)

*Tested and stopped during integrated csynth on 2026-08-04.*

Iter38C merged the serial low-port-pair and high-port-pair state traversals
into one four-port read loop and one four-port write loop. The arithmetic order
within every state column was unchanged, and both native gates passed: exact
6/6 fast positions and exact 32/32 full positions with first divergence `-1`.
The candidate source SHA-256 was
`4e44e0e2faf29e8e116dbd86ee9f16eb571bb507a6635ff0e9a7fc334c565ba6`;
the 7.692 ns csynth Tcl SHA-256 was
`20ec24a5a719005755ec85d9f835de72bb1155c42f6ee746971f500076a610d8`.

Integrated HLS rejected the schedule before report completion:

- `fused_rd0123`: target II=1, **final II=2**;
- `fused_wr0123`: target II=1, **final II=3**; and
- both violations were explicitly attributed to limited ports on the single
  cyclically partitioned `state` array. Low and high columns select the same
  modulo-32 URAM bank, so the merged loop requests two read-modify-write accesses
  per bank per cycle.

At II=2 the merged read merely equals the two former 1,024-cycle II=1 loops;
at II=3 the merged write is about 1,024 cycles/head worse than the two former
write loops. The synthesis was stopped after this decisive negative result to
avoid spending time on RTL/report generation. This variant is rejected and
must not be committed. Iter38D retains concurrent HBM access but splits the
head-local state into independent low/high URAM arrays so each half has its own
bank ports.

### iter38D — concurrent state with struct-paired local storage (superseded)

*Tested 2026-08-04; no hardware build launched.*

Iter38D fixed Iter38C's port conflict by storing each low/high column pair in a
`GDNStatePair` element at the same cyclic bank address. Native compilation and
the six-token exact gate passed, and integrated HLS achieved II=1 for both the
1,024-word four-port read and four-port write loops. Recurrent latency fell
from 43,873--44,081 to **27,329--27,537 cycles/layer**, or 3,416--3,442
cycles/head instead of 5,484--5,510. Unlike the QKVG/GU call-count effect, this
16,544-cycle/layer reduction is a real fixed-trip-count schedule change.

The candidate source SHA-256 was
`eb5cbf5e566e62eac6e25084c94f2c337a4360fe75b567e4a8147904bd20f05f`;
the Vitis HLS 2022.2 7.692 ns Tcl SHA-256 was
`3dd3e05a817b40230b3f0c9ef97b65d46f773f9bf76cbfbcd477f072d2dcbd79`.

| Integrated HLS metric | Iter38B | Iter38D | Delta |
|---|---:|---:|---:|
| Static minimum cycles | 3,141,239 | **2,744,183** | -397,056 |
| Recurrent cycles/layer, minimum | 43,873 | **27,329** | -16,544 |
| RAMB18 | 1,527 | 1,527 | 0 |
| DSP | 3,629 | 3,629 | 0 |
| FF | 851,881 | 859,711 | +7,830 |
| LUT | 911,752 | 933,045 | +21,293 |
| URAM | 48 | **80** | +32 |

The C struct was automatically decomposed into 32 separate low-field and 32
separate high-field 1,024x32 memories, doubling recurrent URAM from 32 to 64
instead of producing the intended 32 memories at 1,024x64. The cycle result is
positive, but this avoidable 32-URAM and 21K-LUT routing risk is not accepted as
the final implementation. Iter38D is superseded and not committed; Iter38E
adds an explicit aggregate/packing directive and requires the same II/cycles
with the original URAM count before any hardware build.

### iter38E — packed four-port concurrent state, QKVG, and GU (hardware candidate)

*Native, integrated csynth, 100 MHz implementation, and on-card validation
completed 2026-08-04.*

Iter38E adds `aggregate compact=bit` to the Iter38D state-pair array. HLS now
implements exactly 32 banks of 1,024 x 64-bit URAM words instead of decomposing
the struct fields into 64 banks. The four external state readers and writers
remain concurrent at II=1, recurrent latency remains 27,329--27,537
cycles/layer, and total URAM returns from 80 to the Iter37/Iter38B value of 48.
The candidate also includes the exact head-major QKVG and pair-interleaved GU
layouts from Iter38A/B.

Pre-hardware evidence and identities:

- source SHA-256: `0791d1d158dc476e7f2cc1e44b6bf7790038ea8bf0458138ea2e015ca7184708`;
- header SHA-256: `7bf6bf9d567241028d916a5def4924c6efb7031e416e573921c11d4b10fa2e74`;
- evaluator SHA-256: `2712848cab0e054a3e43b6d10e4ef4d2e50e5abc149dc2d6dea5ea12c17db953`;
- Vitis HLS 2022.2 7.692 ns Tcl SHA-256:
  `380d76359b4c477d4d353173d09eb8ab602ccde21bc5a267c907b1d0e09d9118`;
- native full-byte weight-shard validation: PASS;
- six-token fast trajectory: exact, first divergence `-1`;
- 32-token full trajectory: exact, first divergence `-1`; and
- integrated csynth: PASS, MM2S II=1, GEMV MAC II=4, recurrent read/write
  II=1, estimated Fmax 167.98 MHz.

| Integrated HLS metric | Iter37D | Iter38E | Delta |
|---|---:|---:|---:|
| Static minimum cycles | 3,957,551 | **2,744,183** | -1,213,368* |
| Recurrent cycles/layer, minimum | 43,873 | **27,329** | -16,544 |
| RAMB18 | 1,511 | 1,527 | +16 / +1.06% |
| DSP | 3,627 | 3,629 | +2 / +0.06% |
| FF | 851,903 | 857,343 | +5,440 / +0.64% |
| LUT | 912,694 | 929,941 | +17,247 / +1.89% |
| URAM | 48 | 48 | 0 |

\*About 0.816M of the top-level static delta is the known dynamic-GEMV
call-count accounting artifact from QKVG/GU. The fixed-trip recurrent delta is
0.397M cycles/token and is the defensible pre-hardware gain. Concurrent use of
all four state ports may also reduce dynamic HBM backpressure, but only the
on-card run can measure that. At 100 MHz the conservative expectation is about
4.75M cycles / 47.5 ms per token versus Iter37D's measured 5.145M / 51.451 ms.

The one-line build command was:

```bash
make -C c_impl iter38
```

It compiled HLS at 130 MHz and linked only at 100 MHz with the exact Iter37D
connectivity, cluster-8 placement, recurrent-SLR2 placement, DMA fanout hook,
BRAM FIFOs, `SSI_SpreadSLLs`, `AlternateCLBRouting`, and pre/post-route
`AggressiveExplore`. The wrapper ran from 01:57:53 to 10:34:16 +03 (8 h 36 m)
and exited zero. Artifact hashes were:

- XO: `8533474f648ded537f7b0b7f92d1f3dc17f8d3670c940a7846858e561a144034`;
- XCLBIN: `c209a41b28ee97d6d897c4eaea66974ebb88773c46079c0aeb059d839b9d79dc`.

Implementation completed with zero failed nets, zero unrouted nets, zero
partially routed nets, and zero node overlaps. One hundred percent of nets were
fully routed. Timing closed without automatic clock scaling:

| Routed timing metric | Result |
|---|---:|
| Encoded kernel frequency | **100 MHz** |
| Design WNS / TNS | **+0.003 ns / 0** |
| Design WHS / THS | **+0.006 ns / 0** |
| Kernel clock WNS | **+0.032 ns** |
| Fixed 250 MHz DMA WNS | **+0.003 ns** |

The automatic on-card target completed both runs; its final summary-only `jq`
command initially failed because a quoted Make continuation passed literal
backslashes to `jq`. The already-written JSON and parity results were valid,
and the Makefile quoting was corrected without rebuilding or rerunning. Both
the 8-token smoke and 64-token decode matched exactly with first divergence
`-1`. Excluding the seed, 63 kernel invocations measured:

| On-card metric | Iter37D | Iter38E | Change |
|---|---:|---:|---:|
| Minimum | 51.422932 ms | **47.066309 ms** | -- |
| Maximum | 51.784876 ms | **47.109197 ms** | -- |
| Median | 51.437899 ms | **47.076699 ms** | -- |
| Mean | 51.450918 ms | **47.079335 ms** | **1.0929x / -8.50% latency** |
| Mean cycles at 100 MHz | 5.145M | **4.708M** | **-0.437M / -8.50%** |
| Speedup over 121.4 ms | 2.360x | **2.579x** | -- |

The measured 0.437M-cycle reduction is close to the 0.397M fixed-trip
recurrent prediction. It confirms that the 2.744M HLS minimum is not an
end-to-end hardware cycle count, while demonstrating a real benefit from the
concurrent four-port packed state traversal. Iter38E is therefore a **retained
positive result**. The QKVG/GU layouts, packed state pairs, concurrent state
loops, exact shard validator, and Makefile-only reproducible build/on-card
recipe become the new production baseline.

### iter39A — head-serial/all-port QKVG prerequisite (native + csynth)

*Logged: 2026-08-04.*

**Hypothesis.** Iter38 called its QKVG layout head-major, but tracing the actual
collector order showed that channel `c = head*4 + kind` assigns only four HBM
ports to a head. All eight heads therefore advance concurrently and become
visible together; the layout cannot feed a bounded head pipeline. Re-stripe
each head's 1,024 concatenated Q/K/V/gate rows across all 32 ports, storing
heads sequentially within each shard. This preserves all dot-product FP32
orders and bytes while making one complete head visible every approximately
4,096 weight beats.

**Change.** `gdn_build_weight_shards` and its full-byte validator now map
channels 0--7 to Q segments, 8--15 to K, 16--23 to V, and 24--31 to the gate;
each channel stores two `Pack16` results per head. The local unpack reverses
the collector's channel-major order back to the existing natural Q/K/V/gate
buffers. No kernel ABI, AXI master, GEMV arithmetic, weight volume, or model
operation changed.

**Identity and validation.** Working source SHA-256 was
`1811e9ca931ea0519839f3fc2553fc2e5161434bcdb9bdcda21982fd6dee2271`
on Iter38 base `ccb16f32f`. `make -C c_impl -j8`, the fast exact decode gate,
and the full 32-token gate passed; the full gate reported exact trajectory,
100% top-1 agreement, and first divergence `-1`. Integrated synthesis used
Vitis HLS 2022.2 and
`c_impl/diagnostics/iter39a_headserial_qkvg/csynth.tcl`; report SHA-256 was
`2e953e303a31e154ec2fe0323c57ffda23ae083f5596aaa1aab3e63c20e6ce1e`.
An initial invocation through the 2022.1 executable stopped during front-end
analysis because that older path did not expand macro-valued HLS pragma
factors; it generated no design result and the production 2022.2 run replaced
it.

**Synthesis result versus Iter38.** The schedule is deliberately unchanged:
2,744,183 minimum cycles, 8,435-cycle minimum shared-GEMV call, 514-cycle QKVG
unpack, all 32 MM2S loops at II=1, and all 16 cluster MAC loops at II=4.
Estimated Fmax remains 167.98 MHz. Resources are 1,527 RAMB18, 3,629 DSP,
857,342 FF, 929,926 LUT, and 48 URAM (only -1 FF/-15 LUT versus Iter38).

**Verdict: neutral enabling prerequisite, not independently committed.** It
does not lower the current serial schedule and therefore is not a positive
iteration under the commit discipline. Keep it only in the working tree for
Iter39B, whose bounded QKVG-result consumer must overlap each completed head's
three depthwise convolutions with production of later heads. If that overlap
does not yield a material integrated cycle reduction, revert both Iter39A and
Iter39B to the committed Iter38 baseline.

### iter39B — first head-local QKVG/convolution overlap (rejected csynth)

*Logged: 2026-08-04.*

**Hypothesis.** Consume each complete 256-element Q/K/V/gate head directly
from the head-serial QKVG collector, persist the gate, and run one time-shared
head-local depthwise-convolution actor for Q, K, and V while the GEMV engine
produces subsequent heads. The actor should replace three serial whole-hidden
convolutions without triplicating their arithmetic.

**Direct-AXI subvariant.** The first implementation passed convolution weights
and tails on `mem_weights_mm0` directly into the GEMV result sink. Vitis HLS
rejected it before scheduling because the dataflow region then had two reader
processes on one bundled master: `gemv32_load_x_and_w0` and the result sink.
This confirms that the shared AXI interface must remain outside the GEMV
dataflow region.

**Preloaded-context subvariant.** The second implementation staged the three
convolution weight arrays and tails into partitioned BRAM before QKVG, invoked
one head-local convolution call site three times per completed head, and wrote
the staged tails back afterward. Native compilation, the fast exact gate, and
the full 32-token exact gate passed with first divergence `-1`. Integrated
synthesis used Vitis HLS 2022.2 and
`c_impl/diagnostics/iter39b_qkvg_conv_overlap/csynth.tcl`. Working source
SHA-256 was
`3abaca74ba96350f3972ab2e206c90fe3a285ea7bdb4bd53023ee016f4e0c767`;
report SHA-256 was
`ff0d1ab6a78a38a867e9ebc6b3d46306da6ff66458863147ee5e4920812b06dc`.

**Synthesis result versus Iter38.** Arithmetic sharing succeeded: the report
contains one `gdn_depthwise_conv_silu_head_kind`, and the GEMV engine retained
its 8,435-cycle minimum, 32 MM2S loops at II=1, and 16 cluster loops at II=4.
However, the runtime `kind` pointer selection caused HLS to scalarize every
`Pack16` context transfer into 16 narrow AXI/BRAM operations. The context load
took 43,459 cycles per layer and the store 18,655, with the critical loops at
II=16. Top minimum latency regressed from 2,744,183 to **3,653,039 cycles**
(+908,856), and layer minimum latency rose from 113,854 to 151,723. Resources
rose from 1,527 to **1,591 RAMB18**, 857,343 to **886,566 FF**, and 929,941 to
**1,041,897 LUT**; DSP stayed at 3,629 and URAM at 48. Estimated Fmax remained
167.98 MHz.

**Verdict: rejected negative implementation; no hardware build and no
commit.** The overlap structure remains plausible, but this generic context
mover destroys both schedule and area. The next bounded subvariant must use
explicit fixed-bank 512-bit load/store loops, keep the staged tail read-only
during head convolution, and capture only each head's new raw Q/K/V row for a
packed final writeback. Iter39A remains an uncommitted prerequisite only while
that corrected subvariant is evaluated.

### iter39C — packed context and head-local QKVG/convolution overlap (retained)

*Logged: 2026-08-04.*

**Change.** The corrected overlap keeps the one time-shared 256-column
convolution actor from Iter39B but removes every runtime-selected external
pointer from the context movers. Six explicit Q/K/V loops load convolution
weights and old tails as 512-bit `Pack16` bursts. The head actor treats the old
tail as read-only; after all three Q/K/V calls consume a head, the result sink
reuses obsolete tail row 0 to capture that head's new raw Q/K/V row. Three
explicit packed stores finally emit old rows 1/2 followed by the captured new
row. This needs no extra tail buffer and preserves the exact recurrent-tail
ABI and FP32 operation order.

**Identity and validation.** Working source SHA-256 is
`d2674931f90897d932fc73915981866e38a233cb6efa4138caaef2029f3d5bbb`
on Iter38 base `ccb16f32f`. `make -C c_impl -j8`, the fast exact gate, and the
full 32-token exact gate passed; both parity reports had first divergence
`-1`. Integrated synthesis used Vitis HLS 2022.2 and
`c_impl/diagnostics/iter39c_packed_qkvg_context/csynth.tcl`; report SHA-256 is
`5c35c51f3e32c3ffafb1ebc4059f69a5149c8786b9618cf31959092fd183b7f8`.

**Synthesis result versus Iter38.** HLS inferred 512-bit bursts for all six
context-read loops and all three tail-write loops. The load takes 3,137 cycles
per layer and the store 1,371, with every external mover loop at II=1. QKVG
retains one physical `gdn_depthwise_conv_silu_head_kind`; its compute loop is
II=1, and its 16-cycle new-tail capture is II=1. The 32 GEMV MM2S readers
remain II=1, all 16 cluster MAC loops remain II=4, and the shared GEMV minimum
remains 8,435 cycles.

| Integrated metric | Iter38 | Iter39C | Delta |
|---|---:|---:|---:|
| Top minimum cycles | 2,744,183 | **2,270,495** | **-473,688 / -17.27%** |
| Layer minimum cycles | 113,854 | **94,117** | **-19,737** |
| RAMB18 | 1,527 | **1,543** | +16 / +1.05% |
| DSP | 3,629 | **3,629** | 0 |
| FF | 857,343 | **863,589** | +6,246 / +0.73% |
| LUT | 929,941 | **937,707** | +7,766 / +0.84% |
| URAM | 48 | **48** | 0 |

Estimated Fmax remains 167.98 MHz. Unlike the unified-call accounting in
Iter38, this 473,688-cycle reduction is exactly 24 times the fixed per-layer
reduction and represents removed serial convolution work.

**Hardware implementation.** `make -C c_impl iter39` compiled at 130 MHz and
linked only at 100 MHz using the exact Iter38 connectivity, cluster-8 and
recurrent-SLR2 floorplans, DMA fanout repair, BRAM FIFOs, and pre/post-route
`AggressiveExplore`. The build ran from 12:50:04 to 21:09:07 +03. The link
reported 7 h 56 m 30 s and exited zero. Artifact hashes are:

- XO: `53a47efc2098f6967fd256edce29aedfbe4dfe4da0383f609bc8b006e73131c0`;
- XCLBIN: `5c81e79ceb51d3faa00a4ae80a5055f716bbedac884700840011257494f11021`.

Routing completed with zero failed nets, zero unrouted nets, zero partially
routed nets, and zero node overlaps. Timing closed without automatic clock
scaling: design WNS/TNS **+0.003/0 ns**, design WHS/THS **+0.009/0 ns**,
kernel-clock WNS **+0.289 ns**, and fixed 250 MHz DMA WNS **+0.003 ns**.

**On-card result.** `make -C c_impl iter39-oncard` passed both the 8-token
smoke and exact 64-token decode with first divergence `-1` and 100% top-1
agreement. Excluding the seed, 63 calls measured:

| On-card metric | Iter38 | Iter39C | Change |
|---|---:|---:|---:|
| Minimum | 47.066309 ms | **43.080265 ms** | -- |
| Maximum | 47.109197 ms | **43.313968 ms** | -- |
| Median | 47.076699 ms | **43.085956 ms** | -- |
| Mean | 47.079335 ms | **43.093000 ms** | **1.0925x / -8.47%** |
| Mean cycles at 100 MHz | 4.707934M | **4.309300M** | **-0.398634M** |
| Speedup over 121.4 ms | 2.579x | **2.817x** | -- |

The measured 0.399M-cycle gain captures 84.2% of the 0.474M static prediction;
the difference is dynamic HBM/control stall outside the HLS minimum.

**Verdict: retained positive result and new production baseline.** Iter39C is
exact, routable, timing-closed, and materially faster than Iter38. Retain the
head-serial/all-port QKVG layout, packed fixed-bank context movers, shared
head-local Q/K/V convolution actor, and one-line Iter39 build/on-card targets.

### iter40A — direct head-streamed recurrence (rejected by dataflow checking)

*Logged: 2026-08-04.*

**Hypothesis.** Extend Iter39C's bounded QKVG consumer so each convolved Q/K/V
head streams directly into the existing recurrent-attention arithmetic. This
removes full-layer Q/K/V BRAM materialization and should overlap the eight
serial 27.3--27.5K-cycle recurrent calls with later QKVG head production.

**Change and validation.** The first implementation added bounded Q/K/V
streams between the existing head-local convolution actor and a single
head-serial recurrent actor. The latter read the four packed state stripes
directly from weight/state ports 28--31. Native compilation, the fast exact
decode gate, and the full 32-token exact gate passed. Integrated synthesis used
Vitis HLS 2022.2 and
`c_impl/diagnostics/iter40a_headstream_recurrent/csynth.tcl`.

**Synthesis failure.** Vitis HLS stopped during dataflow validation with
`HLS 200-1013`/`HLS 200-984` on each of `mem_weights_mm28` through
`mem_weights_mm31`: the existing GEMV MM2S actor and the new recurrent actor
both read the same bundled AXI master. No latency, resource, implementation, or
on-card result exists.

**Verdict: rejected structural subvariant; no hardware build and no commit.**
The bounded Q/K/V stream is correct, but every shared weight/state AXI bundle
must have one read owner. Iter40B retains the stream edge and replaces the two
readers with one owner per port.

### iter40B — single-owner state prefetch and head-streamed recurrence (rejected at route)

*Logged: 2026-08-04.*

**Change.** Ports 28--31 now use one `gemv32_mm2s_with_state` actor each. For a
normal GEMV the actor emits the unchanged weight stream. For QKVG it emits one
head's 4,096 weight packs and then prefetches that head's 1,024 packed state
words into a bounded URAM FIFO before advancing to the next head. The
head-local convolution actor emits Q/K/V packs directly to a single recurrent
actor, which preserves the original scalar and FP32 reduction order, consumes
the four state streams concurrently, writes updated state to the same four
ports, and writes attention output directly to the resident activation buffer.
The Q/K/V whole-layer buffers and standalone recurrent call are removed. Tiny
A/B projections and the eight layer gate scalars are staged before entering
the bounded dataflow graph. The external ABI, weight bytes, 32 masters,
16-cluster GEMV arithmetic, and state representation are unchanged.

**Identity and validation.** Working `gdn_model.cpp` SHA-256 is
`8a2eccae41599d6bfc0fbb311f020d394833b7ad240bf3f2e37026bb0820cc5b`.
Integrated synthesis used Vitis HLS 2022.2 and
`c_impl/diagnostics/iter40b_state_owner_stream/csynth.tcl` (SHA-256
`e898e05ced5cd59f094fc9c9df9ec033c81f6ef92bc4d38d74d95bafea6a712c`).
The top report SHA-256 is
`0e9165218f45e9db7d0fa520b4de31cd40d042967eb167143a15629785786336`.
Native compilation and the fast exact gate passed. The full native gate then
matched all 32 compared token positions with first divergence `-1` and 100%
top-1 agreement.

**Synthesis result versus Iter39C.** All four state-owner weight and prefetch
loops run at II=1, all 32 effective MM2S paths remain II=1, all 16 cluster MAC
loops remain II=4, and recurrent packed state read/write remain II=1. The
recurrent actor is 28,857 cycles for eight heads and is now inside the QKVG
dataflow graph. Estimated Fmax remains 167.98 MHz.

| Integrated metric | Iter39C | Iter40B | Delta |
|---|---:|---:|---:|
| Top minimum cycles | 2,270,495 | **1,939,866** | **-330,629 / -14.56%** |
| Layer minimum cycles | 94,117 | **80,204** | **-13,913** |
| Shared GEMV minimum | 8,435 | 11,718 | +3,283 |
| RAMB18 | 1,543 | 1,572 | +29 / +1.88% |
| DSP | 3,629 | 3,716 | +87 / +2.40% |
| FF | 863,589 | 889,277 | +25,688 / +2.97% |
| LUT | 937,707 | 959,323 | +21,616 / +2.31% |
| URAM | 48 | 80 | +32 |

The minimum-cycle saving clears the 0.25M integrated gate and projects the
4.309M-cycle Iter39C hardware result to about 3.98M cycles/token. The increase
in shared-GEMV minimum is real accounting/flush overhead from the conditional
bounded dataflow graph and is already included in the top saving. The four
1,024x512-bit state FIFOs account for the 32 extra URAMs. HLS also reports a
roughly 28K-fanout control cone in the recurrent read pipeline, so routing and
timing remain explicit acceptance gates rather than assumed consequences of
the positive schedule.

**Verdict: rejected after 100 MHz routing failure; no commit.** The positive
native/csynth schedule did not survive physical implementation. Iter39C remains
the production baseline and no Iter40 source/config change is committable.

**Hardware launch.** The backward-compatible recurrent-SLR2 hook now accepts
either the old standalone recurrent root or Iter40's nested
`gdn_recurrent_attention_stream_U0`, while still requiring exactly one match;
its SHA-256 is
`f00964b6e9455b35ece5fb4927f8d4fb78ddb0e24d7a33e34ad3482ef68358ce`.
The source/config-only build was launched with
`make -C c_impl run_hw RUN_HW_DIR=diagnostics/iter40b_state_owner_stream/hardware`
at 130 MHz HLS / 100 MHz link. PID, persistent wrapper output, exit marker, and
source/config hashes are under
`c_impl/diagnostics/iter40b_state_owner_stream/hardware/`. The XO compile
completed successfully in 23m49s; implementation and automatic on-card gates
were then attempted.

**Hardware result.** The link ran for 6h58m23s and failed in `route_design`.
Both the Iter22 cluster-8 placement and the updated recurrent-SLR2 hook applied
successfully; the latter matched exactly the nested streamed recurrent root.
Placement completed, but post-placement physical optimization still estimated
WNS/TNS at -1.503/-142.164 ns. Initial routing reached global and timing
congestion level 7. Peak directional demand was 103.731% north, 108.89% south,
and 118.105% east. Localized SLL demand exceeded one column's capacity on both
boundaries: SLR0--1 peaked at 127% and SLR1--2 at 120%. Route verification then
reported partially conflicted nets spanning GEMV clusters 3/5, resident helper
loops, and several streamed recurrent control cones. The error checkpoint is
`build.hw.gdn32.h130.f100.o8/_x_temp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed_error.dcp`.

The synthesized kernel used 571,832 LUTs, 615,340 registers, 1,301 BRAM tiles,
3,721 DSPs, and 80 URAMs. Relative to Iter39C, the 32 new whole-head state FIFO
URAMs and the recurrent actor's placement inside the GEMV dataflow hierarchy
created substantial new control and URAM-to-DSP routing pressure. No XCLBIN was
produced and the automatic on-card smoke/performance tests did not run. A
follow-up must first reduce or physically localize the state-buffer/control
cone; retrying the same netlist or changing frequency does not address this
level-7 routing failure.

The post-failure `report_route_status` run on the 2022.2 error checkpoint
confirmed that this was not a timing-only exit or a small set of ordinary
unrouted nets. After the failed recovery, 89 logical nets retained routing
errors, 45 retained resource conflicts, and the discarded/partial route left
1,493,300 routable nets unrouted. The affected set included global constants
and shell reset/control loads as well as the kernel conflicts printed by
`route_design`, which is characteristic of a device-wide routing collapse once
the local level-7 windows can no longer be escaped.

### iter40C — shallow BRAM state drain before recurrent MAC (rejected at route)

*Logged: 2026-08-05.*

**Failure analysis and hypothesis.** Iter39C also began routing with locally
over-capacity SLL columns (up to 145%) but recovered from global congestion
level 6 and routed completely. Iter40B instead entered global level 7, with
post-place WNS already degraded from Iter39C's -0.006 ns to -2.028 ns before
physical optimization. The new whole-head state streams were not merely four
additional memories: because `fused_rd0123` performed four blocking FIFO reads
inside its 32-lane state MAC, HLS converted that dense loop to a free-running
pipeline and estimated a 28,294-load control cone. Four 1,024x512 queues also
added 32 URAMs. Those are the actionable deltas; changing only the link clock
or route directive would leave the failed netlist intact.

**Change.** The four single-owner MM2S actors and the head-streamed Q/K/V edge
are retained. Each recurrent head now immediately drains the four state
streams in a dedicated 1,024-cycle II=1 copy loop into `state_pair`, the same
32-bank head-local URAM buffer already required by the recurrence. The dense
`fused_rd0123` loop subsequently reads only that local buffer, so its 32 MAC
lanes no longer carry FIFO-empty/backpressure control. The external queues are
reduced from depth 1,024 URAM to depth 64 BRAM. They absorb collector/convolution
skew, then naturally backpressure until the state-drain loop starts; no second
whole-head buffer, AXI owner, state-layout change, or FP32 reordering is added.

**Identity and correctness.** Working `gdn_model.cpp` SHA-256 is
`7c7b3d7a3225da4396171981846cf999e2eb1c4eb937fc4e9dde67ba8d396632`.
Integrated synthesis used
`c_impl/diagnostics/iter40c_local_state_drain/csynth.tcl` (SHA-256
`98f071d619b049c5713fad1d5420e6dfaaf406b9ad6e629e519d56c0addb4ab0`);
the top report SHA-256 is
`89e0d33a859dd3df542a36d4bbab1dfe16fdb8e9ed55bdbeb6ef1102678b221b`.
Native compilation, the fast gate, and the full 32-token gate passed. The full
gate compared all 32 positions exactly, with first divergence `-1` and 100%
top-1 agreement.

**Synthesis result.** All four state-owner weight/state loops remain II=1, all
16 GEMV cluster MAC loops remain II=4, and the new state drain, local fused
read, and fused write are II=1. The 28,294-fanout free-running
`fused_rd0123` message is absent; HLS emits ordinary unified pipeline control
for the local read. Estimated Fmax remains 167.98 MHz.

| Integrated metric | Iter39C | Iter40B | Iter40C | Iter40C vs Iter40B |
|---|---:|---:|---:|---:|
| Top minimum cycles | 2,270,495 | 1,939,866 | **1,621,415** | **-318,451 / -16.42%** |
| Layer minimum cycles | 94,117 | 80,204 | **67,072** | **-13,132** |
| Shared GEMV minimum | 8,435 | 11,718 | **8,435** | **-3,283** |
| Recurrent actor maximum | 27,537 | 28,857 | 32,721 | +3,864 |
| RAMB18 | 1,543 | 1,572 | **1,632** | +60 |
| DSP | 3,629 | 3,716 | **3,716** | 0 |
| FF | 863,589 | 889,277 | **888,696** | -581 |
| LUT | 937,707 | 959,323 | **959,673** | +350 |
| URAM | 48 | 80 | **48** | **-32** |

The explicit local copy lengthens the standalone recurrent actor, but it is
hidden behind the state-owner/weight pipeline in the bounded graph; the top
schedule therefore improves rather than regresses. Relative to Iter39C, the
static saving is 649,080 cycles/token (28.59%). BRAM cost rises because a
512-bit FIFO is width-dominated, but BRAM is distributed across the SLR whereas
the removed URAM queues and their stream-controlled MAC cone were column-local
routing pressure.

**Pre-build verdict: accepted as a hardware candidate, not yet retained.** It
clears native correctness and static schedule gates and directly removes both
identified Iter40B routing structures. It remains uncommittable until the
100 MHz implementation routes, closes timing, and improves exact on-card
latency. The hardware result will be appended here.

**Hardware launch.** A clean, iteration-specific build directory was launched
through the single supported Makefile entry:
`make run_hw BUILD_DIR=build.hw.iter40c.local_state_drain.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter40c_local_state_drain/hardware`. HLS targets 130 MHz
and link targets 100 MHz. The build retains Iter39C's connectivity, Iter22
cluster-8 placement, recurrent-SLR2 placement, DMA fanout hook, BRAM GEMV
FIFOs, and pre/post-route `AggressiveExplore`. The detached tmux session is
`gdn_iter40c_build`, root PID is `2546112`, and a separate completion monitor
will normalize the exit marker after automatic exact 8/64-token on-card gates.
Source/config hashes and the start timestamp are in
`diagnostics/iter40c_local_state_drain/hardware/build.manifest`.

**Hardware result: REJECTED AT ROUTE.** The build ran 09:33--16:58 on
2026-08-05 and exited `run_hw.exit=1` with no XCLBIN. `place_design` completed
legally (design state Fully Placed, 0 node overlaps), then `route_design` failed
in global routing:

```
ERROR: [Route 35-3] Design is not routable as its global congestion level is 7.
ERROR: [VPL 18-1000] Routing results verification failed due to
                     partially-conflicted nets
```

The source change did what it claimed: HLS emitted no 28,294-fanout
free-running `fused_rd0123`, URAM returned to 48, and post-place estimated WNS
recovered from Iter40B's **-1.503 ns to -0.412 ns**. It failed for a reason
unrelated to the schedule.

**Root cause: the recurrent SLR2 placement constraint silently stopped
binding.** Measured on the routed-error checkpoint versus Iter39C's routed
checkpoint:

| Metric | Iter39C (routed) | Iter40C (failed) |
|---|---:|---:|
| Recurrent actor leaves in SLR0 / SLR1 / SLR2 | 1,151 / 608 / **220,117** | **238,859** / 514 / 2 |
| SLR0 CLB sites | 99.31% | 99.77% |
| SLR0 CLB LUTs | 281,679 (**64.06%**) | 352,952 (**80.27%**) |
| SLR2 CLB LUTs | 187,200 | 110,395 |
| Recurrent URAM | 48 in SLR2 | **32 in SLR0** |
| Global congestion windows | 16x16 / 64x64 / 8x8 | **128x128 N, S and W** |

Iter39C's hook moved `gdn_forward_1/inst/grp_gdn_recurrent_attention_fu_1833`,
a top-level sibling of `gdn_gemv`. Iter40 nests the actor as
`grp_gdn_gemv_fu_1407/gdn_recurrent_attention_stream_U0`, inside a dataflow
graph anchored to HBM at the south edge of SLR0. The soft
`USER_SLR_ASSIGNMENT SLR2` lost to that gravity: 238,859 leaves, roughly 30% of
everything in SLR0, landed there on top of GEMV clusters 10--15. SLR0 CLB
sites are ~99% occupied in both designs, so the discriminator is **LUT density
inside those sites, 64% versus 80%**, which exhausts local interconnect and
produces device-wide rather than localized congestion. The hook still reports
`GDN_RECURRENT_SLR2 ... slr=SLR2`; it sets the property successfully and the
placer overrides it, so the marker is not evidence that placement obeyed.

**Hypotheses tested and eliminated.** Each was measured, not assumed:

- *SLR-crossing congestion.* Router SLL demand per Laguna column: Iter39C
  needed 19,510 SLLs with **six** SLR0-1 columns over capacity, peaking at
  **145%**; Iter40C needed 18,801 with five columns peaking at 125%. The design
  that routed had the worse crossing profile. An Iter22-style column-steering
  fix is not indicated.
- *High control fanout.* Nets above 1,000 pins: Iter39C **29,858**, Iter40C
  **27,956**; in both, the largest are `ap_clk`/`WCLK` global clock nets at
  ~694k pins. The failing design has fewer. Control-cone replication
  (roadmap 10.4) is not indicated by this evidence.
- *Relocation cost.* The recurrent actor's hierarchical boundary is **7,864
  nets in Iter40C versus 13,132 in Iter39C**, and in both designs essentially
  none of them cross an SLR (1 and 4 respectively). The actor is monolithic:
  >99.7% of its leaves sit in one SLR and its neighbourhood follows it. It is
  therefore *cheaper* to relocate than the block that relocated successfully.
- *`[Place 30-1239]` "failed to find partition obeying SLR constraint".*
  Present exactly once in Iter39C, Iter40B and Iter40C. Not a discriminator.

**Verdict: negative implementation result, not committable.** Iter39C remains
production at 43.093 ms/token. The Iter40C source
(`7c7b3d7a3225da4396171981846cf999e2eb1c4eb937fc4e9dde67ba8d396632`) stays in
the working tree unlanded. Its schedule and correctness evidence remain valid
and are reused unchanged by Iter41A, which changes only the physical recipe.

### iter41A — bind the recurrent actor to SLR2 with a hard pblock (rejected at route; refutes the placement hypothesis)

*Tested 2026-08-05--06; placement gate passed, routing failed at congestion 7.*

**Hypothesis.** Iter40C fails only because a 238,859-leaf block is placed in the
wrong SLR. If the constraint is made binding, SLR0 LUT density falls from 80%
toward Iter39C's 64% and the design routes, retaining Iter40C's measured
schedule gain.

**Change: physical recipe only; source and XO frozen.** The verified Iter40C
`.xo` is reused, so any outcome is attributable to placement alone.

1. `apply_iter37c_recurrent_slr2.tcl` is replaced by
   `apply_iter41_recurrent_slr2_pblock.tcl`. It still sources the Iter22
   cluster-8/`ws16`/`ws17`/`xr8` hook and its `CELL_BLOAT_FACTOR` settings, but
   constrains the recurrent actor with a **pblock covering all of SLR2**
   (clock regions `X0Y8:X7Y11`, queried from the part, not assumed) with
   `CONTAIN_ROUTING 0`, plus the original soft assignment. A pblock is a hard
   constraint the placer must satisfy; `USER_SLR_ASSIGNMENT` alone is advisory
   and demonstrably lost. This is one hierarchy contained to a full SLR, which
   is materially different from Iter17's rejected 7/7/2 sub-SLR floorplan.
2. A new `PLACE_DESIGN.TCL.POST` gate,
   `check_iter41_recurrent_placement.tcl`, counts recurrent leaves per SLR and
   **errors before routing** if fewer than 80% are in SLR2, and reports SLR0 LUT
   density. Iter40B and Iter40C each spent ~7 h to discover a placement problem
   that was already decided at ~2 h; this converts that into a ~2 h answer.

Everything else is byte-identical to Iter40C: `SSI_SpreadSLLs`,
`AlternateCLBRouting`, `AggressiveExplore` at both physical-optimization
stages, the Iter35 DMA fanout pre-place hook, Iter23 fanout, 130 MHz HLS and
100 MHz link.

**Capacity check.** SLR2 has ~54,000 CLB sites and Iter40C uses 25,859
(47.89%). At Iter40C's SLR0 density (~14.4 leaves/CLB), 238,859 recurrent
leaves need ~16,600 CLBs, projecting SLR2 to ~79% and SLR0 down to ~70%.
Iter39C ran with 220,117 recurrent leaves in SLR2 at 72.17%. URAM in SLR2 is
16/320 used and DSP 22%, so the actor's 32 URAMs and its DSPs are not
constraints. The SLR1-2 SLL boundary is the least used resource in every build
(39.5--42.9%), and it is the boundary any new crossings would consume.

**Acceptance gates, in order.** (1) >=80% of recurrent leaves in SLR2 and SLR0
LUT <=75% at post-place, else abort; (2) route with 0 failed, 0 unrouted, 0
overlaps and congestion <=6; (3) kernel and `dma_ip_axi_aclk_1` WNS >=0 at
100 MHz; (4) exact 8- and 64-token on-card trajectories; (5) mean below
43.093 ms/token, else neutral and not committable.

**If the placement gate fails**, the pblock is not binding either and the next
step is Iter41B: keep Iter40C's local state drain and BRAM queues but restore
the recurrent actor as a top-level sibling of `gdn_gemv`, recreating the
Iter39C structure that placed correctly. That costs a resynthesis.

**Hook validation before the build.** Both hooks were sourced against the
Iter40C routed-error checkpoint first. This caught a real defect:
`resize_pblock` rejects clock-region objects with
`[Place 30-342] pblock resize has invalid range X0Y8` and requires a
`CLOCKREGION_X0Y8:CLOCKREGION_X7Y11` range string. Uncaught, that would have
killed the run ~2 h in at `opt_design`. The corrected range derivation was
re-verified on a bare part in ~1 min. The placement gate was validated in the
same pass by confirming it **aborts** on Iter40C's known-bad placement, quoting
`SLR0=238859 SLR1=514 SLR2=2`, which is a third independent confirmation of the
Iter40C diagnosis.

**Result: the placement fix worked completely, and the design still does not
route.** Ran 19:41--00:07 (4 h 26 m), `run_hw.exit=2`, no XCLBIN.

The pblock bound perfectly and the post-place gate passed:

```
GDN_ITER41_PBLOCK    requested=CLOCKREGION_X0Y8:CLOCKREGION_X7Y11
                     ranges=CLOCKREGION_X0Y8:CLOCKREGION_X7Y11
GDN_ITER41_PLACEMENT SLR0=0 SLR1=0 SLR2=239362 total=239362 slr2_frac=1.0000
GDN_ITER41_GATE_PASS proceeding_to_route
```

Every macroscopic physical metric then converged on the routable Iter39C
baseline:

| Metric | Iter39C (routed) | Iter40C (failed) | **Iter41A (failed)** |
|---|---:|---:|---:|
| Recurrent leaves in SLR2 | 220,117 (99.2%) | 2 (0.0%) | **239,362 (100%)** |
| SLR0 CLB LUTs | 281,679 (64.06%) | 352,952 (80.27%) | **294,091 (66.89%)** |
| SLR2 CLB LUTs | 187,200 (43.33%) | 110,395 (25.55%) | **197,035 (45.61%)** |
| URAM placement | 48 all in SLR2 | 32 in SLR0 | **48 all in SLR2** |
| SLR1-0 SLL | 19,463 (84.47%) | 18,954 (82.27%) | 20,399 (88.54%) |
| Post-place estimated WNS | **+0.003 ns** | -0.412 ns | **+0.003 ns** |

Placement, memory placement and post-place timing are now indistinguishable
from the run that routed -- WNS matches Iter39C to the picosecond. Iter41A also
progressed further than either Iter40 build, reaching `Phase 4 Initial Routing
Verification` where Iter40B and Iter40C both stopped inside `Phase 3 Initial
Routing`. It nevertheless ended at global/short **congestion level 7 (128x128)**
and timing congestion level 7, and `route_design` failed verification on
partially-conflicted nets.

**This is the decisive experiment, and it refutes the placement hypothesis.**
SLR0 over-density was real, was fully corrected, and was not sufficient. The
conflicted nets are unchanged in class and now span both ends of the die:
`flow_control_loop_pipe_sequential_init_U/ap_loop_init_int` and `ap_done_cache`
inside `gdn_recurrent_attention_stream_U0` (SLR2), the top-level
`grp_gdn_forward_Pipeline_copy_local*` loops, and
`proc_sys_reset_kernel_slr0/U0/peripheral_aresetn_BUFG[0]`.

The mechanism is now fully explained by the measured boundary data. Iter39C's
recurrent actor is **buffer-coupled** to top-level peers that are free to
migrate, so the whole island moved to SLR2 together and only **4** of its 13,132
boundary nets crossed an SLR. Iter40's actor is **stream-coupled inside
`gdn_gemv`**, whose MM2S readers are pinned to HBM at the south edge of SLR0 and
cannot follow it. Both available placements therefore lose: leaving it south
packs SLR0 to 80% LUT density, and forcing it north converts its 7,864 boundary
nets into SLR crossings that concentrate into a few Laguna columns -- the
Iter41A router reported a single column at **169%** on SLR0-1 and **182%** on
SLR1-2, worse than any prior build.

**Verdict: negative implementation result, not committable.** Iter39C remains
production at 43.093 ms/token. Placement is exhausted as a lever for the nested
topology: two opposite placements have now been built and both fail at
congestion level 7. The remaining fix is structural, not physical -- Iter41B
must un-nest the recurrent actor back to a top-level sibling of `gdn_gemv`,
retaining Iter40C's local state drain and BRAM queues, so the actor is once
again buffer-coupled to peers that can migrate with it. Do not spend another
physical-recipe iteration on this netlist.

### iter42 — reduce GEMV weight-stream depth from 64 to 32 (rejected neutral synthesis)

*Tested: 2026-08-06.*

**Hypothesis.** The Iter41 routing failures might be relieved by halving all
32 `ws` queues while retaining BRAM decoupling. The experiment changed only
the `ws` depth from 64 to 32. Integrated synthesis used Vitis HLS 2022.2 and
`diagnostics/iter42_ws_depth32/csynth.tcl` (SHA-256
`64deb0a06f59f0ec90344b1d3d13728f94c22763a6c5ef82b049e0e29dba277c`);
the report SHA-256 is
`3eec07c65864cf57d8744d69aa238b5b8663775cdde7db8902dbffbf8d4330a3`.

**Result.** HLS still implemented every 512-bit queue as BRAM. Because width,
not depth, determines its RAMB18 footprint in this range, top-level BRAM stayed
at **1,632**. The 1,621,415-cycle minimum, 3,716 DSPs, 48 URAMs and 167.98 MHz
estimate were unchanged. FF fell only 96 (888,696 to 888,600) and LUT only 128
(959,673 to 959,545), far below a meaningful physical change.

**Verdict: rejected neutral; no hardware build and no commit.** Depth 64 was
restored. It provides more decoupling for effectively the same BRAM footprint.

### iter43 — replicate the kernel-reset fanout cone (stopped/inconclusive)

*Tested: 2026-08-06.*

**Hypothesis and identity.** The conflicted Iter41 route contained shell reset
and HLS flow-control nets. This physical-only run reused the frozen Iter40C XO
`d8c706414fa02dac493c8f4199c6db6ac0ea55b01496a11fcf1b00bc502726b4`
and added `MAX_FANOUT_MODE=CLOCK_REGION` plus `FORCE_MAX_FANOUT=512` to the
kernel reset cone. The source/config/Tcl hashes are recorded in
`diagnostics/iter43_reset_fanout/hardware/build.manifest`.

**Result.** The run placed and reached `route_design`/initial timing update,
then the wrapper was externally terminated at 17:39. There is no exit marker,
routed checkpoint, XCLBIN, timing report or on-card result. Therefore this run
cannot establish either benefit or regression.

**Verdict: stopped/inconclusive; no commit.** No reset-fanout change is retained
on this evidence.

### iter44 — force the GEMV reduction fadds into fabric (rejected neutral synthesis)

*Tested: 2026-08-06.*

**Hypothesis.** Move the clustered GEMV reduction tree away from the dense DSP
columns to reduce localized interconnect pressure. Integrated synthesis used
`diagnostics/iter44_fadd_fabric/csynth.tcl` (SHA-256
`eb5597c4dd77de83c30e18422cf487799589ca15c25692533490f7a0e238ad24`);
the report SHA-256 is
`3360e742f41eb60433e81f7d077d54751671fdf1d28f7b706f3a1402c83e11ef`.

**Result.** The top report was byte-for-metric identical to Iter40C:
1,621,415 minimum cycles, 1,632 RAMB18, 3,716 DSP, 888,696 FF, 959,673 LUT,
48 URAM and estimated Fmax 167.98 MHz. The directive did not produce a usable
physical delta in the integrated hierarchy.

**Verdict: rejected neutral; no hardware build and no commit.** The proven DSP
implementation was restored.

### iter45 — reduce AXI read-outstanding capacity to four (routes and times; rejected by on-card deadlock)

*Tested: 2026-08-06--07.*

**Change and identity.** `num_read_outstanding` on weight masters 1--31 was
reduced from 8 to 4; master 0 retained its separately proven setting. The
source SHA-256 is
`9a9582bb7840df6c7689e5c72aa1594948176a3cb27242b0dd8b97453cc67934`.
The 100 MHz physical recipe remained
`hw_iter37c_state4_recur32_slr2_f115.cfg` SHA-256
`998b71e3a8cb3b7f818f12cbe6581f0ffd2e04010dba5db3f20ca2ae844aa08f`.
The integrated report SHA-256 is
`787b6165805d54801647a5ecdcbf5179247dbd3dabbff0ecfe06c27152f6b54f`;
the complete identities and command are in
`diagnostics/iter45_axi_outstanding4/hardware/build.manifest`.

**Synthesis and implementation.** Static latency and compute II were unchanged:
1,621,415 minimum cycles, state-owner read loops II=1 and all 16 cluster MAC
loops II=4. HLS LUT estimate fell from 959,673 to **868,905** while BRAM/DSP/
URAM stayed at 1,632/3,716/48. The 130 MHz HLS / 100 MHz link completed in
9 h 10 m. Routing finished with zero failed, unrouted or partially routed nets
and recovered from global congestion level 6. The design closed timing with
WNS/TNS **+0.003/0 ns**, WHS/THS **+0.008/0 ns**, kernel-clock WNS
**+0.237 ns**, and fixed 250 MHz DMA WNS **+0.003 ns**. Routed kernel use was
509,192 LUT, 27,390 LUTRAM, 613,586 registers, 1,331 BRAM tiles, 48 URAM and
3,721 DSPs.

**On-card failure.** The eight-token gate launched and never completed its
first kernel call. After seven hours the control state remained
`ap_start=1, ap_done=0, ap_idle=0`; the host was blocked in `run.wait()`.
The run was terminated without a token or latency measurement. This is an RTL
liveness failure, not a routing or timing failure.

**Verdict: rejected functional result; no commit.** Iter45 proved the
outstanding-depth reduction is a strong routability lever, but that source is
not usable until the state-owner protocol is made live.

### iter46 — enlarge the four state queues to one complete head (inconclusive RTL liveness experiment)

*Tested: 2026-08-07.*

**Hypothesis.** Iter45's four depth-64 state streams can fill and backpressure
ports 28--31 before the Q/K/V consumer becomes ready. Raising each queue to
1,024 words permits one complete head of state to be buffered. Integrated
synthesis used `diagnostics/iter46_state_fifo_burst/csynth.tcl`; the report
SHA-256 is
`69f842b7f24b90f3715d12b3b910779932a4d519a6e331fe75f888a94d91eaa9`.

**Result.** Native exact gates passed and HLS inferred four
`fifo_w512_d1024_B` BRAM queues. Top minimum latency remained 1,621,415 cycles,
with 1,632 RAMB18, 3,716 DSP, 888,271 FF, 868,961 LUT, 48 URAM and 167.98 MHz
estimated Fmax. A one-layer RTL simulation with depth 64 stopped progressing
near 44K cycles; the depth-1,024 variant advanced beyond 186K cycles. Neither
completed within its 2.5 h / 5 h observation window, so that comparison was a
useful clue but not a liveness proof.

**Verdict: inconclusive; no commit.** The depth-1,024 source advanced to the
separate Iter47 hardware acceptance test.

### iter47 — whole-head state FIFOs with outstanding four (rejected at placement)

*Tested: 2026-08-07.*

**Identity.** Source SHA-256
`fd200e3c552c76a0172d5e3993dfd93be2c827eb8eb1f3e6946cb77146d0c2a1`
combines Iter45 outstanding depth four with Iter46 state depth 1,024. The
unchanged 100 MHz config SHA-256 is
`998b71e3a8cb3b7f818f12cbe6581f0ffd2e04010dba5db3f20ca2ae844aa08f`.
The command and rationale are preserved in
`diagnostics/iter47_state_depth1024/hardware/build.manifest`.

**Result.** The 130 MHz compile succeeded, but 100 MHz implementation failed
after 2 h 32 m in `place_design`: 114 instances remained unplaced, including
43 Laguna/SLL-related cells around the HBM path-12 crossing plus shell
clock-converter registers. No routing, timing, XCLBIN or on-card result exists.

**Verdict: rejected negative implementation; no commit.** The four additional
whole-head memories consume the placement margin recovered by Iter45.

### iter48 — steer cluster 4 away from the exhausted Laguna column (rejected at route)

*Tested: 2026-08-08.*

**Change and identity.** Keep Iter47 source and constrain cluster 4 plus its
`ws_8`/`ws_9` queues into eastern SLR0 clock regions, targeting the measured
HBM path-12 Laguna collision. Config SHA-256 is
`ec279d277285a0944da0aabb9c339f87fdcb0d5569fa7496c41505011e05d48d`;
Tcl SHA-256 is
`d6377be4c3fe227f6b08f561d1d41a128d084b22169b40e5b241bf6d93b58f11`.
The exact command and source identity are in
`diagnostics/iter48_cluster4_sll/hardware/build.manifest`.

**Result.** The targeted placement failure was removed: all cells placed with
zero overlaps and SLR0--1 demand was 18,991/23,040 SLLs. Routing nevertheless
failed at global, short and timing congestion level 7. Southbound demand
peaked at 114.91%, and route verification ended on partially conflicted HLS
control/recurrent nets. No XCLBIN or timing/on-card result exists.

**Measured root cause (per-block SLR histograms, not inference).** The two
checkpoints were opened and every child of the gemv hierarchy was mapped to its
physical SLR with `diagnostics/iter48_cluster4_sll/gemv_children.tcl`
(Iter45 `level0_wrapper_routed.dcp`, Iter48 `level0_wrapper_routed_error.dcp`).
One block explains the failure:

| Block | Leaves | Iter45 (routed, cong 6) | Iter48 (failed, cong 7) |
|---|---:|---|---|
| `gdn_recurrent_attention_stream_U0` | 254,502 | SLR2 **238,300** (94%) | SLR0 **238,062** (94%) |
| `gemv32_mm2s_with_state_28..31_U0` | 4 x ~2,930 | SLR1 ~2,055 each | SLR0 ~2,910 each |
| `gemv32_store_or_qkvg_conv_stream_U0` | 46,662 | SLR2 43,341 (93%) | split 5,304/9,637/28,397 |

The largest block in the design, ~23% of the gemv hierarchy, changed SLR. That
is the entire SLR0 overload: SLR0 CLB LUTs 61.90% -> 80.50%, CLB occupancy
99.78%, and the router's worst window is SOUTH 114.911% over
`INT_X0Y0 -> INT_X127Y79`, i.e. SLR0 in full. The 16 clusters merely permuted
(net-neutral), and total gemv leaves are unchanged (1,113,664 vs 1,113,913).

**This was not caused by the deep state queues' routing footprint.** At
synthesis the two netlists differ by **+78 flops and +28 BRAM tiles**, with
identical DSP (3,725) and URAM (48) - too small to relocate 158k leaves. The
`ws_8`/`ws_9` pin also did not cause it directly: both FIFOs occupied SLR0 in
Iter45 too (68/73 leaves) and were only moved west->east *within* SLR0. The pin
perturbed a placer sitting on a knife edge into a different global solution.

Note that solution bought **nothing** in SLR crossings, so this is not a
crossings-versus-congestion trade. Directional SLL demand from
`slr_util_placed.rpt`:

| Boundary | Iter45 (routed) | Iter48 (failed) |
|---|---:|---:|
| SLR0 -> SLR1 | 13,478 (58.50%) | 14,720 (63.89%) |
| SLR1 -> SLR0 | 6,462 (28.05%) | 4,371 (18.97%) |
| SLR0 <-> SLR1 total | 19,940 | 19,091 |
| SLR1 <-> SLR2 total | 14,319 | 14,004 |

Total crossings are essentially unchanged and the *dominant* direction got
worse. Iter48 therefore paid SLR0 congestion for no crossing saving. (An
earlier reading of "65.90%" for Iter48 came from a non-comparable summary row
and is superseded by these directional figures.)

Iter37C's `USER_SLR_ASSIGNMENT SLR2` is advisory; it printed `slr=SLR2` and the
placer overrode it, the same failure mode Iter41A recorded for Iter40C.

**Consequence for Iter41A's generalization.** Iter41A concluded "placement is
exhausted as a lever for the nested topology ... do not spend another
physical-recipe iteration on this netlist." That holds for the Iter40C netlist,
which never routed under any placement. It does **not** generalize: Iter45
(`num_read_outstanding` 8->4) routed at congestion 6 with 238,300 recurrent
leaves in SLR2 - the configuration Iter41A declared unroutable. The nested actor
demonstrably routes on the current netlist when it is actually held in SLR2.

**Verdict: rejected negative implementation; no commit.** The targeted Laguna
fix worked (0 unplaced vs Iter47's 114) and should be retained; the run failed
for an unrelated reason. Two follow-ups exist. Iter49 (credit protocol) removes
the deep queues entirely and is the preferred path because it keeps the netlist
close to Iter45's. If a build again places the recurrent actor outside SLR2,
the fix is a binding pblock rather than the advisory property: the bundle
`build_iter50_recurrent_hard.sh` / `hw_iter50_recurrent_hard_f100.cfg` /
`apply_iter50_recurrent_hard_ws_east.tcl` is prepared for that, and pairs it
with `check_iter41_recurrent_placement.tcl` as a post-place gate so a
non-binding constraint costs ~5 h instead of ~7.5 h. The frozen depth-1024
source it requires is preserved at
`diagnostics/iter48_cluster4_sll/gdn_model.cpp.d1024`
(SHA-256 `fd200e3c...`), because the working tree has since moved to Iter49.

### iter49 — per-head state credit with shallow queues (rejected at placement)

*Tested: 2026-08-08; hardware build rejected.*

**Fix.** Retain Iter45's routable outstanding depth of four and restore all
four state streams to depth 64. Add one depth-2 Boolean credit stream per
state-owning port. Each owner emits exactly 4,096 contiguous weight packs for
one QKVG head, waits for a credit, then emits that head's 1,024 contiguous
state words. The recurrent actor first captures all 16 Q/K/V packs, returns
the four credits, and immediately drains the state words into its existing
head-local buffer. Thus state can never block delivery of the weights needed
to create the credit, while no full-head FIFO is required. Native-only builds
guard the reverse handshake because C execution serializes dataflow actors;
the synthesized feedback protocol is validated separately at RTL.

**Identity and validation.** Working `gdn_model.cpp` SHA-256 is
`1a3552d0f09c9090891c6b0e96175241dec46744d21e9cc6c704b3266a2e2fa6`.
`make -C c_impl -j8`, the six-position fast exact gate, and the full 32-position
exact gate passed with no divergence. Integrated HLS used
`diagnostics/iter49_state_credit/csynth.tcl` SHA-256
`e5a8348edca3739ba2d3ed3133ccaee228b93ae3fe9152aa6df0716d3631a2c6`;
report SHA-256 is
`162f741bab56baccc401c2461541f6d134d1a949b4afcfbb0f6813c1587706f6`.

**Focused RTL liveness gate.** A bounded four-owner/eight-head protocol harness
used the same 4,096-weight/credit/1,024-state sequence, depth-64 BRAM data
queues and depth-2 SRL credits. C simulation, synthesis and Verilog C/RTL
cosimulation all passed. HLS explicitly reports four backward dataflow
channels and implements `state_credit[0:3]` as `fifo_w1_d2_S`, while all four
state streams are `fifo_w512_d64_B`. The harness source/Tcl hashes are recorded
under `diagnostics/iter49_state_credit_protocol/`.

**Integrated synthesis.** Static latency is unchanged from Iter45/Iter47:
1,621,415 top minimum cycles, 67,072 per layer and 8,435 shared-GEMV minimum.
Every state-owner weight/prefetch loop remains II=1 and all cluster MAC loops
remain II=4. Estimated Fmax is 167.98 MHz. Relative to Iter45, only 32 FF and
114 LUT are added; totals are 1,632 RAMB18, 3,716 DSP, 888,263 FF, 869,019 LUT
and 48 URAM. This restores the routable Iter45 memory footprint while directly
removing its demonstrated circular wait.

**Pre-build verdict: accepted as a 100 MHz hardware candidate, but not
retained.** The hardware command was
`make run_hw BUILD_DIR=build.hw.iter49.credit.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter49_state_credit/hardware HLS_FREQ=130 FREQ=100
LINK_FREQ=100 JOBS=8 HW_DEVICE=0`. The XO compiled, but the detached build
exited 2 at 2026-08-08 17:42:01 +03:00 after 4 h 56 min 55 s total
(4 h 30 min 53 s in link).

**Hardware result: failed placement before routing.** Vivado required 19,036
of 23,040 SLLs across SLR0 to SLR1, classified the design as highly
congested, and failed `place_design` with 70 unplaced instances. Of those, 45
are in `gdn_recurrent_attention_stream`, 15 are other kernel/AXI logic, and
10 are HBM path-31 interconnect registers. The illegal placement's diagnostic
timing degraded from estimated WNS -2.769 ns to post-placement WNS -9.486 ns
and post-replication WNS -8.804 ns; these values are not timing-closure results
because placement never completed. No routed checkpoint, xclbin, or on-card
result was produced.

**Failure diagnosis.** This is not global resource exhaustion: relative to
the routed Iter45 post-link synthesis, Iter49 has only 84 more CLB LUTs
(571,716 versus 571,632), 40 fewer CLB registers (617,142 versus 617,182), and
identical 1,331 BRAM tiles, 48 URAMs and 3,721 DSPs. Nor is aggregate SLL count
the discriminator: Iter45 routed despite a larger placement warning of 19,979
SLLs. Every Iter49 unplaced cell is soft logic (48 FDREs and 22 LUTs), with 45
inside recurrence, 15 in kernel AXI ports 29--31, and 10 in HBM path 31. This
localization, together with the post-placement timing collapse, indicates a
bad coarse partition/local CLB-packing solution for the coupled
recurrence/state-owner cone. The four reverse credit channels add only 32 FF
and 114 LUT at HLS level, but turn that cone into a cyclic dataflow component
and perturb a placer already known from Iter48 to sit on a knife edge. The
`USER_SLR_ASSIGNMENT SLR2` hook matched the recurrent root, but remains
advisory; Vivado has overridden it in prior runs. Because placement never
completed, there is no placed DCP from which to measure an authoritative
per-SLR histogram, so attribution of the bad partition specifically to the
feedback edge is a high-confidence physical inference rather than a measured
SLR mapping.

**Final verdict: rejected and not committable.** The credit protocol passed
native, integrated-synthesis, and focused RTL-liveness gates, but its physical
netlist did not fit the established 100 MHz floorplan. Iter39C remains the
last demonstrated production improvement at 43.093 ms. Do not retry the same
netlist unchanged; retain this result only as a negative experiment in the
working optimization log.

### iter50 — hard-bind the Iter49 recurrent actor to SLR2 (rejected at route)

*Started: 2026-08-09.*

**Hypothesis.** Iter49 failed from a bad coarse partition/local CLB-packing
solution rather than resource growth. Reuse its exact XO and replace the
advisory recurrent `USER_SLR_ASSIGNMENT` with the previously validated full-
SLR2 hard pblock. Preserve Iter22 cluster-8 placement and the DMA fanout hook,
but deliberately omit Iter48's cluster-4/`ws_8`/`ws_9` steering because that
physical perturbation moved the recurrent actor into SLR0. A post-place hook
reports the exact recurrent per-SLR histogram and aborts before route if less
than 80% of its placed leaves are in SLR2.

**Frozen identity and command.** The reused Iter49 XO SHA-256 is
`7b10631b9dae728dc05a9b9fac9f2b851342b7848467910cc2b6f52039a7d270`;
the unchanged source SHA-256 is
`1a3552d0f09c9090891c6b0e96175241dec46744d21e9cc6c704b3266a2e2fa6`.
`make -q xo` returned zero after seeding the new build directory, proving HLS
compilation will be skipped. The physical-only command is `make run_hw
HW_CFG_TEMPLATE=hw_iter41_recur_pblock_f100.cfg
BUILD_DIR=build.hw.iter50.credit.recurhard.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter50_credit_recurrent_hard/hardware HLS_FREQ=130
FREQ=100 LINK_FREQ=100 JOBS=8 HW_DEVICE=0`. Complete source/config/Tcl hashes
are in `diagnostics/iter50_credit_recurrent_hard/hardware/build.manifest`.

**Placement result.** Placement completed and the hard-placement gate passed:
all 239,764 placed recurrent leaves were in SLR2 (`SLR0=0`, `SLR1=0`,
`SLR2=239764`, fraction 1.0000). Post-place WNS was -0.006 ns and pre-route
physical optimization recovered setup to +0.003 ns; the provisional hold
result was WHS -0.248 ns and was not a final timing result. The build therefore
proved that the full-SLR2 pblock enforces the intended recurrent partition,
unlike Iter49's advisory assignment.

**Physical side effect.** The enforced recurrence placement displaced the rest
of GEMV into the wrong SLRs. A read-only checkpoint comparison measured placed
GEMV primitive counts of SLR0/SLR1/SLR2 =
415,186/365,236/333,242 in routed Iter45 versus
464,889/307,837/341,591 in Iter50: SLR0 gained 49,703 primitives while SLR1
lost 57,399. The routed SLR utilization reports show the same redistribution.
Relative to Iter45, Iter50 moved SLR0 from 54,442 to 54,785 occupied CLBs
(99.06% to 99.68%), 272,191 to 284,319 CLB LUTs, 578.5 to 642 BRAM tiles,
and 1,341 to 1,523 DSPs. SLR1 fell from 44,463 to 41,276 CLBs and from
1,170 to 969 DSPs. This is not an aggregate-capacity failure; the hard pblock
perturbed cluster/FIFO placement and packed more hard and soft GEMV resources
into the already saturated SLR0.

The direct-child histogram identifies the dominant mover rather than merely
correlating whole-SLR totals. Cluster 10 had 48,970 placed leaves in SLR1 and
zero in SLR0 in routed Iter45; Iter50 put 46,986 in SLR0 and only 2,001 in
SLR1. Its two weight FIFOs `ws20`/`ws21` and activation FIFO `xr10` moved from
SLR1 to SLR0 with it. Cluster 8 also spilled from 2,001 to 12,943 SLR0 leaves
(SLR1 fell from 47,528 to 36,588). Cluster 0 and cluster 15 largely exchanged
SLR1/SLR2 positions, which changes topology but does not explain the SLR0
density jump. Therefore cluster 10 plus `ws20`, `ws21`, and `xr10` is the
minimal measured relocation cone for the next physical candidate; cluster 8
spill is a secondary gate to monitor rather than grounds for another broad
floorplan.

The route prepass consequently reported severe localized SLL demand. Across
SLR1--SLR2, total demand was only 11,383/23,040 (49.41%), but the worst column
was 2,662/1,440 (185%). Across SLR0--SLR1, total demand was 16,251/23,040
(70.53%), while the worst column was 3,209/1,440 (223%). Vivado emitted
`Route 35-3311` and then `Route 35-447`; estimated global/short and timing
congestion were both level 7.

**Routing result.** `AlternateCLBRouting` spent 4 h 50 min in `route_design`.
The first rip-up pass reduced overlap nodes from 1,209,590 through 514,334,
209,420, 92,786, 43,957, and 24,222, but ended with intermediate WNS
-8.522 ns. A later global iteration reintroduced 944,961 overlaps. Final route
verification reported 44 completely unrouted nets, 20,070 overlapping nodes,
and 26,879 routable nets with resource conflicts. This distinction matters:
the design was not merely 44 isolated nets from success. The conflict report
includes the global-logic-zero distribution, shell/control AXI nets, top-level
FP adders, and recurrent FP adders, confirming device-wide fallout from the
localized hot regions. Vivado wrote
`level0_wrapper_routed_error.dcp`, emitted `Constraints 18-1000`, and exited
the detached wrapper with code 2 at 2026-08-09 09:42:10 +03:00 after about
9 h 41 min total. No xclbin, final timing report, smoke test, or on-card
measurement exists.

**Final verdict: rejected and not committable.** The hard SLR2 pblock fixes
Iter49's coarse recurrent partition but damages the surrounding GEMV placement
enough to create level-7 routing conflicts. Do not retry this full-SLR hard
pblock unchanged and do not treat a route-directive-only change as sufficient:
the next candidate must preserve recurrence in SLR2 while explicitly preventing
the measured 49.7k-primitive GEMV migration into SLR0. Iter39C remains the last
demonstrated production result at 43.093 ms; the Iter49 credit protocol remains
functionally promising but has not produced hardware.

### iter51 — return the measured cluster-10 relocation cone to SLR1 (rejected at placement gate)

*Prepared: 2026-08-09.*

**Hypothesis.** Keep Iter50's effective full-SLR2 recurrent pblock, but correct
the specific GEMV displacement measured in its failed placed checkpoint.
Cluster 10 moved from 48,970 leaves in SLR1 and zero in SLR0 in routed Iter45
to 46,986 leaves in SLR0 and only 2,001 in SLR1 in Iter50. Its `ws20`, `ws21`,
and `xr10` FIFOs moved with it. Assign only those four hierarchy roots to SLR1;
retain the Iter22 cluster-8 constraint and leave every other cluster, FIFO, and
route unconstrained. This should recover most of Iter50's 0.62 percentage-point
SLR0 CLB overfill without recreating the rejected broad cluster floorplans.

**Frozen identity.** The C++ source remains unchanged at SHA-256
`1a3552d0f09c9090891c6b0e96175241dec46744d21e9cc6c704b3266a2e2fa6`.
The build will reuse the exact native-, integrated-csynth-, and focused-RTL-
verified Iter49/Iter50 XO at SHA-256
`7b10631b9dae728dc05a9b9fac9f2b851342b7848467910cc2b6f52039a7d270`;
therefore this is a physical-only experiment and HLS must be skipped. The new
files and SHA-256 identities are `apply_iter51_cluster10_slr1.tcl`
`2a5e0c24deb3238fe4f1d184c1dafe5a7a464b0d2c0c50793ac9df7eb55271e7`,
`check_iter51_placement.tcl`
`b8e00e617a4fce49064b71e96347ad8e7e7cb4448af21b978cd7f005997fd112`,
and `hw_iter51_cluster10_slr1_f100.cfg`
`d08a7633e6ff0db2226a2d74c8c721e9e76059f9a48b88220a877f5c3c6dd508`.

**Early physical gate.** The post-place hook first runs the established
recurrent gate, then requires cluster 10 and each of `ws20`, `ws21`, and
`xr10` to have at least 80% of its placed primitives in SLR1. It also requires
SLR0 occupied-CLB utilization to be at most 99.30%, between routed Iter45's
99.06% and failed Iter50's 99.68%, and reports cluster 8's histogram for spill
diagnosis. Failure aborts before `route_design`; passing placement is not a
success claim.

**Acceptance.** Route must finish with zero failed, unrouted, and overlapping
nets; final WNS and WHS must both be nonnegative at an achieved 100 MHz with
no clock scaling. Only then may the automatic exact 8-token smoke and exact
64-token on-card run execute. A positive performance result must improve the
current production objective; otherwise the constraint remains uncommittable.
The build command and outcome will be added here at launch and completion.

**Prepared command.** `make run_hw
HW_CFG_TEMPLATE=hw_iter51_cluster10_slr1_f100.cfg
BUILD_DIR=build.hw.iter51.credit.c10slr1.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter51_credit_cluster10_slr1/hardware HLS_FREQ=130
FREQ=100 LINK_FREQ=100 JOBS=8 HW_DEVICE=0`. The exact XO was reflinked into
the new build directory and `make -q xo` returned zero, proving that the run
will enter directly at the 100 MHz hardware link. The frozen manifest is
`diagnostics/iter51_credit_cluster10_slr1/hardware/build.manifest`.

**Launch.** An initial 2026-08-09 12:50:13 +03:00 detach inside the transient
sandbox was reaped before `v++` produced output; it performed no synthesis or
implementation and is not a physical attempt. The identical recorded command
was relaunched persistently at 13:31:32 +03:00 under host wrapper PID 1843509.
Its wrapper log, exit marker, timestamps, and PID are under
`diagnostics/iter51_credit_cluster10_slr1/hardware/`; detailed Vivado progress
is in the build directory's `impl_1/runme.log` once implementation starts.

**Placement result.** The persistent run ended with wrapper exit 2 at
2026-08-09 18:21:06 +03:00, after 4 h 49 min. The pre-placement hook proved
that every requested property was present: cluster 10 and `ws20`/`ws21`/
`xr10` all reported `USER_SLR_ASSIGNMENT=SLR1`. Vivado nevertheless overrode
the advisory properties. Post-place, cluster 10 was SLR0/SLR1/SLR2 =
46,986/2,001/0 placed primitives, only 4.08% in SLR1. Each of the three local
FIFOs had all 73 measured primitives in SLR0 and none in SLR1. Cluster 8 was
12,943/36,588/0 (73.87% in SLR1).

The hard recurrent pblock remained fully effective: all 239,764 recurrent
primitives were in SLR2. However, SLR0 stayed at 54,785 occupied CLBs
(99.68%) and 284,316 LUTs, statistically identical to failed Iter50; SLR1 was
41,276 CLBs (76.44%) and SLR2 41,450 (76.76%). Post-place WNS was -0.006 ns.
The congestion estimate remained device-scale (global and short congestion up
to 128x128). The gate correctly aborted before `route_design`, so no route,
xclbin, timing-closure result, smoke test, or on-card measurement exists.

**Final verdict: rejected and not committable.** The measured target cone was
correct, but another soft SLR assignment cannot counter the full-SLR2 recurrent
pblock. The next physical attempt must keep the scope surgical while making
the already-proven Iter45 cluster-10 partition compulsory: a hard pblock over
the full of SLR1 for cluster 10 plus `ws20`, `ws21`, and `xr10`. This has
physical capacity evidence—SLR1 is only 76.44% occupied in the failed
placement and the same cluster occupied SLR1 in routed Iter45—while avoiding a
broad all-cluster floorplan.

### iter52 — hard-pblock the cluster-10 relocation cone in SLR1 (rejected after on-card deadlock)

*Prepared: 2026-08-09.*

**Hypothesis.** Preserve the full-SLR2 recurrent pblock and replace only
Iter51's demonstrably ineffective cluster-10 advisory assignments with a
non-soft pblock spanning all SLR1 clock regions. Add cluster 10 and its three
local FIFO hierarchies to that pblock, leave routing uncontained, and retain
all other physical levers unchanged. This should reproduce the routed Iter45
coarse partition for the measured mover while leaving Vivado freedom within
SLR1 and everywhere else.

**Gates and acceptance.** Reuse the exact Iter49--Iter51 XO and skip HLS. At
post-place, require the recurrent actor to remain at least 80% in SLR2,
cluster 10 and each local FIFO to be at least 95% in SLR1, and SLR0 occupied
CLBs to fall to at most 99.30%; report cluster 8 for secondary spill. Only a
passing placement may enter route. Final acceptance remains zero failed,
unrouted, or overlapping nets, nonnegative WNS/WHS at 100 MHz with no scaling,
then exact 8-token and 64-token on-card runs.

**Frozen identity and command.** Source and XO remain SHA-256
`1a3552d0f09c9090891c6b0e96175241dec46744d21e9cc6c704b3266a2e2fa6`
and `7b10631b9dae728dc05a9b9fac9f2b851342b7848467910cc2b6f52039a7d270`.
The new apply/check/config hashes are respectively
`e8fc4bf76867b1913acc3f210a356fce27538f8447315f90db64525400b02f7b`,
`3db132b7bcb660fe4e5d253532f4d1a22de9573a03e83338f546e803cffc2f12`,
and `f498c1d8158e624f1f072af1c853da9c911d1130fcc73350545a0ba7cc20f2cc`.
Both Tcl files are syntactically complete, the utilization parser matches the
measured report format, and `make -q xo` returned zero after reflinking the XO.
The command is `make run_hw
HW_CFG_TEMPLATE=hw_iter52_cluster10_slr1_pblock_f100.cfg
BUILD_DIR=build.hw.iter52.credit.c10hard.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter52_credit_cluster10_hard/hardware HLS_FREQ=130
FREQ=100 LINK_FREQ=100 JOBS=8 HW_DEVICE=0`; the complete frozen manifest is
under the run directory.

**Launch.** The detached build started at 2026-08-09 19:07:30 +03:00 under
wrapper PID 2309064. The harmless Make warning that the new config timestamp
was 24 seconds ahead of the shell clock does not affect its content or hashes.
The wrapper entered `v++` hardware link using the reused XO.

**Placement and routing result.** The mandatory placement gate passed. The
recurrent actor was entirely in SLR2 (239,762/239,762 leaves); cluster 10 was
entirely in SLR1 (48,998/48,998), and `ws20`, `ws21`, and `xr10` were likewise
entirely in SLR1 (73 leaves each). SLR0/1/2 occupied-CLB counts were
54,237/39,076/44,658, or 98.68%/72.36%/82.70%. Cluster 8 spilled from SLR0
into SLR1 (30,848/18,687 leaves), which was the principal compensating move.
The hard pblock therefore achieved its intended SLR0 density relief.

The SLR0--SLR1 boundary used 16,593/23,040 SLLs (72.02% total), with its
worst physical column at 2,538/1,440 (176%). SLR1--SLR2 used 11,308/23,040
(49.08% total), with its worst column at 2,973/1,440 (206%). Relative to
Iter50, the lower-boundary worst column improved from 223% to 176%, while the
upper-boundary worst column regressed from 185% to 206%. Despite this, the
router converged from 709,476 initial node overlaps to zero. Final route
verification reported zero failed, unrouted, partially routed, or overlapping
nets. This is the first routable credit-protocol implementation and confirms
that the cluster-10 hard relocation solved the Iter50 physical failure.

**Timing and artifact.** The final routed timing report was WNS -0.226 ns,
TNS -4.106 ns over 53 endpoints, and WHS +0.009 ns. Post-route
`AggressiveExplore` improved kernel-clock TNS to -3.724 ns but did not recover
WNS. Vitis consequently auto-scaled the requested 100 MHz DATA clock to
97.7 MHz and encoded 97 MHz in the xclbin. Linking and bitstream generation
completed after 11 h 31 min. The 81,085,994-byte `gdn_forward.xclbin` has
SHA-256
`77182b287ace19765253644e3e94d5756621bc6c8e7e87e4b399f4c270cf2361`.
This misses the no-scaling timing gate but was retained long enough for the
mandatory functional smoke test.

**On-card result.** The automatic 8-token decode loaded the device, xclbin,
weights, 50.3316 MB recurrent-state fixture, and all buffers successfully.
The host reached `run.start()`, which returned, then remained blocked in
`run.wait()` for more than four hours. It never printed the post-wait marker,
produced no smoke JSON, and therefore provided neither parity nor performance.
During the hang, `xbutil examine --report dynamic-regions` showed one use of
`gdn_forward_1` but reported the CU as `IDLE`; the host itself was sleeping at
0% CPU. The host was terminated after preserving these diagnostics, causing
the wrapper to exit 2 (`make` exit 143) at 2026-08-10 11:05:20 +03:00. The
exact last marker is in
`diagnostics/iter52_credit_cluster10_hard/hardware/on_card/oncard_smoke8.log`.

**Deadlock root cause.** The per-head credit boundary is one GEMV input row too
early. Each state-owning MM2S actor writes exactly 4,096 weight packs (32 rows
times 128 input packs) and then waits for the recurrent credit. The clustered
GEMV, however, double-buffers row accumulation: at the end of row `r` it emits
row `r-1`, and only its whole-command epilogue emits the final pending row.
Consequently, after row 31 the cluster has emitted only through row 30. The
second Pack16 for head 0 (rows 16--31) is not written to `ys14`/`ys15` until
row 32 completes, which requires 128 weights from head 1. Ports 28--31 are
already waiting for a credit at that point. The collector therefore cannot
complete head 0, the QKVG/convolution actor cannot emit complete Q/K/V, and
recurrence cannot generate the credit. This is a closed wait cycle:
state-owner credit wait -> cluster weight wait -> collector/result wait ->
recurrent Q/K/V wait -> missing credit.

The earlier integrated depth-128 RTL experiment independently exposed the
same physical channel chain (`ys_14` empty, `ws_29` empty, state owner blocked,
recurrence waiting for Q/K/V); the focused Iter49 protocol harness missed it
because it assumed 4,096 accepted weights immediately produce one complete
head and did not model the cluster's one-row delayed result emission. FIFO
depth and HBM latency can change when the cycle becomes visible, but cannot
make this token graph live.

**Required correction before another build.** Remove the reverse credit FIFOs
and make each state owner supply a one-row lookahead before switching from
weights to state. Head 0 must send 4,224 weights; heads 1--6 then send 4,096
new weights each; head 7 sends the remaining 3,968. This preserves the exact
32,768 total weights per QKVG command and their order. After each boundary the
cluster has all tokens required to emit that head, so a depth-64 state FIFO may
backpressure safely without starving the result that causes recurrence to
drain it. The final head uses the cluster's existing command-end flush. A full
integrated one-layer RTL cosimulation—not the reduced owner-only harness—is a
mandatory liveness gate for this correction.

**Final verdict: rejected and not committable.** The physical intervention is
successful evidence and should be reused, but the synthesized credit-protocol
architecture is not functionally live on hardware. Exact 8/64-token checks and
latency measurement did not run. Before another long link, diagnose the full
32-reader/collector/QKV-convolution/recurrent dataflow cycle with the corrected
one-row-lookahead schedule. Do not treat the routed xclbin as a successful
build.

### iter53 — restore forward-only state owners with whole-head queues (timing failure)

*Prepared: 2026-08-10.*

**Hypothesis and change.** Remove Iter49--Iter52's four reverse credit streams
entirely and return ports 28--31 to Iter45's simple forward-only schedule:
4,096 weight packs followed by 1,024 state packs for each head. Increase only
the four packed state queues from depth 64 to depth 1,024 and bind them to
BRAM. A queue can therefore accept the complete state burst without blocking
its owner. The owner then starts the next head's weights; the first 128 packs
complete the clustered GEMV's one-row-delayed result for the previous head,
which releases Q/K/V and lets recurrence drain the buffered state. This removes
the deadlocked feedback edge while preserving weight order, traffic, GEMV
arithmetic, and the static schedule. Physically, reuse Iter52's successful hard
recurrent-SLR2 and cluster-10-SLR1 floorplan.

**Source and native validation.** The source SHA-256 is
`8c666c47c8f0143ef948bb41aa61d8319114cde71c538b4b3b86867f04c23ca1`.
`make -C c_impl -j8`, the fast six-token exact gate, and the full 32-token
exact gate passed with first divergence -1 and 100% top-1 agreement. No new
AXI master or external ABI change was introduced.

**Integrated synthesis.** Vitis HLS 2022.2 synthesis used
`diagnostics/iter53_state_fifo1024_no_credit/csynth.tcl`; the report SHA-256 is
`9493c999d7d2cb9b5d21bac8622fafb84ff724fdd8973e27d3a118a1023ca197`.
All 16 `gemv32_cl_flat` loops remain II=4. The four state-owner weight loops
remain II=1 for 4,096 words and the state-prefetch loops remain II=1 for 1,024
words. HLS inferred exactly four `fifo_w512_d1024_B` queues using BRAM and no
`state_credit` process or FIFO. Estimated Fmax is 167.98 MHz. The top report is
identical to Iter46: 1,632 RAMB18, 3,716 DSP, 888,271 FF, 868,961 LUT and 48
URAM. Relative to Iter49's credit design, this removes 58 LUT and eight FF in
the HLS estimate while exchanging the shallow queues for complete state-burst
buffers. The earlier one-layer depth-1,024 RTL run advanced beyond 186K cycles
without the depth-64 deadlock signature but timed out before completion, so
on-card completion remains the decisive liveness gate.

**Hardware command and acceptance.** The 130 MHz HLS / 100 MHz link command is
`make run_hw HW_CFG_TEMPLATE=hw_iter52_cluster10_slr1_pblock_f100.cfg
BUILD_DIR=build.hw.iter53.nocredit.state1024.c10hard.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter53_state_fifo1024_no_credit/hardware HLS_FREQ=130
FREQ=100 LINK_FREQ=100 JOBS=8 HW_DEVICE=0`. The config/apply/check SHA-256 values
remain `f498c1d8158e624f1f072af1c853da9c911d1130fcc73350545a0ba7cc20f2cc`,
`e8fc4bf76867b1913acc3f210a356fce27538f8447315f90db64525400b02f7b`,
and `3db132b7bcb660fe4e5d253532f4d1a22de9573a03e83338f546e803cffc2f12`.
Acceptance requires the Iter52 placement gate, zero route defects,
nonnegative setup/hold timing without clock scaling, exact eight-token parity,
and then an exact 64-token performance result. **Status: hardware build
failed timing and the iteration is rejected.** The detached wrapper ran from
2026-08-10 11:54:27 +03:00 to 2026-08-11 07:09:55 +03:00 and exited with
`run_hw.exit=2`; its persistent log, exit marker, timestamps and frozen
manifest are under
`diagnostics/iter53_state_fifo1024_no_credit/hardware/`.

**Implementation result.** Placement preserved the intended coarse
distribution: the recurrent hierarchy was 100% in SLR2; cluster 10 and its
three stream endpoints were 100% in SLR1; cluster 8 was 95.94% in SLR1 with
2,011 leaves left in SLR0. Final SLR CLB occupancies were 98.85%, 80.98%, and
75.65% for SLR0--2. Routing completed with zero failed, unrouted, partially
routed, conflicting, or overlapping nets. The final SLR0--1 SLL demand was
71.92% overall with a 143% worst column, while SLR1--2 was 45.52% overall with
a 99% worst column. Thus the hard floorplan and deeper forward-only queues
preserved routability and substantially reduced the earlier SLR1--2 local SLL
peak, but did not close timing.

Post-route `AggressiveExplore` improved overall setup from WNS -3.741 ns to
WNS **-0.874 ns**, TNS -223.064 ns; hold closed at WHS +0.001 ns. The fixed
250 MHz `dma_ip_axi_aclk_1` remained fatal at WNS -0.874 ns, TNS -86.321 ns
over 409 endpoints. Its dominant paths were the r15 response FIFO
`state[0]` and `fifoaddr_reg[5]` high-fanout cones (529 and 521 loads), followed
by a physically detoured six-LUT path from the path-12 `s01_mmu` reset register
to the `s01_ar_node` payload FIFO. The 100 MHz kernel clock also missed at WNS
-0.550 ns, TNS -136.743 ns over 651 endpoints. Its leading failures were an
SLR0-to-SLR2 reset-to-BRAM path with 99.2% routed delay and a cluster-8 path
that crossed SLR0-to-SLR1 and returned to the 4.06% of cluster leaves left in
SLR0. The final timing report SHA-256 is
`f445b276bbb43aa5c2a0ced013f5b12030a05db10fe9fe77b829a47cc81ee82a`.
No XCLBIN was emitted, so neither on-card liveness nor performance was tested.
The architecture remains a functional native/csynth candidate, but this
physical realization is negative and must not be committed.

### iter54a — target Iter53's routed timing cones at 100 MHz (pre-opt guard failure)

*Prepared: 2026-08-11.*

**Hypothesis and scope.** Keep Iter53's forward-only depth-1,024 state queues,
130 MHz HLS schedule, 100 MHz link target, 32 HBM readers, hard recurrent-SLR2
placement, and hard cluster-10-SLR1 placement byte-for-byte unchanged. Repair
the measured physical timing paths without another architectural perturbation:

- hard-contain cluster 8 in the full SLR1 while keeping `ys_8` movable, so the
  2,011 cluster leaves that spilled into SLR0 cannot create the measured
  SLR0-to-SLR1-to-SLR0 512-load return path;
- apply `MAX_FANOUT_MODE=CLOCK_REGION` and `FORCE_MAX_FANOUT=64` to the exact
  platform-reset driver that produced Iter53's SLR0-to-SLR2 reset-to-BRAM
  path, rather than constraining hundreds of hierarchical aliases as Iter43
  did;
- extend the proven DMA hook to the actual Iter53 r15 response-FIFO
  `state[0]` and `fifoaddr_reg[5]` sources, also at a forced fanout of 64;
- hard-contain only the eight primitives on the measured path-12 AR-control
  path to `CLOCKREGION_X2Y1:CLOCKREGION_X3Y1`, eliminating its detour through
  X4Y2 without constraining the complete HMSS path; and
- retain `SSI_SpreadSLLs` placement but use route directive
  `NoTimingRelaxation`, because Iter53 completed routing cleanly and timing is
  now the acceptance blocker.

**Identity and pre-build validation.** The source remains
`8c666c47c8f0143ef948bb41aa61d8319114cde71c538b4b3b86867f04c23ca1`.
This physical-only run reuses Iter53's bit-identical XO, SHA-256
`fd9c4165dcd3b23ee11dc3498bd6445f66e1a7ebf0b5c5c902d007f874623ee5`;
therefore Iter53's exact native fast/full results and integrated HLS result
(1,621,415 minimum cycles, all 16 clusters II=4) remain applicable. The
config, pre-opt hook, pre-place hook, and post-place check SHA-256 values are
`39a3e907806f3c8864de3960b1f7ffef93641ff488436087b11a5bc715e97329`,
`cd9455dbf6d6b3f299f43a3b940912ca1b29903003343079bdbd172c166b7ebd`,
`e1508e6a8b1567d541a8f74e77056b1c59bccecd221ca37fe8ca03b25f71ed39`,
and `887611568dede19279333584821cd3489cb5441484224765d2e67543971fa627`.
All Tcl files are syntactically complete; the resolved config contains no
placeholder and has SHA-256
`399288483828ba10c824621e70f150db442a9b16b0848084cbd4f135c35e93f1`.
Every exact cell regexp matched one object in the Iter53 final checkpoint,
including all eight AR-control primitives. Vivado 2022.2's command help
confirmed `NoTimingRelaxation` is a supported `route_design` directive.

**Command and acceptance.** The command is `make run_hw
HW_CFG_TEMPLATE=hw_iter54_timing_f100.cfg
BUILD_DIR=build.hw.iter54.timing.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter54_timing/hardware HLS_FREQ=130 FREQ=100
LINK_FREQ=100 JOBS=8 HW_DEVICE=0`. The post-place gate requires Iter52's
recurrent/cluster-10/utilization checks, at least 99.9% of cluster 8 in SLR1,
and all eight AR primitives inside X2Y1:X3Y1. Final acceptance requires zero
route defects, DMA and kernel setup/hold slack all nonnegative without clock
scaling, exact eight-token on-card parity, and an exact 64-token performance
run. **Status: stopped before optimization; rejected as a preflight failure.**
The physical-only link ran from 2026-08-11 12:07:46 to 14:37:22 +03:00 in
persistent tmux session `gdn_iter54` with wrapper PID 2264605 and exited
`run_hw.exit=2`. All 227 block-level synthesis jobs and top-level synthesis
completed, then the pre-opt hook stopped with `reset direct fanout 0 is below
the expected pre-opt floor`. The reset driver cell matched exactly; however,
its local Q-net is a hierarchy-boundary stub with no direct leaf pins at this
stage. The actual flattened kernel-root reset alias carries the loads. No
`opt_design`, placement, routing, timing, XCLBIN, or on-card test ran, so this
result says nothing about the proposed physical fixes. The manifest, wrapper
log, exit marker, and timestamps are under
`diagnostics/iter54_timing/hardware/`. **Verdict: rejected infrastructure
failure; no commit.**

### iter54b — select the flattened reset alias and retry the timing repair (pre-place guard failure)

*Prepared: 2026-08-11.*

**Correction.** Preserve every Iter54a physical constraint, frequency, source,
XO, config, and acceptance gate. Keep the exact platform reset-driver cell as
a hierarchy guard, but set the fanout properties on exactly one flattened
kernel-root alias, `level0_i/ulp/gdn_forward_1/inst/ap_rst_n_inv`. The
inconclusive Iter43 pre-opt log measured this alias at 35,811 flattened pins;
the repaired hook accepts only 20,000--60,000 pins and aborts otherwise. This
selects the physical reset net once without Iter43's 616 repeated hierarchical
aliases. The repaired pre-opt hook SHA-256 is
`ae19a25c10245fbc6a7e2de08ef93ac09c0b3102b41fae199736c7ce50c0730a`.
The config, DMA hook, placement gate, resolved config, and reused XO hashes
remain `39a3e907806f3c8864de3960b1f7ffef93641ff488436087b11a5bc715e97329`,
`e1508e6a8b1567d541a8f74e77056b1c59bccecd221ca37fe8ca03b25f71ed39`,
`887611568dede19279333584821cd3489cb5441484224765d2e67543971fa627`,
`399288483828ba10c824621e70f150db442a9b16b0848084cbd4f135c35e93f1`,
and `fd9c4165dcd3b23ee11dc3498bd6445f66e1a7ebf0b5c5c902d007f874623ee5`.

**Command and result.** Retry with `make run_hw
HW_CFG_TEMPLATE=hw_iter54_timing_f100.cfg
BUILD_DIR=build.hw.iter54.timing.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter54b_timing_reset_alias/hardware HLS_FREQ=130
FREQ=100 LINK_FREQ=100 JOBS=8 HW_DEVICE=0`. Reuse the current build directory
and unchanged XO; the modified hook invalidates the failed link while avoiding
a redundant HLS compile. The retry ran from 2026-08-11 14:44:54 to 15:32:06
+03:00 in persistent tmux session `gdn_iter54b` with wrapper PID 2931041 and
exited `run_hw.exit=2`. The two new r15 DMA sources matched exactly, with
direct fanouts 529 and 521, so that part of the repaired pre-place hook was
validated. The hook then stopped before `place_design` because
`get_clock_regions {CLOCKREGION_X2Y1 CLOCKREGION_X3Y1}` matched zero objects.
The `CLOCKREGION_` spelling is valid in a pblock grid range but is not the
clock-region object-name spelling accepted by `get_clock_regions` in this
checkpoint. Consequently the reset-alias pre-opt repair completed, but the
cluster-8 placement, AR-path placement, routing, timing, XCLBIN, and on-card
tests did not run. This is another script/preflight result and provides no
evidence for or against timing closure. Persistent artifacts are under
`diagnostics/iter54b_timing_reset_alias/hardware/`. **Verdict: rejected
infrastructure failure; no commit.**

### iter54c — correct clock-region object names and retry at 100 MHz (cycle-count improvement; on-card exact)

*Prepared: 2026-08-11.*

**Correction and validation.** Preserve Iter54b's source, XO, 130 MHz HLS
schedule, 100 MHz link target, floorplan, fanout properties, route directive,
and acceptance gates. Correct only the Vivado clock-region naming mismatch:
`get_clock_regions` and the post-place object-name check now use `X2Y1 X3Y1`,
while the pblock resource range correctly remains
`CLOCKREGION_X2Y1:CLOCKREGION_X3Y1`. An independent Vivado 2022.2 query on the
U55C checkpoint resolved exactly two objects, `X2Y1 X3Y1`, and confirmed that
the prefixed names resolve zero objects. Both Tcl files pass `info complete`
and `git diff --check`. The corrected DMA hook and placement-check SHA-256
values are `aa0d8a155684444061aaffa9cfd1f687fc656c1d93745db96c792364683b6bdb`
and `ad6f1e0188775fb16ac8d0a5a3f33c918e9261a2db5786c68cffb00fc5b63af9`.
The pre-opt hook, config, and reused XO remain
`ae19a25c10245fbc6a7e2de08ef93ac09c0b3102b41fae199736c7ce50c0730a`,
`39a3e907806f3c8864de3960b1f7ffef93641ff488436087b11a5bc715e97329`,
and `fd9c4165dcd3b23ee11dc3498bd6445f66e1a7ebf0b5c5c902d007f874623ee5`.

**Command.** Launch `make run_hw
HW_CFG_TEMPLATE=hw_iter54_timing_f100.cfg
BUILD_DIR=build.hw.iter54.timing.f100.o8.v2022_2
RUN_HW_DIR=diagnostics/iter54c_timing_region_fix/hardware HLS_FREQ=130
FREQ=100 LINK_FREQ=100 JOBS=8 HW_DEVICE=0`. Reuse the existing XO and rerun
link plus the automatic exact eight-token and 64-token on-card gates if an
XCLBIN is emitted. The detached run started at 2026-08-11 15:36:42 +03:00 in
tmux session `gdn_iter54c`, with persistent wrapper PID 3263699. Its wrapper
log, PID, exit marker, link artifacts, and on-card results are under
`diagnostics/iter54c_timing_region_fix/hardware/`.

**Implementation result.** Placement honored every hard gate: the 239,370-leaf
recurrent actor was wholly in SLR2, clusters 8 and 10 were wholly in SLR1, and
all eight selected HMSS AR-control primitives were in X2Y1/X3Y1. Routed SLR
CLB occupancy was 98.84/76.87/82.91% for SLR0/1/2. Routing completed with zero
failed/unrouted nets and zero node overlaps and emitted an 81,279,703-byte
XCLBIN. Post-route `AggressiveExplore` improved the routed kernel setup result
from WNS/TNS -1.453/-837.976 ns to **-0.625/-338.949 ns** (1,457 failing
endpoints); hold closed at +0.001 ns. The fixed 250 MHz DMA clock closed at
WNS/WHS **+0.003/+0.009 ns**. Because the kernel did not close at the requested
100 MHz, Vitis auto-scaled it to **94.1 MHz**. This is a routable and usable
image, but it is not 100 MHz timing closure.

**On-card result.** The automatic run first exited `run_hw.exit=2` because the
card was left in a deadlocked XRT load state; this was not an implementation
failure. Resetting the card and rerunning the unchanged XCLBIN completed with
`oncard.rerun.exit=0`. The eight-token smoke and full 64-token trajectory both
passed exact parity (`first_divergence_index=-1`, 100% top-1 agreement). Across
the 63 measured post-seed tokens, kernel latency was 43.667998--43.742944 ms,
median 43.695297 ms and mean **43.702388 ms/token**. At the achieved 94.1 MHz
this is **4.112395M effective cycles/token**, 4.57% fewer cycles than Iter39C's
4.309M. Wall time is 7.17% below Iter38E's 47.079335 ms, but 1.41% above the
100 MHz Iter39C latency record of 43.093 ms; the clock loss masks the cycle
reduction.

**Verdict: retained as a positive cycle-count iteration.** This result advances
the cycle-first roadmap by proving head-streamed QKVG/convolution/recurrent
execution and deep forward-only state queues in a routable, bit-exact image.
Iter39C remains the fastest wall-clock image until timing is recovered. Do not
quote the 1.07727 speedup ratio in `performance_summary.json` as a 7.73%
latency reduction: it corresponds to a correct 7.17% reduction relative to
Iter38E, and it does not compare against Iter39C.

### iter55 — 150 MHz physical islands plus full-logits parity (in progress)

*Prepared: 2026-08-12.*

**Objective and architecture.** Recover and exceed the requested kernel clock
without reducing the 32 weight ports or the 32 aggregate recurrent MAC lanes.
The GEMV chain/collectors are changed from the noncontiguous historical
4/6/6 grouping to contiguous 6/7/3 physical islands, preserving cluster and
result order while limiting each inter-SLR activation/result path to one
boundary crossing. Recurrent attention is split into two concurrent natural
16-column actors: island 0 owns recurrent ports 28/30 and island 1 owns 29/31.
This follows the packed state layout directly, avoids a 32-lane control cone,
and retains 32 aggregate columns/cycle. Q/K/V are duplicated through bounded
BRAM FIFOs and the two outputs are merged in natural Pack16 order. The QKVG
context, tiny-GEMV, and recurrent-scalar reads from the HBM0 auxiliary master
now terminate in explicit producer/consumer streams so the AXI adapter is not
directly coupled to the downstream compute cone.

The corresponding pre-opt floorplan hard-contains clusters 0--5 in SLR0,
6--12 in SLR1, and 13--15 in the west of SLR2; it puts the two recurrent
actors in east SLR2. It retains Iter54c's proven DMA timing hook, uses
clock-region-local reset replication with forced fanout 32, and adds a
post-place gate that recursively checks the placed primitive leaves below all
four hard pblocks. The only Makefile hardware path remains `run_hw`; its
defaults are now HLS/link 150 MHz and
`hw_f150_physical_islands.cfg`.

**Full-logits correctness gate.** The prior standard test compared only
`gen_traj`/`tf_argmax`; it did not preserve or compare the 32,000 values before
argmax. Native-only debug capture now records the final hidden vector and the
actual natural-order LM-head reorder-buffer values without changing the
synthesized ABI or datapath. For every decoded step, `gdn_eval` computes an
independent scalar CPU LM head, compares all 32,000 logits with tolerance
`1e-3 + 1e-4*abs(reference)`, and verifies that the captured-logits argmax is
the token returned by the kernel. It can additionally dump/load a bitwise
reference via `--logits-dump`/`--logits-reference`. The Python parity report
and the one-command decode gate now explicitly fail on any logits tolerance,
exact-reference, or argmax mismatch instead of gating only on token
trajectory.

Before the architecture edit, one Iter54c decode step was frozen as a
32,000-float exact reference at SHA-256
`a11f6981d1bb270abd60f778318471892bf8e599ddea8014aed885ad79272f08`.
After the recurrent split, collector regrouping, and auxiliary buffering, the
same step has zero bitwise mismatches. The fast six-token gate passes the exact
golden trajectory and compares 160,000 logits with maximum absolute/relative
differences `2.86102295e-05`/`7.86781311e-06`, zero CPU tolerance failures,
zero exact-reference mismatches, and zero argmax mismatches. The full
32-token decode-from-state gate also passed: exact trajectory, first
divergence -1, 100% top-1, and 992,000 logits over 31 FPGA-equivalent decode
steps with maximum absolute/relative differences
`4.38690186e-05`/`8.40425491e-06`, zero tolerance failures, and zero argmax
mismatches. Its log SHA-256 is
`90e00db3f06ba33e0eae81bfc843299ef5d7af20a275a25023b680c2f6047bd9`.

**Integrated 150 MHz synthesis.** Vitis HLS 2022.2 completed with report
SHA-256
`f5339ab0f9490fad1497a9a0562f034bb23789fa44d3e323de7110f49cd1be66`.
The 6.667 ns target has 4.867 ns estimated delay plus 1.80 ns uncertainty
(estimated Fmax 205.47 MHz). All 16 `gemv32_cl_flat` loops retain II=4. The
two recurrent actors are concurrent and each is 43,009--43,233 cycles; the
combined wrapper is 43,010--43,234 cycles. Top minimum latency is 1,670,212
cycles versus Iter54c's 1,621,415 (+3.01%). Resources are 1,848 RAMB18,
3,762 DSP, 912,803 FF, 863,815 LUT, and 48 URAM (45.8%, 41.7%, 35.0%, 66.3%,
and 5.0% device-wide). The critical GEMV II and URAM count are unchanged;
the principal cost is 216 RAMB18 for the explicit frequency-boundary FIFOs.

**Identity and pending acceptance.** Current source identities are
`gdn_model.cpp` `7a0457fb...b98d1c3`, `gdn_model.h`
`af4fcf50...e11479`, and `gdn_eval.cpp` `7c3f3e33...b20d9`.
The config/apply/check identities are `4faf23ce...b133436`,
`2a69f4c5...af15b0`, and `ccfb8ef4...4b354`; the Makefile is
`5a53e457...34667b`. The explicit Python logits/trajectory gate is
`13343b8f...d95ab`. With all native and HLS gates passed, launch the single
reproducible command `make -C c_impl run_hw`. Acceptance requires the
four post-place physical gates, zero route defects, WNS/WHS nonnegative at an
achieved 150 MHz without auto-scaling, exact eight-token smoke, and an exact
64-token on-card run.

**Hardware launch.** The fresh v++ XO compilation completed in 29 min 25 s,
again reporting 205.47 MHz estimated Fmax and all 16 GEMV clusters at II=4.
The 17 MiB XO SHA-256 is
`e7c25887f45a156cb2b2e934471424e12b00a38ed4917b9f5bbfd90516812333`.
Inspection of the packaged RTL verified the exact
`gemv32_cluster2[_N]_U0`, `ws_N_U`, `xr_N_U`, `ys_N_U`,
`gemv32_collect6/7/3_U0`, and
`gdn_recurrent_attention_islands_U0` instance names used by the physical
hook. The resolved config SHA-256 is
`3d8d96487a72efd17877b79380e9b61fdd4890d754b87497edf67a5e23b2802b`.
The detached `make run_hw
RUN_HW_DIR=diagnostics/iter55_f150_arch/hardware` link started at
2026-08-12 13:06 +03:00 under persistent wrapper PID 859293. Frozen manifest,
wrapper log, PID, timestamps, and eventual exit marker are in that directory;
the detailed link/implementation tree is
`build.hw.gdn32.h150.f150.o8/_x_temp/`. Status is **native full-logits and
csynth positive; 150 MHz hardware link running, not yet committable**.

**Iter55a implementation result: rejected infrastructure failure.** The first
150 MHz link exited with `run_hw.exit=2` at 2026-08-12 15:42:01 +03:00. All
227 block-synthesis jobs and top-level synthesis completed, but the pre-
`opt_design` floorplan hook stopped before optimization, placement, or routing:

```text
ERROR: [Common 17-161] Invalid option value
'.../gemv32_cluster2_U0' specified for 'objects'.
```

The island lists had been assembled with ordinary Tcl `concat`, which retained
the hierarchy names but stringified Vivado's opaque cell-object handles. The
first `f150_make_pblock` consequently passed a plain cluster name to an
`objects` argument. This is not congestion or timing evidence: no physical
optimization, reset replication, placement, route, timing report, XCLBIN, or
on-card run occurred. The BRAM sufficiency message was a non-fatal platform
utilization warning; the Tcl exception is the sole terminating error. The
wrapper and detailed run log are retained under
`diagnostics/iter55_f150_arch/hardware/` and
`build.hw.gdn32.h150.f150.o8/_x_temp/link/vivado/vpl/prj/prj.runs/impl_1/`.

**Iter55b infrastructure correction and retry.** Preserve the Iter55 source,
XO, 150 MHz clocks, collector/recurrent architecture, physical regions, reset
fanout 32, and all acceptance gates. Correct only `f150_make_pblock`: re-resolve
the accumulated names with `get_cells -hierarchical`, require the resolved
collection count to match the requested count, and pass that collection as a
whole to `add_cells_to_pblock` and `set_property USER_SLR_ASSIGNMENT`. The
corrected hook is Tcl-complete, passes `git diff --check`, and has SHA-256
`12b2919811e842f218b8ab5453343cd914cb19920f199135cd69615f3a796aa8`.
The unchanged XO remains
`e7c25887f45a156cb2b2e934471424e12b00a38ed4917b9f5bbfd90516812333`.
The detached retry launched at 2026-08-12 16:27:49 +03:00 with wrapper PID
1034912 using `make run_hw
RUN_HW_DIR=diagnostics/iter55b_f150_object_fix/hardware`. Its manifest,
wrapper log, timestamps, PID, eventual exit marker, and automatic on-card
outputs are retained under that directory. Status is **infrastructure retry
running; not committable**.

**Iter55b result: rejected lookup-semantics failure.** The retry completed the
cached 55-job synthesis stage, then exited at 2026-08-12 17:06:56 +03:00,
again before `opt_design`, with the new guard reporting
`pb_f150_clusters_slr0 resolved 0 of 31 cells`. Direct read-only testing
against the synthesized 350 MB kernel checkpoint proved the Vivado 2022.2
lookup behavior: an exact full hierarchy name returns one object with
`get_cells -quiet $full`, while `get_cells -hierarchical -quiet $full` returns
zero. Thus the count guard correctly prevented a silently empty floorplan, but
the re-resolution form was wrong. There is still no placement, reset,
congestion, routing, timing, XCLBIN, or on-card evidence from this attempt.
**Verdict: rejected infrastructure attempt; no commit.**

**Iter55c exact-object correction.** Preserve the architecture, XO, clocks,
physical intent, and acceptance gates unchanged. Resolve every stored full
hierarchy name individually using exact `get_cells -quiet $cell_name`, require
exactly one match, and combine the returned opaque objects using
`add_to_collection`. The corrected hook is Tcl-complete, passes
`git diff --check`, and has SHA-256
`c4b7d7dffe7387f7c8ce570362ad1c09dc7368638555f15b06f2f989cb7b002f`.
The direct checkpoint experiment supplied positive API-level evidence for the
exact lookup before this retry. The detached retry launched at
2026-08-12 17:21:44 +03:00 under wrapper PID 1049458 using `make run_hw
RUN_HW_DIR=diagnostics/iter55c_f150_exact_objects/hardware`; its manifest,
wrapper log, timestamps, exit marker, and automatic on-card outputs are kept
there. Status is **corrected retry running; not committable**.

**Iter55c result: rejected non-Vivado collection command.** The retry reused
all cached synthesis products and reached the pre-`opt_design` hook, then
exited at 2026-08-12 17:59:19 +03:00 with `invalid command name
"add_to_collection"`. Exact cell lookup was no longer the failure;
`add_to_collection` is a Synopsys-style collection command absent from Vivado
2022.2. No pblock, reset replication, optimization, placement, route, timing
report, XCLBIN, or on-card run resulted. **Verdict: rejected infrastructure
attempt; no commit.**

**Iter55d Vivado-native pblock correction.** Preserve all architectural and
physical intent. Preflight every exact hierarchy name, create the pblock, then
resolve and pass each opaque cell object individually to Vivado's native
`add_cells_to_pblock` and `set_property USER_SLR_ASSIGNMENT` commands. Require
the final pblock root count to equal the requested count. The post-place check
similarly resolves each primitive name independently. A read-only API test on
the cached `bd_85ad_switch_2to1_12_0.dcp` demonstrated the complete operation
before relaunch: 468 hierarchical candidates, exact lookup 1, pblock roots 1,
and `USER_SLR_ASSIGNMENT=SLR0`. Both corrected hooks are Tcl-complete and pass
`git diff --check`; their SHA-256 values are
`063dafda71428cf106855f087752e4c0eacae859945d7675493ff671a5346210`
and `137b9fc7188178713c24fc02baeda6be5fe501229df77a6480252f3d626f4ffb`.
The detached retry launched at 2026-08-12 18:09:51 +03:00 under wrapper PID
1209174 using `make run_hw
RUN_HW_DIR=diagnostics/iter55d_f150_vivado_objects/hardware`; its manifest,
wrapper log, timestamps, exit marker, and automatic on-card outputs are kept
there.

**Iter55d result: rejected localized-SLL routing failure.** This was the first
Iter55 attempt to complete the physical-hook infrastructure and exercise the
new architecture in implementation. All requested hard pblocks were accepted,
and the post-place checker found every assigned primitive inside its requested
region: 305,667 leaves in the SLR0 cluster group, 356,253 in the SLR1 cluster
group, 152,937 in the SLR2-west cluster group, and 213,477 in the SLR2-east
recurrent group, with zero outside each region. The reset constraint also
applied (`reset_fanout=32`), and `opt_design`, placement, and post-place
physical optimization completed. Physical optimization improved estimated WNS
from -8.639 ns to -2.641 ns, but this is only pre-route evidence and does not
meet the 150 MHz target.

Routing then failed during its first global-routing iteration, before detailed
routing, with `[Route 35-3339] unable to resolve localized SLL routing demand`
and `[Route 35-368] Router failed to resolve global congestion`. Aggregate SLL
use was not exhausted, but individual columns were impossible: the SLR1-SLR2
boundary requested 12,967/23,040 SLLs (56.28%) in total while peaking at
2,516/1,440 (175%) in one column and 2,052/1,440 (142%) in another; the
SLR0-SLR1 boundary requested 18,146/23,040 (78.76%) while peaking at
2,582/1,440 (179%), with five other columns above 100%. Global-routing
congestion fell from 13,979 to 2,665 and then stalled. The reported 1,480,023
failed nets, 1,304,768 unrouted nets, 175,255 partially routed nets, and 95 node
overlaps describe this early aborted global route, not an almost-complete
route. Intermediate route timing was WNS -2.538 ns/TNS -2,407.626 ns, but it
is non-signoff timing because routing never completed. Linked utilization was
592,774 LUTs, 796,658 registers, 1,475.5/1,776 BRAMs, and 3,768 DSPs. The run
exited 2 at 2026-08-13 01:04:57 +03:00; it produced only
`level0_wrapper_routed_error.dcp`, no XCLBIN and therefore no on-card or logits
comparison result. The reset repair was successfully exercised and is not the
reported fatal blocker; the hard whole-hierarchy 6/7/3 plus SLR2-east
partition concentrated inter-SLR data/control traffic into too few SLL
columns. **Verdict: rejected physical architecture/floorplan; no commit.**

### iter56 — surgical 4/6/6 placement and registered result boundaries

*Prepared: 2026-08-13.*

**Hypothesis and measured basis.** Iter55d's post-place report used
22,080/23,040 SLLs (95.83%) at SLR0--SLR1 and 17,044/23,040 (73.98%) at
SLR1--SLR2, including 8,075 direct SLR0--SLR2 signals. Its BRAM distribution
was 93.30/62.50/92.93% across SLR0--2, and none of the 39,124 crossing signals
used dedicated TX/RX crossing registers. The broad hard pblocks therefore
removed useful placement freedom and concentrated both BRAM endpoints and
SLLs. Iter56 retains the two concurrent 16-column recurrent actors but restores
the routed 4/6/6 result-collector cut. It removes the 6/7/3 whole-cluster,
whole-FIFO, local-collector, auxiliary-reader/consumer, and SLR2-west/east
pblocks. Only the full recurrent wrapper in all of SLR2, cluster 8 in SLR1,
cluster 10 plus `ws20`/`ws21`/`xr10` in SLR1, and the small result-boundary
relay/final-collector group in SLR1 are hard-contained.

Three explicit II=1 Pack16 relay actors now separate the local 4/6/6
collectors from the final collector. Their sequential leaves are marked
`USER_SLL_REG` for SSI placement. The six depth-32 Q/K/V duplication FIFOs and
two depth-16 recurrent-output FIFOs are changed from BRAM to LUTRAM. This is a
targeted approximately 3.6K-LUT exchange expected to free roughly 112 RAMB18
equivalents; all 69 high-traffic GEMV MM2S, activation, and result decouplers
remain in BRAM. Reset replication remains clock-region-local at forced fanout
32, and the proven Iter54 DMA timing hook is unchanged.

**Early rejection gate.** The post-place hook requires every surgical pblock
to be honored, at least one `USER_SLL_REG` leaf to survive, SLR0--SLR1
connectivity at most 85%, SLR1--SLR2 at most 65%, direct SLR0--SLR2 signals at
most 6,000, every SLR below 95% occupied CLBs, and every SLR below 90% BRAM.
Failure aborts before routing rather than spending hours on an already
nonviable placement. The 150 MHz HLS/link target, `SSI_SpreadSLLs`,
`NoTimingRelaxation`, and pre/post-route `AggressiveExplore` remain unchanged.

**Native validation and identity.** `make -C c_impl -j8`, the fast exact gate,
and the full 32-token gate pass. Fast validation compared 160,000 pre-argmax
logits with maximum absolute error 2.86102295e-05, zero tolerance failures,
zero exact-reference mismatches, zero argmax mismatches, and exact token
trajectory. Full validation compared 992,000 logits with maximum absolute
error 4.38690186e-05 and the same zero-failure/exact-token result. Source,
apply hook, post-place hook, DMA hook, config, and Makefile SHA-256 values are
respectively `f52b6b64c2267333247b4cbed5f8ffae6f6b582a8f47035e217b58814076a5ff`,
`6d36b0f1c52383c787d543d55a849ded1567113cf25bac4d8523f8cc0c12483f`,
`1cb422d8fc52d91fa5b4de65582e5b9440a8814e76a541a69f0ce9783a05e6fe`,
`aa0d8a155684444061aaffa9cfd1f687fc656c1d93745db96c792364683b6bdb`,
`4faf23ce87da7fb6c397305a82c3a22d24340198787dee9de1e037bd8b133436`,
and `5a53e457a49a1a199f82e31b170b8adb2b4722a8d9c56881babb87759934667b`.
All Tcl sources are syntactically complete and `git diff --check` passes.

**Integrated synthesis and launch gate.** The final 150 MHz XO export
completed successfully after increasing each result-boundary relay FIFO from
the exploratory depths of 2 and 8 to depth 16. HLS had measured a maximum
required depth of 14 on the SLR0 relay; rounding all three complete-burst
buffers to 16 removes every relay-depth/deadlock recommendation while retaining
distributed RAM implementation. The final report estimates 1,670,212 minimum
top-level cycles, 205.47 MHz Fmax, 1,728 RAMB18, 3,762 DSPs, 921,774 FFs,
869,262 LUTs, and 48 URAMs. Relative to Iter55 synthesis this saves 120 RAMB18,
keeps DSP and minimum cycles unchanged, and adds 8,971 FFs and 5,447 LUTs.
The `gemv32_cl_flat` loop remains at its requested and achieved II=4. The
exported XO SHA-256 is
`e7a6153b0b0c7784fb99fd0a85d85c70ecf56c199a4d6e2fcf6b49e1876b34aa`.
These are synthesis results only; routability, 150 MHz timing closure, logits
parity, and on-card latency remain unproven. **Status: hardware launch ready;
not committable.** The detached hardware flow launched at
2026-08-13 03:28:59 +03:00 under wrapper PID 1575633 using `make run_hw
RUN_HW_DIR=diagnostics/iter56_surgical_f150/hardware HLS_FREQ=150
LINK_FREQ=150 FREQ=150 JOBS=8 HW_DEVICE=0`. Its manifest, wrapper log, PID,
eventual exit marker, and automatic 8/64-token on-card outputs are retained
under that run directory. **Implementation status: running.**

**Iter56 result: inconclusive post-place density/SLL gate.** The detached flow
exited 2 at 2026-08-13 07:37:42 +03:00. Synthesis, `opt_design`, and placement
completed, and every surgical pblock was honored: all 217,406 placed recurrent
leaves were in SLR2, all 50,078 cluster-8 leaves and all 50,266 leaves in the
cluster-10/FIFO cone were in SLR1, and all 865 result-boundary leaves were in
SLR1. The placement gate then stopped the flow before `phys_opt_design` and
`route_design`; no routed checkpoint, XCLBIN, or on-card result exists.

The architectural correction did relieve the Iter55d inter-SLR problem.
SLR0--SLR1 connectivity fell from 95.83% to **90.08%**, SLR1--SLR2 fell from
73.98% to **61.05%**, direct SLR0<->SLR2 signals fell from 8,075 to **2,986**,
and total SLL use fell from 39,124 to **34,820**. Post-place estimated WNS also
improved from -2.641 ns to **-1.565 ns**. However, freeing most actors let the
placer collapse too much logic and memory toward HBM: SLR0/1/2 occupied CLBs
became **99.32/85.89/64.61%**, while BRAM became
**93.30/92.63/53.87%**. The run would therefore also have failed the 95% CLB
and 90% BRAM gates even if the lower-boundary SLL threshold had been relaxed.
Vivado explicitly classified the placed design as highly congested, with
128x128 global congestion to the north and 128x128 short congestion in several
directions. Although 80 relay registers retained `USER_SLL_REG`, the SLR report
showed zero crossings using dedicated TX/RX registers, so the relay constraint
did not materialize into dedicated Laguna crossings in this placement.

**Gate calibration correction.** These thresholds were more conservative than
the latest successful routed/on-card Iter54c image. Iter54c routed with SLR0
at 54,322 occupied CLBs (**98.84%**), lower-boundary connectivity at **88.87%**
and 34,646 total SLLs. Iter56 had 54,585 occupied SLR0 CLBs (**99.32%**),
lower-boundary connectivity **90.08%**, and 34,820 total SLLs: only +263 CLBs,
+1.21 percentage points on the lower boundary, and +174 total SLLs. Iter54c
also routed despite congestion levels reaching 7. The meaningful negative
delta is memory distribution: Iter56 used 93.30/92.63/53.87% BRAM versus
Iter54c's routed 86.68/83.93/61.38%. This raises risk but does not prove route
failure. Therefore Iter56 must not be described as an unroutable realization;
the router was never invoked, and the experiment is **inconclusive**.

The malformed text fragment after `90.08` in the fatal line is VPL's rendering
of percent signs in a Tcl error string; the parsed 90.08% value and comparison
were correct and it is not the cause of failure. **Verdict: inconclusive due to
an over-conservative user gate; no commit.** A valid follow-up is to convert
the CLB/BRAM/SLL thresholds to reporting-only checks and let this exact placed
topology enter routing. Redistribution toward SLR2 remains a fallback if that
route actually fails, not a prerequisite inferred from placement alone.

**Reporting-only correction prepared.** The post-place script now keeps exact
pblock placement, complete-report parsing, and missing `USER_SLL_REG` checks
fatal, but converts the SLL, CLB, BRAM, and direct-SLR thresholds into
`GDN_ITER56_ADVISORY` messages. It always emits
`GDN_ITER56_REPORT_ONLY ... proceeding_to_route` after a structurally valid
placement. The corrected script is Tcl-complete, passes `git diff --check`, and
has SHA-256
`ffca9307e73543c7e8a2408fd1a6678dfd2975fdd9b6628e28fb242c084b2e2b`.
No retry has been launched yet. The stopped 150 MHz run contains no placed or
post-physopt DCP—the failed post-place hook prevented the standard checkpoint
write—so it cannot resume directly at `route_design`. A same-clock Vitis retry
can reuse synthesized artifacts but must repeat optimization and placement; a
100 MHz link is a distinct clock-constrained implementation and must likewise
repeat placement before routing.

### iter56b — reporting-only retry at 100 MHz

*Prepared: 2026-08-13.*

Preserve the exact Iter56 architecture, 150 MHz HLS compilation, 4/6/6
collector graph, recurrent split, surgical pblocks, DMA hook, and validated XO.
Change only the link clock to 100 MHz and use the corrected reporting-only
post-place script. Reuse the Iter56 XO at SHA-256
`e7a6153b0b0c7784fb99fd0a85d85c70ecf56c199a4d6e2fcf6b49e1876b34aa`
instead of repeating HLS. Source/apply/check/DMA/config/Makefile SHA-256 values
are respectively
`f52b6b64c2267333247b4cbed5f8ffae6f6b582a8f47035e217b58814076a5ff`,
`6d36b0f1c52383c787d543d55a849ded1567113cf25bac4d8523f8cc0c12483f`,
`ffca9307e73543c7e8a2408fd1a6678dfd2975fdd9b6628e28fb242c084b2e2b`,
`aa0d8a155684444061aaffa9cfd1f687fc656c1d93745db96c792364683b6bdb`,
`4faf23ce87da7fb6c397305a82c3a22d24340198787dee9de1e037bd8b133436`,
and `5a53e457a49a1a199f82e31b170b8adb2b4722a8d9c56881babb87759934667b`.
The build uses the ordinary `run_hw` target with a fresh
`build.hw.gdn32.h150.f100.o8` link directory and, if implementation succeeds,
automatically runs exact 8-token and 64-token on-card gates. Acceptance is zero
route errors, nonnegative setup/hold slack at an actual 100 MHz kernel clock
without automatic scaling, exact token/logits parity, and a measured 63-token
mean latency. **Status: launch pending; not committable.**

The detached retry launched at 2026-08-13 13:42:56 +03:00 under wrapper PID
2307534 using the command recorded above. Its manifest, wrapper log, PID,
eventual exit marker, and automatic on-card outputs are under
`diagnostics/iter56b_surgical_f100/hardware/`. **Status: 100 MHz link running;
not committable.**

**Outcome of the Iter56b link.** The 100 MHz link completed and produced an
XCLBIN (80.9 MB, v++ 12 h 01 m). Both clocks met: `clk_kernel_00` WNS
**+0.016 ns** and `dma_ip_axi_aclk_1` **0.000 ns**, zero failing endpoints, so
the image is a true 100 MHz build with no automatic scaling. **On card it
hung.** `host.exe` blocked in `run.wait()` for 4 h 22 m having consumed 16 s of
CPU, `wchan=do_sys_poll`, log frozen after `decode-from-state seed=21225 N=8`.
Killed with SIGTERM; device `0000:c2:00.1` reset; all four cards returned Ready.
**Verdict: rejected by on-card hang; not committable.** Cause identified in
Iter57 below.

### iter57 — state queues of two heads for the island recurrence (retained; 42.023540 ms/token on card)

*Tested: 2026-08-14--15. Evidence: on-card, bit-exact, timing-closed.*

**Cause, localised by C/RTL cosimulation rather than inference.** The Iter56b
hang was reproduced in simulation and the AESL detector named a seven-node
dependence cycle (`cosim_isl150_d32`, deadlock at 145,101,420 ps):

```
(1) gdn_recurrent_attention_islands_U0  <- q/k/v empty      from store_or_qkvg_conv_stream
(2) gemv32_store_or_qkvg_conv_stream_U0 <- result_U empty   from collect_final
(3) gemv32_collect_final_U0             <- slr2_boundary_U  from boundary_relay_2
(4) gemv32_boundary_relay_2_U0          <- slr2_result_U    from collect6_16
(5) gemv32_collect6_16_U0               <- ys_14_U          from cluster2_14
(6) gemv32_cluster2_14_U0               <- ws_29_U empty    from mm2s_with_state_29
(7) gemv32_mm2s_with_state_29_U0        <- state_stream1_U FULL, read by (1)
```

`gemv32_mm2s_with_state_29` emits, per head, 4,096 weight packs then 1,024 state
packs, and `state_stream1` was depth 1024 -- **exactly one head**. It fills head
*h*'s state, streams head *h+1*'s weights, then blocks writing head *h+1*'s
state because head *h*'s is still unread; island 1 cannot drain it until head
*h*'s q/k/v arrives, and those are in flight behind the island redesign's new
`gemv32_boundary_relay_2` stage. The relay lengthened the q/k/v round trip past
one head, so a buffer that was exactly sufficient before the redesign no longer
is. This is why depth 1024 worked in Iter54c (no boundary relay in that path)
and fails here.

**A wrong hypothesis was eliminated by measurement, not argument.** The initial
suspicion was the `out0`/`out1` merge fan-in. Bisecting it (16/32/64/128) is
decisive against: `d32` and `d64` deadlock at the *identical* simulated time,
145,035,000 ns, so quadrupling the output depth changes nothing. `out0`/`out1`
remain at 16 in the shipped design.

**Fix and bisection.** `state_stream0..3` depth 1024 -> **2048** (two heads).
Cosim with depth 2048 and 4096 both ran to 766,076,490 ps and 762,167,070 ps
respectively -- **5.3x past the depth-1024 deadlock point** -- with no banner and
no frozen interval across twelve consecutive five-minute polls. 4096 bought
nothing over 2048, so 2048 is the minimum sufficient depth. Harnesses were
generated by `make_cosim_islands.sh`, which asserts the only differences from
production are the header include, `GDN_LAYERS 24->1`, the 34 rescaled m_axi
depths and the depth under test (`unexpected=0` on every variant).

**Identity.** `gdn_model.cpp` SHA-256 `2aa7d9c044af627f925a03ef...`;
`hw_f150_physical_islands.cfg` `4faf23ce87da7fb6c397305a...`;
`apply_f150_physical_islands.tcl` `6d36b0f1c52383c787d543d5...`; XCLBIN
`4178d442d956eece...`. Command:
`make run_hw HW_CFG_TEMPLATE=hw_f150_physical_islands.cfg
BUILD_DIR=build.hw.gdn32.h150.f100.o32 HLS_FREQ=150 FREQ=100 LINK_FREQ=100
JOBS=32 HW_DEVICE=0`. Everything except the queue depth is byte-identical to the
Iter56b recipe, so the outcome is attributable to that one change.

**Validation.** Fast decode-correctness gate passed before launch. Route: zero
failed nets, zero unrouted, zero overlaps. Timing closed on both clocks --
`clk_kernel_00` **WNS +0.060 ns, 0 failing**, `dma_ip_axi_aclk_1` **+0.003 ns,
0 failing** -- so this is a genuine 100 MHz image, not an auto-scaled one.

**Cost.** RAMB36 585/576/337 across SLR0/1/2 (87.05/85.71/50.15%), total 1,498
against Iter54c's 1,424: **+74 tiles**. Predicted +64 from capacity arithmetic
(512 bits x 2048 / 32,768 usable = 32 RAMB36 per FIFO, x4, less the 64 already
present); the extra ten are surrounding logic. Note the tiles landed mostly in
SLR0/SLR1 (+32/+31) rather than in SLR2 where the islands live (+11), so SLR0
and SLR1 are now the BRAM-tight SLRs at ~86-87%.

**On-card result, 63 timed runs, one token per call:**

| Metric | Value |
|---|---:|
| mean | **42.023540 ms/token** |
| median | 42.009 |
| min / max | 41.968 / 42.389 |
| vs Iter39C 43.093 | **1.025x** |
| vs Iter38E 47.079 | 1.120x |
| vs 8-port 121.4 | 2.889x |

Both gates exact: 8-token and 64-token trajectories match the GPU golden
bit-for-bit (`RESULT: PASS` on `oncard_smoke8_parity.log` and
`oncard_decode64_parity.log`).

**Verdict: retained.** First image of this arc that routes, closes timing at a
true 100 MHz, and runs bit-exact, and it improves on the Iter39C production
baseline by 2.5%. Artifacts under `diagnostics/iter57_state2048/`.

### Superseded plan (written before iter12 ran)

Pin **only the 16 clusters**, contiguous in chain order, **6/6/4** across
SLR0/SLR1/SLR2. Collectors, FIFOs, AXI adapters and the whole aux path stay
unpinned, exactly where iter10's placer put them. This differs from rows 4-9,
which pinned 16 clusters *plus* FIFOs *plus* aux and drove SLR0 down to 4
clusters. The delta from iter10 is 3 extra remote clusters (~+3,126 nets), so even
with the 39% model error the boundary lands near 20,300 of 23,040 — inside a cap
the routed microbenchmark survived at 94.8% — while SLR0 sheds a third of its
GEMV logic.

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

## Iter58 — LM head emits full logits instead of on-chip argmax

*Logged: 2026-08-18. Status at time of writing: **native-only** (csim). Build
not yet run; no csynth, routed, timing, or on-card evidence.*

**Hypothesis.** Returning the full `GDN_VOCAB` logit vector instead of a single
greedy token id is a prerequisite for scoring log-likelihood benchmark suites
(lm-eval Table 3) on-card. The added HBM traffic should be negligible against
the ~5.195 GB/token of weight streaming, and removing the fused argmax should
not hurt timing.

**Change.**
- `gemv32_store` LM-head branch (`rows_per_ch == GDN_VOCAB / GEMV_CHANNELS`):
  the fused `gemv32_argmax_*` reduction is replaced by `gemv32_logits_pack`,
  which assembles each output `Pack16` completely in registers and issues one
  full 512-bit store. **This shape is the reason the write must be
  pack-assembled:** `rows_per_ch` is 1000, so per-channel spans are not
  pack-aligned, and the pre-existing scalar path at
  `gemv32_store_scalar_r` does a per-lane `out[pack].data[lane] = ...`
  read-modify-write. Against local BRAM that is harmless; aimed at HBM it
  degenerates into 4-byte AXI transactions — the same failure mode that once
  cost 1610 ms/token at 0.99% bus efficiency. `GDN_VOCAB` is a multiple of 16,
  so the natural-order vector is a whole number of packs and no partial line is
  ever written.
- The LM-head `gdn_gemv` call now targets `workspace + GDN_WS_OFF_LOGITS`
  (a workspace region that was already reserved but unused) rather than
  `gemv_out_storage`, which is sized for the 5632-wide MLP at 704 packs and
  could not hold the 2000-pack logit vector.
- The `x_norm[0]` token-id handoff is removed. `gdn_eval.cpp` and `host.cpp`
  now compute the greedy argmax host-side with strict `>` and first-index tie
  breaking, reproducing the retired on-chip reduction exactly.

**Expected cost.** 2000 extra 512-bit writes per token, ~18k cycles at the
scalar-scan II of ~9, against a 4.202354M-cycle token: under 0.5%. Removing the
argmax comparators and their partitioned lane registers should free LUTs.

**Identity.** `gdn_model.cpp` 38285cdd1641074d, `gdn_model.h` af4fcf5048fd941c,
`host.cpp` a23b8e2603b6ca68, `gdn_eval.cpp` 6dfdb65ade937959,
`hw_f150_physical_islands.cfg` 4faf23ce87da7fb6 (unchanged from Iter57).

**Validation so far (native only).**
- `scripts/decode_correctness_check.sh --fast`: PASS.
- `scripts/decode_correctness_check.sh` (full, 32 steps): PASS — exact
  trajectory, first divergence index -1, top-1 agreement 100.00%, and
  **992,000 pre-argmax logits** checked against the independent scalar LM head
  with `cpu_tol_fail=0`, `exact_ref_mismatch=0`, `argmax_mismatch=0`
  (max abs 4.39e-05, max rel 8.40e-06 — the usual reduction-order difference).
- `make host`: builds clean against XRT.

**Result: REJECTED at place_design — device BRAM overflow.** The build ran
5 h 35 m (02:52:42 → 08:27:28) and failed DRC before the placer started:

```
[VPL UTLZ-1] RAMB36E2 over-utilized in Top Level Design:
             requires 2045, only 2016 available          (29 over)
[VPL UTLZ-1] RAMB18 and RAMB36/FIFO over-utilized:
             requires 4303, only 4032 available
[VPL UTLZ-1] RAMB36E2 over-utilized in pblock_dynamic_region:
             requires 1799, only 1776 available          (23 over)
[VPL 4-23]   Error(s) found during DRC. Placer not run.
```

HLS estimated **3019 BRAM (74%)** and **1,211,412 LUT (92%)** against Iter57's
routed 1,728 RAMB18 / 869,262 LUT.

**Root cause (from the reports, not inferred).** `gdn_gemv` carries
`#pragma HLS inline off`, so its `out` parameter is *one shared interface*
across every GEMV call in the design. Routing the LM head's output through it
meant `out` was indexed to `GDN_VOCAB/16 = 2000` packs, so HLS sized that
shared buffer for the entire logit vector — 128 KB, about 29 RAMB36. The
device overflow is 29 RAMB36E2. The costs match, and the pblock overflow of 23
is the same buffer seen inside the dynamic region.

The failure had nothing to do with the pack-aligned HBM write, which was the
risk this entry was written to guard against, and nothing to do with timing —
the placer never ran.

**Remedy (Iter58b, native-validated).** Give the store path its own AXI
pointer, `float *logits_out`, threaded through `gemv32_store`,
`gemv32_store_or_qkvg_conv_stream`, and `gdn_gemv`. The LM-head branch writes
logits there; `out` reverts to `gemv_out_storage` and keeps its MLP size, so
the shared buffer is never sized for the vocabulary. Every `gdn_gemv` call
site passes `workspace + GDN_WS_OFF_LOGITS`; only the LM-head shape reads it.

Full native gate re-passes after the fix: exact trajectory, 992,000 logits,
zero mismatches. `make host` clean.

**Process note.** This iteration cost 5 h 35 m to learn a resource fact that a
30–60 minute `make xo` would have reported. A csynth resource check now
precedes any link on a change that touches buffer sizing or interfaces.
`test.tcl` cannot serve that purpose as it stands — it fails compilation on
`GDN_NORM_LANES` and friends, so `make xo` is the check to use.

**Iter58b csynth confirms the remedy (measured, 32 min).**

| | BRAM | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| Iter58 shared `out` (rejected) | 3019 (74%) | 6416 (71%) | 1,440,496 | 1,211,412 (92%) | 64 |
| **Iter58b dedicated `logits_out`** | **1728 (42%)** | 3777 (41%) | 920,110 (35%) | 867,476 (66%) | 48 |
| Iter57 routed, for reference | 1728 | 3762 | 921,774 | 869,262 | 48 |

A single shared interface was costing 1,291 BRAM, 2,639 DSP and 344k LUT. The
remedy returns the design to Iter57's footprint and is marginally *below* it on
LUT and FF, since the fused argmax comparators and their partitioned lane
registers are gone. BRAM at 42% clears the 2016-cell RAMB36E2 limit that Iter58
overflowed by 29.

Whole-design HLS latency rises 0.84% against the rejected variant
(70,824,933 vs 70,231,813 cycles), consistent with adding ~2000 pack writes per
token. The per-token cost that matters is the on-card measurement, still to come.

Status: **native-validated and csynth-validated; link not yet run.** Timing
remains the open question — Iter57 closed at +0.060 ns / +0.003 ns WNS, so the
margin is thin even at an unchanged footprint.

## Iter59 — memory rebalanced BRAM -> URAM from measured congestion evidence

*Logged: 2026-08-19. Status: **native-validated and csynth-validated**; no link,
no routing, no timing, no on-card evidence.*

**Why.** `report_design_analysis -congestion` on the Iter57 **routed**
checkpoint (the design that closes timing) shows every congestion window shares
one signature: **26 of 29 windows at RAMB >= 96%, and 27 of 27 at URAM = 0%**.
Device BRAM was 1885/2016 (93.5%) against URAM 48/960 (5.0%), with SLR0 and
SLR1 holding zero URAM between them. The named contributors are the GEMV
clusters — `gemv32_cluster2_{1,2,3,4,5,7,9,11}` — plus the shell's `hmss_0`.
Vivado's own `report_qor_suggestions` raises **RQS_UTIL-211: "High BRAM usage.
Convert some BRAM to URAM"** on the same checkpoint, estimating -598 RAMB18 for
+38 URAM.

**Change.** `ws[32]` and the four depth-2048 state queues bind to URAM;
`xr[17]` and `ys[16]` stay in BRAM.

**Measured (csynth, four variants):**

| variant | BRAM | URAM | LUT | FF | latency |
|---|---:|---:|---:|---:|---:|
| A all-BRAM (iter58b) | 1728 (42%) | 48 (5%) | 867,476 | 920,110 | 70,824,933 |
| B +state queues | 1668 (41%) | 80 (8%) | 867,776 | 920,654 | 70,824,933 |
| C ws+xr+ys+state | 693 (17%) | 600 (62%) | 873,561 | 930,144 | 70,824,933 |
| **D balanced (retained)** | **1188 (29%)** | **336 (35%)** | 870,624 | 925,326 | 70,824,933 |

Latency and DSP are identical in all four. D costs +3,148 LUT over A (+0.36%).

**Why D and not C.** C frees the most BRAM but the device has only **five URAM
column positions** (clock-region X = 1,3,4,5,6; none at X0/X2/X7), 16 per clock
region, 320 per SLR — queried from the part, not inferred. Distributed like the
clusters (SLR0 49% / SLR1 39% / SLR2 12%), C's 600 blocks put SLR0 near **92%
URAM**, which relocates the congestion rather than relieving it. D projects to
roughly **SLR0 BRAM 63% / URAM 51%** — the only variant where neither memory
type approaches saturation in the worst SLR.

**Pblock interaction — checked, no change needed.** `f150_slr_range` builds
ranges as `CLOCKREGION_X..Y..`, and a clock-region range includes every site
type in those regions, URAM included. Explicit `SLICE_X…`/`RAMB18_X…` ranges
would have excluded URAM and failed at placement. `pb_iter56_cluster10_slr1`
pins `ws_20_U`, `ws_21_U`, `xr_10_U` by name; the two `ws` FIFOs are now
URAM-backed and confined to SLR1, needing 16 of its 320.

**Vivado suggestion triage** (from `diagnostics/congestion_forensics/gdn_iter59.rqs`):
- **RQS_UTIL-211** (BRAM->URAM): the useful one, and the same direction as this change.
- **RQS_UTIL-206** (BRAM->LUTRAM): -598 RAMB18 but **+81,665 LUTRAM**. Must stay
  disabled — SLR0 is at 98.83% CLB with 30,786 LUTRAM already.
- **RQS_UTIL-12** (SRL->register): targets **exactly one** FSM cell, worth 1 FF.
  Negligible; it does **not** address SLR0's 16,297 SRLs, which Vivado examined
  and left alone.

**Validation.** `decode_correctness_check.sh` full: PASS — exact trajectory,
992,000 pre-argmax logits, zero mismatches. Source `gdn_model.cpp` c688fccca52e6a0b.

**Result: REJECTED at route_design — congestion level 7, worse than baseline.**
Link ran 6 h 15 m (00:52:56 -> 07:09:42) and failed:

```
ERROR: [VPL 35-3]    Design is not routable as its global congestion level is 7.
ERROR: [VPL 18-1000] Routing results verification failed due to
                     partially-conflicted nets
```

**Post-place congestion got worse in every direction, none improved:**

| direction | Iter57 (routes, closes timing) | Iter59 URAM |
|---|---|---|
| North | 128x128 / 128x128 / **64x64** | 128x128 / 128x128 / **128x128** |
| South | **32x32** / 128x128 / **64x64** | **128x128** / 128x128 / **128x128** |
| East | **32x32** / **16x16** / 128x128 | **128x128** / **64x64** / 128x128 |
| West | **64x64** / 32x32 / 128x128 | **128x128** / 32x32 / 128x128 |

Global congestion reached 128x128 on all four directions; Iter57 holds 32x32 on
South and East.

**Why the hypothesis was wrong.** The entry above reasoned that because 26 of 29
congestion windows sat at RAMB >= 96% with URAM at 0%, BRAM scarcity was
*causing* the congestion. That is correlation, not causation: memory columns are
inherently dense regions, so they show high occupancy inside any congestion
window. Cutting BRAM 42% -> 29% changed routability not at all and perturbed a
placement that was only marginally routable to begin with. **Vivado's own
RQS_UTIL-211 pointed the same way and was equally wrong** — a QoR suggestion is
a hypothesis, not a verdict, and must be judged by a routed result.

**What the failure actually points at.** The partially-conflicted nets are not
in the GEMV clusters that dominate the congestion windows. They are dataflow
*control* nets inside the SLR2-pinned recurrent island:

```
.../gdn_recurrent_attention_island_0_U0/grp_gdn_tree_reduce_256_fu_1203/
    grp_gdn_tree_reduce_256_Pipeline_L16_fu_138_ap_start_reg
.../..._Pipeline_recur_island_alpha_product_fu_1232/
    flow_control_loop_pipe_sequential_init_U/ap_done_cache
.../..._Pipeline_recur_island_load_state_fu_1155/..._ap_loop_init_int
.../grp_gdn_tree_reduce_256_fu_1203_arr_0_ce0
```

`ap_start`, `ap_done_cache`, `ap_loop_init_int` and clock enables — handshake
and control, not datapath. Any future attempt should start there rather than on
memory binding.

**Timing was nearly acceptable and is not the reason it failed:** WNS **+0.003**
after physical optimisation, drifting to **-0.013** during routing; hold was the
weak axis (WHS -0.249, THS -657.248).

**Retained artifacts.** `diagnostics/checkpoints/iter59_post_place.dcp` (891 MB)
and `iter59_routed_error.dcp` (890 MB). Routing can be retried from placement
without repeating the ~4 h place, and the error checkpoint can be opened to
inspect the conflicted control nets directly.

**Checkpoint-hook bug found and fixed.** The Iter58b hook fired correctly but
wrote to a directory literally named `@C_IMPL_DIR@` under `impl_1`: that token
is substituted by the Makefile into the *cfg* file only, never inside a sourced
Tcl script. `check_f150_physical_islands.tcl` now derives the directory from
`[file dirname [file normalize [info script]]]`.

**Reverted.** All `bind_storage` FIFO bindings restored to `impl=bram`,
byte-identical to committed HEAD; only the pre-existing `reorder`/`state_pair`
URAM bindings remain. Fast decode gate re-passes. Nothing from Iter59 is
committed.

## Iter61 — LM-head logits streamed to the top level (RETAINED; on-card exact)

*Tested 2026-08-19/20. Evidence: **on-card**, routed, timing-closed.*

**Goal.** Emit the full 32,000-value logit vector from the on-chip LM head so
benchmark suites that need log-likelihoods can be scored on hardware, without
losing the exact-FP32 decode contract.

**Change.** `gemv32_store`'s existing LM-head branch (entered only when
`rows_per_ch == GDN_VOCAB / GEMV_CHANNELS`) additionally assembles the logit
vector into whole `Pack16` lines and pushes them into an `hls::stream`. The top
level drains that queue and writes to `workspace + GDN_WS_OFF_LOGITS`, beside
the token-id handoff Iter57 already performs. The fused strict argmax is
untouched, so the token path and the exact-match gate are unaffected; the
logits are purely additive.

The queue is depth 2048 (the producer completes before the sequential caller
drains it) and bound to URAM, costing 8 blocks of the 912 free and no BRAM.

**Why the queue rather than a memory port.** Two prior attempts put an AXI
write inside `gemv32_store`, which the island pblocks distribute across all
three SLRs:

| | approach | result |
|---|---|---|
| Iter58 | logits through the shared `out` pointer | `out` is one interface across every GEMV call, so HLS sized it for the vocabulary: **+1291 BRAM**, place_design DRC failed |
| Iter58b | dedicated `logits_out` AXI pointer | placed; routing never determined (node shutdown) |
| Iter59 | Iter58b + BRAM->URAM rebinding | congestion **level 7 in all four directions**, `route_design` refused at Phase 3.1 |
| **Iter61** | **stream to the top level** | **routed, timing met, on card** |

Filling a FIFO adds no memory port to the GEMV region at all.

**Resources (csynth) vs Iter57:**

| | BRAM | DSP | FF | LUT | URAM | cycles |
|---|---:|---:|---:|---:|---:|---:|
| Iter57 | 1728 | 3762 | 921,774 | 869,262 | 48 | 70,825,004 |
| **Iter61** | **1728** | 3778 | 923,732 | 873,019 | **56** | 70,827,077 |

BRAM identical, +8 URAM, +0.4% LUT, +0.003% cycles -- the smallest footprint
change of any variant tried.

**Implementation.** Post-place congestion **global 128/64/32/64**
(N/S/E/W), against Iter57's 128/32/32/64 and Iter59's 128 in all four. SLL
crossing 20,590 of 23,040 (89.4%), in line with Iter57's 20,713. Vivado
reported the SLL demand as "currently routable", which it did not for Iter59.
Route completed with **0 node overlaps and 0 failed nets**. Post-route
**WNS +0.003 ns, TNS 0.000, WHS +0.009, THS 0.000**; `dma_ip_axi_aclk_1` clean
over all 309,621 endpoints.

**On-card (U55C, 100 MHz).** 8-token and **64-token** decode both **exact**:
64/64 tokens bit-identical to the GPU golden, first divergence index -1.
**42.170227 ms/token median** (mean 42.167, min 42.121) against Iter57's
42.023540 -- **+0.35%** for the added logit export.

**Native.** Full 32-step gate passes. The stream output was additionally
verified value-by-value against the argmax loop's capture: **0 of 32,000
mismatched, worst absolute difference 0.000000e+00, on every step.** That check
mattered -- the standard gate compares `gdn_native_logits_debug`, which is
filled by the argmax path and would not have caught a fault in the new stream.

**Identity.** `gdn_model.cpp` 7591b223e7b8ac4e, `host.cpp` b47afba5e6791544,
`hw_f150_physical_islands.cfg` 4faf23ce87da7fb6 (unchanged),
`check_f150_physical_islands.tcl` d67c4ca4a13c9730.

**Build environment (both were blockers, both cost a full run):**
- Vivado peaked at **51 GB**. The `light` Slurm partition caps a job at 32 GB
  and killed an earlier attempt at 27 GB during Design Initialization. Use
  `--partition=build --mem=64G`; `run_hw_sbatch.sh` records the recipe.
- `build`-partition nodes `acclnode03/04/05` **cannot write to /home/yaoz0b** --
  probe jobs complete there and produce no files, so batch output vanishes and
  the job fails with exit 2. Pin to `harrier`.
- After a node relaunch the host presented **different hardware** (XRT 2022.2,
  and a U280 added). **XRT device index 0 is the U280; the U55C is index 2.**
  Loading the U55C image on index 0 fails `err = -22`. Confirm the index by BDF
  (`xbutil examine -d 0000:41:00.1`) before any on-card run.

**Verdict: RETAINED.** First design to emit full logits from hardware while
keeping the exact 64-token trajectory. Costs +0.35% per token.
