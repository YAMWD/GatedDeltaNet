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
