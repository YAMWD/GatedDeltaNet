# Frequency-Local Architecture Roadmap: 150 -> 200 -> 250 MHz

**Status (2026-09-05): the Iter68 campaign is CLOSED as stopped/inconclusive
and 250 MHz is no longer a target.** This document is kept as the record of
what Iter68 proposed and implemented (Iter68A–G source, gates, failure
branches). Outcome, from `optimization_log.md` (Iter68 verdict entry and the
Iter69 census): Iter68G never improved timing — every Vitis 2024.2 link,
Iter68G included, was implemented at 10.000 ns regardless of the requested
clock — and its image hung on the first call with all 96 state write-backs
landing 128 MiB below their stripe, a fault the static tooling could not
localise. Iter69 then showed that the retained Iter67c netlist fails 6.667 ns
by 1.594 ns of detoured wire from SLL column congestion (up to 211 % demand),
not from the control-path logic this redesign targeted. Nothing from Iter68 is
committed; [architecture.md](architecture.md) remains authoritative for the
retained Iter67c design. The HBM-aware frequency floor model in §2 (14.4 GB/s
per pseudo-channel versus 12.8 GB/s per port at 200 MHz and 16.0 GB/s at
250 MHz) is still the right way to cost any future clock increase.

This document is also a review contract for Claude Code. Reviewers should
challenge the unproven assumptions and stage ordering below, but must not
silently weaken the arithmetic contract, remove HBM ports or clusters, accept
an automatically scaled clock, or promote native/csynth evidence to a hardware
result.

## 1. Objective and non-negotiable constraints

The objective is to redesign physical communication, control, and feedback so
the current single-token decoder can run at exact DATA clocks of 150, 200, and
ultimately 250 MHz without losing evaluated model quality. A full link must be
attempted at every requested frequency; 250 MHz timing closure is the stretch
result, not an assumption.

The following remain fixed throughout this campaign:

- one XRT-visible `gdn_forward` kernel and the current kernel-argument order;
- the current host ABI, workspace offsets, packed checkpoint schemas, and
  full-logit output;
- 32 independent 512-bit HBM weight ports and 16 two-port GEMV clusters;
- packed BF16 weights, transient activations, convolution tails, and persistent
  recurrent state;
- native BF16-rounded multiplication followed by the existing FP32 dot-tree,
  bank, and final accumulation association;
- FP32 normalization, convolution, residual, recurrent arithmetic, and logits;
- strict lowest-index argmax behavior;
- MM2S/FIFO decoupling rather than direct AXI access from compute clusters;
- Vitis/Vivado 2024.2 and the U55C platform; and
- Slurm-only HLS, cosim, linking, and FPGA execution.

The single-kernel constraint is intentional for this campaign. It preserves
one `xrt::run`, the current token-level ABI, one owner for each HBM port, and
the already validated persistent-state lifetime. A multi-kernel-per-SLR design
connected with `--connectivity.slr` and AXIS could obtain tool-inserted register
slices, but it would introduce cross-CU launch/liveness semantics, a new XCLBIN
topology, and host orchestration changes. It is documented in Section 11 as a
separate architectural fallback requiring explicit approval, not a silent
timing repair.

Cycle count may grow modestly only within the milestone budgets below and only
when the higher achieved clock produces a strict production-TPOT improvement.
Reducing precision, ports, clusters, or evaluated quality is not an allowed
timing fallback. Weight compression is a separate numerical-contract campaign;
it is relevant to sub-10-ms latency but is not automatically authorized here.

## 2. Retained reference and success criteria

Iter67c is the immutable rollback and comparison point.

| Metric | Iter67c reference |
|---|---:|
| Git reference | `caf5125434b7` |
| DATA clock | exact 100 MHz |
| Effective cycles/token | 2.4099M |
| Kernel median | 24.099 ms |
| Production TPOT median | 24.221158 ms |
| Occupied CLB sites, SLR0/1/2 | 99.29% / 87.51% / 81.97% |
| SLL use, SLR0<->1 / SLR1<->2 | 84.76% / 56.91% |
| Dedicated SLL registers | 0 observed |
| Correctness | exact 64-token trajectory; accepted full-logit quality gate |

The frequency milestones are below. F200/F250 HLS targets are initial probes,
not assumed compile-to-link ratios; their final values are selected from the
preceding routed image.

| Milestone | HLS probe | Exact link target | Max effective cycles | Production TPOT acceptance | HBM-aware expectation |
|---|---:|---:|---:|---:|---:|
| F150 | 250 MHz; 200 fallback | 150 MHz | 2.53M | <=17.0 ms | ~16.19 ms |
| F200 | provisional 300 MHz | 200 MHz | 2.68M | <=13.5 ms | ~12.17 ms ideal; ~12.24 ms at estimated 88% HBM efficiency |
| F250 | provisional 333 MHz | 250 MHz | 2.84M | <=11.5 ms | >=10.33 ms Iter67c-derived schedule floor; ~11.10 ms at estimated 88% HBM efficiency |

Every accepted image must have:

- the exact requested DATA clock, with no automatic scaling;
- zero unplaced cells, node overlaps, failed nets, and unrouted nets;
- DATA, fixed 250 MHz DMA, and fixed 450 MHz HBM setup and hold slack >= 0;
- 32 masters, 16 clusters, and II=1 weight consumption in every cluster;
- an eight-token smoke test and a 64-token measurement excluding the seed;
- the accepted native trajectory and full-logit quality behavior; and
- strictly lower measured production TPOT than the last retained image.

### 2.1 HBM pseudo-channel ceiling

Frequency is a clean latency multiplier only while one HBM pseudo-channel can
supply one 512-bit Beat per kernel cycle. Each U55C HBM AXI pseudo-channel is
256 bits at 450 MHz, or **14.4 GB/s peak**. A 512-bit kernel port requests
`64 bytes * DATA frequency`:

| DATA clock | Per-port demand | Fraction of 14.4 GB/s peak |
|---:|---:|---:|
| 100 MHz | 6.4 GB/s | 44.4% |
| 150 MHz | 9.6 GB/s | 66.7% |
| 200 MHz | 12.8 GB/s | 88.9% |
| 225 MHz | 14.4 GB/s | 100% |
| 250 MHz | 16.0 GB/s | 111.1%; cannot be sustained |

Each port reads 1,268,224 weight Beats, or 81,166,336 bytes, per token. Its
absolute transfer floor is therefore 5.6366 ms at 14.4 GB/s and an estimated
6.4052 ms at 88% efficiency. Iter67c's derived exposed non-weight/stall
remainder is:

```text
2,409,900 - 1,268,224 = 1,141,676 cycles
```

At 250 MHz that remainder takes 4.5667 ms. Adding the measured 0.124 ms host
boundary gives:

- **10.33 ms production lower bound** at impossible-to-exceed peak HBM rate;
- approximately **11.10 ms production** at the explicitly estimated 88% rate;
- about **2.55M kernel cycles** at the absolute HBM floor; and
- about **2.74M kernel cycles** at 88% efficiency.

This additive estimate assumes the frequency campaign preserves Iter67c's
exposed phase structure; it is not a theorem about a future overlap redesign.
Under that fixed scope, the uncompressed BF16 F250 target of <=11.5 ms remains
possible but is borderline, while the former <=10.0 ms stretch is below the
derived schedule floor and is removed. A sub-10-ms target needs fewer weight
bytes, materially more GEMV/non-GEMV overlap, or more physical memory
bandwidth—all separate architecture/numerical contracts. F200 to F250 is
expected to improve latency by only about 9% at realistic HBM efficiency, not
the 20% implied by pure clock scaling.

Before the F200 full-model link, run the existing standalone 32-port
[GEMV tile microbenchmark](../microbench/gemv_tile/README.md) at exact 200 and
250 MHz. Record per-port and aggregate GB/s, burst efficiency, achieved clocks,
and any pseudo-channel imbalance.
Replace the 88% estimate with those measurements before accepting or revising
the F200/F250 latency budgets.

## 3. Why Iter67c cannot simply be relinked at 250 MHz

Iter67c closes 100 MHz, but it was optimized primarily for routability at that
clock. Its remaining problem is physical distribution, not a single slow BF16
operator.

### 3.1 Measured path families, not one global-control family

The Iter67c post-place top-200 DATA-path census classifies all 200 starts into
four disjoint families:

| Source/path family | Top-200 paths | Does Iter68A directly remove it? |
|---|---:|---|
| top `ap_CS_fsm_reg[]` to `m_axi` load/store FIFO address or enable | 85 | yes, if persistent services synthesize as intended |
| adapter/FIFO `dout_vld_reg` to downstream FIFO CE | 48 | **no** |
| `m_axi` store-unit `full_n_reg` to `WEBWE`/write-request control | 59 | **no** |
| per-call `ap_start` control | 8 | yes, if no lower-level start cone is recreated |

The worst post-place path is 9.225 ns: 0.376 ns logic and 8.849 ns route
(95.9%), with five LUT levels and a combinational SLR1 -> SLR2 -> SLR0 path.
This supports the locality diagnosis, but it also shows that persistent launch
control addresses only about half the top-200 family. Adapter-internal FIFO and
store-control paths require their own netlist gates and, if they survive, their
own small experiments.

Evidence is in the retained
[top-200 post-place report](../diagnostics/iter68_day1_frequency_census/post_place_census/post_place_data_top200.rpt).

For the `dout_vld -> CE` family, the preferred repair is a registered
adapter-to-actor handshake or skid boundary, not global fanout replication. For
the store `full_n` family, test an explicitly selected
`config_interface -m_axi_buffer_impl` and a smaller write-buffer/outstanding
configuration on the measured writable ports. Retain either only if AXI II,
burst efficiency, and cycle count remain unchanged while the path family and
regional fanout shrink.

### 3.2 The 3/6/7 topology is a CLB-for-SLL trade

Iter67c has one long 16-cluster activation ripple and a 4/6/6 collector split.
Its control and activation paths do not form independently placeable SLR-local
units, and SLR0 is already at 99.29% occupied CLB sites. Moving one more cluster
out of SLR0 creates real CLB relief.

However, all HBM interfaces enter through SLR0. Every cluster port placed above
SLR0 carries roughly one 512-bit stream across an SLR boundary. The current and
estimated topology costs are:

| Boundary metric | Iter67c routed | 3/6/7 estimate before placement |
|---|---:|---:|
| SLR0 -> SLR1 | 12,537, corresponding to 24 remote ports | approximately 13,600 for 26 remote ports |
| SLR1 <-> SLR0 total | 19,529 / 23,040 = 84.76% | approximately 20,700 = ~90% |
| SLR1 -> SLR2 | 6,775 for 12 remote ports | approximately 7,900 for 14 remote ports |
| LAGUNA TX/RX registers | 0 | must be greater than zero |

The HBM0 service also adds wide response, write-data, and logit crossings. The
3/6/7 source is therefore a candidate topology, not assumed port locality. Its
post-place topology gate is <=85% aggregate SLR0<->SLR1 SLL use, no local SLL
column over 100%, and physically used LAGUNA TX/RX registers on every intended
boundary. If it fails, test **4/6/6 while moving sequencer/store/non-port logic
out of SLR0** before moving more port-bound clusters upward.

The measured baseline is available in the retained
[routed per-SLR report](../diagnostics/iter67c_argmax_ii_hw/gdn_final_qor/utilization_slr.rpt).

### 3.3 HLS relays are not physical crossing registers

Earlier HLS collector relays improved routability, but the routed image used no
dedicated SLL TX/RX registers. A `destination.write(source.read())` relay can
still create a combinational `dout_vld -> CE` handshake, and a C++ function or
FIFO boundary does not prove that reverse backpressure was registered.

Because Iter57/Iter67c already measured zero LAGUNA registers, a two-entry RTL
skid buffer is now the **default** crossing mechanism for F150, not a fallback.
It must register both data/valid and ready/backpressure, expose a C simulation
model for cosim, preserve II=1, and place at least one inspected TX/RX register
at each physical boundary.

### 3.4 HLS target frequency is a measured variable

The only retained compile/link point is HLS 150 MHz -> exact link 100 MHz,
roughly a 0.67 link/HLS ratio. It does not establish the provisional
300/200 or 333/250 ratios. Because the worst current paths are routing-dominant,
an unnecessarily aggressive HLS target may add pipeline FFs and control to an
already 99.29%-occupied SLR0 without shortening the physical path.

The current HLS250 result must be compared with Iter67c's 1,002,888 FF and
895,268 LUT, its <=2.53M F150 cycle budget, every II=1 loop, and the GEMV
accumulator's required fadd latency <=8. If HLS250 adds unplaceable logic or
breaks those schedule gates, synthesize the same architecture at HLS200 for the
150 MHz link. Select later HLS targets from the F150/F200 routed timing and
resource response rather than retaining a fixed ratio by convention.

### 3.5 Feedback and serial tail paths become visible above 150 MHz

The five-phase recurrent schedule is correct and II=1 at the retained HLS
target, but its per-head FP32 feedback distance may be insufficient at the
selected F250 HLS target. The single argmax recurrence and some small-operator
reductions may also become the critical paths after global control is removed.
These are later, measured repairs; changing them before control locality is
proven would confound the diagnosis.

### 3.6 HBM0 is a shared physical service

Dense port 0, auxiliary weights, workspace ingress/egress, convolution tails,
logits, and token output share one AXI bundle. Letting several top-level actors
touch it would recreate arbitration and control cones. The redesign therefore
gives HBM0 one owner and a bounded request protocol.

## 4. Iter68A changes already implemented

The following changes exist in the current working source. They have passed a
native fast full-logit gate, but have not yet passed integrated HLS, RTL cosim,
placement, routing, timing, or on-card measurement. The evidence qualifier is
part of every statement in this section.

### 4.1 Fixed 97-command protocol

All large projections are represented by one `GDNGemvCommand`, an
`ap_uint<64>` with this fixed layout:

| Bits | Field | Width |
|---:|---|---:|
| 31:0 | packed-weight Beat offset | 32 |
| 40:32 | activation `k_packs` | 9 |
| 50:41 | output rows per channel | 10 |
| 55:51 | layer index | 5 |
| 58:56 | operation mode | 3 |
| 59 | final-command flag | 1 |
| 63:60 | reserved, always zero | 4 |

One token has exactly 97 commands: QKVG, output projection, gate/up, and down
projection for each of 24 layers, followed by the LM head. The independent
calculator [scripts/gdn_gemv_schedule.py](../../scripts/gdn_gemv_schedule.py)
checks command encoding, section offsets, the single final flag, and the final
per-shard boundary of 1,366,528 BF16 Beats.

The intent is that actors read their dimensions locally from short command
FIFOs. A top-level scalar no longer fans directly into every reader, FIFO,
cluster, and collector.

### 4.2 Three bounded top-level services

The synthesized branch of `gdn_forward` is organized around three long-lived,
bounded services:

```text
                                  SLR1
                    +-----------------------------+
embedding/aux <---->| gdn_token_sequencer         |
small operators     | - owns activation lifetime  |
                    | - emits 97 GEMV commands     |
                    | - has no direct m_axi        |
                    +----------+------------------+
                               |
                    commands + activations
                               v
                    +-----------------------------+
                    | gdn_gemv_service_persistent |
                    | - 32 readers / 16 clusters  |
                    | - 3/6/7 local islands       |
                    | - recurrence in SLR2        |
                    | - final collector in SLR1   |
                    +-----------------------------+
                               |
                          logits/token/data
                               v
                SLR0  +---------------------------+
HBM0 <-------------->| gdn_port0_service         |
                      | sole owner of shared AXI0 |
                      +---------------------------+
```

`gdn_token_sequencer` owns transient activations and the fixed 24-layer small
operator schedule. It communicates with memory only through streams.

`gdn_port0_service` is the sole owner of auxiliary weights, workspace, and
dense weight port 0. Its transaction order is statically bounded at 173
requests, 106,648 read-response Beats, 13,825 write-data Beats, and 2,000
full-logit Beats per token.

`gdn_gemv_service_persistent` owns the other dense ports, the 16 clusters,
collection, QKVG/convolution/recurrent handling, and full-logit generation.

The three names describe architectural roles. HLS must still prove that the
result is one concurrent graph rather than a serialized schedule or replicated
engines.

### 4.3 Persistent GEMV actors

Each MM2S reader, two-port cluster, activation drain, local collector, final
collector, store adapter, and recurrent service loops over the complete
97-command schedule. The actor is intended to start once per token; it consumes
per-command dimensions from a local FIFO rather than receiving 97 top-level
starts.

Three island-local command distributors bound command fanout. The synthesized
path has no direct call to the legacy per-command `gdn_gemv` wrapper.

This is an implementation intent, not yet netlist evidence. Integrated HLS
must confirm that subfunction control did not recreate a per-command `ap_start`
cone below the persistent wrapper.

### 4.4 3/6/7 candidate GEMV islands

The current source trades one SLR0 cluster for a larger upper-SLR island:

| Island | Intended SLR | HBM ports | Clusters | Logical channels |
|---|---:|---:|---:|---:|
| 0 | SLR0 | 0-5 | 3 | 0-5 |
| 1 | SLR1 | 6-17 | 6 | 6-17 |
| 2 | SLR2 | 18-31 | 7 | 18-31 |

Each island has its own activation ripple, reader FIFOs, cluster-result FIFOs,
command distributor, drain, and local collector. The final SLR1 collector
drains 6/12/14 logical channels in the original order, so the packed weight
format and FP32 result association are unchanged.

The motivation is CLB redistribution, not HBM-port locality: the HBM interfaces
are physically rooted in SLR0. SLR0 receives only three clusters because it
also owns the shared HBM0 service and has the least CLB whitespace. SLR2
receives seven because it has more CLB headroom and owns recurrent-state
traffic. This incurs the SLL cost quantified in Section 3.2, so a future
placement report must decide whether 3/6/7 survives or 4/6/6 with non-port logic
moved upward is the better topology.

### 4.5 Explicit two-stage crossing actors

Two non-inlined relay actors were added on each intentional inter-SLR data
path:

- activation and command traffic from SLR1 to islands 0 and 2;
- island 0 and 2 result traffic back to the SLR1 final collector;
- Q/K/V plus recurrent context from SLR1 to the recurrent service in SLR2;
- attention results from SLR2 back to SLR1;
- HBM0 requests and write data from SLR1 to SLR0;
- HBM0 read responses from SLR0 to SLR1; and
- full logits from SLR1 to the HBM0 writer in SLR0.

No data path is intended to cross directly from SLR0 to SLR2; it must relay
through SLR1.

These C++ actors are a source checkpoint for HLS legality, not the accepted
F150 crossing. HLS or Vivado may merge them, place both stages on the same side
of the boundary, or leave a long combinational ready path. Before F150, replace
the crossing implementation with a two-entry RTL skid-buffer black box that
registers forward data/valid and reverse ready/backpressure. Post-synthesis and
post-place checks must prove II=1, preservation, boundary placement, and a
nonzero LAGUNA TX/RX count per used boundary.

### 4.6 Arithmetic and ABI deliberately unchanged

Iter68A has not changed the model arithmetic, product rounding, accumulation
association, state representation, checkpoint layout, workspace offsets,
argument order, vocabulary output, or strict argmax semantics. The top-level
auxiliary pointer is expressed as 512-bit Beats in the synthesis interface so
the HBM0 service sees one raw port; native code casts the same storage back to
FP32 where required.

### 4.7 Verification completed so far

| Evidence | Result |
|---|---|
| Command-only native gate, job 3140 | pass |
| 3/6/7 island native gate, job 3141 | pass |
| Service integration compile/gate, jobs 3153/3154 | pass |
| Persistent graph native compiles, jobs 3168/3169 | pass |
| Fast trajectory/full-logit gate, job 3173 | six-token trajectory, 160,000 logits, first divergence -1, zero tolerance/exact-reference/argmax mismatches |
| Integrated HLS250 | submitted from immutable snapshot; result pending at document time |
| RTL cosim | not run |
| Placement/routing/timing | not run |
| On-card correctness/performance | not run |

The current immutable HLS snapshot is
`diagnostics/iter68_persistent_services/source_snapshot.tar`, SHA-256
`ee55a02924dbe12d22dbcad49898f310505410be4517f17a31b374e66a5931c2`.
The snapshotted kernel source SHA-256 is
`230daf0c9ecf31bc87c24d3d1751919cc4532d4030a42c1b7f348dda03ebb5aa`.

The native/csim build intentionally executes the retained sequential reference
under `#ifndef __SYNTHESIS__`; ordinary C simulation cannot execute the cyclic
top-level stream graph concurrently. Therefore native parity proves arithmetic
and schedule construction, but it does **not** prove the synthesized service
graph. RTL cosim is mandatory for this redesign.

### 4.8 Source map for review

The principal review points are:

| File / symbol | Role | Current evidence |
|---|---|---|
| `gdn_model.h`: `GDNGemvCommand` | shared 64-bit command type and fixed command count | native compile |
| `gdn_model.cpp`: `gdn_make_gemv_command` and accessors | command encoding/decoding | calculator + native gate |
| `gdn_model.cpp`: `gdn_persistent_*` actors | 97-command MM2S, cluster, drain, collector, store, and recurrence wrappers | source/native only |
| `gdn_model.cpp`: `gdn_gemv_island{0,1,2}_service` | 3/6/7 local graphs | source/native only |
| `gdn_model.cpp`: `gdn_fixed_boundary_relay` and typed boundary relays | provisional two-stage inter-SLR cuts | source only; replace with RTL skid buffers before F150 |
| `gdn_model.cpp`: `gdn_port0_service` | sole HBM0 AXI owner | source/native only |
| `gdn_model.cpp`: `gdn_token_sequencer` | 24-layer small-op schedule and service requests | native arithmetic gate only |
| `gdn_model.cpp`: `gdn_gemv_service_persistent` | persistent GEMV/recurrent graph | source/native only |
| `gdn_model.cpp`: synthesized branch of `gdn_forward` | top bounded dataflow connection | HLS/cosim pending |
| `packed_bf16_one_layer_test.cpp` | one-layer/eight-head liveness harness | must be made production-faithful for Iter68 cosim |
| `scripts/gdn_gemv_schedule.py` | independent 97-command and shard-offset calculator | passes |
| `doc/optimization_log.md` | chronological evidence and failed-wrapper record | updated through HLS submission |

No Iter68 physical Tcl is considered final yet. Stable hierarchy selectors,
soft SLR assignments, crossing-register properties, and post-route reports must
be derived from the successful Iter68 HLS netlist rather than guessed from
Iter67c instance names.

## 5. Immediate gate: integrated HLS at 250 MHz

The first long gate is integrated csynth with a 4.0 ns target. It is not a
hardware-frequency claim. It answers whether the service graph is legal and
whether the first 150 MHz image is worth linking.

### 5.1 Required HLS evidence

The report and generated RTL must show:

1. exactly 32 dense AXI masters and one shared owner for HBM0;
2. exactly 16 two-port clusters partitioned 3/6/7;
3. one persistent GEMV graph, not one clone per operation mode or buffer type;
4. all MM2S response and cluster weight loops at II=1;
5. one BF16 Beat consumed per weight port per cycle in steady state;
6. no top-level FSM or persistent-wrapper `ap_start` path directly controlling
   all cluster FIFO addresses, enables, or CEs;
7. bounded 97-command loops and exact transaction counts;
8. no unexpected AXI master, GEMV clone, FP32 multiplier, or arithmetic-order
   change;
9. the same recurrent arithmetic II and state transaction counts as Iter67c;
10. estimated timing consistent with the 250 MHz HLS target, with any misses
    identified by path family rather than only top-level WNS; and
11. resources and hierarchy split by service, island, cluster, relay, store,
    recurrence, and HBM0 service.

For job 3175 specifically, compare total and per-hierarchy FF/LUT against
Iter67c's 1,002,888 FF / 895,268 LUT, verify effective cycles <=2.53M, preserve
every required II=1 loop, and require the eight-context GEMV accumulator's fadd
latency <=8. HLS250 is retained for F150 only if the extra pipeline structure
has a measured schedule benefit and a plausible regional placement budget.
Even a passing job 3175 XO is diagnostic rather than the final F150 XO: replacing
the provisional HLS crossing relays with RTL skid buffers requires a fresh
integration synthesis before linking.

There is no global LUT-percentage rejection gate. Physical feasibility is
judged from cluster density, regional CLB demand, control fanout, and later
placement. However, engine replication, a material per-cluster LUT/FF increase,
or SLR0 logic growth without a compensating removal is a structural failure
that must be understood before linking.

### 5.2 Required post-synthesis and post-place evidence

HLS schedule reports cannot prove that the path families in Section 3.1 were
removed. The generated netlist and first placement must additionally record:

- `report_high_fanout_nets`, grouped by service, island, adapter, and clock
  region;
- setup paths starting at every surviving `*ap_CS_fsm_reg*` and `*ap_start*`;
- the `dout_vld_reg -> FIFO CE` and store-unit
  `full_n_reg -> WEBWE/fifo_wreq` path families;
- actual LAGUNA TX/RX register counts for every intended SLR crossing;
- aggregate and per-column SLL demand on both boundaries;
- per-clock WNS/WHS and endpoint counts for DATA, DMA, and `hbm_aclk`; and
- per-SLR utilization for every service/island wrapper.

The Iter67c reference for `hbm_aclk` is +0.084 ns WNS / +0.010 ns WHS over
269,988 setup and hold endpoints. Every link re-places that clock, so it must
pass independently at post-place and post-route; DATA-clock success does not
cover it. The baseline is in the retained
[post-route timing report](../diagnostics/iter67c_argmax_ii_hw/gdn_final_qor/timing_summary.rpt).

If either adapter-internal family remains among the leading paths, run one
controlled experiment at a time: first a registered adapter-to-actor skid
boundary for `dout_vld -> CE`; then, only for the measured writable ports, an
explicit `config_interface -m_axi_buffer_impl` or smaller
burst/outstanding/write-buffer setting for `full_n`. The gate is unchanged II,
burst throughput, transaction count, and effective cycles plus removal or
material shortening of the target path family.

### 5.3 HLS failure handling

If HLS fails because of language, scheduling, or dataflow legality, fix that
specific issue while preserving the service boundaries. Do not fall back to
97 top-level GEMV launches merely to get a report.

If HLS reports deadlock-prone cyclic dependencies, inspect the fixed-count
producer/consumer equations first. Increase a FIFO only when the report or
cosim proves a bounded burst mismatch; depth is not a substitute for a missing
dependency edge or an unregistered reverse path.

If a supposed persistent wrapper still emits a per-command control handshake,
move the inner loop into the actor or convert that subfunction to a free-running
pipeline. Do not globally inline the full cluster graph without measuring the
resulting control fanout.

If HLS clones the shared engine, trace the incompatible array/stream types or
compile-time caller specialization. All operation modes must use one common
transport type and one physical service.

If the 250 MHz HLS target preserves functionality but adds an unplaceable
FF/LUT delta, breaks an II=1 loop, makes the accumulator fadd latency exceed
eight contexts, or exceeds 2.53M estimated effective cycles, re-run the same
architecture at HLS200 for the F150 link. This is a compile-target correction,
not a return to the old architecture.

## 6. RTL cosim and structural proof before linking

After HLS passes, run a production-faithful one-layer/all-eight-head RTL cosim
with deterministic nonzero data, AXI response latency, and randomized stream
backpressure. This is not optional because the native branch does not execute
the synthesized topology.

Every persistent actor must process at least two consecutive commands with
different shapes in one cosim run. The minimum sequence is QKVG -> output plus
gate/up -> down, covering both `k_packs=64/176` and
`rows_per_channel=64/256/352`; include LM head when the bounded harness can do
so without making the gate impractical. Restarting the RTL between commands
does not test the persistent protocol.

The cosim gate must verify:

- all commands retire in order without deadlock;
- every island receives the same 97 command sequence;
- activation and result Beat counts match each command;
- four QKVG state readers and both recurrent islands complete all eight heads;
- convolution-tail and recurrent-state writes complete;
- all 32,000 FP32 logits are emitted and their checksum/reference comparison
  passes;
- strict argmax returns the expected token;
- no actor terminates early while another remains blocked; and
- total cycles do not exceed the previous bounded one-layer liveness result
  except for the explicitly added relay latency.

Separately inspect the synthesized control netlist. The structural gate is not
"the hierarchy names exist"; it is that the old global path family no longer
reaches cluster-local FIFO addresses/enables or AXI response CEs.

Cosim proves bounded transaction balance, functional RTL, and liveness under
modeled stalls. It does **not** prove HBM pseudo-channel bandwidth, actual SLL
placement, LAGUNA usage, clock timing, or card/runtime behavior. The hardware
smoke test must use a bounded `xrt::run::wait(timeout)` path so a residual relay
count deadlock fails visibly and yields diagnostics instead of blocking the
Slurm allocation indefinitely.

## 7. F150: first physical proof at exact 150 MHz

F150 uses the 250 MHz HLS output if it passes Section 5.1, otherwise the HLS200
fallback, and links at exactly 150 MHz. This is the first test of the locality
redesign and should not include the later C-slow recurrence or banked-argmax
changes unless the 150 MHz timing report proves they are already required.

### 7.1 Physical constraints

Re-derive hierarchy selectors from the new netlist and make every selector
exact-count checked. Do not reuse synthesis-generated Iter67c names blindly.

Use soft assignments as the starting point:

- token sequencer, final collector/store, and central command frontend: SLR1;
- HBM0 service and island 0: SLR0;
- island 1: SLR1;
- island 2 and recurrent wrapper: SLR2.

Do not restore broad hard pblocks. Constrain an individual clock region only
when a measured congestion or timing window identifies the need.

Instantiate the two-entry RTL skid buffer on each inter-SLR stream and select
only its boundary register for `USER_SLL_REG`. Then verify actual LAGUNA TX/RX
placement and report the number of used dedicated SLL registers per boundary.
Property presence without physical placement, or a zero TX/RX count on a used
boundary, is a failed gate.

The initial placement is a topology decision point. Retain 3/6/7 only if the
SLR0<->SLR1 aggregate is <=85%, no local SLL column exceeds 100%, and SLR0 CLB
relief is material. Otherwise regenerate the same persistent-service design as
4/6/6 and move sequencer/store/non-port logic from SLR0 before proceeding to a
full route.

Retain the proven DMA/reset repairs. Begin implementation with
`SSI_SpreadLogic_high`, `AlternateCLBRouting`, pre-route physical optimization,
and post-route `AggressiveExplore`; do not launch an untargeted directive
sweep.

### 7.2 F150 diagnostics and acceptance

Collect:

- route legality and failed/unrouted/overlap counts;
- per-clock setup and hold plus bus-skew reports;
- the top timing paths grouped by control, crossing, arithmetic, and AXI path
  family;
- per-SLR LUT, FF, occupied CLB, BRAM, URAM, and DSP use;
- global/long/short congestion tables and the worst windows;
- SLL column demand and actual TX/RX register placement;
- cluster-local command, enable, reset, multiplier, and adder fanout; and
- QoR suggestions, treated as evidence rather than automatic instructions.

If F150 routes and closes timing, run the eight-token smoke and 64-token test.
Retain only if effective cycles are <=2.53M, production TPOT is <=17.0 ms,
quality passes, and it strictly improves Iter67c.

## 8. F200: local retiming from the routed F150 checkpoint

F200 starts only after the F150 checkpoint has been characterized. Re-synthesize
at a target selected from F150's measured compile/link response; 300 MHz is the
initial probe, not a fixed ratio. Before the full-model link, route and run the
32-port GEMV microbenchmark at exact 200 and 250 MHz to replace the HBM
efficiency estimate. Then link the full model at exactly 200 MHz. Do not guess
the next critical path from the 100 MHz design; increasing the clock changes
which path family is dominant.

Apply one measured repair at a time in this order:

1. **Missing SLR boundary.** Add or preserve the required crossing/skid
   register and prove physical placement.
2. **Island-local control.** Split only the measured command/enable/reset cone
   into cluster- or clock-region-local distributors. Avoid very low fanout
   limits that inflate SLR0 CLB use.
3. **AXI-adapter handshake.** Register the measured `dout_vld -> CE` boundary
   or reduce the measured store write buffer/full-control cone, preserving
   bursts and II=1.
4. **HBM0-service control.** Pipeline request decode/address generation while
   keeping one AXI owner and fixed ordering.
5. **Collector/store path.** Add local registers between collection, mode
   decode, and store adapters without changing output order.
6. **Arithmetic pipeline.** Increase operator latency only for the measured
   BF16, FP32 tree, norm, convolution, residual, or SwiGLU path. Preserve the
   expression and association.

Re-run native/cosim when a source-level pipeline or protocol changes. A pure
physical constraint change still requires route/timing and on-card gates.

F200 acceptance is <=2.68M effective cycles, <=13.5 ms production TPOT, exact
200 MHz, clean timing/route, and unchanged quality.

## 9. F250: high-frequency feedback architecture

F250 uses an HLS target selected from F200's measured response; 333 MHz is the
initial scheduling probe, not an assumed 0.75 link/HLS ratio. It links at an
exact 250 MHz and keeps the locality architecture from F150/F200, adding only
the high-frequency mechanisms proven necessary by the HLS schedule or 200 MHz
routed timing census.

### 9.1 GEMV row-context fallback

Keep the current eight row contexts if the FP32 accumulation recurrence remains
II=1 at the selected F250 HLS target. If it does not:

1. use 16 row contexts so the same accumulator bank is revisited only after a
   longer pipeline distance;
2. repack device shards in 16-row-interleaved groups;
3. preserve the current per-product rounding and four-bank final association;
4. internally pad each 1000-row LM-head channel stripe to 1008 rows; and
5. discard padded outputs before logit emission and argmax.

This is a throughput-retiming change, not additional arithmetic parallelism.
It must retain one Beat per port per cycle and must not clone the GEMV engine.

### 9.2 C-slow recurrent feedback

The first high-frequency recurrence candidate interleaves two independent
heads across one physical pipeline:

```text
slot:  0 1 2 3 4 5 6 7 8 9
work: A0 A1 A2 A3 B0 B1 B2 B3 - -
```

Each head's feedback distance becomes ten cycles while two heads still consume
the same ten slots as the current two five-phase heads. The operation order
within each head, state ownership, FP32 association, and total useful recurrence
work remain unchanged.

Bind the recurrent feedback add to latency <=10 cycles. If the selected F250
HLS implementation still requires more than ten cycles, use three heads over a
12-slot schedule. Do not reassociate the recurrent sum or narrow it to BF16.

The gate is not just II=1: total recurrence cycles per two or three heads must
remain within the F250 token budget and cosim must prove state/head ordering.

### 9.3 Banked strict argmax

Retain the current argmax if it remains II=1 and timing-clean. Otherwise use
four temporal banks: each bank compares every fourth logit, then a final merge
selects the maximum with the strict lowest-index tie rule. Full FP32 logits
must still be emitted. The merge pipeline must not change tie behavior.

### 9.4 Clock-region locality

At F250, every reported critical path must either remain within one clock
region or terminate at a registered SLR/clock-region boundary. Soft island SLR
assignment alone is insufficient. Use small measured sub-pblocks only for an
actor and its local memories when placement repeatedly splits that actor across
clock regions; never weld the complete design to generated leaf names.

F250 acceptance is <=2.84M effective cycles, <=11.5 ms production TPOT, exact
250 MHz with all clocks clean, and unchanged quality. This is expected to be
only about a 9% latency step beyond F200 because weight traffic saturates each
pseudo-channel. Sub-10-ms performance is not an uncompressed F250 acceptance
target; it requires a separate, quality-qualified compression result.

## 10. Fixed failure-repair decision tree

Use the first matching category. Do not skip directly to a large floorplan or
precision change.

| Observed failure | Required diagnosis | First permitted repair | Forbidden shortcut |
|---|---|---|---|
| HLS duplicates GEMV | hierarchy/resources/caller types | unify transport and service call signature | accept clones because device-wide LUT appears available |
| HLS250 over-pipelines F150 | FF/LUT delta, II, cycles, fadd latency | synthesize identical source at HLS200 | force HLS250 because a fixed ratio was planned |
| HLS/dataflow deadlock | command and Beat balance; blocking edge | fix protocol or one proven FIFO bound | blindly deepen every FIFO |
| Cosim stalls | last successful transaction per actor | repair missing command/data dependency | skip cosim and discover it in hardware |
| SLR relay optimized away | netlist and physical register placement | use/preserve the default RTL skid buffer | trust an HLS relay or `USER_SLL_REG` text property alone |
| 3/6/7 exceeds 85% lower SLL | per-boundary/column SLL and SLR0 CLB delta | restore 4/6/6; move non-port logic upward | add more remote clusters or hard-pblock all clusters |
| Unplaced cells/CLB exhaustion | per-SLR hierarchy utilization | move/split only measured actor; reduce control replication | broad all-cluster pblocks |
| SLL column hotspot | crossing endpoints and column demand | relocate boundary register or route through SLR1 | direct SLR0<->SLR2 crossing |
| Long island-local CE/control | fanout loads by clock region | bounded local distributor; moderate fanout replication | global `FORCE_MAX_FANOUT=32` |
| AXI `dout_vld/full_n` path remains | adapter timing/fanout plus burst counters | registered boundary or one measured buffer setting | assume persistent `ap_start` removed adapter paths |
| Arithmetic setup path | operator and feedback schedule | pipeline same expression; contexts/C-slow as specified | change accumulation order or precision |
| DMA clock failure | exact path family on fixed 250 MHz clock | retain/retune proven local DMA repair | lower DATA clock and assume DMA improves |
| HBM clock failure | `hbm_aclk` WNS/WHS over all endpoints | repair measured shell/adapter placement path | accept because DATA and DMA clocks pass |
| F250 stalls at HBM ceiling | measured per-PC GB/s versus byte floor | report bandwidth-limited result; separately evaluate compression | treat stalls as failed II or alter arithmetic silently |
| DATA clock auto-scales | requested vs achieved clock report | reject and repair timing | count the image as a frequency milestone |

Regional congestion level is diagnostic, not a routability guarantee. A level
5 design can still fail from local CLB exhaustion, SLL demand, overlaps, or one
unroutable control cone; a higher early level may improve after placement and
physical optimization.

## 11. Verification matrix

| Gate | Iter68A source | F150 | F200 | F250 |
|---|---:|---:|---:|---:|
| Command calculator | required | required | required | required |
| Native fast/full trajectory | required | required after source change | required after source change | required after source change |
| Full-logit quality comparison | required | required | required | required |
| Integrated csynth | HLS250, HLS200 fallback | selected F150 target | selected from F150 evidence | selected from F200 evidence |
| One-layer/eight-head randomized cosim | required | required if protocol changes | required if protocol changes | required |
| Global-control structural audit | required | required | required | required |
| AXI `dout_vld/full_n` netlist census | required | required | required | required |
| LAGUNA TX/RX count per used boundary | no | >0 after placement | >0 | >0 |
| 32-port HBM microbenchmark | no | no | exact 200 and 250 before full link | measured model retained |
| Exact-clock full link | none | 150 MHz | 200 MHz | 250 MHz |
| Bounded-timeout eight/64-token on-card run | none | required | required | required |
| Complete 62-document WikiText gate | no | optional | optional | final fastest image |

The final fastest retained image must rerun the complete 62-document WikiText
teacher-forced gate and remain within the current accepted perplexity bound.

**Multi-kernel alternative, outside the current gate.** If a single-kernel
netlist still cannot produce registered SLR handshakes after RTL skid buffers
and the measured failure is specifically cross-SLR control—not CLB, SLL, HBM,
or arithmetic timing—prepare a separate feasibility plan for three SLR-bound
kernels connected by AXIS and `--connectivity.slr`. It may gain tool-inserted
register slices, but it changes launch/liveness, host orchestration, and XCLBIN
topology. Do not implement it without explicit approval and new ABI/correctness
gates.

## 12. Build and evidence workflow

Use only the production entry point, with frequency overrides:

```bash
make -C c_impl run_hw HLS_FREQ=<selected_hls_mhz> LINK_FREQ=<150|200|250>
```

The command is the inner workflow and must run inside Slurm. Do not add
iteration-specific Make targets or permanent launch scripts. Use immutable
source/config snapshots so source work cannot change a running build.

Before the first F150 on-card job:

- key `RUN_HW_DIR` and `ONCARD_DIR` by the Slurm job ID, for example
  `diagnostics/<tag>-${SLURM_JOB_ID}`, so equal-frequency runs cannot overwrite
  one another; and
- add a bounded `xrt::run::wait(timeout)` path that records a timeout verdict,
  resets/releases the run safely, and preserves the last completed protocol
  counters when available.

Every long job must have shared Slurm stdout and detailed tool logs under
`c_impl/diagnostics/<tag>/`. After a bounded launch-sentry window, report the
job ID, ETA, and clickable live-log path and end the chat turn; do not poll a
multi-hour HLS or implementation job.

For every candidate, archive or record hashes for:

- Git reference and dirty source snapshot;
- `gdn_model.cpp`/`.h`, Makefile, link config, and Tcl hooks;
- XO and XCLBIN when generated;
- route/timing/congestion/utilization reports;
- native/GPU reference inputs; and
- on-card JSON and first-divergence result.

Record every success, failure, stopped run, wrapper failure, or inconclusive
experiment in [optimization_log.md](optimization_log.md) before starting the
next named iteration. Revert negative or neutral implementation/config changes.
Commit only after a real on-card TPOT improvement, then update
`architecture.md`, the relevant block documents, and
`cycle_optimization_roadmap.md` in the same positive commit series.

## 13. Explicit review questions for Claude Code

Claude Code should review the current source and reports against these
questions, in order:

1. Does the synthesized branch elaborate as one bounded concurrent graph, or
   do nested function calls reintroduce per-command start propagation?
2. Are all 97 command and Beat-count equations balanced for every operation
   mode, including LM-head tail handling and QKVG state traffic?
3. Can any cyclic blocking sequence arise between the sequencer, HBM0 service,
   GEMV store, recurrence, or full-logit writer under delayed AXI responses?
4. Did HLS create exactly one physical GEMV service and 16 clusters, or
   specialize/clone it because of differing callers, buffers, or modes?
5. Are both the forward and reverse handshake paths actually registered at
   every SLR boundary by the default RTL skid buffer, with nonzero physical
   LAGUNA TX/RX placement?
6. Does the 3/6/7 split reduce SLR0 occupied CLB enough to justify its estimated
   ~90% lower-boundary SLL use, or should 4/6/6 be retained while non-port logic
   moves out of SLR0?
7. Is island 2 plus recurrence physically feasible in SLR2 without a local
   SLL-column or BRAM/URAM hotspot?
8. Are command distributors truly island-local, and what are their post-synth
   and post-place fanouts by clock region?
9. Did moving all HBM0 traffic behind one service introduce avoidable bubbles
   or an oversized address/decode mux on its critical path?
10. Do `dout_vld -> CE` and store `full_n -> WEBWE/fifo_wreq` remain leading
    path families after persistent control is synthesized, and which single
    adapter-level change addresses each?
11. Does job 3175's HLS250 FF/LUT delta, cycle estimate, II, and fadd latency
    justify using it for F150, or is HLS200 the safer compile target?
12. Which source objects are provisional duplicates of the legacy path and can
    be removed only after HLS/cosim proves the service graph?
13. At the selected high-frequency HLS target, is eight-context GEMV still
    II=1? If not, does 16-context
    interleaving meet timing without excessive BRAM/control growth?
14. What is the measured recurrent feedback latency at the selected F250 HLS
    target, and is the
    two-head/10-slot C-slow schedule sufficient before considering 12 slots?
15. Does the current argmax remain II=1 at 250 MHz, or is the four-bank strict
    merge actually necessary?
16. What per-pseudo-channel efficiency do the exact 200/250 MHz GEMV
    microbenchmarks measure, and does the resulting BF16 F250 bound still fit
    <=11.5 ms?
17. Are any proposed physical constraints coupled to fragile generated leaf
    names rather than stable service/island wrappers and exact-count selectors?
18. Does every suggested improvement preserve the arithmetic rounding points,
    FP32 association, full logits, strict tie rule, 32 ports, and 16 clusters?

Suggestions should identify the evidence they answer, the smallest changed
variable, expected timing/routing effect, verification needed, and rollback.
Do not recommend a broad directive sweep, global fanout limit, large hard
pblock, precision downgrade, or reduced topology without first showing why the
fixed repair order cannot address the measured path family.

## 14. Current interpretation

The path to 250 MHz is not "add more pipeline pragmas everywhere." It is:

1. make launch/control state local and bounded;
2. separately remove or register the AXI-adapter FIFO handshake path families;
3. choose 3/6/7 versus 4/6/6 from measured CLB relief **and** SLL cost;
4. make every crossing use a two-entry RTL skid buffer and prove physical
   LAGUNA placement;
5. prove the new concurrent protocol across consecutive unequal commands and
   randomized stalls;
6. choose HLS targets from measured schedule/resource response, not a fixed
   compile/link ratio;
7. close 150 MHz and use its routed checkpoint to identify the next real path;
8. measure the 200/250 MHz HBM ceiling, then close 200 MHz;
9. continue to exact 250 MHz knowing BF16 will be bandwidth-bound and the gain
   over F200 is likely about 9%; and
10. use context interleaving, C-slow recurrence, and banked argmax only where
    the selected high-frequency schedule proves feedback distance insufficient.

Iter68A implements the first source-level version of persistent control and the
3/6/7 candidate. The adapter repairs, RTL skid buffers, topology gate, cosim,
and all physical-frequency results remain plans rather than achieved evidence.
Sub-10-ms latency is now explicitly a later compression-quality decision, not
an implied consequence of reaching 250 MHz.
