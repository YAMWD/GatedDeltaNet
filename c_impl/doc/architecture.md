# GatedDeltaNet Decode Accelerator Architecture

**Status:** Current production architecture (Iter57), routed, timing-closed,
and exact on an Alveo U55C on 2026-08-15. It is HLS-synthesized at 150 MHz and
links at the requested **100 MHz** without automatic clock scaling. The exact
64-token run measures **42.023540 ms/token / 4.202354M cycles**, making Iter57
both the fastest wall-clock image and the lowest-cycle timing-closed image.

The accelerator in `c_impl/` is decode-only. Prompt prefill runs on the GPU and
exports fixed-size recurrent and convolution state. The FPGA then advances one
token per `gdn_forward` invocation. The successful design combines:

- 32 independent 512-bit HBM weight readers;
- 16 two-port FP32 GEMV clusters;
- BRAM-backed MM2S/FIFO decoupling;
- a 4/6/6 cluster distribution across SLR0/SLR1/SLR2;
- transient activations resident in local BRAM for the complete 24-layer
  forward;
- four packed recurrent-state stripes appended to HBM weight ports 28--31;
- two concurrent 16-column, head-local recurrent islands with HBM retrieval and
  update fused into the two required attention passes;
- concurrent transfer from all four state ports into packed low/high URAM
  pairs;
- head-serial/all-port Q/K/V/gate (`QKVG`) and pair-interleaved MLP gate/up
  (`GU`) weight layouts, each issued as one shared-GEMV command;
- a time-shared head-local Q/K/V convolution actor that consumes each QKVG
  head while later heads are still streaming;
- immediate head-local Q/K/V streams from that convolution actor into the
  recurrent actor, avoiding whole-hidden Q/K/V materialization;
- state reads on ports 28--31 interleaved with each QKVG head's weights and
  decoupled through four two-head, depth-2048 BRAM queues;
- duplicated Q/K/V head streams feeding the two recurrent islands, followed by
  a deterministic half-head merge;
- explicit registered relay actors between each 4/6/6 SLR-local GEMV collector
  and the final collector;
- one forward-only dataflow graph that overlaps later QKVG heads with
  convolution, state transfer, and recurrence of earlier heads;
- packed external recurrent-state and convolution-tail transfers; and
- on-chip LM-head argmax, with no external logits materialization.

The architecture routes with zero failed nets, zero unrouted nets, and zero
node overlaps and preserves exact 64-token parity. Post-route physical
optimization closes the 100 MHz kernel clock at **+0.060 ns WNS** and the fixed
250 MHz DMA clock at **+0.003 ns WNS**. Iter57 is 2.48% faster and uses 2.48%
fewer cycles than Iter39C; it is 2.889x faster than the 121.4 ms eight-port
baseline.

## Fixed Model Shape

`gdn_forward` is specialized to the committed GDN-1.3B FP32 decode shape:

| Parameter | Value |
|---|---:|
| Hidden dimension | 2,048 |
| Attention heads | 8 |
| Q/K/V head dimension | 256 |
| MLP intermediate dimension | 5,632 |
| Layers | 24 |
| Convolution kernel | 4 |
| Vocabulary | 32,000 |
| Tokens per invocation | 1 |

The recurrent state is `24 x 8 x 256 x 256` FP32 values, or 48 MiB. The
convolution state is three prior rows for Q, K, and V in every layer, or about
1.69 MiB. These two state classes persist in HBM across kernel calls; transient
activations do not.

## System Overview

```text
GPU prefill
    |
    | .gdnstate: recurrent state + Q/K/V convolution tails + seed token
    v
XRT host
    |-- keeps the original embedding table in host memory
    |-- builds 32 compact, output-row-striped GEMV weight shards
    |-- builds one compact non-GEMV auxiliary-weight image
    |-- writes the selected 8 KiB embedding row to workspace[X]
    v
+---------------------------------------------------------------------+
| gdn_forward, one token                                               |
|                                                                     |
|  workspace[X] --> local activation BRAMs --> 24 GDN layers          |
|                         |                     |                      |
|                         |                     +--> recurrent state --+--> HBM28..31
|                         |                     +--> convolution tails +--> HBM0
|                         v                                            |
|             32 HBM readers --> 16 GEMV clusters --> local results   |
|                         |                                            |
|                  final norm + LM head                                |
|                         |                                            |
|                  on-chip strict argmax                               |
|                         v                                            |
|                workspace[X_NORM][0] = token                          |
+---------------------------------------------------------------------+
```

The host writes an embedding rather than a token ID because the embedding table
remains in host memory. After the kernel completes, the host reads one 512-bit
line at the existing `GDN_WS_OFF_X_NORM` offset and interprets lane zero as the
next token ID.

## Kernel Interface and HBM Mapping

The HLS top has 34 pointer arguments:

- `aux_weights`;
- `workspace`; and
- `weight_data_mm0` through `weight_data_mm31`.

They synthesize to exactly 32 AXI masters. `aux_weights`, `workspace`, and shard
0 intentionally share `mem_weights_mm0`; shards 1-31 each have their own
master. The successful connectivity is:

| AXI bundle | HBM bank | Contents |
|---|---:|---|
| `mem_weights_mm0` | HBM[0] | GEMV shard 0, compact auxiliary weights, workspace, convolution tails, embedding ingress, token egress |
| `mem_weights_mm1` ... `mem_weights_mm27` | HBM[1] ... HBM[27] | One compact GEMV shard per bank, read-only during decode |
| `mem_weights_mm28` ... `mem_weights_mm31` | HBM[28] ... HBM[31] | One GEMV shard plus one 12 MiB recurrent-state stripe per bank |

Every large matrix, including `lm_head`, is split across the 32 compact shards,
with one total copy of every projection weight. Conventional O, MLP-down, and
LM-head sections retain output-row stripes. For each sequential head, QKVG
concatenates its 256 Q, K, V, and gate rows and stripes those 1,024 rows across
all 32 ports: channels 0--7 carry Q segments, 8--15 K, 16--23 V, and 24--31
gate. Two collector rounds therefore expose one complete head every roughly
4,096 weight beats. GU assigns channel `c = chunk*2 + kind` one 352-row chunk.
Within every shard the sections are stored in this order:

```text
per layer: head-serial/all-port qkvg, output, pair-interleaved gu, mlp_down
global:    lm_head
```

The host-side exact shard validator compares every copied FP32 weight against
the source blob before native decode, rather than relying only on token parity.

One weight shard contains 43,728,896 FP32 values (166.8125 MiB). Ports 28--31
append 3,145,728 FP32 state values (12 MiB) after that fixed boundary. The small norms,
A/B projections, convolution kernels, and final norm are packed into
`aux_weights`; the full 5.6 GB source weight blob is not duplicated on the
device.

Port 0 is special. Its loader first copies the local GEMV activation into the
cluster ripple and then streams shard-0 weights. This gives the dataflow region
one reader for the shared port-0 bundle and avoids the local congestion caused
by giving compute blocks direct AXI access.

## External Workspace Contract

The host-visible `GDN_WS_OFF_*` layout is unchanged from Iter31. This preserves
the external ABI and XRT allocation behavior, but most activation offsets are
now reserved compatibility space rather than active HBM traffic.

| Workspace region | Current synthesized use |
|---|---|
| `GDN_WS_OFF_X` | Host writes one 2,048-float embedding; kernel reads it once |
| `GDN_WS_OFF_X_NORM` | Kernel writes one 512-bit result line; lane zero is the selected token |
| Q/K/V, gate, attention, temporary, MLP and logits offsets | Reserved but not accessed by the activation datapath |
| `GDN_WS_OFF_REC_STATE` | Reserved compatibility space; recurrent state moved to weight-shard tails 28--31 |
| `GDN_WS_OFF_HEAD_BUF` | Persistent packed Q/K/V convolution-tail read/update/write |

The recurrent state cannot remain on chip across the complete decode. One layer
is 2 MiB; all 24 layers would require roughly 2,304 URAMs before any other
storage, versus 960 URAMs on the U55C. Iter38 buffers one 256 x 256 FP32 head at
a time. Within each state row, the eight low-half Pack16 words alternate between
ports 28/29 and the eight high-half words alternate between ports 30/31. All
four ports now advance together at one Pack16 word per port per cycle. Each
low/high column pair is stored as one compact 64-bit word in a 32-bank cyclic
URAM array, allowing both halves to share a bank address without doubling URAM.
The update loop writes all four striped words concurrently and immediately.
Every port retains a monotonic 1,024-beat transfer per head; no new AXI master
or public `.gdnstate` format is required.

## Activation-Resident Datapath

The embedding is loaded once from HBM into local `Pack16` storage. A `Pack16`
is one 512-bit word containing 16 FP32 lanes. The activation lifetime then
remains on chip until the final token is written.

The residual and conventional projections use local activation buffers, while
Q/K/V now pass through bounded head streams rather than whole-hidden storage.
Two equally shaped buffers provide a single shared GEMV aperture:

| Local storage | Physical size | Lifetime/reuse |
|---|---:|---|
| `x_storage` | 128 Pack16 / 8 KiB | Residual stream across all 24 layers |
| `norm_attn_storage` | 128 Pack16 / 8 KiB | RMSNorm output, then reused for attention output |
| `q_mlp_gate_storage` | 352 Pack16 / 22 KiB | MLP gate after GU deinterleave |
| `k_mlp_up_storage` | 352 Pack16 / 22 KiB | MLP up after GU deinterleave |
| `gate_storage` | 128 Pack16 / 8 KiB | Attention output gate |
| `gemv_in_storage` | 352 Pack16 / 22 KiB | Common input aperture for every large projection |
| `gemv_out_storage` | 704 Pack16 / 44 KiB | Unified GU output aperture and LM-head token result |
| `a_storage`, `b_storage` | 16 FP32 each | Fully partitioned eight-element head vectors |

All large buffers are explicitly bound to dual-port BRAM. The shared 352-word
GEMV input and 704-word output apertures let HLS time-share one physical
`gdn_gemv` graph across all projection shapes. The top also enforces
`allocation function instances=gdn_gemv limit=1`. Without this, HLS specializes
complete 32-reader fabrics for different pointer shapes and multiplies the
hardware.

Inside the QKVG command, bounded Q/K/V streams carry 16 Pack16 words per head
and a broadcaster duplicates each head for the two recurrent islands. Four
additional BRAM queues each hold two 1,024-word recurrent-state stripes. Their
depth is a correctness condition, not speculative buffering: the registered
collector boundary increases the Q/K/V round-trip latency beyond one head, so
a one-head state queue forms a cyclic wait with the state-owning MM2S actors.
One-layer RTL cosimulation reproduced that seven-actor deadlock at depth 1024;
depths 2048 and 4096 both progressed more than 5.3x beyond it, so Iter57 retains
the minimum sufficient depth of 2048.

The aperture copies are local 512-bit II=1 loops. They cost tens of thousands
of cycles per token, rather than the millions of cycles and dynamic stalls
caused by materializing every activation through HBM.

## Per-Layer Schedule

Each of the 24 layers executes serially:

```text
local x
  -> RMSNorm -> norm_attn
  -> tiny GEMV A and B -> partitioned scalar arrays
  -> load packed Q/K/V convolution weights and old tails
  -> shared GEMV aperture
       -> one head-serial/all-port QKVG command
       -> for each completed head while later heads stream:
            store gate
            run one shared head-local convolution actor for Q, K, and V
            capture the new raw Q/K/V tail row
            stream convolved Q/K/V directly to recurrence
       -> on ports 28--31, after each head's QKVG weights:
            stream four 1,024-word old-state stripes into depth-2048 queues
       -> duplicate Q/K/V into two recurrent islands
       -> recurrent islands, concurrently with later QKVG heads:
            drain one head of Q/K/V and state into local scratch
            process disjoint 128-column halves at 16 columns/cycle each
            fuse head-local retrieval and update
            write updated state directly to ports 28--31
       -> merge the two half-head outputs into norm_attn
  -> packed convolution-tail writeback
  -> output norm and gate, in place
  -> output-projection GEMV
  -> local residual accumulation into x
  -> RMSNorm
  -> one pair-interleaved GU GEMV
  -> local deinterleave into reused Q/K buffers
  -> SwiGLU, in place
  -> MLP down GEMV
  -> local residual accumulation into x
```

The QKVG sink holds only one 256-element head. One convolution datapath is
called serially for that head's Q, K, and V, then emits them to a broadcaster
while the upstream graph produces later heads. Two independent recurrent
actors each consume all Q/K/V values but own disjoint 128-column state/output
halves; a small merge actor restores natural output order. The four state
readers interleave a complete state-head burst between adjacent QKVG weight
head bursts. Each island drains blocking FIFO reads in a narrow II=1 copy loop
before touching its local URAM scratch. There is no feedback credit or
readiness handshake in the graph. The two-head state FIFO depth makes the
strictly forward schedule deadlock-free despite the registered GEMV collector
boundaries.

The old three-row convolution tails and weights are staged through fixed-bank
512-bit loops. After a head consumes its old context, obsolete tail row 0 stores
the new raw row; the final writeback emits old rows 1/2 and that new row. Output
projection and MLP-down results are added directly to local `x`.

There are four shared-GEMV commands per layer and one global LM-head command:

```text
24 * (QKVG + output + GU + MLP down)
  + LM head
= 97 large GEMV calls per token
```

This call reduction does not reduce FP32 MACs or HBM weight bytes. It exposes a
stream-friendly logical layout and removes top-level command boundaries. The
raw HLS top minimum therefore overstates its direct cycle benefit; hardware
measurement, not call count, determines the retained gain.

The architecture does not wrap the whole layer in one HLS `dataflow` region.
The shared GEMV fabric is time-multiplexed between projections; only the
internals of each GEMV use dataflow.

## 32-Port GEMV Microarchitecture

### Reader/compute decoupling

Each projection starts 32 monotonic MM2S readers. A reader issues one 512-bit
weight beat per cycle into a depth-64 BRAM FIFO:

```text
HBM[c] -- MM2S II=1 --> ws[c] BRAM FIFO --> compute cluster
```

This decouples AXI burst issue from FP32 compute. The reader can keep HBM
transactions long and contiguous while the FIFO absorbs pipeline startup,
row-boundary handling, and short compute stalls. Earlier direct-AXI compute
coupled address generation and MAC placement around the HBM interfaces,
creating severe local congestion. BRAM FIFOs also move about 60K LUTs and 112K
FFs of wide queue storage out of the already dense SLR0 CLB fabric.

### Sixteen two-port clusters

Two adjacent weight channels feed one `gemv32_cluster2`; 16 clusters consume all
32 ports. The activation travels through a one-way ripple:

```text
xr[0] -> cluster 0 -> xr[1] -> cluster 1 -> ... -> cluster 15 -> drain
              |                         |
           ws[0:1]                  ws[30:31]
              |                         |
            ys[0]                    ys[15]
```

Every cluster captures a private, 16-bank activation copy in BRAM before
computing. This avoids one 32-way high-fanout activation broadcast. A cluster
computes two output stripes in parallel:

- 16 FP32 multiplies per channel per weight beat;
- a balanced 16-input FP32 adder tree;
- `GEMV_PARTIAL=4` rotating partial sums;
- a flattened row/reduction loop at II=4, unrolled by four; and
- ping-pong accumulation banks so the prior row drains while the next row
  starts.

II=4 with four unrolled packs still consumes one weight beat from each port per
cycle. Reducing the sharing factor from eight to four preserved the arithmetic
rate and DSP count while cutting the cluster operand-mux LUT cost.

### SLR-local result collection

The compute topology is physically grouped 4/6/6:

| Region | Clusters | Weight channels | Collector output per result-pack step |
|---|---|---|---:|
| SLR0 | 0-3 | 0-7 | 8 channel packs |
| SLR1 | 4-9 | 8-19 | 12 channel packs |
| SLR2 | 10-15 | 20-31 | 12 channel packs |

Each SLR has a local collector. A final collector concatenates the three
streams, avoiding a single 16-input global collector and keeping most result
wiring local.

The physical Iter22 hook moves cluster 8 and its two FIFO endpoints into the
SLR1 east side. This narrow topology-boundary adjustment relieved the SLR0/SLL
hotspot without imposing a global hard floorplan. The Iter23 hook replicates
the measured high-fanout DMA FIFO-address cone. The final configuration keeps
the successful connectivity and hooks unchanged.

### Output reorder and store modes

Collectors emit pack-major/channel-minor order. `gemv32_store` first buffers
that stream in a shared URAM reorder memory, then exposes natural output-row
order to the rest of the kernel.

Normal projections copy natural-order `Pack16` results into the local output
aperture. The LM head uses a different store mode:

1. Read every reorder word once at II=1.
2. Compare its 16 lanes in parallel.
3. Keep 16 lane-local winners.
4. Merge the winners using strict `>` comparison and lowest-index tie
   resolution.
5. Return only the selected token.

No 32,000-float logits tensor is written to or reread from HBM. The comparison
order preserves the original first-index argmax behavior.

## Auxiliary Compute

Non-GEMV blocks remain shared, relatively narrow implementations:

| Block | Parallelism |
|---|---:|
| RMSNorm/reductions | 8 lanes |
| Tiny A/B GEMV | 2 output lanes |
| Depthwise convolution, output gate, SwiGLU | 4 elementwise lanes |
| Recurrent state arithmetic | **32 lanes** |

Iter35 selectively restored recurrent-state arithmetic from 8 to 16 lanes and
Iter37 widened its aggregate width to 32. Wider normalization, output-gate, and
SwiGLU variants were rejected because their measured cycle savings did not
justify their DSP/LUT cost. Iter57 keeps those narrow auxiliary blocks and
realizes the 32 recurrent columns as two concurrent physical 16-column islands,
with both state halves supplied from all four stripes.

## Static Schedule and HLS Resources

Because `gdn_gemv` accepts runtime dimensions, its top-level call accounting is
not a complete token-latency model. The last dimension-correct schedule before
QKVG/GU command merging was:

| Component | Iter32 cycles | Iter36 cycles | Iter37 cycles |
|---|---:|---:|---:|
| Equivalent dense GEMV work | 2,785,524 | 2,785,524 | 2,785,524 |
| RMSNorm | 139,405 | 139,405 | 139,405 |
| Tiny GEMV | 103,584 | 103,584 | 103,584 |
| Q/K/V convolution | 624,744 | 624,744 | 624,744 |
| Recurrent attention | 5,760,144 | **1,841,112** | **1,057,944** |
| Output norm/gate | 128,712 | 128,712 | 128,712 |
| SwiGLU | 321,048 | 321,048 | 321,048 |
| Activation handoff/copy/add/argmax | 54,069 | 54,069 | 54,069 |
| **Reconstructed total** | **9,917,230** | **5,998,198** | **5,215,030** |

Iter38 preserves that dense weight/MAC work, reduces the fixed-trip recurrence
by **397,056 cycles/token**, and adds bounded QKVG/GU deinterleave loops. Its raw
HLS top minimum is 2,744,183 cycles versus 3,957,551 for Iter37, but about
0.816M of that difference is dynamic-call accounting rather than removed work.
Iter39C then overlaps the three Q/K/V convolutions with head-serial QKVG
production. Its top minimum is **2,270,495 cycles**, a fixed
**473,688-cycle / 17.27%** reduction from Iter38; the measured hardware result
is the authoritative total. Iter54c extends the bounded graph through recurrent
attention and interleaves old-state reads with the QKVG weight stream. Its top
minimum is **1,621,415 cycles**, 649,080 below Iter39C, while the on-card
effective count falls by 196,905 cycles to **4.112395M**. The smaller measured
reduction reflects dynamic HBM/control stalls omitted by the raw minimum.
Iter57 trades some of that theoretical minimum for physical timing boundaries:
two recurrent islands, duplicated head streams, a merge, and three registered
collector relays raise the HLS minimum to **1,670,212 cycles**. On hardware it
executes **4.202354M cycles**, still 106,646 cycles below Iter39C while restoring
true 100 MHz timing closure.

Final integrated csynth resources:

| Resource | Iter37 | Iter38 | Iter39C | Iter54c | Iter57 |
|---|---:|---:|---:|---:|---:|
| RAMB18 | 1,511 | 1,527 | 1,543 | 1,632 | **1,728** |
| DSP | 3,627 | 3,629 | 3,629 | 3,716 | **3,762** |
| FF | 851,903 | 857,343 | 863,589 | 888,271 | **921,774** |
| LUT | 912,694 | 929,941 | 937,707 | 868,961 | **869,262** |
| URAM | 48 | 48 | 48 | 48 | **48** |

The recurrent call falls from 43,873--44,081 cycles in Iter37 to
**27,329--27,537 cycles**. The combined four-port state read and write loops
both achieve II=1; GEMV MAC II remains four, no AXI master was added, and all
32 MM2S readers remain II=1. Iter39C's six context-read and three tail-write
loops are 512-bit II=1 transfers; its one shared head-convolution compute loop
is II=1. Iter54c keeps all 16 GEMV cluster MAC loops at II=4 and all four
state-owner streams at II=1. The added RAMB18 capacity is primarily the four
depth-1024, 512-bit state FIFOs that eliminate the rejected credit-handshake
feedback path.
Iter57 preserves those IIs and the same 32 masters. Its four state FIFOs are
depth 2048, while the recurrent arithmetic is split into two 16-column actors
that preserve the prior aggregate 32-column width. The additional DSPs and
registers are the cost of the duplicated Q/K/V-side arithmetic and control
needed to create physically separable timing islands.

## Physical Implementation

The reproducible Iter57 build compiles HLS at 150 MHz and links a 100 MHz
U55C link. The Make target uses only relocatable configuration/Tcl source:

```bash
make -C c_impl run_hw
```

This single target builds the current 150 MHz HLS / 100 MHz image and
then runs the exact 8-token smoke and 64-token on-card measurement. Historical
iteration-specific Make targets are not part of the production interface.

Its physical recipe uses:

- exact 32-bank connectivity from the routed Iter31 topology;
- the Iter35 DMA fanout repairs plus the measured Iter54 r15/AR-control repair;
- hard containment of the two-island recurrent wrapper in the full SLR2;
- hard full-SLR1 containment for clusters 8 and 10 and cluster 10's immediate
  weight/activation streams;
- hard SLR1 containment and `USER_SLL_REG` marking for the three registered
  collector relays and the final collector;
- clock-region replication of the 35,670-load kernel reset alias;
- clock-region replication of the two measured r15 response-FIFO DMA cones;
- hard containment of eight measured HMSS AR-control primitives in X2Y1/X3Y1;
- BRAM implementation for all wide MM2S, activation-ripple, result, and
  collector FIFOs, plus the four depth-2048 state queues;
- a post-place structural gate that verifies all four pblocks and reports SLR,
  SLL, CLB, and BRAM pressure without prematurely rejecting a routable design;
- `SSI_SpreadSLLs` placement;
- `NoTimingRelaxation` routing; and
- pre-route and post-route `AggressiveExplore` physical optimization.

Final implementation result:

| Metric | Result |
|---|---:|
| Requested / encoded kernel clock | **100 / 100 MHz** |
| Failed nets | 0 |
| Unrouted nets | 0 |
| Node overlaps | 0 |
| Kernel-clock WNS / TNS | **+0.060 ns / 0** |
| Kernel-clock WHS / THS | **+0.009 ns / 0** |
| Fixed 250 MHz DMA WNS | **+0.003 ns** |
| Automatic clock scaling | **none** |

The final router reports 100% of nets fully routed. Routed whole-device usage
is 636,109 CLB LUTs, 915,909 CLB registers, 1,667.5 BRAM tiles, 48 URAMs and
3,772 DSPs. SLR0 remains physically critical at 98.92% CLB occupancy versus
73.29% in SLR1 and 76.14% in SLR2. RAMB36/FIFO use is 585/576/337 across
SLR0/SLR1/SLR2 (87.05/85.71/50.15%), so any additional storage, arithmetic, or
cross-SLR control still requires a new full route rather than inference from
HLS resources.

### Previous Iter36 frequency result

For comparison, the Iter36 source was built with a 130 MHz link request:

```bash
bash c_impl/build_iter36_headlocal.sh 130
```

Post-route `AggressiveExplore` improved the kernel paths but did not close the
7.692 ns requirement. At the requested 130 MHz, the post-physopt report has
WNS/TNS **-0.948/-10111.593 ns**, 19,904 failing setup endpoints, and WHS
**+0.001 ns**. Vivado auto-frequency scaling encoded `DATA_CLK` at an achieved
**115.7 MHz**. The fixed 250 MHz DMA same-clock setup path remained positive.
The build completed in 31 h 31 m and emitted a valid XCLBIN; this is a routed,
on-card-validated 115.7 MHz result, not a 130 MHz closure result.

## Correctness and On-Card Performance

The Iter57 image passed:

- native C++ build;
- fast exact decode, 6/6;
- full native exact decode, 32/32;
- eight-token on-card smoke parity; and
- 64-token on-card decode-from-state parity, first divergence `-1`.

Excluding the initial seed entry, 63 measured kernel calls produced:

| Metric | Iter57 at 100 MHz |
|---|---:|
| Minimum | 41.968075 ms/token |
| Maximum | 42.389207 ms/token |
| Median | **42.009069 ms/token** |
| Mean | **42.023540 ms/token** |
| Effective cycles | **4.202354M** |
| Cycle/latency reduction from Iter39C | **2.48%** |
| Latency reduction from Iter38E, 47.079335 ms | **10.74%** |
| Speedup over eight-port baseline, 121.4 ms | **2.889x** |

Iter57 is modestly higher-cycle than auto-scaled Iter54c, but it removes that
image's frequency downgrade and is the first version of the timing-island arc
to route, close at a true 100 MHz, and run exact on card. The deeper state
queues are part of the functional architecture: the otherwise identical
depth-1024 Iter56b bitstream closed timing but deadlocked on its first token.

For comparison, the Iter39C image produced:

| Metric | Iter39C at 100 MHz |
|---|---:|
| Minimum | 43.080 ms/token |
| Maximum | 43.314 ms/token |
| Median | **43.086 ms/token** |
| Mean | **43.093 ms/token** |
| Mean cycles at 100 MHz | **4.309M** |
| Speedup over Iter38E, 47.079 ms | **1.093x / 8.47% latency reduction** |
| Speedup over Iter37D, 51.451 ms | **1.194x** |
| Speedup over Iter36 100 MHz, 59.578 ms | **1.382x** |
| Speedup over eight-port baseline, 121.4 ms | **2.817x** |

At 100 MHz, Iter39C removes 0.399M measured cycles from Iter38, capturing 84.2%
of the 0.474M fixed static reduction. The residual difference confirms that
the raw 2.270M HLS minimum is not an end-to-end prediction; external traffic
and dynamic command stalls remain outside that minimum.

For comparison, the previous Iter36 auto-scaled follow-up ran the same smoke
and full fixtures with its 115.7 MHz image. Both remained exact. Excluding the
seed, its 63 kernel calls measured:

| Metric | Auto-scaled 115.7 MHz |
|---|---:|
| Minimum | 51.776 ms/token |
| Maximum | 52.104 ms/token |
| Median | **51.813 ms/token** |
| Mean | **51.844 ms/token** |
| Improvement over Iter36 100 MHz | **1.149x / 12.98%** |
| Speedup over Iter35, 75.062 ms | **1.448x** |
| Speedup over Iter32, 98.660 ms | **1.903x** |
| Speedup over eight-port baseline, 121.4 ms | **2.342x** |

The ideal clock-ratio prediction was 51.493 ms/token; the measured result is
within 0.68%, so the frequency increase did not expose a material new stall.

Reproducibility hashes for the retained Iter57 result:

| Artifact | SHA-256 |
|---|---|
| `gdn_model.cpp` | `2aa7d9c044af627f925a03ef9e42a2e997111235bba07177d6cce4d4b1e46cd6` |
| Compiled XO | `4df8788f6221b84d38ee8ea795b1ed6a4c91ea22d83f2115f7e413b3751a7096` |
| Relocatable link configuration | `4faf23ce87da7fb6c397305a82c3a22d24340198787dee9de1e037bd8b133436` |
| Timing-island/floorplan hook | `6d36b0f1c52383c787d543d55a849ded1567113cf25bac4d8523f8cc0c12483f` |
| Corrected Iter54 DMA hook | `aa0d8a155684444061aaffa9cfd1f687fc656c1d93745db96c792364683b6bdb` |
| Timing-island placement gate | `ffca9307e73543c7e8a2408fd1a6678dfd2975fdd9b6628e28fb242c084b2e2b` |
| Routed, timing-closed 100 MHz XCLBIN | `4178d442d956eece4d86d2812f12e8bd1894a5036646c14336d5f3de81cc1bcd` |

## Design Invariants

Preserve these unless a replacement is validated through native parity,
integrated csynth, full implementation, and on-card measurement:

1. Keep 32 independent weight masters and 16 two-port clusters.
2. Keep MM2S readers separate from compute and keep the wide FIFOs in BRAM.
3. Keep the activation ripple and 4/6/6 SLR-local collectors.
4. Keep only one shared physical `gdn_gemv` instance.
5. Keep transient activations local; do not reintroduce workspace
   materialization.
6. Keep recurrent state and convolution tails packed and externally
   persistent.
7. Keep the recurrent buffer head-local, its low/high pairs compacted into
   64-bit URAM words, all four state ports concurrent, and HBM transfer fused
   into the required retrieval/update traversals.
8. Keep LM-head argmax on chip and preserve strict first-index tie behavior.
9. Keep the external workspace offsets and host ABI stable.
10. Treat the Iter35/54 DMA hooks, recurrent-SLR2 assignment, cluster 8/10
    placement, and registered collector boundary placement as part of the
    successful physical architecture.
11. Judge changes against both 100 MHz timing and zero-conflict routing; HLS
    resource savings alone do not predict routability.
12. Preserve the head-serial/all-port QKVG and pair-interleaved GU layouts,
    including the full-byte host validation, until a replacement proves
    exactness and a lower measured cycle count.
13. Keep one time-shared head-local convolution actor inside the QKVG result
    sink, plus fixed-bank packed context movers; do not restore three serial
    whole-hidden convolution passes.
14. Keep convolved Q/K/V head-local and streamed directly to recurrence; do not
    restore whole-hidden Q/K/V buffers.
15. Keep the state graph forward-only with a two-head depth-2048 BRAM queue per
    state port. Depth 1024 is a proven hardware deadlock for this registered
    collector topology. Do not reintroduce feedback credit/readiness handshakes
    without a bounded proof and a better routed result.
16. Keep the two concurrent 16-column recurrent islands and deterministic
    half-head merge until a replacement proves equal correctness, lower cycles,
    and at least the current 100 MHz timing margin.
17. Keep native full-logits comparison enabled as a pre-hardware correctness
    gate; token parity alone cannot detect non-argmax numerical regressions.

Use [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md) for the
historical GEMV scaling progression and
[optimization_log.md](optimization_log.md) for the exhaustive build,
congestion, timing, and performance record.
