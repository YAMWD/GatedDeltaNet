# GatedDeltaNet Decode Accelerator Architecture

**Status:** Current routed production architecture (Iter39C). The image closes
timing at the requested 100 MHz and was measured 2026-08-04 on an Alveo U55C.

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
- 32-lane, head-local recurrent arithmetic with HBM retrieval and update fused
  into the two required attention passes;
- concurrent transfer from all four state ports into packed low/high URAM
  pairs;
- head-serial/all-port Q/K/V/gate (`QKVG`) and pair-interleaved MLP gate/up
  (`GU`) weight layouts, each issued as one shared-GEMV command;
- a time-shared head-local Q/K/V convolution actor that consumes each QKVG
  head while later heads are still streaming;
- packed external recurrent-state and convolution-tail transfers; and
- on-chip LM-head argmax, with no external logits materialization.

The architecture routes with zero failed nets, zero unrouted nets, and zero
node overlaps. It closes setup and hold with design WNS/WHS
**+0.003/+0.009 ns** and preserves exact 64-token parity at
**43.093 ms/token / 4.309M cycles**: 1.093x faster than Iter38E and 2.817x
faster than the 121.4 ms eight-port baseline.

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

| Workspace region | Iter39C synthesized use |
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

Six logical activation buffers hold the layer state, and two equally shaped
buffers provide a single shared GEMV aperture:

| Local storage | Physical size | Lifetime/reuse |
|---|---:|---|
| `x_storage` | 128 Pack16 / 8 KiB | Residual stream across all 24 layers |
| `norm_attn_storage` | 128 Pack16 / 8 KiB | RMSNorm output, then reused for attention output |
| `q_mlp_gate_storage` | 352 Pack16 / 22 KiB | Q for attention, then MLP gate |
| `k_mlp_up_storage` | 352 Pack16 / 22 KiB | K for attention, then MLP up |
| `v_storage` | 352 Pack16 / 22 KiB | V; only the first 128 words are logically used |
| `gate_storage` | 128 Pack16 / 8 KiB | Attention output gate |
| `gemv_in_storage` | 352 Pack16 / 22 KiB | Common input aperture for every large projection |
| `gemv_out_storage` | 704 Pack16 / 44 KiB | Unified GU output aperture and LM-head token result |
| `a_storage`, `b_storage` | 16 FP32 each | Fully partitioned eight-element head vectors |

All large buffers are explicitly bound to dual-port BRAM. The equal 352-word
shape for Q, K, V, and both GEMV apertures is deliberate: it makes HLS share one
physical convolution implementation and one physical `gdn_gemv` graph across
all projection shapes. The top also enforces
`allocation function instances=gdn_gemv limit=1`. Without this, HLS specializes
complete 32-reader fabrics for different pointer shapes and multiplies the
hardware.

The aperture copies are local 512-bit II=1 loops. They cost tens of thousands
of cycles per token, rather than the millions of cycles and dynamic stalls
caused by materializing every activation through HBM.

## Per-Layer Schedule

Each of the 24 layers executes serially:

```text
local x
  -> RMSNorm -> norm_attn
  -> load packed Q/K/V convolution weights and old tails
  -> shared GEMV aperture
       -> one head-serial/all-port QKVG command
       -> for each completed head while later heads stream:
            store gate
            run one shared head-local convolution actor for Q, K, and V
            capture the new raw Q/K/V tail row
  -> packed convolution-tail writeback
  -> tiny GEMV A and B -> partitioned scalar arrays
  -> recurrent gated-delta attention
       reads local Q/K/V
       fuses head-local HBM restore with retrieval
       fuses head-local state update with HBM save
       overwrites norm_attn with attention output
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
called serially for that head's Q, K, and V, while the upstream dataflow graph
continues producing later heads into bounded FIFOs. The old three-row tails and
convolution weights are staged through fixed-bank 512-bit loops. After a head
has consumed its old context, obsolete tail row 0 stores the new raw row; the
final writeback emits old rows 1/2 and that new row. Output projection and
MLP-down results are added directly to local `x`.

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

Iter35 selectively restored recurrent-state arithmetic from 8 to 16 lanes.
Wider normalization, output-gate, and SwiGLU variants were rejected because
their measured cycle savings did not justify their DSP/LUT cost. Iter38 keeps
those narrow blocks and Iter37's 32 recurrent lanes, while supplying both state
halves concurrently from all four stripes.

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
is the authoritative total.

Final integrated csynth resources:

| Resource | Iter37 | Iter38 | Iter39C |
|---|---:|---:|---:|
| RAMB18 | 1,511 | 1,527 | **1,543** |
| DSP | 3,627 | 3,629 | **3,629** |
| FF | 851,903 | 857,343 | **863,589** |
| LUT | 912,694 | 929,941 | **937,707** |
| URAM | 48 | 48 | **48** |

The recurrent call falls from 43,873--44,081 cycles in Iter37 to
**27,329--27,537 cycles**. The combined four-port state read and write loops
both achieve II=1; GEMV MAC II remains four, no AXI master was added, and all
32 MM2S readers remain II=1. Iter39C's six context-read and three tail-write
loops are 512-bit II=1 transfers; its one shared head-convolution compute loop
is II=1.

## Physical Implementation

The reproducible Iter39C build compiles HLS at 130 MHz and links the U55C image
at 100 MHz. The Make target owns the checksum guards and uses only relocatable
configuration/Tcl source:

```bash
make -C c_impl iter39
```

Its physical recipe uses:

- exact 32-bank connectivity from the routed Iter31 topology;
- `apply_iter22_cluster8_slr1_east.tcl`;
- `apply_iter35_dma_w15_fifoaddr_fanout.tcl` for the fixed 250 MHz DMA path;
- an exact recurrent hierarchy constraint to the full SLR2;
- BRAM implementation for all wide MM2S, activation-ripple, result, and
  collector FIFOs;
- `SSI_SpreadSLLs` placement;
- `AlternateCLBRouting`; and
- pre-route and post-route `AggressiveExplore` physical optimization.

Final implementation result:

| Metric | Result |
|---|---:|
| Build interval | 8 h 19 m |
| Requested / encoded kernel clock | **100 / 100 MHz** |
| Failed nets | 0 |
| Unrouted nets | 0 |
| Node overlaps | 0 |
| Design WNS / TNS | **+0.003 ns / 0** |
| Design WHS / THS | **+0.009 ns / 0** |
| Kernel-clock WNS | **+0.289 ns** |
| Fixed 250 MHz DMA WNS | **+0.003 ns** |
| Automatic clock scaling | none |

The final router reports 100% of nets fully routed. Initial routing still
reported level-6 global/short congestion and level-7 timing congestion before
rip-up/reroute converged to zero overlaps, so any additional storage,
arithmetic, or cross-SLR control still requires a new full route rather than
inference from HLS resources.

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

The Iter39C image passed:

- native C++ build;
- fast exact decode, 6/6;
- full native exact decode, 32/32;
- eight-token on-card smoke parity; and
- 64-token on-card decode-from-state parity, first divergence `-1`.

Excluding the initial seed entry, 63 measured kernel calls produced:

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

Reproducibility hashes for the retained Iter39C result:

| Artifact | SHA-256 |
|---|---|
| `gdn_model.cpp` | `d2674931f90897d932fc73915981866e38a233cb6efa4138caaef2029f3d5bbb` |
| Compiled XO | `53a47efc2098f6967fd256edce29aedfbe4dfe4da0383f609bc8b006e73131c0` |
| Recurrent-SLR2 hook | `15587403b5e904345abdee72cd84cfc0fa24f8be559f94f1e5554cdcedbd059e` |
| Relocatable link configuration | `998b71e3a8cb3b7f818f12cbe6581f0ffd2e04010dba5db3f20ca2ae844aa08f` |
| Timing-closed 100 MHz XCLBIN | `5c81e79ceb51d3faa00a4ae80a5055f716bbedac884700840011257494f11021` |

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
10. Treat the Iter22 cluster placement, Iter35 DMA hook, and recurrent-SLR2
    assignment as part of the successful physical architecture.
11. Judge changes against both 100 MHz timing and zero-conflict routing; HLS
    resource savings alone do not predict routability.
12. Preserve the head-serial/all-port QKVG and pair-interleaved GU layouts,
    including the full-byte host validation, until a replacement proves
    exactness and a lower measured cycle count.
13. Keep one time-shared head-local convolution actor inside the QKVG result
    sink, plus fixed-bank packed context movers; do not restore three serial
    whole-hidden convolution passes.

Use [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md) for the
historical GEMV scaling progression and
[optimization_log.md](optimization_log.md) for the exhaustive build,
congestion, timing, and performance record.
