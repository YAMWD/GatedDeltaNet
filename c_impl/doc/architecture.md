# GatedDeltaNet Decode Accelerator Architecture

**Status:** Current routed production architecture (Iter36), measured
2026-07-31 on an Alveo U55C.

The accelerator in `c_impl/` is decode-only. Prompt prefill runs on the GPU and
exports fixed-size recurrent and convolution state. The FPGA then advances one
token per `gdn_forward` invocation. The successful design combines:

- 32 independent 512-bit HBM weight readers;
- 16 two-port FP32 GEMV clusters;
- BRAM-backed MM2S/FIFO decoupling;
- a 4/6/6 cluster distribution across SLR0/SLR1/SLR2;
- transient activations resident in local BRAM for the complete 24-layer
  forward;
- a 16-bank, head-local recurrent-state buffer with HBM restore/retrieval and
  update/save fused into the two required attention passes;
- packed external recurrent-state and convolution-tail transfers; and
- on-chip LM-head argmax, with no external logits materialization.

The production image routes with zero failed nets, zero unrouted nets, and zero
node overlaps at 100 MHz. It produces an exact 64-token trajectory at
59.578 ms/token mean latency, 2.04x faster than the 121.4 ms eight-port
baseline and 1.26x faster than Iter35.

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
|                         |                     +--> recurrent state --+--> HBM0
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
| `mem_weights_mm0` | HBM[0] | GEMV shard 0, compact auxiliary weights, workspace, recurrent state, convolution tails, embedding ingress, token egress |
| `mem_weights_mm1` ... `mem_weights_mm31` | HBM[1] ... HBM[31] | One compact GEMV shard per bank, read-only during decode |

Every large matrix, including `lm_head`, is split across the 32 compact shards.
There is one total copy of the projection weights: channel `c` holds output rows
`[c * out_dim/32, (c + 1) * out_dim/32)`. Within every shard the matrices are
stored in this order:

```text
per layer: q, k, v, gate, output, mlp_gate, mlp_up, mlp_down
global:    lm_head
```

One shard contains 43,728,896 FP32 values (166.8125 MiB). The small norms,
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

| Workspace region | Iter36 synthesized use |
|---|---|
| `GDN_WS_OFF_X` | Host writes one 2,048-float embedding; kernel reads it once |
| `GDN_WS_OFF_X_NORM` | Kernel writes one 512-bit result line; lane zero is the selected token |
| Q/K/V, gate, attention, temporary, MLP and logits offsets | Reserved but not accessed by the activation datapath |
| `GDN_WS_OFF_REC_STATE` | Persistent packed recurrent-state read/update/write |
| `GDN_WS_OFF_HEAD_BUF` | Persistent packed Q/K/V convolution-tail read/update/write |

The recurrent state cannot remain on chip across the complete decode. One layer
is 2 MiB; all 24 layers would require roughly 2,304 URAMs before any other
storage, versus 960 URAMs on the U55C. Iter36 instead buffers one 256 x 256 FP32
head at a time in 16 cyclic URAM banks. The retrieval loop loads each 512-bit
state word from HBM while consuming it, and the update loop writes each updated
word back immediately. The unavoidable packed state round trip remains, but
the two redundant full-layer restore/save traversals and the 128-URAM
all-head buffer are gone.

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
| `gemv_out_storage` | 352 Pack16 / 22 KiB | Common output aperture and LM-head token result |
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
  -> shared GEMV aperture
       -> Q -> q_mlp_gate
       -> K -> k_mlp_up
       -> V -> v
  -> tiny GEMV A and B -> partitioned scalar arrays
  -> shared GEMV aperture -> attention gate
  -> Q/K/V depthwise convolution + SiLU, in place
  -> recurrent gated-delta attention
       reads local Q/K/V
       fuses head-local HBM restore with retrieval
       fuses head-local state update with HBM save
       overwrites norm_attn with attention output
  -> output norm and gate, in place
  -> output-projection GEMV
  -> local residual accumulation into x
  -> RMSNorm
  -> MLP gate GEMV -> reuse q_mlp_gate
  -> MLP up GEMV   -> reuse k_mlp_up
  -> SwiGLU, in place
  -> MLP down GEMV
  -> local residual accumulation into x
```

Q, K, and V convolution is safe in place because the convolution function
captures the complete current input row before it emits the output row. Output
projection and MLP-down results are added directly to local `x`, eliminating
two external residual-materialization passes.

There are eight large GEMVs per layer and one global LM-head GEMV:

```text
24 * (Q + K + V + gate + output + MLP gate + MLP up + MLP down)
  + LM head
= 193 large GEMV calls per token
```

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
| Recurrent state arithmetic | 16 lanes |

Iter35 selectively restored recurrent-state arithmetic from 8 to 16 lanes.
Wider normalization, output-gate, and SwiGLU variants were rejected because
their measured cycle savings did not justify their DSP/LUT cost. Iter36 keeps
those choices and changes recurrent-state materialization only.

## Static Schedule and HLS Resources

Because `gdn_gemv` accepts runtime dimensions, the raw top-level HLS latency
uses the maximum LM-head trip count for all 193 calls and is not a token
latency. Reconstructing the schedule with each real projection shape gives:

| Component | Iter32 cycles | Iter36 cycles |
|---|---:|---:|
| 193 large GEMVs | 2,785,524 | 2,785,524 |
| RMSNorm | 139,405 | 139,405 |
| Tiny GEMV | 103,584 | 103,584 |
| Q/K/V convolution | 624,744 | 624,744 |
| Recurrent attention | 5,760,144 | **1,841,112** |
| Output norm/gate | 128,712 | 128,712 |
| SwiGLU | 321,048 | 321,048 |
| Activation handoff/copy/add/argmax | 54,069 | 54,069 |
| **Reconstructed total** | **9,917,230** | **5,998,198** |

Final Iter36 integrated csynth resources:

| Resource | Iter36 |
|---|---:|
| RAMB18 | 1,511 |
| DSP | 3,323 |
| FF | 798,733 |
| LUT | 829,526 |
| URAM | **32** |

The head-local fusion reduces the recurrent buffer from Iter35's 128 URAMs to
16 and whole-kernel use from 144 to 32 URAMs. GEMV MAC II remains four, no AXI
master was added, and all 32 MM2S readers remain II=1.

## Physical Implementation

The reproducible build compiles HLS at 130 MHz and links the U55C image at
100 MHz:

```bash
bash c_impl/build_iter36_headlocal.sh 100
```

The launcher is hash-guarded and refuses to overwrite its existing build
directory. Its physical recipe uses:

- exact 32-bank connectivity from the routed Iter31 topology;
- `apply_iter22_cluster8_slr1_east.tcl`;
- `apply_iter23_dma_fanout.tcl`;
- `apply_iter35_dma_w15_fifoaddr_fanout.tcl` for the fixed 250 MHz DMA path;
- BRAM implementation for all wide MM2S, activation-ripple, result, and
  collector FIFOs;
- `AltSpreadLogic_high` placement;
- `Explore` routing; and
- pre-route and post-route `AggressiveExplore` physical optimization.

Final implementation result:

| Metric | Result |
|---|---:|
| Build duration | 7 h 47 m (7 h 26 m link) |
| Kernel clock | 100 MHz |
| Failed nets | 0 |
| Unrouted nets | 0 |
| Node overlaps | 0 |
| Overall WNS / TNS | +0.003 ns / 0 |
| Overall WHS / THS | +0.009 ns / 0 |
| 100 MHz kernel WNS / WHS | +0.104 ns / +0.009 ns |
| 250 MHz `dma_ip_axi_aclk_1` WNS / WHS | +0.003 ns / +0.009 ns |
| Automatic clock scaling | None |

Local routing remains dense: SLR0 reports effective south congestion level 6
and west level 5. The design is therefore routed, but it does not have enough
physical margin to justify casually widening compute blocks or changing FIFO
implementation. Any such change requires a new full route.

## Correctness and On-Card Performance

The final image passed:

- native C++ build;
- fast exact decode, 6/6;
- full native exact decode, 32/32;
- eight-token on-card smoke parity; and
- 64-token on-card decode-from-state parity, first divergence `-1`.

Excluding the initial seed entry, 63 measured kernel calls produced:

| Metric | Result |
|---|---:|
| Minimum | 59.517 ms/token |
| Maximum | 59.851 ms/token |
| Median | **59.563 ms/token** |
| Mean | **59.578 ms/token** |
| Speedup over Iter35, 75.062 ms | **1.260x** |
| Speedup over Iter32, 98.660 ms | **1.656x** |
| Speedup over eight-port baseline, 121.4 ms | **2.038x** |

The measured mean is within 0.40 ms of the reconstructed 59.982 ms static
schedule. Head-local fusion therefore removes the recurrent restore/save cost
almost one-for-one without introducing a material dynamic-stall gap.

Reproducibility hashes:

| Artifact | SHA-256 |
|---|---|
| `gdn_model.cpp` | `b0a380365d00a7535dd1256f62f6a21f97a3eee6158e3e4b53bb92ce2df5dafb` |
| Iter35 link configuration reused by Iter36 | `240855bd65ba1cf4525c88b34f9d07012bf73c90e75f524b2c9547eaa9e5922b` |
| Compiled XO | `c699dcc8c678cba970906d1a9ab0d62ab2315ad7d67c77fdcdf921dbd752171e` |
| Successful XCLBIN | `b251ca1c89a2d56111c1c336e003db3c3c0bbaf02556207a50a184578c6f3c34` |

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
7. Keep the recurrent buffer head-local and fuse HBM state transfer into the
   required retrieval/update traversals.
8. Keep LM-head argmax on chip and preserve strict first-index tie behavior.
9. Keep the external workspace offsets and host ABI stable.
10. Treat the Iter22 floorplan and Iter23/Iter35 DMA hooks as part of the successful
   physical architecture, not optional tuning.
11. Judge changes against both 100 MHz timing and zero-conflict routing; HLS
    resource savings alone do not predict routability.

Use [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md) for the
historical GEMV scaling progression and
[optimization_log.md](optimization_log.md) for the exhaustive build,
congestion, timing, and performance record.
