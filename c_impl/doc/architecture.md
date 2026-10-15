# GatedDeltaNet Decode Accelerator Architecture

**Status:** Current production architecture: the **225 MHz three-kernel image
with the re-mapped HBM channels (Iter88)**, promoted 2026-10-15 after the full
qualification (card gates on both cards, drift, WikiText-2, same-card A/B and
power; § *Correctness and On-Card Performance*). It is the Iter82–87 datapath
and schedule, unchanged, synthesized and linked at a true 4.444 ns, with one
connectivity change: the SLR2 kernel's eight HBM masters sit on channels 2–9
and the SLR0 kernel's ports 2–9 on channels 24–31, so that no die-crossing
master uses the channels whose shell pipes cross the SLR0/SLR1 boundary in
clock-region column X6 — which is what lets the shell's HBM AXI clock close at
its full 450 MHz. The Iter82–87 image is the **200 MHz three-kernel image
of the Iter82–87 series**, promoted 2026-10-13 after the full qualification
(drift, WikiText-2 and paired power, § *Correctness and On-Card Performance*).
It is the Iter81 image — the same three kernels, one per die of the U55C:
`gdn_forward_p` (control: ports 0/1/18–23, store, convolution, norms,
residuals, LM-head argmax) on SLR1, `gdn_k_slr0` (ports 2–17, GEMV clusters
1–8) on SLR0 and `gdn_k_slr2` (ports 24–31, clusters 12–15, the recurrent
islands and the state readers/writers) on SLR2, joined by seven registered
AXI-Stream links under one static 97-call schedule, with the Iter81
three-stage recurrent islands and prefetch-first state readers — plus four
schedule changes that hide the control kernel's serial work and the state
write-back: a packed BF16 aux-weight image read in bursts and both tiny a/b
projections in one fused II=1 loop (Iter83), the state write-back deferred
until each reader's QKVG weight loop has ended (Iter84), the residual add fused
with the following RMSNorm and the SwiGLU evaluated inside the GU store
(Iter85), and the per-head output norm/gate and the convolution-tail writer as
processes of the QKVG region (Iter86), the tail writes held until port 0's QKVG
weights are read (Iter87). Every change is element-wise or
order-preserving: native bit-exact and the production figures on card. The
image is HLS-synthesized and linked at **200 MHz** under **Vitis 2024.2**,
routed with zero routing errors. **The Iter88 image closes the kernel clock at
+0.035 / +0.010 ns at 4.444 ns, the fixed 250 MHz DMA clock at +0.003 / +0.009
ns and the 450 MHz HBM AXI clock at +0.039 / +0.010 ns** (the Iter82–87 image:
+0.027 at 5.000 ns, +0.003, and −0.040 on the HBM clock, run at 442.0 MHz under
the flow's 440 MHz floor). The 64-token card run measures **8.233 ms/token
production TPOT / 8.141 ms kernel = 1,831,725 cycles at 225 MHz** by median
over 63 generated tokens (build 10440 / job 10441, acclnode01's card; build
10168, the first draw of the recipe, 8.253 / 8.147 on the same card; the
Iter82–87 image: 8.484 / 8.361 ms = 1,672,120 cycles at 200 MHz), with an exact
token trajectory and the production vector-gate figures over 2,016,000 logits.
Against the Iter82–87 image in one job on harrier's card, the two images
alternating A B B A A B, Iter88 is **2.8 % faster** (8.488 → 8.253 ms), every
interval separated, at +2.7 % power and unchanged energy per token. The clock
rose 12.5 % but the token takes 9.6 % more cycles: the weight ports already
run at the HBM's sustained ceiling (§ *Physical Implementation*). Against
Iter81 on the same card (10.256 ms kernel over the WikiText run) the Iter82–87
image was **−18.5 % kernel time**; against the Iter82–86 image Iter87 was faster
by 16 µs per token in short runs and by 2.5 µs under the 4.5-minute power
protocol. The dual-kernel
call probe (Iter82, compiled out of production) measured every step
(`optimization_log.md`, Iter82–87). After Iter86 the four state ports'
traffic — 4,096 beats read before and 4,096 written after each layer's QKVG
weights per port — bounds the QKVG call: the floor is ≈ 62.5K cycles per layer
(≈ 1.564M per token).

Production XCLBIN SHA-256
`497c87255a20b418…` (build 10440, `diagnostics/hw_20261015_121355_3074494/`,
recipe `hw_f225_p_remap.cfg`, produced with its gates and qualification by the
single command `make -C c_impl run_hw`; build 10168's `74c8ec59605d3b81…`,
`diagnostics/hw_20261014_073923_2548162/`, is the first draw of the same
recipe, implemented identically to the picosecond),
from `gdn_model.cpp`
`b21f9b94d2993d5eb64fdc0ede43374c5b64fca31799138cb3a9a58f2ff195d4` (commit
`42959cfcf`, the Iter87 source) and the unchanged `host.cpp` `def56de3…`. The
Iter82–87 image `e3e88f96…` (build 8396, 8.484 ms/token at 200 MHz, the same
source with the original HBM map) is the immediate predecessor, the Iter82–86
image `6ee78021…` (build 8236, 8.486 ms/token, source `56622ec53`) and the
Iter81 image `a95b9a8a…` (build 7370, 10.470 ms/token on harrier's card) the
ones before. The
Iter79 image `0326f01be04ff628…` (build 6548, 12.350 ms/token; reproduced by
builds 6568 and 6642) is the recorded predecessor in § *Correctness and
On-Card Performance*, with the 150 MHz Iter76 image (`b2a0a478…`, 16.255 ms)
and Iter67c (`fb4fc63f…`, 24.208 ms at 100 MHz) before it. The GEMV datapath is
the 150 MHz image's datapath unchanged — Iter66e's all-BF16 arithmetic, the fused
strict argmax, the five-phase II=1 recurrent read, the Iter73a/75b registered
state write-back and the Iter73b2 straight-line cluster emit — cut along die
boundaries; the one arithmetic-neutral source fix the partition needed (start
propagation in the recurrent islands' dataflow region, see § *Design
Invariants* item 24) removed a hardware-only deadlock that no C simulation can
see. The integrated production flow reproduced the implementation a third time
from a clean snapshot (`make run_hw`, build 6642, image `0479e581…`, card job
6643: 12.387 ms TPOT, exact gates). The 150 MHz Iter76 image (`b2a0a478…`, 16.255 ms/token) and Iter67c
(`fb4fc63f…`, 24.208 ms/token at 100 MHz) are the recorded predecessors in
§ *Correctness and On-Card Performance*. Iter81 changed only the SLR2 kernel's
recurrent islands and state readers, plus the state readers' address form
(§ *Design Invariants* item 26).

**Read § *Arithmetic contract and what "correct" means* before treating any
hardware/native mismatch as a defect.** With BF16 the end-to-end bit-exact
gate no longer applies, for a measured and understood reason.

The accelerator in `c_impl/` is decode-only. Prompt prefill runs on the GPU and
exports fixed-size recurrent and convolution state. The FPGA then advances one
token per `gdn_forward` invocation. The successful design combines:

- 32 independent 512-bit HBM weight readers over **packed-BF16** shards;
- 16 two-port GEMV clusters whose products use a native `ap_float<16,8>`
  multiplier and reduce in FP32;
- **free-running cluster pipelines** (`style=frp`), which replace the former
  5,100--9,300-load per-stage clock-enable cones with interface handshakes;
- BRAM-backed MM2S/FIFO decoupling;
- a 4/6/6 cluster distribution across SLR0/SLR1/SLR2;
- transient activations resident in local BRAM for the complete 24-layer
  forward;
- four packed **BF16** recurrent-state stripes appended to HBM weight ports
  28--31, behind full-window 4,096-deep 512-bit URAM queues;
- two concurrent 16-column, head-local recurrent islands with HBM retrieval and
  update fused into the two required attention passes — since Iter81 each a
  three-stage head pipeline (prep → read → update, an 8,192-word URAM replay
  FIFO between read and update) whose interval, about 1,640 cycles per head,
  sits under the 2,048-cycle weight pace;
- a five-phase recurrent read schedule that maintains enough accumulator
  recurrence distance for II=1 without changing FP32 association;
- concurrent transfer from all four state ports into packed low/high URAM
  pairs;
- head-serial/all-port Q/K/V/gate (`QKVG`) and pair-interleaved MLP gate/up
  (`GU`) weight layouts, each issued as one shared-GEMV command;
- a time-shared head-local Q/K/V convolution actor that consumes each QKVG
  head while later heads are still streaming;
- immediate head-local Q/K/V streams from that convolution actor into the
  recurrent actor, avoiding whole-hidden Q/K/V materialization;
- state reads on ports 28--31 issued whole-layer (4,096 beats per port) before
  the call's weights, in the control kernel's pre-call gap, into four
  full-window depth-4096 URAM queues (Iter81; until then they were interleaved
  with each head's weights);
- duplicated Q/K/V head streams feeding the two recurrent islands, followed by
  a deterministic half-head merge;
- explicit registered relay actors between each 4/6/6 SLR-local GEMV collector
  and the final collector;
- one forward-only dataflow graph that overlaps later QKVG heads with
  convolution, state transfer, and recurrence of earlier heads;
- packed external recurrent-state and convolution-tail transfers; and
- on-chip LM-head argmax, plus (Iter61) the full GDN_VOCAB logit vector
  streamed out to the workspace for benchmark scoring;
- (Iter79) **three kernels, one per SLR**, from the same source: the GEMV
  clusters sit on the die of the HBM pseudo-channels they read, and seven
  registered AXI-Stream links carry the activation broadcast, the partial
  results, the convolved Q/K/V, the per-head gate scalars and the attention
  output between them — data only, no control, under one static schedule.

The architecture routes every routable net with zero routing errors and
preserves the exact 64-token trajectory. The in-link closure sequence closes
the 225 MHz kernel clock at **+0.035 ns WNS / +0.010 ns WHS**, the fixed
250 MHz DMA clock at **+0.003 / +0.009 ns** and the 450 MHz HBM AXI clock at
**+0.039 / +0.010 ns**. The production image runs **2.6 % faster by kernel
time** than the Iter82–87 image on the same card (8.147 against 8.361 ms) and
**20.6 % faster** than the Iter81 image (10.256 ms), and **14.7x** the
121.4 ms eight-port baseline by production TPOT (8.233 ms).

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

Recurrent arithmetic remains FP32, while its persistent
`24 x 8 x 256 x 256` state is stored as BF16, or 24 MiB. The three prior Q/K/V
convolution rows per layer are also stored as BF16, about 0.84 MiB. These two
state classes persist in HBM across kernel calls; transient activations do not.

## System Overview

```text
GPU prefill
    |
    | .gdnstate: recurrent state + Q/K/V convolution tails + seed token
    v
XRT host
    |-- keeps the original embedding table in host memory
    |-- builds 32 compact, output-row-striped GEMV weight shards
    |-- builds and validates one packed-BF16 non-GEMV auxiliary-weight image (Iter83)
    |-- writes the selected 8 KiB embedding row to workspace[X]
    v
+---------------------------------------------------------------------------+
| one token = three kernels started together, one static 97-call schedule     |
|                                                                             |
|  SLR0  gdn_k_slr0      ports 2-17 -> clusters 1-8 -> collect8 ---ys---+     |
|          ^ xr                                                         |     |
|          |                                                            v     |
|  SLR1  gdn_forward_p   workspace[X] -> activation BRAMs -> 24 layers         |
|          loader/x broadcast -> cluster 0, 9-11; ports 0/1/18-23             |
|          collect3 + final collector <-- ys from both dies                   |
|          store / Q-K-V conv -> conv, scalars ---> | attn <--- (64 beats)    |
|          norms, residuals, SwiGLU, final norm, LM head, strict argmax        |
|          + logit vector -> FIFO -> HBM0; workspace[X_NORM][0] = token       |
|          |  xr                                                 ^            |
|          v                                                     |            |
|  SLR2  gdn_k_slr2      ports 24-31 -> clusters 12-15 -> collect4 --ys-+     |
|          conv/scalars -> duplicate -> two recurrent islands -> merge        |
|          state readers (ports 28-31) -> URAM queues -> islands -> writers   |
+---------------------------------------------------------------------------+
```

Seven AXI-Stream links join the kernels (`register_mode=both` on both ends,
direct `stream_connect`, no FIFO inserted by the linker):

| Link | Direction | Width | Beats per call |
|---|---|---:|---|
| `xr_to_slr0`, `xr_to_slr2` | control → GEMV kernels | 512 | the activation, `in_dim/32` (64, or 176 for MLP-down) |
| `ys_from_slr0` | SLR0 → control | 64 | `row_groups × 64` result words (eight per cluster per row group) |
| `ys_from_slr2` | SLR2 → control | 64 | `row_groups × 32` |
| `conv_to_slr2` | control → SLR2 | 512 | 192 convolved Q/K/V beats on a QKVG call, else 0 |
| `scalars_to_slr2` | control → SLR2 | 512 | 2 beats (per-head a, b, a_log, dt_bias) on a QKVG call, else 0 |
| `attn_from_slr2` | SLR2 → control | 512 | 64 merged attention beats on a QKVG call, else 0 |

The host starts the two GEMV kernels, then the control kernel, and waits for all
three; the remote kernels block on their activation link until the control
kernel sends it. Every kernel derives the per-call dimensions, weight offsets
and the "recurrence on" flag from the same `gdn_call_schedule`, so the links
never carry control and the three kernels cannot disagree on a call.

The host writes an embedding rather than a token ID because the embedding table
remains in host memory. After the kernel completes, the host reads one 512-bit
line at the existing `GDN_WS_OFF_X_NORM` offset and interprets lane zero as the
next token ID.

## Kernel Interface and HBM Mapping

The three HLS tops together take the 34 pointer arguments the monolith took
(`aux_weights`, `workspace`, `weight_data_mm0` … `weight_data_mm31`) plus the
seven stream ports:

| Kernel | Die | Pointer arguments (AXI masters) | Stream ports |
|---|---|---|---|
| `gdn_forward_p` | SLR1 | `aux_weights`, `workspace`, `mm0`, `mm1`, `mm18`–`mm23` (8 masters) | `xr_to_slr0`, `xr_to_slr2`, `ys_from_slr0`, `ys_from_slr2`, `conv_to_slr2`, `scalars_to_slr2`, `attn_from_slr2` |
| `gdn_k_slr0` | SLR0 | `mm2`–`mm17` (16 masters) | `xr_in`, `ys_out` |
| `gdn_k_slr2` | SLR2 | `mm24`–`mm31` (8 masters) | `xr_in`, `conv_in`, `scalars_in`, `ys_out`, `attn_out` |

They synthesize to exactly 32 AXI masters. `aux_weights`, `workspace`, and shard
0 intentionally share `mem_weights_mm0`; shards 1-31 each have their own
master. The host maps every buffer to the bank of the master that reads it
(`port_group(c)`, from the image's connectivity), and the host ABI, workspace
offsets and the `.gdnstate` contract are unchanged from the monolith. The
production connectivity (Iter88; link-only, the host follows the image) is:

| AXI bundle | HBM bank | Contents |
|---|---:|---|
| `mem_weights_mm0` | HBM[0] | GEMV shard 0, compact auxiliary weights, workspace, convolution tails, embedding ingress, token egress |
| `mem_weights_mm1`, `mm10`–`mm23` | HBM[1], HBM[10]–[23] | One compact GEMV shard per bank, read-only during decode |
| `mem_weights_mm2` ... `mem_weights_mm9` (SLR0 kernel) | HBM[24] ... HBM[31] | One compact GEMV shard per bank, read-only; the right-hand channels, reached without a die crossing |
| `mem_weights_mm24` ... `mem_weights_mm27` (SLR2 kernel) | HBM[2] ... HBM[5] | One compact GEMV shard per bank, read-only; two die crossings on the shell's pipes |
| `mem_weights_mm28` ... `mem_weights_mm31` (SLR2 kernel) | HBM[6] ... HBM[9] | One GEMV shard plus one **6 MiB BF16** recurrent-state stripe per bank |

Why the SLR2 kernel's masters sit on the left-hand channels: the shell's
SLR-crossing register pipes for a channel land near that channel's column, and
columns X6–X7 at the SLR0/SLR1 boundary are shared with the static PCIe/DMA
shell. With the SLR2 masters on channels 24–31 those pipes failed the 450 MHz
HBM AXI clock on every draw (−0.018…−0.061 ns on the same 46 endpoints at 200
and 225 MHz); on channels 2–9 they land in columns X0–X4 and close with margin
(+0.039 and +0.069 ns on two draws). The original map (`weight_data_mmN` →
`HBM[N]`) lives on in `hw_f200_p.cfg`.

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

One weight shard is **1,366,528 beats = 87,457,792 bytes (83.4 MiB)** of packed
BF16, 32 values per 512-bit beat; the 32 shards hold exactly
1,399,324,672 dense parameters with no padding, so a token reads **2.799 GB of
weight bytes** (it was 5.598 GB in FP32).  That count is the five hidden-size
projections per layer (`q/k/v/g/o_proj`) plus `gate/up/down_proj`, times 24,
plus `lm_head`; the embedding table stays on the host.  It is confirmed twice
over: from the checkpoint tensor shapes, and from the reader's own bound
`total_weight_beats = rows_per_ch * k_packs` summed over the 97 GEMV calls,
which totals 1,366,528 beats per port and equals
`GDN_COMPILED_WEIGHT_SHARD_BEATS`.  (Earlier revisions of this document said
1,298,661,376 parameters / 2.597 GB / 1,268,224 beats; that omitted one
2048x2048 projection per layer, 98,304 beats per port.) Ports 28--31 append one 3,145,728-value
BF16 state stripe (**6 MiB**) after that fixed boundary. The small norms, A/B
projections, convolution kernels, and final norm are packed into
`aux_weights` — since Iter83 a host-built **packed BF16 beat image** (46,360
beats = 2.97 MB; per layer 1,929 beats: attn_norm 64, one FP32 beat of
a_log/dt_bias, o_norm 8, mlp_norm 64, the a/b projections 1,024 as beat
j·16 + c, the q/k/v conv kernels 3 × 256 in blob order; the final norm after
layer 23), validated lane by lane against the blob at load by
`gdn_validate_aux_image()`; it is not part of the `.gdnstate` ABI. The
consumers read it in 512-bit bursts (RMSNorm 64 beats, output norm 8, conv
context 3 × 256) and the two tiny a/b projections are one II=1 loop
(`gdn_gemv_tiny_ab`, 1,024 beats, the per-output accumulation order of the
retired per-output form). Measured by the Iter82 probe on card: the control
kernel's serial work between GEMV regions fell from 487.7K to 278.2K cycles
per token (−10.2 % of the token; per layer 11,168 → 4,865 before QKVG,
3,502 → 1,468 after o_proj, 3,529 → 3,244 after QKVG), with every GEMV region
unchanged.

Two weight sizes coexist and are easy to confuse. The `.gdnw` file on disk is
still a **5.87 GB FP32-container** blob whose values are required to be
BF16-exact — `gdn_validate_bf16_exact_weights()` rejects a non-conforming blob
before decode. The *device* image is the packed-BF16 shard set above. Neither
is "the weights are 5.6 GB" in the bandwidth sense; per-token traffic is
2.799 GB.

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
| Q/K/V, gate, attention, temporary and MLP offsets | Reserved but not accessed by the activation datapath |
| Logits offset (`GDN_WS_OFF_LOGITS`) | Written once per token by the top level, drained from the LM-head FIFO (Iter61) |
| `GDN_WS_OFF_REC_STATE` | Reserved compatibility space; recurrent state moved to weight-shard tails 28--31, stored BF16 |
| `GDN_WS_OFF_HEAD_BUF` | Persistent packed Q/K/V convolution-tail read/update/write |

The recurrent state still cannot remain on chip across the complete decode,
though BF16 narrowed the margin. One layer is now **1 MiB** and 24 layers are
24 MiB, which is 683 of the device's 960 URAMs by raw capacity — but this
design's 16-column partitioned access has historically cost about **1.68x raw**
(the FP32 version budgeted 96 URAM/layer against 57 raw), which puts full
residency near 1,152 URAM, over budget. The binding constraint is per-SLR
anyway: 320 URAM each, and the recurrent islands are pblock-pinned to SLR2,
which has 272 free — nine layers. See
[cycle_optimization_roadmap.md](cycle_optimization_roadmap.md) for the measured
ROI of partial residency.

Iter38 buffers one 256 x 256 head at a time; Iter66 stores those values BF16,
32 per Beat512. Within each state row, the eight low-half Pack16 words
alternate between
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
  -> norm_attn (layer 0: RMSNorm of the embedding row before the loop; later
     layers: from the fused residual + RMSNorm pass that ended the previous layer)
  -> tiny GEMV A and B -> partitioned scalar arrays
  -> load packed Q/K/V convolution weights and old tails
  -> shared GEMV aperture
       -> one head-serial/all-port QKVG command
       -> for each completed head while later heads stream:
            stream the head's gate to the norm/gate process
            run one shared head-local convolution actor for Q, K, and V
            stream the old tail rows 1-2 and the new raw row to the
            tails writer (mm0, 8-beat bursts; Iter86), which holds them in
            a 1,024-deep URAM queue until port 0's QKVG weights are read
            (one-token go from the loader; Iter87)
            stream convolved Q/K/V directly to recurrence
       -> on ports 28--31, before the call's first weight beat:
            stream the layer's four 4,096-beat old-state stripes into
            full-window URAM queues (hidden in the pre-call gap)
       -> duplicate Q/K/V into two recurrent islands
       -> recurrent islands, each a three-stage head pipeline:
            prep:   norms and alpha of head h+1
            read:   retrieval/partial pass of head h straight off the queue,
                    16 columns/cycle, replaying the state to the update stage
            update: decayed state + k x delta of head h-1, to the state writers
       -> merge the two half-head outputs and send them to the control kernel
       -> per head, as its attention arrives: output norm and gate
          (gdn_attn_norm_gate, the retired pass's arithmetic; Iter86)
  -> 64-beat copy of the gated attention into the engine's input
  -> output-projection GEMV
  -> residual accumulation into x fused with the MLP RMSNorm (Iter85: one pass
     forms the rounded sum beats and their sum of squares, then the scale pass)
  -> one pair-interleaved GU GEMV; the SwiGLU is evaluated per result word
     inside the store's fill loop and written out in the down projection's
     176-beat layout (Iter85), then copied into the engine's input
  -> MLP down GEMV
  -> residual accumulation into x fused with the next layer's attention
     RMSNorm (the final norm after layer 23)
```

The QKVG sink holds only one 256-element head. One convolution datapath is
called serially for that head's Q, K, and V, then emits them to a broadcaster
while the upstream graph produces later heads. Two independent recurrent
actors each consume all Q/K/V values but own disjoint 128-column state/output
halves; a small merge actor restores natural output order. Since Iter81 the
four state readers fetch the layer's whole state before the call's weights, so
every port streams 16,384 weight beats per QKVG call, and each island's read
stage takes the state beats straight off its full-window queue with
non-blocking reads — the queue holds the complete layer, filled before the
first head is needed — and hands every state word to the update stage through
an 8,192-word replay FIFO. The three stages overlap across heads, so an
island's interval is its slowest stage, about 1,640 cycles per head, below the
2,048-cycle weight pace. There is no feedback credit or readiness handshake in
the graph. The full-window state FIFO depth makes the strictly forward schedule
deadlock-free despite the registered GEMV collector boundaries.

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

### Iter79 — the kernel split

The 16 clusters are the same modules as before; what changed is which kernel
instantiates them and how their results travel:

| Kernel | Clusters | Ports | Collector |
|---|---|---|---|
| `gdn_forward_p` | 0, 9, 10, 11 | 0/1, 18–23 | `gemv32_collect3` (clusters 9–11) and the final collector |
| `gdn_k_slr0` | 1–8 | 2–17 | `gemv32_collect8` → `ys_out` |
| `gdn_k_slr2` | 12–15 | 24–31 | `gemv32_collect4` → `ys_out` |

The control kernel's loader (`gemv32_load_x_bcast`) writes every activation
beat to its local chain and to two relay FIFOs; two relay processes forward
them over `xr_to_slr0` and `xr_to_slr2`. Each GEMV kernel receives the beats
into its own chain head, so the per-cluster activation copy, the ripple and the
drain are the monolith's. Results come back as 64-bit words (one FP32 per port
of a cluster); the final collector (`gemv32_collect_final_p`) consumes, per row
group, 8 words from cluster 0, 64 from SLR0, 24 from clusters 9–11 and 32 from
SLR2 — the order the monolith's 4/6/6 collectors produced — so the store sees
the identical word stream and the result is bit-identical. On a QKVG call the
store's convolution output (192 beats), the per-head gate scalars (2 beats) and
the recurrent islands' merged attention (64 beats) cross between the control
kernel and SLR2 over three more links; on every other call those relays run
zero iterations. The relay FIFOs are 64 deep (the two activation copies, the
two returned result streams) and 32 or 16 deep (convolution and attention).
Per-token cost of the hand-offs, measured: +25,750 cycles (+1.06 %).

The partition is physical, not algorithmic: the control kernel is pinned to
SLR1 and the two GEMV kernels to the dies of their pseudo-channels
(`hw_f200_p.cfg` `slr=` lines), so the only paths that cross a die are the
seven links — each a registered AXI-Stream hop — and the shell's own HBM
pipes. That is what let the placer close 200 MHz where the monolith, at
97 % CLB on SLR0, failed by −1.6 ns and every pblock variant was unroutable
(Iter77/78 record in `optimization_log.md`).

### Iter81 — balanced recurrent islands and the prefetch-first state reader

Through Iter79 the QKVG call was paced by the recurrent islands, not by the
weight stream: each island ran one head at a time through a serial chain —
copy the head's state from the queue into a local array (1,027 cycles), prep
(load Q/K/V, norms, alpha, ~1,480), the five-phase read pass (1,291),
delta/output (~43) and the update loop (1,036) — 4,887 cycles per head,
39,100 per call, while ports 0–27 streamed 16,384 weight beats per call and
ports 28–31 20,480 (the 512-beat state slot after every head's 2,048 weight
beats). Iter81 makes the call stream-paced with two changes, both bit-exact
by construction (every value is the original expression with the original
operands in the original order):

- **Prefetch-first state reader** (`gemv32_mm2s_with_state_p`). In QKVG mode
  each of ports 28–31 fetches the layer's 4,096 state beats first, then all
  16,384 weight beats. The fetch sits in the ~9.4K-cycle gap in which the
  control kernel runs its norm, the two tiny projections and the convolution
  context before it broadcasts the activation, and the full-window queues hold
  it. The address is the state writer's proven form (item 26): the second
  Iter81 link read every state beat 128 MiB low after Vivado synthesis folded
  the earlier form's 27-bit offset, and a reader address gate now covers it.
- **Three-stage island** (`gdn_recurrent_attention_island_p<I>`): `prep`
  reads the head's Q/K/V packs, the two squared norms (tree reductions) and
  the merged normalize-and-alpha loop, and sends normalized q/k, the raw V
  beats and the three scalars; `read` loads q/k into BRAM copies, runs the
  five-phase II=1 read pass straight off the state queues (phases 0 and 2 take
  a low and a high beat with `read_nb` and cache the upper halves), writes one
  576-bit replay word per phase (16 low + 16 high state values, k and its copy)
  into an 8,192-deep URAM FIFO, then the delta/output loop; `update` reads the
  replay words back (`read_nb`), computes decay × old + k × delta per lane in
  the original order and writes the joined BF16 beats to the write-back
  queues. The dataflow region keeps start propagation on (item 24). csynth:
  prep 9,793, read 13,129, update 8,409 cycles per call; interval 13,244
  against 39,100 (RTL cosim 14,883–18,556 per call).
- **Two timing changes found by an endpoint census** of the new island in a
  quarter-SLR pblock: the update loop's 64 multipliers share the `decay`
  register (32 of them) and the `k` registers, which Vivado merges into inputs
  fanning out to 64 unregistered DSPs, so the four update products are bound
  one stage deeper (`bind_op fmul maxdsp latency=4`, a five-stage core); and
  the 8-trip delta loop runs at II=4 as a free-running pipeline, which removes
  its clock-enable cone. Out of context the island then closes at +0.429 ns
  against the original's +0.434 ns.

Cost per island (Vivado): LUT 76,893 vs 69,705, FF 65,745 vs 54,778, DSP 526
vs 353, BRAM 32 vs 46, URAM 16 vs 16. Measured on card: the token fell from
2,445,400 to **2,069,400 cycles** (−376,000, −15.4 %), against the ≈ −550,000
the stream-paced model predicts. The model's error was its "today" figure:
the old QKVG call was ≈ 50K cycles on card, not the csynth-derived 41.7K, so
Iter81's 15.5K-per-layer saving is the whole measured −376K. The dual-kernel
call probe (`GDN_CALL_PROBE`, Iter82: build 7602 / dump 7604, data-dependent
stamps on both kernels; the Iter81 probe's reading that SLR2 "never waits" and
that the Q/K/V convolution actor paces QKVG was a stamp artifact) measured the
2,053,983-cycle token exactly: the weight stream at one beat per cycle is
1,366,528 (66.5 %); the **control kernel's serial work between its GEMV
regions is 487,494 (23.7 %)** — per layer 11,168 before QKVG, 3,529 after it,
3,502 after o_proj, 2,061 after GU, 18 % above csynth where that code reads
scalar weights over mm0 — and the SLR2 kernel enters every region before the
control kernel does and waits there for the x broadcast; the QKVG region runs
6,001 cycles beyond its 16,384-beat stream (144K, 7.0 %) because ports 28–31
stream at 0.833 beats per cycle while the **state write-back** shares their
pseudo-channels (78.8K), the islands' last head follows the last Q/K/V beat by
2,018 cycles (48.4K) and the x broadcast and first results cost ≈ 620 (14.9K);
the other calls' fill and drain are 502 / 1,028 / 639 per layer (52K, 2.5 %);
the LM head 3.7K. The islands read a head in 1,622 cycles and update it in
1,044, so at the stream's 2,465-cycle head pace they wait ≈ 840 cycles per head
for Q/K/V — they are not the pacer; the convolution actor takes ≈ 600 cycles
per head; the state prefetch is done at 4.3K of the 11.1K pre-QKVG wait and is
hidden. The levers, in measured order: the serial work (packed aux image and
fused tiny GEMV, residual/SOS, per-head output norm, loader tails — Iter83
took the first two: −209.7K), deferring the state write-back until after the
weight loop (**Iter84, done: each state writer drains its 4,096-deep URAM queue
only after a one-token `wr_go` from its reader's weight loop; measured
−67.1K, the ports back at 0.985 beats/cycle and the collector's wait on SLR2
gone**), the residual add fused with the following RMSNorm and the SwiGLU inside
the GU store (**Iter85, done:** `gdn_residual_rmsnorm_bf16` and the store's fill
loop; measured −55.2K, the serial work outside the control regions 278.2K →
222.6K, after o_proj ≈ 900 and after GU ≈ 180 cycles per layer), the per-head
output norm/gate and the conv-tail writer inside the QKVG region (**Iter86, done:**
`gdn_attn_norm_gate`, `gdn_tails_writer`; measured −37.4K: the post-QKVG serial
work 3.2K → 0.13K per layer, partly returned as the islands' tail now exposed at
the o_proj boundary and the tails writes slowing port 0 — **Iter87, done:** the
writer drains a 1,024-deep URAM queue only after a one-token go from port 0's
loader, written after its weight loop; −0.5K to −3.2K cycles per token in
same-job A/Bs, capped by the state ports' write-back drain), the islands' last head, and the
per-call fill/drain; see `optimization_log.md`, Iter82–87.
**Measured structure after Iter86 (1,684,509 cycles on acclnode01's card, probe
build):** per layer on the SLR2 kernel 67.4K = weights 54.3K + state prefetch
4.4K + the QKVG tail 4.3K (the islands' last head and the deferred write-back) +
waits for the control kernel 1.9K + fill/drain 1.4K + region boundaries 4 × ≈ 265.
The four state ports' traffic (4,096 read + 4,096 written beats per port per
layer, before and after the QKVG weights) puts the floor at ≈ 62.5K per layer
(≈ 1.564M per token); cutting the control's pre-QKVG work cannot gain beyond the
prefetch it hides. The
monolith keeps the original island and reader
(`gdn_recurrent_attention_island`, `gemv32_mm2s_with_state`) for A/B.

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

| Resource | Iter37 | Iter38 | Iter39C | Iter54c | Iter57 | **Iter66e** |
|---|---:|---:|---:|---:|---:|---:|
| RAMB18 | 1,511 | 1,527 | 1,543 | 1,632 | 1,728 | **1,995** (49%) |
| DSP | 3,627 | 3,629 | 3,629 | 3,716 | 3,762 | **3,325** (36%) |
| FF | 851,903 | 857,343 | 863,589 | 888,271 | 921,774 | **1,002,888** (38%) |
| LUT | 912,694 | 929,941 | 937,707 | 868,961 | 869,262 | **895,268** (68%) |
| URAM | 48 | 48 | 48 | 48 | 48 | **80** (8%) |

Iter66e is measured under Vitis 2024.2, so its column is not a like-for-like
tool comparison with the 2022.2 columns: the same design re-estimated at
+520 BRAM18 purely from the version change, and that was carried as the leading
physical risk into place-and-route rather than treated as a regression. The
directions that *are* attributable to the design are DSP (-437, the native BF16
multiplier replacing FP32 operators) and URAM (+32, the four full-window state
queues).

### Iter61 — LM-head logit export

The LM head additionally emits its full 32,000-value logit vector. `gemv32_store`
pushes whole `Pack16` lines into an `hls::stream`; `gdn_forward` drains that
queue and writes `workspace + GDN_WS_OFF_LOGITS`, beside the token-id handoff.
The fused strict argmax is unchanged, so the decode trajectory and its
exact-match gate are untouched.

The queue, not a memory port, is the point. Two earlier attempts gave
`gemv32_store` its own AXI write; because the island pblocks distribute that
block across all three SLRs, a port there is a port everywhere, and
`route_design` refused at global congestion level 7. Filling a FIFO adds no
port to the GEMV region: BRAM is unchanged at 1728, URAM rises by 8 of 912
free, LUT by 0.4%, cycles by 0.003%.

Measured on card at 100 MHz: **42.170227 ms/token** median, 64 of 64 tokens
bit-identical to the GPU golden, post-route WNS +0.003 ns with zero overlaps and
zero failed nets. The export costs **+0.35%** against Iter57's 42.023540.

**Correction, and an open question.** This document previously reported the
recurrent call falling from 43,873--44,081 cycles in Iter37 to
**27,329--27,537 cycles** at Iter38D. That 27.3K figure cannot be reproduced
from any report currently on disk: every `csynth.rpt` from 2026-08-19 onward,
including builds that predate the BF16 work, reports
`gdn_recurrent_attention_islands` at **43,235--43,427 cycles per layer** —
essentially Iter37's number. Either the Iter57 split into two 16-column islands
restored the cost, or the two figures count different things (one island versus
the wrapper). It is recorded as unresolved rather than silently dropped,
because the block is **40.7% of the measured token** and a 1.6x error here
misprices every downstream decision. Resolving it needs one instrumented
csynth, not inference from these reports.

The remaining per-loop figures below are from the Iter66e report and are
current. The combined four-port state read and write loops
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

### Iter66e — all-BF16 arithmetic, BF16 state, and free-running pipelines

Five changes separate Iter66e from Iter61, and the order matters because only
the last one made the design routable.

**1. Packed BF16 weights.** Each 512-bit beat carries 32 BF16 values instead of
16 FP32. Per-token weight traffic halves to 2.799 GB. This alone did *not*
halve the token: the measured gain is **1.582x**, because at 2.799 GB the 32
ports are busy only 53.3% of the token and the design is no longer
bandwidth-bound.

**2. A native `ap_float<16,8>` multiplier.** `gemv32_four_dots` now emits 64
`floatingpoint_mul_16ns_16ns_16ns` instances and **zero FP32 multipliers**.
Against the exact-product BF16 reference this cut complete-cluster FF from
45,716 to 36,806 (-19.49%) and cluster DSP from 257 to 140; the hot weight loop
fell 40,322 to 31,042 FF (-23.01%) with DSP halving 256 to 128. The operator
itself is latency 2, II=1, 230 LUT, 173 FF, 0 DSP.

**3. BF16 recurrent state.** 32 values per Beat512, 24 MiB device-wide. Quality
cost is 0.02 accuracy points on Table 3, measured on GPU before adoption.

**4. Full-window state FIFOs.** The four 512-bit state queues moved from
depth-2048 BRAM to **depth-4096 URAM**, holding a complete layer window. This
is a liveness bound, not a tuning choice: it guarantees every later QKVG weight
beat is delivered before the queue can backpressure its reader, so a final
state-write stall has no dependency back to GEMV production.

**5. Free-running pipelines.** `#pragma HLS pipeline II=1 style=frp` on
`gemv32_cl_weight_stream`. The cluster datapath now runs always-valid instead
of gating each stage on a clock enable. The generated weight-stream RTL holds
**96 `ap_ce` references (interface handshakes) in place of the former per-stage
enable cones**, and the count of cluster nets above 2,000 pins fell 96 → 32.
Cost is +15 FF / +172 LUT per cluster and all 16 loops keep `yes(frp)` at II=1.

**Why this is the lesson of the campaign.** Iter66b--d attacked the same
high-fanout enable network with *constraints* — `FORCE_MAX_FANOUT`
clock-region replication, then `SOFT_HLUTNM`/`HLUTNM` un-pairing — and reached
16, then 5, then 3 overlaps without closing. Seven checkpoint repair jobs then
made every case worse (16→51, 5→22, 3→42), because per-net rerouting cannot
re-permute the site pins of routed neighbours and, where it can move wires, it
simply displaces contention into packed neighbours. Deleting the enable cones
at the HLS source closed the design in one attempt. The un-pairing hook is
retained (113,648 soft + 15,360 hard pairings cleared); the CE-replication hook
is deliberately dropped because frp supersedes it and its fail-closed
zero-match gate would abort a frp build.

## Arithmetic Contract and What "Correct" Means

The kernel's dense product is: **BF16 weight x BF16 activation -> RNE-rounded
BF16 product -> FP32 reduction** in four partial banks with balanced 16-product
trees. Matching AMD's Floating-Point Operator semantics in the *native*
reference cost five rejected qualification jobs, each on a different wrong
rule, so the measured rule is recorded here:

> Round to nearest-even **first**, then flush only if the rounded BF16 exponent
> is still zero. An exact product underflows to zero **unless** normalizing its
> 8-bit significand carries `0xff` -> `0x100` — equivalently, unless the exact
> FP32 magnitude exceeds `0x007fc000`.

Applying FTZ before RNE is wrong at the normal boundary, because an
FP32-subnormal exact product can round up into a normal BF16 result; the
halfway threshold `0x007f8000` is too permissive. `gdn_native_bf16_mul_to_fp32()`
is the reference implementation, validated over 2,572,928 exact products in
csim and 64 on card.

**The hardware/native bit-exact gate has been removed, and this is a decision
about what is measurable, not a relaxation of standards.** `LOGITS_REFERENCE`
defaults to empty, `--logits-reference` is a diagnostic, the on-card JSON
reports `exact_reference_required: false` for it, and only a non-finite value
still aborts a run -- a NaN or Inf can never be a rounding difference. The
native csim gate is untouched and still requires bit-exactness against the
cached golden and an independent scalar LM head; the independent-GPU
scale-aware vector gate remains the on-card gate and was not weakened.

The reason the old gate had to go: `expf` and `log1pf` are the only operations here that
IEEE-754 does not require to be correctly rounded. glibc and the AMD FPO cores
both conform and legally disagree in the last bit — measured on the real
operands, **82 of 384 per-head `decay` values (21.4%)** and 18 of 384 `beta`
values differ by 1--2 ULP. Each such scalar multiplies all 65,536 state
elements of its head, shifting them ~2^-24 relative, which is invisible after
BF16 rounding except where the FP32 result sits within that distance of an RNE
tie. The result is **129 of 12,582,912 BF16 state lanes off by +-1 BF16 ULP**,
uniformly scattered across 20 of 24 layers and all 8 heads, with the
convolution tails bit-identical and step-1 logits bit-exact. From step 2 those
lanes — now off by 2^-9 — produce macroscopic logit differences.

Everything else in the arithmetic was audited and cleared by direct RTL-vs-C
measurement: FP32 multiply, divide, sqrt and add are bit-identical for normal
operands; hardware emits no fused multiply-add; HLS unsafe-math and
reassociation are absent. One real hazard is latent but excluded as the cause:
**every HLS FP32 adder is DAZ/FTZ**, and csim is structurally blind to it
because a `bind_op` has no effect in C. It is excluded here only because the
native state dump computed with MXCSR FTZ+DAZ enabled is byte-identical to the
default, so no subnormal arises in this path.

Two methodology traps from that investigation, worth not repeating:

1. **An isolated probe can synthesize a different core than the kernel does.**
   An early probe placed the transcendental in an `II=1` pipelined loop and
   cleared it; the kernel's scheduling context selects a different FPO
   implementation.
2. **Grid sweeps are weak evidence for transcendentals.** An 8,192-point linear
   sweep passed while 21% of the *real captured operands* failed. Audit
   transcendentals on captured operands, never on a synthetic grid.

A fix exists and is deliberately unapplied: `gdn_exp_reproducible` /
`gdn_log1p_reproducible` (range-reduced forms using only IEEE-mandated
operations, preserved at
`diagnostics/iter66m_head_scalars/PROPOSED_FIX.cpp`) would make native and RTL
identical. It is not applied because it would make hardware match *native*,
which itself forks from the GPU at step 83 — see the drift measurement under
*Correctness and On-Card Performance*.

## Physical Implementation

The production build compiles HLS at 225 MHz under Vitis 2024.2 and links at
a true 225 MHz. All hardware work goes through Slurm as chained jobs, and
`make run_hw` on the login node is the one command for the whole flow: the
build, the `afterok` card gates, and the `afterok` qualification (512-token
drift and WikiText-2, then the power protocol's FPGA arm):

```bash
cd c_impl
make run_hw                                  # BUILD_CPUS=8 BUILD_MEM=96G on a busy node; QUAL=0 stops after the gates
```

Every knob the design needs is already the default: `HW_CFG_TEMPLATE=hw_f225_p_remap.cfg`,
HLS and link at 225 MHz, the three kernels, the BF16 weight blob, the
native-product `.gdnstate` and golden, the GPU reference logits, and the
qualification's references (the 512-step GPU logit dump, the WikiText-2 GPU
score, the 4,096-token power fixture — regenerable with the `scripts/`
producers; the power job generates the fixture on a GPU if it is missing).
Every card-side job builds its host on its own node from the frozen source
snapshot (`slurm/build_host_on_node.sh`), so the card jobs run on any node
with a free U55C.
`LOGITS_REFERENCE` defaults to empty because the hardware/native comparison is
a diagnostic, not a gate. `BUILD_EXCLUSIVE=user` keeps other users' jobs off the
node for the whole link (Iter76 evidence: three quiet-node links of one netlist
reproduced each other checksum for checksum, the one sharing its node diverged
in the placer's physical synthesis); the exclude list reflects measured
platform gaps (`acclnode04/05` have no U55C platform) and `harrier`'s full
node-local disk. The two 200 MHz links of the promoted netlist (builds 6548 on
`acclnode03`, 6568 on `acclnode01`) implemented identically without
exclusivity, so it is a recommendation, not a requirement.
`hw_f150.cfg` (the 150 MHz monolith recipe, with `xo_mono`/`xclbin_mono`),
`hw_iter66e_frp_unpair_f100.cfg` (100 MHz) and `hw_f150_physical_islands.cfg`
(pre-Iter66) are retained for A/B comparison only.

The production recipe lives in `hw_f225_p_remap.cfg` and its hooks
(`hw_f200_p.cfg`, the Iter79–87 recipe, differs only in the HBM map and the
clock it is linked at):

- `nk=` one compute unit per kernel, `slr=` pinning `gdn_forward_p_1` to SLR1,
  `gdn_k_slr0_1` to SLR0 and `gdn_k_slr2_1` to SLR2; one `sp=` line per master
  with the one-bank-per-port HBM map of § *Kernel Interface and HBM Mapping*
  (`aux_weights`/`workspace`/`mm0` on HBM[0]; the SLR2 kernel on HBM[2..9],
  the SLR0 kernel's ports 2–9 on HBM[24..31]); seven `stream_connect=` lines.
- `apply_p_islands.tcl` (OPT_DESIGN.PRE): sets the floorplan hand-off and
  sources `apply_iter69_kernel_clock_f150.tcl`, which overrides the kernel-clock
  XDC that Vitis 2024.2 otherwise implements at 10.000 ns (target from
  `GDN_KERNEL_TARGET_MHZ`, exported by the build job) and then sources
  `apply_p_islands_body.tcl`: the two recurrent islands of `gdn_k_slr2_1` go into
  quarter-SLR pblocks (`CLOCKREGION_X0Y10:X3Y11` and `X4Y10:X7Y11`). Nothing
  else is floorplanned; the kernels' `slr=` assignment does the rest.
- `SSI_SpreadLogic_high` placement, `AlternateCLBRouting`, pre- and post-route
  `AggressiveExplore`.
- `finish_p_timing.tcl` (POST_ROUTE_PHYS_OPT_DESIGN.TCL.POST): if kernel setup is
  still negative, one `Explore` pass, then one focused kernel path-group pass;
  then `check_p_final_timing.tcl` — the exact-clock gate on all three clocks,
  route status, DRC and bus skew; any failure aborts the link. The HBM AXI
  clock's closure at 450 MHz was, through Iter87, a per-draw ±0.02 ns coin on
  the shell's own SLR-crossing pipes in clock-region column X6 (Iter79 P6
  +0.015 ns; the Iter81 links +0.000, then −0.021 twice on identical inputs;
  −0.018 and −0.040 on the Iter86/87 images) — the column the static shell
  shares, and the shell's crossing flops cannot use the Laguna registers under
  the platform's reconfigurable pblock (Iter88 diagnosis). Since Iter81
  `run_hw_sbatch.sh` defaults `GDN_HBM_MIN_MHZ=440`: the gate accepts a miss
  that a clock scaled to that floor absorbs, vpl programs the achieved
  frequency and the build job derives the matching tolerance for the metadata
  reconciliation. Since Iter88 the re-mapped channels keep those pipes out of
  column X6 and the clock closes at exactly 450 MHz (+0.039 and +0.069 ns on
  two draws); the floor stays as the safety net. An explicit empty value
  demands exact 450 MHz.
- `reconcile_exact_clock.py` after the link, as before (a no-op on these
  images: vpl wrote DATA_CLK 200 natively because every clock closed).

Routed utilization of the Iter88 production image (build 10168): CLB slices
50,422 / 49,571 / 49,930 (91.74 / 91.80 / 92.46 %), LUTs 202,119 / 232,073 /
252,651 (45.97 / 53.72 / 58.48 %), DSP 2,157 / 1,549 / 2,137, BRAM tiles 429 /
541.5 / 298.5, URAM 0 / 64 / 96 per SLR; SLLs 24,869 (SLR1↔SLR0 57.60 %,
SLR2↔SLR1 50.34 %) — the Iter82–87 image's numbers within 0.3 points, the
re-map having moved the shell's crossings without adding any. csynth at
4.444 ns per kernel (`gdn_forward_p` / `gdn_k_slr0` / `gdn_k_slr2`): BRAM18
604 / 697 / 472, DSP 1,032 / 1,121 / 1,609, FF 249K / 279K / 336K, LUT 252K /
236K / 320K, URAM 64 / 0 / 96. For the record, the routed utilization of the
first three-kernel image (build 6548, identical on 6568 and 6642), beside the
150 MHz monolith (build 4022) like for like:

| | SLR0 | SLR1 | SLR2 |
|---|---:|---:|---:|
| CLB slices occupied, 200 MHz image | 50,199 (91.3 %) | 49,348 (91.4 %) | 49,634 (91.9 %) |
| CLB slices occupied, 150 MHz monolith | 53,375 (97.1 %) | 40,917 (75.8 %) | 42,002 (77.8 %) |
| CLB LUTs, 200 MHz image | 202,602 (46.1 %) | 227,761 (52.7 %) | 247,187 (57.2 %) |
| CLB LUTs, 150 MHz monolith | 266,530 (60.6 %) | 205,786 (47.6 %) | 210,884 (48.8 %) |
| CLB registers, 200 / 150 | 314,767 / 363,058 | 303,298 / 227,627 | 283,555 / 239,356 |
| DSP, 200 / 150 | 2,154 / 2,378 | 1,465 / 1,681 | 1,792 / 1,486 |
| URAM, 200 / 150 | 0 / 0 | 56 / 34 | 96 / 78 |

The LUT total barely moved (677,550 against 683,200); it moved *off SLR0*
(−24 % there) onto the other two dies, the slices evened out at ~91 %, and
SLL use fell by a quarter: SLR1↔SLR0 13,228 (57.4 %, was 19,749 = 85.7 %),
SLR2↔SLR1 11,605 (50.4 %, was 13,216 = 57.4 %), 24,833 in total (was 32,965). csynth per kernel (`gdn_forward_p` / `gdn_k_slr0` / `gdn_k_slr2`):
BRAM18 644 / 697 / 494, DSP 949 / 1,121 / 1,263, FF 250K / 278K / 307K, LUT
249K / 236K / 308K, URAM 56 / 0 / 96 — 1,835 BRAM18, 3,333 DSP, 835K FF, 792K
LUT, 152 URAM in all, each kernel at an estimated 3.650 ns.

### The 150 MHz monolith recipe (predecessor)

The text below describes the Iter76 recipe (`hw_f150.cfg`) that the three-kernel
recipe replaced; its build-job mechanics are unchanged.

The build job freezes a source snapshot, stages it on node-local disk, and runs
the native trajectory gate, the XO architecture gate and the synthesized
state-address gate before the link. The link runs the in-link closure hook and
its fail-closed exact-clock gate; then `reconcile_exact_clock.py` patches the
XCLBIN's `DATA_CLK` to the requested 150 MHz — Vitis derives `AUTO-FREQ-SCALING`
from the timing report it writes *before* the hook runs, so a closed design
otherwise ships labelled 149 MHz — but only when the hook's gate file proves
every clock non-negative at the requested period, never touching the bitstream.
The `afterok` card job selects the U55C by BDF, verifies the copied image and
host by hash, and runs the 8- and 64-token gates with the submission's frozen
`reproduction.Makefile` (`GDN_ONCARD=1`, no build prerequisites), so an 8-core
job can never start a link. Usage, outputs and the evidence boundary are in
[reproduce_f150.md](reproduce_f150.md).

The launcher freezes the complete chain into its submission snapshot:
`Makefile`, `gdn_model.{cpp,h}`, `host.cpp`, `hls_gdn_forward.tcl`, the three
link configs, `apply_iter69_kernel_clock_f150.tcl`,
`apply_iter75d_control_fanout.tcl`, `apply_iter66e_unpair.tcl`,
`apply_f150_physical_islands.tcl`, the three chained DMA hooks,
`check_f150_physical_islands.tcl`, `finish_f150_timing.tcl`,
`check_iter75d_final_timing.tcl`, `report_final_qor.tcl`,
`check_native_bf16_xo.py`, `check_state_writer_addresses.py` and
`reconcile_exact_clock.py`. Four inputs are gitignored and must be regenerated
first — the weight blob, the GPU reference logits, the `.gdnstate` handoff, and
(optionally) the native reference. See the root `CLAUDE.md` § *Reproducing a
hardware build from a clean clone*.

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
  collector FIFOs; the four state queues are **depth-4096 URAM** full-window
  queues;
- `SOFT_HLUTNM`/`HLUTNM` un-pairing on the three families every measured pin
  conflict named — `gemv32_four_dots` cones, `gemv32_cl_flush` pipelines, and
  `m_axi` `bus_write`/`fifo_burst` counters — trading SLR0's spare LUTs for
  pin-assignment freedom (113,648 soft plus 15,360 hard pairings cleared);
- a post-place structural gate that verifies all four pblocks and reports SLR,
  SLL, CLB, and BRAM pressure without prematurely rejecting a routable design;
- the Iter69 override of vpl's `_user_impl_clk.xdc` at OPT_DESIGN.PRE, without
  which Vitis 2024.2 implements the kernel clock at 10.000 ns regardless of
  `--kernel_frequency`;
- Iter75d's eight measured pre-place control-fanout repairs (`FORCE_MAX_FANOUT`
  32 on the SLR0 kernel reset, 8 on `gdn_gemv`'s `ap_start`,
  `ap_CS_fsm_reg[91]`, two RMSNorm and two output-norm address truncations and
  a `gemv32_store` state bit), the families a locality census measured as the
  largest slice of the 150 MHz residual;
- `SSI_SpreadLogic_high` placement and `AlternateCLBRouting` routing (the
  100 MHz recipe used `SSI_SpreadSLLs` and `NoTimingRelaxation`);
- pre-route and post-route `AggressiveExplore` physical optimization; and
- the in-link finishing hook (`finish_f150_timing.tcl`): if kernel setup is
  still negative after post-route `AggressiveExplore`, one `Explore` pass, then
  one focused kernel-path-group pass (`phys_opt_design -placement_opt
  -routing_opt -critical_cell_opt -path_groups clk_kernel_00_unbuffered_net`),
  then the exact-clock gate on all three clocks, route status, DRC and bus
  skew, each of which aborts the link on failure.

Final implementation result of the production image (build 4022; build 3987
and closure jobs 3953/3955 are checksum-identical at every phase):

| Metric | 150 MHz production (build 4022) |
|---|---:|
| Requested / encoded kernel clock | **150 / 150 MHz** (metadata reconciled from vpl's 149) |
| Failed / unrouted nets | 0 / 0 of 1,592,229 |
| Setup WNS / TNS (design-wide) | **0.000 ns / 0.000**, 0 failing of 2,039,265 endpoints |
| Hold WHS / THS | **+0.002 ns / 0.000**, 0 failing of 2,036,610 |
| Setup / hold, **kernel clock** at 6.667 ns | **0.000 / +0.002 ns** |
| Setup / hold, fixed 250 MHz DMA | +0.003 / +0.009 ns |
| Setup / hold, fixed 450 MHz HBM | +0.052 / +0.010 ns |
| Closure ladder inside the link | −0.013 (AggressiveExplore) → −0.001 (Explore) → **0.000** (focused pass) |
| Automatic clock scaling | **none** (the gate aborts the link on any negative slack) |
| Total build, 48 CPUs on `acclnode01` | **12:23:54** (fresh XO ≈ 0:40, link ≈ 11:00 including 3:40 of closure passes and reports) |

The 100 MHz predecessor, for comparison:

| Metric | Iter66e |
|---|---:|
| Requested / encoded kernel clock | **100 / 100 MHz** |
| Failed / unrouted nets | 0 / 0 |
| Node overlaps | **0** |
| Setup WNS / TNS (design-wide) | **+0.003 ns / 0.000**, 0 failing of 2,277,369 endpoints |
| Hold WHS / THS | **+0.009 ns / 0.000**, 0 failing |
| Setup WNS, **kernel clock** | **+0.195 ns** over 1,580,815 endpoints |
| Setup WNS, fixed 250 MHz DMA | +0.003 ns over 307,211 endpoints |
| Setup WNS, fixed 450 MHz HBM | +0.079 ns over 270,010 endpoints |
| Automatic clock scaling | **none** |
| `route_design` wall time | **1:30:56** (every prior attempt ground 5+ h and failed) |
| Total build | 8:07:32 on `harrier`, job 2502 |

Raw post-route setup was WNS -0.017 / TNS -0.132; post-route
`AggressiveExplore` recovered it to the values above. Both the kernel and the
fixed 250 MHz DMA clock domains are clean.

**Read the per-clock rows, not the design-wide one.** At 100 MHz the
design-wide +0.003 ns belonged to the *fixed* 250 MHz `dma_ip_axi_aclk_1`
while the kernel clock had +0.195 ns, which is why frequency was pursued. That
headroom is now spent: at 150 MHz the kernel clock closes with **0.000 ns** to
spare, so any change to the netlist is a timing miss until it re-closes, and
the Iter69–75 campaign measured the wall as wire and SLL congestion, not logic
(the unchanged Iter67c netlist at 6.667 ns missed by −1.594 ns; source-side
locality work and the closure passes bought the rest). Port occupancy is not
invariant with clock either: one HBM pseudo-channel peaks at 14.4 GB/s, a busy
512-bit port draws 9.6 GB/s at 150 MHz (66.7% of peak), 12.8 GB/s at 200 MHz
and 16.0 GB/s at 250 MHz, and the last cannot sustain one Beat per cycle. The
full floor model is in [frequency_250mhz_roadmap.md](frequency_250mhz_roadmap.md).

Routed whole-device usage and its per-SLR distribution (build 4022,
`report_utilization -slr` on the closed design):

| | SLR0 | SLR1 | SLR2 | device |
|---|---:|---:|---:|---:|
| CLB sites occupied | **97.12%** | 75.77% | 77.78% | 136,294 |
| CLB LUTs | 266,530 (60.62%) | 205,786 (47.64%) | 210,884 (48.82%) | 683,200 |
| CLB registers | 363,058 | 227,627 | 239,356 | 830,041 |
| Block RAM tiles | 554.5 (82.51%) | 461 (68.60%) | 357.5 (53.20%) | 1,373 |
| URAM | 0 | 34 (10.63%) | 78 (24.38%) | **112 of 960** |
| DSPs | 2,378 (82.57%) | 1,681 (54.72%) | 1,486 (48.37%) | 5,545 |

SLL usage is **SLR1<->SLR0 19,749 of 23,040 (85.72%)** and SLR2<->SLR1 13,216
(57.36%), 32,965 total. Against Iter66e (96.53 / 94.68 / 76.09% CLB sites,
SLL 89.57%) the Iter73/75 source changes moved logic out of SLR1 and onto the
SLR2 side, which is where the recurrent islands and their new state-writer
FIFOs live; SLR0 remains the full SLR.

**Two numbers to treat carefully.** URAM is the one abundant resource — 848
free — but it is 320 per SLR, and the recurrent islands are pinned to SLR2,
so device-wide headroom is not usable headroom. And the routed DSP total of
5,545 is **2,092 above the 3,453 csynth estimate** (Iter66e: 5,401 vs 3,325),
where Iter57's routed total exceeded its estimate by only 10; the gap is
measured but not explained here, and DSP columns were reported at 94--100%
saturation inside every level-7 congestion window during the failed attempts.
Any additional storage, arithmetic, or cross-SLR control still requires a new
full route rather than inference from HLS resources.

### Previous Iter36 frequency result

For comparison, the Iter36 source was built with a 130 MHz link request. Its
per-iteration launcher has since been retired from the active tree; the exact
command and hashes remain in `optimization_log.md` and the launcher remains
recoverable from Git history. Current builds use `make -C c_impl run_hw`.

Post-route `AggressiveExplore` improved the kernel paths but did not close the
7.692 ns requirement. At the requested 130 MHz, the post-physopt report has
WNS/TNS **-0.948/-10111.593 ns**, 19,904 failing setup endpoints, and WHS
**+0.001 ns**. Vivado auto-frequency scaling encoded `DATA_CLK` at an achieved
**115.7 MHz**. The fixed 250 MHz DMA same-clock setup path remained positive.
The build completed in 31 h 31 m and emitted a valid XCLBIN; this is a routed,
on-card-validated 115.7 MHz result, not a 130 MHz closure result.

## Correctness and On-Card Performance

### Iter88 image, 225 MHz with the re-mapped HBM channels (current)

The Iter87 source, synthesized at 4.444 ns and linked by `hw_f225_p_remap.cfg`
(the Iter79 physical recipe with eight `sp=` pairs swapped). It passed the
production chain — native gates, the XO gate (every reader II=1 at 4.444 ns),
the state writer/reader and convolution-tail address gates, the island gate,
the in-link exact-clock gate with every clock at its nominal period, route,
DRC and bus skew, the clock-metadata check, the 8- and 64-token card gates —
on both cards, and the qualification on harrier (build 10168, job 10169 on
acclnode01, 2026-10-14; jobs 10401/10402/10404/10406 on harrier). Then the
one-command flow reproduced it: `make -C c_impl run_hw` with no knob gave
build 10440 — the same three slacks to the picosecond — and ran the gates and
the qualification itself on acclnode01 (jobs 10441/10442/10443, 2026-10-15,
8 h 22 m end to end); that image is the production image:

| Metric | build 10440, jobs 10441–10443 (production, acclnode01) | build 10168, first draw (acclnode01 / harrier) | Iter82–87 image, build 8396 / job 8397 |
|---|---:|---:|---:|
| Production TPOT, median of 63 | **8.233 ms/token** | 8.253 | 8.484 ms/token |
| Kernel execution, median of 63 | **8.141 ms/token** (mean 8.143) | 8.147 (mean 8.149, 8.133–8.201) | 8.361 (mean 8.362, 8.353–8.413) |
| Effective kernel cycles | **1,831,725 at 225 MHz** | 1,833,075 | 1,672,120 at 200 MHz |
| Same-job A/B on harrier, short runs (job 10402) | — | **8.2533 ms** (6 intervals, 8.2392–8.2705) | 8.4879 (8.4697–8.5061) |
| 512-token drift (fork, ratio, slope) | **BOUNDED**: 447, 1.063, 2.133e-06 (job 10442) | the same (job 10404, harrier) | the same (job 8398) |
| WikiText-2 word perplexity, 314,843 tokens | **16.774839771** (−0.0077 %), 8.141 ms/token | 16.774839771, 8.237 ms/token (harrier) | 16.774839771, 8.365 ms/token |
| Power, FPGA arm (same card as the column's gates) | **71.4 W, 33.2 W idle, 0.5895 J/token gross, 0.3154 idle-subtracted** (job 10443) | 71.5 W, 0.598 J, 0.317 J (job 10406, harrier, with the predecessor at 69.6 W, 0.599 J in the same job) | 69.9 W, 31.7 W, 0.5929 J, 0.3240 J (job 8239) |
| 8- and 64-token trajectories | exact | exact | exact |
| CUDA vector gate, 2,016,000 logits | NRMSE 0.004663, worst step 0.011908, cosine 0.999989, 0 argmax mismatches | identical | identical |
| Kernel / DMA / HBM clocks | +0.035 / +0.003 / **+0.039** ns (HBM at 450.0 MHz) | +0.035 / +0.003 / +0.039 | +0.027 / +0.003 / −0.040 ns (HBM at 442.0 MHz) |

Against the latest paired A100 eager arms (jobs 8399/8400, not re-run because
they do not depend on the image: 31.32 ms / 95.4 W / 2.998 J gross / 0.890 J
idle-subtracted in fp32, 31.57 ms / 85.6 W / 2.711 J / 0.585 J in bf16) the
production image is **3.80x / 3.83x faster** and **5.09x / 4.60x** more
efficient in gross energy (**2.82x / 1.85x** idle-subtracted).

Why only 2.8 % for 12.5 % more clock: the achieved-bandwidth diagnosis (Iter88
log entry) measured the HBM's sustained sequential-read ceiling at **12.87 GB/s
per pseudo-channel** (`microbench/hbm_rd`, 89.4 % of the 14.4 GB/s pin rate)
and, with the probe build of this image (10417), every weight port streaming at
**0.876–0.893 beats per 4.444 ns cycle = 12.6–12.85 GB/s, 98–100 % of that
ceiling** (at 200 MHz: 0.969–0.989 = 12.4–12.66 GB/s). A 512-bit port's demand
meets the ceiling at 201 MHz; above it the clock turns into stalled beats, and
only the non-stream work gets faster. The one-command reproduction of this
image is the table's first column; the log's promotion entry records the run.

### Iter82–87 series image, 200 MHz (predecessor)

Iter87 changes one thing in the control kernel: the convolution-tail writer's
576 writes per QKVG call wait for a one-token go that the port-0 loader writes
after its weight loop, so they no longer share HBM pseudo-channel 0 with the
weight reads. The image passed the same production chain (native gates, XO
gate, the state writer/reader and convolution-tail address gates, the island
gate, the in-link exact-clock gate, route/DRC/bus-skew, clock metadata, card
gates; build 8396, job 8397, 2026-10-13, acclnode01's card), with a new
pre-link control-region RTL cosim (no deadlock, checksums equal to Iter86's):

| Metric | build 8396, job 8397 (production) | Iter82–86 image, build 8236 / job 8237 |
|---|---:|---:|
| Production TPOT, median of 63 | **8.484 ms/token** | 8.486 ms/token |
| Kernel execution, median of 63 | **8.361 ms/token** (mean 8.362, 8.353–8.413) | 8.369 (mean 8.370, 8.360–8.424) |
| Effective kernel cycles at 200 MHz | **1,672,120** | 1,673,800 |
| Same-job A/B, short runs (job 8419: no idle, 10 s warm-up, 2 × 30 s) | **8.3915 ms** (6 intervals, 8.3907–8.3999) | 8.4076 (8.4063–8.4085) |
| Same-job A/B, power protocol (job 8470: 60 s idle, 30 s warm-up, 3 × 60 s) | **8.4059 ms** kernel median (runs 8.4053 / 8.4064) | 8.4083 (8.4091 / 8.4076) |
| Same-job energy, power protocol (job 8470) | 0.600 J gross, 0.3217 J idle-subtracted | 0.595 J gross (runs 0.591 / 0.599), 0.3225 |
| 8- and 64-token trajectories | exact | exact |
| CUDA vector gate, 2,016,000 logits | NRMSE 0.004663, worst step 0.011908, cosine 0.999989, 0 argmax mismatches | identical |
| Kernel / DMA / HBM clocks | +0.027 / +0.003 / −0.040 ns (HBM at 442.0 MHz) | +0.041 / +0.003 / −0.018 ns (HBM at 446.4 MHz) |

The A/B rows are the controlled comparisons: one card job each, one host
binary, the two images alternating over the power benchmark's 4,096-token
context. Iter87 is faster in every run of both jobs, but by a margin that
depends on the run: **−16 µs per token (−0.19 %, the predicted −14 µs) in the
short runs, −2.5 µs (−0.03 %) under the 4.5-minute power protocol**. Its kernel
time moves by up to 15 µs between runs (8.391 in the first runs of job 8419,
8.400 in its last, 8.406 in job 8470) while the predecessor's stays within
3 µs (8.406–8.409); the faster runs came first in a job, before minutes of load
— cause not identified (candidate: card temperature with this image's lower
442 MHz HBM clock). Energy per token is unchanged within noise: the two
predecessor runs of job 8470 differ by 1.4 % in gross energy, more than the
images do. `kernel_ms` is timed by the host and carries tens of microseconds of
host latency that moves between jobs (the predecessor's own median moved 32 µs
between its power run and job 8419), so cross-job figures cannot resolve a
change this small.

Qualification on this image, 2026-10-13: 512-token free-running drift
**BOUNDED**, fork at step 447, ratio 1.063, slope 2.133e-06 per step (job 8398);
WikiText-2 word perplexity **16.774839771** against GPU 16.776123769 on the
verified-identical workload, **−0.0077 %**, identical to the predecessor's to
every printed digit, kernel 8.365 ms/token over the run (8.371 for the
predecessor, cross-job); paired power (jobs 8399 fp32 / 8400 bf16 arms): 70.0 /
69.6 W active, 31.7 / 31.1 W idle, **0.598 / 0.594 J/token gross, 0.327 / 0.328
J idle-subtracted** at a sustained 8.53 ms TPOT, against the paired A100 eager
runs' 31.32 ms / 95.4 W / 2.998 J (FP32) and 31.57 ms / 85.6 W / 2.711 J (BF16):
3.68x / 3.71x faster, 5.02x / 4.57x gross and 2.72x / 1.78x idle-subtracted
energy efficiency. These power runs met a slower host than the predecessor's
(every host-side time 20–100 % longer, the A100 arm's device time too), so their
TPOT and energy per token are higher than the predecessor's by host time, not
device time (the A/B above). The card reads back DATA_CLK 200 MHz and hbm_aclk
442 MHz.

### Iter82–86 series image, 200 MHz (predecessor)

The image passed the production chain — the native trajectory gates on the
three-thread C model, the XO architecture gate, the synthesized address gates on
the state writers and readers of `gdn_k_slr2` and (new) on the control kernel's
convolution-tail writer, the island-name and HLS 200-656 gate, the in-link
exact-clock gate (kernel and DMA exact, HBM within the 440 MHz floor), route,
DRC and bus-skew gates, the clock-metadata check, and the 8- and 64-token card
gates (build 8236, job 8237, 2026-10-11, acclnode01's card):

| Metric | build 8236, job 8237 (production) | Iter81 image |
|---|---:|---:|
| Production TPOT, median of 63 | **8.486 ms/token** | 10.470 ms/token (harrier's card) |
| Kernel execution, median of 63 | **8.369 ms/token** (mean 8.370, 8.360–8.424) | 10.347 (harrier); 10.256 over the WikiText run on acclnode01's card |
| Effective kernel cycles at 200 MHz | **1,673,800** | 2,069,400 (harrier) |
| 8- and 64-token trajectories | exact | exact |
| CUDA vector gate, 2,016,000 logits | NRMSE 0.004663, worst step 0.011908, cosine 0.999989, 0 argmax mismatches | identical |
| Kernel / DMA / HBM clocks | +0.041 / +0.003 / −0.018 ns (HBM at 446.4 MHz) | +0.027 / +0.003 / −0.021 ns (HBM at 445.8 MHz) |

Qualification on this image, all on 2026-10-11: 512-token free-running drift **BOUNDED**, fork at step 447, window-NRMSE ratio 1.063, slope 2.133e-06 per step (job 8238) — the same three numbers as every image since Iter66n; WikiText-2 teacher-forced word perplexity **16.774839771** against GPU **16.776123769** on the verified-identical 62-document / 183-window / 314,843-token workload, **−0.0077 %** (gate 5 %), kernel 8.371 ms/token over the run (Iter81 10.256 on the same card and workload: −18.4 %); paired power (jobs 8239 fp32 arm / 8240 bf16 arm, 4,096-token context, 60 s idle, 30 s warm-up, 3 × 60 s): **69.9 / 69.5 W active, 31.7 / 31.1 W idle, 0.593 / 0.589 J/token gross, 0.324 / 0.326 J idle-subtracted** at a sustained 8.48 ms TPOT, against the paired A100 eager runs' 30.99 ms / 96.0 W / 2.977 J (FP32) and 31.88 ms / 85.5 W / 2.728 J (BF16): **3.66x / 3.76x faster, 5.02x / 4.63x gross and 2.76x / 1.74x idle-subtracted energy efficiency**; against Iter81 (62.1 W, 0.646 J gross, 0.322 J idle-subtracted) +7.8 W active and −8.2 % gross energy per token. The card reads back DATA_CLK 200 MHz and hbm_aclk 446 MHz.

A first link of the same source (build 8230) stopped at the new convolution-tail
address gate because the launcher's frozen snapshot lacked the gate script; the
gate passed on that build's own XO, the snapshot list was fixed, and build 8236 is
the second link.

### Iter81 balanced-island image, 200 MHz (predecessor)

The image passed the production chain — the native 6/32-step trajectory gates
on the three-thread C model with the stored native logits as a bit-exact
reference, the BF16 layout check, the XO architecture gate, the synthesized
state-address gates on both the writers and (new) the readers of `gdn_k_slr2`,
the island-name and HLS 200-656 gate, the in-link exact-clock (kernel and DMA
exact, HBM within the 440 MHz floor), route, DRC and bus-skew gates, the
clock-metadata check, and the 8- and 64-token card gates (build 7370, job
7371, 2026-10-07):

| Metric | build 7370, job 7371 (production) | Iter79 image, build 6548 / job 6549 |
|---|---:|---:|
| Production TPOT, median of 63 | **10.470 ms/token** | 12.350 ms/token |
| Kernel execution, median of 63 | **10.347 ms/token** (mean 10.348, 10.303–10.453) | 12.227 ms/token |
| Host/XRT/PCIe overhead | 0.123 ms/token | 0.123 ms/token |
| Effective kernel cycles at 200 MHz | **2,069,400** | 2,445,400 |
| 8- and 64-token trajectories | exact | exact |
| CUDA vector gate, 2,016,000 logits | NRMSE 0.004663, worst step 0.011908, cosine 0.999989, min step cosine 0.999946, top-5 5/5, 0 argmax mismatches | identical |
| Kernel / DMA / HBM clocks | +0.027 / +0.003 / −0.021 ns (HBM at 445.8 MHz) | +0.029 / +0.003 / +0.015 ns |

Qualification on this image, all on 2026-10-08: 512-token free-running drift **BOUNDED**, fork at step 447, window-NRMSE ratio 1.063, slope 2.133e-06 per step (job 7401) — the same three numbers as every image since Iter66n; WikiText-2 teacher-forced word perplexity **16.774839771** against GPU **16.776123769** on the verified-identical 62-document / 183-window / 314,843-token workload, **−0.0077 %** (gate 5 %), kernel 10.256 ms/token over the run (job 7401) — the Iter79 image's figures to the last digit, as identical arithmetic requires; paired power (jobs 7437 fp32 arm / 7438 bf16 arm, 4,096-token context, 60 s idle, 30 s warm-up, 3 × 60 s): **62.1 W active, 31.1 W idle, 0.646 J/token gross, 0.322 J idle-subtracted** at a sustained 10.40 / 10.34 ms TPOT, against the paired A100 eager runs' 32.06 ms / 94.9 W / 3.045 J (FP32) and 32.44 ms / 85.1 W / 2.761 J (BF16): **3.08x / 3.14x faster, 4.71x / 4.29x gross and 2.76x / 1.80x idle-subtracted energy efficiency**; against the Iter79 image (57.8 W, 0.717 J gross, 0.325 J idle-subtracted) +4.3 W active and −9.9 % energy per token. The card reads back DATA_CLK 200 MHz and hbm_aclk 445 MHz.

Two earlier links of the same design are part of the record: the first
(build 7261) closed timing and ran without a hang but computed wrong tokens
from step 2 — its state reader, in the per-head address form, read every state
beat 128 MiB low after Vivado synthesis folded a 27-bit signed offset into the
pointer add (HLS RTL correct, post-synthesis wrong, exactly the Iter75 writer
defect on the read side), found by on-card state dumps and confirmed by a
reader address gate that is now part of the build; the second (build 7323) was
refused by the exact-clock gate on `hbm_aclk` (−0.021 ns, seven shell-pipe
endpoints) and reproduced to the picosecond by build 7370 with the 440 MHz
floor.

### Iter79 three-kernel image, 200 MHz (predecessor)

The promoted image passed the production chain — native 6/32-step trajectory
gates on the three-thread C model and the BF16 layout check, the XO
architecture gate over the three kernels, the synthesized state-address gate on
`gdn_k_slr2`, the in-link exact-clock, route, DRC and bus-skew gates, the
clock-metadata check, and the 8- and 64-token card gates — and the full
qualification set that promoted the 150 MHz image:

| Metric | build 6548, job 6549 (production) | build 6568, job 6569 (reproduction) |
|---|---:|---:|
| Production TPOT, median of 63 | **12.350 ms/token** | 12.389 ms/token |
| Kernel execution, median of 63 | **12.227 ms/token** | 12.256 ms/token |
| Host/XRT/PCIe overhead | 0.123 ms/token | 0.133 ms/token |
| Effective kernel cycles at 200 MHz | **2,445,400** | 2,451,200 |
| 8- and 64-token trajectories | exact | exact |
| CUDA vector gate, 2,016,000 logits | NRMSE 0.00466269633, worst step 0.0119, cosine 0.999989, 0 tolerance failures | identical |

A second card run of the production image (job 6567) measured 12.341 / 12.229
ms. The vector-gate figures equal the 150 MHz image's and Iter67c's, as they
must for identical arithmetic. Qualification on the production image:
512-token free-running drift **BOUNDED**, fork at step 447, window-NRMSE ratio
1.063, slope 2.133e-06 per step — the same three numbers as every image since
Iter66n (job 6570); WikiText-2 teacher-forced word perplexity **16.774839771**
against GPU **16.776123769** on a verified-identical workload of 62 documents /
183 windows / 314,843 tokens, −0.0077 %, with the kernel at 12.267 ms/token over
the whole run; paired power (jobs 6571/6572, 4,096-token context, 3 × 60 s)
**57.8 W** active and 31.5 W idle, **0.717 J/token** gross and 0.325 J
idle-subtracted, against the paired A100 eager runs' 31.74 ms / 94.9 W /
3.012 J (FP32) and 32.55 ms / 84.7 W / 2.756 J (BF16): 2.56x faster, 4.20x /
3.87x gross and 2.60x / 1.72x idle-subtracted energy efficiency. Against the
150 MHz image the kernel is 1.319x faster on 1.06 % more cycles and uses 8 %
less energy per token: the clock gain outweighs the +9.8 W of active power.

The hang this image shipped without is worth recording here because the
simulations could not see it. The first 200 MHz link (P5) closed timing but its
image never finished a token: all three kernels stayed in START, layer 0's
state was written back, layer 1's never was. HLS had warned (`200-656`) that
the recurrent islands' region ran two auto-rewound single-loop processes
without start propagation — the duplicator, as in production, and the new
stream-writing merge. The merge's loop rewinds because it writes a stream, and
it is part of the region's output synchronization because that stream leaves
the region, so the region never reported done and the SLR2 kernel never left
the QKVG call. One line (`#pragma HLS dataflow` with start propagation on in
that region) fixed it; the controlled rebuild is the proof.

### Iter76 flow, 150 MHz (predecessor)

The production image passed, in order: the native 6- and 32-step trajectory
gates and BF16 layout check; the XO architecture gate; the synthesized
state-address gate (the failure class that lost Iter68G and Iter73a on card is
now caught before the link); the in-link exact-clock, route, DRC and bus-skew
gates; the post-link clock-metadata check; and the 8- and 64-token card gates
through the production on-card half. Three images of the checksum-identical
closed design have run on card:

| Metric | build 4022, job 4023 (production) | build 3987, job 4021 | image 3956, job 3957 |
|---|---:|---:|---:|
| Production TPOT | **16.255 ms/token** | 16.238 ms/token | 16.256 ms/token |
| Kernel execution | **16.131 ms/token** | 16.129 ms/token | 16.131 ms/token |
| Production host/XRT/PCIe overhead | 0.124 ms/token | 0.109 ms/token | 0.125 ms/token |
| Effective kernel cycles at 150 MHz | **2.4197M** | 2.4194M | 2.4197M |
| 64-token trajectory | exact | exact | exact |
| CUDA vector gate, 2,016,000 logits | NRMSE 0.00466269633, cosine 0.999989, top-5 5/5, argmax mismatches 0 | identical | identical |

The vector-gate figures are bit-identical across the three images and equal to
Iter67c's, as expected for the same arithmetic. Qualification ran on the 3956
image (jobs 3958–3960): 512-token drift **BOUNDED** with the trajectory fork at
step 447, the same step as every prior image; WikiText-2 teacher-forced word
perplexity **16.774839771** against GPU **16.776123769** on a verified-identical
workload, −0.0077%; paired power at a matched 1 s sampling period **48.0 W**
active and 26.0 W idle, **0.7792 J/token** gross and 0.3566 J idle-subtracted,
against the paired A100 FP32 run's 31.519 ms / 95.8 W / 3.0243 J (1.94x faster,
3.88x gross and 2.30x idle-subtracted efficiency; the pooled A100 medians used
in the paper are 34.98 ms FP32 and 36.04 ms BF16). Relative to Iter67c the
kernel is 1.494x faster on 0.4% more cycles: the whole gain is clock.

### Iter67c (predecessor, 100 MHz)

Iter67c retains Iter66e's all-BF16 physical architecture and changes three
bounded datapaths: argmax is fused into the existing logit-emission traversal,
`recur_island_read` uses a five-phase schedule to reach II=1 without FP32
reassociation, and the convolution window has 16 banks. Build 2993 routed and
closed timing at 100 MHz; on-card jobs 2994 and 3101 passed the exact trajectory
and independent-GPU full-logit gates.

Job 3101 defines production TPOT at the token-ID-to-token-ID system boundary.
Its timer includes the host-resident embedding lookup and 8 KiB upload, XRT
launch/wait, the full kernel including fused argmax, and one 64-byte token-line
read-back. It stops before optional full-logit transfer and all reference
scans. Across 63 generated tokens:

| Metric | Median | Mean |
|---|---:|---:|
| Production TPOT | **24.208 ms/token** | **24.219 ms/token** |
| Kernel execution | **24.099 ms/token** | **24.092 ms/token** |
| Production host/XRT/PCIe overhead | **0.109 ms/token** | **0.126 ms/token** |
| Effective kernel cycles at 100 MHz | **2.4099M** | **2.4092M** |

The full-logit gate remains enabled after the production timer: global NRMSE
0.00466, worst-step NRMSE 0.0119, minimum cosine 0.99995, exact top-5, zero
argmax mismatches, exact 64-token trajectory, and first divergence -1.

### Iter66e (predecessor)

The Iter66e image passed, in order: native C++ build; fast exact decode 6/6;
full native exact decode 32/32 with all 992,000 logits bit-identical to the
captured native reference; a production-faithful one-layer/all-eight-head RTL
cosim (job 2500, no deadlock — `style=frp` changes pipeline control semantics
and Iter56b's depth-1024 deadlock reached the card without such a gate); legal
route and timing closure; and the on-card gates.

Excluding the seed entry, 63 measured kernel calls (job 2507) produced:

| Metric | Iter66e at 100 MHz |
|---|---:|
| Median wall | **26.654 ms/token** |
| Median kernel | **25.625 ms/token** |
| Effective cycles | **2.5625M** |
| Host loop overhead | 1.029 ms/token (**3.9%**) |
| Speedup over Iter61, 42.170 ms | **1.582x** |
| Speedup over eight-port baseline, 121.4 ms | **4.55x** |
| Margin vs stock-GPU 35 ms reference | **24% faster** |

Host loop overhead was measured directly here for the first time as the
wall-minus-kernel difference. **That 1.029 ms figure is an overestimate and
the wall column is not a production number**: the timed window contained the
two `compare_logits_step` calls, each sweeping all 32,000 logits against a
reference, so a gated run charged itself for work a deployment never does.
`host.cpp` now stops the clock after the host argmax and runs the comparisons
outside it. The kernel column is unaffected -- it only ever spanned
`run.wait()` -- so 25.625 ms and the 2.5625M-cycle figure stand. The next
on-card run will report a lower wall time that must not be compared against
26.654.

**Quality gates.** The trajectory is exact over 64 tokens. The independent CUDA
vector gate over all 2,016,000 logits reports global NRMSE **0.00466**,
worst-step **0.0119**, minimum cosine **0.99995**, top-5 exact, and **zero
argmax mismatches**. Hardware-versus-native bit-exactness does *not* hold and
is not expected to — see § *Arithmetic contract and what "correct" means*.

**Long-run drift is bounded, measured at 512 tokens** (jobs 2523/2529/2672).
Pre-fork windowed NRMSE is flat at 0.0048 +- 0.001 with slope ~1.5e-7, zero
argmax mismatches and no non-finite values, consistent with the gated delta
rule being contractive. The trajectory comparison is more interesting than it
looks:

| pair | pre-fork NRMSE | first argmax divergence |
|---|---:|---:|
| hardware vs native | 0.0046 +- 0.001 | 83 |
| native vs GPU (control) | 0.0045 +- 0.0003 | 83 |
| **hardware vs GPU** | 0.0048 +- 0.001 | **447** |

Native and the GPU fork at exactly the same step 83 and their post-fork windows
agree with the hardware comparison to four digits, so hardware and the GPU took
the *same* alternative token there and native is the outlier. Step 83 is a
near-tie argmax that any implementation difference flips: "zero trajectory
divergence over 512 tokens" is unachievable by any independent implementation
pair, including the CUDA reference this project already accepts. Against the
reference that matters, hardware tracks the GPU **5.4x longer** than native
does. Caveat: one prompt, one seed — this shows hardware is not systematically
worse, not that it is systematically better.

**Task-level quality on card (Iter66o): COMPLETE and PASSING.** Teacher-forced
WikiText-2 rolling perplexity feeds the known next token and reads
log-probabilities from the kernel's own exported logits, so it is immune to the
trajectory forking above. Card job **2822** ran 2 h 36 m on `acclnode01`; GPU
reference job **2821** ran the same fixture on an A100 in 5 m 26 s.

| | FPGA | GPU | delta |
|---|---:|---:|---:|
| Word perplexity | **16.774840** | 16.776124 | **-0.0077%** |
| Byte perplexity | 1.6944051 | 1.6944293 | -0.0014% |
| Bits per byte | 0.7607788 | 0.7607995 | |
| Total log-probability | -680,535.77 | -680,554.24 | |

The gate was 5% relative, so this passes by a factor of 650, and it is 7.6x
tighter than the 2-document smoke (-0.058%) — the expected direction for more
samples. Workload identity was verified rather than assumed: 62 documents, 183
windows, 314,843 scored tokens, 241,335 words and 1,290,527 bytes all match on
both sides. `kernel_ms_per_token` is 25.6078, matching the free-running decode
study's 25.625 to 0.07%, so the scoring path costs the same per token.

The FPGA is nominally the *lower* perplexity of the two. At 1.3e-3 absolute on
a 16.78 baseline that is indistinguishable, not better; the GPU reference also
carries a known contract difference (it scores each window in a single batched
forward and so does not reproduce the accelerator's per-token BF16 state and
conv rounding).

**This is the result that closes the BF16 question on hardware.** The step-2
logit divergence and the step-83 trajectory fork are real and documented above,
but they do not move task quality: the kernel reproduces GPU-measured
perplexity over 314,843 tokens to within 0.008%.

### Iter57 (historical)

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

1. Keep 32 independent weight masters and 16 two-port clusters, split 4 / 8 /
   4 across the control, SLR0 and SLR2 kernels with every master on the die of
   its pseudo-channel.
2. Keep MM2S readers separate from compute and keep the wide FIFOs in BRAM.
3. Keep the activation ripple, the per-kernel collectors (`collect3`,
   `collect8`, `collect4`) and the final collector's 8 | 64 | 24 | 32 word order:
   it is what makes the three-kernel result stream bit-identical to the
   monolith's.
4. Keep only one shared physical GEMV region instance per kernel
   (`gdn_gemv_part_ctrl`, `gdn_gemv_part_slr0`, `gdn_gemv_part_slr2`).
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
11. Judge changes against both 200 MHz timing on all three clocks and
    zero-conflict routing; HLS resource savings alone do not predict
    routability.
12. Preserve the head-serial/all-port QKVG and pair-interleaved GU layouts,
    including the full-byte host validation, until a replacement proves
    exactness and a lower measured cycle count.
13. Keep one time-shared head-local convolution actor inside the QKVG result
    sink, plus fixed-bank packed context movers; do not restore three serial
    whole-hidden convolution passes.
14. Keep convolved Q/K/V head-local and streamed directly to recurrence; do not
    restore whole-hidden Q/K/V buffers.
15. Keep the state graph forward-only with a **full-window depth-4096 URAM
    queue** per state port. Depth 1024 is a proven hardware deadlock for this
    registered collector topology and depth 2048 was the Iter57 minimum; the
    full-window depth is a liveness bound, not a tuning choice. Do not
    reintroduce feedback credit/readiness handshakes without a bounded proof
    and a better routed result.
16. Keep the two concurrent 16-column recurrent islands and deterministic
    half-head merge until a replacement proves equal correctness, lower cycles,
    and timing closure at the current 200 MHz constraint, whose kernel-clock
    margin is +0.027 ns, so any regression is a miss until it re-closes.
17. Keep native full-logits comparison enabled as a pre-hardware correctness
    gate; token parity alone cannot detect non-argmax numerical regressions.
18. Keep `style=frp` on the cluster weight stream. It is what made this design
    routable, and reverting it restores the 5,100--9,300-load clock-enable
    cones that four routing strategies and seven repair jobs could not close.
    Any change to that loop must be re-checked by RTL cosim: frp alters
    pipeline control semantics, and a deadlock of this class reached the card
    once before a cosim gate existed.
19. Keep the native `ap_float<16,8>` product and its AMD-FPO DAZ/FTZ/RNE
    reference in lockstep. The native model is the only thing that makes the
    csim gate meaningful; a divergence between them silently invalidates every
    pre-hardware result.
20. Keep the BF16 state and conv-tail packing (32 values per Beat512) and the
    FP32-container `.gdnstate` ABI. The external format is unchanged and the
    host validates BF16-exactness at load; do not couple the two.
21. Do not re-run a listed routing experiment without stating why the outcome
    would differ. Checkpoint route-repair on a dense placement is closed:
    seven jobs across both failure classes made every case worse.
22. Do not reinstate the hardware/native bit-exact gate. It gates on something
    no conforming implementation pair can satisfy while the recurrence uses
    library transcendentals. The gates that must stay are the native csim
    bit-exact comparison, the independent-GPU scale-aware vector gate, and the
    on-card WikiText perplexity comparison. If an arithmetic change needs the
    old comparison for localization, set `LOGITS_REFERENCE` for that run only.
23. Keep the three-kernel partition and its seven data-only links. The
    kernels share one static schedule (`gdn_call_schedule`); a link must never
    carry control, a call's beat counts must match on both ends by construction,
    and a kernel must be able to finish every call from its own parameters.
    New cross-die traffic goes through a registered AXI-Stream link, never
    through a shared memory region or a pblock that spans dies.
24. Never put `disable_start_propagation` on a dataflow region whose
    single-loop processes write a stream that leaves the region. HLS
    auto-rewinds such a loop and counts the process in the region's output
    synchronization; without start tokens the region never reports done. The
    recurrent islands' region (`gdn_recurrent_attention_islands_dataflow_p`)
    runs with start propagation on for this reason. Treat `WARNING: [HLS
    200-656]` as a build blocker, not a note: the C model, the native
    three-thread model and RTL cosim of a single kernel all pass with the
    deadlock in place; only the card (or a closed-loop emulation) shows it.
25. Keep the hardware-emulation and card probes that found that hang
    reproducible: `GDN_KERNEL_WAIT_TIMEOUT_MS` in `host.cpp` (bounded kernel
    wait plus a recurrent-state read-back that names the last layer written)
    is diagnostic-only and must stay off by default.
26. Every `m_axi` address that combines a runtime layer offset with a port
    base uses the state writer's form — the original base pointer plus a
    complete `ap_uint<64>` index, the offset add registered
    (`bind_op op=add impl=fabric latency=1`) — and passes both synthesized
    address gates (`check_state_writer_addresses.py`,
    `check_state_reader_addresses.py`: HLS RTL, post-synthesis, post-opt). A
    narrower signed offset is folded into the pointer add 128 MiB low by Vivado
    synthesis on the write side (Iter68G/73a/75) and on the read side (Iter81
    link 1); the HLS RTL is right in every case, so no simulation sees it.
27. Keep the islands' three-stage head pipeline (prep → read → update) with
    its 8,192-word URAM replay FIFO, the read stage's non-blocking consumption
    of the full-window state queues, the prefetch-first state reader, the
    five-stage bound update multipliers and the free-running delta loop. They
    are what hide the islands under the weight stream at the original island's
    out-of-context timing (+0.429 against +0.434 ns); binding the read
    multipliers the same way made timing worse and is recorded as rejected.

Use [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md) for the
historical GEMV scaling progression and
[optimization_log.md](optimization_log.md) for the exhaustive build,
congestion, timing, and performance record.
