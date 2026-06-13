# Disaggregated Decode-Only Accelerator (GEMV datapath)

This documents the pivot to a **decode-only** GatedDeltaNet accelerator: the GPU
prefills the prompt and exports a constant-size recurrent + conv state to disk;
the FPGA loads that state and decodes token-by-token through a new **GEMV**
datapath. The prefill GEMM, the prefill/decode mode flags, and all prefill code
were removed — the FPGA's only job is decode.

Supersedes the dual-mode Step-1 design in
[decode_step1_single_token.md](decode_step1_single_token.md) (which kept the
prefill GEMM and a `decode_flags` mux). Follows the decode roadmap in
[decode_roadmap.md](decode_roadmap.md).

## 1. Why disaggregate

Decode and prefill want opposite hardware, and they never run at the same time:

| | Prefill (rows ≫ 1) | Decode (1 token) |
|---|---|---|
| Compute | GEMM, compute-bound, batchable | GEMV, **memory-bound**, single-stream |
| Best device | GPU (high FLOPs, big batches) | FPGA (flat low-latency HBM dataflow) |
| Weight reuse | each weight × many rows | each weight × **1 MAC** |

GatedDeltaNet makes the split clean because it is a **linear-attention
(recurrent)** model: the entire prompt compresses into a **fixed-size** state,
independent of prompt length. The GPU→FPGA handoff is therefore a constant
**~50 MB**, not a growing KV cache:

```
recurrent_state : num_layers × H × K × V  fp32 = 24×8×256×256×4 = 48 MB
conv tails      : num_layers × {q,k,v} × (W-1) × hidden fp32 ≈ 1.7 MB
```

Over PCIe gen3×16 (~16 GB/s) that handoff is **~3 ms, once per request** —
negligible against decode. (A transformer would ship a context-growing KV cache,
often making disaggregation a loss; for GDN it is free.)

## 2. The GPU→FPGA state contract (`.gdnstate`)

`scripts/export_gdn_state.py` runs the FLA GPU model over an (arbitrary) prompt
with `use_cache=True`, pulls each layer's `(conv_state_q/k/v, recurrent_state)`
from the cache, and writes a flat blob. Layout derivation (validated bit-exact):

- **recurrent_state** — FLA `final_state` is `[N,H,K,V]` row-major, V contiguous.
  The FPGA stores `state[h][k][v] = recurrent_state[(h·K+k)·V+v]` — **identical,
  no transpose.**
- **conv_state** — FLA `cache` is `[N,D,W]` (newest at index W−1). The FPGA conv
  tail is `[W-1][D]` with `tail[r][d] = cache[0][d][r+1]` (drop the oldest of the
  W cached inputs; transpose D↔W). Per layer the conv order is q(0), k(1), v(2),
  matching the `head_buffer` slot `(layer·3 + conv)`.

Blob format (little-endian):

```
"GDNSTAT1" (8B)
u32 ×9 : version, num_layers, H, K, V, hidden, W(conv_size), prompt_len, seed_token
i32 ×prompt_len : prompt ids (informational)
f32 : Section A — recurrent_state, all layers   → recurrent_state BO (48 MB)
f32 : Section B — conv tails, all layers         → head_buffer BO (~1.7 MB)
```

`seed_token` is the argmax of the prompt's last-position logits — the **first
generated token**. The FPGA decode produces tokens 1..N-1 from that seed; the
full trajectory `[seed, …]` is checked against the committed golden.

The blob is large + regenerable, so it is gitignored (like the weight blob).

## 3. The GEMV engine (`gdn_gemv`)

Decode is a GEMV: `out[o] = Σ_k in[k]·W[o][k]`, one token, each weight read once.
The design follows the SOTA bandwidth-bound FPGA-HBM decode engines (see
§8 References) — **FlightLLM** [1] (LLaMA2-7B on one U280, "always-on-chip
decode"), **DFX** [2] (single-token GPT core), and the embedded-FPGA decode work
[3] (KV260, multiplier + adder tree at 85% of the decode-bandwidth limit):

1. **Activation-stationary.** The single activation vector (≤ 5632 fp32 = 22 KB)
   is loaded once into on-chip `a_loc` and reused for every output; **weights
   stream from HBM once and are never cached** — the opposite of the prefill
   weight-stationary GEMM (which reused each weight across 256 rows).
2. **Contiguous weight stream.** Weights are row-major `[out_dim][in_dim]`, so one
   output's row `W[o][:]` is contiguous. Scanning **one output at a time** keeps
   the 512-bit weight port reading one Pack16 (16 fp32) beat/cycle at **II=1** — a
   clean burst. (Interleaving 16 output rows would scatter the port to II=16.)
3. **Adder-tree + partial banks.** Each beat does 16 multiplies reduced by a
   combinational tree; the running sum rotates across `GEMV_PARTIAL=8` banks to
   hide FP32 fadd latency so the k-loop holds II=1.
4. **Pack16 output.** 16 dot-products are buffered and written as one 512-bit
   beat (every projection's `out_dim` is a multiple of 16).

Per-token cost is then `weight_bytes / port_bandwidth`: one 512-bit master at
100 MHz = 6.4 GB/s → ~1 s for the 5.6 GB fp32 blob — the **single-port GEMV
floor**, vs the 2.56 s the prefill GEMM spent at num_rows=1 (255/256 of its 16×16
array idle, weights not streamed back-to-back). The next lever (roadmap Step 2)
replicates the reader across N HBM channels toward HBM's ~460 GB/s aggregate.

All eight large projections (q/k/v/gate/o, mlp gate/up/down) use `gdn_gemv`; the
two tiny gate projections (a/b, out_dim=8) keep `gdn_matmul_tiled`.

## 4. What was removed

- **`gdn_matmul_2d`** (the weight-stationary 16×16 GEMM) and `gdn_matmul2d_top`.
- **`decode_flags` / `GDN_DECODE_RESTORE` / `GDN_DECODE_SAVE`** — there is one
  mode. The conv and recurrent attention always restore state at the start and
  save at the end; `gdn_forward` always forwards exactly one token.
- **Prefill paths** — `gdn_attn_forward`, `gdn_attn_forward_layer`, the old
  systolic test top `gdn_matmul_top`, and the prefill host wrappers
  (`gdn_forward_host`, `gdn_prefill_save_host`, `gdn_forward_flags_host`).
- **`gdn_eval` prefill modes** — the MC/LL perplexity scoring and the re-prefill
  decode benchmark; `gdn_eval` is now a decode-only csim driven by
  `--decode-from-state <blob>`.

`gdn_forward` lost its `decode_flags` argument — it is now a 20-arg decode-only
kernel top. `hw.cfg` is unchanged (the flag was an s_axilite scalar, not a memory
port; all `sp=` port names are identical).

## 5. Files

| File | Change |
|---|---|
| `scripts/export_gdn_state.py` | **new** — GPU prefill → `.gdnstate` export + self-check |
| `gdn_model.cpp` | `gdn_gemv` engine; decode-only `gdn_forward` (no GEMM/flags); conv + attention always restore/save; single `gdn_decode_step_host` |
| `gdn_model.h` | decode-only `gdn_forward` (20 args); flag macros + prefill/matmul decls removed |
| `gdn_eval.cpp` | `--decode-from-state` loader + decode loop; MC/LL/re-prefill modes removed |
| `host.cpp` | `upload_decode_state` into the resident BOs; `run_decode_hw` decodes from `.gdnstate`; `--decode-from-state`; 20-arg kernel call |
| `scripts/decode_correctness_check.sh` | gates on the decode-only-from-state path |
| `Makefile` | `host_tb` = `gdn_eval` only; retired the prefill testbenches |

## 6. Validation

Bit-exact to the GPU golden at every stage (fp32 → exact-match gate):

| Stage | Result |
|---|---|
| GPU export self-check (cache-decode vs golden, 8 tok) | exact |
| Native csim, state contract through old kernel (8 tok) | exact |
| Native csim, **decode-only GEMV** (8 / 32 / 64 tok) | **exact, first_div −1, 100% top-1** |
| **On-card, decode-only GEMV (64 tok)** | **exact, first_div −1, 100% top-1; flat 1949.6–1949.8 ms** |

The standing PostToolUse hook (`decode_correctness_check.sh --fast`) now exercises
the decode-only path and stays green.

### On-card result (U55C, 1×64, device 0)

| metric | value |
|---|---|
| correctness | 64/64 bit-exact vs GPU golden |
| per-token shape | **flat 1949.6–1949.8 ms (0.2 ms spread)** — O(1) |
| kernel TPOT | **1.95 s/token** |
| vs Step-1 (GEMM at num_rows=1, 2.56 s) | **1.31× faster** |
| effective weight bandwidth | 2.87 GB/s = **45%** of the 6.4 GB/s single-port peak |
| build | 4 h 21 m; kernel timing +4.26 ns, `hbm_aclk` +0.065 ns (0 failing) |

Removing the GEMM (16×16 grid + 5.76 MB URAM) let the kernel close timing with
margin and **no `hbm_aclk` marginality** (Step-1 had −50 ps/21 paths).

In the coupled version, `gdn_gemv` read weights and MAC'd in the *same* loop
nest, one output row at a time — each output restarted the `gemv_k` pipeline
(Depth=54) and **broke the AXI burst**, chopping the 5.6 GB stream into `out_dim`
latency-bound reads (2.87 GB/s = 45% of one port).

## 6b. Stage 1 — decoupled reader → MAC (on-card, bit-exact)

`gdn_gemv` is now an HLS **dataflow** of two processes: `gemv_read` issues one
monotonic sweep over the whole projection's weights (a single long 512-bit
burst) into an `hls::stream`; `gemv_compute` drains the FIFO and MACs against the
resident activation. The burst no longer breaks between output rows.

| metric | coupled | **decoupled** |
|---|---:|---:|
| kernel TPOT (flat) | 1949 ms | **1543 ms** (1.26×) |
| weight-port rate (XRT) | 2.87 GB/s (45%) | **5.30 GB/s (83% of one 512-bit port)** |
| avg read burst | — | **4.1 KB (full 64-beat)**, 97 ns latency |
| vs Step-1 (2.56 s) | 1.31× | **1.66×** |
| build / timing | — | 3 h 41 m; +3.77 ns, `hbm_aclk` 0 failing |

**1.54 s is the single-port floor**, and the decouple reached it. XRT shows the
port reads 5.6 GB/token at 5.30 GB/s ⇒ **~1.06 s of weight streaming**; the
kernel is 1.54 s ⇒ **~0.48 s is non-gemv serial work** (recurrent-attention state
R/W, the a/b `gdn_matmul_tiled`, conv) during which the weight port idles. So
`1.06 (gemv @ 83% of one port) + 0.48 (non-gemv) = 1.54 s`. The ~1.0 s target had
assumed a full 6.4 GB/s port *and* zero non-gemv — neither reachable on one
master. The decouple's *purpose* (burst efficiency) was met and exceeded:
**port 2.87 → 5.30 GB/s, 45% → 83%.** Further single-port tuning
(`num_read_outstanding`) has <0.1 s of headroom (latency already hidden under
4 KB bursts) — the lever that breaks the floor is multi-port.

## 6c. Stage 2 — multi-channel weight readers, N=2 (on-card, bit-exact)

The weight stream is split across **two parallel 512-bit HBM readers**. The
projection weights are **compact-sharded**: each projection's output rows are cut
into N stripes, shard *c* holds stripe *c* of every projection (one copy split N
ways — no replication). The two shards sit on **disjoint HBM channels**, so the
two readers' bursts hit different channels concurrently. `gdn_gemv` is now a
dataflow of **one PE per channel**: `gemv_pe_bcast` reads the resident activation
once and fans it to the PEs; each `gemv_pe_mac` MACs its shard's stripe; a single
MAC-free `gemv_collect` writes the output. The bit-exactness is preserved (same
resident `a_loc`, same beat order, same partial-bank adder-tree reduction).

| metric | singleport (Stage 1) | **N=2 (Stage 2)** |
|---|---:|---:|
| kernel TPOT (flat) | 1543 ms | **875 ms** (1.76×) |
| TPOT incl. host | 1540 ms | **937 ms** (1.64×) |
| weight readers | 1 × 512-bit | **2 × 512-bit, disjoint HBM** |
| decode bit-exact (1×64) | ✓ | **✓ top1 100%, first_div −1** |
| kernel timing (WNS) | +3.77 ns | **+0.506 ns, 0 failing** |
| shell `hbm_aclk` | 0 failing | −0.044 ns, 23 ep (**benign** — bit-exact) |
| build | — | xo (PE) 3 h 50 m + relink 3 h 24 m |

**Two routing/allocation walls, both load-bearing for N≥2:**

1. **PE distribution (routing).** A *monolithic* 2-wide consumer (one module doing
   both channels' 16-wide fp32 reductions) failed `route_design` **twice** —
   partially-conflicted `full_dsp` fp-adder nets, congestion level 7, after ~7 h
   routes. Splitting into **one independent dataflow PE per channel** (each with
   the Stage-1 single-port density that routed) lets Vivado place them apart;
   `SSI_SpreadLogic_high` + a LUT-fabric `bind_op` on the lane adds did the rest.
   Closed at WNS **+0.506 ns**, 0 failing kernel endpoints.

2. **Disjoint HBM bank map (allocation).** The first disjoint-sharding map put the
   shards on **narrow HBM groups that overlap** weight_data's wide group
   (`mm:HBM[10:20]` ⊂ `weight_data:HBM[10:31]`). XRT 2022.1 allocates each `xrt::bo`
   bottom-up/contiguous, so weight_data (5.46 GiB → 11 banks) takes banks 10–20
   and the shard's HBM[10:20] group is left with **zero free banks → `std::bad_alloc`**.
   `probe_alloc.cpp` pinned it (weight_data OK, shard0 FAIL as the 2nd bo). Fix:
   **dedicated, non-overlapping bank ranges** — `weight_data:HBM[20:31]` (12 banks),
   `weight_data_mm:HBM[10:14]`, `weight_data_mm2:HBM[15:19]` (5 each). 12+5+5 = 22 =
   HBM[10:31] exactly; activations stay on HBM[0:9]. Link-only change (the PE `.xo`
   is unchanged), so the second build was a relink.

**Why 1.76× and not 2×, and the path past it.** Amdahl on the kernel:
`1543 = W + S`, `875 = W/2 + S` ⇒ **W ≈ 1336 ms** (parallel weight streaming) and
**S ≈ 207 ms** serial floor — recurrent-attention state R/W, the a/b
`gdn_matmul_tiled`, conv, and the scalar `weight_data` master (notably the 256 MB
`lm_head` read per token). The readers halved W exactly; S is untouched. Extending
to more readers (W/N + S): **N=4 → ~540 ms, N=8 → ~374 ms** kernel — after which
S dominates and becomes the target.

## 7. Next

Decode-only is bit-exact and flat at **875 ms/token kernel (937 ms incl. host)**,
1.76× over the single-port Stage 1. The remaining levers, in order:

- **N=4 then N=8 weight readers.** Same compact-shard + per-PE-channel structure,
  more stripes (`GEMV_CHANNELS` 4, 8). The shards stay one copy split N ways
  (total weight HBM constant ~10.4 GiB), so plan the **disjoint bank ranges** up
  front (reclaim the tiny activation channels HBM[1:9] for the extra shards). The
  16×16-derived MAC width already covers N=8. Expected ~540 / ~374 ms.
- **Attack the S ≈ 207 ms serial floor.** Once readers collapse W, the next
  targets are recurrent-attention state I/O (50 MB R/W per token), the a/b
  projection, and moving the per-token `lm_head` read off the shared scalar
  master (or computing logits host-side).

## 8. References

The `gdn_gemv` engine (§3) follows the established design principles for
bandwidth-bound LLM decode on HBM-equipped FPGAs — activations resident on-chip,
weights streamed once, MAC width matched to the memory port, and (for scaling)
the weight stream sharded across HBM channels.

1. **FlightLLM** — S. Zeng, J. Liu, G. Dai, X. Yang, T. Fu, H. Wang, W. Ma,
   H. Sun, S. Li, Z. Huang, Y. Dai, J. Li, Z. Wang, R. Zhang, K. Wen, X. Ning,
   and Y. Wang. "FlightLLM: Efficient Large Language Model Inference with a
   Complete Mapping Flow on FPGAs." *ACM/SIGDA Int'l Symp. on Field-Programmable
   Gate Arrays (FPGA)*, 2024, pp. 223–234. arXiv:2401.03868. doi:10.1145/3626202.3637562.
   — *Source of the "always-on-chip decode" scheme (activations resident across
   the whole decode pass); LLaMA2-7B on a single Alveo U280, 65.9% HBM bandwidth
   utilization.*

2. **DFX** — S. Hong, S. Moon, J. Kim, S. Lee, M. Kim, D. Lee, and J. Kim.
   "DFX: A Low-latency Multi-FPGA Appliance for Accelerating Transformer-based
   Text Generation." *55th IEEE/ACM Int'l Symp. on Microarchitecture (MICRO)*,
   2022. arXiv:2209.10797. doi:10.1109/MICRO56248.2022.00051.
   — *Single-token GEMV-centric decode core with an HBM-bandwidth-oriented tiling
   and dataflow; GPT-2 on 4× Alveo U280.*

3. **Embedded-FPGA decode (bandwidth/capacity limit)** — J. Li, T. Li, G. Shen,
   D. Zhao, Q. Zhang, and Y. Zeng. "Pushing up to the Limit of Memory Bandwidth
   and Capacity Utilization for Efficient LLM Decoding on Embedded FPGA." 2025.
   arXiv:2502.10659.
   — *The closest analogue to `gdn_gemv`'s datapath: hidden state kept on-chip,
   N multipliers + an adder tree sized to the 512-bit memory beat, and a weight
   data-arrangement that enables consecutive bursts — reaching 85% of the
   theoretical decode-bandwidth limit.*

4. **Serpens** — L. Song, Y. Chi, L. Guo, and J. Cong. "Serpens: A High Bandwidth
   Memory Based Accelerator for General-Purpose Sparse Matrix-Vector
   Multiplication." *59th ACM/IEEE Design Automation Conference (DAC)*, 2022,
   pp. 211–216. arXiv:2111.12555. doi:10.1145/3489517.3530420.
   — *Reference for the multi-channel HBM streaming pattern that the Step-2
   widening of `gdn_gemv` adopts (sharding the weight stream across HBM pseudo-
   channels with on-chip partial accumulation).*
