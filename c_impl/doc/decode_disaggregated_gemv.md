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

**Why 1.95 s and not the ~1.0 s single-port floor.** `gdn_gemv` reads weights
and MACs in the *same* loop nest, one output row at a time. Each output restarts
the `gemv_k` pipeline (Depth=54) and — the dominant cost — **breaks the AXI
burst**: the contiguous 5.6 GB weight stream is chopped into `out_dim` separate
HBM reads, each paying read latency, so the port sustains ~2.87 GB/s instead of
6.4. The fix is the FlightLLM/Serpens streaming structure (§8 [1],[4]): a
**dedicated weight-reader dataflow process** that issues one long contiguous
burst per projection and feeds the MAC through an `hls::stream` FIFO, decoupling
read from compute so the burst never breaks. That recovers toward ~1.0 s, and it
is the same reader structure Step 2 replicates across HBM channels.

## 7. Next

On-card decode-only is **bit-exact and flat at 1.95 s/token** (1.31× over Step 1),
sustaining 45% of the single weight port. The two remaining levers:

1. **Decouple read from MAC (single port → ~1.0 s).** Split `gdn_gemv` into a
   weight-reader dataflow process that bursts each projection's weights
   contiguously into an `hls::stream`, and a MAC process that consumes it — so
   the 5.6 GB stream is one long burst, not `out_dim` latency-bound reads. Closes
   the 2.87 → 6.4 GB/s gap.
2. **Step 2 — N parallel HBM weight readers (~55 ms).** Replicate that reader
   across HBM pseudo-channels with on-chip partial accumulation (Serpens [4]
   pattern); the 16-lane MAC widens to keep pace.

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
