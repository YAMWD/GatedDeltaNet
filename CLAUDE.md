# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hardware accelerator (Vitis HLS) for GatedDeltaNet inference**, based on the paper *Gated Delta Networks: Improving Mamba2 with Delta Rule* (ICLR '25, NVIDIA). The primary development target is `c_impl/`, which contains the HLS-synthesizable **C++** implementation of the GatedDeltaNet forward pass and the on-card hardware flow for the **Xilinx Alveo U55C**. The Python code (`lit_gpt/`, `pretrain.py`, `scripts/`) and the original Triton/FLA kernels serve as **golden references** for correctness verification.

Paper: https://arxiv.org/abs/2412.06464
Checkpoint: `m-a-p/1.3B-100B-GatedDeltaNet-pure` (HuggingFace) — consumed by `scripts/export_gdn_c.py` to produce the flat `.gdnw` weight blob.

For the **current decode-only accelerator**, the authoritative deep reference is `c_impl/doc/decode_disaggregated_gemv.md` (plus the other `decode_*.md` docs and `decode_roadmap.md`). **Caveat:** `c_impl/README.md` and `c_impl/doc/architecture.md` predate the decode-only pivot — they describe the now-retired prefill design (single-layer `gdn_attn_forward`, a `gdn_model.c` filename, the systolic matmul, `gcc -std=c11`) and are stale on the current kernel; trust the `decode_*.md` docs and this file over them.

## Model Configuration (GatedDeltaNet-1.3B)

| Parameter       | Value |
|-----------------|-------|
| Hidden dim      | 2048  |
| Heads           | 8     |
| Head dim (Q/K)  | 256   |
| Value dim       | 256   |
| Intermediate    | 5632  |
| Layers          | 24    |
| Conv size       | 4     |
| Max seq len     | 2048  |
| Vocab size      | 32000 |

Recurrent state per layer: 8 heads × 256 × 256 FP32 = 2 MB. Full weight blob ≈ 5.6 GB.

## Build & Run Commands

All accelerator work happens in `c_impl/`. The kernel is now **decode-only** — the GPU prefills the prompt and exports a fixed-size recurrent+conv state to disk, and the FPGA decodes token-by-token from it (see *Decode* below). The prefill GEMM, the prefill tops, and their testbenches/TCLs were removed. The Makefile (`make -C c_impl help`) drives three artifact classes: the host testbench (`gdn_eval`), the v++ kernel (`.xo`/`.xclbin`), and the XRT host program.

### Native C++ testbench (csim entry point — fast iteration & parity)
```bash
make -C c_impl                              # builds gdn_eval only (decode-only csim; c++ -O3 -std=c++14)
./c_impl/gdn_eval <weights.gdnw> <fixture.gdnreq> <out.json> --decode --decode-from-state <state.gdnstate> [--decode-len N]
```
`gdn_eval` is now a **decode-only** csim and hard-requires `--decode --decode-from-state`. The native build `#include`s `hls_stream.h`, so it needs the Vitis HLS include dir (`XILINX_HLS_INC`, default `/tools/Xilinx/Vitis_HLS/2022.1/include`). The old `gdn_attn_test` / `gdn_matmul_test` harnesses are retired — the decode-only pivot removed the prefill tops (`gdn_attn_forward`, `gdn_matmul_top`/`gdn_matmul2d_top`) they drove, so the default `host_tb` builds only `gdn_eval`.

### Parity testing (prefill era — vs Python golden)
```bash
cd c_impl && bash test_parity.sh            # rebuild, run every fixtures_smoke/*.gdnreq, diff vs results_smoke_python/
```
Tolerance 1e-3 (observed diffs ~1e-5). Diff logic is `scripts/check_gdn_c_parity.py`. **Note:** this is the prefill-parity flow; it no longer runs as-is on the decode-only branch (`gdn_eval` rejects non-decode invocations). The committed `results_smoke_python/` and the diff script remain; decode correctness is now gated by the check below.

### Decode correctness + TPOT (vs cached GPU golden — no GPU needed)
```bash
bash scripts/decode_correctness_check.sh            # full: 1 example × 32 steps, gates on exact match
bash scripts/decode_correctness_check.sh --fast     # 1×6 smoke (~1–2 min) — used by the inference-edit hook
./c_impl/gdn_eval <w.gdnw> fixtures_decode/decode.gdnreq <out.json> --decode --decode-from-state <state.gdnstate> [--decode-len N]
./c_impl/host.exe <xclbin> <w.gdnw> fixtures_decode/decode.gdnreq <out.json> 0 --decode --decode-from-state <state.gdnstate> [--decode-len N]   # on-card
```
The GPU prefills a prompt and exports the fixed-size recurrent+conv state with `scripts/export_gdn_state.py` → `.gdnstate` (~50 MB); the FPGA decodes from it through the `gdn_gemv` engine. The decode fixture (`fixtures_decode/decode.gdnreq`) and fp32 GPU golden (`results_decode_golden/`) are committed, but `decode_ex0.gdnstate` is **gitignored/regenerable** (like the `.gdnw` weight blob) — so the gate and the standing hook require both to be generated locally first (`export_gdn_state.py` + `export_gdn_c.py weights`). On-card decode is **bit-exact** to the golden over 64 tokens and **flat at ~470 ms/token** for a complete decode step (forward + lm_head + argmax all on-chip) — see `doc/decode_disaggregated_gemv.md`. A PostToolUse hook in `.claude/settings.json` auto-runs the fast check on every edit to `c_impl/gdn_model.{cpp,h}`, `host.cpp`, `gdn_eval.cpp`, `lit_gpt/gated_delta_net.py` — keep it passing. Details: `c_impl/doc/decode_correctness.md`.

### Vitis HLS synthesis (csim/csynth/cosim — no board needed)
```bash
cd c_impl
vitis_hls -f test.tcl                        # decode-only gdn_forward (24 layers, 1 token/step)
```
Targets Alveo U55C (`xcu55c-fsvh2892-2L-e`) at 100 MHz (10 ns clock). `test_single_GDN_attn_synth.tcl` (top `gdn_attn_forward`) and `test_matmul.tcl` (top `gdn_matmul_top`) are **stale** — both set tops the decode-only pivot removed. The TCL-variant map in `c_impl/doc/architecture.md` §2 predates that pivot.

### On-card hardware flow (v++ → bitstream → run on U55C)
```bash
cd c_impl
make run_hw                                  # xo → xclbin → host → ./host.exe on the board (dependency-driven)
make run_hw FIXTURE=fixtures_smoke/piqa.gdnreq HW_MAX_EX=4   # override input / cap examples
```
Phase times (cold): `xo` ~30–60 min, `xclbin` ~4–6 hr, `host` seconds. Individual phases: `make xo|xclbin|host`. Clean: `make clean|clean_hw|distclean`.

**Critical knob — `FREQ=100`** (default): the kernel was synthesized for a 10 ns target. The U55C gen3x16 platform defaults to 300 MHz; without `--kernel_frequency 100` the route fails with WNS ≈ −8.7 ns. The link is shaped by `hw.cfg` in `c_impl/` — kernel instance, the **disjoint HBM bank map** for the 8 GEMV weight readers + activations (overlapping `sp=` ranges → `std::bad_alloc`), and congestion-oriented Vivado strategies (`SSI_SpreadLogic_high`). `pblock_pe_split.tcl` floorplanned the prefill 16-PE systolic grid across SLR0/SLR1; it is **disabled** in the decode-only build (no systolic grid; `SSI_SpreadLogic_high` handles SLR spreading) but kept for reference.

### Exporting weights, state & fixtures (Python golden reference)
```bash
python scripts/export_gdn_c.py weights    --output c_impl/artifacts/gdn-1.3b-f32.gdnw
python scripts/export_gdn_state.py        ...   # GPU prefill → c_impl/fixtures_decode/*.gdnstate (recurrent+conv state)
python scripts/export_gdn_c.py fixtures   --tasks piqa hellaswag --output-dir c_impl/fixtures_smoke
```
`export_gdn_state.py` is the decode handoff producer (it self-checks bit-exactness vs the cache-decode golden). `export_block_fixture.py` (→ `.gdnblk`) targeted the retired single-layer attention harness.

### Running Python golden reference
```bash
python scripts/compare_gdn_c.py --fixture c_impl/fixtures_smoke/piqa.gdnreq --output results/piqa_python.json --device cuda --dtype float32
python scripts/check_gdn_c_parity.py --python-dir c_impl/results_smoke_python --c-dir c_impl/results_smoke_c --output c_impl/results_smoke_parity.json
python scripts/fla_lm_eval.py               # lm-eval-harness evaluation
```

## Architecture

### HLS Accelerator (`c_impl/`) — primary target
- **`gdn_model.cpp` / `gdn_model.h`** — all HLS-synthesizable compute and one **decode-only** kernel top:
  - `gdn_forward` — 24-layer, **one token per call** (token id in → next token id out); restores the recurrent+conv state at entry, saves it at exit. Takes the `weight_data` scalar-weight master plus `GEMV_CHANNELS` (=8) `weight_data_mm*` masters that stream the sharded projection weights in parallel.
  - `gdn_gemv` — the decode compute engine: **activation-stationary** GEMV — the activation vector is resident on-chip and weights stream from HBM once and are never cached (an HLS `dataflow` of one reader→MAC PE per HBM channel, with the projection weights compact-sharded across the 8 channels). It replaces the prefill weight-stationary systolic GEMM (`gdn_matmul_2d`), which was removed. `gdn_argmax` does the on-chip greedy argmax over the `lm_head` logits.
  - Submodules: `gdn_embed_tokens`, `gdn_rmsnorm_rows`, `gdn_gemv` (all large projections incl. `lm_head`) / `gdn_matmul_tiled` (the two tiny a/b gate projections), `gdn_depthwise_conv_silu`, `gdn_recurrent_attention` (gated delta rule with persistent BRAM state), `gdn_output_norm_and_gate`, SwiGLU.
  - Pragmas: `#pragma HLS array_partition`, `pipeline II=1`, `unroll`, `dataflow` (the gemv reader/MAC split); every loop carries a `loop_tripcount` pragma. `memcpy`/`memset` in synthesized paths are replaced with explicit labeled loops for accurate latency estimation.
- **`gdn_eval.cpp`** — the decode-only host testbench (loads weights + `.gdnstate`, runs the decode loop, writes/checks JSON). `gdn_attn_test.cpp` / `gdn_matmul_test.cpp` remain on disk but are retired (their tops were removed).
- **`host.cpp`** — XRT host program for on-card execution: builds the `GEMV_CHANNELS` compact weight shards from the flat blob, allocates one `xrt::bo` per kernel arg (weight shards on disjoint HBM banks — overlapping ranges `std::bad_alloc`), uploads the 5.6 GB of weights in 16 MiB chunks (`sync_bo_chunked` works around the XRT 2022.1 ~16 MiB sync cap), loads the `.gdnstate` into the resident state BOs, then decodes token-by-token from the seed, emitting the same JSON schema as `gdn_eval`.
- **`hw.cfg`, `pblock_pe_split.tcl`** — v++ link config (HBM bank map for the 8 GEMV readers) and the now-disabled prefill SLR floorplan (see the `FREQ=100` note above).
- **Formats**: `.gdnw` (flat F32 weights), `.gdnstate` (GPU-exported recurrent+conv decode state, ~50 MB — the prefill→decode handoff), `.gdnreq` (pretokenized eval fixtures), `.gdnblk` (single-layer attention fixture — retired with the prefill harness).
- **`doc/`** — design docs per submodule + `architecture.md`, `optimization_log.md` (v0→v7, prefill), `weight_stationary_matmul.md`, `decode_roadmap.md`, and **`decode_disaggregated_gemv.md`** (authoritative for the current decode-only accelerator: GEMV engine, the multi-reader scaling ladder, on-card results).

### Golden Reference: Python Model (`lit_gpt/`)
- **`model.py`** — `GPT`, `Block`, `MBlock`, `CausalSelfAttention`, `LLaMAMLP`.
- **`gated_delta_net.py`** — GatedDeltaNet layer: chunk-based linear attention with gated delta rule.
- **`config.py`** — model configs (0.4B, 1.3B, H1 variants).
- **`gated_delta_rule_ops/`** — Triton and FLA kernels; the `fla_version/` kernels are preferred for golden runs. The HLS C++ mirrors these function-for-function (mapping table in `README.md` / `c_impl/README.md`).
- **`rmsnorm.py`, `fused_rotary_embedding.py`, `fused_cross_entropy.py`** — fused-op golden references.

### Training Infrastructure (produces golden weights)
- **`pretrain.py`** — PyTorch Lightning Fabric + FSDP, BF16 mixed precision. **`packed_dataset.py`** — tokenized data loading (SlimPajama). Not exercised by the synthesis flow.

## Key Design Patterns & Current Focus

- The HLS C++ mirrors the Python computation graph exactly to maintain numerical parity (validated end-to-end within 1e-3, observed ~1e-5; on-card Wikitext perplexity matches Python golden to ~1e-7).
- **Optimization arc (prefill, on U55C):** the runtime was weight-HBM-traffic bound. Sequenced levers took wikitext-2048 prefill from **25.9 min → 4.2 min (~6.2×)**: weight-stationary blocking (kills ~95× weight re-reads), 512-bit bursted weight reads (aligned base + dedicated AXI bundle), Pack16 activation widening, and splitting activations across 3 HBM channels. Prefill is now **compute-bound** (matmul ≈ 76% of the kernel). PE-grid widening (Phase C) was attempted and **reverted** — it's a prefill lever with high routing risk and no decode benefit.
- **The accelerator is now decode-only (TPOT-focused).** Decode is a GEMV (1 token/step), **weight-bandwidth bound** not compute bound, so the prefill GEMM (16×16 systolic grid) was *removed* and replaced by the activation-stationary `gdn_gemv`. The disaggregated split (`doc/decode_disaggregated_gemv.md`): the GPU prefills + exports a fixed-size recurrent+conv state (`.gdnstate`, ~50 MB — free for a linear-attention model, no growing KV cache), and the FPGA decodes from it.
- **The decode work is built and on-card bit-exact.** Status: a complete decode step (forward + lm_head + argmax all on-chip) is **flat at ~470 ms/token, bit-exact** to the GPU golden over 64 tokens. The multi-reader bandwidth lever is **exhausted** — the weight stream is sharded across **8 parallel HBM readers** (`GEMV_CHANNELS=8`), 3.34× over the single-port floor with diminishing returns per doubling. The dominant remaining cost is the **serial floor S ≈ 325 ms** (recurrent-attention state I/O on one HBM master + the gated-delta-rule update); the FPGA is ~14× the A100's launch-bound 33.5 ms fp32, so the defensible angle is perf/Watt. Open levers: raise the 100 MHz kernel clock, parallelize/overlap the state I/O (attack S), INT8 weights (4× less traffic, costs bit-exactness). The fast correctness check runs as a standing hook on inference-code edits — keep it green.
- For the decode GEMV the deep reference is `doc/decode_disaggregated_gemv.md`; for the (removed) prefill matmul / memory layout see `doc/weight_stationary_matmul.md`, `doc/systolic_matmul.md`, and `doc/optimization_log.md`.

## Dependencies

**HLS / hardware**: Vitis HLS + Vitis 2022.1, targeting Alveo **U55C** (`xcu55c-fsvh2892-2L-e`, platform `xilinx_u55c_gen3x16_xdma_3_202210_1`). On-card runs use XRT (default `/opt/xilinx/xrt`).

**Native C++ build**: any C++14 compiler + `libm`, plus the Vitis HLS include dir for `hls_stream.h`. No BLAS.

**Python (golden reference)**: container-based (see `Dockerfile`). Key deps: PyTorch 2.3.1+CUDA 12.1, Lightning 2.1.2, Triton 2.3.0, flash-attn, mamba-ssm, causal-conv1d, flash-linear-attention (FLA), lm-eval 0.4.1.
