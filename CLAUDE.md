# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hardware accelerator (Vitis HLS) for GatedDeltaNet inference**, based on the paper *Gated Delta Networks: Improving Mamba2 with Delta Rule* (ICLR '25, NVIDIA). The primary development target is `c_impl/`, which contains the HLS-synthesizable **C++** implementation of the GatedDeltaNet forward pass and the on-card hardware flow for the **Xilinx Alveo U55C**. The Python code (`lit_gpt/`, `pretrain.py`, `scripts/`) and the original Triton/FLA kernels serve as **golden references** for correctness verification.

Paper: https://arxiv.org/abs/2412.06464
Checkpoint: `m-a-p/1.3B-100B-GatedDeltaNet-pure` (HuggingFace) — consumed by `scripts/export_gdn_c.py` to produce the flat `.gdnw` weight blob.

`c_impl/README.md` and `c_impl/doc/` are the authoritative, up-to-date references for the accelerator — consult them for any detail beyond the big picture below.

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

All accelerator work happens in `c_impl/`. The Makefile (`make -C c_impl help`) drives three artifact classes: host testbenches, the v++ kernel (`.xo`/`.xclbin`), and the XRT host program.

### Native C++ testbenches (csim entry points — fast iteration & parity)
```bash
make -C c_impl                              # builds gdn_eval, gdn_attn_test, gdn_matmul_test (c++ -O3 -std=c++14)
./c_impl/gdn_eval      <weights.gdnw> <fixture.gdnreq> <output.json>   # full 24-layer forward
./c_impl/gdn_attn_test <weights.gdnw> <block.gdnblk>                   # single attention layer in isolation
```
The native build `#include`s `hls_stream.h`, so it needs the Vitis HLS include dir (`XILINX_HLS_INC`, default `/tools/Xilinx/Vitis_HLS/2022.1/include`).

### Parity testing (end-to-end vs Python golden)
```bash
cd c_impl && bash test_parity.sh            # rebuild, run every fixtures_smoke/*.gdnreq, diff vs results_smoke_python/
```
Tolerance 1e-3 (observed diffs ~1e-5). Diff logic is `scripts/check_gdn_c_parity.py`.

### Decode correctness + TPOT (vs cached GPU golden — no GPU needed)
```bash
bash scripts/decode_correctness_check.sh            # full: 4 examples × 32 steps, gates on exact match
bash scripts/decode_correctness_check.sh --fast     # 1×6 smoke (~1–2 min) — used by the inference-edit hook
./c_impl/gdn_eval <w.gdnw> fixtures_decode/decode.gdnreq <out.json> --decode [--decode-len N] [--limit E]
./c_impl/host.exe <xclbin> <w.gdnw> fixtures_decode/decode.gdnreq <out.json> 0 --decode ...   # on-card
```
Decode fixture + fp32 GPU golden are committed (`c_impl/fixtures_decode/`, `c_impl/results_decode_golden/`); regenerate via `scripts/export_gdn_c.py decode` + `scripts/compare_gdn_c.py --decode-golden`. On-card decode is bit-exact to the golden over 64 tokens; TPOT today is the O(n²) re-prefill baseline (state clears every call). A PostToolUse hook in `.claude/settings.json` auto-runs the fast check on every edit to `c_impl/gdn_model.{cpp,h}`, `host.cpp`, `gdn_eval.cpp`, `lit_gpt/gated_delta_net.py` — keep it passing. Details: `c_impl/doc/decode_correctness.md`.

### Vitis HLS synthesis (csim/csynth/cosim — no board needed)
```bash
cd c_impl
vitis_hls -f test.tcl                        # full 24-layer gdn_forward
vitis_hls -f test_single_GDN_attn_synth.tcl  # single attention layer (fast csynth iteration)
vitis_hls -f test_matmul.tcl                 # matmul-only harness (gdn_matmul_top)
```
All target Alveo U55C (`xcu55c-fsvh2892-2L-e`) at 100 MHz (10 ns clock). The exact set of TCL variants is mapped in `c_impl/doc/architecture.md` §2.

### On-card hardware flow (v++ → bitstream → run on U55C)
```bash
cd c_impl
make run_hw                                  # xo → xclbin → host → ./host.exe on the board (dependency-driven)
make run_hw FIXTURE=fixtures_smoke/piqa.gdnreq HW_MAX_EX=4   # override input / cap examples
```
Phase times (cold): `xo` ~30–60 min, `xclbin` ~4–6 hr, `host` seconds. Individual phases: `make xo|xclbin|host`. Clean: `make clean|clean_hw|distclean`.

**Critical knob — `FREQ=100`** (default): the kernel was synthesized for a 10 ns target. The U55C gen3x16 platform defaults to 300 MHz; without `--kernel_frequency 100` the route fails with WNS ≈ −8.7 ns. The link is shaped by two files in `c_impl/`: `hw.cfg` (kernel instance, HBM channel mapping, congestion-oriented Vivado strategies) and `pblock_pe_split.tcl` (pre-place floorplan splitting the 16 systolic PEs across SLR0/SLR1 so the design closes timing — without it the router quit at congestion level 7).

### Exporting weights & fixtures (Python golden reference)
```bash
python scripts/export_gdn_c.py weights  --output c_impl/artifacts/gdn-1.3b-f32.gdnw
python scripts/export_gdn_c.py fixtures --tasks piqa hellaswag --output-dir c_impl/fixtures_smoke
python scripts/export_block_fixture.py  --layer 0 --output c_impl/fixtures_block/block0_attn.gdnblk
```

### Running Python golden reference
```bash
python scripts/compare_gdn_c.py --fixture c_impl/fixtures_smoke/piqa.gdnreq --output results/piqa_python.json --device cuda --dtype float32
python scripts/check_gdn_c_parity.py --python-dir c_impl/results_smoke_python --c-dir c_impl/results_smoke_c --output c_impl/results_smoke_parity.json
python scripts/fla_lm_eval.py               # lm-eval-harness evaluation
```

## Architecture

### HLS Accelerator (`c_impl/`) — primary target
- **`gdn_model.cpp` / `gdn_model.h`** — all HLS-synthesizable compute and three top functions sharing the same in-file systolic-matmul kernel:
  - `gdn_forward` — full 24-layer inference (takes a second `weight_data_mm` pointer aliasing the weights onto a dedicated 512-bit AXI bundle).
  - `gdn_attn_forward` — a single attention layer in isolation (fast synthesis target).
  - `gdn_matmul_top` / `gdn_matmul2d_top` — matmul-only harnesses for head-to-head matmul optimization.
  - Submodules: `gdn_embed_tokens`, `gdn_rmsnorm_rows`, `gdn_matmul_tiled` / `gdn_matmul_2d` (weight-stationary), `gdn_depthwise_conv_silu`, `gdn_recurrent_attention` (gated delta rule with persistent BRAM state), `gdn_output_norm_and_gate`, SwiGLU.
  - Tiled/systolic matmul with `#pragma HLS array_partition`, `pipeline II=1`, `unroll`; every loop carries a `loop_tripcount` pragma. `memcpy`/`memset` in synthesized paths are replaced with explicit labeled loops for accurate latency estimation.
- **`gdn_eval.cpp` / `gdn_attn_test.cpp` / `gdn_matmul_test.cpp`** — host testbenches (load weights/fixtures, run a top, write/check JSON).
- **`host.cpp`** — XRT host program for on-card execution: allocates one `xrt::bo` per kernel arg, uploads the 5.6 GB weight blob in 16 MiB chunks (`sync_bo_chunked` works around the XRT 2022.1 ~16 MiB sync cap), streams contexts window-by-window, emits the same JSON schema as `gdn_eval`.
- **`hw.cfg`, `pblock_pe_split.tcl`** — v++ link config and SLR floorplan for the bitstream (see the `FREQ=100` note above).
- **Formats**: `.gdnw` (flat F32 weights), `.gdnreq` (pretokenized eval fixtures), `.gdnblk` (single-layer attention fixture: input + golden output for one block).
- **`doc/`** — design docs per submodule + `architecture.md`, `optimization_log.md` (v0→v7), `weight_stationary_matmul.md`, and `decode_roadmap.md` (current direction).

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
- **Current target is decode (TPOT), not prefill.** Decode is a GEMV (1 token/step) and is **weight-bandwidth bound**, not compute bound — the 16×16 PE grid is already ~19× over-provisioned for the single weight port. The roadmap (`doc/decode_roadmap.md`): a GEMV decode datapath, multi-channel parallel weight readers (the dominant lever, toward HBM's ~460 GB/s), INT8 weights (5.6→1.4 GB), and keeping the kernel resident across tokens. The 16×16 grid stays.
- **Decode Phase 0 is done:** the premise is GPU-validated (`doc/decode_premise.md` — GDN flat ~35 ms/token vs transformer O(n)) and on-card decode is bit-exact to the GPU golden over 64 tokens (`doc/decode_correctness.md`); the fast correctness check runs as a standing hook on inference-code edits. Next is the Phase-1 single-token datapath: persist recurrent state (`gdn_model.cpp` clears it every call) + conv window, GEMV path, resident kernel.
- When changing matmul or memory layout, the relevant deep references are `doc/weight_stationary_matmul.md`, `doc/systolic_matmul.md`, and `doc/optimization_log.md`.

## Dependencies

**HLS / hardware**: Vitis HLS + Vitis 2022.1, targeting Alveo **U55C** (`xcu55c-fsvh2892-2L-e`, platform `xilinx_u55c_gen3x16_xdma_3_202210_1`). On-card runs use XRT (default `/opt/xilinx/xrt`).

**Native C++ build**: any C++14 compiler + `libm`, plus the Vitis HLS include dir for `hls_stream.h`. No BLAS.

**Python (golden reference)**: container-based (see `Dockerfile`). Key deps: PyTorch 2.3.1+CUDA 12.1, Lightning 2.1.2, Triton 2.3.0, flash-attn, mamba-ssm, causal-conv1d, flash-linear-attention (FLA), lm-eval 0.4.1.
