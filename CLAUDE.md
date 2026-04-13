# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hardware accelerator (Vitis HLS) for GatedDeltaNet inference**, based on the paper *Gated Delta Networks: Improving Mamba2 with Delta Rule* (ICLR '25, NVIDIA). The primary development target is `c_impl/`, which contains the HLS-synthesizable C implementation of the GatedDeltaNet forward pass. The Python code (`lit_gpt/`, `pretrain.py`, `scripts/`) and the original Triton/FLA kernels serve as **golden references** for correctness verification.

Paper: https://arxiv.org/abs/2412.06464

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

## Build & Run Commands

### HLS Accelerator (`c_impl/`)
```bash
make -C c_impl                              # build C testbench (C11, no BLAS)
./c_impl/gdn_eval <weights.gdnw> <fixture.gdnreq> <output.json>
```

**Vitis HLS flow** (via `c_impl/test.tcl`):
```bash
cd c_impl
vitis_hls -f test.tcl                       # runs csim, csynth, and cosim
```

The TCL script targets `xcvu11p-flga2577-1-e` at 100 MHz (10 ns clock) and runs:
1. `csim_design` — C simulation against a smoke fixture
2. `csynth_design` — RTL synthesis with latency/resource estimates
3. `cosim_design` — RTL co-simulation for functional verification

### Parity Testing
```bash
cd c_impl && bash test_parity.sh            # build, run all smoke fixtures, check against Python golden
```

This rebuilds `gdn_eval`, runs every `fixtures_smoke/*.gdnreq` through the C implementation, and diffs against `results_smoke_python/` using `scripts/check_gdn_c_parity.py`. Tolerance is 1e-3 (observed diffs ~1e-5).

### Exporting Weights & Fixtures (Python golden reference)
```bash
python scripts/export_gdn_c.py weights --output c_impl/artifacts/gdn-1.3b-f32.gdnw
python scripts/export_gdn_c.py fixtures --tasks piqa hellaswag --output-dir c_impl/fixtures
```

### Running Python Golden Reference
```bash
python scripts/compare_gdn_c.py --fixture c_impl/fixtures/piqa.gdnreq --output results/piqa_python.json --device cuda --dtype float32
python scripts/check_gdn_c_parity.py --python-dir c_impl/results_smoke_python --c-dir c_impl/results_smoke_c --output c_impl/results_smoke_parity.json
```

### Python Training (golden reference model)
```bash
python pretrain.py \
  --train_data_dir $TRAIN_DATA --val_data_dir $VALIDATION_DATA \
  --output_root $SAVE_DIR --exp_name $NAME \
  --model_name $MODEL --train_config $CONFIG \
  --eval_iters $EVAL_ITERS --learning_rate $LR \
  --micro_batch_size $MICRO_BATCH_SIZE
```

### Evaluation
```bash
python scripts/fla_lm_eval.py   # lm-eval-harness based evaluation
```

## Architecture

### HLS Accelerator (`c_impl/`)
The primary development target. Forward-only HLS-synthesizable C implementation of GatedDeltaNet-1.3B inference.

- **`gdn_model.c` / `gdn_model.h`** — HLS top function `gdn_forward` and all synthesized subfunctions: `embed_tokens`, `rms_norm`, `gdn_matmul` (tiled with BRAM local buffers), `silu_mul`, `recurrent_attention` (gated delta rule with depthwise conv1d), `residual_add`.
  - Uses tiled matrix multiplication with `#pragma HLS array_partition`, `#pragma HLS pipeline II=1`, and `#pragma HLS unroll` for compute-bound loops.
  - All loops carry `#pragma HLS loop_tripcount` annotations with comments explaining min/max values.
- **`gdn_eval.c`** — Testbench: loads weights (`.gdnw`), reads pretokenized fixtures (`.gdnreq`), runs `gdn_forward`, writes JSON results.
- **`test.tcl`** — Vitis HLS project script (csim / csynth / cosim).
- **`test_parity.sh`** — Automated parity validation against Python golden results.
- **`Makefile`** — Builds the native C testbench.
- **Formats**: `.gdnw` (flat F32 weight file), `.gdnreq` (pretokenized evaluation fixtures).

### Golden Reference: Python Model (`lit_gpt/`)
- **`model.py`** — Main model classes: `GPT`, `Block`, `MBlock`, `CausalSelfAttention`, `LLaMAMLP`.
- **`gated_delta_net.py`** — GatedDeltaNet layer: chunk-based linear attention with gated delta rule.
- **`config.py`** — Model configs (0.4B, 1.3B, H1 variants).
- **`gated_delta_rule_ops/`** — Triton and FLA kernel implementations (golden reference for HLS).

### Golden Reference: Fused Operations
- **`rmsnorm.py`** — RMSNorm with Triton kernels
- **`fused_rotary_embedding.py`** — Fused RoPE
- **`fused_cross_entropy.py`** — Fused cross-entropy loss

### Training Infrastructure (for producing golden weights)
- **`pretrain.py`** — PyTorch Lightning Fabric + FSDP, BF16 mixed precision.
- **`packed_dataset.py`** — Efficient tokenized data loading (SlimPajama-672B).

## Key Design Patterns

- The HLS C code mirrors the Python model's computation graph exactly to maintain numerical parity.
- Matrix multiplications use a tiled strategy (16x16x16 tiles) with BRAM local buffers and HLS pragmas for parallelism.
- All `memcpy`/`memset` in synthesized code paths have been replaced with explicit labeled loops carrying `loop_tripcount` pragmas for accurate latency estimation.
- Parity is validated end-to-end: C output must match Python golden results within 1e-3 tolerance (observed ~1e-5).
- FLA kernels (`gated_delta_rule_ops/fla_version/`) are preferred for Python golden reference runs.

## Dependencies

**HLS**: Vitis HLS 2022.1, targeting Xilinx VU11P (`xcvu11p-flga2577-1-e`).

**Python (golden reference)**: Container-based setup (see `Dockerfile`). Key deps: PyTorch 2.3.1+CUDA 12.1, Lightning 2.1.2, Triton 2.3.0, flash-attn, mamba-ssm, causal-conv1d, flash-linear-attention (FLA), lm-eval 0.4.1.
