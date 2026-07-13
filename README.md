# GatedDeltaNet HLS Accelerator

A Vitis HLS hardware accelerator for **GatedDeltaNet-1.3B** inference on
the Xilinx **Alveo U55C** (`xcu55c-fsvh2892-2L-e`). The CPP source under
[`c_impl/`](c_impl/) is the primary development target; the surrounding
PyTorch / Triton implementation is the **golden reference** used to verify
correctness.

The model is from
**[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)**
(Yang, Kautz, Hatamizadeh — ICLR '25).

## Project layout

| Path | Role |
|------|------|
| `c_impl/` | **HLS synthesis target.** Forward-only CPP implementation of GDN-1.3B inference, plus host testbenches and Vitis HLS scripts. See [`c_impl/README.md`](c_impl/README.md). |
| `c_impl/doc/` | Submodule docs: architecture, tiled matmul, depthwise conv, recurrent attention, output-norm, and a chronological optimisation log (v0 → v7). |
| `lit_gpt/` | Python golden reference: `model.py`, `gated_delta_net.py`, FLA / Triton kernels under `gated_delta_rule_ops/`. |
| `pretrain.py`, `packed_dataset.py` | Pretraining harness (Lightning Fabric + FSDP, BF16). Used to produce the weight checkpoint that drives the HLS testbenches; not exercised by the synthesis flow. |
| `scripts/` | Weight / fixture export and parity-check tools: `export_gdn_c.py`, `export_block_fixture.py`, `compare_gdn_c.py`, `check_gdn_c_parity.py`. |
| `Dockerfile` | Reproducible Python environment (PyTorch 2.3.1 + CUDA 12.1, Triton 2.3.0, FLA, lm-eval). |

## Python golden reference

The PyTorch / Triton implementation in [`lit_gpt/`](lit_gpt/) (and the FLA
kernels under [`lit_gpt/gated_delta_rule_ops/fla_version/`](lit_gpt/gated_delta_rule_ops/fla_version/))
is the **golden reference** for the HLS work. Every intermediate tensor that
the CPP testbenches compare against is produced by running the Python model.
The CPP implementation in [`c_impl/gdn_model.cpp`](c_impl/gdn_model.cpp) mirrors
the corresponding Python code path, function for function:

| Python (`lit_gpt/`) | C / HLS (`c_impl/gdn_model.cpp`) |
|---------------------|--------------------------------|
| `model.py` — `GPT`, `Block`, `MBlock` | `gdn_forward` (full 24-layer top function) |
| `gated_delta_net.py` — `GatedDeltaNet` block | `gdn_attn_forward` (single-layer top function) |
| `rmsnorm.py` (Triton) | `gdn_rmsnorm_rows` |
| Linear projections in `gated_delta_net.py` | `gdn_matmul` (tiled) |
| `ShortConvolution` | `gdn_depthwise_conv_silu` |
| `gated_delta_rule_ops/fla_version/` (gated delta rule kernels) | `gdn_recurrent_attention` (persistent BRAM state) |
| `FusedRMSNormSwishGate` (output norm + gate) | `gdn_output_norm_and_gate` |
| SwiGLU MLP (`LLaMAMLP`) | `gdn_swiglu_inplace` + the surrounding matmul calls in `gdn_forward` |

### Model checkpoint

The Python golden reference and all `c_impl/` testbenches consume the
**`m-a-p/1.3B-100B-GatedDeltaNet-pure`** checkpoint published on HuggingFace:

> https://huggingface.co/m-a-p/1.3B-100B-GatedDeltaNet-pure

This is the GDN-1.3B model used throughout the HLS work. `pretrain.py` in
this repo can also be used to train your own checkpoint with the same
architecture; either source is consumed by `scripts/export_gdn_c.py weights`
to produce the flat `.gdnw` weight blob in `c_impl/artifacts/`.

### Parity flow

1. **Obtain the Python checkpoint** — download `m-a-p/1.3B-100B-GatedDeltaNet-pure`
   from HuggingFace (link above), or train your own with `pretrain.py`.
2. **Export** the weights to a flat FP32 blob (`scripts/export_gdn_c.py
   weights → c_impl/artifacts/*.gdnw`) and pretokenise eval fixtures
   (`scripts/export_gdn_c.py fixtures → c_impl/fixtures_smoke/*.gdnreq`,
   `scripts/export_block_fixture.py → c_impl/fixtures_block/*.gdnblk`).
3. **Run the Python reference** on the same fixtures
   (`scripts/compare_gdn_c.py`) to capture the per-task golden JSON.
4. **Run the CPP/HLS implementation** on those fixtures (`gdn_eval`,
   `gdn_attn_test`, or Vitis HLS csim/cosim).
5. **Diff** with `scripts/check_gdn_c_parity.py` (1 × 10⁻³ absolute
   tolerance; observed diffs are ~1 × 10⁻⁵). `c_impl/test_parity.sh`
   automates steps 4–5 for the smoke fixture set.

Any change to the CPP source — and especially anything that perturbs the FP
order of operations (e.g. tree-reduce reordering, tile-shape changes) — is
gated through this loop before being merged.

## Status (single-layer attention, `gdn_attn_forward`)

Vitis HLS 2022.1 csynth on the canonical target `xcu55c-fsvh2892-2L-e`
(Alveo U55C) at a 100 MHz target clock, run via `test_single_GDN_attn.tcl`
(post v7 optimisations):

| Metric | Value |
|--------|------:|
| Top-level latency | **141.03 G cycles** (1.41 s) |
| Timing slack | **0.00 ns** (closes timing with zero margin) |
| Outstanding II violations | **0** |
| BRAM_18K | 322 (7 % of device, 23 % per SLR) |
| DSP | 1042 (11 % of device, 34 % per SLR) |
| LUT | 237 k (18 % of device, 54 % per SLR) |
| FF | 210 k (8 % of device, 24 % per SLR) |
| URAM | 0 (0 %) |

The optimisation work (v0 → v7) drops top-level latency by ≈ 26 % and clears
all II and timing violations vs the un-optimised baseline; the iteration
history is documented in
[`c_impl/doc/optimization_log.md`](c_impl/doc/optimization_log.md). Re-run
`vitis_hls -f test_single_GDN_attn.tcl` to refresh
`c_impl/GDN_single_attn/solution2/syn/report/csynth.rpt` after any source
change.

| Verification | Result |
|--------------|--------|
| `gdn_attn_test` (single-layer parity vs Python golden) | **PASS** (max abs diff 1.2 × 10⁻⁶) |
| `test_parity.sh` (full 24-layer, 9 fixtures) | **PASS** (worst diff 6.22 × 10⁻⁵, tol 1 × 10⁻³) |

Detailed iteration history with concrete latency / II / resource numbers per
pass: [`c_impl/doc/optimization_log.md`](c_impl/doc/optimization_log.md).

## Model configuration

GDN-1.3B is the only configuration the HLS path is wired for:

| Parameter | Value |
|-----------|-------|
| Hidden dim | 2048 |
| Num heads | 8 |
| Head dim (Q/K and V) | 256 |
| Intermediate | 5632 |
| Num layers | 24 |
| Conv kernel | 4 |
| Max seq len | 2048 |
| Vocab | 32 000 |

## Quick start

### Build the CPP testbenches and run parity
```bash
make -C c_impl
cd c_impl && bash test_parity.sh
```

`test_parity.sh` rebuilds `gdn_eval`, runs every `fixtures_smoke/*.gdnreq`
through the CPP implementation, and diffs the JSON outputs against
`results_smoke_python/` via `scripts/check_gdn_c_parity.py`.

### Vitis HLS synthesis
```bash
cd c_impl
vitis_hls -f test_single_GDN_attn.tcl   # primary: single-layer attention on Alveo U55C, csim + csynth
vitis_hls -f test.tcl                   # full 24-layer model on Alveo U55C, csim + csynth + cosim
```

The single-layer csynth report is written to
`c_impl/GDN_single_attn/solution2/syn/report/csynth.rpt`. Both TCL scripts
target `xcu55c-fsvh2892-2L-e` at 10 ns clock period.

### Hardware build & on-card test (Vitis v++)
```bash
cd c_impl
make run_hw                             # xo → xclbin → host → ./host.exe on the U55C
```

`make run_hw` is dependency-driven: it builds the kernel (`gdn_forward.xo`),
links the bitstream (`gdn_forward.xclbin`), and the XRT host (`host.exe`)
only if they are missing or stale, then invokes the on-card test. Override
the fixture or output via Make variables:

```bash
make run_hw FIXTURE=fixtures_smoke/piqa.gdnreq          # HW_OUT auto-derives to hw_piqa.json
make run_hw HW_MAX_EX=4                                  # cap example count
make run_hw HW_DEVICE=2                                  # XRT device index
```

The link wires in `c_impl/pblock_pe_split.tcl` as a pre-place script so the
matmul `ProcessingElement` chain is split across SLR0/SLR1 and the recurrent
attention / output norm modules land in SLR2/SLR1 respectively — this is what
gets the design through route_design at 100 MHz on U55C. The individual
targets (`make xo`, `make xclbin`, `make host`) are still available if you
want to stop at any phase. See [`c_impl/README.md`](c_impl/README.md) for the
full hardware build flow, knobs, and the latest on-card parity numbers.

### Generate weights and fixtures
```bash
# Flat float32 weight blob (~5.6 GB)
python scripts/export_gdn_c.py weights \
    --output c_impl/artifacts/gdn-1.3b-f32.gdnw

# Pretokenised eval fixtures
python scripts/export_gdn_c.py fixtures \
    --tasks piqa hellaswag winogrande arc_easy arc_challenge \
            social_iqa boolq lambada_openai wikitext \
    --output-dir c_impl/fixtures_smoke

# Single-block attention fixture for gdn_attn_test
python scripts/export_block_fixture.py \
    --layer 0 \
    --output c_impl/fixtures_block/block0_attn.gdnblk

# Python golden reference for parity
python scripts/compare_gdn_c.py \
    --fixture c_impl/fixtures_smoke/piqa.gdnreq \
    --output  c_impl/results_smoke_python/piqa.json \
    --device cuda --dtype float32
```

### Pretrain the golden model (optional)
```bash
python pretrain.py \
    --train_data_dir $TRAIN_DATA --val_data_dir $VALIDATION_DATA \
    --output_root $SAVE_DIR --exp_name $NAME \
    --model_name $MODEL --train_config $CONFIG \
    --eval_iters $EVAL_ITERS --learning_rate $LR \
    --micro_batch_size $MICRO_BATCH_SIZE
```

The HLS path consumes the resulting checkpoint via `scripts/export_gdn_c.py`.

## Where the architecture work is documented

- [`c_impl/doc/architecture.md`](c_impl/doc/architecture.md) — top-level
  HLS overview, file map, AXI bundle topology, optimisation history table.
- [`c_impl/doc/decode_disaggregated_gemv.md`](c_impl/doc/decode_disaggregated_gemv.md) —
  decode-only GEMV architecture, multi-reader scaling, and on-card results.
- [`c_impl/doc/depthwise_conv.md`](c_impl/doc/depthwise_conv.md) — pre-buffered
  weights, 4-row sliding window, two-phase per-row split (load vs compute).
- [`c_impl/doc/recurrent_attention.md`](c_impl/doc/recurrent_attention.md) —
  persistent BRAM state, fused two-pass pipeline, P_K=16 column parallelism,
  delta_drain split, tree-reduce, q/k bundle split.
- [`c_impl/doc/output_norm.md`](c_impl/doc/output_norm.md) — on-chip attn /
  gate / weight buffers, tree-reduced sum-of-squares.
- [`c_impl/doc/optimization_log.md`](c_impl/doc/optimization_log.md) —
  v0 → v7 iteration log, per-pass before/after numbers, list of structural
  follow-ups (streaming GEMM, dataflow at `gdn_attn_forward` scope) needed to
  break through the matmul-dominated runtime.

## Dependencies

- **Vitis HLS** 2022.1, targeting Alveo U55C (`xcu55c-fsvh2892-2L-e`).
- **C build**: any C11 compiler; the Makefile uses `gcc -O3 -std=c11` and
  links only `libm`. No BLAS.
- **Python golden reference**: see [`Dockerfile`](Dockerfile). Key pins:
  PyTorch 2.3.1 + CUDA 12.1, Lightning 2.1.2, Triton 2.3.0,
  flash-linear-attention (FLA), lm-eval 0.4.1.

## Citation

If you use this work, please cite the original GatedDeltaNet paper:

```
@inproceedings{yang2025gated,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=r8H7xhYPwz}
}
```

## License

Copyright © 2025, NVIDIA Corporation. All rights reserved.

Licensed under the NVIDIA Source Code License-NC. See [LICENSE](LICENSE) for
details. The HLS accelerator code under `c_impl/` is a derivative work and is
released under the same terms.

## Acknowledgements

Built on:
- Upstream Python implementation: [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (preferred Triton kernels for the golden reference)
- [Samba](https://github.com/microsoft/Samba), [LitGPT](https://github.com/Lightning-AI/litgpt), [TinyLlama](https://github.com/jzhang38/TinyLlama) (training scaffold)
