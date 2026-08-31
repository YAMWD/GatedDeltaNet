# GatedDeltaNet HLS Accelerator

This repository contains a decode-only Vitis HLS accelerator for
GatedDeltaNet-1.3B on the Xilinx Alveo U55C (`xcu55c-fsvh2892-2L-e`). The
C++ implementation under `c_impl/` is the hardware target; `lit_gpt/` and
the evaluation scripts provide the PyTorch/Triton golden reference.

The model is based on [Gated Delta Networks: Improving Mamba2 with Delta
Rule](https://openreview.net/forum?id=r8H7xhYPwz).

## Repository layout

| Path | Purpose |
|---|---|
| `c_impl/` | Decode kernel, native driver, XRT host, physical configuration, and Slurm hardware flow |
| `c_impl/doc/` | Current architecture, block documentation, cycle roadmap, and complete optimization log |
| `lit_gpt/` | PyTorch model and Triton/FLA reference implementation |
| `scripts/` | Weight/state export, correctness checks, profiling, and quality evaluation |
| `pretrain.py` | Optional Lightning/FSDP training entry point |

Historical prefill tops, standalone matmul/attention harnesses, and per-iteration
hardware launchers have been retired. Their source and results remain available
through Git history and `c_impl/doc/optimization_log.md`.

## Accelerator configuration

The synthesized kernel is specialized for one model shape:

| Parameter | Value |
|---|---:|
| Layers | 24 |
| Hidden dimension | 2048 |
| Heads | 8 |
| Head dimension | 256 |
| Intermediate dimension | 5632 |
| Convolution width | 4 |
| Vocabulary | 32,000 |

The external kernel ABI, workspace offsets, weight-port ordering, and recurrent
state layout are documented in `c_impl/doc/architecture.md`.

## Build and verification

Build the native decode driver:

```bash
make -C c_impl
```

Run the short exact correctness gate, or omit `--fast` for the full gate:

```bash
bash scripts/decode_correctness_check.sh --fast
```

Run integrated HLS synthesis with Vitis HLS **2024.2** (pinned as
`VITIS_VERSION` in `c_impl/Makefile`; the native BF16 multiplier requires its
`ap_float`):

```bash
cd c_impl
vitis_hls -f test.tcl
```

Submit the production hardware build and dependent U55C test through Slurm:

```bash
cd c_impl
bash run_hw_sbatch.sh
```

The wrapper creates separate build and FPGA jobs and chains the test with
`afterok`. `make -C c_impl run_hw` is the inner production flow used by
those jobs, not the cluster submission command. Run `make -C c_impl help`
for the current weight, state, logit-reference, clock, device, and output
options.

Weights, exported recurrent state, logit dumps, XOs, XCLBINs, build trees, and
diagnostic reports are generated artifacts and are intentionally not committed.

## Documentation

- `c_impl/doc/README.md` — **start here**; says which document is current.
- `c_impl/doc/architecture.md` — current top-level architecture and ABI
  (Iter66e: all-BF16, 26.654 ms/token on card).
- `c_impl/doc/decode_disaggregated_gemv.md` — 32-port GEMV engine.
- `c_impl/doc/recurrent_attention.md` — recurrent-state implementation.
- `c_impl/doc/cycle_optimization_roadmap.md` — measured cycle roadmap.
- `c_impl/doc/optimization_log.md` — every successful and failed iteration.

## Citation

```bibtex
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

Licensed under the NVIDIA Source Code License-NC. See [LICENSE](LICENSE).
