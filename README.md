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

## Evaluation

Three measurements support this accelerator's claims: decode **latency**,
decode **energy**, and **task quality**. Each is collected independently, on
its own hardware, and reported on its own terms. They are comparable only
because they share one workload definition and one pair of measurement
boundaries, which the rest of this section states explicitly. A number
collected under any other protocol is not part of this evaluation.

### The three arms

| Arm | Hardware | Reports | Entry point |
|---|---|---|---|
| Accelerator decode | one Alveo U55C | ms/token, J/token | `c_impl/run_hw_sbatch.sh`, then the power/quality launchers |
| GPU decode | one NVIDIA H100 or A100 | ms/token, J/token | `scripts/run_gpu_power_eval.py` |
| Task quality | GPU prefill, then either decoder | WikiText-2 word perplexity | `scripts/fpga_lm_eval.py`, `scripts/fla_lm_eval.py` |

The two decode arms run in separate jobs, on separate machines, at separate
times. Do not require them to run together: collect each one, verify that both
used the same workload and boundaries, then compare the saved results.

### One workload

Every decode measurement uses batch 1, one token per step, and teacher forcing.
The tracked fixture `c_impl/fixtures_full/wikitext.gdnreq` generates a fixed
4,096-token prompt and a fixed 4,096-token teacher stream;
`scripts/prepare_steady_decode_fixture.py` derives both and records their
hashes. The prompt is prefilled once, then the teacher stream is consumed
cyclically, with recurrent state continuing across wraps. This is sustained
teacher-forced decode, not free-running generation: it fixes the token
sequence so that two implementations do the same work, and it keeps a
trajectory fork from changing what is being timed.

Batch 1 is the point of the design, not a limitation of the harness. The
architectural claim is flat per-token latency with constant memory and no KV
cache, which is a single-stream property.

### Two boundaries, and which one to quote

Both decoders report an inner and an outer time, defined to be analogous:

| Boundary | Accelerator | GPU |
|---|---|---|
| Inner | `kernel_ms`, kernel launch to completion | `device_time_ms`, CUDA events around forward plus argmax |
| Outer | `per_step_tpot_ms`, embedding lookup, upload, launch and wait, on-chip argmax, token read-back | `production_tpot_ms`, input construction and transfer, dispatch, LM head, argmax, token returned to the host |

**Quote the outer boundary when comparing to a GPU.** Emitting a token includes
selecting it, and the GPU reference includes its own argmax inside its timed
window. The inner boundary is for attributing a change to the kernel itself.
Never compare one arm's inner time against the other's outer time.

### Energy

Energy is measured over sustained windows after warmup, never per token.
Report gross joules per token, and idle-subtracted joules per token beside it,
with the idle power that was subtracted. Prefill and warmup are excluded from
the active window.

On the GPU, the runner prefers NVML total-energy-counter differences and falls
back to integrating the sampled power trace, recording per interval which
method was used. Sampling at 0.2 s does not give 0.2 s of sensor resolution:
the standard NVML power query returns a one-second average, which is why the
windows are tens of seconds. The sampler binds to the allocated device by CUDA
UUID rather than by ordinal, so a scheduler or container remapping cannot
silently meter a different card. Board energy needs an exclusively allocated
whole GPU; MIG instances are rejected.

### Running the GPU arm

```bash
python -m pip install -r scripts/requirements_gpu_eval.txt
python scripts/run_gpu_power_eval.py --require-gpu-name H100 \
    --output-dir c_impl/diagnostics/gpu_power/h100-bf16-r1
```

This needs no FPGA, no XRT, and no exported hardware weights. It downloads the
pinned checkpoint unless `--model` names a local one, and refuses to start if
the allocated GPU does not match `--require-gpu-name`. On a cluster, submit
`scripts/slurm_gpu_power_eval.sh` instead of running on a login node. BF16 is
the default; run FP32 as a control into its own output directory. Each run
writes a manifest with the source commit, script hashes, model identity and
resolved environment, so a result can be traced back to what produced it.

The full protocol, the smoke test, the output-file reference and the
validation limits are in
[the GPU evaluation guide](c_impl/doc/gpu_latency_energy_evaluation.md).

### Running the accelerator arm

Build and validate the image first (see *Build and verification* below), then
run the power and quality launchers against that exact image. The historical
FPGA launchers pin a frozen artifact and source pair and will reject a newer
kernel; reproduce them in the original checkout rather than relaxing their
identity gates.

### Comparing the arms

```bash
python scripts/aggregate_power_eval.py \
    --gpu-timing  <gpu>/gpu_timing.json  --gpu-power  <gpu>/gpu_power.jsonl \
    --fpga-timing <fpga>/fpga_timing.json --fpga-power <fpga>/fpga_power.jsonl \
    --output-json comparison.json --output-csv comparison.csv
```

Before running it, confirm both sides used the same fixture hashes and the
same boundary. The aggregator combines results; it does not verify that they
were comparable.

### What these numbers do not claim

The GPU arm measures **eager** PyTorch and FLA decode, which is the
configuration a user gets by default. It is not an optimized GPU baseline: a
CUDA-graph capture of the same step reached roughly 4.2 ms/token on an A100 in
a separate experiment on `worktree-gpu-decode-opt`, verified token-identical to
eager, and that work is deliberately not part of this workflow. Do not present
the eager figure as the best a GPU can do, and do not describe the accelerator
as faster in raw latency than a GPU can be made to go. The durable claims are
flat per-token latency, constant memory with no KV cache, and performance per
watt.

Quality is evaluated with teacher forcing and the decoder's own exported
log-probabilities, because free-running trajectories fork between any two
independent implementations of this model. A fork is not an error; see the
arithmetic-contract discussion in `c_impl/doc/architecture.md`.

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

The wrapper freezes a source snapshot, creates separate build and FPGA jobs
and chains the test with `afterok`; `make -C c_impl run_hw` on the login node
invokes the same wrapper. The flow builds the three kernels of the production
image, closes them at 200 MHz and validates the image on the card (see
`c_impl/doc/reproduce_f200.md`). Run
`make -C c_impl help` for the current weight, state, logit-reference, clock,
device, and output options.

Weights, exported recurrent state, logit dumps, XOs, XCLBINs, build trees, and
diagnostic reports are generated artifacts and are intentionally not committed.

## Documentation

- `c_impl/doc/README.md` — **start here**; says which document is current.
- `c_impl/doc/architecture.md` — current top-level architecture and ABI
  (the 200 MHz three-kernel production image: 12.350 ms/token production
  TPOT and 12.227 ms kernel on card, 0.717 J/token gross).
- `c_impl/doc/gpu_latency_energy_evaluation.md` — the GPU latency and energy
  arm, from a fresh clone.
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
