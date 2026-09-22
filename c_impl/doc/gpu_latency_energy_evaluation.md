# GPU latency and energy evaluation (H100 / A100)

This workflow runs on a single NVIDIA GPU without an FPGA, XRT, exported
hardware weights, or the old ACCL directory layout. It measures batch-1
**eager** decode with the complete LM head, argmax and token return to the
CPU. BF16 is the default model dtype; FP32 and FP16 are explicit alternatives.
CUDA graphs from `worktree-gpu-decode-opt` are a separate integration and are
not enabled here. Do not describe this eager baseline as optimized H100
performance.

The implementation is covered by processor-side energy-accounting, identity and
protocol tests, and the complete path has been run end to end on an **A100**
(see *Validation record* below). It has **not** been run on an H100. Nothing
here is H100-specific, but the first run on a destination machine is a
compatibility check, not a result; use the smoke run in section 3.

## 1. Clone and install

Use Linux x86_64, Python 3.11 and an NVIDIA driver compatible with the pinned
PyTorch CUDA 12.6 build. Use an exclusively allocated whole GPU for board
energy measurements; MIG-instance energy attribution is not supported.

```bash
git clone --depth 1 --branch main https://github.com/YAMWD/GatedDeltaNet.git
cd GatedDeltaNet
conda create -y -n gdn-gpu-eval python=3.11 pip
conda activate gdn-gpu-eval
python -m pip install -r scripts/requirements_gpu_eval.txt
python -m pip check
```

The numerical stack is pinned to PyTorch 2.7.1+cu126, Triton 3.3.1,
FLA/fla-core 0.4.2 and Transformers 4.51.3. NVML Python bindings come from
`nvidia-ml-py`. The requirements cover latency/energy; they do not install the
separate lm-evaluation-harness quality suite. Each run saves the complete
resolved `pip freeze` as well.

## 2. Obtain the checkpoint

The runner downloads model `m-a-p/1.3B-100B-GatedDeltaNet-pure`, revision
`930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa`, when `--model` is omitted.
To download before allocating the GPU (or before using an offline compute
node), run:

```bash
export MODEL_ID="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "m-a-p/1.3B-100B-GatedDeltaNet-pure",
    revision="930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa",
))
PY
)"
```

`--model "$MODEL_ID"` accepts the resulting local checkpoint directory. It
does not mean that any local directory has verified weights: the manifest
records its path and config hash, not a fresh hash of every weight shard.
Keep the pinned HF snapshot or record hashes for a custom checkpoint.

## 3. Smoke-test the allocated H100

On a standalone H100, or inside the cluster's allocated GPU job:

```bash
python -u scripts/run_gpu_power_eval.py \
    --model "$MODEL_ID" --require-gpu-name H100 \
    --output-dir c_impl/diagnostics/gpu_power/h100-smoke \
    --idle-seconds 3 --warmup-seconds 2 --interval-seconds 5 --intervals 1
```

This is a compatibility check, not a publishable energy measurement. Require
`status.json` to say `complete`, inspect `summary.json` and confirm the correct
GPU UUID, model and execution dtype. Use a fresh output directory for each
invocation; existing results are never overwritten. `--device cuda:0` means
the first logical CUDA device in the allocation, not physical NVML GPU 0.
The sampler selects the exact UUID reported by that CUDA device.

For Slurm, submit from the repository root. Replace `YOUR_H100_PARTITION` and
the GRES name if your destination uses different names. The wrapper's Slurm
log and detailed live log both remain under repository diagnostics.

```bash
mkdir -p c_impl/diagnostics/gpu_power
export PYTHON_BIN="$(command -v python)"
sbatch --partition=YOUR_H100_PARTITION --gres=gpu:h100:1 \
    --chdir="$PWD" \
    --output="$PWD/c_impl/diagnostics/gpu_power/slurm-%j.out" \
    scripts/slurm_gpu_power_eval.sh \
    --idle-seconds 3 --warmup-seconds 2 --interval-seconds 5 --intervals 1
```

The wrapper preserves `CUDA_VISIBLE_DEVICES`. `GPU_NAME` defaults to H100;
set it to A100 to require an A100 instead. `MODEL_ID` and `PYTHON_BIN` are
inherited through Slurm's normal environment export. `GPU_EVAL_OUTPUT` can
override the default `c_impl/diagnostics/gpu_power/run-JOBID` directory.

## 4. Measure sustained decode

```bash
python -u scripts/run_gpu_power_eval.py \
    --model "$MODEL_ID" --require-gpu-name H100 --dtype bfloat16 \
    --output-dir c_impl/diagnostics/gpu_power/h100-bf16-r1

python -u scripts/run_gpu_power_eval.py \
    --model "$MODEL_ID" --require-gpu-name H100 --dtype float32 \
    --output-dir c_impl/diagnostics/gpu_power/h100-fp32-r1
```

On a cluster, submit the corresponding wrapper jobs with `--dtype bfloat16`
and `--dtype float32`, rather than running these commands on a login node.
The defaults are 60 seconds of loaded-idle measurement, 30 seconds of decode
warmup, then three 60-second active intervals. Repeat whole runs into new
directories to measure variability. Each interval continues the state from
the preceding interval; intervals within a run are not independent restarts.

The tracked WikiText fixture generates a fixed 4,096-token prompt and a
4,096-token teacher stream. The implementation prefills `prompt[:-1]` with
the chunk kernel, then cyclically consumes `[prompt[-1], *teacher]` through
warmup and measurement. Decode uses `fused_recurrent` by default. The state
continues across token-stream wraps. Record any changes to this protocol;
this is sustained teacher-forced decode, not free-running text generation.
The clean FLA environment retains its normal recurrent-state precision;
loading BF16 model weights does not apply the separate BF16-state patch.

## 5. Read the outputs

| File | Contents |
|---|---|
| `summary.json`, `summary.csv` | TPOT, tokens/s, gross J/token, tokens/J, idle-subtracted values |
| `gpu_timing.json` | Per-window timing, energy-counter endpoints, GPU identity, dtype and numerical settings |
| `gpu_power.jsonl` | NVML identity and timestamped power samples, including a closing sample |
| `manifest.json`, `environment.txt` | Command, source commit/hashes, model config identity, environment and Slurm allocation |
| `fixture/manifest.json` | Fixture, prompt and teacher-token hashes |
| `benchmark.log`, `status.json` | Progress/failures and completion status |

`production_tpot_ms` includes input construction/transfer, eager dispatch,
LM head, argmax and return of the selected token to the CPU. `device_time_ms`
uses CUDA events around forward plus argmax; it excludes input construction.
Energy and throughput cover the entire active window, including loop and
timing overhead. Prefill and warmup are excluded from active energy; host CPU
and whole-server power are not measured. GPU clocks, power limit, driver,
memory size and compute capability are recorded rather than assumed from a
generic H100 label.

Gross energy uses advancing NVML total-energy-counter differences where
supported, converting millijoules to joules. Otherwise it integrates the
sampled power trace using trapezoids. Each interval records the method and
the integrated sampled energy for comparison. A counter reset is an error.
An unsupported or non-advancing counter falls back to the power trace.
Timing windows must be covered by telemetry even when counters are used.
Idle-subtracted tokens/J is null if the measured net energy is nonpositive.

H100's standard NVML power-usage query returns a one-second average. Polling
at 0.2 seconds does not create 0.2-second independent sensor readings; use
the sustained windows above. Counter snapshots occur immediately outside
the timed loops, so their small read overhead is not included in per-token
latency. See NVIDIA's [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html).

To compare separately collected FPGA and H100 runs, after verifying the
same workload and measurement boundaries:

```bash
python scripts/aggregate_power_eval.py \
    --gpu-timing /path/to/h100/gpu_timing.json \
    --gpu-power /path/to/h100/gpu_power.jsonl \
    --fpga-timing /path/to/fpga/fpga_timing.json \
    --fpga-power /path/to/fpga/fpga_power.jsonl \
    --output-json /path/to/comparison.json --output-csv /path/to/comparison.csv
```

The old `slurm_fpga_eval_*`, `slurm_fpga_tpot_power.sh` and
`collect_fpga_eval_manifest.py` target a frozen Iter67c artifact/source
combination. Their identity gates deliberately reject the newer kernel
source on main. Reproduce that historical flow in the original eval checkout
`a56e1fb6cae622df2eae533980563ce4a423551f`, or qualify a new matched hardware
artifact/source set before adapting the gates. Do not substitute a new source
hash merely to bypass a failed check. The GPU-only workflow does not use any
of these legacy FPGA launchers.

## Validation record

Run on 2026-09-22, Slurm job 4531 on `acclnode01`, submitted through
`scripts/slurm_gpu_power_eval.sh` exactly as documented above. Slurm reports
COMPLETED in 3 min 56 s, the wrapper exits 0 and `status.json` says `complete`.
This is what establishes that the workflow runs; it is a validation of the
tooling on the GPU that was available, not a published A100 result and not an
H100 result.

**Allocation and stack.** NVIDIA A100 80GB PCIe, UUID
`GPU-94af67ba-264c-0507-4650-d893ae1d3021`, driver 570.133.07, 300 W power
limit, SM 1320 MHz, memory 1512 MHz. Python 3.11.15, PyTorch 2.7.1+cu126,
CUDA 12.6, Triton 3.3.1, FLA 0.4.2, Transformers 4.51.3. Checkpoint at the
pinned revision `930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa`. Configuration
BF16, `fused_recurrent`, 20 s idle, 20 s warmup, three 30 s intervals.

**Result.**

| Quantity | Value |
|---|---:|
| Production TPOT | 32.331 ms/token |
| Device time (CUDA events) | 32.257 ms/token |
| Throughput | 30.92 tokens/s |
| Gross energy | 2.731 J/token |
| Idle-subtracted energy | 0.579 J/token |
| Idle power | 66.55 W |
| Active power, mean / median / p95 | 84.44 / 84.77 / 85.95 W |

**What the run proves, beyond completing.**

- The NVML **total-energy counter was supported and used** on all three
  intervals, so the preferred path was exercised rather than the fallback.
- Counter energy and integrated sampled power **agree to within 0.09%, 0.13%
  and 0.23%** on the three intervals (2504.59 vs 2506.86 J, 2546.18 vs
  2549.58, 2552.21 vs 2558.16). Two independent methods measuring the same
  window is the strongest internal check available, and it passes.
- The sampler bound to the **UUID that CUDA reported** for the allocated
  device, and the first and closing samples both landed, so the timing windows
  are fully covered by telemetry.
- The fixture, prompt and teacher hashes were recorded, and the manifest
  captured the source commit, script hashes, model identity and environment.
- TPOT of 32.3 ms is consistent with the repository's independently obtained
  ~35 ms/token A100 BF16 eager reference, which is a sanity check on the
  workload rather than a new measurement of it.

**One finding worth acting on.** The first interval ran 884 steps and the next
two ran 950 each, 7% slower, and its energy per token was correspondingly
higher (2.833 vs 2.680 and 2.687 J). Twenty seconds of warmup was not enough
on this stack. Use the documented default of 30 s or more, and treat a first
interval that differs from the rest as unwarmed rather than averaging it in.

## Software checks

```bash
PYTHONPATH=scripts python -m unittest -v \
    scripts/test_gpu_power_eval.py scripts/test_fpga_eval_client.py
```

Use Python 3.9 or newer; the pinned and tested version is 3.11. Under 3.8 the
CUDA-UUID test fails inside `str.removeprefix`, which looks like a device-
selection defect and is only the interpreter being too old.

These checks cover numerical energy integration, uneven sampling, counter
units/reset/fallback, UUID mismatches, incomplete traces, zero net energy,
GPU-only and paired output formats, CUDA-to-NVML UUID selection, MIG
rejection and the sampler's first/closing samples. They do not establish GPU
model correctness or measured H100 latency/energy.
