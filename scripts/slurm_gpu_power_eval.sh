#!/usr/bin/env bash
# Submit from the repository root. Select the destination's H100 partition/GRES
# on the sbatch command line; no FPGA or XRT installation is required.
#SBATCH --job-name=gdn-gpu-power
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=c_impl/diagnostics/gpu_power/slurm-%j.out

set -euo pipefail
REPO="${GDN_REPO:-${SLURM_SUBMIT_DIR:?submit from the repository root}}"
cd "$REPO"
PY="${PYTHON_BIN:-python}"
OUT="${GPU_EVAL_OUTPUT:-$REPO/c_impl/diagnostics/gpu_power/run-${SLURM_JOB_ID}}"
# The runner insists on a new/empty OUT. Keep its wrapper log beside it.
mkdir -p "$(dirname "$OUT")"
exec > >(tee -a "${OUT}.live.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "${OUT}.exit"' EXIT
args=(--output-dir "$OUT" --require-gpu-name "${GPU_NAME:-H100}")
if [[ -n "${MODEL_ID:-}" ]]; then
    args+=(--model "$MODEL_ID")
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}-${SLURM_JOB_ID}/gdn-triton}"
export TOKENIZERS_PARALLELISM=false
"$PY" -u scripts/run_gpu_power_eval.py "${args[@]}" "$@"
