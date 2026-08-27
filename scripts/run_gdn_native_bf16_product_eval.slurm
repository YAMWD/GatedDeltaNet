#!/bin/bash
#SBATCH --job-name=gdn_bf16_native_mul
#SBATCH --partition=light
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/bf16_native_product_quality_20260827/slurm-%j.out

# Quality evaluation for the prospective native ap_float<16,8> FPGA multiply:
# BF16 x BF16 -> BF16 (RNE at every product), followed by FP32 accumulation.

set -uo pipefail

REPO=/home/yaoz0b/GatedDeltaNet
DIAG="$REPO/c_impl/diagnostics/bf16_native_product_quality_20260827"
EVAL_ROOT=/home/yaoz0b/gdn_precision_eval_20260827
ARM_ROOT="$EVAL_ROOT/native_bf16_product"
CHECKPOINT=/home/yaoz0b/gdn_precision_eval_20260817/checkpoint_bf16_mixed
PY="$REPO/.micromamba/envs/gdn-hf/bin/python"
LIVE_LOG="$DIAG/evaluation.live.log"
EXIT_MARKER="$DIAG/exit_code"
TRITON_WORK="/tmp/${USER}-${SLURM_JOB_ID}/triton-cache"

mkdir -p "$DIAG" "$ARM_ROOT" "$TRITON_WORK"
exec > >(tee -a "$LIVE_LOG") 2>&1

cleanup() {
    rc=$?
    trap - EXIT
    echo "--- restoring FLA recurrent kernel ---"
    "$REPO/scripts/gdn_bf16_state_patch.sh" restore || true
    "$REPO/scripts/gdn_bf16_state_patch.sh" status || true
    printf '%s\n' "$rc" > "$EXIT_MARKER"
    echo "JOB_EXIT_CODE=$rc END=$(date --iso-8601=seconds)"
    exit "$rc"
}
trap cleanup EXIT

cd "$REPO"
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$TRITON_WORK"
export GDN_NATIVE_BF16_PRODUCT=1
export GDN_NATIVE_BF16_PRODUCT_MANIFEST="$ARM_ROOT/native_bf16_product_manifest.json"
export CUDA_VISIBLE_DEVICES=0

echo "=== native-BF16-product quality evaluation ==="
echo "node=$(hostname) job=${SLURM_JOB_ID:-unset} start=$(date --iso-8601=seconds)"
echo "checkpoint=$CHECKPOINT"
echo "output=$ARM_ROOT"
echo "contract=BF16xBF16 -> RNE BF16 product -> FPGA-associated FP32 reduction"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

echo "--- gate 1: isolated arithmetic and generated PTX ---"
"$PY" scripts/test_gdn_native_bf16_product.py --benchmark \
    2>&1 | tee "$DIAG/arithmetic_gate.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "FATAL: arithmetic/PTX gate failed ($rc)" >&2
    exit "$rc"
fi

echo "--- gate 2: unpatched recurrent state is genuinely FP32 ---"
scripts/gdn_bf16_state_patch.sh status
"$PY" scripts/verify_bf16_state_patch.py --model "$CHECKPOINT" \
    --dtype bfloat16 --expect fp32 2>&1 | tee "$DIAG/state_control.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "FATAL: recurrent-state control gate failed ($rc)" >&2
    exit "$rc"
fi

echo "--- applying and verifying per-token BF16 recurrent persistence ---"
scripts/gdn_bf16_state_patch.sh apply
scripts/gdn_bf16_state_patch.sh status
"$PY" scripts/verify_bf16_state_patch.py --model "$CHECKPOINT" \
    --dtype bfloat16 --expect bf16 2>&1 | tee "$DIAG/state_bf16_gate.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "FATAL: BF16 recurrent-state gate failed ($rc)" >&2
    exit "$rc"
fi

echo "--- gate 3: full-model dense patch and FP32-logit smoke test ---"
"$PY" scripts/verify_gdn_native_bf16_product_model.py \
    --model "$CHECKPOINT" --sequence-length 8 \
    2>&1 | tee "$DIAG/model_smoke.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "FATAL: full-model product-rounded gate failed ($rc)" >&2
    exit "$rc"
fi

echo "--- full Tables 2, 3 and 5 evaluation ---"
MODEL_ID="$CHECKPOINT" DTYPE=bfloat16 OUT_ROOT="$ARM_ROOT" \
    PYTHON_BIN="$PY" scripts/run_gdn_precision_eval_arm.sh \
    2>&1 | tee "$ARM_ROOT/arm.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "FATAL: quality evaluation failed ($rc)" >&2
    exit "$rc"
fi

echo "QUALITY_EVALUATION_COMPLETE $(date --iso-8601=seconds)"
exit 0
