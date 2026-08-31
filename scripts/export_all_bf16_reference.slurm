#!/bin/bash
#SBATCH --job-name=gdn_allbf16_ref
#SBATCH --partition=light
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

GDN_REPO_ROOT="${GDN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/yaoz0b/GatedDeltaNet}}"
REPO_ROOT="$(cd "${GDN_REPO_ROOT}" && pwd)"
TAG="${TAG:-all_bf16_reference_${SLURM_JOB_ID:-manual}}"
DIAG="${REPO_ROOT}/c_impl/diagnostics/${TAG}"
mkdir -p "${DIAG}"
exec > >(tee "${DIAG}/gpu_reference.live.log") 2>&1

finish() {
    rc=$?
    printf '%s\n' "${rc}" > "${DIAG}/gpu_reference.exit"
    echo "EXIT_CODE=${rc} end=$(date --iso-8601=seconds)"
}
trap finish EXIT

MODEL_DIR="${MODEL_DIR:-/home/yaoz0b/gdn_precision_eval_20260817/checkpoint_bf16_mixed}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.micromamba/envs/gdn-hf/bin/python}"

cd "${REPO_ROOT}"

STATE_OUT="${STATE_OUT:-c_impl/fixtures_decode/decode_ex0_all_bf16.gdnstate}"
GOLDEN_OUT="${GOLDEN_OUT:-c_impl/results_decode_golden/decode_all_bf16.decode.json}"
LOGITS_OUT="${LOGITS_OUT:-c_impl/artifacts/decode_all_bf16_64.gdnlog}"
HIDDEN_OUT="${HIDDEN_OUT:-}"
NATIVE_BF16_PRODUCT="${NATIVE_BF16_PRODUCT:-0}"

echo "=== independent all-BF16 decode reference ==="
echo "  node       : $(hostname)"
echo "  job        : ${SLURM_JOB_ID:-none}"
echo "  model      : ${MODEL_DIR}"
echo "  state      : ${STATE_OUT}"
echo "  golden     : ${GOLDEN_OUT}"
echo "  logits     : ${LOGITS_OUT}"
echo "  hidden     : ${HIDDEN_OUT:-<not requested>}"
echo "  product    : $([ "${NATIVE_BF16_PRODUCT}" = 1 ] && echo native-bf16-rounded || echo exact-bf16)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum \
    "${REPO_ROOT}/scripts/export_gdn_state.py" \
    "${REPO_ROOT}/scripts/gdn_native_bf16_product.py" \
    "${REPO_ROOT}/scripts/export_all_bf16_reference.slurm" \
    > "${DIAG}/source_hashes.sha256"

hidden_args=()
if [ -n "${HIDDEN_OUT}" ]; then
    hidden_args=(--hidden-output "${HIDDEN_OUT}")
fi
product_args=()
if [ "${NATIVE_BF16_PRODUCT}" = "1" ]; then
    product_args=(--native-bf16-product)
fi

"${PYTHON_BIN}" scripts/export_gdn_state.py \
    --model-name "${MODEL_DIR}" \
    --fixture c_impl/fixtures_decode/decode.gdnreq \
    --example 0 \
    --output "${STATE_OUT}" \
    --device cuda \
    --dtype bfloat16 \
    --recurrent-storage bf16 \
    --conv-storage bf16 \
    --require-all-bf16 \
    --fp32-logits \
    --verify-steps 8 \
    --decode-steps 64 \
    --golden-output "${GOLDEN_OUT}" \
    --logits-output "${LOGITS_OUT}" \
    "${product_args[@]}" \
    "${hidden_args[@]}"

sha256sum "${STATE_OUT}" "${GOLDEN_OUT}" "${LOGITS_OUT}"
if [ -n "${HIDDEN_OUT}" ]; then
    sha256sum "${HIDDEN_OUT}"
fi
