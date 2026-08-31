#!/bin/bash
#SBATCH --job-name=gdn_allbf16_native
#SBATCH --partition=build
#SBATCH --nodelist=acclnode03
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

GDN_REPO_ROOT="${GDN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/yaoz0b/GatedDeltaNet}}"
REPO_ROOT="$(cd "${GDN_REPO_ROOT}" && pwd)"
C_IMPL="${REPO_ROOT}/c_impl"
TAG="${TAG:-iter64_native_reference}"
DIAG="${C_IMPL}/diagnostics/${TAG}"
STAGE="/tmp/${USER}-${SLURM_JOB_ID}/native_all_bf16"
WEIGHTS="${WEIGHTS:-${C_IMPL}/artifacts/gdn-1.3b-bf16w.gdnw}"
FIXTURE="${FIXTURE:-${C_IMPL}/fixtures_decode/decode.gdnreq}"
STATE="${STATE:-${C_IMPL}/fixtures_decode/decode_ex0_all_bf16.gdnstate}"
GOLDEN="${GOLDEN:-${C_IMPL}/results_decode_golden/decode_all_bf16.decode.json}"
GPU_LOGITS_REFERENCE="${GPU_LOGITS_REFERENCE:-${C_IMPL}/artifacts/decode_all_bf16_64.gdnlog}"
LOGITS_OUT="${LOGITS_OUT:-${C_IMPL}/artifacts/decode_all_bf16_native_64.gdnlog}"
HIDDEN_OUT="${HIDDEN_OUT:-}"
DECODE_LEN="${DECODE_LEN:-64}"
RUN_GPU_LOGITS_GATE="${RUN_GPU_LOGITS_GATE:-1}"
LAUNCH_HW="${LAUNCH_HW:-0}"
HW_TAG="${HW_TAG:-iter65_fulldsp_reduce_hw}"
EXPECTED_SOURCE_SHA="${EXPECTED_SOURCE_SHA:-}"

mkdir -p "${DIAG}" "${STAGE}"
exec > >(tee "${DIAG}/native_reference.live.log") 2>&1

finish() {
    rc=$?
    printf '%s\n' "${rc}" > "${DIAG}/native_reference.exit"
    echo "EXIT_CODE=${rc} end=$(date --iso-8601=seconds)"
}
trap finish EXIT

for source_file in gdn_model.cpp gdn_model.h gdn_eval.cpp; do
    cp "${C_IMPL}/${source_file}" "${STAGE}/${source_file}"
done
sha256sum "${STAGE}/gdn_model.cpp" "${STAGE}/gdn_model.h" \
          "${STAGE}/gdn_eval.cpp" > "${DIAG}/source_hashes.txt"

echo "=== all-BF16 native exact-logit reference ==="
echo "  node    : $(hostname)"
echo "  job     : ${SLURM_JOB_ID}"
echo "  stage   : ${STAGE}"
echo "  weights : ${WEIGHTS}"
echo "  state   : ${STATE}"
echo "  golden  : ${GOLDEN}"
echo "  GPU log : ${GPU_LOGITS_REFERENCE}"
echo "  hidden  : ${HIDDEN_OUT:-<not requested>}"
echo "  length  : ${DECODE_LEN}"

for required in "${WEIGHTS}" "${FIXTURE}" "${STATE}" "${GOLDEN}" \
                "${GPU_LOGITS_REFERENCE}"; do
    if [ ! -s "${required}" ]; then
        echo "FATAL: missing all-BF16 validation artifact: ${required}" >&2
        exit 2
    fi
done

staged_source_sha="$(sha256sum "${STAGE}/gdn_model.cpp" | awk '{print $1}')"
if [ -n "${EXPECTED_SOURCE_SHA}" ] && \
   [ "${staged_source_sha}" != "${EXPECTED_SOURCE_SHA}" ]; then
    echo "FATAL: staged gdn_model.cpp SHA ${staged_source_sha} does not match" >&2
    echo "       submitted SHA ${EXPECTED_SOURCE_SHA}" >&2
    exit 2
fi

g++ -O3 -std=c++14 -Wno-unknown-pragmas -Wno-unused-label \
    -I/tools/Xilinx/Vitis_HLS/2022.1/include \
    -o "${STAGE}/gdn_eval" "${STAGE}/gdn_eval.cpp" \
    "${STAGE}/gdn_model.cpp" -lm

hidden_args=()
if [ -n "${HIDDEN_OUT}" ]; then
    hidden_args=(--hidden-dump "${STAGE}/decode_native.gdnhid")
fi

"${STAGE}/gdn_eval" "${WEIGHTS}" "${FIXTURE}" \
    "${STAGE}/decode_native.json" \
    --decode --decode-from-state "${STATE}" --decode-len "${DECODE_LEN}" \
    --logits-dump "${STAGE}/decode_all_bf16_native.gdnlog" \
    "${hidden_args[@]}"

python3 "${REPO_ROOT}/scripts/check_gdn_c_parity.py" \
    --decode --golden "${GOLDEN}" --c "${STAGE}/decode_native.json" \
    | tee "${DIAG}/parity.log"

# The native driver already compares every produced logit against its
# independent scalar FP32 LM-head implementation.  This second comparison
# covers every one of the 63 x 32,000 values against the independent CUDA
# all-BF16 trajectory. CUDA and HLS associate FP32 reductions differently
# before BF16 boundaries, so use the documented BF16 vector envelope rather
# than the scalar FP32 implementation-reproduction tolerance. Hardware versus
# this native file remains bit-exact in the XRT gate.
if [ "${RUN_GPU_LOGITS_GATE}" = "1" ]; then
    python3 "${REPO_ROOT}/scripts/compare_gdn_logits.py" \
        "${STAGE}/decode_all_bf16_native.gdnlog" \
        "${GPU_LOGITS_REFERENCE}" --profile bf16-gpu \
        | tee "${DIAG}/gpu_logits_parity.log"
else
    echo "GPU_ALL_LOGITS_GATE_SKIPPED diagnostic-only"
fi

cp "${STAGE}/decode_native.json" "${DIAG}/decode_native.json"
cp "${STAGE}/decode_all_bf16_native.gdnlog" "${LOGITS_OUT}"
if [ -n "${HIDDEN_OUT}" ]; then
    cp "${STAGE}/decode_native.gdnhid" "${HIDDEN_OUT}"
fi
sha256sum "${LOGITS_OUT}" "${GPU_LOGITS_REFERENCE}" "${STATE}" "${GOLDEN}" \
    > "${DIAG}/reference_hashes.txt"
if [ -n "${HIDDEN_OUT}" ]; then
    sha256sum "${HIDDEN_OUT}" >> "${DIAG}/reference_hashes.txt"
fi
echo "NATIVE_ALL_BF16_REFERENCE_DONE $(date --iso-8601=seconds)"

if [ "${LAUNCH_HW}" = "1" ]; then
    if [ "${DECODE_LEN}" != "64" ] || [ "${RUN_GPU_LOGITS_GATE}" != "1" ]; then
        echo "FATAL: hardware launch requires DECODE_LEN=64 and GPU logit gate" >&2
        exit 2
    fi
    current_source_sha="$(sha256sum "${C_IMPL}/gdn_model.cpp" | awk '{print $1}')"
    if [ "${current_source_sha}" != "${staged_source_sha}" ]; then
        echo "FATAL: working source changed during the native gate; refusing" >&2
        echo "       to launch a different hardware snapshot" >&2
        echo "       gate=${staged_source_sha} current=${current_source_sha}" >&2
        exit 2
    fi
    echo "NATIVE_GATE_PASS_SUBMIT_HW tag=${HW_TAG}"
    (
        cd "${C_IMPL}"
        JOBS=48 HLS_FREQ=150 LINK_FREQ=100 \
        VIVADO_SYNTH_JOBS=16 VIVADO_IMPL_JOBS=8 \
        BUILD_NODE=acclnode03 BUILD_MEM=192G \
        WEIGHTS="${WEIGHTS}" \
        DECODE_STATE="${STATE}" \
        DECODE_GOLDEN="${GOLDEN}" \
        LOGITS_REFERENCE="${LOGITS_OUT}" \
        GPU_LOGITS_REFERENCE="${GPU_LOGITS_REFERENCE}" \
        bash run_hw_sbatch.sh "${HW_TAG}"
    ) | tee "${DIAG}/hw_submission.log"
fi
