#!/usr/bin/env bash
#SBATCH --job-name=gdn-fpga-eval-full
#SBATCH --partition=light
#SBATCH --gres=gpu:a100:1,fpga:u55c:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet-eval/c_impl/diagnostics/fpga_eval_full/slurm-%j.out

set -euo pipefail
set -o pipefail

: "${EVAL_TABLE:?set EVAL_TABLE to 2, 3, or 5}"
case "$EVAL_TABLE" in
    2|3|5) ;;
    *) echo "FATAL: EVAL_TABLE must be 2, 3, or 5" >&2; exit 2 ;;
esac

REPO=/home/yaoz0b/GatedDeltaNet-eval
DIAG="$REPO/c_impl/diagnostics/fpga_eval_table${EVAL_TABLE}"
RAW_ROOT=/home/yaoz0b/gdn_fpga_eval/full
RAW="$RAW_ROOT/table${EVAL_TABLE}"
JOB_TMP="/tmp/${USER}-${SLURM_JOB_ID}/fpga-eval-table${EVAL_TABLE}"
PY=/home/yaoz0b/GatedDeltaNet/.micromamba/envs/gdn-hf/bin/python
MODEL=/home/yaoz0b/.cache/huggingface/hub/models--m-a-p--1.3B-100B-GatedDeltaNet-pure/snapshots/930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa
XCLBIN=/home/yaoz0b/gdn_fpga_eval/artifacts/iter67c/gdn_forward.xclbin
WEIGHTS=/home/yaoz0b/gdn_fpga_eval/artifacts/model/gdn-1.3b-bf16w.gdnw
LIVE="$DIAG/table${EVAL_TABLE}-${SLURM_JOB_ID}.live.log"
EXIT_MARKER="$DIAG/table${EVAL_TABLE}-${SLURM_JOB_ID}.exit"

mkdir -p "$DIAG" "$RAW" "$JOB_TMP" "$JOB_TMP/triton"
exec > >(tee -a "$LIVE") 2>&1

host_pid=""
host_done=""
cleanup() {
    rc=$?
    trap - EXIT
    if [[ -n "$host_done" ]]; then
        mkdir -p "$(dirname "$host_done")"
        printf 'done\n' > "$host_done"
    fi
    if [[ -n "$host_pid" ]] && kill -0 "$host_pid" 2>/dev/null; then
        kill "$host_pid" 2>/dev/null || true
        wait "$host_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$rc" > "$EXIT_MARKER"
    echo "TABLE${EVAL_TABLE}_EXIT_CODE=$rc end=$(date --iso-8601=seconds)"
    exit "$rc"
}
trap cleanup EXIT

set +u
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
set -u
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$JOB_TMP/triton"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

cd "$REPO"
echo "table=$EVAL_TABLE node=$(hostname -s) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
echo "gres=${SLURM_JOB_GRES:-unset}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

mapfile -t u55c_cards < <(
    xbutil examine 2>/dev/null |
        grep -i 'xilinx_u55c' |
        grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' |
        tr -d '[]' |
        sort -u
)
if [[ "${#u55c_cards[@]}" -ne 1 ]]; then
    echo "FATAL: expected exactly one allocated U55C, found ${#u55c_cards[@]}" >&2
    exit 2
fi
BDF="${u55c_cards[0]}"
echo "fpga_bdf=$BDF"

echo "[gate] immutable artifact identities"
test "$(sha256sum "$XCLBIN" | awk '{print $1}')" = \
    fb4fc63f76bc1ee485665f21102596270930d6d39643b8eb5ae7d4f899d289ab
test "$(sha256sum "$WEIGHTS" | awk '{print $1}')" = \
    ba81d3536e868e1057b81cc71354060cfba968c1b356b2fa70add3b83a84c298
test "$(sha256sum c_impl/gdn_model.cpp | awk '{print $1}')" = \
    2bc240e6a5cf24b23bd83aa3ad518552bd475e72c8ac5c54a6cdf8bc991ad020
make -C c_impl host

start_host() {
    local phase="$1"
    local results="$2"
    local queue="$JOB_TMP/$phase/queue"
    local log="$DIAG/host_${phase}-${SLURM_JOB_ID}.live.log"
    mkdir -p "$queue" "$results"
    host_done="$queue/producer.done"
    stdbuf -oL -eL c_impl/host.exe "$XCLBIN" "$WEIGHTS" \
        c_impl/fixtures_decode/decode.gdnreq - "$BDF" \
        --eval-queue "$queue" --eval-results "$results" \
        --eval-producer-done "$host_done" --eval-idle-timeout 7200 \
        > "$log" 2>&1 &
    host_pid=$!
    for _ in $(seq 1 60); do
        if grep -q '\[eval\] serving' "$log" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$host_pid" 2>/dev/null; then
            wait "$host_pid"
        fi
        sleep 2
    done
    grep -q '\[eval\] serving' "$log"
    FPGA_QUEUE="$queue"
    FPGA_RESULTS="$results"
    export FPGA_QUEUE FPGA_RESULTS
}

finish_host() {
    printf 'done\n' > "$host_done"
    wait "$host_pid"
    host_pid=""
    host_done=""
}

case "$EVAL_TABLE" in
    2)
        echo "[full] Table 2 S1/S2"
        start_host table2_s12 "$RAW/s12/raw"
        MODEL_ID="$MODEL" OUTPUT_DIR="$RAW/s12/results" \
            REQUEST_PREFIX=full-t2-s12 \
            TASKS=gdn_niah_single_1,gdn_niah_single_2 \
            SEQ_LENGTHS='[1024,2048,4096,8192]' PYTHON_BIN="$PY" \
            scripts/run_fpga_table2_eval.sh --log_samples
        finish_host

        echo "[full] Table 2 S3"
        start_host table2_s3 "$RAW/s3/raw"
        MODEL_ID="$MODEL" OUTPUT_DIR="$RAW/s3/results" \
            REQUEST_PREFIX=full-t2-s3 TASKS=gdn_niah_single_3 \
            SEQ_LENGTHS='[1024,2048,4096]' PYTHON_BIN="$PY" \
            scripts/run_fpga_table2_eval.sh --log_samples
        finish_host
        ;;
    3)
        echo "[full] Table 3"
        start_host table3 "$RAW/raw"
        MODEL_ID="$MODEL" OUTPUT_DIR="$RAW/results" \
            REQUEST_PREFIX=full-t3 PYTHON_BIN="$PY" \
            scripts/run_fpga_table3_eval.sh --log_samples
        finish_host
        ;;
    5)
        echo "[full] Table 5"
        start_host table5 "$RAW/raw"
        "$PY" scripts/run_gdn_longbench_eval.py \
            --model "$MODEL" --dtype bfloat16 --output-dir "$RAW/results" \
            --batch-size 1 --resume \
            --fpga-queue "$FPGA_QUEUE" --fpga-results "$FPGA_RESULTS" \
            --fpga-request-prefix full-t5
        finish_host
        ;;
esac

echo "FPGA_TABLE${EVAL_TABLE}_COMPLETE $(date --iso-8601=seconds)"
