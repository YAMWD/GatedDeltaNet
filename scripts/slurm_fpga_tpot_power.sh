#!/usr/bin/env bash
#SBATCH --job-name=gdn-fpga-power
#SBATCH --partition=light
#SBATCH --gres=gpu:a100:1,fpga:u55c:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet-eval/c_impl/diagnostics/fpga_eval_power/slurm-%j.out

set -euo pipefail
set -o pipefail

REPO=/home/yaoz0b/GatedDeltaNet-eval
DIAG="$REPO/c_impl/diagnostics/fpga_eval_power"
RAW=/home/yaoz0b/gdn_fpga_eval/full/performance
PY=/home/yaoz0b/GatedDeltaNet/.micromamba/envs/gdn-hf/bin/python
MODEL=/home/yaoz0b/.cache/huggingface/hub/models--m-a-p--1.3B-100B-GatedDeltaNet-pure/snapshots/930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa
XCLBIN=/home/yaoz0b/gdn_fpga_eval/artifacts/iter67c/gdn_forward.xclbin
WEIGHTS=/home/yaoz0b/gdn_fpga_eval/artifacts/model/gdn-1.3b-bf16w.gdnw
WIKITEXT=/home/yaoz0b/GatedDeltaNet/c_impl/fixtures_full/wikitext.gdnreq
LIVE="$DIAG/power-${SLURM_JOB_ID}.live.log"
EXIT_MARKER="$DIAG/power-${SLURM_JOB_ID}.exit"

mkdir -p "$DIAG" "$RAW/fixture"
exec > >(tee -a "$LIVE") 2>&1

fpga_sampler=""
gpu_sampler=""
cleanup() {
    rc=$?
    trap - EXIT
    touch "$RAW/fpga_power.stop" "$RAW/gpu_power.stop"
    for pid in "$fpga_sampler" "$gpu_sampler"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    printf '%s\n' "$rc" > "$EXIT_MARKER"
    echo "POWER_EXIT_CODE=$rc end=$(date --iso-8601=seconds)"
    exit "$rc"
}
trap cleanup EXIT

set +u
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
set -u
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/${USER}-${SLURM_JOB_ID}/triton"
mkdir -p "$TRITON_CACHE_DIR"

cd "$REPO"
echo "node=$(hostname -s) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
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

test "$(sha256sum "$XCLBIN" | awk '{print $1}')" = \
    fb4fc63f76bc1ee485665f21102596270930d6d39643b8eb5ae7d4f899d289ab
test "$(sha256sum "$WEIGHTS" | awk '{print $1}')" = \
    ba81d3536e868e1057b81cc71354060cfba968c1b356b2fa70add3b83a84c298
test "$(sha256sum c_impl/gdn_model.cpp | awk '{print $1}')" = \
    2bc240e6a5cf24b23bd83aa3ad518552bd475e72c8ac5c54a6cdf8bc991ad020
make -C c_impl host

echo "[fixture] deterministic 4K prompt and teacher stream"
"$PY" scripts/prepare_steady_decode_fixture.py \
    --fixture "$WIKITEXT" --output-dir "$RAW/fixture"
"$PY" scripts/export_fpga_handoff_state.py \
    --model "$MODEL" --tokens "$RAW/fixture/prompt_4096.tokens" \
    --output "$RAW/fixture/prompt_4096.gdnstate" \
    --metadata "$RAW/fixture/prompt_4096.state.json"

echo "[benchmark] U55C idle, warmup, and three active intervals"
rm -f "$RAW/fpga_power.stop"
"$PY" scripts/sample_device_power.py --device fpga --fpga-bdf "$BDF" \
    --output "$RAW/fpga_power.jsonl" --stop-file "$RAW/fpga_power.stop" &
fpga_sampler=$!
c_impl/host.exe "$XCLBIN" "$WEIGHTS" c_impl/fixtures_decode/decode.gdnreq \
    "$RAW/fpga_timing.json" "$BDF" --decode \
    --decode-from-state "$RAW/fixture/prompt_4096.gdnstate" \
    --benchmark-tokens "$RAW/fixture/fpga_stream_4097.tokens" \
    --benchmark-idle-seconds 60 --benchmark-warmup-seconds 30 \
    --benchmark-interval-seconds 60 --benchmark-intervals 3
touch "$RAW/fpga_power.stop"
wait "$fpga_sampler"
fpga_sampler=""

echo "[benchmark] A100 FP32 eager idle, warmup, and three active intervals"
rm -f "$RAW/gpu_power.stop"
"$PY" scripts/sample_device_power.py --device gpu \
    --output "$RAW/gpu_power.jsonl" --stop-file "$RAW/gpu_power.stop" &
gpu_sampler=$!
"$PY" scripts/benchmark_gpu_fp32.py \
    --model "$MODEL" \
    --prompt-tokens "$RAW/fixture/prompt_4096.tokens" \
    --teacher-tokens "$RAW/fixture/teacher_4096.tokens" \
    --output "$RAW/gpu_timing.json" \
    --idle-seconds 60 --warmup-seconds 30 \
    --interval-seconds 60 --intervals 3
touch "$RAW/gpu_power.stop"
wait "$gpu_sampler"
gpu_sampler=""

"$PY" scripts/aggregate_power_eval.py \
    --fpga-timing "$RAW/fpga_timing.json" \
    --fpga-power "$RAW/fpga_power.jsonl" \
    --gpu-timing "$RAW/gpu_timing.json" \
    --gpu-power "$RAW/gpu_power.jsonl" \
    --output-json "$RAW/comparison.json" \
    --output-csv "$RAW/comparison.csv"

echo "FPGA_GPU_POWER_EVAL_COMPLETE $(date --iso-8601=seconds)"
