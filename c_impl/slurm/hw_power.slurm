#!/bin/bash
#SBATCH --partition=light
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Qualification, part 2 (Iter88: chained by run_hw_sbatch.sh after the card gates): the
# paired-power protocol's FPGA arm on this submission's image -- 4,096-token context, 60 s
# loaded idle, 30 s warm-up, 3 x 60 s of sustained teacher-forced decode, the card's power
# sensor sampled once per second -- summarized per interval and overall (J/token gross and
# idle-subtracted).  POWER_GPU_ARMS="float32 bfloat16" adds the paired A100 arms in the same
# job and the full FPGA-vs-GPU comparison (aggregate_power_eval.py); the launcher requests the
# GPU only then, or when the 4,096-token fixture must be generated.  The gres comes from the
# launcher (--gres on sbatch).  Env: TAG SUBMIT_C_IMPL POWER_FIXTURE_DIR POWER_GPU_ARMS WEIGHTS.
# Writes diagnostics/<TAG>/qual/power/ and power.exit.
set -o pipefail
TAG="${TAG:?TAG must be set by the submitter}"
C="${SUBMIT_C_IMPL:?SUBMIT_C_IMPL must be set by the submitter}"
cd "$C" || exit 2
REPO="$(cd "$C/.." && pwd)"
DIAG="$C/diagnostics/$TAG"
V="$DIAG/qual/power"; mkdir -p "$V"
MARK="$V/power.exit"; rm -f "$MARK"
PY="${GDN_PY:-$REPO/.micromamba/envs/gdn-hf/bin/python}"; [ -x "$PY" ] || PY=python3
FX="${POWER_FIXTURE_DIR:-diagnostics/power_fixture_4096}"; [ "${FX:0:1}" = / ] || FX="$C/$FX"
WEIGHTS="${WEIGHTS:-artifacts/gdn-1.3b-bf16w.gdnw}"
MODEL="${GDN_HF_MODEL:-$HOME/.cache/huggingface/hub/models--m-a-p--1.3B-100B-GatedDeltaNet-pure/snapshots/930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa}"
ARMS="${POWER_GPU_ARMS:-}"
sampler=""
cleanup() { rc=$?; trap - EXIT; touch "$V/stop" 2>/dev/null; [ -n "$sampler" ] && wait "$sampler" 2>/dev/null; echo "$rc" > "$MARK"; echo "POWER_EXIT=$rc end=$(date -u +%FT%TZ)"; exit "$rc"; }
trap cleanup EXIT
echo "node=$(hostname -s) job=$SLURM_JOB_ID tag=$TAG gres=${SLURM_JOB_GRES:-?} start=$(date -u +%FT%TZ)"
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || { echo FATAL xrt; exit 2; }
set -u
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
XCL="$(cat "$DIAG/xclbin.path")"; test -s "$XCL" || exit 2
grep -E 'gdn_p\.xclbin$|gdn_forward\.xclbin$' "$DIAG/copied_artifacts.sha256" | sha256sum -c - || exit 2
test -s "$WEIGHTS" || { echo "FATAL missing $WEIGHTS"; exit 2; }
HOST="$V/host.$(hostname -s).exe"
bash "$DIAG/build_host_on_node.sh" "$DIAG" "$HOST" || exit 3
BDF="$(xbutil examine 2>/dev/null | grep -i xilinx_u55c | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | tr -d '[]' | sort -u)"
[ "$(printf '%s\n' "$BDF" | wc -l)" -eq 1 ] && [ -n "$BDF" ] || { echo "FATAL: U55C BDF '$BDF'"; exit 2; }
echo "bdf=$BDF"
if [ ! -s "$FX/prompt_4096.gdnstate" ] || [ ! -s "$FX/fpga_stream_4097.tokens" ]; then
  echo "[fixture] generating the 4,096-token context on the GPU -> $FX"
  mkdir -p "$FX"
  "$PY" "$REPO/scripts/prepare_steady_decode_fixture.py" --fixture fixtures_full/wikitext.gdnreq --output-dir "$FX" || exit 4
  "$PY" "$REPO/scripts/export_fpga_handoff_state.py" --model "$MODEL" --tokens "$FX/prompt_4096.tokens" \
      --output "$FX/prompt_4096.gdnstate" --metadata "$FX/prompt_4096.state.json" || exit 4
fi
sha256sum "$FX/prompt_4096.gdnstate" "$FX/fpga_stream_4097.tokens" | cut -c1-16,65-
echo "[benchmark] U55C: 60 s loaded idle, 30 s warm-up, 3 x 60 s  $(date -u +%T)Z"
rm -f "$V/stop"
"$PY" "$REPO/scripts/sample_device_power.py" --device fpga --fpga-bdf "$BDF" --period 1.0 \
    --output "$V/fpga_power.jsonl" --stop-file "$V/stop" &
sampler=$!
"$HOST" "$XCL" "$WEIGHTS" fixtures_decode/decode.gdnreq "$V/fpga_timing.json" "$BDF" --decode \
    --decode-from-state "$FX/prompt_4096.gdnstate" --benchmark-tokens "$FX/fpga_stream_4097.tokens" \
    --benchmark-idle-seconds 60 --benchmark-warmup-seconds 30 --benchmark-interval-seconds 60 --benchmark-intervals 3 \
    2>&1 | grep -vE '^\[progress\] (upload|shard)' | tail -8
rc=${PIPESTATUS[0]}; touch "$V/stop"; wait "$sampler"; sampler=""
[ "$rc" -eq 0 ] || { echo "FATAL: FPGA benchmark rc=$rc"; exit 5; }
xbutil examine -d "$BDF" -r platform 2>/dev/null | grep -iA6 "^Clocks" | grep -iE "clk|hbm" | tee "$V/fpga_clocks.txt" | head -3
"$PY" - "$V" <<'PYEOF'
import json, sys, statistics as st
from pathlib import Path
from aggregate_power_eval import summarize_device
v = Path(sys.argv[1])
s = summarize_device("U55C", v / "fpga_timing.json", v / "fpga_power.jsonl")
t = json.loads((v / "fpga_timing.json").read_text())
s["kernel_median_ms_per_interval"] = [iv["kernel"]["median_ms"] for iv in t["intervals"]]
(v / "fpga_summary.json").write_text(json.dumps(s, indent=2) + "\n")
print("FPGA_POWER_SUMMARY tpot_ms=%.4f device_ms=%.4f kernel_median_ms=%s active_W=%.2f idle_W=%.2f gross_J=%.4f idle_sub_J=%.4f steps=%d" % (
    s["production_tpot_ms"], s["device_time_ms"], ",".join("%.4f" % k for k in s["kernel_median_ms_per_interval"]),
    s["active_mean_watts"], s["idle_mean_watts"], s["gross_joules_per_token"], s["idle_subtracted_joules_per_token"], s["steps"]))
PYEOF
for dt in $ARMS; do
  echo "[benchmark] A100 $dt eager: 60 s idle, 30 s warm-up, 3 x 60 s  $(date -u +%T)Z"
  R="$V/gpu_$dt"; mkdir -p "$R"; rm -f "$R/stop"
  export TRITON_CACHE_DIR="/tmp/${USER}-${SLURM_JOB_ID}/triton"; mkdir -p "$TRITON_CACHE_DIR"
  "$PY" "$REPO/scripts/sample_device_power.py" --device gpu --period 1.0 --output "$R/gpu_power.jsonl" --stop-file "$R/stop" &
  sampler=$!
  "$PY" "$REPO/scripts/benchmark_gpu_fp32.py" --model "$MODEL" --prompt-tokens "$FX/prompt_4096.tokens" \
      --teacher-tokens "$FX/teacher_4096.tokens" --output "$R/gpu_timing.json" --dtype "$dt" \
      --idle-seconds 60 --warmup-seconds 30 --interval-seconds 60 --intervals 3
  rc=$?; touch "$R/stop"; wait "$sampler"; sampler=""
  [ "$rc" -eq 0 ] || { echo "FATAL: GPU $dt benchmark rc=$rc"; exit 6; }
  "$PY" "$REPO/scripts/aggregate_power_eval.py" --fpga-timing "$V/fpga_timing.json" --fpga-power "$V/fpga_power.jsonl" \
      --gpu-timing "$R/gpu_timing.json" --gpu-power "$R/gpu_power.jsonl" \
      --output-json "$R/comparison.json" --output-csv "$R/comparison.csv" || exit 6
  python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['comparison']; print('PAIRED_%s speed=%.3fx gross_eff=%.3fx idle_sub_eff=%.3fx' % (sys.argv[2], 1/c['fpga_over_gpu_production_tpot'], c['fpga_over_gpu_gross_energy_efficiency'], c['fpga_over_gpu_idle_subtracted_energy_efficiency']))" "$R/comparison.json" "$dt"
done
echo "POWER_DONE $(date -u +%FT%TZ)"
