#!/bin/bash
#SBATCH --partition=light
#SBATCH --gres=fpga:u55c:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Qualification, part 1 (Iter88: chained by run_hw_sbatch.sh after the card gates):
#   1. 512-token free-running decode against the GPU 512-step reference (QUAL_GPU512),
#      windowed drift trend -> BOUNDED verdict (logit_drift_trend.py).
#   2. Teacher-forced WikiText-2, all 62 documents, against the GPU score (QUAL_WIKITEXT_GPU),
#      5 % word-perplexity gate on the verified-identical workload.
# The protocol is the Iter66n/Iter66o one, unchanged since, so the numbers compare across images.
# Env from the launcher: TAG SUBMIT_C_IMPL QUAL_GPU512 QUAL_WIKITEXT_GPU QUAL_WIKITEXT_FIXTURE WEIGHTS
# DECODE_FIXTURE DECODE_STATE.  Writes diagnostics/<TAG>/qual/robustness/ and robustness.exit ("rc1 rc2 rc3").
set -o pipefail
TAG="${TAG:?TAG must be set by the submitter}"
C="${SUBMIT_C_IMPL:?SUBMIT_C_IMPL must be set by the submitter}"
cd "$C" || exit 2
DIAG="$C/diagnostics/$TAG"
V="$DIAG/qual/robustness"; mkdir -p "$V"
MARK="$V/robustness.exit"; rm -f "$MARK"
PY="${GDN_PY:-$C/../.micromamba/envs/gdn-hf/bin/python}"; [ -x "$PY" ] || PY=python3
GPU512="${QUAL_GPU512:-diagnostics/iter66n_long_horizon/gpu512.gdnlog}"
GPUPPL="${QUAL_WIKITEXT_GPU:-diagnostics/iter66o_wikitext/wikitext_gpu_reference.json}"
WIKI="${QUAL_WIKITEXT_FIXTURE:-fixtures_full/wikitext.gdnreq}"
WEIGHTS="${WEIGHTS:-artifacts/gdn-1.3b-bf16w.gdnw}"
DECODE_FIXTURE="${DECODE_FIXTURE:-fixtures_decode/decode.gdnreq}"
DECODE_STATE="${DECODE_STATE:-fixtures_decode/decode_ex0_native_bf16_product.gdnstate}"
echo "node=$(hostname -s) job=$SLURM_JOB_ID tag=$TAG start=$(date -u +%FT%TZ)"
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || { echo FATAL xrt; echo "2 2 2" > "$MARK"; exit 2; }
set -u
XCL="$(cat "$DIAG/xclbin.path")"
for f in "$XCL" "$GPU512" "$GPUPPL" "$WIKI" "$WEIGHTS" "$DECODE_FIXTURE" "$DECODE_STATE"; do
  test -s "$f" || { echo "FATAL missing $f"; echo "2 2 2" > "$MARK"; exit 2; }
done
grep -E 'gdn_p\.xclbin$|gdn_forward\.xclbin$' "$DIAG/copied_artifacts.sha256" | sha256sum -c - || { echo "2 2 2" > "$MARK"; exit 2; }
HOST="$V/host.$(hostname -s).exe"
bash "$DIAG/build_host_on_node.sh" "$DIAG" "$HOST" || { echo "2 2 2" > "$MARK"; exit 3; }
BDF="$(xbutil examine 2>/dev/null | grep -i xilinx_u55c | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | tr -d '[]' | sort -u)"
[ "$(printf '%s\n' "$BDF" | wc -l)" -eq 1 ] && [ -n "$BDF" ] || { echo "FATAL: U55C BDF '$BDF'"; echo "2 2 2" > "$MARK"; exit 2; }
echo "bdf=$BDF"; xbutil examine -d "$BDF" -r platform 2>/dev/null | grep -iA6 "^Clocks" | grep -iE "clk|hbm" | head -3

echo "===== 1. 512-token decode vs GPU reference  $(date -u +%T)Z"
"$HOST" "$XCL" "$WEIGHTS" "$DECODE_FIXTURE" "$V/card512.json" "$BDF" \
    --decode --decode-from-state "$DECODE_STATE" --decode-len 512 \
    --gpu-logits-reference "$GPU512" --logits-dump "$V/card512.gdnlog" 2>&1 \
    | grep -vE '^\[progress\] (upload|shard)' | tail -20
rc1=${PIPESTATUS[0]}
echo "card512 rc=$rc1 (1 = post-fork GPU comparison failed, expected for a free-running 512 run)"
verdict="MISSING"
if [ -s "$V/card512.gdnlog" ]; then
  "$PY" "$C/../scripts/logit_drift_trend.py" "$GPU512" "$V/card512.gdnlog" --window 64 2>&1 | tee "$V/card512_trend.txt" | tail -8
  verdict="$(grep -oE 'TREND_VERDICT=[A-Z]+' "$V/card512_trend.txt" | head -1 | cut -d= -f2)"
fi
echo "drift verdict: ${verdict:-MISSING}"

echo "===== 2. WikiText-2 teacher-forced, all documents  $(date -u +%T)Z"
"$HOST" "$XCL" "$WEIGHTS" "$WIKI" "$V/wikitext_full62.json" "$BDF" --score --score-doc-limit 0 2>&1 \
    | stdbuf -oL grep --line-buffered -vE '^\[progress\] (upload|shard)' | tail -30
rc2=${PIPESTATUS[0]}
echo "wikitext rc=$rc2  $(date -u +%T)Z"
rc3=9
if [ "$rc2" -eq 0 ]; then
python3 - "$V/wikitext_full62.json" "$GPUPPL" "$V/wikitext_full62_vs_gpu.json" 0.05 <<'PYEOF'
import json, math, sys
from pathlib import Path
fp, gp, op = map(Path, sys.argv[1:4]); limit = float(sys.argv[4])
f = json.loads(fp.read_text()); g = json.loads(gp.read_text())
ident = {k: {"fpga": f.get(k), "gpu": g.get(k), "match": f.get(k) == g.get(k)}
         for k in ("documents", "windows", "scored_tokens", "words", "bytes")}
same = all(v["match"] for v in ident.values())
fpp, gpp = float(f["word_perplexity"]), float(g["word_perplexity"])
rel = (fpp - gpp) / gpp
ok = same and math.isfinite(rel) and abs(rel) <= limit
r = {"pass": ok, "same_workload": same, "identity": ident, "fpga_word_perplexity": fpp,
     "gpu_word_perplexity": gpp, "relative_delta_percent": 100 * rel,
     "fpga_byte_perplexity": f.get("byte_perplexity"), "gpu_byte_perplexity": g.get("byte_perplexity"),
     "fpga_kernel_ms_per_token": f.get("average_kernel_ms")}
op.write_text(json.dumps(r, indent=2) + "\n"); print(json.dumps(r, indent=2))
raise SystemExit(0 if ok else 3)
PYEOF
rc3=$?
fi
echo "ppl-gate rc=$rc3"
echo "$rc1 $rc2 $rc3 $verdict" > "$MARK"
echo "ROBUSTNESS_SUMMARY drift=$verdict wikitext_rc=$rc2 ppl_gate_rc=$rc3 end=$(date -u +%FT%TZ)"
[ "$rc2" -eq 0 ] && [ "$rc3" -eq 0 ] && [ "$verdict" = "BOUNDED" ]
