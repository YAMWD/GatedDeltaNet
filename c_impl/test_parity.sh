#!/usr/bin/env bash
# Run C gdn_eval on all smoke fixtures and compare against Python golden results.
#
# Usage:
#   cd c_impl && bash test_parity.sh
#
# Prerequisites:
#   - artifacts/gdn-1.3b-f32.gdnw must exist
#   - fixtures_smoke/ must contain .gdnreq files
#   - results_smoke_python/ must contain matching .json golden results
#
# The script rebuilds gdn_eval, runs every smoke fixture through it,
# then calls check_gdn_c_parity.py to diff the C results against the
# Python golden results.  It fails (exit 1) if any per-example logprob
# diff exceeds the tolerance or if predictions disagree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS="$SCRIPT_DIR/artifacts/gdn-1.3b-f32.gdnw"
FIXTURES_DIR="$SCRIPT_DIR/fixtures_smoke"
PYTHON_DIR="$SCRIPT_DIR/results_smoke_python"
PARITY_SCRIPT="$ROOT_DIR/scripts/check_gdn_c_parity.py"

# Tolerance for logprob / score diffs (absolute).
# Observed diffs are ~1e-5; 1e-3 gives comfortable headroom.
LOGPROB_TOL="1e-3"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
for f in "$WEIGHTS" "$PARITY_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: required file not found: $f" >&2
        exit 1
    fi
done

if [[ ! -d "$FIXTURES_DIR" ]]; then
    echo "FATAL: fixtures directory not found: $FIXTURES_DIR" >&2
    exit 1
fi

if [[ ! -d "$PYTHON_DIR" ]]; then
    echo "FATAL: python golden results not found: $PYTHON_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "==> Building gdn_eval"
make -C "$SCRIPT_DIR" clean
make -C "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Run C evaluations
# ---------------------------------------------------------------------------
C_OUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gdn_parity_c.XXXXXX")
trap 'rm -rf "$C_OUT_DIR"' EXIT

num_fixtures=0
for req in "$FIXTURES_DIR"/*.gdnreq; do
    task="$(basename "$req" .gdnreq)"
    # Only run tasks that have a Python golden result
    if [[ ! -f "$PYTHON_DIR/${task}.json" ]]; then
        echo "  skip $task (no python golden)"
        continue
    fi
    echo "==> Running C eval: $task"
    "$SCRIPT_DIR/gdn_eval" "$WEIGHTS" "$req" "$C_OUT_DIR/${task}.json"
    num_fixtures=$((num_fixtures + 1))
done

if [[ "$num_fixtures" -eq 0 ]]; then
    echo "FATAL: no fixtures were evaluated" >&2
    exit 1
fi

echo "==> Ran $num_fixtures fixture(s)"

# ---------------------------------------------------------------------------
# Compute parity report
# ---------------------------------------------------------------------------
PARITY_JSON="$C_OUT_DIR/parity.json"
python3 "$PARITY_SCRIPT" \
    --python-dir "$PYTHON_DIR" \
    --c-dir "$C_OUT_DIR" \
    --output "$PARITY_JSON"

echo "==> Parity report:"
cat "$PARITY_JSON"
echo

# ---------------------------------------------------------------------------
# Check tolerances
# ---------------------------------------------------------------------------
FAIL=0

python3 - "$PARITY_JSON" "$LOGPROB_TOL" <<'PYEOF'
import json, sys

parity_path = sys.argv[1]
tol = float(sys.argv[2])
report = json.loads(open(parity_path).read())

fail = False
for task in report["tasks"]:
    name = task["task"]
    kind = task["kind"]

    # Check prediction agreement (MC tasks)
    if kind == 1:
        if task.get("pred_mismatch", 0) != 0:
            print(f"FAIL {name}: pred_mismatch = {task['pred_mismatch']}")
            fail = True
        if task.get("pred_norm_mismatch", 0) != 0:
            print(f"FAIL {name}: pred_norm_mismatch = {task['pred_norm_mismatch']}")
            fail = True
        for key in ("max_score_diff", "max_score_norm_diff"):
            val = task.get(key, 0.0)
            if val > tol:
                print(f"FAIL {name}: {key} = {val:.6e} > {tol:.0e}")
                fail = True
            else:
                print(f"  OK {name}: {key} = {val:.6e}")

    # Check loglikelihood tasks
    if kind == 2:
        if task.get("greedy_mismatch", 0) != 0:
            print(f"FAIL {name}: greedy_mismatch = {task['greedy_mismatch']}")
            fail = True
        val = task.get("max_logprob_diff", 0.0)
        if val > tol:
            print(f"FAIL {name}: max_logprob_diff = {val:.6e} > {tol:.0e}")
            fail = True
        else:
            print(f"  OK {name}: max_logprob_diff = {val:.6e}")

    # Check rolling perplexity tasks
    if kind == 3:
        val = task.get("max_logprob_diff", 0.0)
        if val > tol:
            print(f"FAIL {name}: max_logprob_diff = {val:.6e} > {tol:.0e}")
            fail = True
        else:
            print(f"  OK {name}: max_logprob_diff = {val:.6e}")

if fail:
    print("\nPARITY CHECK FAILED")
    sys.exit(1)
else:
    print("\nPARITY CHECK PASSED")
    sys.exit(0)
PYEOF

exit $?
