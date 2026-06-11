#!/usr/bin/env bash
#
# decode_correctness_check.sh — one-command decode-correctness driver for the
# GatedDeltaNet HLS/native testbench.
#
# Compares the native testbench's decode trajectory (gen_traj / tf_argmax)
# against the CACHED decode golden produced by the GPU reference run
# (c_impl/results_decode_golden/decode.decode.json). Reports decode-correctness
# metrics via scripts/check_gdn_c_parity.py --decode and gates on
# exact_traj_match (non-zero exit on mismatch) so it can drive a git/editor hook.
#
# Modes:
#   --fast        Quick, NO GPU. Rebuilds gdn_eval, runs a TINY decode subset
#                 (default --limit 1 --decode-len 6) and checks it against the
#                 cached golden. Intended for a pre-commit / on-save hook.
#                 NOTE: the native re-prefill forward is ~16 s/step on CPU, so
#                 even this tiny run takes ~1-2 min. That is expected.
#   (default)     Full run: larger subset (default --limit 4 --decode-len 32),
#                 same cached golden, full metric report.
#
# Optional passthrough (override the per-mode defaults):
#   --limit N         number of decode examples to run
#   --decode-len N    number of decode steps per example
#
# This NEVER regenerates the Python/GPU golden — it always compares against the
# cached golden on disk.
#
set -euo pipefail

# --- resolve repo root from this script's location ---------------------------
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_PATH}/.." && pwd)"
C_IMPL="${REPO_ROOT}/c_impl"

WEIGHTS="${C_IMPL}/artifacts/gdn-1.3b-f32.gdnw"
FIXTURE="${C_IMPL}/fixtures_decode/decode.gdnreq"
GOLDEN="${C_IMPL}/results_decode_golden/decode.decode.json"
CHECKER="${REPO_ROOT}/scripts/check_gdn_c_parity.py"
# Disaggregated decode: the FPGA never prefills. State (example 0's post-prompt
# recurrent + conv window) is produced once by the GPU (scripts/export_gdn_state.py)
# and lives on disk (gitignored, like the weight blob). The native decode-only
# path loads it and decodes from the exported seed token — no prefill, no
# re-prefill — and is gated bit-exact against the same cached golden.
STATE="${C_IMPL}/fixtures_decode/decode_ex0.gdnstate"

# --- parse args --------------------------------------------------------------
MODE="full"
LIMIT=""
DECODE_LEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            MODE="fast"
            shift
            ;;
        --limit)
            LIMIT="${2:?--limit needs a value}"
            shift 2
            ;;
        --decode-len)
            DECODE_LEN="${2:?--decode-len needs a value}"
            shift 2
            ;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "usage: $0 [--fast] [--limit N] [--decode-len N]" >&2
            exit 2
            ;;
    esac
done

# Per-mode defaults (only applied if not explicitly overridden).
if [[ "${MODE}" == "fast" ]]; then
    : "${LIMIT:=1}"
    : "${DECODE_LEN:=6}"
    OUT_JSON="${C_IMPL}/results_decode_c/decode_fast.c.json"
else
    : "${LIMIT:=4}"
    : "${DECODE_LEN:=32}"
    OUT_JSON="${C_IMPL}/results_decode_c/decode.c.json"
fi

PYTHON_BIN="${PYTHON:-python3}"

# --- preflight ---------------------------------------------------------------
fail() { echo "FATAL: $*" >&2; exit 1; }

for f in "${WEIGHTS}" "${FIXTURE}" "${GOLDEN}" "${CHECKER}" "${STATE}"; do
    [[ -f "${f}" ]] || fail "required file not found: ${f}"
done
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "python interpreter not found: ${PYTHON_BIN}"

mkdir -p "$(dirname "${OUT_JSON}")"

echo "======================================================================"
echo "GatedDeltaNet decode-correctness check  [mode=${MODE}]"
echo "======================================================================"
echo "  repo root   : ${REPO_ROOT}"
echo "  weights     : ${WEIGHTS}"
echo "  fixture     : ${FIXTURE}"
echo "  golden      : ${GOLDEN}  (cached, NOT regenerated)"
echo "  candidate   : ${OUT_JSON}"
echo "  decode args : --limit ${LIMIT} --decode-len ${DECODE_LEN}"
echo "----------------------------------------------------------------------"

# --- build -------------------------------------------------------------------
echo "==> Building native testbench: make -C ${C_IMPL} gdn_eval"
make -C "${C_IMPL}" gdn_eval

# --- run decode --------------------------------------------------------------
# gdn_eval expects to find its relative default paths from inside c_impl; we use
# an absolute output path so the cwd does not matter, but cd anyway to match the
# documented invocation and to resolve any relative artifacts consistently.
echo "==> Running native decode-only from GPU state (no prefill / no re-prefill)"
echo "    cd ${C_IMPL} && ./gdn_eval <weights> <fixture> ${OUT_JSON} --decode --decode-from-state ${STATE} --decode-len ${DECODE_LEN}"
(
    cd "${C_IMPL}"
    ./gdn_eval \
        "${WEIGHTS}" \
        "${FIXTURE}" \
        "${OUT_JSON}" \
        --decode \
        --decode-from-state "${STATE}" \
        --decode-len "${DECODE_LEN}"
)

# --- check -------------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo "==> Comparing candidate decode trajectory against cached golden"
echo "    ${PYTHON_BIN} ${CHECKER} --decode --golden ${GOLDEN} --c ${OUT_JSON}"
echo

set +e
"${PYTHON_BIN}" "${CHECKER}" --decode --golden "${GOLDEN}" --c "${OUT_JSON}"
CHECK_RC=$?
set -e

echo
echo "======================================================================"
if [[ "${CHECK_RC}" -eq 0 ]]; then
    echo "  DECODE CORRECTNESS: PASS  [mode=${MODE}]"
    echo "======================================================================"
    exit 0
else
    echo "  DECODE CORRECTNESS: FAIL  [mode=${MODE}]  (exact_traj_match mismatch)"
    echo "======================================================================"
    exit 1
fi
