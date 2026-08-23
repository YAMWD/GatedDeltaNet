#!/bin/bash
#SBATCH --job-name=gdn_bf16_all
#SBATCH --partition=light
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#
# The missing arm: BF16 weights + BF16 activations + BF16 recurrent state.
#
# The recorded arms each vary one precision against a common reference:
#   Arm A            BF16 weights, BF16 activations, FP32 state
#   Arm B            BF16 weights, FP32 activations, FP32 state
#   state follow-up  BF16 weights, FP32 activations, BF16 state
# Nothing combines them, so the configuration the accelerator would actually
# implement has no data. The state gain (+1.55 on Table 2) came from swamping,
# which BF16 activations also act on, so the two are not assumed to add.
#
# Runs Tables 2, 3 and 5 at full reference sample counts, same runner and same
# converted checkpoint as the earlier arms, so the only new variables are the
# activation dtype and the state patch.

set -uo pipefail

REPO=/home/yaoz0b/GatedDeltaNet
cd "$REPO"

ARM_ROOT=/home/yaoz0b/gdn_precision_eval_20260817/bf16_all
CKPT=/home/yaoz0b/gdn_precision_eval_20260817/checkpoint_bf16_mixed
PY=.micromamba/envs/gdn-hf/bin/python

mkdir -p "$ARM_ROOT"
echo "=== all-BF16 arm ==="
echo "  node   : $(hostname)"
echo "  job    : ${SLURM_JOB_ID:-<none>}"
echo "  start  : $(date --iso-8601=seconds)"
echo "  gpu    : ${SLURM_JOB_GRES:-<unset>}"
echo "  ckpt   : $CKPT"
echo "  out    : $ARM_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  card   : /'

# site-packages is shared: restore the kernel on every exit path, including
# failure, timeout and scancel.
restore() {
    echo "--- restoring FLA kernel (exit path) ---"
    scripts/gdn_bf16_state_patch.sh restore || true
    scripts/gdn_bf16_state_patch.sh status  || true
}
trap restore EXIT INT TERM

echo "--- control: unpatched kernel must show unrounded FP32 state ---"
scripts/gdn_bf16_state_patch.sh status
"$PY" scripts/verify_bf16_state_patch.py --model "$CKPT" \
    --dtype bfloat16 --expect fp32 2>&1 | tee "$ARM_ROOT/verify_control.log"
vrc=${PIPESTATUS[0]}
if [ "$vrc" -ne 0 ]; then
    case "$vrc" in
    1) echo "FATAL: control gate failed -- the state already looks BF16-exact" >&2
       echo "       without the patch, so the comparison has no control." >&2 ;;
    2) echo "FATAL: no recurrent-state tensor found in the cache." >&2 ;;
    *) echo "FATAL: the verifier crashed (exit ${vrc}) -- a tooling fault, not" >&2
       echo "       a measurement. See verify_control.log." >&2 ;;
    esac
    exit 3
fi

echo "--- applying the BF16-state patch ---"
scripts/gdn_bf16_state_patch.sh apply
scripts/gdn_bf16_state_patch.sh status
# Triton caches on kernel source; clear it so the patched kernel is recompiled.
rm -rf "${HOME}/.triton/cache" 2>/dev/null || true

echo "--- gate: patched kernel must show BF16-exact state ---"
"$PY" scripts/verify_bf16_state_patch.py --model "$CKPT" \
    --dtype bfloat16 --expect bf16 2>&1 | tee "$ARM_ROOT/verify_patched.log"
vrc=${PIPESTATUS[0]}
if [ "$vrc" -ne 0 ]; then
    if [ "$vrc" -eq 1 ]; then
        echo "FATAL: the patch did not take effect. Not spending GPU hours on a" >&2
        echo "       run whose headline variable is not actually applied." >&2
    else
        echo "FATAL: the verifier crashed (exit ${vrc}) -- a tooling fault." >&2
    fi
    exit 4
fi

echo "--- running Tables 2/3/5 (DTYPE=bfloat16 = BF16 activations) ---"
MODEL_ID="$CKPT" DTYPE=bfloat16 OUT_ROOT="$ARM_ROOT" \
    scripts/run_gdn_precision_eval_arm.sh 2>&1 | tee "$ARM_ROOT/arm.log"
rc=${PIPESTATUS[0]}

echo "EXIT_CODE=${rc} end=$(date --iso-8601=seconds)"
printf '%s\n' "$rc" > "$ARM_ROOT/arm.exit"
exit "$rc"
