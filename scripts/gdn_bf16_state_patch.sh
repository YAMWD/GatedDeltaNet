#!/bin/bash
#
# Apply / restore the temporary BF16 recurrent-state patch to FLA's
# fused_recurrent gated-delta-rule kernel.
#
#   usage: scripts/gdn_bf16_state_patch.sh apply|restore|status
#
# FLA allocates the recurrent accumulator as tl.float32 regardless of model
# dtype, so BF16 recurrent state cannot be selected -- it has to be patched in.
# This rounds the accumulator to BF16 at the end of every token, modelling
# hardware that computes in FP32 registers and stores the state to BF16 memory.
# The rounding is placed AFTER the output store, so the output is taken from the
# unrounded state, matching a design that rounds only on the store.
#
# The patch is unconditional rather than env-gated on purpose: Triton's JIT
# cache keys on kernel source, so a flag read from the environment can silently
# reuse a kernel compiled without the rounding. Editing the source guarantees a
# recompile. Always restore afterwards -- site-packages is shared.

set -euo pipefail

KERNEL="${GDN_FLA_KERNEL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.micromamba/envs/gdn-hf/lib/python3.11/site-packages/fla/ops/gated_delta_rule/fused_recurrent.py}"
BACKUP="${KERNEL}.gdn_orig"
MARK="GDN_BF16_STATE_PATCH"

action="${1:?usage: $0 apply|restore|status}"

case "$action" in
apply)
    if grep -q "$MARK" "$KERNEL"; then
        echo "already patched: $KERNEL"; exit 0
    fi
    cp -p "$KERNEL" "$BACKUP"
    python3 - "$KERNEL" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); lines = p.read_text().split('\n')
store = [i for i, l in enumerate(lines)
         if 'tl.store(p_o,' in l and 'mask=mask_v' in l]
if len(store) != 1:
    sys.exit(f"expected exactly 1 output-store site, found {len(store)}")
i = store[0]
indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
lines[i+1:i+1] = [
    f"{indent}# GDN_BF16_STATE_PATCH: round the recurrent accumulator to BF16 once",
    f"{indent}# per token, after the output store so the output uses the unrounded",
    f"{indent}# state. Models FP32 compute with BF16 state storage. TEMPORARY.",
    f"{indent}b_h = b_h.to(tl.bfloat16).to(tl.float32)",
]
p.write_text('\n'.join(lines))
print(f"patched after output store at line {i+1}")
PY
    echo "backup: $BACKUP"
    ;;
restore)
    if [ ! -f "$BACKUP" ]; then echo "no backup at $BACKUP" >&2; exit 1; fi
    mv -f "$BACKUP" "$KERNEL"
    if grep -q "$MARK" "$KERNEL"; then echo "RESTORE FAILED: marker still present" >&2; exit 1; fi
    echo "restored: $KERNEL"
    ;;
status)
    if grep -q "$MARK" "$KERNEL"; then echo "PATCHED   $KERNEL"; else echo "clean     $KERNEL"; fi
    sha256sum "$KERNEL" | sed 's/^/  /'
    ;;
*) echo "usage: $0 apply|restore|status" >&2; exit 2 ;;
esac
