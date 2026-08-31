#!/bin/bash
#SBATCH --job-name=gdn_oncard
#SBATCH --partition=light
#SBATCH --gres=fpga:u55c:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#
# On-card half of the hardware flow: runs the exact 8-token and 64-token
# decode gates against an image built by slurm/hw_build.slurm.
#
# `light` is the only partition granted an FPGA. The card exists only inside
# this job -- never ssh to the node to reach it, which bypasses the scheduler
# and can open a card allocated to someone else (the device nodes are 0666).
#
# The user is capped at ONE FPGA across all jobs at once, so this job queues
# behind any other allocation of the user's that holds a card, even an idle
# one, and even when other U55Cs sit free.

# pipefail only for now: XRT's setup.sh is not set -u clean, and sourcing it
# with stderr discarded under set -u kills the shell with no message at all --
# job 2503 failed in two seconds exactly this way (same lesson as hw_build
# job 1338: source the toolchain BEFORE enabling strict shell options).
set -o pipefail

JOBS="${JOBS:-48}"
HLS_FREQ="${HLS_FREQ:-150}"
LINK_FREQ="${LINK_FREQ:-100}"
TAG="${TAG:?TAG must be set by the submitter}"

echo "=== gdn on-card validation ==="
echo "  node    : $(hostname)"
echo "  job     : ${SLURM_JOB_ID:-<none>}"
echo "  start   : $(date --iso-8601=seconds)"
echo "  gres    : ${SLURM_JOB_GRES:-<unset>}"

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || {
    echo "FATAL: cannot source XRT setup" >&2; exit 2; }
set -u

# Under ConstrainDevices=yes only the allocated card is visible, so the XRT
# index is normally 0 even though the node holds several cards. Verify that
# rather than trusting it: a wrong index fails with a bare err = -22.
# Some nodes expose cards beyond the allocation to `xbutil examine` --
# acclnode01 lists an unallocated U280 beside the allocated U55C (on-card
# jobs 1354 and 2504), so counting all visible cards is the wrong guard.
# Require exactly one U55C and select it explicitly by BDF; host.exe accepts
# a BDF in place of an enumeration index (the qualification-proven pattern
# from job 1355).
examine="$(xbutil examine 2>/dev/null)"
mapfile -t all_bdfs < <(printf '%s\n' "$examine" \
    | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | sort -u)
mapfile -t u55c_bdfs < <(printf '%s\n' "$examine" \
    | grep -i 'xilinx_u55c' \
    | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' \
    | tr -d '[]' | sort -u)
echo "  cards visible: ${#all_bdfs[@]}  ${all_bdfs[*]:-none}"
echo "  u55c cards   : ${#u55c_bdfs[@]}  ${u55c_bdfs[*]:-none}"
if [ "${#u55c_bdfs[@]}" -ne 1 ]; then
    echo "FATAL: expected exactly 1 U55C in this allocation, found ${#u55c_bdfs[@]}." >&2
    printf '%s\n' "$examine" | head -30 >&2
    exit 2
fi
HW_DEVICE="${u55c_bdfs[0]}"

# JOBS only names the build directory here -- nothing is compiled in this job,
# which has 8 cores. It MUST match the build job's JOBS or make would look in
# a directory that does not exist (or worse, a stale one from another run).
BUILD_DIR="$(cat "diagnostics/${TAG}/build_dir.path" 2>/dev/null \
             || make -s --eval='__p:;@echo $(BUILD_DIR)' __p JOBS="$JOBS" \
                  HLS_FREQ="$HLS_FREQ" LINK_FREQ="$LINK_FREQ" 2>/dev/null | tail -1)"
XCLBIN="${BUILD_DIR}/gdn_forward.xclbin"
echo "  image   : ${XCLBIN}"
test -s "$XCLBIN" || { echo "FATAL: no image at ${XCLBIN}" >&2; exit 2; }
test -x ./host.exe || { echo "FATAL: host.exe missing" >&2; exit 2; }

# Data knobs are forwarded explicitly rather than left to the Makefile default.
# They matter: a kernel built for packed BF16 weights rejects the FP32 blob
# outright (`weight payload is not BF16-exact`), and it does so only after the
# host has read the whole 5.6 GB file. This branch defaults to
# WEIGHTS=artifacts/gdn-1.3b-bf16w.gdnw; an override is still forwarded here.
mk_args=(JOBS="$JOBS" HLS_FREQ="$HLS_FREQ" LINK_FREQ="$LINK_FREQ" HW_DEVICE="$HW_DEVICE")
for knob in WEIGHTS DECODE_STATE DECODE_FIXTURE DECODE_GOLDEN \
            LOGITS_REFERENCE GPU_LOGITS_REFERENCE; do
    # Forward when SET, even if empty: an explicitly empty knob must override
    # the Makefile default. LOGITS_REFERENCE now defaults to empty because the
    # hardware/native comparison is a diagnostic, not a gate (Iter66m); set it
    # to a GDNLOG1 path to re-enable the comparison for arithmetic debugging.
    if [ "${!knob+set}" = "set" ]; then
        mk_args+=("${knob}=${!knob}")
        echo "  ${knob} = '${!knob}'"
    fi
done

# -o marks the image and host as up to date, so make runs only the gates and
# can never start a 32-core link inside this 8-core, 32 GB allocation.
make -o "$XCLBIN" -o host.exe run_hw "${mk_args[@]}"
rc=$?

echo "EXIT_CODE=${rc} end=$(date --iso-8601=seconds)"
printf '%s\n' "$rc" > "diagnostics/${TAG}/oncard.exit"
exit "$rc"
