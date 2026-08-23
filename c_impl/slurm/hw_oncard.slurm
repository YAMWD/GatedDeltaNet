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

set -uo pipefail

JOBS="${JOBS:-32}"
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

# Under ConstrainDevices=yes only the allocated card is visible, so the XRT
# index is normally 0 even though the node holds several cards. Verify that
# rather than trusting it: a wrong index fails with a bare err = -22.
examine="$(xbutil examine 2>/dev/null)"
mapfile -t bdfs < <(printf '%s\n' "$examine" \
    | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | sort -u)
echo "  cards visible: ${#bdfs[@]}  ${bdfs[*]:-none}"
if [ "${#bdfs[@]}" -ne 1 ]; then
    echo "FATAL: expected exactly 1 visible card inside this allocation," >&2
    echo "       found ${#bdfs[@]}. Not guessing a device index." >&2
    printf '%s\n' "$examine" | head -30 >&2
    exit 2
fi
if ! printf '%s\n' "$examine" | grep -qi 'u55c'; then
    echo "FATAL: the visible card is not a U55C. Refusing to load this image." >&2
    printf '%s\n' "$examine" | head -30 >&2
    exit 2
fi
HW_DEVICE=0

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
# host has read the whole 5.6 GB file. Pass WEIGHTS=artifacts/gdn-1.3b-bf16w.gdnw
# for a BF16 image; the FP32 default is correct for the FP32 production kernel.
mk_args=(JOBS="$JOBS" HLS_FREQ="$HLS_FREQ" LINK_FREQ="$LINK_FREQ" HW_DEVICE="$HW_DEVICE")
for knob in WEIGHTS DECODE_STATE DECODE_FIXTURE DECODE_GOLDEN LOGITS_REFERENCE; do
    if [ -n "${!knob:-}" ]; then
        mk_args+=("${knob}=${!knob}")
        echo "  ${knob} = ${!knob}"
    fi
done

# -o marks the image and host as up to date, so make runs only the gates and
# can never start a 32-core link inside this 8-core, 32 GB allocation.
make -o "$XCLBIN" -o host.exe run_hw "${mk_args[@]}"
rc=$?

echo "EXIT_CODE=${rc} end=$(date --iso-8601=seconds)"
printf '%s\n' "$rc" > "diagnostics/${TAG}/oncard.exit"
exit "$rc"
