#!/bin/bash
#
# Submit the hardware flow to Slurm as two chained jobs.
#
#   usage:  bash run_hw_sbatch.sh [TAG]
#   knobs:  JOBS HLS_FREQ LINK_FREQ BUILD_MEM BUILD_TIME ONCARD_TIME SKIP_ONCARD
#   data:   WEIGHTS DECODE_STATE DECODE_FIXTURE DECODE_GOLDEN LOGITS_REFERENCE
#           exported here and forwarded to make by the on-card job. A packed-BF16
#           kernel needs WEIGHTS=artifacts/gdn-1.3b-bf16w.gdnw -- it rejects the
#           FP32 blob, and only after reading all 5.6 GB of it.
#
# Why two jobs: the `build` QOS grants no FPGA and the `light` QOS grants only
# 8 cores, so a single job cannot both link the image and run it on the card.
# The on-card job is chained with afterok, so it starts only if the build
# succeeds and is cancelled automatically if the build fails.
#
# This replaces the pre-migration single-job script, which ran `make run_hw`
# (build *and* on-card) in the `build` partition, where the card gates could
# never have worked, and pinned to harrier for a filesystem problem that no
# longer exists -- /home is now the same NFS share on the login node and on
# every compute node.

set -euo pipefail

C_IMPL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$C_IMPL"

TAG="${1:-hw_$(date +%Y%m%d_%H%M%S)}"
JOBS="${JOBS:-32}"
HLS_FREQ="${HLS_FREQ:-150}"
LINK_FREQ="${LINK_FREQ:-100}"
BUILD_MEM="${BUILD_MEM:-128G}"
BUILD_TIME="${BUILD_TIME:-2-00:00:00}"
ONCARD_TIME="${ONCARD_TIME:-4:00:00}"

DIAG="diagnostics/${TAG}"
mkdir -p "$DIAG"

echo "tag        : ${TAG}"
echo "knobs      : JOBS=${JOBS} HLS_FREQ=${HLS_FREQ} LINK_FREQ=${LINK_FREQ}"
echo "diagnostics: ${C_IMPL}/${DIAG}"

# Pre-flight: the user is capped at one FPGA across all jobs at once. An idle
# allocation still holds its card, so say so now rather than letting the
# on-card job sit in the queue behind it with an opaque reason.
held="$(squeue -u "$USER" -h -o '%i %j %b %T %M' 2>/dev/null | grep -i 'fpga' || true)"
if [ -n "$held" ]; then
    echo
    echo "note: you already hold a job requesting an FPGA. The on-card job will"
    echo "      queue until it ends -- the per-user cap is 1 FPGA, regardless of"
    echo "      how many cards are free."
    printf '        %s\n' "$held"
fi

sha256sum gdn_model.cpp gdn_model.h host.cpp hw_f150_physical_islands.cfg \
    > "${DIAG}/source_hashes.txt" 2>/dev/null || true

build_id="$(sbatch --parsable \
    --job-name="${TAG}_build" \
    --chdir="$C_IMPL" \
    --mem="$BUILD_MEM" \
    --time="$BUILD_TIME" \
    --output="${C_IMPL}/${DIAG}/build.slurm-%j.log" \
    --export=ALL,TAG="$TAG",JOBS="$JOBS",HLS_FREQ="$HLS_FREQ",LINK_FREQ="$LINK_FREQ" \
    slurm/hw_build.slurm)"
echo
echo "build  job ${build_id}  -> ${DIAG}/build.slurm-${build_id}.log"

if [ "${SKIP_ONCARD:-0}" = "1" ]; then
    echo "on-card    : skipped (SKIP_ONCARD=1)"
else
    oncard_id="$(sbatch --parsable \
        --job-name="${TAG}_oncard" \
        --chdir="$C_IMPL" \
        --time="$ONCARD_TIME" \
        --dependency="afterok:${build_id}" \
        --kill-on-invalid-dep=yes \
        --output="${C_IMPL}/${DIAG}/oncard.slurm-%j.log" \
        --export=ALL,TAG="$TAG",JOBS="$JOBS",HLS_FREQ="$HLS_FREQ",LINK_FREQ="$LINK_FREQ" \
        slurm/hw_oncard.slurm)"
    echo "oncard job ${oncard_id}  -> ${DIAG}/oncard.slurm-${oncard_id}.log   (afterok:${build_id})"
fi

echo
echo "watch  : tail -F ${C_IMPL}/${DIAG}/build.slurm-${build_id}.log"
echo "queue  : squeue -u ${USER}"
