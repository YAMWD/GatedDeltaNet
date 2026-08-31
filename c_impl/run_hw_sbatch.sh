#!/bin/bash
#
# Submit the hardware flow to Slurm as two chained jobs.
#
#   usage:  bash run_hw_sbatch.sh [TAG]
#   knobs:  JOBS HLS_FREQ LINK_FREQ VIVADO_SYNTH_JOBS VIVADO_IMPL_JOBS
#           BUILD_CONSTRAINT BUILD_EXCLUDE BUILD_NODE (optional exceptional pin)
#           BUILD_MEM BUILD_TIME ONCARD_TIME SKIP_ONCARD
#   data:   WEIGHTS DECODE_STATE DECODE_FIXTURE DECODE_GOLDEN LOGITS_REFERENCE
#           GPU_LOGITS_REFERENCE
#           exported here and forwarded to make by the on-card job. A packed-BF16
#           kernel needs WEIGHTS=artifacts/gdn-1.3b-bf16w.gdnw -- it rejects the
#           FP32 blob, and only after reading all 5.6 GB of it.
#
# Why two jobs: the `build` QOS grants no FPGA and the `light` QOS grants only
# 8 cores, so a single job cannot both link the image and run it on the card.
# The on-card job is chained with afterok, so it starts only if the build
# succeeds and is cancelled automatically if the build fails.
#
# The compute-only build requests the required Vivado feature and lets Slurm
# select any eligible node. BUILD_NODE is empty by default and exists only for
# an explicit, measured node-specific requirement. The build is staged on
# node-local NVMe. The source archive freezes dirty-but-intended working-tree
# changes at submission time, so a queued build cannot silently pick up later
# edits from shared /home.

set -euo pipefail

C_IMPL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$C_IMPL"

TAG="${1:-hw_$(date +%Y%m%d_%H%M%S)}"
JOBS="${JOBS:-48}"
HLS_FREQ="${HLS_FREQ:-150}"
LINK_FREQ="${LINK_FREQ:-100}"
VIVADO_SYNTH_JOBS="${VIVADO_SYNTH_JOBS:-16}"
VIVADO_IMPL_JOBS="${VIVADO_IMPL_JOBS:-8}"
BUILD_CONSTRAINT="${BUILD_CONSTRAINT:-vivado2024.2}"
# Measured platform gaps: acclnode04 lacks the U55C platform and XRT;
# acclnode05 advertises vivado2024.2 but lacks the U55C platform (probes
# 2026-08-22 / 2026-08-28; re-confirmed by 2-second preflight FATALs on jobs
# 2255, 2448/2450, and 2452/2454). BUILD_EXCLUDE= (empty) re-probes after a
# cluster change.
BUILD_EXCLUDE="${BUILD_EXCLUDE-acclnode04,acclnode05}"
BUILD_NODE="${BUILD_NODE:-}"
BUILD_MEM="${BUILD_MEM:-192G}"
BUILD_TIME="${BUILD_TIME:-2-00:00:00}"
ONCARD_TIME="${ONCARD_TIME:-4:00:00}"
WEIGHTS="${WEIGHTS:-artifacts/gdn-1.3b-bf16w.gdnw}"
DECODE_STATE="${DECODE_STATE:-fixtures_decode/decode_ex0_native_bf16_product.gdnstate}"
DECODE_GOLDEN="${DECODE_GOLDEN:-results_decode_golden/decode_native_bf16_product.decode.json}"
# Diagnostic only, off by default: hardware/native bit-exactness is not
# achievable (Iter66m) and is not a gate. Set LOGITS_REFERENCE=<file> to
# re-enable the comparison when localizing an arithmetic change.
LOGITS_REFERENCE="${LOGITS_REFERENCE:-}"
GPU_LOGITS_REFERENCE="${GPU_LOGITS_REFERENCE:-artifacts/decode_native_bf16_product_64.gdnlog}"
export VIVADO_SYNTH_JOBS VIVADO_IMPL_JOBS WEIGHTS DECODE_STATE DECODE_GOLDEN
export LOGITS_REFERENCE GPU_LOGITS_REFERENCE

DIAG="diagnostics/${TAG}"
mkdir -p "$DIAG"
printf '%s\n' "PENDING: Slurm build has not started; live tool output will replace this line after allocation." > "${DIAG}/build.live.log"

echo "tag        : ${TAG}"
echo "knobs      : JOBS=${JOBS} HLS_FREQ=${HLS_FREQ} LINK_FREQ=${LINK_FREQ}"
echo "vivado     : SYNTH_JOBS=${VIVADO_SYNTH_JOBS} IMPL_JOBS=${VIVADO_IMPL_JOBS}"
echo "constraint : ${BUILD_CONSTRAINT:-<none>}"
echo "exclude    : ${BUILD_EXCLUDE:-<none>}"
echo "build node : ${BUILD_NODE:-scheduler-selected}"
echo "weights    : ${WEIGHTS}"
echo "state      : ${DECODE_STATE}"
echo "golden     : ${DECODE_GOLDEN}"
echo "native ref : ${LOGITS_REFERENCE:-<disabled: diagnostic only>}"
echo "GPU ref    : ${GPU_LOGITS_REFERENCE}"
echo "diagnostics: ${C_IMPL}/${DIAG}"

if [ "${JOBS}" -gt 48 ]; then
    echo "FATAL: JOBS=${JOBS} exceeds the 48-CPU build allocation" >&2
    exit 2
fi
# LOGITS_REFERENCE is intentionally excluded from the required set: it is a
# diagnostic and defaults to empty. It is still checked when explicitly set.
for required in "${WEIGHTS}" "${DECODE_STATE}" "${DECODE_GOLDEN}" \
                "${GPU_LOGITS_REFERENCE}"; do
    if [ ! -s "${required}" ]; then
        echo "FATAL: required all-BF16 validation artifact missing: ${required}" >&2
        exit 2
    fi
done
if [ -n "${LOGITS_REFERENCE}" ] && [ ! -s "${LOGITS_REFERENCE}" ]; then
    echo "FATAL: LOGITS_REFERENCE set but missing: ${LOGITS_REFERENCE}" >&2
    exit 2
fi

# Pre-flight: the user is capped at one FPGA across all jobs at once. An idle
# allocation still holds its card, so say so now rather than letting the
# on-card job sit in the queue behind it with an opaque reason.
held="$(/opt/slurm/current/bin/squeue -u "$USER" -h \
    -o '%i %j %b %T %M' 2>/dev/null | grep -i 'fpga' || true)"
if [ -n "$held" ]; then
    echo
    echo "note: you already hold a job requesting an FPGA. The on-card job will"
    echo "      queue until it ends -- the per-user cap is 1 FPGA, regardless of"
    echo "      how many cards are free."
    printf '        %s\n' "$held"
fi

snapshot_files=(
    Makefile gdn_model.cpp gdn_model.h host.cpp hls_gdn_forward.tcl
    hw_f150_physical_islands.cfg apply_f150_physical_islands.tcl
    apply_iter54_dma_timing.tcl apply_iter35_dma_w15_fifoaddr_fanout.tcl
    apply_iter23_dma_fanout.tcl check_f150_physical_islands.tcl
    report_final_qor.tcl check_native_bf16_xo.py
)
# A variant build (HW_CFG_TEMPLATE=... and/or extra hook Tcl) must freeze its
# extra inputs too: EXTRA_SNAPSHOT_FILES="fileA fileB" appends to the list.
if [ -n "${EXTRA_SNAPSHOT_FILES:-}" ]; then
    for extra in ${EXTRA_SNAPSHOT_FILES}; do
        snapshot_files+=("${extra}")
    done
fi
echo "cfg tmpl   : ${HW_CFG_TEMPLATE:-hw_f150_physical_islands.cfg (default)}"
for snapshot_file in "${snapshot_files[@]}"; do
    test -s "${snapshot_file}" || {
        echo "FATAL: source/config file missing: ${snapshot_file}" >&2
        exit 2
    }
done
SNAPSHOT_PATH="${C_IMPL}/${DIAG}/source_snapshot.tar"
tar -cf "${SNAPSHOT_PATH}" "${snapshot_files[@]}"
sha256sum "${snapshot_files[@]}" > "${DIAG}/source_hashes.txt"
sha256sum "${SNAPSHOT_PATH}" > "${DIAG}/source_snapshot.sha256"
export SNAPSHOT_PATH

build_placement_args=()
if [ -n "${BUILD_CONSTRAINT}" ]; then
    build_placement_args+=(--constraint="${BUILD_CONSTRAINT}")
fi
if [ -n "${BUILD_EXCLUDE}" ]; then
    build_placement_args+=(--exclude="${BUILD_EXCLUDE}")
fi
if [ -n "${BUILD_NODE}" ]; then
    build_placement_args+=(--nodelist="${BUILD_NODE}")
fi

build_id="$(/opt/slurm/current/bin/sbatch --parsable \
    --job-name="${TAG}_build" \
    --chdir="$C_IMPL" \
    "${build_placement_args[@]}" \
    --cpus-per-task=48 \
    --mem="$BUILD_MEM" \
    --time="$BUILD_TIME" \
    --output="${C_IMPL}/${DIAG}/build.slurm-%j.log" \
    --export=ALL,TAG="$TAG",JOBS="$JOBS",HLS_FREQ="$HLS_FREQ",LINK_FREQ="$LINK_FREQ",SUBMIT_C_IMPL="$C_IMPL",SNAPSHOT_PATH="$SNAPSHOT_PATH" \
    slurm/hw_build.slurm)"
echo
echo "build  job ${build_id}  -> ${DIAG}/build.slurm-${build_id}.log"
echo "detail log              -> ${DIAG}/build.live.log"

if [ "${SKIP_ONCARD:-0}" = "1" ]; then
    echo "on-card    : skipped (SKIP_ONCARD=1)"
else
    oncard_id="$(/opt/slurm/current/bin/sbatch --parsable \
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
echo "watch  : tail -F ${C_IMPL}/${DIAG}/build.live.log"
echo "queue  : squeue -u ${USER}"
