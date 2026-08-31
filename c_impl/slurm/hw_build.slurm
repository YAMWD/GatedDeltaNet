#!/bin/bash
#SBATCH --job-name=gdn_build
#SBATCH --partition=build
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#
# Compute-only half of the production flow. The submitter freezes the exact
# source/config set into a tar archive. This job expands that archive onto
# the allocated node's local NVMe, builds there, and copies reproducible outputs
# and diagnostic evidence back to the shared repository. It never opens a
# card; validation is a separate afterok job in slurm/hw_oncard.slurm.

set -uo pipefail

JOBS="${JOBS:-48}"
HLS_FREQ="${HLS_FREQ:-150}"
LINK_FREQ="${LINK_FREQ:-100}"
VIVADO_SYNTH_JOBS="${VIVADO_SYNTH_JOBS:-16}"
VIVADO_IMPL_JOBS="${VIVADO_IMPL_JOBS:-8}"
TAG="${TAG:?TAG must be set by the submitter}"
SUBMIT_C_IMPL="${SUBMIT_C_IMPL:?SUBMIT_C_IMPL must be set by the submitter}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:?SNAPSHOT_PATH must be set by the submitter}"

if [ "${SLURM_CPUS_PER_TASK:-0}" -lt 48 ] || [ "${JOBS}" -gt 48 ]; then
    echo "FATAL: production build requires 48 allocated CPUs and JOBS<=48" >&2
    exit 2
fi
if [ "$(hostname -s)" = "acclhead1" ]; then
    echo "FATAL: production build cannot run on acclhead1" >&2
    exit 2
fi
# One version knob, forwarded to make so this gate and the Makefile's
# XILINX_VITIS can never disagree.
VITIS_VERSION="${VITIS_VERSION:-2024.2}"
test -r "/tools/Xilinx/Vitis/${VITIS_VERSION}/settings64.sh" || {
    echo "FATAL: Vitis ${VITIS_VERSION} is unavailable on $(hostname -s)" >&2
    exit 2
}
test -d /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1 || {
    echo "FATAL: U55C platform is unavailable on $(hostname -s)" >&2
    exit 2
}
test -s "${SNAPSHOT_PATH}" || {
    echo "FATAL: source snapshot missing: ${SNAPSHOT_PATH}" >&2
    exit 2
}

STAGE_ROOT="/tmp/${USER}-${SLURM_JOB_ID}"
STAGE_C_IMPL="${STAGE_ROOT}/c_impl"
SHARED_DIAG="${SUBMIT_C_IMPL}/diagnostics/${TAG}"
mkdir -p "${STAGE_C_IMPL}" "${SHARED_DIAG}"
tar -xf "${SNAPSHOT_PATH}" -C "${STAGE_C_IMPL}"

echo "=== gdn hardware build ==="
echo "  node       : $(hostname)"
echo "  job        : ${SLURM_JOB_ID:-<none>}"
echo "  start      : $(date --iso-8601=seconds)"
echo "  cpus       : ${SLURM_CPUS_PER_TASK:-?}   JOBS=${JOBS}"
echo "  memory     : ${SLURM_MEM_PER_NODE:-?} MiB"
echo "  vivado     : synth=${VIVADO_SYNTH_JOBS} impl=${VIVADO_IMPL_JOBS}"
echo "  freqs      : HLS=${HLS_FREQ} LINK=${LINK_FREQ}"
echo "  stage      : ${STAGE_C_IMPL}"
echo "  snapshot   : ${SNAPSHOT_PATH}"
printf '%s\n' "${STAGE_C_IMPL}" > "${SHARED_DIAG}/stage_dir.path"
sha256sum "${SNAPSHOT_PATH}" > "${SHARED_DIAG}/source_snapshot.sha256"

BUILD_DIR="$(make -s -C "${STAGE_C_IMPL}" \
    --eval='__p:;@echo $(BUILD_DIR)' __p JOBS="${JOBS}" \
    HLS_FREQ="${HLS_FREQ}" LINK_FREQ="${LINK_FREQ}" 2>/dev/null | tail -1)"
STAGE_BUILD="${STAGE_C_IMPL}/${BUILD_DIR}"
SHARED_BUILD="${SUBMIT_C_IMPL}/${BUILD_DIR}"
echo "  build dir  : ${BUILD_DIR}"
printf '%s\n' "${BUILD_DIR}" > "${SHARED_DIAG}/build_dir.path"

# Compile the production XO once, inspect that exact HLS result, and only then
# authorize the long Vivado link.  `make xclbin` reuses the up-to-date XO, so
# this is a gate inside one build rather than a duplicate HLS synthesis.
printf '%s\n' "xo" > "${SHARED_DIAG}/build.phase"
echo "=== STEP 2: full-model XO / integrated HLS ===" \
    | tee "${SHARED_DIAG}/build.live.log"
set +e
make -C "${STAGE_C_IMPL}" xo \
    JOBS="${JOBS}" HLS_FREQ="${HLS_FREQ}" LINK_FREQ="${LINK_FREQ}" \
    VITIS_VERSION="${VITIS_VERSION}" \
    VIVADO_SYNTH_JOBS="${VIVADO_SYNTH_JOBS}" \
    VIVADO_IMPL_JOBS="${VIVADO_IMPL_JOBS}" 2>&1 \
    | tee -a "${SHARED_DIAG}/build.live.log"
xo_rc="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${xo_rc}" > "${SHARED_DIAG}/xo.exit"

gate_rc=125
if [ "${xo_rc}" -eq 0 ]; then
    printf '%s\n' "xo_gate" > "${SHARED_DIAG}/build.phase"
    echo "=== STEP 2 GATE: architecture / II / timing / local resources ===" \
        | tee -a "${SHARED_DIAG}/build.live.log"
    set +e
    python3 "${STAGE_C_IMPL}/check_native_bf16_xo.py" "${STAGE_BUILD}" \
        --json-out "${SHARED_DIAG}/xo_gate_summary.json" 2>&1 \
        | tee "${SHARED_DIAG}/xo_gate.log" \
        | tee -a "${SHARED_DIAG}/build.live.log"
    gate_rc="${PIPESTATUS[0]}"
    set -e
fi
printf '%s\n' "${gate_rc}" > "${SHARED_DIAG}/xo_gate.exit"

if [ "${xo_rc}" -ne 0 ]; then
    rc="${xo_rc}"
    echo "STEP_2_XO_FAIL rc=${rc}; Vivado link was not started" \
        | tee -a "${SHARED_DIAG}/build.live.log"
elif [ "${gate_rc}" -ne 0 ]; then
    rc="${gate_rc}"
    echo "STEP_2_GATE_FAIL rc=${rc}; Vivado link was not started" \
        | tee -a "${SHARED_DIAG}/build.live.log"
else
    printf '%s\n' "vivado_link" > "${SHARED_DIAG}/build.phase"
    echo "=== STEP 3: full Vivado implementation / XCLBIN ===" \
        | tee -a "${SHARED_DIAG}/build.live.log"
    set +e
    make -C "${STAGE_C_IMPL}" xclbin host \
        JOBS="${JOBS}" HLS_FREQ="${HLS_FREQ}" LINK_FREQ="${LINK_FREQ}" \
        VITIS_VERSION="${VITIS_VERSION}" \
        VIVADO_SYNTH_JOBS="${VIVADO_SYNTH_JOBS}" \
        VIVADO_IMPL_JOBS="${VIVADO_IMPL_JOBS}" 2>&1 \
        | tee -a "${SHARED_DIAG}/build.live.log"
    rc="${PIPESTATUS[0]}"
    set -e
fi

# Preserve readable reports regardless of success. The archive retains path
# context for every log/report, while hot files are copied separately for
# convenient diagnosis after the job exits.
if [ -d "${STAGE_BUILD}" ]; then
    (
        cd "${STAGE_C_IMPL}"
        find "${BUILD_DIR}" -type f \
            \( -name '*.log' -o -name '*.rpt' -o -name '*.jou' \
               -o -name '*.xml' -o -name '*.pb' -o -name '*.tsv' \) \
            -print0 | tar --null -czf \
                "${SHARED_DIAG}/build_diagnostics.tar.gz" --files-from=-
    ) || true

    impl_log="$(find "${STAGE_BUILD}" -path '*/impl_1/runme.log' \
        -type f -print -quit 2>/dev/null)"
    if [ -n "${impl_log}" ]; then
        cp -f "${impl_log}" "${SHARED_DIAG}/impl_1.runme.log"
    fi
    final_qor="$(find "${STAGE_BUILD}" -type d -name gdn_final_qor \
        -print -quit 2>/dev/null)"
    if [ -n "${final_qor}" ]; then
        rm -rf "${SHARED_DIAG}/gdn_final_qor"
        cp -a "${final_qor}" "${SHARED_DIAG}/gdn_final_qor"
    fi
    for report_dir in reports reports_compile; do
        if [ -d "${STAGE_BUILD}/${report_dir}" ]; then
            rm -rf "${SHARED_DIAG}/${report_dir}"
            cp -a "${STAGE_BUILD}/${report_dir}" "${SHARED_DIAG}/${report_dir}"
        fi
    done
    for build_file in gdn_forward_compile.log gdn_forward_link.log \
                      xo_manifest.sha256 build_manifest.sha256 resolved_hw.cfg; do
        if [ -f "${STAGE_BUILD}/${build_file}" ]; then
            cp -f "${STAGE_BUILD}/${build_file}" "${SHARED_DIAG}/${build_file}"
        fi
    done
fi

# The post-place Tcl lives beside the staged source rather than under the
# generated build tree.  Preserve its reports and checkpoint independently so
# a later diagnosis or route continuation does not depend on node-local /tmp.
if [ -d "${STAGE_C_IMPL}/diagnostics/placement_reports" ]; then
    rm -rf "${SHARED_DIAG}/placement_reports"
    cp -a "${STAGE_C_IMPL}/diagnostics/placement_reports" \
          "${SHARED_DIAG}/placement_reports"
fi
if [ -s "${STAGE_C_IMPL}/diagnostics/checkpoints/post_place.dcp" ]; then
    cp -f "${STAGE_C_IMPL}/diagnostics/checkpoints/post_place.dcp" \
          "${SHARED_DIAG}/post_place.dcp"
    sha256sum "${SHARED_DIAG}/post_place.dcp" > \
              "${SHARED_DIAG}/post_place.dcp.sha256"
fi

if [ "${rc}" -eq 0 ]; then
    test -s "${STAGE_BUILD}/gdn_forward.xclbin" || rc=3
    test -x "${STAGE_C_IMPL}/host.exe" || rc=3
fi
if [ "${rc}" -eq 0 ]; then
    mkdir -p "${SHARED_BUILD}"
    cp -f "${STAGE_BUILD}/gdn_forward.xclbin" \
          "${SHARED_BUILD}/gdn_forward.xclbin"
    for manifest in xo_manifest.sha256 build_manifest.sha256 resolved_hw.cfg; do
        if [ -f "${STAGE_BUILD}/${manifest}" ]; then
            cp -f "${STAGE_BUILD}/${manifest}" "${SHARED_BUILD}/${manifest}"
        fi
    done
    cp -f "${STAGE_C_IMPL}/host.exe" "${SUBMIT_C_IMPL}/host.exe"
    sha256sum "${SHARED_BUILD}/gdn_forward.xclbin" \
              "${SUBMIT_C_IMPL}/host.exe" > \
              "${SHARED_DIAG}/copied_artifacts.sha256"
    ls -lh "${SHARED_BUILD}/gdn_forward.xclbin" | sed 's/^/  image: /'
fi

echo "EXIT_CODE=${rc} end=$(date --iso-8601=seconds)"
printf '%s\n' "${rc}" > "${SHARED_DIAG}/build.exit"
# On failure, keep the phase marker naming the step that died (xo / xo_gate /
# vivado_link) instead of clobbering it with "complete".
if [ "${rc}" -eq 0 ]; then
    printf '%s\n' "complete" > "${SHARED_DIAG}/build.phase"
fi
exit "${rc}"
