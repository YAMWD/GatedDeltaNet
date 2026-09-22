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
LINK_FREQ="${LINK_FREQ:-150}"
VIVADO_SYNTH_JOBS="${VIVADO_SYNTH_JOBS:-16}"
VIVADO_IMPL_JOBS="${VIVADO_IMPL_JOBS:-8}"
TAG="${TAG:?TAG must be set by the submitter}"
SUBMIT_C_IMPL="${SUBMIT_C_IMPL:?SUBMIT_C_IMPL must be set by the submitter}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:?SNAPSHOT_PATH must be set by the submitter}"

# Shared diagnostics first, so every early exit below still leaves the
# build.exit / build.phase markers the watchers and the on-card chain rely on
# (builds 3961 and 3986 died in the disk preflight with no marker at all).
SHARED_DIAG="${SUBMIT_C_IMPL}/diagnostics/${TAG}"
mkdir -p "${SHARED_DIAG}"
printf '%s\n' "preflight" > "${SHARED_DIAG}/build.phase"
trap 'rc=$?; test -s "${SHARED_DIAG}/build.exit" || printf "%s\n" "$rc" > "${SHARED_DIAG}/build.exit"' EXIT

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
df -h /tmp
available_kb="$(df -Pk /tmp | awk 'NR==2 {print $4}')"
if [ "$available_kb" -lt 62914560 ]; then
    echo "FATAL: less than 60 GiB available in /tmp; refusing a production link"
    exit 2
fi
STAGE_C_IMPL="${STAGE_ROOT}/c_impl"
export GDN_SHARED_LOG_DIR="$SHARED_DIAG"
mkdir -p "${STAGE_C_IMPL}"
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
SHARED_BUILD="${SHARED_DIAG}/artifacts/${BUILD_DIR}"
echo "  build dir  : ${BUILD_DIR}"
printf '%s\n' "${BUILD_DIR}" > "${SHARED_DIAG}/build_dir.path"

# Address-fix candidates must pass native and generated/synthesized address
# checks before linking. The source snapshot contains the exact gate script.
if [ "${STATE_ADDRESS_GATE:-0}" = 1 ]; then
    printf '%s\n' "native_gate" > "${SHARED_DIAG}/build.phase"
    set +e
    (
        set -e
        make -C "${STAGE_C_IMPL}" gdn_eval all_bf16_layout_test
        "${STAGE_C_IMPL}/all_bf16_layout_test"
        # Dataset overrides can be either repository-relative or absolute.
        cd "$SUBMIT_C_IMPL"
        for steps in 6 32; do
            "${STAGE_C_IMPL}/gdn_eval" \
                "${WEIGHTS}" \
                "${DECODE_FIXTURE:-fixtures_decode/decode.gdnreq}" \
                "${SHARED_DIAG}/native${steps}.json" --decode \
                --decode-from-state "${DECODE_STATE}" \
                --decode-len "${steps}"
            python3 "${SHARED_DIAG}/check_gdn_c_parity.py" \
                --decode --golden "${DECODE_GOLDEN}" \
                --c "${SHARED_DIAG}/native${steps}.json"
        done
    ) 2>&1 | tee "${SHARED_DIAG}/native_gate.live.log"
    native_rc="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "$native_rc" > "${SHARED_DIAG}/native_gate.exit"
    if [ "$native_rc" -ne 0 ]; then
        printf '%s\n' "$native_rc" > "${SHARED_DIAG}/build.exit"
        exit "$native_rc"
    fi
fi

# Compile the production XO once, inspect that exact HLS result, and only then
# authorize the long Vivado link.  `make xclbin` reuses the up-to-date XO, so
# this is a gate inside one build rather than a duplicate HLS synthesis.
if [ -n "${REUSE_XO:-}" ]; then
    printf '%s\n' "reuse_xo_identity" > "${SHARED_DIAG}/build.phase"
    python3 "${STAGE_C_IMPL}/prepare_reused_xo.py" "$REUSE_XO" \
        "${REUSE_XO_REFERENCE:?REUSE_XO_REFERENCE required}" \
        "$STAGE_C_IMPL" "$STAGE_BUILD" "$HLS_FREQ" \
        > "${SHARED_DIAG}/xo_reuse.log" 2>&1 || {
        cat "${SHARED_DIAG}/xo_reuse.log"
        echo 2 > "${SHARED_DIAG}/build.exit"
        exit 2
    }
    cp "$STAGE_BUILD/xo_reuse.json" "$SHARED_DIAG/"
fi
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

if [ "$xo_rc" -eq 0 ] && [ "$gate_rc" -eq 0 ] && [ "${STATE_ADDRESS_GATE:-0}" = 1 ]; then
    printf '%s\n' "state_address_gate" > "${SHARED_DIAG}/build.phase"
    set +u
    source "/tools/Xilinx/Vitis/${VITIS_VERSION}/settings64.sh"
    set -u
    set +e
    python3 "${STAGE_C_IMPL}/check_state_writer_addresses.py" \
        "${STAGE_BUILD}/gdn_forward.xo" "${SHARED_DIAG}/state_address_gate" \
        2>&1 | tee "${SHARED_DIAG}/state_address_gate.live.log"
    gate_rc="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "$gate_rc" > "${SHARED_DIAG}/state_address_gate.exit"
fi

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
    # v++ stdout goes directly to NFS. The finishing Tcl independently writes
    # physical_finish.live.log there; Vitis removes pre-created runme symlinks.
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
    while IFS= read -r -d '' checkpoint; do
        mkdir -p "${SHARED_DIAG}/checkpoints"
        cp "$checkpoint" "${SHARED_DIAG}/checkpoints/"
    done < <(find "$STAGE_BUILD" -type f -name 'gdn_f150_*.dcp' -print0)
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
if [ "$rc" -eq 0 ] && [ "${REQUIRE_EXACT_CLOCK:-0}" = 1 ]; then
    # vpl's AUTO-FREQ-SCALING reads the timing report it wrote BEFORE the
    # post-route TCL.POST hook closed the design, so a closed image can still
    # carry a scaled DATA_CLK (build 3987: design at 6.667 ns, metadata 149).
    # Reconcile the metadata with the hook's own gate evidence; the tool patches
    # only when every clock is non-negative at the requested period, never
    # touches the bitstream, and fails closed otherwise.
    set +e
    gate_tsv="$(find "${STAGE_BUILD}" -path '*/gdn_final_qor/exact_clock_gate.tsv' \
        -print -quit 2>/dev/null)"
    python3 "${STAGE_C_IMPL}/reconcile_exact_clock.py" \
        "${STAGE_BUILD}/gdn_forward.xclbin" "${gate_tsv:-/nonexistent}" "$LINK_FREQ" \
        "${SHARED_DIAG}" 2>&1 | tee "${SHARED_DIAG}/exact_clock.log"
    clock_rc="${PIPESTATUS[0]}"
    if [ "$clock_rc" -eq 0 ] && grep -q '^EXACT_CLOCK_RECONCILED' "${SHARED_DIAG}/exact_clock.log"; then
        # The reconciled image replaces the one Make hashed into
        # build_manifest.sha256; rewrite that entry and re-verify the whole
        # manifest so the published digests describe the shipped file. The
        # vpl-written original's digest stays on record beside it.
        (
            cd "${STAGE_C_IMPL}" || exit 1
            manifest="${BUILD_DIR}/build_manifest.sha256"
            new_line="$(sha256sum "${BUILD_DIR}/gdn_forward.xclbin")"
            old_line="$(grep -E " ${BUILD_DIR}/gdn_forward.xclbin\$" "$manifest")"
            printf '%s\n' "$old_line" | sed 's/$/  (vpl-written metadata, before reconciliation)/' \
                > "${BUILD_DIR}/build_manifest.pre_reconcile.txt"
            grep -vE " ${BUILD_DIR}/gdn_forward.xclbin\$" "$manifest" > "${manifest}.tmp"
            printf '%s\n' "$new_line" >> "${manifest}.tmp"
            mv "${manifest}.tmp" "$manifest"
            sha256sum -c --quiet "$manifest"
        ) 2>&1 | tee -a "${SHARED_DIAG}/exact_clock.log"
        clock_rc="${PIPESTATUS[0]}"
        [ "$clock_rc" -eq 0 ] && echo "BUILD_MANIFEST_UPDATED for the reconciled image" | tee -a "${SHARED_DIAG}/exact_clock.log"
    fi
    set -e
    printf '%s\n' "$clock_rc" > "${SHARED_DIAG}/exact_clock.exit"
    if [ "$clock_rc" -ne 0 ]; then rc=$clock_rc; fi
fi
if [ "${rc}" -eq 0 ]; then
    mkdir -p "${SHARED_BUILD}"
    cp -f "${STAGE_BUILD}/gdn_forward.xclbin" \
          "${SHARED_BUILD}/gdn_forward.xclbin"
    for manifest in xo_manifest.sha256 build_manifest.sha256 build_manifest.pre_reconcile.txt resolved_hw.cfg; do
        if [ -f "${STAGE_BUILD}/${manifest}" ]; then
            cp -f "${STAGE_BUILD}/${manifest}" "${SHARED_BUILD}/${manifest}"
        fi
    done
    cp -f "${STAGE_C_IMPL}/host.exe" "${SHARED_BUILD}/host.exe"
    cp "${STAGE_C_IMPL}/Makefile" "${SHARED_DIAG}/reproduction.Makefile"
    printf '%s\n' "$SHARED_BUILD/gdn_forward.xclbin" > "$SHARED_DIAG/xclbin.path"
    printf '%s\n' "$SHARED_BUILD/host.exe" > "$SHARED_DIAG/host.path"
    sha256sum "${SHARED_BUILD}/gdn_forward.xclbin" \
              "${SHARED_BUILD}/host.exe" > \
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
