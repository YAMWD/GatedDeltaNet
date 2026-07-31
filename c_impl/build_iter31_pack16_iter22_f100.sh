#!/usr/bin/env bash
set -euo pipefail

c_impl_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_file="${c_impl_dir}/gdn_model.cpp"
expected_source_sha256=e570ccf623abb0801d1aca5652f132fb39a2e97f5eec2c083fd16671b20c257a
hls_pre_tcl="${c_impl_dir}/hls_gdn_forward.tcl"
expected_hls_pre_tcl_sha256=a76930f3332baac50bf7540b58f9242a2efcd235154215b26e95e8da5a057633
expected_xo_sha256=8421d5dea9ddfe0920f66013d6be44acab9bad6e8d4ee4a41a2a5d3fa9fe0823
config="${c_impl_dir}/hw_iter31_pack16_iter22_postphys_f100.cfg"
expected_config_sha256=3fd8fa3ae0c088d12557d635b0821ca81653828bedcee61783a6881ca95c0316
iter22_hook="${c_impl_dir}/apply_iter22_cluster8_slr1_east.tcl"
expected_iter22_hook_sha256=b0f07d2128589789641c649d8c127949dd06ca72f6f3140154eda5e24c331b1f
iter23_hook="${c_impl_dir}/apply_iter23_dma_fanout.tcl"
expected_iter23_hook_sha256=268f9e9fb8e0e6178317edd88be46811372351251758dc781ba950e84f4ba4d8
build_dir="${c_impl_dir}/build.hw.iter31.pack16state.iter22.postphys.f100.o8.v2022_2"
diagnostics_dir="${c_impl_dir}/diagnostics/iter31_pack16state_iter22_postphys_f100"
vpp=${VPP:-/tools/Xilinx/Vitis/2022.2/bin/v++}
vitis_settings=${VITIS_SETTINGS:-/tools/Xilinx/Vitis/2022.2/settings64.sh}
platform=${PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}
hls_frequency_mhz=130
link_frequency_mhz=100
jobs=8

mkdir -p "${diagnostics_dir}"
printf "%s\n" "$$" > "${diagnostics_dir}/build.pid"
on_exit() {
    local status=$?
    trap - EXIT
    printf "%s\n" "${status}" > "${diagnostics_dir}/build.exit"
    printf "wrapper_exited_at=%s\nwrapper_exit=%s\n" \
        "$(date --iso-8601=seconds)" "${status}" \
        >> "${diagnostics_dir}/build.manifest"
    exit "${status}"
}
trap on_exit EXIT

if [[ ! -f "${vitis_settings}" ]]; then
    echo "iter31: Vitis settings script not found: ${vitis_settings}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${vitis_settings}"

if [[ ! -x "${vpp}" ]]; then
    echo "iter31: Vitis v++ not found or not executable: ${vpp}" >&2
    exit 1
fi

verify_sha256() {
    local path=$1
    local expected=$2
    local label=$3
    local actual
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ "${actual}" != "${expected}" ]]; then
        echo "iter31: ${label} hash mismatch: expected ${expected}, got ${actual}" >&2
        exit 1
    fi
}

verify_sha256 "${source_file}" "${expected_source_sha256}" "kernel source"
verify_sha256 "${hls_pre_tcl}" "${expected_hls_pre_tcl_sha256}" "HLS pre-Tcl"
verify_sha256 "${config}" "${expected_config_sha256}" "config"
verify_sha256 "${iter22_hook}" "${expected_iter22_hook_sha256}" "iter22 hook"
verify_sha256 "${iter23_hook}" "${expected_iter23_hook_sha256}" "iter23 hook"

if [[ -e "${build_dir}" ]]; then
    echo "iter31: refusing to overwrite existing build directory: ${build_dir}" >&2
    exit 1
fi

mkdir -p "${build_dir}"

{
    printf "iteration=iter31_pack16_iter22_postphys_f100\n"
    printf "change=revert iter30 cluster9 move; restore iter22 route-complete topology; target 100 MHz; enable post-route AggressiveExplore\n"
    printf "source=%s\n" "${source_file}"
    printf "source_sha256=%s\n" "${expected_source_sha256}"
    printf "hls_pre_tcl=%s\n" "${hls_pre_tcl}"
    printf "hls_pre_tcl_sha256=%s\n" "${expected_hls_pre_tcl_sha256}"
    printf "hls_frequency_mhz=%s\n" "${hls_frequency_mhz}"
    printf "link_frequency_mhz=%s\n" "${link_frequency_mhz}"
    printf "platform=%s\n" "${platform}"
    printf "config=%s\n" "${config}"
    printf "config_sha256=%s\n" "${expected_config_sha256}"
    printf "iter22_floorplan_sha256=%s\n" "${expected_iter22_hook_sha256}"
    printf "iter23_dma_hook_sha256=%s\n" "${expected_iter23_hook_sha256}"
    printf "started_at=%s\n" "$(date --iso-8601=seconds)"
} > "${diagnostics_dir}/build.manifest"

"${vpp}" -c -t hw \
    --platform "${platform}" \
    --save-temps \
    --optimize 2 \
    --hls.jobs "${jobs}" \
    --kernel_frequency "${hls_frequency_mhz}" \
    --temp_dir "${build_dir}/_x_compile" \
    --report_dir "${build_dir}/reports_compile" \
    --hls.pre_tcl "${hls_pre_tcl}" \
    -k gdn_forward \
    -o "${build_dir}/gdn_forward.xo" \
    "${source_file}" \
    2>&1 | tee "${build_dir}/gdn_forward_compile.log"

verify_sha256 "${build_dir}/gdn_forward.xo" "${expected_xo_sha256}" \
    "compiled XO"
{
    printf "compile_completed_at=%s\n" "$(date --iso-8601=seconds)"
    printf "xo_sha256=%s\n" "${expected_xo_sha256}"
} >> "${diagnostics_dir}/build.manifest"

(
    cd "${build_dir}"
    "${vpp}" -l -t hw \
        --platform "${platform}" \
        --save-temps \
        --optimize 2 \
        --hls.jobs "${jobs}" \
        --kernel_frequency "${link_frequency_mhz}" \
        --config "${config}" \
        --vivado.synth.jobs "${jobs}" \
        --vivado.impl.jobs "${jobs}" \
        --temp_dir _x_temp/ \
        --report_dir reports/ \
        -o gdn_forward.xclbin \
        gdn_forward.xo \
        2>&1 | tee gdn_forward_link.log
)

test -s "${build_dir}/gdn_forward.xclbin"
{
    printf "completed_at=%s\n" "$(date --iso-8601=seconds)"
    printf "xclbin_sha256=%s\n" \
        "$(sha256sum "${build_dir}/gdn_forward.xclbin" | awk '{print $1}')"
} >> "${diagnostics_dir}/build.manifest"

echo "iter31 build complete: ${build_dir}/gdn_forward.xclbin"
