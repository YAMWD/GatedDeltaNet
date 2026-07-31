#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(100|130)$ ]]; then
    echo "usage: $0 <100|130>" >&2
    exit 2
fi

link_frequency_mhz=$1
c_impl_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_file="${c_impl_dir}/gdn_model.cpp"
expected_source_sha256=b0a380365d00a7535dd1256f62f6a21f97a3eee6158e3e4b53bb92ce2df5dafb
hls_pre_tcl="${c_impl_dir}/hls_gdn_forward.tcl"
expected_hls_pre_tcl_sha256=a76930f3332baac50bf7540b58f9242a2efcd235154215b26e95e8da5a057633
expected_xo_sha256=${EXPECTED_XO_SHA256:-}
config="${c_impl_dir}/hw_iter35_recur16_dma_w15_fanout64_postphys_f100.cfg"
expected_config_sha256=240855bd65ba1cf4525c88b34f9d07012bf73c90e75f524b2c9547eaa9e5922b
iter22_hook="${c_impl_dir}/apply_iter22_cluster8_slr1_east.tcl"
expected_iter22_hook_sha256=b0f07d2128589789641c649d8c127949dd06ca72f6f3140154eda5e24c331b1f
iter23_hook="${c_impl_dir}/apply_iter23_dma_fanout.tcl"
expected_iter23_hook_sha256=268f9e9fb8e0e6178317edd88be46811372351251758dc781ba950e84f4ba4d8
iter35_hook="${c_impl_dir}/apply_iter35_dma_w15_fifoaddr_fanout.tcl"
expected_iter35_hook_sha256=d6cd2074b8bd987cf00db922c390c3c92ecd48947108d4d1b54a46e1445263bb
donor_ip_cache="${c_impl_dir}/build.hw.iter35.recur16.dmaw15f64.postphys.f100.o8.v2022_2/.ipcache"
build_dir="${c_impl_dir}/build.hw.iter36.headlocal.iter35.postphys.f${link_frequency_mhz}.o8.v2022_2"
diagnostics_dir="${c_impl_dir}/diagnostics/iter36_headlocal_f${link_frequency_mhz}"
vpp=${VPP:-/tools/Xilinx/Vitis/2022.2/bin/v++}
vitis_settings=${VITIS_SETTINGS:-/tools/Xilinx/Vitis/2022.2/settings64.sh}
platform=${PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}
hls_frequency_mhz=130
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
    echo "iter36 f${link_frequency_mhz}: Vitis settings script not found: ${vitis_settings}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${vitis_settings}"

if [[ ! -x "${vpp}" ]]; then
    echo "iter36 f${link_frequency_mhz}: Vitis v++ not executable: ${vpp}" >&2
    exit 1
fi

verify_sha256() {
    local path=$1
    local expected=$2
    local label=$3
    local actual
    if [[ ! -f "${path}" ]]; then
        echo "iter36 f${link_frequency_mhz}: ${label} is missing: ${path}" >&2
        exit 1
    fi
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ "${actual}" != "${expected}" ]]; then
        echo "iter36 f${link_frequency_mhz}: ${label} hash mismatch: expected ${expected}, got ${actual}" >&2
        exit 1
    fi
}

verify_sha256 "${source_file}" "${expected_source_sha256}" "kernel source"
verify_sha256 "${hls_pre_tcl}" "${expected_hls_pre_tcl_sha256}" "HLS pre-Tcl"
verify_sha256 "${config}" "${expected_config_sha256}" "config"
verify_sha256 "${iter22_hook}" "${expected_iter22_hook_sha256}" "Iter22 hook"
verify_sha256 "${iter23_hook}" "${expected_iter23_hook_sha256}" "Iter23 hook"
verify_sha256 "${iter35_hook}" "${expected_iter35_hook_sha256}" "Iter35 hook"

if [[ -e "${build_dir}" ]]; then
    echo "iter36 f${link_frequency_mhz}: refusing to overwrite build directory: ${build_dir}" >&2
    exit 1
fi

mkdir -p "${build_dir}"
ip_cache_reused=false
if [[ -d "${donor_ip_cache}" ]]; then
    mkdir -p "${build_dir}/.ipcache"
    cp -a --reflink=auto "${donor_ip_cache}/." "${build_dir}/.ipcache/"
    ip_cache_reused=true
fi

{
    printf "iteration=iter36_headlocal_f%s\n" "${link_frequency_mhz}"
    printf "change=head-local recurrent state with HBM restore/read and update/save fusion\n"
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
    printf "iter35_dma_hook_sha256=%s\n" "${expected_iter35_hook_sha256}"
    printf "donor_ip_cache=%s\n" "${donor_ip_cache}"
    printf "ip_cache_reused=%s\n" "${ip_cache_reused}"
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

actual_xo_sha256=$(sha256sum "${build_dir}/gdn_forward.xo" | awk '{print $1}')
if [[ -n "${expected_xo_sha256}" &&
      "${actual_xo_sha256}" != "${expected_xo_sha256}" ]]; then
    echo "iter36 f${link_frequency_mhz}: XO hash mismatch: expected ${expected_xo_sha256}, got ${actual_xo_sha256}" >&2
    exit 1
fi
{
    printf "compile_completed_at=%s\n" "$(date --iso-8601=seconds)"
    printf "xo_sha256=%s\n" "${actual_xo_sha256}"
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

echo "iter36 f${link_frequency_mhz} build complete: ${build_dir}/gdn_forward.xclbin"
