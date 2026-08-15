#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(100|115)$ ]]; then
    echo "usage: $0 <100|115>" >&2
    exit 2
fi

link_frequency_mhz=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

source_file=gdn_model.cpp
header_file=gdn_model.h
host_file=host.cpp
eval_file=gdn_eval.cpp
hls_pre_tcl=hls_gdn_forward.tcl
config_template=hw_iter37c_state4_recur32_slr2_f115.cfg
iter22_hook=apply_iter22_cluster8_slr1_east.tcl
iter23_hook=apply_iter23_dma_fanout.tcl
iter35_hook=apply_iter35_dma_w15_fifoaddr_fanout.tcl
iter37c_hook=apply_iter37c_recurrent_slr2.tcl
expected_source_sha256=88e68abdcba29f4355216571440d70d8611f0855717466e8f73891a3db58b216
expected_header_sha256=3896fa2fa95f10dc65481720ce3393803699df7c3985117224134d1fed2e1716
expected_host_sha256=8fd114f8fd5deb679003a6f62a822c7f1c7c8ae3b8450a80f4bc08e02d2e8969
expected_eval_sha256=673ee4edf3cf372e3283c98794372a34f9f57b91e295e9651f06fbab93ab3e66
expected_hls_pre_tcl_sha256=a76930f3332baac50bf7540b58f9242a2efcd235154215b26e95e8da5a057633
expected_config_template_sha256=998b71e3a8cb3b7f818f12cbe6581f0ffd2e04010dba5db3f20ca2ae844aa08f
expected_iter22_hook_sha256=b0f07d2128589789641c649d8c127949dd06ca72f6f3140154eda5e24c331b1f
expected_iter23_hook_sha256=268f9e9fb8e0e6178317edd88be46811372351251758dc781ba950e84f4ba4d8
expected_iter35_hook_sha256=d6cd2074b8bd987cf00db922c390c3c92ecd48947108d4d1b54a46e1445263bb
expected_iter37c_hook_sha256=15587403b5e904345abdee72cd84cfc0fa24f8be559f94f1e5554cdcedbd059e
expected_xo_sha256=5846289626acf27d200a098038ed934432fe16730071c8e185d9e9c5fc766626
donor_ip_cache=build.hw.iter35.recur16.dmaw15f64.postphys.f100.o8.v2022_2/.ipcache
donor_xo_dir=build.hw.iter37b.state4.recur32.iter35.postphys.f115.o8.v2022_2
build_dir=build.hw.iter37d.state4.recur32.slr2.f${link_frequency_mhz}.o8.v2022_2
diagnostics_dir=diagnostics/iter37d_state4_recur32_slr2_f${link_frequency_mhz}
vpp=${VPP:-/tools/Xilinx/Vitis/2022.2/bin/v++}
vitis_settings=${VITIS_SETTINGS:-/tools/Xilinx/Vitis/2022.2/settings64.sh}
platform=${PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}
hls_frequency_mhz=130
jobs=${JOBS:-8}

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

verify_sha256() {
    local path=$1 expected=$2 label=$3 actual
    if [[ ! -f "${path}" ]]; then
        echo "iter37: ${label} is missing: ${path}" >&2
        exit 1
    fi
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ "${actual}" != "${expected}" ]]; then
        echo "iter37: ${label} hash mismatch: expected ${expected}, got ${actual}" >&2
        exit 1
    fi
}

verify_sha256 "${source_file}" "${expected_source_sha256}" "kernel source"
verify_sha256 "${header_file}" "${expected_header_sha256}" "kernel header"
verify_sha256 "${host_file}" "${expected_host_sha256}" "XRT host"
verify_sha256 "${eval_file}" "${expected_eval_sha256}" "native evaluator"
verify_sha256 "${hls_pre_tcl}" "${expected_hls_pre_tcl_sha256}" "HLS pre-Tcl"
verify_sha256 "${config_template}" "${expected_config_template_sha256}" "link config template"
verify_sha256 "${iter22_hook}" "${expected_iter22_hook_sha256}" "Iter22 hook"
verify_sha256 "${iter23_hook}" "${expected_iter23_hook_sha256}" "Iter23 hook"
verify_sha256 "${iter35_hook}" "${expected_iter35_hook_sha256}" "Iter35 hook"
verify_sha256 "${iter37c_hook}" "${expected_iter37c_hook_sha256}" "Iter37D hook"

resolved_config="${diagnostics_dir}/resolved_hw.cfg"
sed "s|@C_IMPL_DIR@|${script_dir}|g" "${config_template}" > "${resolved_config}"
if grep -q '@C_IMPL_DIR@' "${resolved_config}"; then
    echo "iter37d: unresolved path placeholder in ${resolved_config}" >&2
    exit 1
fi
resolved_config=$(realpath "${resolved_config}")

if [[ ! -f "${vitis_settings}" ]]; then
    echo "iter37: Vitis settings script not found: ${vitis_settings}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${vitis_settings}"
if [[ ! -x "${vpp}" ]]; then
    echo "iter37: Vitis v++ not executable: ${vpp}" >&2
    exit 1
fi
if [[ -e "${build_dir}" ]]; then
    echo "iter37: refusing to overwrite build directory: ${build_dir}" >&2
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
    printf "iteration=iter37d_state4_recur32_slr2_f%s\n" "${link_frequency_mhz}"
    printf "change=Iter37B source; recurrent hierarchy assigned to full SLR2; SSI_SpreadSLLs placement; AlternateCLBRouting\n"
    printf "source_sha256=%s\n" "${expected_source_sha256}"
    printf "header_sha256=%s\n" "${expected_header_sha256}"
    printf "host_sha256=%s\n" "${expected_host_sha256}"
    printf "eval_sha256=%s\n" "${expected_eval_sha256}"
    printf "hls_pre_tcl_sha256=%s\n" "${expected_hls_pre_tcl_sha256}"
    printf "config_template_sha256=%s\n" "${expected_config_template_sha256}"
    printf "resolved_config_sha256=%s\n" "$(sha256sum "${resolved_config}" | awk '{print $1}')"
    printf "iter37d_hook_sha256=%s\n" "${expected_iter37c_hook_sha256}"
    printf "hls_frequency_mhz=%s\n" "${hls_frequency_mhz}"
    printf "link_frequency_mhz=%s\n" "${link_frequency_mhz}"
    printf "platform=%s\n" "${platform}"
    printf "ip_cache_reused=%s\n" "${ip_cache_reused}"
    printf "started_at=%s\n" "$(date --iso-8601=seconds)"
} > "${diagnostics_dir}/build.manifest"

xo_reused=false
donor_xo="${donor_xo_dir}/gdn_forward.xo"
donor_csynth_report="${donor_xo_dir}/reports_compile/gdn_forward/hls_reports/gdn_forward_csynth.rpt"
if [[ -s "${donor_xo}" && -s "${donor_csynth_report}" ]] && \
   [[ $(sha256sum "${donor_xo}" | awk '{print $1}') == "${expected_xo_sha256}" ]]; then
    cp --reflink=auto "${donor_xo}" "${build_dir}/gdn_forward.xo"
    csynth_report="${donor_csynth_report}"
    xo_reused=true
else
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
    csynth_report="${build_dir}/reports_compile/gdn_forward/hls_reports/gdn_forward_csynth.rpt"
fi

if [[ ! -s "${csynth_report}" ]]; then
    echo "iter37d: missing integrated csynth report: ${csynth_report}" >&2
    exit 1
fi

recurrent_max=$(awk -F'|' '/grp_gdn_recurrent_attention_fu_/ {
    gsub(/[[:space:]]/, "", $5); print $5; exit
}' "${csynth_report}")
top_min=$(awk -F'|' '/^[[:space:]]*\|[[:space:]]*[0-9]+\|[[:space:]]*[0-9]+\|/ {
    gsub(/[[:space:]]/, "", $2); print $2; exit
}' "${csynth_report}")
resource_line=$(awk -F'|' '/^\|Total[[:space:]]*\|/ {print; exit}' "${csynth_report}")
if [[ -z "${recurrent_max}" || -z "${top_min}" || -z "${resource_line}" ]]; then
    echo "iter37: could not parse csynth acceptance metrics" >&2
    exit 1
fi
read -r bram dsp ff lut uram < <(
    awk -F'|' '/^\|Total[[:space:]]*\|/ {
        for (i=3; i<=7; ++i) gsub(/[[:space:]]/, "", $i);
        print $3, $4, $5, $6, $7; exit
    }' "${csynth_report}"
)

{
    printf "compile_completed_at=%s\n" "$(date --iso-8601=seconds)"
    printf "xo_reused=%s\n" "${xo_reused}"
    printf "xo_sha256=%s\n" "$(sha256sum "${build_dir}/gdn_forward.xo" | awk '{print $1}')"
    printf "csynth_top_min_cycles=%s\n" "${top_min}"
    printf "csynth_recurrent_max_cycles=%s\n" "${recurrent_max}"
    printf "csynth_bram18=%s\ncsynth_dsp=%s\ncsynth_ff=%s\ncsynth_lut=%s\ncsynth_uram=%s\n" \
        "${bram}" "${dsp}" "${ff}" "${lut}" "${uram}"
} >> "${diagnostics_dir}/build.manifest"

# Stop before spending hours in implementation when the intended recurrence
# speedup did not synthesize or the extra arithmetic grew beyond the planned
# resource envelope. These are conservative pre-route gates, not success claims.
if (( recurrent_max > 50000 )); then
    echo "iter37: csynth gate failed: recurrence ${recurrent_max} cycles > 50000" >&2
    exit 1
fi
if (( top_min > 4100000 )); then
    echo "iter37: csynth gate failed: top minimum ${top_min} cycles > 4100000" >&2
    exit 1
fi
if (( dsp > 3700 || uram > 64 )); then
    echo "iter37: csynth resource gate failed: DSP=${dsp}, URAM=${uram}" >&2
    exit 1
fi

(
    cd "${build_dir}"
    "${vpp}" -l -t hw \
        --platform "${platform}" \
        --save-temps \
        --optimize 2 \
        --hls.jobs "${jobs}" \
        --kernel_frequency "${link_frequency_mhz}" \
        --config "${resolved_config}" \
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

echo "iter37d f${link_frequency_mhz} build complete: ${build_dir}/gdn_forward.xclbin"
