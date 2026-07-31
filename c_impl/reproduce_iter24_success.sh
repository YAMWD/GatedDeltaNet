#!/usr/bin/env bash
set -euo pipefail

c_impl_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir="${c_impl_dir}/build.hw.gdn32x2p4auxsharec8s1eastdmaf64v2.f130.o8.v2022_2"
vpp=${VPP:-/tools/Xilinx/Vitis/2022.2/bin/v++}
vitis_settings=${VITIS_SETTINGS:-/tools/Xilinx/Vitis/2022.2/settings64.sh}
platform=${PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}
frequency_mhz=130
jobs=8

check_sha256() {
    local expected=$1
    local path=$2
    local actual

    if [[ ! -f "${path}" ]]; then
        echo "iter24 reproduction: missing input: ${path}" >&2
        exit 1
    fi
    actual=$(sha256sum "${path}" | awk '{print $1}')
    if [[ "${actual}" != "${expected}" ]]; then
        echo "iter24 reproduction: input checksum mismatch: ${path}" >&2
        echo "  expected: ${expected}" >&2
        echo "  actual:   ${actual}" >&2
        exit 1
    fi
}

# These are the retained source and config-only implementation inputs for the
# correct, routable iter24b design lineage. Refuse to label a build as the
# iter24 reproduction if any reviewed input has drifted.
check_sha256 69952f839a2bced016bb79eda16bbade4e09e176779cbac7a3229badb8c3949e \
    "${c_impl_dir}/gdn_model.cpp"
check_sha256 d4f3e16b097798a79b5ef7771d813e652884d3702e2ba8a4b55de318099e0c30 \
    "${c_impl_dir}/gdn_model.h"
check_sha256 a76930f3332baac50bf7540b58f9242a2efcd235154215b26e95e8da5a057633 \
    "${c_impl_dir}/hls_gdn_forward.tcl"
check_sha256 e701b5211df4c2443f25fde6bb656bf4cb839c3da4fe32aa5a11bf45e01d05b4 \
    "${c_impl_dir}/hw_iter23_cluster8_slr1_east_dmaf64.cfg"
check_sha256 b0f07d2128589789641c649d8c127949dd06ca72f6f3140154eda5e24c331b1f \
    "${c_impl_dir}/apply_iter22_cluster8_slr1_east.tcl"
check_sha256 268f9e9fb8e0e6178317edd88be46811372351251758dc781ba950e84f4ba4d8 \
    "${c_impl_dir}/apply_iter23_dma_fanout.tcl"

if [[ ! -f "${vitis_settings}" ]]; then
    echo "iter24 reproduction: Vitis settings script not found: ${vitis_settings}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${vitis_settings}"

if [[ ! -x "${vpp}" ]]; then
    echo "iter24 reproduction: Vitis v++ not found or not executable: ${vpp}" >&2
    exit 1
fi
vpp_version=$("${vpp}" --version 2>&1 || true)
if [[ "${vpp_version}" != *"v++ v2022.2"* ]]; then
    echo "iter24 reproduction: expected Vitis v++ 2022.2: ${vpp}" >&2
    exit 1
fi
if [[ -e "${build_dir}" ]]; then
    echo "iter24 reproduction: refusing to overwrite existing build directory:" >&2
    echo "  ${build_dir}" >&2
    echo "Remove that directory before starting a clean reproduction." >&2
    exit 1
fi

mkdir -p "${build_dir}"

{
    printf "iteration=iter24b_dmaf64v2_f130\n"
    printf "historical_result=correct routable XCLBIN; requested 130 MHz; DATA_CLK 109 MHz\n"
    printf "historical_xo_sha256=0b4454bfca064627d5e929ebd91721bd989082b16dd1760bae051ded5965cf73\n"
    printf "historical_xclbin_sha256=3b58e7b4272b0d268fa06d74485d4c72e0a578be3eb2a6b01badc170b672fcdc\n"
    printf "vpp=%s\n" "${vpp}"
    printf "platform=%s\n" "${platform}"
    printf "frequency_mhz=%s\n" "${frequency_mhz}"
    printf "jobs=%s\n" "${jobs}"
    printf "config=%s\n" "${c_impl_dir}/hw_iter23_cluster8_slr1_east_dmaf64.cfg"
    printf "hls_pre_tcl=%s\n" "${c_impl_dir}/hls_gdn_forward.tcl"
    printf "implementation_inputs=config_and_tcl_only\n"
    printf "started_at=%s\n" "$(date --iso-8601=seconds)"
} > "${build_dir}/reproduction_manifest.txt"

"${vpp}" -c -t hw \
    --platform "${platform}" \
    --save-temps \
    --optimize 2 \
    --hls.jobs "${jobs}" \
    --kernel_frequency "${frequency_mhz}" \
    --temp_dir "${build_dir}/_x_compile" \
    --report_dir "${build_dir}/reports_compile" \
    --hls.pre_tcl "${c_impl_dir}/hls_gdn_forward.tcl" \
    -k gdn_forward \
    -o "${build_dir}/gdn_forward.xo" \
    "${c_impl_dir}/gdn_model.cpp" \
    2>&1 | tee "${build_dir}/gdn_forward_compile.log"

(
    cd "${build_dir}"
    "${vpp}" -l -t hw \
        --platform "${platform}" \
        --save-temps \
        --optimize 2 \
        --hls.jobs "${jobs}" \
        --kernel_frequency "${frequency_mhz}" \
        --config "${c_impl_dir}/hw_iter23_cluster8_slr1_east_dmaf64.cfg" \
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
    printf "xo_sha256=%s\n" "$(sha256sum "${build_dir}/gdn_forward.xo" | awk '{print $1}')"
    printf "xclbin_sha256=%s\n" "$(sha256sum "${build_dir}/gdn_forward.xclbin" | awk '{print $1}')"
} >> "${build_dir}/reproduction_manifest.txt"

echo "iter24 reproduction complete: ${build_dir}/gdn_forward.xclbin"
