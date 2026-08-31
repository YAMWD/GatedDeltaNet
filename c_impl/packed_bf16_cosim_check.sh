#!/usr/bin/env bash
# Generate and run a production-faithful one-layer/all-eight-head RTL cosim.
# Only layer-dependent storage/interface depths and the layer loop are scaled;
# all 32 MM2S readers, 16 clusters, collectors, recurrent islands, FIFO depths,
# arithmetic, and stream handshakes remain identical to production.
set -euo pipefail

# Vitis HLS 2022.2 reuses an exported DEBUG value as a literal compiler flag
# in its generated co-simulation Makefile.  Some compute-node environments set
# DEBUG=release, which makes g++ interpret "release" as an input filename.
unset DEBUG

c_impl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-${c_impl_dir}/diagnostics/iter64_all_bf16_cosim}"
clock_ns="${2:-6.667}"

mkdir -p "${output_dir}"
if find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "cosim output directory is not empty: ${output_dir}" >&2
    exit 2
fi

require_count() {
    local expected="$1"
    local pattern="$2"
    local path="$3"
    local actual
    actual="$(grep -Fc -- "${pattern}" "${path}" || true)"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "cosim source guard failed: expected ${expected} occurrence(s) of" >&2
        echo "  ${pattern}" >&2
        echo "in ${path}, found ${actual}" >&2
        exit 3
    fi
}

require_count 1 '#define GDN_WSF_STATE   12582912u' "${c_impl_dir}/gdn_model.h"
require_count 1 '#define GDN_WSF_HEADBUF 442368u' "${c_impl_dir}/gdn_model.h"
require_count 1 '#define GDN_COMPILED_WEIGHT_SHARD_BEATS 1366528u' "${c_impl_dir}/gdn_model.h"
require_count 1 '#define GDN_CONV_TAIL_STRIPES (24u * 3u)' "${c_impl_dir}/gdn_model.h"
require_count 1 '#define GDN_LAYERS      24' "${c_impl_dir}/gdn_model.cpp"
require_count 1 'depth=2000000' "${c_impl_dir}/gdn_model.cpp"
require_count 28 'depth=1366528' "${c_impl_dir}/gdn_model.cpp"
require_count 4 'depth=1464832' "${c_impl_dir}/gdn_model.cpp"
require_count 1 'depth=817810' "${c_impl_dir}/gdn_model.cpp"
require_count 4 'depth=4096' "${c_impl_dir}/gdn_model.cpp"

sed \
    -e 's/#define GDN_WSF_STATE   12582912u.*/#define GDN_WSF_STATE   524288u      \/* cosim: 1*8*256*256 *\//' \
    -e 's/#define GDN_WSF_HEADBUF 442368u.*/#define GDN_WSF_HEADBUF 18432u       \/* cosim: 1*3*3*2048 *\//' \
    -e 's/#define GDN_COMPILED_WEIGHT_SHARD_BEATS 1366528u/#define GDN_COMPILED_WEIGHT_SHARD_BEATS 118272u/' \
    -e 's/#define GDN_CONV_TAIL_STRIPES (24u \* 3u)/#define GDN_CONV_TAIL_STRIPES (1u * 3u)/' \
    "${c_impl_dir}/gdn_model.h" > "${output_dir}/gdn_model.h"

sed \
    -e 's/#define GDN_LAYERS      24/#define GDN_LAYERS       1/' \
    -e 's/depth=2000000/depth=63760/' \
    -e 's/depth=1366528/depth=118272/g' \
    -e 's/depth=1464832/depth=122368/g' \
    -e 's/depth=817810/depth=37650/' \
    -e 's/loop_tripcount min=24 max=24/loop_tripcount min=1 max=1/' \
    "${c_impl_dir}/gdn_model.cpp" > "${output_dir}/gdn_model.cpp"

cp "${c_impl_dir}/packed_bf16_one_layer_test.cpp" "${output_dir}/"
cp "${c_impl_dir}/hls_gdn_forward.tcl" "${output_dir}/"
sed -e "s/@CLOCK_NS@/${clock_ns}/g" \
    "${c_impl_dir}/cosim_all_bf16.tcl.in" > "${output_dir}/cosim.tcl"

if [[ "${GDN_COSIM_PREPARE_ONLY:-0}" == 1 ]]; then
    (
        cd "${output_dir}"
        sha256sum gdn_model.cpp gdn_model.h packed_bf16_one_layer_test.cpp \
            hls_gdn_forward.tcl cosim.tcl > source_hashes.txt
    )
    echo "ALL_BF16_COSIM_PREPARED ${output_dir}"
    exit 0
fi

(
    cd "${output_dir}"
    sha256sum gdn_model.cpp gdn_model.h packed_bf16_one_layer_test.cpp \
        hls_gdn_forward.tcl cosim.tcl > source_hashes.txt
    echo "ALL_BF16_COSIM_START $(date --iso-8601=seconds) clock_ns=${clock_ns}"
    set +e
    vitis_hls -f cosim.tcl 2>&1 | tee cosim.out
    status="${PIPESTATUS[0]}"
    set -e
    echo "${status}" > exit_code
    echo "ALL_BF16_COSIM_END $(date --iso-8601=seconds) status=${status}"
    exit "${status}"
)
