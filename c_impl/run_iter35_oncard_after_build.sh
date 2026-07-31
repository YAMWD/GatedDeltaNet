#!/usr/bin/env bash
set -euo pipefail

c_impl_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${c_impl_dir}/.." && pwd)
diagnostics_dir="${c_impl_dir}/diagnostics/iter35_recur16_dma_w15_fanout64_f100"
runtime_dir="${diagnostics_dir}/on_card"
profile_dir="${runtime_dir}/profile_single"
build_dir="${c_impl_dir}/build.hw.iter35.recur16.dmaw15f64.postphys.f100.o8.v2022_2"
build_pid_file="${diagnostics_dir}/build.pid"
build_exit_file="${diagnostics_dir}/build.exit"
xclbin="${build_dir}/gdn_forward.xclbin"
host="${c_impl_dir}/host.exe"
weights="${c_impl_dir}/artifacts/gdn-1.3b-f32.gdnw"
fixture="${c_impl_dir}/fixtures_decode/decode.gdnreq"
state="${c_impl_dir}/fixtures_decode/decode_ex0.gdnstate"
golden="${c_impl_dir}/results_decode_golden/decode.decode.json"
device_index=0

mkdir -p "${runtime_dir}"
printf "%s\n" "$$" > "${diagnostics_dir}/oncard_monitor.pid"
on_exit() {
    local status=$?
    trap - EXIT
    printf "%s\n" "${status}" > "${diagnostics_dir}/oncard_monitor.exit"
    printf "monitor_exited_at=%s\nmonitor_exit=%s\n" \
        "$(date --iso-8601=seconds)" "${status}" \
        >> "${runtime_dir}/monitor.manifest"
    exit "${status}"
}
trap on_exit EXIT

if [[ ! -f "${build_pid_file}" ]]; then
    echo "iter35 on-card monitor: missing build PID: ${build_pid_file}" >&2
    exit 1
fi
build_pid=$(<"${build_pid_file}")
if [[ ! "${build_pid}" =~ ^[0-9]+$ ]]; then
    echo "iter35 on-card monitor: invalid build PID: ${build_pid}" >&2
    exit 1
fi

{
    printf "monitor_started_at=%s\n" "$(date --iso-8601=seconds)"
    printf "build_pid=%s\n" "${build_pid}"
    printf "device_index=%s\n" "${device_index}"
    printf "smoke_decode_len=8\n"
    printf "performance_decode_len=64\n"
    printf "profile_decode_len=2\n"
} > "${runtime_dir}/monitor.manifest"

while kill -0 "${build_pid}" 2>/dev/null; do
    sleep 30
done

for _ in 1 2 3 4 5; do
    [[ -f "${build_exit_file}" ]] && break
    sleep 1
done
if [[ ! -f "${build_exit_file}" ]]; then
    echo "iter35 on-card monitor: build ended without an exit marker" >&2
    exit 1
fi
build_exit=$(<"${build_exit_file}")
printf "build_exit=%s\n" "${build_exit}" >> "${runtime_dir}/monitor.manifest"
if [[ "${build_exit}" != "0" ]]; then
    echo "iter35 on-card monitor: hardware build failed; run skipped" \
        | tee "${runtime_dir}/monitor.skipped"
    exit 0
fi
if [[ ! -s "${xclbin}" ]]; then
    echo "iter35 on-card monitor: successful build has no XCLBIN" >&2
    exit 1
fi

make -C "${c_impl_dir}" host
{
    printf "oncard_started_at=%s\n" "$(date --iso-8601=seconds)"
    printf "xclbin_sha256=%s\n" "$(sha256sum "${xclbin}" | awk '{print $1}')"
} >> "${runtime_dir}/monitor.manifest"

run_decode() {
    local decode_len=$1
    local stem=$2
    local output="${runtime_dir}/${stem}.json"
    local host_log="${runtime_dir}/${stem}.log"
    local parity_log="${runtime_dir}/${stem}_parity.log"

    stdbuf -oL -eL \
        "${host}" \
        "${xclbin}" \
        "${weights}" \
        "${fixture}" \
        "${output}" \
        "${device_index}" \
        1 \
        --decode \
        --decode-from-state "${state}" \
        --decode-len "${decode_len}" \
        2>&1 | tee "${host_log}"

    python3 "${repo_dir}/scripts/check_gdn_c_parity.py" \
        --decode \
        --golden "${golden}" \
        --c "${output}" \
        2>&1 | tee "${parity_log}"
}

run_decode 8 oncard_iter35_smoke8
run_decode 64 oncard_iter35_decode64

jq '
    def median:
        sort as $s
        | ($s | length) as $n
        | if $n % 2 == 1 then
              $s[($n / 2 | floor)]
          else
              (($s[$n / 2 - 1] + $s[$n / 2]) / 2)
          end;
    [.examples[0].kernel_ms[] | select(. > 0)] as $runs
    | {
        decode_len,
        measured_runs: ($runs | length),
        min_kernel_ms: ($runs | min),
        max_kernel_ms: ($runs | max),
        median_kernel_ms: ($runs | median),
        mean_kernel_ms: ($runs | add / length),
        speedup_vs_iter32_98_6595983ms:
            (98.6595983 / ($runs | add / length)),
        speedup_vs_8port_121_4ms:
            (121.4 / ($runs | add / length))
      }
' "${runtime_dir}/oncard_iter35_decode64.json" \
    | tee "${runtime_dir}/performance_summary.json"

# Capture a separate low-overhead XRT profile after the unprofiled performance
# run, so counter collection cannot bias the 64-token latency measurement.
# This production image has no AXI monitor IP; the summary still records the
# exact kernel invocation and explicitly exposes whether counters are present.
mkdir -p "${profile_dir}"
cp "${c_impl_dir}/xrt.ini" "${profile_dir}/xrt.ini"
(
    cd "${profile_dir}"
    stdbuf -oL -eL \
        "${host}" \
        "${xclbin}" \
        "${weights}" \
        "${fixture}" \
        oncard_iter35_profile2.json \
        "${device_index}" \
        1 \
        --decode \
        --decode-from-state "${state}" \
        --decode-len 2 \
        2>&1 | tee oncard_iter35_profile2.log
)
python3 "${repo_dir}/scripts/check_gdn_c_parity.py" \
    --decode \
    --golden "${golden}" \
    --c "${profile_dir}/oncard_iter35_profile2.json" \
    2>&1 | tee "${profile_dir}/oncard_iter35_profile2_parity.log"

printf "oncard_completed_at=%s\n" "$(date --iso-8601=seconds)" \
    >> "${runtime_dir}/monitor.manifest"
