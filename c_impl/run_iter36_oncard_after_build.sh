#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(100|130)$ ]]; then
    echo "usage: $0 <100|130>" >&2
    exit 2
fi

link_frequency_mhz=$1
c_impl_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${c_impl_dir}/.." && pwd)
diagnostics_dir="${c_impl_dir}/diagnostics/iter36_headlocal_f${link_frequency_mhz}"
runtime_dir="${diagnostics_dir}/on_card"
build_dir="${c_impl_dir}/build.hw.iter36.headlocal.iter35.postphys.f${link_frequency_mhz}.o8.v2022_2"
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

for _ in $(seq 1 30); do
    [[ -f "${build_pid_file}" ]] && break
    sleep 1
done
if [[ ! -f "${build_pid_file}" ]]; then
    echo "iter36 f${link_frequency_mhz} on-card monitor: missing build PID" >&2
    exit 1
fi
build_pid=$(<"${build_pid_file}")
if [[ ! "${build_pid}" =~ ^[0-9]+$ ]]; then
    echo "iter36 f${link_frequency_mhz} on-card monitor: invalid build PID: ${build_pid}" >&2
    exit 1
fi

{
    printf "monitor_started_at=%s\n" "$(date --iso-8601=seconds)"
    printf "build_pid=%s\n" "${build_pid}"
    printf "device_index=%s\n" "${device_index}"
    printf "smoke_decode_len=8\n"
    printf "performance_decode_len=64\n"
} > "${runtime_dir}/monitor.manifest"

while kill -0 "${build_pid}" 2>/dev/null; do
    sleep 30
done

for _ in 1 2 3 4 5; do
    [[ -f "${build_exit_file}" ]] && break
    sleep 1
done
if [[ ! -f "${build_exit_file}" ]]; then
    echo "iter36 f${link_frequency_mhz} on-card monitor: build ended without exit marker" >&2
    exit 1
fi
build_exit=$(<"${build_exit_file}")
printf "build_exit=%s\n" "${build_exit}" >> "${runtime_dir}/monitor.manifest"
if [[ "${build_exit}" != "0" ]]; then
    echo "iter36 f${link_frequency_mhz}: hardware build failed; on-card run skipped" \
        | tee "${runtime_dir}/monitor.skipped"
    exit 2
fi
if [[ ! -s "${xclbin}" ]]; then
    echo "iter36 f${link_frequency_mhz}: successful build has no XCLBIN" >&2
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

run_decode 8 "oncard_iter36_f${link_frequency_mhz}_smoke8"
run_decode 64 "oncard_iter36_f${link_frequency_mhz}_decode64"

output="${runtime_dir}/oncard_iter36_f${link_frequency_mhz}_decode64.json"
summary="${runtime_dir}/performance_summary.json"
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
        speedup_vs_iter35_75_061694ms:
            (75.061694 / ($runs | add / length)),
        speedup_vs_iter32_98_6595983ms:
            (98.6595983 / ($runs | add / length)),
        speedup_vs_8port_121_4ms:
            (121.4 / ($runs | add / length))
      }
' "${output}" | tee "${summary}"

jq -e '.measured_runs == 63 and .mean_kernel_ms > 0' "${summary}" >/dev/null
printf "oncard_completed_at=%s\n" "$(date --iso-8601=seconds)" \
    >> "${runtime_dir}/monitor.manifest"
