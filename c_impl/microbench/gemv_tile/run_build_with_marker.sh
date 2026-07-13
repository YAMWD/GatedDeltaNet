#!/usr/bin/env bash

set -uo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 <marker> <log> <artifact-root> <command> [args...]" >&2
    exit 2
fi

marker=$1
log=$2
artifact_root=$3
shift 3

mkdir -p "$(dirname "$marker")" "$(dirname "$log")"
rm -f "$marker"

started_epoch=$(date +%s)
started_at=$(date --iso-8601=seconds)

set +e
"$@" 2>&1 | tee "$log"
command_status=${PIPESTATUS[0]}
set -e

ended_epoch=$(date +%s)
ended_at=$(date --iso-8601=seconds)
elapsed_seconds=$((ended_epoch - started_epoch))

xclbin=$(find "$artifact_root" -type f -name '*.xclbin' -print 2>/dev/null | sort | tail -1)
routed_error_dcp=$(find "$artifact_root" -type f -name '*routed_error.dcp' -print 2>/dev/null | sort | tail -1)
routed_dcp=$(find "$artifact_root" -type f -name '*routed.dcp' -print 2>/dev/null | sort | tail -1)

{
    printf 'exit_code=%s\n' "$command_status"
    printf 'started_at=%s\n' "$started_at"
    printf 'ended_at=%s\n' "$ended_at"
    printf 'elapsed_seconds=%s\n' "$elapsed_seconds"
    printf 'log=%s\n' "$log"
    printf 'xclbin=%s\n' "$xclbin"
    printf 'routed_dcp=%s\n' "$routed_dcp"
    printf 'routed_error_dcp=%s\n' "$routed_error_dcp"
} >"$marker"

echo "BUILD_WAITER_COMPLETE marker=$marker exit_code=$command_status elapsed_seconds=$elapsed_seconds"
exit "$command_status"
