#!/bin/bash
#SBATCH --job-name=iter79_slr_stream_card
#SBATCH --partition=light
#SBATCH --gres=fpga:u55c:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/slr_stream/logs/card-%j.log
set -o pipefail
cd /home/yaoz0b/GatedDeltaNet/c_impl/microbench/slr_stream
FREQ=${FREQ:-200}; SCDEPTH=${SCDEPTH:-}; B=build.hw.slr_stream.f${FREQ}${SCDEPTH:+.sc$SCDEPTH}
trap 'rc=$?; printf "%s\n" "$rc" > logs/card-$B.exit; echo "CARD_EXIT=$rc end=$(date -u +%FT%TZ)"' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || { echo "FATAL: cannot source XRT setup" >&2; exit 2; }
source /tools/Xilinx/Vitis/2024.2/settings64.sh >/dev/null 2>&1 || true
make host || { echo 'FATAL: host build failed' >&2; exit 2; }
set -u
examine="$(xbutil examine 2>/dev/null)"
mapfile -t u55c_bdfs < <(printf '%s\n' "$examine" | grep -i 'xilinx_u55c' | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | tr -d '[]' | sort -u)
[ "${#u55c_bdfs[@]}" -eq 1 ] || { echo "FATAL: expected exactly 1 U55C, found ${#u55c_bdfs[@]}" >&2; exit 2; }
HW_DEVICE="${u55c_bdfs[0]}"; X=$B/slr_stream.xclbin
echo "xclbin: $X sha256 $(sha256sum $X | cut -c1-16)"; xclbinutil --input $X --info 2>/dev/null | grep -iE 'DATA_CLK' | head -n 2
ACH=$(xclbinutil --input $X --info 2>/dev/null | grep -iE 'DATA_CLK' | grep -oE '[0-9]+ MHz' | head -n1 | grep -oE '[0-9]+'); ACH=${ACH:-200}
echo "=== 1M beats x 4 reps, 5 timed runs, clock $ACH MHz ==="; ./host_slr_stream.exe $X $HW_DEVICE 1048576 4 5 $ACH
