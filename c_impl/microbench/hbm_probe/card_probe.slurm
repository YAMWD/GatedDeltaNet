#!/bin/bash
#SBATCH --job-name=iter78_hbm_probe_card
#SBATCH --partition=light
#SBATCH --gres=fpga:u55c:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_probe/logs/card-%j.log
set -o pipefail
cd /home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_probe
trap 'rc=$?; printf "%s\n" "$rc" > /home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_probe/logs/card.exit; echo "CARD_EXIT=$rc end=$(date -u +%FT%TZ)"' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || { echo "FATAL: cannot source XRT setup" >&2; exit 2; }
set -u
# Select the allocated U55C by BDF, as hw_oncard.slurm does (a visible
# unallocated U280 makes "exactly one card" the wrong guard).
examine="$(xbutil examine 2>/dev/null)"
mapfile -t u55c_bdfs < <(printf '%s\n' "$examine" | grep -i 'xilinx_u55c' \
    | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | tr -d '[]' | sort -u)
echo "  u55c cards   : ${#u55c_bdfs[@]}  ${u55c_bdfs[*]:-none}"
[ "${#u55c_bdfs[@]}" -eq 1 ] || { echo "FATAL: expected exactly 1 U55C, found ${#u55c_bdfs[@]}" >&2; exit 2; }
HW_DEVICE="${u55c_bdfs[0]}"
X=build.hw.hbm_probe.f${FREQ:-200}.o${OUTSTANDING:-4}/hbm_probe.xclbin
echo "xclbin: $X  sha256 $(sha256sum $X | cut -c1-16)"
echo "achieved kernel clock (DATA_CLK) from the image:"
xclbinutil --input $X --info 2>/dev/null | grep -iE 'DATA_CLK|KERNEL_CLK|Frequency' | head -n 6
echo "=== 87.5 MB/port x 4 reps, 9 timed runs ==="
./host_hbm_probe.exe $X $HW_DEVICE 1366528 4 9
echo "=== 87.5 MB/port x 1 rep (single pass), 9 timed runs ==="
./host_hbm_probe.exe $X $HW_DEVICE 1366528 1 9
