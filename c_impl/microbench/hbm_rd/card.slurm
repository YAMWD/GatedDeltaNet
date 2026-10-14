#!/bin/bash
#SBATCH --job-name=hbm_rd_card
#SBATCH --partition=light
#SBATCH --nodelist=harrier
#SBATCH --gres=fpga:u55c:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --chdir=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_rd
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_rd/card-%j.log
set -o pipefail
B=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_rd
MARK=$B/card.exit; rm -f "$MARK"
trap 'rc=$?; echo "$rc" > "$MARK"; echo "HBM_RD_CARD_EXIT=$rc end=$(date -u +%FT%TZ)"' EXIT
echo "node=$(hostname -s) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || exit 2
set -u
XCL=$B/build.hw.hbm_rd.f300/hbm_rd.xclbin; test -s "$XCL" || { echo "no xclbin"; exit 2; }
make -s host HOST_BIN=$B/host_hbm_rd_harrier.exe || exit 3
BDF="$(xbutil examine 2>/dev/null | grep -i xilinx_u55c | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | tr -d '[]' | sort -u)"
[ "$(printf '%s\n' "$BDF" | wc -l)" -eq 1 ] && [ -n "$BDF" ] || { echo "FATAL: U55C BDF '$BDF'"; exit 2; }
echo "bdf=$BDF"; xclbinutil --input "$XCL" --info 2>/dev/null | grep -A2 -E "DATA_CLK|hbm_aclk" | grep -E "Name|Frequency"
$B/host_hbm_rd_harrier.exe "$XCL" "$BDF" 256 16 3 4
rc=$?
echo "--- clocks read back"; xbutil examine -d "$BDF" -r platform 2>/dev/null | grep -iA6 "^Clocks" | grep -iE "clk|hbm" | head -4
exit $rc
