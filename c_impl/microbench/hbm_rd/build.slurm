#!/bin/bash
#SBATCH --job-name=hbm_rd_build
#SBATCH --partition=build
#SBATCH --constraint=vivado2024.2
#SBATCH --exclude=acclnode03,acclnode04,acclnode05,harrier
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --chdir=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_rd
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_rd/build-%j.log
set -o pipefail
B=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_rd
MARK=$B/build.exit; rm -f "$MARK"
trap 'rc=$?; echo "$rc" > "$MARK"; echo "HBM_RD_BUILD_EXIT=$rc end=$(date -u +%FT%TZ)"' EXIT
echo "node=$(hostname -s) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p "$TMPDIR"
set +u; source /tools/Xilinx/Vitis/2024.2/settings64.sh; source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; set -u
make xo JOBS=8 FREQ=300 || exit 3
echo "--- csynth: II and clock"; R=$(find build.hw.hbm_rd.f300/reports_compile -name "hbm_rd_csynth.rpt" | head -1); grep -E "^\s*\|- rd_loop|\|ap_clk" "$R" | head -3
make xclbin JOBS=8 FREQ=300 || exit 4
echo "--- routed clocks"; cat $(find build.hw.hbm_rd.f300 -name hbm_rd_clocks.tsv | head -1)
echo "--- xclbin clocks"; xclbinutil --input build.hw.hbm_rd.f300/hbm_rd.xclbin --info 2>/dev/null | grep -A2 -E "DATA_CLK|hbm_aclk" | grep -E "Name|Frequency"
grep -h "AUTO-FREQ-SCALING" build.hw.hbm_rd.f300/hbm_rd_link.log | head -3
sha256sum build.hw.hbm_rd.f300/hbm_rd.xclbin | cut -c1-16
