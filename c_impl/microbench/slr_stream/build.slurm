#!/bin/bash
#SBATCH --job-name=iter79_slr_stream_build
#SBATCH --partition=build
#SBATCH --constraint=vivado2024.2
#SBATCH --exclude=acclnode04,acclnode05,harrier
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/slr_stream/logs/build-%j.log
set -o pipefail
cd /home/yaoz0b/GatedDeltaNet/c_impl/microbench/slr_stream
FREQ=${FREQ:-200}; SCDEPTH=${SCDEPTH:-}
B=build.hw.slr_stream.f${FREQ}${SCDEPTH:+.sc$SCDEPTH}
trap 'rc=$?; printf "%s\n" "$rc" > logs/build-$B.exit; echo "BUILD_EXIT=$rc end=$(date -u +%FT%TZ)"' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ) FREQ=$FREQ SCDEPTH=${SCDEPTH:-none} dir=$B"
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p "$TMPDIR"
set +u; source /tools/Xilinx/Vitis/2024.2/settings64.sh; source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; set -u
make xo FREQ=$FREQ SCDEPTH="$SCDEPTH" JOBS=16 || exit 3
make host || exit 4
make xclbin FREQ=$FREQ SCDEPTH="$SCDEPTH" JOBS=16 || exit 5
set +o pipefail; set +e   # everything below is reporting; the link result is already decided
X=$B/slr_stream.xclbin
echo "=== image"; xclbinutil --input $X --info 2>/dev/null | grep -iE 'DATA_CLK|Clock|Frequency' | head -n 6; sha256sum $X
R=$(ls -d $B/_x_temp/link/vivado/vpl/prj/prj.runs/impl_1 2>/dev/null | head -n1)
echo "=== routed timing per clock ($R)"
T=$(ls $R/*timing_summary_routed.rpt 2>/dev/null | head -n1); [ -n "$T" ] && awk '/Clock Summary/{c=1} /Intra Clock Table/{p=1} p&&NR<2000' "$T" | grep -E '^(clk_|dma_|hbm_|Clock|  )' | head -n 20
grep -E 'WNS|TNS|WHS' "$T" 2>/dev/null | head -n 4
echo "=== what v++ put on the stream connections (BD cells)"
for bd in $(find $B/_x_temp/link/vivado/vpl -name '*.bd' 2>/dev/null | head -n 3); do echo "-- $bd"; grep -oE '"VLNV": "[^"]*(axis_register_slice|axis_data_fifo|axis_switch|axis_clock_converter|axi_register_slice|axis_dwidth)[^"]*"' $bd | sort | uniq -c; done
find $B/_x_temp/link/vivado/vpl -name '*.bd.tcl' 2>/dev/null | head -n 2 | while read f; do echo "-- $f"; grep -nE 'axis_register_slice|axis_data_fifo|REG_CONFIG|slr' $f | head -n 20 | cut -c1-200; done
echo "=== SLR utilization of the kernels"
U=$(ls $R/*utilization_placed.rpt 2>/dev/null | head -n1); [ -n "$U" ] && grep -nE 'SLR|slr_src|slr_sink' "$U" | head -n 20 | cut -c1-160
