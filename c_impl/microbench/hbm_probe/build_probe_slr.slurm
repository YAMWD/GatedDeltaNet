#!/bin/bash
#SBATCH --job-name=iter79_hbm_probe_slr_build
#SBATCH --partition=build
#SBATCH --constraint=vivado2024.2
#SBATCH --exclude=acclnode04,acclnode05,harrier
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_probe/logs/build-%j.log
set -o pipefail
cd /home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_probe
trap 'rc=$?; printf "%s\n" "$rc" > /home/yaoz0b/GatedDeltaNet/c_impl/microbench/hbm_probe/logs/build-np${NPORTS:-32}-${SLR:-SLR0}.exit; echo "BUILD_EXIT=$rc end=$(date -u +%FT%TZ)"' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ) FREQ=${FREQ:-200} OUTSTANDING=${OUTSTANDING:-4} NPORTS=${NPORTS:-32} SLR=${SLR:-}"
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p "$TMPDIR"
set +u; source /tools/Xilinx/Vitis/2024.2/settings64.sh; source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; set -u
make xo FREQ=${FREQ:-200} OUTSTANDING=${OUTSTANDING:-4} NPORTS=${NPORTS:-32} SLR=${SLR:-} JOBS=16 || exit 3
make host || exit 4
make xclbin FREQ=${FREQ:-200} OUTSTANDING=${OUTSTANDING:-4} NPORTS=${NPORTS:-32} SLR=${SLR:-} JOBS=16 || exit 5
X=build.hw.hbm_probe.f${FREQ:-200}.o${OUTSTANDING:-4}$( [ "${NPORTS:-32}" = 32 ] || echo .np${NPORTS} )${SLR:+.$SLR}/hbm_probe.xclbin
xclbinutil --input $X --info 2>/dev/null | grep -iE 'DATA_CLK|Clock|Frequency' | head -n 6
sha256sum $X
