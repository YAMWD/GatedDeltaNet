#!/bin/bash
#SBATCH --job-name=gdn_build
#SBATCH --partition=build
#SBATCH --exclude=acclnode04,acclnode05
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#
# Build half of the hardware flow: .xo -> .xclbin -> host.exe. No card is
# touched here, and none can be: the `build` QOS grants gres/fpga=0. The
# on-card gates run as a separate `light` job (slurm/hw_oncard.slurm),
# because one job cannot hold both 32 cores and an FPGA.
#
# Why each setting, all measured on 2026-08-22 (see the accl-cluster skill):
#   --partition=build   `light` caps a job at 32 GB and the OOM killer took
#                       Vivado there at 27 GB, during Design Initialization.
#                       This design has peaked at 51 GB.
#   --exclude=...04,05  those two nodes have no U55C platform under
#                       /opt/xilinx/platforms, so v++ cannot link for this
#                       card there. acclnode01, acclnode03 and harrier can.
#                       Pinning to harrier also works but serialises against
#                       card jobs for no reason.
#   --mem=128G          QOS ceiling is 196800M. 128G is ~2.5x the measured peak.
#   --time=2-00:00:00   the `build` default is 12 h and links here have taken
#                       7-32 h. MaxTime is 3 days, so this cannot go higher
#                       than 3-00:00:00.
#
# Submit through slurm/../run_hw_sbatch.sh, which chains the on-card job to
# this one and keeps the build knobs identical across the pair.

set -uo pipefail

JOBS="${JOBS:-32}"
HLS_FREQ="${HLS_FREQ:-150}"
LINK_FREQ="${LINK_FREQ:-100}"
TAG="${TAG:?TAG must be set by the submitter}"

echo "=== gdn hardware build ==="
echo "  node       : $(hostname)"
echo "  job        : ${SLURM_JOB_ID:-<none>}"
echo "  start      : $(date --iso-8601=seconds)"
echo "  cpus       : ${SLURM_CPUS_PER_TASK:-?}   JOBS=${JOBS}"
echo "  freqs      : HLS=${HLS_FREQ} LINK=${LINK_FREQ}"
echo "  workdir    : $(pwd)"
sha256sum gdn_model.cpp gdn_model.h host.cpp hw_f150_physical_islands.cfg 2>/dev/null | sed 's/^/  /'

# Resolve the build directory the same way the Makefile does, so the on-card
# job can find the image. The directory name embeds JOBS, so a mismatched
# JOBS between the two jobs silently points at a different (or absent) build.
BUILD_DIR="$(make -s --eval='__p:;@echo $(BUILD_DIR)' __p JOBS="$JOBS" \
             HLS_FREQ="$HLS_FREQ" LINK_FREQ="$LINK_FREQ" 2>/dev/null | tail -1)"
echo "  build dir  : ${BUILD_DIR}"
mkdir -p "diagnostics/${TAG}"
printf '%s\n' "$BUILD_DIR" > "diagnostics/${TAG}/build_dir.path"

make xo xclbin host JOBS="$JOBS" HLS_FREQ="$HLS_FREQ" LINK_FREQ="$LINK_FREQ"
rc=$?

echo "EXIT_CODE=${rc} end=$(date --iso-8601=seconds)"
printf '%s\n' "$rc" > "diagnostics/${TAG}/build.exit"
if [ "$rc" -eq 0 ]; then
    ls -l "${BUILD_DIR}/gdn_forward.xclbin" 2>/dev/null | sed 's/^/  image: /'
fi
exit "$rc"
