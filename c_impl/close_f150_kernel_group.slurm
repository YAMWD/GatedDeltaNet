#!/bin/bash
#SBATCH --job-name=close_f150_kernel_group
#SBATCH --partition=build
#SBATCH --constraint=vivado2024.2
#SBATCH --exclude=acclnode03,harrier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/close_f150_kernel_group/slurm-%j.log
# One focused post-route pass on a routed checkpoint, confined to the kernel
# clock's path group (job 3955 took the -0.001 ns checkpoint to 0.000 this way).
#   inputs (--export): CLOSE_INPUT_DCP     routed checkpoint to close (required)
#                      CLOSE_INPUT_SHA256  its expected digest (optional; verified when set)
#   Everything else is staged from the repository. Slurm opens --output before
#   the script runs, so create the log directory first:
#     mkdir -p c_impl/diagnostics/close_f150_kernel_group && \
#     sbatch --export=ALL,CLOSE_INPUT_DCP=/abs/after_explore.dcp c_impl/close_f150_kernel_group.slurm
#   Outputs: reports-<job>/ (per-stage route/timing/CE evidence, after_focused.dcp,
#   closed_f150.dcp + physical_gate.pass on success), manifest-<job>.sha256,
#   vivado-<job>.log, repair-<job>.{phase,exit}.
set -eo pipefail
C=/home/yaoz0b/GatedDeltaNet/c_impl
ROOT=$C/diagnostics/close_f150_kernel_group
INPUT="${CLOSE_INPUT_DCP:?set CLOSE_INPUT_DCP to the routed checkpoint to close}"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/repair-$SLURM_JOB_ID.exit"; echo "REPAIR_EXIT=$rc end=$(date -u +%FT%TZ)"' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ) input=$INPUT"
test -s "$INPUT" || { echo "FATAL: checkpoint missing: $INPUT"; exit 2; }
if [ -n "${CLOSE_INPUT_SHA256:-}" ]; then
    test "$(sha256sum "$INPUT" | awk '{print $1}')" = "$CLOSE_INPUT_SHA256" || { echo "FATAL: input digest mismatch"; exit 2; }
fi
# Two checkpoints (~1.5 GB) are written; use node-local disk when it has room,
# otherwise shared storage (harrier's and acclnode03's /tmp were full 2026-09-10/11).
df -h /tmp
if [ "$(df -Pk /tmp | awk 'NR==2 {print $4}')" -ge 20971520 ]; then
    WORK=/tmp/$USER-$SLURM_JOB_ID/close_f150
else
    WORK=$ROOT/work-$SLURM_JOB_ID; echo "NOTE: /tmp has <20 GiB; working on shared storage at $WORK"
fi
mkdir -p "$WORK"
cp "$INPUT" "$WORK/input.dcp"
cp "$C/close_f150_common.tcl" "$C/close_f150_kernel_group.tcl" "$C/close_f150_parser_test.tcl" "$WORK/"
sha256sum "$WORK/input.dcp" "$WORK"/close_f150_*.tcl | tee "$ROOT/manifest-$SLURM_JOB_ID.sha256"
export REPAIR_DCP="$WORK/input.dcp" REPAIR_OUT="$ROOT/reports-$SLURM_JOB_ID" REPAIR_STATUS="$ROOT/repair-$SLURM_JOB_ID.phase"
cd "$WORK"
tclsh close_f150_parser_test.tcl
source /tools/Xilinx/Vitis/2024.2/settings64.sh
vivado -mode batch -nojournal -log "$ROOT/vivado-$SLURM_JOB_ID.log" -source close_f150_kernel_group.tcl
