#!/bin/bash
#SBATCH --job-name=iter75f_ce_locality
#SBATCH --partition=build
#SBATCH --constraint=vivado2024.2
#SBATCH --exclude=acclnode03,harrier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter75f_ce_locality/slurm-%j.log
# Shared checkpoint: no node pin or U55C platform/GRES needed for this repair.
# Exclude the two recorded disk-full nodes; retain an explicit scratch check.
set -eo pipefail
ROOT=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter75f_ce_locality
INPUT=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter75e_postroute_explore/reports-3953/after_explore.dcp
WORK="/tmp/${USER}-${SLURM_JOB_ID}/ce_locality"
exec > >(tee "$ROOT/repair.live.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/repair.exit"; echo "REPAIR_EXIT=$rc"' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
df -h /tmp
available_kib=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$available_kib" -ge 62914560 || { echo 'FATAL: less than60GiB scratch'; exit 2; }
test "$(sha256sum "$INPUT" | awk '{print $1}')" = ee44dbecbb8d71675dfed80cc2a7c9ffd7c8b6b1d17ce462984271ea1f582f2e
test "$(sha256sum "$ROOT/scripts_snapshot.tar" | awk '{print $1}')" = "${REPAIR_SNAPSHOT_SHA256:?missing snapshot hash}"
mkdir -p "$WORK"
cp "$INPUT" "$WORK/input.dcp"
test "$(sha256sum "$WORK/input.dcp" | awk '{print $1}')" = ee44dbecbb8d71675dfed80cc2a7c9ffd7c8b6b1d17ce462984271ea1f582f2e
tar -xf "$ROOT/scripts_snapshot.tar" -C "$WORK"
sha256sum "$WORK/input.dcp" "$WORK/close_f150_common.tcl" "$WORK/close_f150_kernel_group.tcl" "$WORK/close_f150_kernel_group.slurm" | tee "$ROOT/manifest-$SLURM_JOB_ID.sha256"
export REPAIR_DCP="$WORK/input.dcp" REPAIR_OUT="$ROOT/reports-$SLURM_JOB_ID"
export REPAIR_STATUS="$ROOT/repair.phase"
cd "$WORK"
tclsh parser_test.tcl
source /tools/Xilinx/Vitis/2024.2/settings64.sh
vivado -mode batch -nojournal -log "$ROOT/vivado.live.log" -source close_f150_kernel_group.tcl
