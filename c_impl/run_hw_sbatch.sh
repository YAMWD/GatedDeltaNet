#!/bin/bash
#SBATCH --job-name=gdn_iter61
#SBATCH --partition=build
#SBATCH --nodelist=harrier
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/yaoz0b/GatedDeltaNet/c_impl
#SBATCH --output=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter61_stream/slurm-%j.out
#SBATCH --error=/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/iter61_stream/slurm-%j.out
#
# Iter61 hardware build.
#
# Two environment constraints, both measured:
#   * the `light` partition caps a job at 32 GB, and Vivado was OOM-killed there
#     at 27 GB during Design Initialization, before placement ran. `build`
#     allows up to 192 GB; 64 GB is double the highest this design has been seen
#     to reach (31.9 GB on the Iter59 routing run) and fits the ~72 GB currently
#     schedulable on harrier.
#   * pinned to harrier because acclnode03 runs jobs but CANNOT WRITE to
#     /home/yaoz0b -- probe job 314 completed there and produced no file, which
#     is why jobs 312/313 failed with exit 2 and no output. The other `build`
#     nodes are unusable until that mount is fixed.

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
awk '{printf "mem_limit=%.0f GB\n", $1/1073741824}' \
  /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_$SLURM_JOB_ID/memory.limit_in_bytes 2>/dev/null
sha256sum gdn_model.cpp gdn_model.h host.cpp hw_f150_physical_islands.cfg

make run_hw JOBS=16
rc=$?
echo "EXIT_CODE=$rc end=$(date --iso-8601=seconds)"
echo "$rc" > diagnostics/iter61_stream/sbatch_exit_code
exit $rc
