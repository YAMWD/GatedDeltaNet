#!/bin/bash -l
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=48G
#SBATCH --job-name=gdn-eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd /home/yaoz0b/GatedDeltaNet
module purge
module load cuda/12.1

ENV_PREFIX="${ENV_PREFIX:-/ibex/user/yaoz0b/conda-environments/GatedDeltaNet}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHON_BIN="${ENV_PREFIX}/bin/python"
LM_EVAL_HOME="${LM_EVAL_HOME:-$HOME/.cache/lm-evaluation-harness}"

if [ ! -d "${LM_EVAL_HOME}/.git" ]; then
  mkdir -p "$(dirname "${LM_EVAL_HOME}")"
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness "${LM_EVAL_HOME}"
fi
export PYTHONPATH="${LM_EVAL_HOME}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-m-a-p/1.3B-100B-GatedDeltaNet-pure}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"
EXTRA_ARGS=("${@}")

export MODEL_ID DTYPE BATCH_SIZE OUTPUT_DIR
scripts/run_gdn_table3_eval.sh "${EXTRA_ARGS[@]}"
