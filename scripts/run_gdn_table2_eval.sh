#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-m-a-p/1.3B-100B-GatedDeltaNet-pure}"
DTYPE="${DTYPE:-float32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASKS="${TASKS:-gdn_niah_single_1,gdn_niah_single_2}"
SEQ_LENGTHS="${SEQ_LENGTHS:-[1024,2048,4096,8192]}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/gdn-table2-${DTYPE}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GDN_ATTN_MODE="${GDN_ATTN_MODE:-fused_recurrent}"
TASK_ROOT="${TASK_ROOT:-$PWD/scripts/eval_tasks/gdn_ruler_table2}"
EXTRA_ARGS=("${@}")

mkdir -p "${OUTPUT_DIR}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/fla_lm_eval.py \
  --model gdn_hf \
  --model_args "pretrained=${MODEL_ID},dtype=${DTYPE},max_length=${MAX_LENGTH},trust_remote_code=True,gdn_attn_mode=${GDN_ATTN_MODE}" \
  --tasks "${TASKS}" \
  --include_path "${TASK_ROOT}" \
  --metadata "{\"max_seq_lengths\":${SEQ_LENGTHS}}" \
  --batch_size "${BATCH_SIZE}" \
  --num_fewshot 0 \
  --seed 42 \
  --device cuda \
  --output_path "${OUTPUT_DIR}" \
  --show_config \
  --trust_remote_code \
  "${EXTRA_ARGS[@]}"
