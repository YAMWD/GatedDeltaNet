#!/usr/bin/env bash
set -euo pipefail

: "${FPGA_QUEUE:?set FPGA_QUEUE to the node-local request directory}"
: "${FPGA_RESULTS:?set FPGA_RESULTS to the persistent raw-result directory}"

MODEL_ID="${MODEL_ID:-m-a-p/1.3B-100B-GatedDeltaNet-pure}"
TASKS="${TASKS:-gdn_niah_single_1,gdn_niah_single_2}"
SEQ_LENGTHS="${SEQ_LENGTHS:-[1024,2048,4096,8192]}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/fpga-table2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_ROOT="${TASK_ROOT:-$PWD/scripts/eval_tasks/gdn_ruler_table2}"
REQUEST_PREFIX="${REQUEST_PREFIX:-table2}"

mkdir -p "${OUTPUT_DIR}" "${FPGA_QUEUE}" "${FPGA_RESULTS}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/fpga_lm_eval.py \
  --model gdn_fpga \
  --model_args "pretrained=${MODEL_ID},dtype=bfloat16,max_length=8192,trust_remote_code=True,gdn_attn_mode=chunk,fpga_queue=${FPGA_QUEUE},fpga_results=${FPGA_RESULTS},fpga_request_prefix=${REQUEST_PREFIX}" \
  --tasks "${TASKS}" \
  --include_path "${TASK_ROOT}" \
  --metadata "{\"max_seq_lengths\":${SEQ_LENGTHS}}" \
  --batch_size 1 \
  --num_fewshot 0 \
  --seed 42 \
  --device cuda \
  --output_path "${OUTPUT_DIR}" \
  --show_config \
  --trust_remote_code \
  "$@"

