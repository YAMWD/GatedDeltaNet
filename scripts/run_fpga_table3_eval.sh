#!/usr/bin/env bash
set -euo pipefail

: "${FPGA_QUEUE:?set FPGA_QUEUE to the node-local request directory}"
: "${FPGA_RESULTS:?set FPGA_RESULTS to the persistent raw-result directory}"

MODEL_ID="${MODEL_ID:-m-a-p/1.3B-100B-GatedDeltaNet-pure}"
TASKS="${TASKS:-wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/fpga-table3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUEST_PREFIX="${REQUEST_PREFIX:-table3}"

mkdir -p "${OUTPUT_DIR}" "${FPGA_QUEUE}" "${FPGA_RESULTS}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/fpga_lm_eval.py \
  --model gdn_fpga \
  --model_args "pretrained=${MODEL_ID},dtype=bfloat16,trust_remote_code=True,gdn_attn_mode=chunk,fpga_queue=${FPGA_QUEUE},fpga_results=${FPGA_RESULTS},fpga_request_prefix=${REQUEST_PREFIX}" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --num_fewshot 0 \
  --device cuda \
  --output_path "${OUTPUT_DIR}" \
  --show_config \
  --trust_remote_code \
  "$@"

