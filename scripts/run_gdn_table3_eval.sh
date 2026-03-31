#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-m-a-p/1.3B-100B-GatedDeltaNet-pure}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TASKS="${TASKS:-wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/$(basename "${MODEL_ID//\//_}")-table3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LM_EVAL_MODEL="${LM_EVAL_MODEL:-gdn_hf}"
GDN_ATTN_MODE="${GDN_ATTN_MODE:-fused_recurrent}"
EXTRA_ARGS=("${@}")

mkdir -p "${OUTPUT_DIR}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/fla_lm_eval.py \
  --model "${LM_EVAL_MODEL}" \
  --model_args "pretrained=${MODEL_ID},dtype=${DTYPE},trust_remote_code=True,gdn_attn_mode=${GDN_ATTN_MODE}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_fewshot 0 \
  --device cuda \
  --output_path "${OUTPUT_DIR}" \
  --show_config \
  --trust_remote_code \
  "${EXTRA_ARGS[@]}"
