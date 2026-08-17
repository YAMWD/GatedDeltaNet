#!/usr/bin/env bash
# Run one full precision arm - Tables 2, 3, and 5 - against a single checkpoint.
#
# The three arms recorded in c_impl/doc/fp32_bf16_quality_evaluation.md differ
# only in MODEL_ID and DTYPE:
#
#   FP32    MODEL_ID=<original snapshot>      DTYPE=float32
#   BF16 A  MODEL_ID=<converted checkpoint>   DTYPE=bfloat16   (weights + activations BF16)
#   BF16 B  MODEL_ID=<converted checkpoint>   DTYPE=float32    (BF16 weights, FP32 math)
#
# Arm B loads the BF16 file and upcasts on load, which is the configuration the
# FPGA implements: BF16 in memory, FP32 arithmetic.
#
# Usage:
#   MODEL_ID=<checkpoint> DTYPE=float32 OUT_ROOT=<dir> scripts/run_gdn_precision_eval_arm.sh
#
# Batch sizes are pinned to the values used for every recorded arm so results
# stay comparable; override only if you intend to break that comparability.
set +e
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 97

MODEL_ID="${MODEL_ID:?set MODEL_ID to the checkpoint directory}"
DTYPE="${DTYPE:?set DTYPE to float32 or bfloat16}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT to the output directory for this arm}"
PYTHON_BIN="${PYTHON_BIN:-.micromamba/envs/gdn-hf/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
T3_BATCH="${T3_BATCH:-4}"
T2_BATCH="${T2_BATCH:-16}"
T5_BATCH="${T5_BATCH:-16}"

mkdir -p "$OUT_ROOT/full_evaluation"

echo "ARM MODEL_ID=$MODEL_ID DTYPE=$DTYPE OUT_ROOT=$OUT_ROOT"

echo "PHASE=table3 START=$(date --iso-8601=seconds)"
env CUDA_VISIBLE_DEVICES="$GPU" PYTHON_BIN="$PYTHON_BIN" MODEL_ID="$MODEL_ID" \
    DTYPE="$DTYPE" BATCH_SIZE="$T3_BATCH" OUTPUT_DIR="$OUT_ROOT/table3/results" \
    scripts/run_gdn_table3_eval.sh
rc=$?

if [ "$rc" -eq 0 ]; then
  echo "PHASE=table2_s12 START=$(date --iso-8601=seconds)"
  env CUDA_VISIBLE_DEVICES="$GPU" PYTHON_BIN="$PYTHON_BIN" MODEL_ID="$MODEL_ID" \
      DTYPE="$DTYPE" BATCH_SIZE="$T2_BATCH" \
      TASKS=gdn_niah_single_1,gdn_niah_single_2 \
      SEQ_LENGTHS="[1024,2048,4096,8192]" MAX_LENGTH=8192 \
      OUTPUT_DIR="$OUT_ROOT/table2/s12" \
      scripts/run_gdn_table2_eval.sh --log_samples
  rc=$?
fi

if [ "$rc" -eq 0 ]; then
  echo "PHASE=table2_s3 START=$(date --iso-8601=seconds)"
  env CUDA_VISIBLE_DEVICES="$GPU" PYTHON_BIN="$PYTHON_BIN" MODEL_ID="$MODEL_ID" \
      DTYPE="$DTYPE" BATCH_SIZE="$T2_BATCH" TASKS=gdn_niah_single_3 \
      SEQ_LENGTHS="[1024,2048,4096]" MAX_LENGTH=4096 \
      OUTPUT_DIR="$OUT_ROOT/table2/s3" \
      scripts/run_gdn_table2_eval.sh --log_samples
  rc=$?
fi

if [ "$rc" -eq 0 ]; then
  echo "PHASE=table5 START=$(date --iso-8601=seconds)"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/run_gdn_longbench_eval.py \
      --model "$MODEL_ID" --dtype "$DTYPE" --output-dir "$OUT_ROOT/table5" \
      --batch-size "$T5_BATCH" --overwrite
  rc=$?
fi

echo "EXIT_CODE=$rc END=$(date --iso-8601=seconds)"
printf "%s\n" "$rc" > "$OUT_ROOT/full_evaluation/exit_code"
exit "$rc"
