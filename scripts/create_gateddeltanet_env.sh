#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-GatedDeltaNet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

conda run -n "${ENV_NAME}" pip install torch==2.7.1 --index-url "${TORCH_INDEX_URL}"
conda run -n "${ENV_NAME}" pip install \
  flash-linear-attention==0.4.2 \
  accelerate==1.13.0 \
  datasets==3.6.0 \
  transformers==4.51.3 \
  sentencepiece==0.2.1 \
  huggingface-hub==0.36.2 \
  einops==0.8.2
conda run -n "${ENV_NAME}" pip install "git+https://github.com/EleutherAI/lm-evaluation-harness.git"
