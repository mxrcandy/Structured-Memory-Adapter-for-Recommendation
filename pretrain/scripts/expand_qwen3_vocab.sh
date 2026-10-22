#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_MODEL_DIR="${HF_MODEL_DIR:?Set HF_MODEL_DIR to the base Qwen3 checkpoint.}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:?Set OUTPUT_MODEL_DIR to the expanded checkpoint destination.}"
ITEMIC_LAYER_N="${ITEMIC_LAYER_N:-3}"
VOCAB_SIZE_PER_LAYER="${VOCAB_SIZE_PER_LAYER:-8192}"

if [ ! -d "$HF_MODEL_DIR" ]; then
  echo "Base model not found: $HF_MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_MODEL_DIR")"

python tools/model_converter/expand_qwen3_vocab.py \
  --hf_model_dir "$HF_MODEL_DIR" \
  --output_model_dir "$OUTPUT_MODEL_DIR" \
  --itemic_layer_n "$ITEMIC_LAYER_N" \
  --vocab_size_per_layer "$VOCAB_SIZE_PER_LAYER"

echo "Expanded itemic model written to: $OUTPUT_MODEL_DIR"

