#!/bin/bash

set -e

BASE_MODEL_DIR=$1
MODEL_HOME=$2
STEP=$3
CKPT_DIR=${MODEL_HOME}/step${STEP}/global_step${STEP}

OUTPUT_DIR=$CKPT_DIR/converted
ITEMIC_START_INDEX="${ITEMIC_START_INDEX:-151669}"
ITEMIC_END_INDEX="${ITEMIC_END_INDEX:-176246}"
ADAPTER_SID_TOKEN_START="${ADAPTER_SID_TOKEN_START:-$ITEMIC_START_INDEX}"
ADAPTER_SID_TOKEN_END="${ADAPTER_SID_TOKEN_END:-$ITEMIC_END_INDEX}"

EXTRA_ARGS=()
if [ -n "${ADAPTER_TYPE:-}" ]; then
    EXTRA_ARGS+=(--adapter_type "$ADAPTER_TYPE")
fi
if [ "${HOULSBY_ADAPTER:-false}" = "true" ]; then
    EXTRA_ARGS+=(--houlsby_adapter --houlsby_adapter_bottleneck "${HOULSBY_ADAPTER_BOTTLENECK:-256}")
fi
if [ -n "${ADAPTER_HIDDEN_SIZE:-}" ]; then
    EXTRA_ARGS+=(--adapter_hidden_size "$ADAPTER_HIDDEN_SIZE")
fi
if [ -n "${ADAPTER_SPM_HIDDEN_SIZE:-}" ]; then
    EXTRA_ARGS+=(--adapter_spm_hidden_size "$ADAPTER_SPM_HIDDEN_SIZE")
fi
if [ -n "${ADAPTER_ITM_HIDDEN_SIZE:-}" ]; then
    EXTRA_ARGS+=(--adapter_itm_hidden_size "$ADAPTER_ITM_HIDDEN_SIZE")
fi
if [ -n "${ADAPTER_LAYERS:-}" ]; then
    EXTRA_ARGS+=(--adapter_layers "$ADAPTER_LAYERS")
fi
if [ -n "${ADAPTER_MEMORY_SIZE:-}" ]; then
    EXTRA_ARGS+=(--adapter_memory_size "$ADAPTER_MEMORY_SIZE")
fi
if [ -n "${ADAPTER_SPM_MEMORY_SIZE:-}" ]; then
    EXTRA_ARGS+=(--adapter_spm_memory_size "$ADAPTER_SPM_MEMORY_SIZE")
fi
if [ -n "${ADAPTER_ITM_MEMORY_SIZE:-}" ]; then
    EXTRA_ARGS+=(--adapter_itm_memory_size "$ADAPTER_ITM_MEMORY_SIZE")
fi
if [ -n "${ADAPTER_NGRAM_ORDER:-}" ]; then
    EXTRA_ARGS+=(--adapter_ngram_order "$ADAPTER_NGRAM_ORDER")
fi
if [ -n "${ADAPTER_LNGRAM_BITS_PER_ROUTE:-}" ]; then
    EXTRA_ARGS+=(--adapter_lngram_bits_per_route "$ADAPTER_LNGRAM_BITS_PER_ROUTE")
fi
if [ -n "${ADAPTER_TOPK:-}" ]; then
    EXTRA_ARGS+=(--adapter_topk "$ADAPTER_TOPK")
fi
if [ "${ADAPTER_USE_ALPHA_LAYER:-false}" = "true" ]; then
    EXTRA_ARGS+=(--adapter_use_alpha_layer --adapter_alpha_init "${ADAPTER_ALPHA_INIT:-0.1}")
fi
if [ -n "${ADAPTER_FIXED_CONTEXT_GATE:-}" ]; then
    EXTRA_ARGS+=(--adapter_fixed_context_gate "$ADAPTER_FIXED_CONTEXT_GATE")
fi
if [ "${ADAPTER_DISABLE_CONV:-false}" = "true" ]; then
    EXTRA_ARGS+=(--adapter_disable_conv)
fi
if [ "${ADAPTER_USE_SID_SPAN:-false}" = "true" ]; then
    EXTRA_ARGS+=(--adapter_use_sid_span)
    EXTRA_ARGS+=(--adapter_sid_token_start "$ADAPTER_SID_TOKEN_START")
    EXTRA_ARGS+=(--adapter_sid_token_end "$ADAPTER_SID_TOKEN_END")
fi
if [ "${BUNDLE_REMOTE_CODE:-0}" = "1" ]; then
    EXTRA_ARGS+=(--bundle_remote_code)
fi
if [ "${MERGE_LORA:-0}" = "1" ]; then
    EXTRA_ARGS+=(--merge_lora)
    EXTRA_ARGS+=(--lora_rank "${LORA_RANK:?Set LORA_RANK when MERGE_LORA=1}")
    EXTRA_ARGS+=(--lora_alpha "${LORA_ALPHA:?Set LORA_ALPHA when MERGE_LORA=1}")
fi

python3 tools/model_converter/convert_checkpoint_to_hf.py --checkpoint_dir $CKPT_DIR \
    --output_dir $OUTPUT_DIR \
    --source_hf_model_path $BASE_MODEL_DIR \
    "${EXTRA_ARGS[@]}"
