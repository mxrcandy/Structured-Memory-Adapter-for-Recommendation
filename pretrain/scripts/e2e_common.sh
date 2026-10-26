#!/usr/bin/env bash

E2E_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_COMMON_PRETRAIN_DIR="$(cd "$E2E_COMMON_DIR/.." && pwd)"
E2E_COMMON_REPO_ROOT="$(cd "$E2E_COMMON_PRETRAIN_DIR/.." && pwd)"

# shellcheck source=scripts/data_profiles.sh
source "$E2E_COMMON_DIR/data_profiles.sh"
read_latest_step() {
  local model_home="$1"
  local latest_file="$model_home/latest"
  local tag step

  if [ ! -f "$latest_file" ]; then
    echo "Missing latest checkpoint marker: $latest_file" >&2
    return 1
  fi

  tag="$(tr -d '[:space:]' < "$latest_file")"
  case "$tag" in
    step*) step="${tag#step}" ;;
    global_step*) step="${tag#global_step}" ;;
    *) step="$tag" ;;
  esac

  [[ "$step" =~ ^[0-9]+$ ]] || {
    echo "Cannot parse checkpoint step from $latest_file: $tag" >&2
    return 1
  }
  printf '%s\n' "$step"
}

convert_e2e_checkpoint() {
  local base_model_dir="$1"
  local model_home="$2"
  local step="$3"
  local bundle_remote_code="$4"
  local conversion_env=("BUNDLE_REMOTE_CODE=$bundle_remote_code" "HOULSBY_ADAPTER=${HOULSBY_ADAPTER:-false}" "HOULSBY_ADAPTER_BOTTLENECK=${HOULSBY_ADAPTER_BOTTLENECK:-256}")

  if [ "${E2E_MERGE_LORA:-false}" = "true" ]; then
    conversion_env+=("MERGE_LORA=1" "LORA_RANK=$LORA_RANK" "LORA_ALPHA=$LORA_ALPHA")
  fi

  env "${conversion_env[@]}" \
    bash "$E2E_COMMON_PRETRAIN_DIR/scripts/convert_checkpoint_to_hf.sh" \
      "$base_model_dir" "$model_home" "$step"
}

write_e2e_summary() {
  mkdir -p "$(dirname "$SUMMARY_FILE")"
  {
    echo "run_id: $RUN_ID"
    echo "data_profile: $DATA_PROFILE"
    echo "method_run_tag: $E2E_RUN_TAG"
    echo "finetune_mode: $FINETUNE_MODE"
    echo "adapter_type: ${ADAPTER_TYPE:-none}"
    echo "adapter_itm_mode: ${ADAPTER_ITM_MODE:-default}"
    echo "adapter_itm_aux_loss_weight: ${ADAPTER_ITM_AUX_LOSS_WEIGHT:-inherit}"
    echo "adapter_itm_pattern_file: ${ADAPTER_ITM_PATTERN_FILE:-}"
    echo "adapter_itm_pattern_min_frequency: ${ADAPTER_ITM_PATTERN_MIN_FREQUENCY:-}"
    echo "adapter_memory_size: ${ADAPTER_MEMORY_SIZE:-}"
    echo "adapter_spm_memory_size: ${ADAPTER_SPM_MEMORY_SIZE:-${ADAPTER_MEMORY_SIZE:-}}"
    echo "adapter_itm_memory_size: ${ADAPTER_ITM_MEMORY_SIZE:-${ADAPTER_MEMORY_SIZE:-}}"
    echo "adapter_memory_init: ${ADAPTER_MEMORY_INIT:-default}"
    echo "adapter_memory_init_std: ${ADAPTER_MEMORY_INIT_STD:-0.01}"
    echo "adapter_hidden_size: ${ADAPTER_HIDDEN_SIZE:-}"
    echo "adapter_spm_hidden_size: ${ADAPTER_SPM_HIDDEN_SIZE:-${ADAPTER_HIDDEN_SIZE:-model_hidden_size}}"
    echo "adapter_itm_hidden_size: ${ADAPTER_ITM_HIDDEN_SIZE:-${ADAPTER_HIDDEN_SIZE:-model_hidden_size}}"
    echo "houlsby_adapter_bottleneck: ${HOULSBY_ADAPTER_BOTTLENECK:-}"
    echo "lora_rank: ${LORA_RANK:-}"
    echo "lora_alpha: ${LORA_ALPHA:-}"
    echo "partial_train_layers: ${PARTIAL_TRAIN_LAYERS:-}"
    echo "stage1_source_file_list: $STAGE1_SOURCE_FILE_LIST"
    echo "stage2_source_file_list: $STAGE2_SOURCE_FILE_LIST"
    echo "sft_source_file_list: $SFT_SOURCE_FILE_LIST"
    echo "stage1_converted_model_dir: $STAGE1_CONVERTED_MODEL_DIR"
    echo "stage2_output_dir: $STAGE2_OUTPUT_DIR"
    echo "stage2_step: ${STAGE2_STEP:-}"
    echo "stage2_converted_model_dir: ${STAGE2_CONVERTED_MODEL_DIR:-}"
    echo "sft_output_dir: $SFT_OUTPUT_DIR"
    echo "sft_step: ${SFT_STEP:-}"
    echo "sft_converted_model_dir: ${SFT_CONVERTED_MODEL_DIR:-}"
  } > "$SUMMARY_FILE"
}

run_e2e_pipeline() {
  apply_data_profile
  : "${STAGE1_CONVERTED_MODEL_DIR:?Set STAGE1_CONVERTED_MODEL_DIR to the Stage1 converted HF checkpoint.}"

  export SEED="${SEED:-19260817}"
  local seed_suffix=""
  if [ "$SEED" != "19260817" ]; then seed_suffix="_seed${SEED}"; fi
  RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
  STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-$E2E_COMMON_REPO_ROOT/model_output/e2e_stage2_pretrain_${DATASET_TAG}_${E2E_RUN_TAG}${seed_suffix}_${RUN_ID}}"
  SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$E2E_COMMON_REPO_ROOT/model_output/e2e_sft_${DATASET_TAG}_${E2E_RUN_TAG}${seed_suffix}_${RUN_ID}}"
  SUMMARY_FILE="${SUMMARY_FILE:-$SFT_OUTPUT_DIR/end_to_end_summary.log}"

  echo "[1/4] Stage2 $FINETUNE_MODE pretrain -> $STAGE2_OUTPUT_DIR"
  MODEL_DIR="$STAGE1_CONVERTED_MODEL_DIR" \
  OUTPUT_DIR="$STAGE2_OUTPUT_DIR" \
  RUN_ID="$RUN_ID" \
  DATA_PROFILE="$DATA_PROFILE" \
    bash "$E2E_COMMON_PRETRAIN_DIR/scripts/train_stage2_pretrain.sh"

  STAGE2_STEP="$(read_latest_step "$STAGE2_OUTPUT_DIR")"
  STAGE2_CONVERTED_MODEL_DIR="$STAGE2_OUTPUT_DIR/step${STAGE2_STEP}/global_step${STAGE2_STEP}/converted"
  echo "[2/4] Convert Stage2 checkpoint step $STAGE2_STEP -> $STAGE2_CONVERTED_MODEL_DIR"
  convert_e2e_checkpoint "$STAGE1_CONVERTED_MODEL_DIR" "$STAGE2_OUTPUT_DIR" "$STAGE2_STEP" "$E2E_STAGE2_BUNDLE_REMOTE_CODE"
  [ -d "$STAGE2_CONVERTED_MODEL_DIR" ] || {
    echo "Stage2 converted model not found: $STAGE2_CONVERTED_MODEL_DIR" >&2
    return 1
  }

  echo "[3/4] SFT $FINETUNE_MODE train -> $SFT_OUTPUT_DIR"
  MODEL_DIR="$STAGE2_CONVERTED_MODEL_DIR" \
  STAGE2_CONVERTED_MODEL_DIR="$STAGE2_CONVERTED_MODEL_DIR" \
  OUTPUT_DIR="$SFT_OUTPUT_DIR" \
  RUN_ID="$RUN_ID" \
  DATA_PROFILE="$DATA_PROFILE" \
    bash "$E2E_COMMON_PRETRAIN_DIR/scripts/train_sft_video.sh"

  SFT_STEP="$(read_latest_step "$SFT_OUTPUT_DIR")"
  SFT_CONVERTED_MODEL_DIR="$SFT_OUTPUT_DIR/step${SFT_STEP}/global_step${SFT_STEP}/converted"
  echo "[4/4] Convert SFT checkpoint step $SFT_STEP -> $SFT_CONVERTED_MODEL_DIR"
  convert_e2e_checkpoint "$STAGE2_CONVERTED_MODEL_DIR" "$SFT_OUTPUT_DIR" "$SFT_STEP" "$E2E_SFT_BUNDLE_REMOTE_CODE"
  [ -d "$SFT_CONVERTED_MODEL_DIR" ] || {
    echo "SFT converted model not found: $SFT_CONVERTED_MODEL_DIR" >&2
    return 1
  }

  write_e2e_summary
  echo "Final converted model: $SFT_CONVERTED_MODEL_DIR"
  echo "Run recommendation and general-capability evaluations separately with the commands in benchmarks/lm_eval/README.md."
  echo "Summary: $SUMMARY_FILE"
}
