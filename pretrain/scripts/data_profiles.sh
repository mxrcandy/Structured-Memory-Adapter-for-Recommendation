#!/usr/bin/env bash

# Data profiles centralize the three stage-specific shard lists. Explicit
# variables from the calling shell always win over profile defaults.
DATA_PROFILE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PROFILE_REPO_ROOT="$(cd "$DATA_PROFILE_SCRIPT_DIR/../.." && pwd)"

apply_data_profile() {
  DATA_PROFILE="${DATA_PROFILE:-10pct}"

  case "$DATA_PROFILE" in
    amazon_letter_beauty|amazon_letter_beauty_general1_rec8|amazon_letter_beauty_general1_rec16|amazon_letter_instruments|amazon_letter_instruments_general1_rec16|amazon_letter_toys|amazon_letter_sports|amazon_letter_yelp|recif_10pct|1pct|10pct|full|all_item_understand|mixed_1pct_general2_rec1|mixed_10pct_general1_rec4|mixed_10pct_general2_rec1) ;;
    *)
      echo "Unknown DATA_PROFILE: $DATA_PROFILE (expected amazon_letter_beauty, amazon_letter_beauty_general1_rec8, amazon_letter_beauty_general1_rec16, amazon_letter_instruments, amazon_letter_instruments_general1_rec16, amazon_letter_yelp, amazon_letter_toys, amazon_letter_sports, recif_10pct, 1pct, 10pct, full, all_item_understand, mixed_1pct_general2_rec1, mixed_10pct_general1_rec4, or mixed_10pct_general2_rec1)" >&2
      return 1
      ;;
  esac

  DATASET_TAG="${DATASET_TAG:-$DATA_PROFILE}"
  local data_root
  case "$DATA_PROFILE" in
    amazon_letter_beauty|amazon_letter_beauty_general1_rec8|amazon_letter_beauty_general1_rec16|amazon_letter_instruments|amazon_letter_instruments_general1_rec16|amazon_letter_toys|amazon_letter_sports|amazon_letter_yelp)
      data_root="$DATA_PROFILE_REPO_ROOT/process_data/output_${DATA_PROFILE}"
      MAX_LENGTH="${MAX_LENGTH:-2048}"
      # Qwen3-0.6B: a/b/c/d (256 each), then sid_begin and sid_end.
      ITEMIC_START_INDEX="${ITEMIC_START_INDEX:-151669}"
      ITEMIC_END_INDEX="${ITEMIC_END_INDEX:-152694}"
      export MAX_LENGTH ITEMIC_START_INDEX ITEMIC_END_INDEX
      ;;
    recif_10pct) data_root="$DATA_PROFILE_REPO_ROOT/process_data/output_recif_10pct" ;;
    1pct|10pct|full) data_root="$DATA_PROFILE_REPO_ROOT/process_data/output_${DATA_PROFILE}" ;;
    all_item_understand) data_root="$DATA_PROFILE_REPO_ROOT/process_data/output_all_item_understand" ;;
    mixed_1pct_general2_rec1) data_root="$DATA_PROFILE_REPO_ROOT/process_data/output_1pct_general2_rec1" ;;
    mixed_10pct_general1_rec4) data_root="$DATA_PROFILE_REPO_ROOT/process_data/output_10pct_general1_rec4" ;;
    mixed_10pct_general2_rec1) data_root="$DATA_PROFILE_REPO_ROOT/process_data/output_10pct_general2_rec1" ;;
  esac

  STAGE1_SOURCE_FILE_LIST="${STAGE1_SOURCE_FILE_LIST:-$data_root/split_data_pretrain_item_understand/file_list.json}"
  if [ "$DATA_PROFILE" = "all_item_understand" ]; then
    STAGE2_SOURCE_FILE_LIST="${STAGE2_SOURCE_FILE_LIST-}"
    SFT_SOURCE_FILE_LIST="${SFT_SOURCE_FILE_LIST-}"
  else
    STAGE2_SOURCE_FILE_LIST="${STAGE2_SOURCE_FILE_LIST:-$data_root/split_data_pretrain/file_list.json}"
    SFT_SOURCE_FILE_LIST="${SFT_SOURCE_FILE_LIST:-$data_root/split_data_sft/file_list.json}"
  fi

  export DATA_PROFILE DATASET_TAG STAGE1_SOURCE_FILE_LIST STAGE2_SOURCE_FILE_LIST SFT_SOURCE_FILE_LIST
}

resolve_stage_source_file_list() {
  local stage="$1"

  if [ -n "${SOURCE_FILE_LIST:-}" ]; then
    printf '%s\n' "$SOURCE_FILE_LIST"
    return 0
  fi

  case "$stage" in
    stage1) printf '%s\n' "$STAGE1_SOURCE_FILE_LIST" ;;
    stage2) printf '%s\n' "$STAGE2_SOURCE_FILE_LIST" ;;
    sft) printf '%s\n' "$SFT_SOURCE_FILE_LIST" ;;
    *)
      echo "Unknown training stage: $stage" >&2
      return 1
      ;;
  esac
}
