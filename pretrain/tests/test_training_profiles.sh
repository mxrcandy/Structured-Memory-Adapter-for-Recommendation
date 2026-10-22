#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE_LIB="$REPO_ROOT/pretrain/scripts/data_profiles.sh"
E2E_ADAPTER="$REPO_ROOT/pretrain/scripts/train_e2e_adapter.sh"
E2E_ADAPTER_FULL="$REPO_ROOT/pretrain/scripts/train_e2e_adapter_full.sh"
E2E_SID_ONLY="$REPO_ROOT/pretrain/scripts/train_e2e_sid_only.sh"

fail() {
  echo "test_training_profiles: $*" >&2
  exit 1
}

assert_equals() {
  local expected="$1"
  local actual="$2"
  [ "$expected" = "$actual" ] || fail "expected '$expected', got '$actual'"
}

[ -f "$PROFILE_LIB" ] || fail "missing data profile library: $PROFILE_LIB"
[ -f "$E2E_ADAPTER" ] || fail "missing adapter E2E entrypoint: $E2E_ADAPTER"
[ -f "$E2E_ADAPTER_FULL" ] || fail "missing adapter-full E2E entrypoint: $E2E_ADAPTER_FULL"
[ -f "$E2E_SID_ONLY" ] || fail "missing sid-only E2E entrypoint: $E2E_SID_ONLY"
[ ! -d "$REPO_ROOT/pretrain/scripts/lib" ] || fail "obsolete lib directory still exists"

full_profile="$(
  DATA_PROFILE=full bash -c '
    source "$1"
    apply_data_profile
    printf "%s|%s|%s|%s\n" "$DATASET_TAG" "$STAGE1_SOURCE_FILE_LIST" "$STAGE2_SOURCE_FILE_LIST" "$SFT_SOURCE_FILE_LIST"
  ' _ "$PROFILE_LIB"
)"
assert_equals \
  "full|$REPO_ROOT/process_data/output_full/split_data_pretrain_item_understand/file_list.json|$REPO_ROOT/process_data/output_full/split_data_pretrain/file_list.json|$REPO_ROOT/process_data/output_full/split_data_sft/file_list.json" \
  "$full_profile"

override_source="$(
  DATA_PROFILE=full SOURCE_FILE_LIST=/tmp/custom-file-list.json bash -c '
    source "$1"
    apply_data_profile
    resolve_stage_source_file_list stage2
  ' _ "$PROFILE_LIB"
)"
assert_equals "/tmp/custom-file-list.json" "$override_source"

if DATA_PROFILE=unknown bash -c 'source "$1"; apply_data_profile' _ "$PROFILE_LIB" >/dev/null 2>&1; then
  fail "unknown data profile unexpectedly succeeded"
fi

for retired_path in \
  "$REPO_ROOT/pretrain/scripts/train_stage2_route.sh" \
  "$REPO_ROOT/pretrain/scripts/train_sft_video_route.sh"; do
  [ ! -e "$retired_path" ] || fail "retired route dispatcher still exists: $retired_path"
done

if rg -n '\bROUTE\b|ROUTE_PRESET|configure_training_route|E2E_LABEL' \
  "$REPO_ROOT/pretrain/scripts/e2e_common.sh" \
  "$REPO_ROOT/pretrain/scripts/train_e2e_"*.sh \
  "$REPO_ROOT/pretrain/scripts/slurm_train.sbatch"; then
  fail "active launchers still depend on route dispatch"
fi

rg -Fq 'E2E_ENTRY' "$REPO_ROOT/pretrain/scripts/slurm_e2e.sbatch" || fail "Slurm E2E launcher does not select an explicit E2E entrypoint"

assert_e2e_port() {
  local script="$1"
  local port="$2"
  rg -Fq "MASTER_PORT=\"\${MASTER_PORT:-$port}\"" "$REPO_ROOT/pretrain/scripts/$script" || fail "missing E2E default MASTER_PORT=$port in $script"
}

assert_e2e_port train_e2e_full.sh 29625
assert_e2e_port train_e2e_adapter.sh 29624
assert_e2e_port train_e2e_lora.sh 29627
assert_e2e_port train_e2e_adapter_lora.sh 29628
assert_e2e_port train_e2e_adapter_partial.sh 29629
assert_e2e_port train_e2e_adapter_full.sh 29630
assert_e2e_port train_e2e_sid_only.sh 29631
rg -Fq 'train_e2e_houlsby_adapter.sh' "$REPO_ROOT/pretrain/scripts/slurm_e2e.sbatch" || fail "standard Slurm E2E launcher does not allow Houlsby Adapter"
rg -Fq 'train_e2e_houlsby_adapter.sh' "$REPO_ROOT/pretrain/scripts/slurm_e2e_amazon.sbatch" || fail "Amazon Slurm E2E launcher does not allow Houlsby Adapter"
rg -Fq 'E2E_SFT_BUNDLE_REMOTE_CODE=1' "$REPO_ROOT/pretrain/scripts/train_e2e_houlsby_adapter.sh" || fail "Houlsby final conversion must bundle remote code"
rg -Fq 'HOULSBY_ADAPTER_BOTTLENECK' "$REPO_ROOT/pretrain/scripts/train_e2e_houlsby_adapter.sh" || fail "Houlsby entrypoint does not expose bottleneck width"

rg -Fq 'FINETUNE_MODE="${FINETUNE_MODE:-adapter_full}"' "$E2E_ADAPTER_FULL" || fail "adapter-full E2E entrypoint has the wrong default mode"
rg -Fq 'FINETUNE_MODE="${FINETUNE_MODE:-sid_only}"' "$E2E_SID_ONLY" || fail "sid-only E2E entrypoint has the wrong default mode"
for script in \
  "$E2E_ADAPTER" \
  "$REPO_ROOT/pretrain/scripts/train_e2e_adapter_lora.sh" \
  "$REPO_ROOT/pretrain/scripts/train_e2e_adapter_partial.sh" \
  "$E2E_ADAPTER_FULL"; do
  rg -Fq 'E2E_STAGE2_BUNDLE_REMOTE_CODE=0' "$script" || fail "Engram Stage 2 conversion should not bundle remote code: $script"
  rg -Fq 'E2E_SFT_BUNDLE_REMOTE_CODE=1' "$script" || fail "Engram final conversion must bundle remote code: $script"
done
for script in \
  "$REPO_ROOT/pretrain/scripts/train_stage2_pretrain.sh" \
  "$REPO_ROOT/pretrain/scripts/train_sft_video.sh"; do
  rg -Fq '"adapter_full"' "$script" || fail "adapter_full is not wired into $script"
done

if rg -n 'train_(stage2_pretrain|sft_video)_10pct\.sh|output_10pct' \
  "$REPO_ROOT/pretrain/scripts/train_stage2_pretrain.sh" \
  "$REPO_ROOT/pretrain/scripts/train_sft_video.sh" \
  "$REPO_ROOT/pretrain/scripts/smoke_pretrain.sh" \
  "$REPO_ROOT/pretrain/scripts/smoke_sft.sh"; then
  fail "active training scripts still hard-code the 10pct workflow"
fi

if STAGE1_CONVERTED_MODEL_DIR= bash "$E2E_ADAPTER" >/tmp/test_training_profiles_e2e.out 2>&1; then
  fail "E2E adapter entrypoint unexpectedly accepted an empty Stage 1 checkpoint"
fi
rg -Fq 'STAGE1_CONVERTED_MODEL_DIR' /tmp/test_training_profiles_e2e.out || fail "E2E missing-checkpoint error is unclear"

echo "test_training_profiles: PASS"
