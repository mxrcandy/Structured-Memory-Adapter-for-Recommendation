#!/usr/bin/env bash
set -euo pipefail
PRETRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PRETRAIN_DIR/scripts/e2e_common.sh"

export FINETUNE_MODE="${FINETUNE_MODE:-lora}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + PORT_SEED % 20000))}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
export LORA_LAYERS="${LORA_LAYERS:-}"
export E2E_RUN_TAG="${E2E_RUN_TAG:-${LORA_RUN_TAG:-lora_${LORA_TARGET_MODULES//,/-}_r${LORA_RANK}_a${LORA_ALPHA}_drop${LORA_DROPOUT}}}"
export E2E_MERGE_LORA=true
export E2E_STAGE2_BUNDLE_REMOTE_CODE=0
export E2E_SFT_BUNDLE_REMOTE_CODE=0
run_e2e_pipeline
