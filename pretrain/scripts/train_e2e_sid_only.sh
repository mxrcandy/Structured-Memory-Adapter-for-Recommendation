#!/usr/bin/env bash
set -euo pipefail
PRETRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PRETRAIN_DIR/scripts/e2e_common.sh"

export FINETUNE_MODE="${FINETUNE_MODE:-sid_only}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + PORT_SEED % 20000))}"
export E2E_RUN_TAG="${E2E_RUN_TAG:-sid_only}"
export E2E_MERGE_LORA=false
export E2E_STAGE2_BUNDLE_REMOTE_CODE=0
export E2E_SFT_BUNDLE_REMOTE_CODE=0
run_e2e_pipeline
