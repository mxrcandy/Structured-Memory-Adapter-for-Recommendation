#!/usr/bin/env bash
set -euo pipefail
PRETRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PRETRAIN_DIR/scripts/e2e_common.sh"

export FINETUNE_MODE="${FINETUNE_MODE:-houlsby_adapter}"
export HOULSBY_ADAPTER_BOTTLENECK="${HOULSBY_ADAPTER_BOTTLENECK:-256}"
export HOULSBY_ADAPTER=true
PORT_SEED="${SLURM_JOB_ID:-$$}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + PORT_SEED % 20000))}"
export E2E_RUN_TAG="${E2E_RUN_TAG:-houlsby_adapter_b${HOULSBY_ADAPTER_BOTTLENECK}}"
export E2E_MERGE_LORA=false
# SFT constructs the repository's Qwen3 class directly; only final inference
# needs the self-contained remote-code bundle.
export E2E_STAGE2_BUNDLE_REMOTE_CODE=0
export E2E_SFT_BUNDLE_REMOTE_CODE=1
run_e2e_pipeline
