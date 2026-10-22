#!/usr/bin/env bash
set -euo pipefail

# Structured SPM+ITM baseline. All values remain overridable for ablations.
PRETRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PRETRAIN_DIR/scripts/e2e_common.sh"

export FINETUNE_MODE="${FINETUNE_MODE:-adapter}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + PORT_SEED % 20000))}"
export ADAPTER_TYPE="${ADAPTER_TYPE:-engram_structure}"
export ADAPTER_LAYERS="${ADAPTER_LAYERS:-1,14}"
export ADAPTER_HIDDEN_SIZE="${ADAPTER_HIDDEN_SIZE:-128}"
export ADAPTER_SPM_HIDDEN_SIZE="${ADAPTER_SPM_HIDDEN_SIZE:-}"
export ADAPTER_ITM_HIDDEN_SIZE="${ADAPTER_ITM_HIDDEN_SIZE:-}"
export ADAPTER_MEMORY_SIZE="${ADAPTER_MEMORY_SIZE:-65521}"
export ADAPTER_SPM_MEMORY_SIZE="${ADAPTER_SPM_MEMORY_SIZE:-}"
export ADAPTER_ITM_MEMORY_SIZE="${ADAPTER_ITM_MEMORY_SIZE:-}"
export ADAPTER_MEMORY_INIT="${ADAPTER_MEMORY_INIT:-default}"
export ADAPTER_MEMORY_INIT_STD="${ADAPTER_MEMORY_INIT_STD:-0.01}"
export ADAPTER_MEMORY_PROTOTYPE_PATH="${ADAPTER_MEMORY_PROTOTYPE_PATH:-$PRETRAIN_DIR/../model_output/prototypes_recif_videorec_pretrain_10pct/engram_structure/engram_structure_proto_tokenemb_randomproj2048to128_seed42_spm2-3_itm1-1_heads4_mem65521.pt}"
export ADAPTER_NGRAM_ORDER="${ADAPTER_NGRAM_ORDER:-3}"
export ADAPTER_TOPK="${ADAPTER_TOPK:-4}"
export ADAPTER_LNGRAM_BITS_PER_ROUTE="${ADAPTER_LNGRAM_BITS_PER_ROUTE:-4}"
export ADAPTER_USE_SPM="${ADAPTER_USE_SPM:-true}"
export ADAPTER_USE_ITM="${ADAPTER_USE_ITM:-true}"
export ADAPTER_ITM_ORDER="${ADAPTER_ITM_ORDER:-1}"
export ADAPTER_ITM_MODE="${ADAPTER_ITM_MODE:-default}"
# Empty inherits the input checkpoint (old checkpoints use zero).
export ADAPTER_ITM_AUX_LOSS_WEIGHT="${ADAPTER_ITM_AUX_LOSS_WEIGHT:-}"
export ADAPTER_ITM_PATTERN_FILE="${ADAPTER_ITM_PATTERN_FILE:-}"
export ADAPTER_ITM_PATTERN_MIN_FREQUENCY="${ADAPTER_ITM_PATTERN_MIN_FREQUENCY:-5}"
# Naming only: these labels do not select/filter patterns or configure the model.
# Example: ADAPTER_ITM_PATTERN_TAG=ab2a ADAPTER_ITM_CONCENTRATION_TAG=03
# The actual mapping is selected exclusively by ADAPTER_ITM_PATTERN_FILE.
export ADAPTER_ITM_PATTERN_TAG="${ADAPTER_ITM_PATTERN_TAG:-}"
export ADAPTER_ITM_CONCENTRATION_TAG="${ADAPTER_ITM_CONCENTRATION_TAG:-}"
export ADAPTER_USE_ALPHA_LAYER="${ADAPTER_USE_ALPHA_LAYER:-false}"
export ADAPTER_ALPHA_INIT="${ADAPTER_ALPHA_INIT:-0.1}"
# Empty means dynamic context gate; pass a numeric value to use a fixed gate.
export ADAPTER_FIXED_CONTEXT_GATE="${ADAPTER_FIXED_CONTEXT_GATE:-}"
export ADAPTER_DISABLE_CONV=true
export ADAPTER_USE_SID_SPAN=false
E2E_GATE_TAG="${ADAPTER_FIXED_CONTEXT_GATE:-dynamic}"
if [ "$ADAPTER_MEMORY_INIT" = "default" ]; then E2E_INIT_TAG=""; else E2E_INIT_TAG="_init${ADAPTER_MEMORY_INIT}"; fi
E2E_HIDDEN_TAG="d${ADAPTER_HIDDEN_SIZE}"
if [ -n "$ADAPTER_SPM_HIDDEN_SIZE" ] || [ -n "$ADAPTER_ITM_HIDDEN_SIZE" ]; then
  E2E_HIDDEN_TAG="spmd${ADAPTER_SPM_HIDDEN_SIZE:-$ADAPTER_HIDDEN_SIZE}_itmd${ADAPTER_ITM_HIDDEN_SIZE:-$ADAPTER_HIDDEN_SIZE}"
fi
if [[ "$ADAPTER_TYPE" == "lngram_structure" ]]; then
  E2E_DEFAULT_TAG="${ADAPTER_TYPE}_layers${ADAPTER_LAYERS//,/-}_spm${ADAPTER_USE_SPM}_itm${ADAPTER_USE_ITM}_order${ADAPTER_ITM_ORDER}_bits${ADAPTER_LNGRAM_BITS_PER_ROUTE:-4}_${E2E_HIDDEN_TAG}${E2E_INIT_TAG}_gate${E2E_GATE_TAG}"
else
  E2E_MEMORY_TAG="mem${ADAPTER_MEMORY_SIZE}"
  if [ -n "$ADAPTER_SPM_MEMORY_SIZE" ] || [ -n "$ADAPTER_ITM_MEMORY_SIZE" ]; then
    E2E_MEMORY_TAG="spmmem${ADAPTER_SPM_MEMORY_SIZE:-$ADAPTER_MEMORY_SIZE}_itmmem${ADAPTER_ITM_MEMORY_SIZE:-$ADAPTER_MEMORY_SIZE}"
  fi
  E2E_DEFAULT_TAG="${ADAPTER_TYPE}_layers${ADAPTER_LAYERS//,/-}_spm${ADAPTER_USE_SPM}_itm${ADAPTER_USE_ITM}_order${ADAPTER_ITM_ORDER}_heads${ADAPTER_TOPK}_${E2E_MEMORY_TAG}_${E2E_HIDDEN_TAG}${E2E_INIT_TAG}_gate${E2E_GATE_TAG}"
fi
if [ "$ADAPTER_ITM_MODE" = "exact" ]; then
  E2E_DEFAULT_TAG="${E2E_DEFAULT_TAG}_itmexact_min${ADAPTER_ITM_PATTERN_MIN_FREQUENCY}"
elif [ "$ADAPTER_ITM_MODE" = "successor_prior" ]; then
  E2E_DEFAULT_TAG="${E2E_DEFAULT_TAG}_itmsuccessorprior_min${ADAPTER_ITM_PATTERN_MIN_FREQUENCY}"
elif [ "$ADAPTER_ITM_MODE" = "successor_prior_learned" ]; then
  E2E_DEFAULT_TAG="${E2E_DEFAULT_TAG}_itmsuccessorpriorlearned_min${ADAPTER_ITM_PATTERN_MIN_FREQUENCY}"
fi
if [ -n "$ADAPTER_ITM_AUX_LOSS_WEIGHT" ]; then
  E2E_AUX_TAG="$(python - "$ADAPTER_ITM_AUX_LOSS_WEIGHT" "$ADAPTER_ITM_MODE" "$ADAPTER_TYPE" "$ADAPTER_USE_ITM" <<'PY'
import math
import sys
weight = float(sys.argv[1])
if not math.isfinite(weight) or weight < 0:
    raise SystemExit('ADAPTER_ITM_AUX_LOSS_WEIGHT must be finite and nonnegative')
if weight > 0 and sys.argv[2:] != ['exact', 'engram_structure', 'true']:
    raise SystemExit('next-a auxiliary supervision requires engram_structure + exact ITM enabled')
print(f'_auxa{weight:g}' if weight > 0 else '')
PY
)"
  E2E_DEFAULT_TAG="${E2E_DEFAULT_TAG}${E2E_AUX_TAG}"
fi
if [ -n "$ADAPTER_ITM_PATTERN_TAG" ]; then
  E2E_DEFAULT_TAG="${E2E_DEFAULT_TAG}_${ADAPTER_ITM_PATTERN_TAG}"
fi
if [ -n "$ADAPTER_ITM_CONCENTRATION_TAG" ]; then
  E2E_DEFAULT_TAG="${E2E_DEFAULT_TAG}_c${ADAPTER_ITM_CONCENTRATION_TAG}"
fi
# An explicit E2E_RUN_TAG remains a complete override of the default name.
export E2E_RUN_TAG="${E2E_RUN_TAG:-$E2E_DEFAULT_TAG}"
export E2E_MERGE_LORA=false
export E2E_STAGE2_BUNDLE_REMOTE_CODE=0
export E2E_SFT_BUNDLE_REMOTE_CODE=1

run_e2e_pipeline
