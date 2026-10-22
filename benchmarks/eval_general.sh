#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
RESULT_NAME="${RESULT_NAME:-}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
ADAPTER_ENABLED="${ADAPTER_ENABLED:-true}"
GENERAL_EVAL_SUITE="${GENERAL_EVAL_SUITE:-stage4_core}"
# Shared generation budget for all models; full evaluation unless SAMPLE_SIZE is set.
GENERAL_MAX_NEW_TOKENS="${GENERAL_MAX_NEW_TOKENS:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
# Parallel execution modes:
#   data   (default): one single-GPU model replica per DATA_PARALLEL_SIZE;
#                    recommendation VllmWorkers split the same task's samples.
#   tensor:           one model split across TENSOR_PARALLEL_SIZE GPUs.
GENERAL_PARALLEL_MODE="${GENERAL_PARALLEL_MODE:-data}"
GENERAL_NUM_WORKERS="${GENERAL_NUM_WORKERS:-}"
GENERAL_GPU_IDS="${GENERAL_GPU_IDS:-}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-4}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/openonerec_general_ray_${SLURM_JOB_ID:-$$}}"
if [[ -z "${LM_EVAL_BIN:-}" && "$PYTHON_BIN" == */python* ]]; then
  LM_EVAL_BIN="$(dirname "$PYTHON_BIN")/lm_eval"
fi
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
export LM_EVAL_BIN
if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then echo "Model path not found: ${MODEL_PATH:-<empty>}" >&2; exit 1; fi
if [[ "$ENABLE_THINKING" != true && "$ENABLE_THINKING" != false ]]; then echo "ENABLE_THINKING must be true or false" >&2; exit 1; fi
if [[ "$ADAPTER_ENABLED" != true && "$ADAPTER_ENABLED" != false ]]; then echo "ADAPTER_ENABLED must be true or false" >&2; exit 1; fi
DATA_VERSION="${DATA_VERSION:-v1.0}"
MODEL_PARENT_DIR="$(cd "$MODEL_PATH/.." && pwd)"
MODEL_PARENT_BASENAME="$(basename "$MODEL_PARENT_DIR")"
if [ "$(basename "$MODEL_PATH")" = "converted" ] && [[ "$MODEL_PARENT_BASENAME" == global_step* ]]; then
  TRAIN_RUN_DIR="$(cd "$MODEL_PATH/../../.." && pwd)"
else
  TRAIN_RUN_DIR="${GENERAL_BENCHMARK_TRAIN_RUN_DIR:-$MODEL_PARENT_DIR}"
fi
if [ -z "$RESULT_NAME" ]; then
  TRAIN_RUN_BASENAME="$(basename "$TRAIN_RUN_DIR")"
  RESULT_BASE_NAME="$TRAIN_RUN_BASENAME"
  if [[ "$RESULT_BASE_NAME" =~ ^(.+)_[0-9]{8}_[0-9]{6}$ ]]; then
    RESULT_BASE_NAME="${BASH_REMATCH[1]}"
  fi
  RESULT_NAME="$RESULT_BASE_NAME"
  if [[ "$(basename "$MODEL_PATH")" != converted ]]; then RESULT_NAME="$(basename "$MODEL_PATH")"; fi
fi
if [[ "$RESULT_NAME" != *_general_* ]]; then
  RESULT_NAME="${RESULT_NAME}_general_${GENERAL_EVAL_SUITE}"
fi
if [ "$ADAPTER_ENABLED" != "true" ] && [[ "$RESULT_NAME" != *_adapter_off ]]; then
  RESULT_NAME="${RESULT_NAME}_adapter_off"
fi
OUTPUT_ROOT="${GENERAL_BENCHMARK_OUTPUT_ROOT:-$TRAIN_RUN_DIR/benchmark_results}"
LOG_ROOT="${GENERAL_BENCHMARK_LOG_ROOT:-$TRAIN_RUN_DIR/benchmark_logs}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${DATA_VERSION}/results_${RESULT_NAME}}"
BASE_LOG_NAME="${LOG_NAME:-${LOG_ROOT}/${DATA_VERSION}/${RESULT_NAME}}"
DATA_DIR="${GENERAL_BENCHMARK_DATA_DIR:-$REPO_ROOT/raw_data/general_data/benchmark_data}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-${BASE_LOG_NAME}.gpu.log}"
GPU_MONITOR_PID=""
MODEL_CACHE_KEY="$(printf '%s' "$MODEL_PATH" | sha256sum | cut -c1-16)"
mkdir -p "$BASE_OUTPUT_DIR" "$(dirname "$BASE_LOG_NAME")" "$DATA_DIR/hf_cache"
export HF_HOME="${HF_HOME:-$DATA_DIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/openonerec_hf_modules_${MODEL_CACHE_KEY}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENONEREC_ADAPTER_ENABLED="$ADAPTER_ENABLED"
export TENSOR_PARALLEL_SIZE
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export RAY_TEMP_DIR
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# Do not accidentally attach to a stale Ray head inherited from another job.
unset RAY_ADDRESS
# prepare_data.py stores the official lm-eval datasets under this cache.
# lm-eval reads HF_DATASETS_CACHE, so no network access is needed when the
# cache has been prepared for the selected suite.

start_gpu_monitor() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while true; do
        date
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
        sleep "$GPU_MONITOR_INTERVAL"
      done
    ) > "$GPU_MONITOR_LOG" 2>&1 &
    GPU_MONITOR_PID="$!"
  fi
}

stop_gpu_monitor() {
  if [ -n "$GPU_MONITOR_PID" ]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  stop_gpu_monitor
  if [[ "$GENERAL_PARALLEL_MODE" == data && -d "$RAY_TEMP_DIR/session_latest/logs" ]]; then
    mkdir -p "${BASE_LOG_NAME}.ray_logs"
    cp -a "$RAY_TEMP_DIR/session_latest/logs/." "${BASE_LOG_NAME}.ray_logs/" || true
  fi
}
trap cleanup EXIT

EFFECTIVE_TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE"
if [[ "$GENERAL_PARALLEL_MODE" == data || "$GENERAL_PARALLEL_MODE" == task ]]; then
  EFFECTIVE_TENSOR_PARALLEL_SIZE=1
fi
{ echo "time: $(date)"; echo "model_path: $MODEL_PATH"; echo "result_name: $RESULT_NAME"; echo "data_version: $DATA_VERSION"; echo "suite: $GENERAL_EVAL_SUITE"; echo "adapter_enabled: $ADAPTER_ENABLED"; echo "enable_thinking: $ENABLE_THINKING"; echo "output_dir: $BASE_OUTPUT_DIR"; echo "data_dir: $DATA_DIR"; echo "python_bin: $PYTHON_BIN"; echo "lm_eval_bin: $LM_EVAL_BIN"; echo "parallel_mode: $GENERAL_PARALLEL_MODE"; echo "data_parallel_size: $DATA_PARALLEL_SIZE"; echo "num_workers: ${GENERAL_NUM_WORKERS:-auto}"; echo "gpu_ids: ${GENERAL_GPU_IDS:-auto}"; echo "tensor_parallel_size: $EFFECTIVE_TENSOR_PARALLEL_SIZE"; } | tee "${BASE_LOG_NAME}.log"
start_gpu_monitor
ARGS=(-m benchmarks.lm_eval.wrapper --model-path "$MODEL_PATH" --suite "$GENERAL_EVAL_SUITE" --output-dir "$BASE_OUTPUT_DIR" --adapter-enabled "$ADAPTER_ENABLED" --parallel-mode "$GENERAL_PARALLEL_MODE" --data-parallel-size "$DATA_PARALLEL_SIZE")
ARGS+=(--max-new-tokens "$GENERAL_MAX_NEW_TOKENS")
echo "max_new_tokens: $GENERAL_MAX_NEW_TOKENS" | tee -a "${BASE_LOG_NAME}.log"
if [[ -n "$GENERAL_NUM_WORKERS" ]]; then ARGS+=(--num-workers "$GENERAL_NUM_WORKERS"); fi
if [[ -n "$GENERAL_GPU_IDS" ]]; then ARGS+=(--gpu-ids "$GENERAL_GPU_IDS"); fi
if [[ "$ENABLE_THINKING" == true ]]; then ARGS+=(--enable-thinking); fi
if [[ -n "${SAMPLE_SIZE:-}" ]]; then ARGS+=(--limit "$SAMPLE_SIZE"); fi
"$PYTHON_BIN" "${ARGS[@]}" 2>&1 | tee -a "${BASE_LOG_NAME}.log"
