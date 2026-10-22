#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

: "${MODEL_PATH:?Set MODEL_PATH to the trained Amazon checkpoint (converted HF directory).}"
# Every export directory is named `converted`; isolate Transformers' dynamic
# module cache by full model path so different checkpoints cannot collide.
MODEL_CACHE_KEY="$(printf '%s' "$MODEL_PATH" | sha256sum | cut -c1-16)"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/openonerec_hf_modules_${MODEL_CACHE_KEY}}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
ADAPTER_ENABLED="${ADAPTER_ENABLED:-true}"
export OPENONEREC_ADAPTER_ENABLED="$ADAPTER_ENABLED"
export BENCHMARK_MEMORY_BACKEND="${BENCHMARK_MEMORY_BACKEND:-auto}"
echo "Requested memory backend: $BENCHMARK_MEMORY_BACKEND (vLLM Transformers backend)"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path not found: $MODEL_PATH" >&2
  echo "Set MODEL_PATH=/path/to/converted before running $0." >&2
  exit 1
fi
if [ "$ADAPTER_ENABLED" != "true" ] && [ -f "$MODEL_PATH/modeling_qwen3.py" ] && ! grep -q "OPENONEREC_ADAPTER_ENABLED" "$MODEL_PATH/modeling_qwen3.py"; then
  echo "ADAPTER_ENABLED=false requested, but checkpoint remote code lacks OPENONEREC_ADAPTER_ENABLED." >&2
  echo "Re-convert the checkpoint with the current model converter before rerunning." >&2
  exit 1
fi

case "${DATA_PROFILE:-}" in
  amazon_letter_instruments|amazon_letter_instruments_general1_rec16) DEFAULT_TASK_TYPES="amazon_instruments" ;;
  amazon_letter_toys) DEFAULT_TASK_TYPES="amazon_toys" ;;
  amazon_letter_sports) DEFAULT_TASK_TYPES="amazon_sports" ;;
  amazon_letter_yelp) DEFAULT_TASK_TYPES="amazon_yelp" ;;
  *) DEFAULT_TASK_TYPES="amazon_beauty" ;;
esac
TASK_TYPES="${TASK_TYPES:-$DEFAULT_TASK_TYPES}"
DATA_VERSION="${DATA_VERSION:-v1.0}"
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-.}"
case "$TASK_TYPES" in
  amazon_instruments) DEFAULT_BENCHMARK_DATA_DIR="$REPO_ROOT/process_data/output_amazon_letter_instruments/benchmark_data" ;;
  amazon_toys) DEFAULT_BENCHMARK_DATA_DIR="$REPO_ROOT/process_data/output_amazon_letter_toys/benchmark_data" ;;
  amazon_sports) DEFAULT_BENCHMARK_DATA_DIR="$REPO_ROOT/process_data/output_amazon_letter_sports/benchmark_data" ;;
  amazon_yelp) DEFAULT_BENCHMARK_DATA_DIR="$REPO_ROOT/process_data/output_amazon_letter_yelp/benchmark_data" ;;
  *) DEFAULT_BENCHMARK_DATA_DIR="$REPO_ROOT/process_data/output_amazon_letter_beauty/benchmark_data" ;;
esac
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-$DEFAULT_BENCHMARK_DATA_DIR}"
NUM_GPUS="${NUM_GPUS:-4}"
BENCHMARK_RAY_ADDRESS="${BENCHMARK_RAY_ADDRESS:-local}"
# Resolve the sample-size alias once for both generation and result naming.
SAMPLE_SIZE="${SAMPLE_SIZE:-${REC_EVAL_SAMPLE_SIZE:-}}"
EFFECTIVE_SAMPLE_SIZE="$SAMPLE_SIZE"

MODEL_PARENT_DIR="$(cd "$MODEL_PATH/.." && pwd)"
MODEL_PARENT_BASENAME="$(basename "$MODEL_PARENT_DIR")"
if [ "$(basename "$MODEL_PATH")" = "converted" ] && [[ "$MODEL_PARENT_BASENAME" == global_step* ]]; then
  TRAIN_RUN_DIR="$(cd "$MODEL_PATH/../../.." && pwd)"
else
  TRAIN_RUN_DIR="$MODEL_PARENT_DIR"
fi

if [ -z "${RESULT_NAME:-}" ]; then
  TRAIN_RUN_BASENAME="$(basename "$TRAIN_RUN_DIR")"
  RESULT_BASE_NAME="$TRAIN_RUN_BASENAME"
  if [[ "$RESULT_BASE_NAME" =~ ^(.+)_[0-9]{8}_[0-9]{6}$ ]]; then
    RESULT_BASE_NAME="${BASH_REMATCH[1]}"
  fi
  if [ -n "$EFFECTIVE_SAMPLE_SIZE" ]; then
    RESULT_NAME="${RESULT_BASE_NAME}_sample_${EFFECTIVE_SAMPLE_SIZE}"
  else
    RESULT_NAME="${RESULT_BASE_NAME}_full"
  fi
fi
if [ "$ADAPTER_ENABLED" != "true" ] && [[ "$RESULT_NAME" != *_adapter_off ]]; then
  RESULT_NAME="${RESULT_NAME}_adapter_off"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$TRAIN_RUN_DIR/benchmark_results}"
LOG_ROOT="${LOG_ROOT:-$TRAIN_RUN_DIR/benchmark_logs}"

BASE_OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${DATA_VERSION}/results_${RESULT_NAME}}"
BASE_LOG_NAME="${LOG_NAME:-${LOG_ROOT}/${DATA_VERSION}/${RESULT_NAME}}"

DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
WORKER_BATCH_SIZE="${WORKER_BATCH_SIZE:-1875}"
NUM_BEAMS="${NUM_BEAMS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-32}"
NUM_RETURN_THINKING_SEQUENCES="${NUM_RETURN_THINKING_SEQUENCES:-1}"
MAX_LOGPROBS="${MAX_LOGPROBS:-384}"
# Use the optimization-disabled path while investigating adapter CUDA failures.
FORCE_DISABLE_OPTIMIZATIONS="${FORCE_DISABLE_OPTIMIZATIONS:-true}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/openonerec_ray_${SLURM_JOB_ID:-$$}}"
export RAY_TEMP_DIR
RAY_LOG_DIR="${RAY_LOG_DIR:-${BASE_LOG_NAME}.ray_logs}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-${BASE_LOG_NAME}.gpu.log}"
GPU_MONITOR_PID=""
RAY_LOG_SYNC_INTERVAL="${RAY_LOG_SYNC_INTERVAL:-10}"
RAY_LOG_MONITOR_PID=""

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

copy_ray_logs() {
  mkdir -p "$RAY_LOG_DIR"
  {
    echo "ray_temp_dir: $RAY_TEMP_DIR"
    echo "copy_time: $(date)"
  } > "$RAY_LOG_DIR/ray_log_info.txt"

  local session_dir=""
  if [ -e "$RAY_TEMP_DIR/session_latest" ]; then
    session_dir="$(readlink -f "$RAY_TEMP_DIR/session_latest" 2>/dev/null || true)"
  fi
  if [ -z "$session_dir" ] || [ ! -d "$session_dir" ]; then
    session_dir="$(ls -td "$RAY_TEMP_DIR"/session_* 2>/dev/null | head -n 1 || true)"
  fi
  if [ -n "$session_dir" ] && [ -d "$session_dir/logs" ]; then
    echo "ray_session_dir: $session_dir" >> "$RAY_LOG_DIR/ray_log_info.txt"
    cp -a "$session_dir/logs/." "$RAY_LOG_DIR/" 2>/dev/null || true
  else
    echo "ray_session_dir: not found" >> "$RAY_LOG_DIR/ray_log_info.txt"
  fi
}

start_ray_log_monitor() {
  (
    while true; do
      copy_ray_logs
      sleep "$RAY_LOG_SYNC_INTERVAL"
    done
  ) >/dev/null 2>&1 &
  RAY_LOG_MONITOR_PID="$!"
}

stop_ray_log_monitor() {
  if [ -n "$RAY_LOG_MONITOR_PID" ]; then
    kill "$RAY_LOG_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  stop_gpu_monitor
  stop_ray_log_monitor
  copy_ray_logs
}
trap cleanup EXIT

mkdir -p "$(dirname "$BASE_LOG_NAME")" "$BASE_OUTPUT_DIR" "$RAY_TEMP_DIR" "$RAY_LOG_DIR"
start_gpu_monitor
start_ray_log_monitor

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = "true" ]; then
  THINKING_ARGS=(--enable_thinking)
fi

OPTIONAL_ARGS=()
if [ "$FORCE_DISABLE_OPTIMIZATIONS" = "true" ]; then
  OPTIONAL_ARGS+=(--force_disable_optimizations)
fi
if [ -n "${SAMPLE_SIZE:-}" ]; then
  OPTIONAL_ARGS+=(--sample_size "$SAMPLE_SIZE")
fi
if [ -n "${MAX_MODEL_LEN:-}" ]; then
  OPTIONAL_ARGS+=(--max_model_len "$MAX_MODEL_LEN")
fi
if [ -n "${NUM_GPUS:-}" ]; then
  OPTIONAL_ARGS+=(--num_gpus "$NUM_GPUS")
fi
if [ -n "${GPU_IDS:-}" ]; then
  # Use space-separated GPU ids, for example: GPU_IDS="0" or GPU_IDS="0 1".
  # shellcheck disable=SC2206
  GPU_ID_ARRAY=($GPU_IDS)
  OPTIONAL_ARGS+=(--gpu_ids "${GPU_ID_ARRAY[@]}")
fi
if [ -n "${TENSOR_PARALLEL_SIZE:-}" ]; then
  OPTIONAL_ARGS+=(--tensor_parallel_size "$TENSOR_PARALLEL_SIZE")
fi
BENCHMARK_RAY_ADDRESS="${BENCHMARK_RAY_ADDRESS:-${RAY_ADDRESS:-}}"
if [ -n "$BENCHMARK_RAY_ADDRESS" ]; then
  OPTIONAL_ARGS+=(--ray_address "$BENCHMARK_RAY_ADDRESS")
fi

{
  echo "========== Amazon Benchmark Configuration =========="
  echo "time: $(date)"
  echo "script_dir: $SCRIPT_DIR"
  echo "model_path: $MODEL_PATH"
  echo "task_types: $TASK_TYPES"
  echo "result_name: $RESULT_NAME"
  echo "benchmark_base_dir: $BENCHMARK_BASE_DIR"
  echo "benchmark_data_dir: $BENCHMARK_DATA_DIR"
  echo "train_run_dir: $TRAIN_RUN_DIR"
  echo "output_root: $OUTPUT_ROOT"
  echo "log_root: $LOG_ROOT"
  echo "data_version: $DATA_VERSION"
  echo "output_dir: $BASE_OUTPUT_DIR"
  echo "enable_thinking: $ENABLE_THINKING"
  echo "adapter_enabled: $ADAPTER_ENABLED"
  echo "dtype: $DTYPE"
  echo "gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
  echo "worker_batch_size: $WORKER_BATCH_SIZE"
  echo "num_beams: $NUM_BEAMS"
  echo "num_return_sequences: $NUM_RETURN_SEQUENCES"
  echo "num_return_thinking_sequences: $NUM_RETURN_THINKING_SEQUENCES"
  echo "max_logprobs: $MAX_LOGPROBS"
  echo "force_disable_optimizations: $FORCE_DISABLE_OPTIMIZATIONS"
  echo "sample_size: ${SAMPLE_SIZE:-full}"
  echo "max_model_len: ${MAX_MODEL_LEN:-unset}"
  echo "num_gpus: ${NUM_GPUS:-auto}"
  echo "gpu_ids: ${GPU_IDS:-auto}"
  echo "tensor_parallel_size: ${TENSOR_PARALLEL_SIZE:-1}"
  echo "ray_address: ${BENCHMARK_RAY_ADDRESS:-default}"
  echo "ray_temp_dir: $RAY_TEMP_DIR"
  echo "ray_log_dir: $RAY_LOG_DIR"
  echo "ray_log_sync_interval: $RAY_LOG_SYNC_INTERVAL"
  echo "gpu_monitor_log: $GPU_MONITOR_LOG"
  echo "gpu_monitor_interval: $GPU_MONITOR_INTERVAL"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
  python3 - <<'PY'
import benchmark
print(f"benchmark_package: {benchmark.__file__}")
PY
  echo "==================================================="
} | tee -a "${BASE_LOG_NAME}.log"

unset RAY_ADDRESS

python3 -u scripts/ray-vllm/evaluate.py \
  --task_types "$TASK_TYPES" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --model_path "$MODEL_PATH" \
  --data_dir "$BENCHMARK_DATA_DIR" \
  --output_dir "$BASE_OUTPUT_DIR" \
  --dtype "$DTYPE" \
  --worker_batch_size "$WORKER_BATCH_SIZE" \
  --max_logprobs "$MAX_LOGPROBS" \
  --overwrite \
  --num_beams "$NUM_BEAMS" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --num_return_thinking_sequences "$NUM_RETURN_THINKING_SEQUENCES" \
  "${THINKING_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}" \
  >> "${BASE_LOG_NAME}.log" 2>&1

echo "Amazon benchmark completed successfully"
echo "Results: $BASE_OUTPUT_DIR"
echo "Log: ${BASE_LOG_NAME}.log"
