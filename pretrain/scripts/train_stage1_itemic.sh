#!/bin/bash
set -euo pipefail

PRETRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PRETRAIN_DIR/.." && pwd)"
cd "$PRETRAIN_DIR"
# shellcheck source=scripts/data_profiles.sh
source "$PRETRAIN_DIR/scripts/data_profiles.sh"
apply_data_profile

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/hf_models/Qwen3-1.7B_itemic}"
SOURCE_FILE_LIST="${SOURCE_FILE_LIST:-$(resolve_stage_source_file_list stage1)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/model_output/stage1_itemic_alignment_${DATASET_TAG}_$RUN_ID}"
RUN_CONFIG_DIR="${RUN_CONFIG_DIR:-$OUTPUT_DIR/configs}"
DATASET_CONFIG="$RUN_CONFIG_DIR/pretrain_stage1.json"
RESOLVED_SOURCE_FILE_LIST="$RUN_CONFIG_DIR/pretrain_stage1_file_list.resolved.json"
PYTHON_BIN="${PYTHON_BIN:-python}"

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
MASTER_PORT="${MASTER_PORT:-$((20000 + PORT_SEED % 20000))}"

MAX_LENGTH="${MAX_LENGTH:-32768}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOCAL_SHUFFLE_BUFFER_SIZE="${LOCAL_SHUFFLE_BUFFER_SIZE:-10000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MIN_LR="${MIN_LR:-1e-4}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-200}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-2000}"
SAVE_CHECKPOINT_PER_STEP="${SAVE_CHECKPOINT_PER_STEP:-0}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"
LOGGING_PER_STEP="${LOGGING_PER_STEP:-5}"
ITEMIC_START_INDEX="${ITEMIC_START_INDEX:-151669}"
ITEMIC_END_INDEX="${ITEMIC_END_INDEX:-176246}"
SEED="${SEED:-19260817}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model not found: $MODEL_DIR" >&2
  exit 1
fi
if [ ! -f "$SOURCE_FILE_LIST" ]; then
  echo "Pretrain file_list not found: $SOURCE_FILE_LIST" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$RUN_CONFIG_DIR" /tmp/_wids_cache

GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-$OUTPUT_DIR/gpu_monitor.log}"
GPU_MONITOR_PID=""
start_gpu_monitor() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while true; do
        date
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
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
trap stop_gpu_monitor EXIT
start_gpu_monitor

"$PYTHON_BIN" - "$SOURCE_FILE_LIST" "$REPO_ROOT" "$RESOLVED_SOURCE_FILE_LIST" <<'PY'
import json
import sys
from pathlib import Path

src, repo_root, dst = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
files = json.loads(Path(src).read_text(encoding="utf-8"))
resolved = []
for file_name in files:
    path = Path(file_name)
    if not path.is_absolute():
        path = repo_root / path
    resolved.append(str(path.resolve()))

Path(dst).write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"resolved {len(resolved)} source shards to {dst}")
PY

"$PYTHON_BIN" - "$RESOLVED_SOURCE_FILE_LIST" "$MODEL_DIR" "$DATASET_CONFIG" "$MAX_LENGTH" "$NUM_EPOCHS" "$NUM_WORKERS" "$LOCAL_SHUFFLE_BUFFER_SIZE" "$ITEMIC_START_INDEX" "$ITEMIC_END_INDEX" <<'PY'
import json
import sys
from transformers import AutoConfig

sources, model_dir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
max_length = int(sys.argv[4])
num_epochs = int(sys.argv[5])
num_workers = int(sys.argv[6])
shuffle_buffer = int(sys.argv[7])
itemic_start = int(sys.argv[8])
itemic_end = int(sys.argv[9])
vocab_size = AutoConfig.from_pretrained(model_dir, trust_remote_code=True).vocab_size

config = {
    "name": "chat_completion_parquet",
    "sources": sources,
    "only_assistant_loss": False,
    "max_length": max_length,
    "base_model_dir": model_dir,
    "num_workers": num_workers,
    "num_epochs": num_epochs,
    "cut_to_pad": 1,
    "model_class": "Qwen3ForCausalLM",
    "full_attention": False,
    "local_shuffle_buffer_size": shuffle_buffer,
    "max_sample_length": max_length,
    "local_shuffle_random_fetch": 0.0001,
}
if itemic_end < vocab_size:
    config["itemic_id_range"] = [itemic_start, itemic_end]

with open(dst, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print(f"wrote {dst}; vocab_size={vocab_size}; itemic_enabled={'itemic_id_range' in config}")
PY

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR
export MASTER_PORT
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$PRETRAIN_DIR/scripts/mpi_env_compat.sh"
mpi_env_args PATH LD_LIBRARY_PATH PYTHONPATH CUDA_VISIBLE_DEVICES MASTER_ADDR MASTER_PORT NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
START_SECONDS="$(date +%s)"
SCRIPT_FILE="$(readlink -f "$0")"
{
  echo "start_time: $START_TS"
  echo "script: $SCRIPT_FILE"
  echo "model_dir: $MODEL_DIR"
  echo "source_file_list: $SOURCE_FILE_LIST"
  echo "resolved_source_file_list: $RESOLVED_SOURCE_FILE_LIST"
  echo "dataset_config: $DATASET_CONFIG"
  echo "output_dir: $OUTPUT_DIR"
  echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo "num_gpus: $NUM_GPUS"
  echo "master_addr: $MASTER_ADDR"
  echo "master_port: $MASTER_PORT"
  echo "gpu_monitor_log: $GPU_MONITOR_LOG"
} >> "$OUTPUT_DIR/task_info.log"

set -x
mpirun -np "$NUM_GPUS" \
  "${MPI_ENV_ARGS[@]}" \
  "$PYTHON_BIN" recipes/train_qwen3.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_config "$DATASET_CONFIG" \
    --freeze_llm \
    --use_tie_weights \
    --start_optimize_embedding_index "$ITEMIC_START_INDEX" \
    --model_class Qwen3ForCausalLM \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --max_length "$MAX_LENGTH" \
    --learning_rate "$LEARNING_RATE" \
    --min_lr "$MIN_LR" \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --num_warmup_steps "$NUM_WARMUP_STEPS" \
    --num_training_steps "$NUM_TRAINING_STEPS" \
    --save_checkpoint_per_step "$SAVE_CHECKPOINT_PER_STEP" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --logging_per_step "$LOGGING_PER_STEP" \
    --seed "$SEED" \
    --enable_gradient_checkpointing \
    --use_chunked_loss_computer \
  > "$OUTPUT_DIR/stdout.log" 2> "$OUTPUT_DIR/stderr.log"
set +x

END_TS="$(date '+%Y-%m-%d %H:%M:%S')"
END_SECONDS="$(date +%s)"
{
  echo "end_time: $END_TS"
  echo "elapsed_seconds: $((END_SECONDS - START_SECONDS))"
  echo "latest_checkpoint: $(cat "$OUTPUT_DIR/latest" 2>/dev/null || true)"
  echo "========================="
} >> "$OUTPUT_DIR/task_info.log"

echo "Stage1 output: $OUTPUT_DIR"
