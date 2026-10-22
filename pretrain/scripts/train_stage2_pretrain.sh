#!/bin/bash
set -euo pipefail

PRETRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PRETRAIN_DIR/.." && pwd)"
cd "$PRETRAIN_DIR"
# shellcheck source=scripts/data_profiles.sh
source "$PRETRAIN_DIR/scripts/data_profiles.sh"
apply_data_profile

SEED="${SEED:-19260817}"
SEED_SUFFIX=""
if [ "$SEED" != "19260817" ]; then SEED_SUFFIX="_seed${SEED}"; fi
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
STAGE1_CONVERTED_MODEL_DIR="${STAGE1_CONVERTED_MODEL_DIR:-}"
MODEL_DIR="${MODEL_DIR:-$STAGE1_CONVERTED_MODEL_DIR}"
SOURCE_FILE_LIST="${SOURCE_FILE_LIST:-$(resolve_stage_source_file_list stage2)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/model_output/stage2_pretrain_${DATASET_TAG}${SEED_SUFFIX}_${RUN_ID}}"
RUN_CONFIG_DIR="${RUN_CONFIG_DIR:-$OUTPUT_DIR/configs}"
DATASET_CONFIG="$RUN_CONFIG_DIR/pretrain_stage2.json"
RESOLVED_SOURCE_FILE_LIST="$RUN_CONFIG_DIR/pretrain_stage2_file_list.resolved.json"
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
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-500}"
NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-5000}"
SAVE_CHECKPOINT_PER_STEP="${SAVE_CHECKPOINT_PER_STEP:-0}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"
LOGGING_PER_STEP="${LOGGING_PER_STEP:-5}"
ITEMIC_START_INDEX="${ITEMIC_START_INDEX:-151669}"
ITEMIC_END_INDEX="${ITEMIC_END_INDEX:-176246}"
FINETUNE_MODE="${FINETUNE_MODE:-full}"
ADAPTER_TYPE="${ADAPTER_TYPE:-}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-}"
ADAPTER_HIDDEN_SIZE="${ADAPTER_HIDDEN_SIZE:-}"
ADAPTER_SPM_HIDDEN_SIZE="${ADAPTER_SPM_HIDDEN_SIZE:-}"
ADAPTER_ITM_HIDDEN_SIZE="${ADAPTER_ITM_HIDDEN_SIZE:-}"
ADAPTER_MEMORY_SIZE="${ADAPTER_MEMORY_SIZE:-1024}"
ADAPTER_SPM_MEMORY_SIZE="${ADAPTER_SPM_MEMORY_SIZE:-}"
ADAPTER_ITM_MEMORY_SIZE="${ADAPTER_ITM_MEMORY_SIZE:-}"
ADAPTER_MEMORY_INIT="${ADAPTER_MEMORY_INIT:-default}"
ADAPTER_MEMORY_INIT_STD="${ADAPTER_MEMORY_INIT_STD:-0.01}"
ADAPTER_MEMORY_PROTOTYPE_PATH="${ADAPTER_MEMORY_PROTOTYPE_PATH:-}"
ADAPTER_NGRAM_ORDER="${ADAPTER_NGRAM_ORDER:-2}"
ADAPTER_TOPK="${ADAPTER_TOPK:-4}"
ADAPTER_USE_ALPHA_LAYER="${ADAPTER_USE_ALPHA_LAYER:-false}"
ADAPTER_ALPHA_INIT="${ADAPTER_ALPHA_INIT:-0.1}"
ADAPTER_FIXED_CONTEXT_GATE="${ADAPTER_FIXED_CONTEXT_GATE:-}"
ADAPTER_DISABLE_CONV="${ADAPTER_DISABLE_CONV:-false}"
ADAPTER_USE_SID_SPAN="${ADAPTER_USE_SID_SPAN:-false}"
ADAPTER_SID_TOKEN_START="${ADAPTER_SID_TOKEN_START:-$ITEMIC_START_INDEX}"
ADAPTER_SID_TOKEN_END="${ADAPTER_SID_TOKEN_END:-$ITEMIC_END_INDEX}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
LORA_LAYERS="${LORA_LAYERS:-}"
PARTIAL_TRAIN_LAYERS="${PARTIAL_TRAIN_LAYERS:-}"
PARTIAL_TRAIN_TARGET_MODULES="${PARTIAL_TRAIN_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

if [ -z "$MODEL_DIR" ]; then
  echo "MODEL_DIR is required for Stage2." >&2
  echo "Set MODEL_DIR or STAGE1_CONVERTED_MODEL_DIR to the converted HF checkpoint from Stage1." >&2
  echo "Example: STAGE1_CONVERTED_MODEL_DIR=/path/to/stage1/stepX/global_stepX/converted bash scripts/train_stage2_pretrain.sh" >&2
  exit 1
fi
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
  echo "finetune_mode: $FINETUNE_MODE"
  echo "adapter_type: ${ADAPTER_TYPE:-none}"
  echo "adapter_layers: ${ADAPTER_LAYERS:-all}"
  echo "adapter_hidden_size: ${ADAPTER_HIDDEN_SIZE:-model_hidden_size}"
  echo "adapter_spm_hidden_size: ${ADAPTER_SPM_HIDDEN_SIZE:-${ADAPTER_HIDDEN_SIZE:-model_hidden_size}}"
  echo "adapter_itm_hidden_size: ${ADAPTER_ITM_HIDDEN_SIZE:-${ADAPTER_HIDDEN_SIZE:-model_hidden_size}}"
  echo "adapter_memory_size: $ADAPTER_MEMORY_SIZE"
  echo "adapter_spm_memory_size: ${ADAPTER_SPM_MEMORY_SIZE:-$ADAPTER_MEMORY_SIZE}"
  echo "adapter_itm_memory_size: ${ADAPTER_ITM_MEMORY_SIZE:-$ADAPTER_MEMORY_SIZE}"
  echo "adapter_memory_init: $ADAPTER_MEMORY_INIT"
  echo "adapter_memory_init_std: $ADAPTER_MEMORY_INIT_STD"
  echo "adapter_memory_prototype_path: ${ADAPTER_MEMORY_PROTOTYPE_PATH:-}"
  echo "adapter_ngram_order: $ADAPTER_NGRAM_ORDER"
  echo "adapter_topk: $ADAPTER_TOPK"
  echo "adapter_use_alpha_layer: $ADAPTER_USE_ALPHA_LAYER"
  echo "adapter_alpha_init: $ADAPTER_ALPHA_INIT"
  echo "adapter_fixed_context_gate: ${ADAPTER_FIXED_CONTEXT_GATE:-dynamic}"
  echo "adapter_use_conv: $([[ \"$ADAPTER_DISABLE_CONV\" == \"true\" ]] && echo false || echo true)"
  echo "adapter_use_sid_span: $ADAPTER_USE_SID_SPAN"
  echo "adapter_sid_token_start: $ADAPTER_SID_TOKEN_START"
  echo "adapter_sid_token_end: $ADAPTER_SID_TOKEN_END"
  echo "lora_rank: $LORA_RANK"
  echo "lora_alpha: $LORA_ALPHA"
  echo "lora_dropout: $LORA_DROPOUT"
  echo "lora_target_modules: $LORA_TARGET_MODULES"
  echo "lora_layers: ${LORA_LAYERS:-all}"
  echo "partial_train_layers: ${PARTIAL_TRAIN_LAYERS:-all}"
  echo "partial_train_target_modules: $PARTIAL_TRAIN_TARGET_MODULES"
} >> "$OUTPUT_DIR/task_info.log"

TRAIN_EXTRA_ARGS=(--finetune_mode "$FINETUNE_MODE")
TRAIN_EXTRA_ARGS+=(--adapter_itm_mode "${ADAPTER_ITM_MODE:-default}")
if [ -n "${ADAPTER_ITM_AUX_LOSS_WEIGHT:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--adapter_itm_aux_loss_weight "$ADAPTER_ITM_AUX_LOSS_WEIGHT")
fi
printf 'adapter_itm_aux_loss_weight: %s\n' "${ADAPTER_ITM_AUX_LOSS_WEIGHT:-inherit}" >> "$OUTPUT_DIR/task_info.log"
if [[ "${ADAPTER_ITM_MODE:-default}" = "exact" || "${ADAPTER_ITM_MODE:-default}" = "successor_prior" ]]; then
  TRAIN_EXTRA_ARGS+=(--adapter_itm_pattern_min_frequency "${ADAPTER_ITM_PATTERN_MIN_FREQUENCY:-5}")
fi
if [ -n "${ADAPTER_ITM_PATTERN_FILE:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--adapter_itm_pattern_file "$ADAPTER_ITM_PATTERN_FILE")
fi
printf 'adapter_itm_mode: %s\nadapter_itm_pattern_file: %s\n' "${ADAPTER_ITM_MODE:-default}" "${ADAPTER_ITM_PATTERN_FILE:-}" >> "$OUTPUT_DIR/task_info.log"
if [[ "$ADAPTER_TYPE" == "engram_structure" || "$ADAPTER_TYPE" == "lngram_structure" ]]; then
  TRAIN_EXTRA_ARGS+=(--adapter_use_spm "${ADAPTER_USE_SPM:-true}" --adapter_use_itm "${ADAPTER_USE_ITM:-true}" --adapter_itm_order "${ADAPTER_ITM_ORDER:-1}")
  printf 'adapter_use_spm: %s\nadapter_use_itm: %s\nadapter_itm_order: %s\n' "${ADAPTER_USE_SPM:-true}" "${ADAPTER_USE_ITM:-true}" "${ADAPTER_ITM_ORDER:-1}" >> "$OUTPUT_DIR/task_info.log"
fi
if { [ "$FINETUNE_MODE" = "adapter" ] || [ "$FINETUNE_MODE" = "adapter_lora" ] || [ "$FINETUNE_MODE" = "adapter_partial" ] || [ "$FINETUNE_MODE" = "adapter_full" ]; }; then
  if [ -n "$ADAPTER_TYPE" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_type "$ADAPTER_TYPE")
  fi
  if [ -n "$ADAPTER_LAYERS" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_layers "$ADAPTER_LAYERS")
  fi
  if [ -n "$ADAPTER_HIDDEN_SIZE" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_hidden_size "$ADAPTER_HIDDEN_SIZE")
  fi
  if [ -n "$ADAPTER_SPM_HIDDEN_SIZE" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_spm_hidden_size "$ADAPTER_SPM_HIDDEN_SIZE")
  fi
  if [ -n "$ADAPTER_ITM_HIDDEN_SIZE" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_itm_hidden_size "$ADAPTER_ITM_HIDDEN_SIZE")
  fi
  TRAIN_EXTRA_ARGS+=(--adapter_memory_size "$ADAPTER_MEMORY_SIZE")
  if [ -n "$ADAPTER_SPM_MEMORY_SIZE" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_spm_memory_size "$ADAPTER_SPM_MEMORY_SIZE")
  fi
  if [ -n "$ADAPTER_ITM_MEMORY_SIZE" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_itm_memory_size "$ADAPTER_ITM_MEMORY_SIZE")
  fi
  TRAIN_EXTRA_ARGS+=(--adapter_memory_init "$ADAPTER_MEMORY_INIT")
  TRAIN_EXTRA_ARGS+=(--adapter_memory_init_std "$ADAPTER_MEMORY_INIT_STD")
  if [ -n "$ADAPTER_MEMORY_PROTOTYPE_PATH" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_memory_prototype_path "$ADAPTER_MEMORY_PROTOTYPE_PATH")
  fi
  TRAIN_EXTRA_ARGS+=(--adapter_ngram_order "$ADAPTER_NGRAM_ORDER")
  TRAIN_EXTRA_ARGS+=(--adapter_topk "$ADAPTER_TOPK")
  if [ "${ADAPTER_TYPE:-}" = "lngram" ] || [ "${ADAPTER_TYPE:-}" = "lngram_structure" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_lngram_bits_per_route "$ADAPTER_LNGRAM_BITS_PER_ROUTE")
  fi
  if [ "$ADAPTER_USE_ALPHA_LAYER" = "true" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_use_alpha_layer --adapter_alpha_init "$ADAPTER_ALPHA_INIT")
  fi
  if [ -n "$ADAPTER_FIXED_CONTEXT_GATE" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_fixed_context_gate "$ADAPTER_FIXED_CONTEXT_GATE")
  fi
  if [ "$ADAPTER_DISABLE_CONV" = "true" ]; then
    TRAIN_EXTRA_ARGS+=(--adapter_disable_conv)
  fi
  if [ "$ADAPTER_USE_SID_SPAN" = "true" ]; then
    TRAIN_EXTRA_ARGS+=(
      --adapter_use_sid_span
      --adapter_sid_token_start "$ADAPTER_SID_TOKEN_START"
      --adapter_sid_token_end "$ADAPTER_SID_TOKEN_END"
    )
  fi
fi
if { [ "$FINETUNE_MODE" = "lora" ] || [ "$FINETUNE_MODE" = "adapter_lora" ]; }; then
  TRAIN_EXTRA_ARGS+=(--lora_rank "$LORA_RANK")
  TRAIN_EXTRA_ARGS+=(--lora_alpha "$LORA_ALPHA")
  TRAIN_EXTRA_ARGS+=(--lora_dropout "$LORA_DROPOUT")
  TRAIN_EXTRA_ARGS+=(--lora_target_modules "$LORA_TARGET_MODULES")
  if [ -n "$LORA_LAYERS" ]; then
    TRAIN_EXTRA_ARGS+=(--lora_layers "$LORA_LAYERS")
  fi
fi
if [ "$FINETUNE_MODE" = "houlsby_adapter" ]; then
  TRAIN_EXTRA_ARGS+=(--houlsby_adapter_bottleneck "${HOULSBY_ADAPTER_BOTTLENECK:-256}")
fi
if [ "$FINETUNE_MODE" = "adapter_partial" ]; then
  if [ -n "$PARTIAL_TRAIN_LAYERS" ]; then
    TRAIN_EXTRA_ARGS+=(--partial_train_layers "$PARTIAL_TRAIN_LAYERS")
  fi
  TRAIN_EXTRA_ARGS+=(--partial_train_target_modules "$PARTIAL_TRAIN_TARGET_MODULES")
fi
if { [ "$FINETUNE_MODE" = "sid_only" ] || [ "$FINETUNE_MODE" = "adapter" ] || [ "$FINETUNE_MODE" = "lora" ] || [ "$FINETUNE_MODE" = "adapter_lora" ] || [ "$FINETUNE_MODE" = "adapter_partial" ] || [ "$FINETUNE_MODE" = "houlsby_adapter" ]; } && [ "${ITEMIC_START_INDEX:-0}" -gt 0 ]; then
  TRAIN_EXTRA_ARGS+=(--start_optimize_embedding_index "$ITEMIC_START_INDEX")
fi

set -x
mpirun -np "$NUM_GPUS" \
  "${MPI_ENV_ARGS[@]}" \
  "$PYTHON_BIN" recipes/train_qwen3.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_config "$DATASET_CONFIG" \
    --use_tie_weights \
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
    "${TRAIN_EXTRA_ARGS[@]}" \
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

echo "Stage2 output: $OUTPUT_DIR"
