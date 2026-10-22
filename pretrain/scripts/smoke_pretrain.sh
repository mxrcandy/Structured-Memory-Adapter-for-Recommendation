#!/bin/bash
set -euo pipefail

PRETRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PRETRAIN_DIR/.." && pwd)"
cd "$PRETRAIN_DIR"
# shellcheck source=scripts/data_profiles.sh
source "$PRETRAIN_DIR/scripts/data_profiles.sh"
apply_data_profile

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/hf_models/Qwen3-1.7B_itemic}"
SOURCE_FILE_LIST="${SOURCE_FILE_LIST:-$(resolve_stage_source_file_list stage2)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/model_output/smoke/pretrain_${DATASET_TAG}}"
SMOKE_DIR="${SMOKE_DIR:-$OUTPUT_DIR/configs}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SMOKE_ROWS="${SMOKE_ROWS:-4}"
ITEMIC_START_INDEX="${ITEMIC_START_INDEX:-151669}"
ITEMIC_END_INDEX="${ITEMIC_END_INDEX:-176246}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model not found: $MODEL_DIR" >&2
  exit 1
fi
if [ ! -f "$SOURCE_FILE_LIST" ]; then
  echo "Pretrain file_list not found: $SOURCE_FILE_LIST" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$SMOKE_DIR" /tmp/_wids_cache

SMOKE_FILE_LIST="$SMOKE_DIR/pretrain_file_list_smoke.json"
SMOKE_DATASET_CONFIG="$SMOKE_DIR/pretrain_smoke.json"
SMOKE_PARQUET="$SMOKE_DIR/pretrain_smoke.parquet"

"$PYTHON_BIN" - "$SOURCE_FILE_LIST" "$SMOKE_FILE_LIST" "$SMOKE_PARQUET" "$SMOKE_ROWS" <<'PY'
import json
import sys
import pandas as pd

src, dst, smoke_parquet, smoke_rows = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
files = json.load(open(src, encoding="utf-8"))
if not files:
    raise SystemExit(f"empty file list: {src}")
df = pd.read_parquet(files[0]).head(smoke_rows)
if df.empty:
    raise SystemExit(f"empty smoke shard: {files[0]}")
df.to_parquet(smoke_parquet, index=False)
json.dump([smoke_parquet], open(dst, "w", encoding="utf-8"), indent=2)
print(f"wrote {smoke_parquet}: {len(df)} rows from {files[0]}")
print(f"wrote {dst}")
PY

"$PYTHON_BIN" - "$SMOKE_FILE_LIST" "$MODEL_DIR" "$SMOKE_DATASET_CONFIG" "$MAX_LENGTH" "$ITEMIC_START_INDEX" "$ITEMIC_END_INDEX" <<'PY'
import json
import sys
from transformers import AutoConfig

sources, model_dir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
max_length = int(sys.argv[4])
itemic_start = int(sys.argv[5])
itemic_end = int(sys.argv[6])
vocab_size = AutoConfig.from_pretrained(model_dir, trust_remote_code=True).vocab_size

config = {
    "name": "chat_completion_parquet",
    "sources": sources,
    "only_assistant_loss": False,
    "max_length": max_length,
    "base_model_dir": model_dir,
    "num_workers": 1,
    "num_epochs": 1,
    "cut_to_pad": 1,
    "model_class": "Qwen3ForCausalLM",
    "full_attention": False,
    "local_shuffle_buffer_size": 100,
    "max_sample_length": max_length,
    "local_shuffle_random_fetch": 0.0001
}
if itemic_end < vocab_size:
    config["itemic_id_range"] = [itemic_start, itemic_end]
json.dump(config, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {dst}; vocab_size={vocab_size}; itemic_enabled={'itemic_id_range' in config}")
PY

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + PORT_SEED % 20000))}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
START_SECONDS="$(date +%s)"
SCRIPT_FILE="$(readlink -f "$0")"
{
  echo "start_time: $START_TS"
  echo "script: $SCRIPT_FILE"
  echo "model_dir: $MODEL_DIR"
  echo "source_file_list: $SOURCE_FILE_LIST"
  echo "dataset_config: $SMOKE_DATASET_CONFIG"
  echo "output_dir: $OUTPUT_DIR"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "num_gpus: 1"
  echo "master_addr: $MASTER_ADDR"
  echo "master_port: $MASTER_PORT"
} >> "$OUTPUT_DIR/task_info.log"

mpirun -np 1 \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x PYTHONPATH \
  -x MASTER_ADDR \
  -x MASTER_PORT \
  -x NCCL_DEBUG \
  -x PYTORCH_CUDA_ALLOC_CONF \
  -x TOKENIZERS_PARALLELISM \
  "$PYTHON_BIN" recipes/train_qwen3.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_config "$SMOKE_DATASET_CONFIG" \
    --use_tie_weights \
    --model_class Qwen3ForCausalLM \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --max_length "$MAX_LENGTH" \
    --learning_rate 2e-4 \
    --min_lr 1e-4 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --num_training_steps 10 \
    --save_checkpoint_per_step 5 \
    --minibatch_size 1024 \
    --logging_per_step 1 \
    --seed 19260817 \
    --enable_gradient_checkpointing \
    --use_chunked_loss_computer \
  > "$OUTPUT_DIR/stdout.log" 2> "$OUTPUT_DIR/stderr.log"

END_TS="$(date '+%Y-%m-%d %H:%M:%S')"
END_SECONDS="$(date +%s)"
{
  echo "end_time: $END_TS"
  echo "elapsed_seconds: $((END_SECONDS - START_SECONDS))"
  echo "latest_checkpoint: $(cat "$OUTPUT_DIR/latest" 2>/dev/null || true)"
  echo "========================="
} >> "$OUTPUT_DIR/task_info.log"

echo "Smoke pretrain output: $OUTPUT_DIR"
