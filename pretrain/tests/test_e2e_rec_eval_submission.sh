#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HELPER="$REPO_ROOT/pretrain/scripts/e2e_rec_eval.sh"

if [ ! -f "$HELPER" ]; then
  echo "Missing e2e evaluation helper: $HELPER" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$HELPER"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
SBATCH_ARGS="$TMP_DIR/sbatch.args"

assert_contains() {
  local expected="$1"
  if ! rg -Fqx -- "$expected" "$SBATCH_ARGS"; then
    echo "Expected sbatch argument not found: $expected" >&2
    cat "$SBATCH_ARGS" >&2
    exit 1
  fi
}

export SBATCH_ARGS
export E2E_SBATCH_BIN="$SCRIPT_DIR/mock_sbatch.sh"
export SLURM_JOB_ID=1234

EVAL_JOB_ID=""
submit_e2e_rec_eval \
  "/tmp/final-converted"

[ "$EVAL_JOB_ID" = "4242" ]
assert_contains '--dependency=afterok:1234'
if rg -n -- '--gres=gpu:' "$SBATCH_ARGS"; then
  echo 'GPU allocation must come from the evaluation Slurm script.' >&2
  exit 1
fi
assert_contains '--export=ALL'
rg -Fxq 'MODEL_PATH=/tmp/final-converted' "$SBATCH_ARGS.env"
rg -Fxq 'BENCHMARK_CONDA_ENV=openonerec_benchmark' "$SBATCH_ARGS.env"
assert_contains "$REPO_ROOT/benchmarks/slurm_eval_video.sbatch"
if rg -n 'OUTPUT_ROOT|LOG_ROOT|TRAIN_RUN_DIR|RESULT_NAME' "$SBATCH_ARGS"; then
  echo 'Unexpected evaluation configuration was passed to sbatch.' >&2
  exit 1
fi

rm -f "$SBATCH_ARGS"
unset SLURM_JOB_ID
EVAL_JOB_ID=""
submit_e2e_rec_eval \
  "/tmp/final-converted"

[ -z "$EVAL_JOB_ID" ]
[ ! -e "$SBATCH_ARGS" ]

echo 'test_e2e_rec_eval_submission: PASS'
