#!/usr/bin/env bash

# Source this file from a train_e2e_*.sh script after its final checkpoint is
# converted. It intentionally does not set shell options.
E2E_GENERAL_EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_GENERAL_EVAL_REPO_ROOT="$(cd "$E2E_GENERAL_EVAL_SCRIPT_DIR/../.." && pwd)"

submit_e2e_general_eval() {
  local model_path="$1"
  local eval_sbatch_script="${E2E_GENERAL_EVAL_SBATCH_SCRIPT:-$E2E_GENERAL_EVAL_REPO_ROOT/benchmarks/slurm_eval_general.sbatch}"
  local sbatch_bin="${E2E_SBATCH_BIN:-sbatch}"
  local submit_output
  local -a eval_env submit_command

  GENERAL_EVAL_JOB_ID=""
  GENERAL_EVAL_SUBMISSION_MODE="not_submitted"

  if [ "${RUN_GENERAL_EVAL:-false}" != "true" ]; then
    GENERAL_EVAL_SUBMISSION_MODE="disabled"
    echo "General evaluation is disabled (RUN_GENERAL_EVAL=${RUN_GENERAL_EVAL:-false})."
    return 0
  fi

  if [ ! -f "$eval_sbatch_script" ]; then
    echo "General evaluation Slurm script not found: $eval_sbatch_script" >&2
    return 1
  fi

  eval_env=("REPO_ROOT=${E2E_GENERAL_EVAL_REPO_ROOT}" "MODEL_PATH=$model_path")
  eval_env+=("CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}")
  eval_env+=("GENERAL_EVAL_CONDA_ENV=${GENERAL_EVAL_CONDA_ENV:-openonerec_lm_eval}")
  eval_env+=("ADAPTER_ENABLED=${GENERAL_EVAL_ADAPTER_ENABLED:-true}")
  if [ -n "${GENERAL_EVAL_RESULT_NAME:-}" ]; then
    eval_env+=("RESULT_NAME=$GENERAL_EVAL_RESULT_NAME")
  fi

  # A plain ALL export avoids implicit Slurm login-environment retrieval.
  submit_command=(env "${eval_env[@]}" "$sbatch_bin")

  if [ -z "${SLURM_JOB_ID:-}" ]; then
    GENERAL_EVAL_SUBMISSION_MODE="manual"
    echo "Training is not running under Slurm; general evaluation was not submitted."
    printf "Submit it with: "
    printf "%q " "${submit_command[@]}" "--export=ALL" "$eval_sbatch_script"
    printf "\n"
    return 0
  fi

  if ! command -v "$sbatch_bin" >/dev/null 2>&1; then
    echo "Cannot submit general evaluation because sbatch is unavailable: $sbatch_bin" >&2
    return 1
  fi

  if ! submit_output="$("${submit_command[@]}" \
    "--dependency=afterok:${SLURM_JOB_ID}" \
    "--export=ALL" \
    "$eval_sbatch_script")"; then
    echo "Failed to submit dependent general evaluation job." >&2
    echo "$submit_output" >&2
    return 1
  fi
  if [[ "$submit_output" =~ [Ss]ubmitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
    GENERAL_EVAL_JOB_ID="${BASH_REMATCH[1]}"
  else
    echo "Could not parse general evaluation job ID from sbatch output: $submit_output" >&2
    return 1
  fi
  GENERAL_EVAL_SUBMISSION_MODE="slurm_dependency"
  echo "General evaluation submitted as job $GENERAL_EVAL_JOB_ID with afterok dependency on training job $SLURM_JOB_ID."
}
