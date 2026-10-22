#!/usr/bin/env bash

# Source this file from a train_e2e_*.sh script after its final checkpoint is
# converted. It intentionally does not set shell options because it runs in the
# caller's shell and follows the caller's error-handling policy.
E2E_REC_EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_REC_EVAL_REPO_ROOT="$(cd "$E2E_REC_EVAL_SCRIPT_DIR/../.." && pwd)"

submit_e2e_rec_eval() {
  local model_path="$1"
  # Shared recommendation submission; task and entrypoint follow DATA_PROFILE.
  local default_eval_script eval_task_types
  case "${DATA_PROFILE:-10pct}" in
    amazon_letter_beauty|amazon_letter_beauty_general1_rec8|amazon_letter_beauty_general1_rec16)
      default_eval_script="slurm_eval_amazon.sbatch"
      eval_task_types="amazon_beauty"
      ;;
    amazon_letter_instruments|amazon_letter_instruments_general1_rec16)
      default_eval_script="slurm_eval_amazon.sbatch"
      eval_task_types="amazon_instruments"
      ;;
    amazon_letter_toys)
      default_eval_script="slurm_eval_amazon.sbatch"
      eval_task_types="amazon_toys"
      ;;
    amazon_letter_sports)
      default_eval_script="slurm_eval_amazon.sbatch"
      eval_task_types="amazon_sports"
      ;;
    amazon_letter_yelp)
      default_eval_script="slurm_eval_amazon.sbatch"
      eval_task_types="amazon_yelp"
      ;;
    *)
      default_eval_script="slurm_eval_video.sbatch"
      eval_task_types="video"
      ;;
  esac
  local eval_sbatch_script="${E2E_EVAL_SBATCH_SCRIPT:-$E2E_REC_EVAL_REPO_ROOT/benchmarks/$default_eval_script}"
  local sbatch_bin="${E2E_SBATCH_BIN:-sbatch}"
  local submit_output
  local -a eval_env submit_command

  EVAL_JOB_ID=""
  EVAL_SUBMISSION_MODE="not_submitted"

  if [ "${RUN_REC_EVAL:-true}" != "true" ]; then
    EVAL_SUBMISSION_MODE="disabled"
    echo "Recommendation evaluation is disabled (RUN_REC_EVAL=${RUN_REC_EVAL:-true})."
    return 0
  fi

  if [ ! -f "$eval_sbatch_script" ]; then
    echo "Recommendation evaluation Slurm script not found: $eval_sbatch_script" >&2
    return 1
  fi

  eval_env=("REPO_ROOT=${E2E_REC_EVAL_REPO_ROOT}" "MODEL_PATH=$model_path")
  eval_env+=("CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}")
  eval_env+=("BENCHMARK_CONDA_ENV=${BENCHMARK_CONDA_ENV:-openonerec_benchmark}")
  eval_env+=("TASK_TYPES=$eval_task_types")
  eval_env+=("ADAPTER_ENABLED=${REC_EVAL_ADAPTER_ENABLED:-true}")
  if [ -n "${REC_EVAL_SAMPLE_SIZE:-}" ]; then
    eval_env+=("SAMPLE_SIZE=${REC_EVAL_SAMPLE_SIZE}")
  fi
  if [ -n "${REC_EVAL_RESULT_NAME:-}" ]; then
    eval_env+=("RESULT_NAME=${REC_EVAL_RESULT_NAME}")
  fi
  for eval_var in NUM_BEAMS NUM_RETURN_SEQUENCES NUM_RETURN_THINKING_SEQUENCES MAX_LOGPROBS MAX_MODEL_LEN WORKER_BATCH_SIZE GPU_MEMORY_UTILIZATION; do
    if [ -n "${!eval_var:-}" ]; then
      eval_env+=("${eval_var}=${!eval_var}")
    fi
  done

  # A plain ALL export avoids implicit Slurm login-environment retrieval.
  submit_command=(env "${eval_env[@]}" "$sbatch_bin")

  if [ -z "${SLURM_JOB_ID:-}" ]; then
    EVAL_SUBMISSION_MODE="manual"
    echo "Training is not running under Slurm; recommendation evaluation was not submitted."
    printf "Submit it with: "
    printf "%q " "${submit_command[@]}" "--export=ALL" "$eval_sbatch_script"
    printf "\n"
    return 0
  fi

  if ! command -v "$sbatch_bin" >/dev/null 2>&1; then
    echo "Cannot submit recommendation evaluation because sbatch is unavailable: $sbatch_bin" >&2
    return 1
  fi

  if ! submit_output="$("${submit_command[@]}" \
    "--dependency=afterok:${SLURM_JOB_ID}" \
    "--export=ALL" \
    "$eval_sbatch_script")"; then
    echo "Failed to submit dependent recommendation evaluation job." >&2
    echo "$submit_output" >&2
    return 1
  fi

  if [[ "$submit_output" =~ [Ss]ubmitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
    EVAL_JOB_ID="${BASH_REMATCH[1]}"
  else
    echo "Could not parse recommendation evaluation job ID from sbatch output: $submit_output" >&2
    return 1
  fi

  EVAL_SUBMISSION_MODE="slurm_dependency"
  echo "Recommendation evaluation submitted as job $EVAL_JOB_ID with afterok dependency on training job $SLURM_JOB_ID."
}
