#!/bin/bash
#SBATCH -J one_e2e
#SBATCH -o e2e-%j.log
#SBATCH -e e2e-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu_8a100

set -eo pipefail

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on:"
echo "$SLURM_JOB_NODELIST"

. /etc/profile.d/modules.sh
module load cuda/12.4
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
. "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${PRETRAIN_CONDA_ENV:-openonerec}"
set -u

REPO_ROOT="${REPO_ROOT:-/home/wanghao3group/ywliangsc/Projects/OpenOneRec-mxr}"
E2E_ENTRY="${E2E_ENTRY:?Set E2E_ENTRY to a train_e2e_*.sh entrypoint.}"
case "$E2E_ENTRY" in
  train_e2e_full.sh|train_e2e_sid_only.sh|train_e2e_lora.sh|train_e2e_adapter.sh|train_e2e_adapter_structure.sh|train_e2e_adapter_lora.sh|train_e2e_adapter_partial.sh|train_e2e_adapter_full.sh|train_e2e_houlsby_adapter.sh) ;;
  *)
    echo "Unsupported E2E_ENTRY: $E2E_ENTRY" >&2
    exit 1
    ;;
esac

cd "$REPO_ROOT/pretrain"
echo "job_id: ${SLURM_JOB_ID:-unknown}"
echo "data_profile: ${DATA_PROFILE:-10pct}"
echo "e2e_entry: $E2E_ENTRY"
echo "finetune_mode: ${FINETUNE_MODE:-script_default}"
echo "run_general_eval: ${RUN_GENERAL_EVAL:-false}"
echo "run_rec_eval: ${RUN_REC_EVAL:-true}"
echo "rec_eval_sample_size: ${REC_EVAL_SAMPLE_SIZE:-full}"

bash "$REPO_ROOT/pretrain/scripts/$E2E_ENTRY"
