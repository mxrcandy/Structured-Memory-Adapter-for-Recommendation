#!/bin/bash
#SBATCH -J one_e2e
#SBATCH -o e2e-%j.log
#SBATCH -e e2e-%j.err
#SBATCH -p A800
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:A800:4
#SBATCH --qos=normal

set -eo pipefail

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on:"
echo "$SLURM_JOB_NODELIST"

source /etc/profile
module load cuda/12.9.1
module load mpi/2021.16
export CONDA_ROOT="${CONDA_ROOT:-/opt/Software/Anaconda3}"
PRETRAIN_CONDA_ENV="${PRETRAIN_CONDA_ENV:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr/envs/openonerec}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$PRETRAIN_CONDA_ENV"
set -u

REPO_ROOT="${REPO_ROOT:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr}"
export E2E_EVAL_SBATCH_SCRIPT="${E2E_EVAL_SBATCH_SCRIPT:-$REPO_ROOT/benchmarks/slurm_eval_video_saids.sbatch}"
export E2E_GENERAL_EVAL_SBATCH_SCRIPT="${E2E_GENERAL_EVAL_SBATCH_SCRIPT:-$REPO_ROOT/benchmarks/slurm_eval_general_saids.sbatch}"
export BENCHMARK_CONDA_ENV="${BENCHMARK_CONDA_ENV:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr/envs/openonerec_benchmark}"
export GENERAL_EVAL_CONDA_ENV="${GENERAL_EVAL_CONDA_ENV:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr/envs/openonerec_lm_eval}"
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
