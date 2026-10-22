#!/bin/bash
#SBATCH -J one_convert
#SBATCH -o convert-%j.log
#SBATCH -e convert-%j.err
#SBATCH -p A800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:A800:1
#SBATCH --qos=normal

set -eo pipefail

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on:"
echo "$SLURM_JOB_NODELIST"

source /etc/profile
module load cuda/12.9.1
export CONDA_ROOT="${CONDA_ROOT:-/opt/Software/Anaconda3}"
PRETRAIN_CONDA_ENV="${PRETRAIN_CONDA_ENV:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr/envs/openonerec}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$PRETRAIN_CONDA_ENV"
set -u

REPO_ROOT="${REPO_ROOT:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr}"
: "${BASE_MODEL_DIR:?Set BASE_MODEL_DIR to the HF model used to start this training stage.}"
: "${MODEL_HOME:?Set MODEL_HOME to the FSDP checkpoint output directory.}"
: "${STEP:?Set STEP to the checkpoint step to convert.}"

cd "$REPO_ROOT/pretrain"
bash "$REPO_ROOT/pretrain/scripts/convert_checkpoint_to_hf.sh" \
  "$BASE_MODEL_DIR" "$MODEL_HOME" "$STEP"
