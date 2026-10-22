#!/bin/bash
#SBATCH -J one_train_amazon
#SBATCH -o amazon-stage-train-%j.log
#SBATCH -e amazon-stage-train-%j.err
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
module load mpi/2021.16
export CONDA_ROOT="${CONDA_ROOT:-/opt/Software/Anaconda3}"
PRETRAIN_CONDA_ENV="${PRETRAIN_CONDA_ENV:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr/envs/openonerec}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$PRETRAIN_CONDA_ENV"
set -u

# Amazon training uses one GPU; preserve the device assigned by Slurm.
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA_PROFILE="${DATA_PROFILE:-amazon_letter_beauty}"

REPO_ROOT="${REPO_ROOT:-/data2/group_王皓/mhFang/Projects/OpenOneRec-mxr}"
: "${TRAIN_ENTRY:?Set TRAIN_ENTRY to train_stage1_itemic.sh, train_stage2_pretrain.sh, or train_sft_video.sh.}"

cd "$REPO_ROOT/pretrain"
echo "job_id: ${SLURM_JOB_ID:-unknown}"
echo "data_profile: $DATA_PROFILE"
echo "train_entry: $TRAIN_ENTRY"
echo "finetune_mode: ${FINETUNE_MODE:-script_default}"

bash "$REPO_ROOT/pretrain/scripts/$TRAIN_ENTRY"
