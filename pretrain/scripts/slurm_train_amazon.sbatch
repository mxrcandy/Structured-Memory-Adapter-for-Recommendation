#!/bin/bash
#SBATCH -J one_train_amazon
#SBATCH -o amazon-stage-train-%j.log
#SBATCH -e amazon-stage-train-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
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

# Amazon training uses one GPU; preserve the device assigned by Slurm.
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA_PROFILE="${DATA_PROFILE:-amazon_letter_beauty}"

REPO_ROOT="${REPO_ROOT:-/home/wanghao3group/ywliangsc/Projects/OpenOneRec-mxr}"
: "${TRAIN_ENTRY:?Set TRAIN_ENTRY to train_stage1_itemic.sh, train_stage2_pretrain.sh, or train_sft_video.sh.}"

cd "$REPO_ROOT/pretrain"
echo "job_id: ${SLURM_JOB_ID:-unknown}"
echo "data_profile: $DATA_PROFILE"
echo "train_entry: $TRAIN_ENTRY"
echo "finetune_mode: ${FINETUNE_MODE:-script_default}"

bash "$REPO_ROOT/pretrain/scripts/$TRAIN_ENTRY"
