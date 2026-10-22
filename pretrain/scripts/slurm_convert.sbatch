#!/bin/bash
#SBATCH -J one_convert
#SBATCH -o convert-%j.log
#SBATCH -e convert-%j.err
#SBATCH -p CPU-64C256GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos=qos_cpu_64c256gb

set -eo pipefail

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on:"
echo "$SLURM_JOB_NODELIST"

export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
. "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${PRETRAIN_CONDA_ENV:-openonerec}"
set -u

REPO_ROOT="${REPO_ROOT:-/home/wanghao3group/ywliangsc/Projects/OpenOneRec-mxr}"
: "${BASE_MODEL_DIR:?Set BASE_MODEL_DIR to the HF model used to start this training stage.}"
: "${MODEL_HOME:?Set MODEL_HOME to the FSDP checkpoint output directory.}"
: "${STEP:?Set STEP to the checkpoint step to convert.}"

cd "$REPO_ROOT/pretrain"
bash "$REPO_ROOT/pretrain/scripts/convert_checkpoint_to_hf.sh" \
  "$BASE_MODEL_DIR" "$MODEL_HOME" "$STEP"
