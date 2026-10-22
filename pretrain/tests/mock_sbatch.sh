#!/usr/bin/env bash
set -euo pipefail

: "${SBATCH_ARGS:?Set SBATCH_ARGS to capture mock sbatch arguments.}"
printf '%s\n' "$@" > "$SBATCH_ARGS"
env > "$SBATCH_ARGS.env"
echo 'Submitted batch job 4242'
