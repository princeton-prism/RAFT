#!/bin/bash
# Submit one 1-GPU Slurm job per evaluation axis.
#   TASK=pickplace MODEL_DIR=checkpoints/RAFT-pickplace bash scripts/slurm/eval_axes.sh [extra sbatch args, e.g. --partition=...]
set -euo pipefail
: "${MODEL_DIR:?}"
source "$(dirname "$0")/../env.sh"
for AXIS in ${RAFT_AXES}; do
    sbatch -J "raft_${AXIS}" -N 1 -n 1 -c 16 --gres=gpu:1 --mem=160G -t 24:00:00 "$@" \
        --export=ALL,RAFT_HOME="$RAFT_HOME",TASK="$TASK",MODEL_DIR="$MODEL_DIR",AXIS="$AXIS" \
        --wrap "bash $RAFT_HOME/scripts/eval/eval_axis.sh"
done
echo "when done: python3 $RAFT_HOME/scripts/results/raft_results.py $OUT_ROOT/eval/$(basename "$MODEL_DIR")"
