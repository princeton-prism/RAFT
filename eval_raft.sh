#!/bin/bash
# Evaluate one checkpoint on every axis of $TASK in sequence, then print the results table.
# scripts/slurm/eval_axes.sh submits one Slurm job per axis instead.
#
# Usage: TASK=pickplace MODEL_DIR=checkpoints/RAFT-pickplace bash eval_raft.sh
set -uo pipefail
source "$(dirname "$0")/scripts/env.sh"
: "${MODEL_DIR:?set MODEL_DIR}"
TAG=${TAG:-$(basename "${MODEL_DIR}")}
for AXIS in ${RAFT_AXES}; do
    AXIS=$AXIS TAG=$TAG bash "${RAFT_HOME}/scripts/eval/eval_axis.sh" || echo "[warn] ${AXIS} exited nonzero; checking results anyway"
done
python3 "${RAFT_HOME}/scripts/results/raft_results.py" "${OUT_ROOT}/eval/${TAG}"
