#!/bin/bash
# Build valid_trajectory_index.json + aggregated_stats.json for an action dataset in the
# MolmoSpaces h5 + mp4 layout.
#
# Usage: bash scripts/data/preprocess_action_data.sh [ACTION_DATA]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
DATA=${1:-${ACTION_DATA}}
cd "${MOLMOSPACES_DIR}"
uv run --project "${RAFT_HOME}" --no-sync python scripts/data/repair_video_paths.py "${DATA}" --num-workers "${NUM_WORKERS:-8}"
uv run --project "${RAFT_HOME}" --no-sync python scripts/data/validate_trajectories.py "${DATA}" --num-workers "${NUM_WORKERS:-8}" --no-video
uv run --project "${RAFT_HOME}" --no-sync python scripts/data/calculate_stats.py "${DATA}" --num-workers "${NUM_WORKERS:-8}" \
    --keys actions/joint_pos actions/joint_pos_rel obs/agent/qpos
ls -la "${DATA}/valid_trajectory_index.json" "${DATA}/aggregated_stats.json"
