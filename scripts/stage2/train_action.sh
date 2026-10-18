#!/bin/bash
# Stage II: full action finetune (pi0 flow matching) from a flat backbone, vision tower frozen.
# Optimizer, schedule and model settings come from the molmobot_pi0_droid train config
# (src/model/config_openpi.py). The last checkpoint is linked as <run>/final.
#
# Usage:
#   TASK=pickplace NUM_TRAIN_STEPS=<steps> bash scripts/stage2/train_action.sh            # from Stage I
#   INIT_CKPT_DIR=$CKPT_ROOT/MolmoBot-Pi0-DROID EXP_NAME=plainft NUM_TRAIN_STEPS=<steps> bash ...  # plain FT
set -euo pipefail
source "$(dirname "$0")/../env.sh"
INIT_CKPT_DIR=${INIT_CKPT_DIR:-${STAGE1_CKPT_DIR}}
EXP_NAME=${EXP_NAME:-raft_stage2_${TASK}}
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}                       # global, split across GPUs
export MOLMOBOT_GRAD_ACCUM=${MOLMOBOT_GRAD_ACCUM:-8}
: "${NUM_TRAIN_STEPS:?set NUM_TRAIN_STEPS}"
export FREEZE_FILTER=${FREEZE_FILTER:-vision_tower}
# Checkpoint retention: keep the latest, the 2 best by validation loss, and the final step.
export KEEP_BEST_CHECKPOINTS=${KEEP_BEST_CHECKPOINTS:-2} VAL_BATCHES=${VAL_BATCHES:-8}
export KEEP_STEPS=${KEEP_STEPS:-${NUM_TRAIN_STEPS}}
if [[ -n "${VAL_DATA}" ]]; then
    need "${VAL_DATA}/valid_trajectory_index.json" "held-out val split"
    export VAL_DATA_ROOT=${VAL_DATA} VAL_INTERVAL=${VAL_INTERVAL:-50}
fi
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

need "${ACTION_DATA}/aggregated_stats.json" "preprocessed action data (scripts/data/preprocess_action_data.sh)"
need "${INIT_CKPT_DIR}/model.safetensors" "Stage-II init checkpoint"
echo "init=${INIT_CKPT_DIR} exp=${EXP_NAME} effective_batch=$((BATCH_SIZE * MOLMOBOT_GRAD_ACCUM)) gpus=${NUM_GPUS}"

cd "${RAFT_HOME}"
uv run --no-sync torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    src/agent/train_action.py molmobot_pi0_droid \
    --data.repo-id "${ACTION_DATA}" \
    --data.exo-camera randomized_zed2_analogue_1 \
    --data.wrist-camera wrist_camera_zed_mini \
    --pytorch-weight-path "${INIT_CKPT_DIR}" \
    --exp-name "${EXP_NAME}" \
    --checkpoint-base-dir "${OUT_ROOT}/train_output" \
    --project-name raft \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --save-interval "${SAVE_INTERVAL:-500}" \
    --log-interval 10 \
    "$@"

# Link the last checkpoint as <run>/final.
RUN_DIR=${OUT_ROOT}/train_output/molmobot_pi0_droid/${EXP_NAME}
ln -sfn "${NUM_TRAIN_STEPS}" "${RUN_DIR}/final"
echo "final checkpoint: ${RUN_DIR}/final"
