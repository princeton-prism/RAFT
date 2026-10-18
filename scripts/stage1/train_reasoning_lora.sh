#!/bin/bash
# Stage I: train a reasoning LoRA on the PaliGemma backbone of MolmoBot-Pi0-DROID with next-token
# cross-entropy on (image, instruction) -> reasoning text. Adapters are saved under $OUTPUT_DIR.
#
# Usage: TASK=pickplace|opendrawer bash scripts/stage1/train_reasoning_lora.sh [extra args]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
OUTPUT_DIR=${OUTPUT_DIR:-${OUT_ROOT}/stage1_reasoning_lora}
need "${REASONING_DATA}/train.jsonl" "reasoning JSONL (download, or scripts/stage1/annotate_dataset_paligemma.py)"
need "${BASE_CKPT_DIR}/model.safetensors" "MolmoBot-Pi0-DROID weights"

cd "${RAFT_HOME}"
uv run --no-sync python src/agent/train_reasoning.py \
    --data-root "${REASONING_DATA}" \
    --molmobot-ckpt "${BASE_CKPT_DIR}/model.safetensors" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 12 --grad-accum 4 --num-epochs 10 \
    --lr 2e-4 --warmup-steps 100 \
    --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 \
    --save-steps 500 --seed 42 \
    "$@"
