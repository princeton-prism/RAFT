#!/bin/bash
# Stage I -> flat backbone: fold the reasoning LoRA into MolmoBot-Pi0-DROID.
# The PaliGemma weights round-trip through bf16, so fp32 weights pick up bf16 rounding.
#
# Usage: TASK=pickplace|opendrawer bash scripts/stage1/merge.sh
#        ADAPTER=<adapter dir> to merge an intermediate checkpoint instead of the final one
set -euo pipefail
source "$(dirname "$0")/../env.sh"
ADAPTER=${ADAPTER:-${OUT_ROOT}/stage1_reasoning_lora/final_adapter}
need "${ADAPTER}/adapter_config.json" "Stage-I LoRA adapter"

cd "${RAFT_HOME}"
uv run --no-sync python scripts/stage1/merge_paligemma_lora_into_molmobot.py \
    --molmobot-ckpt "${BASE_CKPT_DIR}/model.safetensors" \
    --lora-adapter "${ADAPTER}" \
    --output-ckpt "${STAGE1_CKPT_DIR}/model.safetensors"
