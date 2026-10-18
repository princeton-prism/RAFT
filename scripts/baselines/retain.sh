#!/bin/bash
# RETAIN baseline (arXiv:2512.08333): interpolate the finetuned LLM weights back toward the base,
# theta_llm = (1-a)*base + a*ft; the vision tower and action expert keep their finetuned weights.
#
# Usage: FINETUNED=<plain-FT checkpoint dir> ALPHA=<a> bash scripts/baselines/retain.sh
#        then eval with MODEL_DIR=$CKPT_ROOT/RAFT-baseline-retain-$TASK/alpha$ALPHA
set -euo pipefail
source "$(dirname "$0")/../env.sh"
: "${FINETUNED:?set FINETUNED to a plain-FT checkpoint dir}" "${ALPHA:?set ALPHA, the weight a on the finetuned LLM}"
cd "${RAFT_HOME}"
uv run --no-sync python scripts/baselines/merge_retain_pi0.py \
    --base "${BASE_CKPT_DIR}/model.safetensors" \
    --finetuned "${FINETUNED}" \
    --alpha "${ALPHA}" \
    --output-dir "${CKPT_ROOT}/RAFT-baseline-retain-${TASK}/alpha${ALPHA}"
