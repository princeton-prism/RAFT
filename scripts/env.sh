#!/bin/bash
# Shared environment for every RAFT launcher (sourced). Override any path by exporting it first.
#   TASK=pickplace (default) | opendrawer
RAFT_HOME=${RAFT_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TASK=${TASK:-pickplace}

CKPT_ROOT=${CKPT_ROOT:-${RAFT_HOME}/checkpoints}
DATA_ROOT=${DATA_ROOT:-${RAFT_HOME}/data}
BASE_CKPT_DIR=${BASE_CKPT_DIR:-${CKPT_ROOT}/MolmoBot-Pi0-DROID}          # allenai/MolmoBot-Pi0-DROID
BENCH_ROOT=${BENCH_ROOT:-${RAFT_HOME}/config/benchmarks/${TASK}}
MOLMOSPACES_DIR=${MOLMOSPACES_DIR:-${RAFT_HOME}/molmospaces}  # git submodule, patched by scripts/setup_molmospaces.sh
export MLSPACES_CACHE_DIR=${MLSPACES_CACHE_DIR:-${HOME}/.cache/molmo-spaces-resources}
# The resource manager takes a write lock in MLSPACES_CACHE_DIR, so it must be writable by you.
# MLSPACES_ASSETS_DIR (optional) is passed through to molmospaces unchanged.
OUT_ROOT=${OUT_ROOT:-${RAFT_HOME}/outputs/${TASK}}

case "${TASK}" in
  pickplace)
    ACTION_DATA=${ACTION_DATA:-${DATA_ROOT}/pickplace/action_train}
    VAL_DATA=${VAL_DATA:-}                                   # empty: val loss on the train pool
    REASONING_DATA=${REASONING_DATA:-${DATA_ROOT}/pickplace/reasoning}
    ;;
  opendrawer)
    ACTION_DATA=${ACTION_DATA:-${DATA_ROOT}/opendrawer/action_train_h95}
    VAL_DATA=${VAL_DATA:-${DATA_ROOT}/opendrawer/action_val_h11}   # held-out val split
    REASONING_DATA=${REASONING_DATA:-${DATA_ROOT}/opendrawer/reasoning_trace_only}
    ;;
  *) echo "unknown TASK=${TASK} (pickplace|opendrawer)" >&2; exit 2 ;;
esac
case "${TASK}" in
  pickplace)  RAFT_AXES="id oodlight texall oodobjv3 oodprompt" ;;
  opendrawer) RAFT_AXES="id oodlight texall oodprompt" ;;
esac
STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-${CKPT_ROOT}/RAFT-stage1-${TASK}}
RAFT_CKPT_DIR=${RAFT_CKPT_DIR:-${CKPT_ROOT}/RAFT-${TASK}}

export USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1

need() { [[ -e "$1" ]] || { echo "missing: $1 ($2)" >&2; exit 2; }; }
