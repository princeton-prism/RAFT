#!/bin/bash
# Evaluate one checkpoint on one axis: picks the benchmark and randomization flags for the axis.
#
#   TASK=pickplace                                   TASK=opendrawer
#   axis       benchmark    randomization            axis       benchmark   randomization
#   id         id/          none                     id         id/         none
#   oodlight   id/          lighting                 oodlight   id/         lighting
#   texall     id/          all textures (no light)  texall     id/         all textures (no light)
#   oodobjv3   oodobj/      none (novel objects)
#   oodprompt  oodprompt/   none (rephrased)         oodprompt  oodprompt/  none (rephrased)
#
# Usage: TASK=pickplace MODEL_DIR=checkpoints/RAFT-pickplace AXIS=oodobjv3 bash scripts/eval/eval_axis.sh
#        (MODEL_DIR = a checkpoint dir with model.safetensors + assets/)
# Output: $OUT_DIR (default $OUT_ROOT/eval/<tag>/<AXIS>), read by scripts/results/raft_results.py
set -euo pipefail
source "$(dirname "$0")/../env.sh"
: "${MODEL_DIR:?set MODEL_DIR}" "${AXIS:?set AXIS}"
TAG=${TAG:-$(basename "${MODEL_DIR}")}
OUT_DIR=${OUT_DIR:-${OUT_ROOT}/eval/${TAG}/${AXIS}${SUFFIX:-}}

bench=${BENCH_ROOT}/id light=false tex=false texall=false workers=3
case "${TASK}:${AXIS}" in
    pickplace:id)          workers=4 ;;
    pickplace:oodlight)    light=true ;;
    pickplace:texall)      tex=true; texall=true ;;
    pickplace:oodobjv3)    bench=${BENCH_ROOT}/oodobj ;;
    pickplace:oodprompt)   bench=${BENCH_ROOT}/oodprompt ;;
    opendrawer:id)         ;;
    opendrawer:oodlight)   light=true; workers=4 ;;
    opendrawer:texall)     tex=true; texall=true; workers=4 ;;
    opendrawer:oodprompt)  bench=${BENCH_ROOT}/oodprompt; workers=4 ;;
    *) echo "unknown axis ${AXIS} for TASK=${TASK}" >&2; exit 2 ;;
esac
need "${MODEL_DIR}/model.safetensors" "checkpoint"
MODEL_DIR=$(cd "${MODEL_DIR}" && pwd)
need "${bench}/benchmark.json" "benchmark"
mkdir -p "${OUT_DIR}"

# NUM_WORKERS is a GPU-memory budget: each worker holds its own copy of the policy.
cd "${RAFT_HOME}"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --no-sync python -m src.agent.eval \
    policy=pi \
    policy.action_type=jointpos \
    policy.model_name=null \
    policy.checkpoint_dir="${MODEL_DIR}" \
    benchmark="${bench}" \
    horizon=40.0 \
    num_workers="${NUM_WORKERS:-${workers}}" \
    n_videos="${N_VIDEOS:-0}" \
    randomize_lighting=${light} \
    randomize_textures=${tex} \
    +randomize_textures_all=${texall} \
    hydra.run.dir="${OUT_DIR}" \
    ${EXTRA_HYDRA:-}
# The runner can exit nonzero on a temp-dir cleanup race AFTER writing results; trust only the
# "Success count:" line in ${OUT_DIR}/**/running_log.log (raft_results.py does this).
