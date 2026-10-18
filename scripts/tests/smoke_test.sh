#!/bin/bash
# End-to-end check: evaluate a checkpoint on the first N episodes of the id benchmark and fail
# unless all N ran. Checks install, assets, checkpoint loading and the simulator, not success rates.
#
# Usage: TASK=pickplace MODEL_DIR=checkpoints/RAFT-pickplace N=8 bash scripts/tests/smoke_test.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
: "${MODEL_DIR:?}"
N=${N:-8}
SMOKE_BENCH=${OUT_ROOT}/smoke_bench
mkdir -p "${SMOKE_BENCH}/id"
python3 - "${BENCH_ROOT}/id/benchmark.json" "${SMOKE_BENCH}/id/benchmark.json" "${N}" <<'EOF'
import json, sys
eps = json.load(open(sys.argv[1]))
json.dump(eps[: int(sys.argv[3])], open(sys.argv[2], "w"))
print(f"smoke benchmark: {sys.argv[3]} of {len(eps)} episodes")
EOF
BENCH_ROOT=${SMOKE_BENCH} AXIS=id TAG=smoke NUM_WORKERS=${NUM_WORKERS:-2} \
    bash "${RAFT_HOME}/scripts/eval/eval_axis.sh" || echo "[warn] eval exited nonzero; checking results"
line=$(grep -rh "Success count" "${OUT_ROOT}/eval/smoke/id" --include=running_log.log | tail -1 || true)
echo "${line:-no result line}"
total=$(sed -n 's/.*Total count: \([0-9]*\).*/\1/p' <<<"${line}")
# Pass only if every episode actually ran; rollout errors are logged but leave Total count short.
[[ "${total:-0}" -eq "${N}" ]] || { echo "SMOKE FAIL: ${total:-0} of ${N} episodes completed (see rollout errors above)"; exit 1; }
echo "SMOKE OK (${N} episodes ran; success rate is not checked here)"
