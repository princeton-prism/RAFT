#!/bin/bash
# Install the simulator: check out the molmospaces submodule at the pinned commit and apply
# scripts/molmospaces_raft.patch (lighting/texture randomization flags for JSON benchmarks,
# per-episode sharding, open-task and JSON-eval fixes).
#
# Usage: bash scripts/setup_molmospaces.sh   (inits + patches the submodule, `uv sync --extra eval`,
#                                            openpi transformers overrides)
set -euo pipefail
source "$(dirname "$0")/env.sh"
COMMIT=cd23becebcf72dd93a4aa5872a60802d5eff03ef

if [[ "${MOLMOSPACES_DIR}" == "${RAFT_HOME}/molmospaces" && -d "${RAFT_HOME}/.git" ]]; then
    git -C "${RAFT_HOME}" submodule update --init molmospaces
elif [[ ! -e "${MOLMOSPACES_DIR}/.git" ]]; then
    git clone https://github.com/allenai/molmospaces.git "${MOLMOSPACES_DIR}"
fi
cd "${MOLMOSPACES_DIR}"
git checkout --quiet "${COMMIT}"
PATCH="${RAFT_HOME}/scripts/molmospaces_raft.patch"
if patch -p1 -R --dry-run -s -f < "${PATCH}" >/dev/null 2>&1; then
    echo "patch already applied"
else
    patch -p1 -s < "${PATCH}"     # plain `diff -ruN` patch: includes new files, so use patch(1)
    echo "applied molmospaces_raft.patch on ${COMMIT}"
fi
cd "${RAFT_HOME}"
[[ "${MOLMOSPACES_DIR}" == "${RAFT_HOME}/molmospaces" ]] || \
    echo "NOTE: pyproject [tool.uv.sources] points at ./molmospaces; symlink it to ${MOLMOSPACES_DIR}"
uv sync --extra eval
# openpi's PyTorch pi0 needs its patched gemma/paligemma/siglip modules copied into transformers.
uv run --no-sync python scripts/install_openpi_extras.py
echo "molmospaces ready at ${MOLMOSPACES_DIR}; assets cache: ${MLSPACES_CACHE_DIR}"
