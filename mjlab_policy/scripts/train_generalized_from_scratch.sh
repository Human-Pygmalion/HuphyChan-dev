#!/bin/bash
# New one-stage training. This file only defines the command; it is not run automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${MJLAB_PYTHON:-python}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERS="${MAX_ITERS:-50000}"

"$PY" scripts/train.py Pygmalion-Velocity-Generalized-Terrain10 \
  --agent.seed 43 \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.run-name generalized_terrain10_from_scratch
