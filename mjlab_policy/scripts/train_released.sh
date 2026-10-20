#!/bin/bash
# Train Pygmalion-Velocity-Rough-Box the way the released policy was trained
# (actor-only warm start from the flat parent, 300-update critic warm-up), with training video on.
#
#   bash scripts/train_released.sh [max_iterations=12000] [num_envs=4096] [run_name=released] [extra args...]
#
# Extra args go to mjlab / scripts/train.py, e.g. the optional randomization
# --obs-delay 1 --act-delay 1 --gain-dr 0.8 1.2 (all off by default; README section 9).
#
# Set MJLAB_PYTHON to the python that has mjlab installed (default: python).
# Logs: logs/rsl_rl/pygmalion_velocity/<date>_<run_name>/ (checkpoints, params/, videos/train/).
set -euo pipefail
cd "$(dirname "$0")/.."
ITER="${1:-12000}"; ENVS="${2:-4096}"; RUN="${3:-released}"; shift $(( $# < 3 ? $# : 3 ))
PY="${MJLAB_PYTHON:-python}"
# One video clip every 200 iterations (24 env steps per iteration).
exec "$PY" scripts/train.py Pygmalion-Velocity-Rough-Box \
  --warmstart-actor checkpoints/parent_flat_actor_model_29998.pt --critic-warmup 300 \
  --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITER" --agent.seed 43 \
  --agent.run-name "$RUN" --video True --video-interval 4800 --video-length 500 "$@"
