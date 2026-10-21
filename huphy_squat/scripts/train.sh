#!/usr/bin/env bash
# Train Huphy-Squat-Flat-v0 with stock mjlab `train` (5,001 iterations, 4096 envs, wandb project huphy_squat).
#   scripts/train.sh [RUN_NAME] [extra mjlab train args, e.g. --agent.max-iterations 2000]
# Logs/checkpoints: huphy_squat/logs/rsl_rl/huphy_squat/<date>_<RUN_NAME>/  (model_<it>.pt every 500 it)
# Offline wandb (no upload): WANDB_MODE=offline scripts/train.sh ...
set -euo pipefail
cd "$(dirname "$0")/.."
MJLAB_VENV=${MJLAB_VENV:-$HOME/huphy/mjlab/.venv}
RUN=${1:-main}
shift || true
exec "$MJLAB_VENV/bin/train" Huphy-Squat-Flat-v0 --env.scene.num-envs 4096 --agent.run-name "$RUN" "$@"
