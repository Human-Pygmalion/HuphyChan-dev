#!/usr/bin/env bash
set -euo pipefail

UNITREE_RL_MJLAB=${UNITREE_RL_MJLAB:?Set UNITREE_RL_MJLAB to the unitree_rl_mjlab checkout}
MJLAB_PYTHON=${MJLAB_PYTHON:-/home/jooyoungbae/miniconda3/envs/mjlab/bin/python}
GPU=${GPU:-0}
NUM_ENVS=${NUM_ENVS:-4096}
# The original launch budget was 17,001 additional iterations, but the released
# run was intentionally stopped at iteration 20,000 (+7,000 from the parent).
MAX_ITERATIONS=${MAX_ITERATIONS:-7001}
RUN_NAME=${RUN_NAME:-huphy_00399_uniondr_resume13000_reproduce}

PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOTION_FILE=${MOTION_FILE:-$PACKAGE_DIR/motions/00399_interactionmesh_50hz.npz}
PARENT_RUN=${PARENT_RUN:-huphy_00399_parent_model_13000}
PARENT_DIR="$UNITREE_RL_MJLAB/logs/rsl_rl/huphy_tracking/$PARENT_RUN"
mkdir -p "$PARENT_DIR"
cp "$PACKAGE_DIR/model_13000.pt" "$PARENT_DIR/model_13000.pt"

cd "$UNITREE_RL_MJLAB"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$MJLAB_PYTHON" scripts/train.py \
  HUPHY-Tracking-ProjectedGravity-UnionDR \
  --motion-file "$MOTION_FILE" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.seed 42 \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.resume True \
  --agent.load-run "$PARENT_RUN" \
  --agent.load-checkpoint model_13000.pt \
  --agent.run-name "$RUN_NAME" \
  --gpu-ids '[0]'
