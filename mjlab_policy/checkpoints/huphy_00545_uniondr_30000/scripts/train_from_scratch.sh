#!/usr/bin/env bash
set -euo pipefail

# The executable source snapshot must first be overlaid onto UNITREE_RL_MJLAB.
UNITREE_RL_MJLAB=${UNITREE_RL_MJLAB:?Set UNITREE_RL_MJLAB to the unitree_rl_mjlab checkout}
MJLAB_PYTHON=${MJLAB_PYTHON:-/home/jooyoungbae/miniconda3/envs/mjlab/bin/python}
GPU=${GPU:-0}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-30001}
RUN_NAME=${RUN_NAME:-huphy_00545_uniondr_footfric03to17_fromscratch_4096_reproduce}

PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOTION_FILE=${MOTION_FILE:-$PACKAGE_DIR/motions/00545_interactionmesh_50hz.npz}

cd "$UNITREE_RL_MJLAB"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$MJLAB_PYTHON" scripts/train.py \
  HUPHY-Tracking-ProjectedGravity-UnionDR \
  --motion-file "$MOTION_FILE" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.seed 42 \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.resume False \
  --agent.run-name "$RUN_NAME" \
  --gpu-ids '[0]'
