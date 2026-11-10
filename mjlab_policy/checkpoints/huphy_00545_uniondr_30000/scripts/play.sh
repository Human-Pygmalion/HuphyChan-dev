#!/usr/bin/env bash
set -euo pipefail

UNITREE_RL_MJLAB=${UNITREE_RL_MJLAB:?Set UNITREE_RL_MJLAB to the unitree_rl_mjlab checkout}
MJLAB_PYTHON=${MJLAB_PYTHON:-/home/jooyoungbae/miniconda3/envs/mjlab/bin/python}
GPU=${GPU:-0}
VIEWER=${VIEWER:-viser}
VISER_PORT=${VISER_PORT:-8081}

PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$UNITREE_RL_MJLAB"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$MJLAB_PYTHON" scripts/play.py \
  HUPHY-Tracking-ProjectedGravity-UnionDR \
  --motion-file "$PACKAGE_DIR/motions/00545_interactionmesh_50hz.npz" \
  --checkpoint-file "$PACKAGE_DIR/model_30000.pt" \
  --viewer "$VIEWER" \
  --viser-port "$VISER_PORT" \
  --gpu-ids '[0]' \
  "$@"
