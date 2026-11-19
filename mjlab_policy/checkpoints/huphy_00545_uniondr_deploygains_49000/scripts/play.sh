#!/usr/bin/env bash
set -euo pipefail
UNITREE_RL_MJLAB=${UNITREE_RL_MJLAB:?Set UNITREE_RL_MJLAB}
MJLAB_PYTHON=${MJLAB_PYTHON:-python}
GPU=${GPU:-0}
VISER_PORT=${VISER_PORT:-8082}
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$UNITREE_RL_MJLAB"
exec env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$MJLAB_PYTHON" scripts/play.py \
  HUPHY-Tracking-ProjectedGravity-UnionDR-DeployGains \
  --motion-file "$PACKAGE_DIR/motions/00545_interactionmesh_50hz.npz" \
  --checkpoint-file "$PACKAGE_DIR/model_49000.pt" \
  --num-envs 1 --device cuda:0 --viewer viser --viser-port "$VISER_PORT" "$@"
