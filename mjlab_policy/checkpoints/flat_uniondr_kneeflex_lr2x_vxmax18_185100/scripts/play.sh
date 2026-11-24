#!/bin/bash
set -euo pipefail

POLICY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${MJLAB_PYTHON:-python}"
cd "$POLICY_ROOT"

exec env PYTHONPATH="$POLICY_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" scripts/play.py \
  Pygmalion-Velocity-Flat-UnionDR-FullPlanar-Clearance10-KneeFlex-LR2x \
  --checkpoint-file checkpoints/flat_uniondr_kneeflex_lr2x_vxmax18_185100/model_185100.pt \
  "$@"
