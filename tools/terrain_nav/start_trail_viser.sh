#!/usr/bin/env bash
# Drive a walking policy yourself on the mountain-trail terrain in viser (2026-10-12, user request).
# Reuses pygviewer: the baked trail model (10 cm heightfield + tree trunks = the colliders), the policy panel
# (vx / vy / wz sliders, mode, reset), drawn terrain thinned to 30 cm for the browser (collision unchanged).
#   bash tools/terrain_nav/start_trail_viser.sh [policy name in the cache, default roughmain] [port, default 8098]
# then open http://<this host>:8098  - Policy folder: vx / wz sliders; "reset to" puts the robot back up.
cd "$(dirname "$0")/../.." || exit 1
POL=${1:-roughmain}; PORT=${2:-8098}; CACHE=${CACHE:-$HOME/pyg_demo/cache_trail}
PYG_VIEW_CACHE=$CACHE setsid nohup mujoco-sim/mjlab/.venv/bin/python3 tools/pygviewer/run.py --variant LegOnly-AB \
  --cache "$CACHE" --port "$PORT" --no-api --host 0.0.0.0 --base free --policy "$POL" \
  > "$HOME/pyg_demo/trail_viser.log" 2>&1 < /dev/null &
echo "started (log ~/pyg_demo/trail_viser.log)"
