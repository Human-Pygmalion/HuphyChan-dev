#!/usr/bin/env bash
# The article shots (hero + top-down) on a black and on a white background, for compose_backgrounds.py.
# 3090, in ~/pyg_demo/terrain (needs capture_terrain.sh, terrain.rviz, terrain_top.rviz there).  ~4 min.
# usage: bash capture_bg_pair.sh [out dir, default nosurf]
set -uo pipefail
cd ~/pyg_demo/terrain
OUT=${1:-nosurf}
mkdir -p "$OUT"
for c in 0 255; do
  for v in terrain terrain_top; do sed "s/Background Color: 12; 15; 20/Background Color: $c; $c; $c/" $v.rviz > ${v}_bg$c.rviz; done
  timeout 300 bash capture_terrain.sh terrain_bg$c.rviz "$OUT/hero_bg$c.png" 45 > "$OUT/cap_hero_$c.log" 2>&1
  PUB_ARGS="--dots --corridor 6" timeout 300 bash capture_terrain.sh terrain_top_bg$c.rviz "$OUT/top_bg$c.png" 45 > "$OUT/cap_top_$c.log" 2>&1
done
ls -la "$OUT"/*_bg*.png
