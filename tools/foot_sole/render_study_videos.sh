#!/bin/bash
# Qualitative comparison videos for the foot-sole study (run ON the 3090 after the evaluations; skips videos already made).
#   bash tools/foot_sole/render_study_videos.sh [parallel=3]
# Every video replays the recorded evaluation (env 0) of each run on the same clock, all 9 commands x 15 s, real time at 25 fps
# (stride 2 of the 50 Hz policy steps). Tiles show the run, the command and the right-foot load.
set -uo pipefail
cd "$HOME/MikuchanRemote/Human-Pygmalion"
W=Feet_design/Feet_Design_Workspace/mjcf_sim/study
EV=$W/eval
OUT=$W/videos
mkdir -p "$OUT"
PAR="${1:-3}"
PY=mujoco-sim/mjlab/.venv/bin/python3
job() {  # name view terrain tag:label ...
  local name=$1 view=$2 ter=$3; shift 3
  local tags=() labs=()
  for tl in "$@"; do
    local t="${tl%%:*}_${ter}"
    [ -f "$EV/$t.npz" ] || { echo "skip $name ($t missing)"; return; }
    tags+=("$t"); labs+=("${tl#*:}")
  done
  [ -f "$OUT/$name.mp4" ] && return
  MUJOCO_GL=egl $PY tools/foot_sole/render_eval_video.py --eval-dir "$EV" --tags "${tags[@]}" --labels "${labs[@]}" --view "$view" \
    --stride 2 --size 480x360 --out "$OUT/$name.mp4" > "$OUT/$name.log" 2>&1 && echo "made $name"
}
S0=("orig_49999:box foot, no fine-tune" "base_52498:box foot +2500" "v3full_52498:v3 +2500" "v5full_52498:v5 +2500" "v5fullm_52498:v5 CAD mass +2500")
REF="v5fullm_51000:v5 (reference)"
HEEL=("v5_hf10_51000:heel flat x10" "$REF" "v5_hf30_51000:heel flat x30" "v5_hr2_51000:heel rise 2" "v5_hr5_51000:heel rise 5")
TOE=("v5_tf200_51000:toe flat x200" "$REF" "v5_tf230_51000:toe flat x230" "v5_tr3_51000:toe rise 3" "v5_tr8_51000:toe rise 8")
ARCH=("v5_a0_51000:arch 0" "v5_a2_51000:arch 2" "$REF" "v5_a6_51000:arch 6" "v5_a9_51000:arch 9" "v5_a12_51000:arch 12")
ARCHS=("$REF" "v5_ax130_51000:apex x130" "v5_span80_51000:span 80-145" "v5_span40_51000:span 40-185" "v5_ao40_51000:cone apex 40" "v5_ao160_51000:cone apex 160")
OTHER=("base_51000:box foot" "$REF" "v5_w13_51000:rubber -13 mm" "v5_softarch_51000:soft arch" "v5_human_51000:human proportions")
{
  for ter in flat ridge slope_rough pebbles stones_lo stones; do echo "stage0_side_$ter side $ter S0"; done
  for ter in flat pebbles stones_lo; do echo "stage0_foot_$ter foot $ter S0"; done
  for g in HEEL TOE ARCH ARCHS OTHER; do
    for ter in flat stones_lo; do echo "sweep_${g,,}_foot_$ter foot $ter $g"; done
    echo "sweep_${g,,}_side_pebbles side pebbles $g"
  done
} | while read -r name view ter grp; do
  unset -n arr 2>/dev/null; declare -n arr="$grp"
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
  job "$name" "$view" "$ter" "${arr[@]}" &
done
wait
echo "videos done: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l)"
