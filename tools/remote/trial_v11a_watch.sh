#!/usr/bin/env bash
# Watcher for rough_box_trial_v11a (docs/experiments/2026-10-14_rough_box_trial_v11a.md) on the remote
# 3090. Same contract as v11_watch.sh: run with run_in_background; it exits ONLY on events that need
# judgement - the trainer stopped, iteration 4,000 / 8,000 reached (mid-training checks), or the host
# unreachable 3 times in a row. The hourly heartbeat goes to the briefing from HERE, no model woken.
# Usage: bash tools/remote/trial_v11a_watch.sh [next-mark]   (marks: 4000 8000; default = first unpassed)
cd "$(dirname "$0")/../.." || exit 1
H=syaro@10.8.0.14
L='~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out/rough_box_trial_v11a_launch.out'
r() { ssh -o BatchMode=yes -o ConnectTimeout=15 $H "$@"; }
iter() { r "grep -o 'Learning iteration [0-9]*/[0-9]*' $L | tail -1" | grep -o '[0-9]*/' | tr -d /; }
alive() { r 'pgrep -f "[r]ough_box_trial_v11a" >/dev/null && echo yes || echo no'; }
heartbeat() {
  python3 tools/briefing/briefing.py done trial-v11a-heartbeat \
    "v11a 권장안 시험 런 정기 점검 ($(date +%H:%M)) — 원격 3090 학습 중" \
    --finding "자동 점검(스크립트). 진행 중입니다." --number "반복 $1 / 12,000" >/dev/null 2>&1
  ( flock -w 300 9 && git add "docs/000.Real-time Brefing.md" docs/.briefing_state.json \
    && git commit -q -m "brief: trial v11a heartbeat (auto)" ) 9>/home/syaro/pyg_fea/locks/git.lock >/dev/null 2>&1
}
i0=$(iter); mark=${1:-}
[ -z "$mark" ] && for m in 4000 8000 99999; do [ "${i0:-0}" -lt $m ] && { mark=$m; break; }; done
next=$(( $(date +%s) + 3600 )); fails=0
while :; do
  a=$(alive) || a=""
  if [ -z "$a" ]; then fails=$((fails+1)); [ $fails -ge 3 ] && { why="REMOTE UNREACHABLE"; break; }; sleep 120; continue; fi
  fails=0; i=$(iter)
  [ "$a" = no ] && { why="TRAINER STOPPED"; break; }
  [ "${i:-0}" -ge "$mark" ] && { why="MARK $mark REACHED"; break; }
  [ "$(date +%s)" -ge "$next" ] && { heartbeat "$i"; next=$(( $(date +%s) + 3600 )); }
  sleep 300
done
echo "$why $(date +%H:%M) iter ${i:-?}"
r "tail -c 1500 $L"
