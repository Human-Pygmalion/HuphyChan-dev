#!/usr/bin/env bash
# Generic watcher for ONE training run on the remote 3090 (same contract as gain_ab_watch.sh):
# run with run_in_background; exits only when the run reaches the next mark (4000 / 8000), its trainer
# stops, or the host is unreachable 3 times. Hourly heartbeat goes to the briefing from here.
# Usage: bash tools/remote/run_watch.sh <run-name> [mark]
cd "$(dirname "$0")/../.." || exit 1
RUN=$1; H=syaro@10.8.0.14
D='~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out'
r() { ssh -o BatchMode=yes -o ConnectTimeout=15 $H "$@"; }
iter() { r "grep -a -o 'Learning iteration [0-9]*/[0-9]*' \$(ls -t $D/${RUN}_resume.out $D/${RUN}_launch.out 2>/dev/null | head -1) 2>/dev/null | tail -1" | grep -o '[0-9]*/' | tr -d /; }
alive() { r "pgrep -f '[a]gent.run-name $RUN' >/dev/null && echo yes || echo no"; }
i0=$(iter); mark=${2:-}
[ -z "$mark" ] && for m in 4000 8000 99999; do [ "${i0:-0}" -lt $m ] && { mark=$m; break; }; done
next=$(( $(date +%s) + 3600 )); fails=0
while :; do
  a=$(alive) || a=""
  if [ -z "$a" ]; then fails=$((fails+1)); [ $fails -ge 3 ] && { why="REMOTE UNREACHABLE"; break; }; sleep 120; continue; fi
  fails=0; i=$(iter)
  [ "$a" = no ] && { why="TRAINER STOPPED"; break; }
  [ "${i:-0}" -ge "$mark" ] && { why="MARK $mark REACHED"; break; }
  if [ "$(date +%s)" -ge "$next" ]; then
    python3 tools/briefing/briefing.py done "${RUN}-heartbeat" "$RUN 정기 점검 ($(date +%H:%M)) — 원격 3090 학습 중" \
      --finding "자동 점검(스크립트). 진행 중입니다." --number "반복 $i / 12,000" >/dev/null 2>&1
    ( flock -w 300 9 && git add "docs/000.Real-time Brefing.md" docs/.briefing_state.json \
      && git commit -q -m "brief: $RUN heartbeat (auto)" ) 9>/home/syaro/pyg_fea/locks/git.lock >/dev/null 2>&1
    next=$(( $(date +%s) + 3600 ))
  fi
  sleep 300
done
echo "$why $(date +%H:%M) $RUN iter ${i:-?}"
r "tail -c 1200 \$(ls -t $D/${RUN}_resume.out $D/${RUN}_launch.out 2>/dev/null | head -1)"
