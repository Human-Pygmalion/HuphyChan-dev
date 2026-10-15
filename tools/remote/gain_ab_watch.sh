#!/usr/bin/env bash
# Watcher for the gain-only A/B on the remote 3090 (docs/experiments/2026-10-15_ab_rough_gain_compare.md):
#   A = rough_gain_crank60_v11a, B = rough_gain_mountain_v11a (B waits in a queue until A's trainer exits,
#   see mujoco-sim/mjlab/analysis/gain_ab_queue_b.sh, or runs beside A if memory allowed).
# Same contract as trial_v11a_watch.sh: run it with run_in_background; it exits ONLY on events that need
# judgement - a watched trainer stopped (finished or died; the last log lines say which), either arm reached
# iteration 4,000 or 8,000 (mid-training checks), or the host unreachable 3 times in a row.  The hourly
# heartbeat goes to the briefing from HERE, no model woken.
# "Watched" = alive when the script starts, or B not yet launched.  An arm that already finished is ignored,
# so restarting the script after A's finish waits for B.
# Usage: bash tools/remote/gain_ab_watch.sh [markA] [markB]   (marks: 4000 8000; default = first unpassed)
cd "$(dirname "$0")/../.." || exit 1
H=syaro@10.8.0.14
D='~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out'
A=rough_gain_crank60_v11a
B=rough_gain_mountain_v11a
r() { ssh -o BatchMode=yes -o ConnectTimeout=15 $H "$@"; }
iter() { r "grep -a -o 'Learning iteration [0-9]*/[0-9]*' $D/$1_launch.out 2>/dev/null | tail -1" | grep -o '[0-9]*/' | tr -d /; }
alive() { r "pgrep -f '[a]gent.run-name $1' >/dev/null && echo yes || echo no"; }
firstmark() { for m in 4000 8000 99999; do [ "${1:-0}" -lt $m ] && { echo $m; return; }; done; }
heartbeat() {
  python3 tools/briefing/briefing.py done gain-ab-heartbeat \
    "게인만 바꾼 두 팔 비교 학습 정기 점검 ($(date +%H:%M)) — 원격 3090 학습 중" \
    --finding "자동 점검(스크립트). 진행 중입니다. 팔 A = 크랭크 60/2.7 만, 팔 B = 등반 권장 묶음 (대기 중이면 반복 '-')." \
    --number "팔 A 반복 ${1:--} / 12,000 · 팔 B 반복 ${2:--} / 12,000" >/dev/null 2>&1
  ( flock -w 300 9 && git add "docs/000.Real-time Brefing.md" docs/.briefing_state.json \
    && git commit -q -m "brief: gain A/B heartbeat (auto)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" ) 9>/home/syaro/pyg_fea/locks/git.lock >/dev/null 2>&1
}
ia=$(iter $A); ib=$(iter $B)
aa=$(alive $A) || aa=""; ab=$(alive $B) || ab=""
watchA=0; [ "$aa" = yes ] && watchA=1
watchB=0; if [ "$ab" = yes ] || [ -z "$ib" ]; then watchB=1; fi   # alive, or not launched yet
markA=${1:-$(firstmark "$ia")}; markB=${2:-$(firstmark "$ib")}
next=$(( $(date +%s) + 3600 )); fails=0; why=""
while :; do
  aa=$(alive $A) || aa=""; ab=$(alive $B) || ab=""
  if [ -z "$aa" ] || [ -z "$ab" ]; then fails=$((fails+1)); [ $fails -ge 3 ] && { why="REMOTE UNREACHABLE"; break; }; sleep 120; continue; fi
  fails=0; ia=$(iter $A); ib=$(iter $B)
  [ $watchA = 1 ] && [ "$aa" = no ] && { why="TRAINER A STOPPED ($A)"; break; }
  [ $watchB = 1 ] && [ "$ab" = no ] && [ -n "$ib" ] && { why="TRAINER B STOPPED ($B)"; break; }
  [ -n "$ia" ] && [ "$ia" -ge "$markA" ] && { why="ARM A MARK $markA REACHED"; break; }
  [ -n "$ib" ] && [ "$ib" -ge "$markB" ] && { why="ARM B MARK $markB REACHED"; break; }
  [ "$(date +%s)" -ge "$next" ] && { heartbeat "$ia" "$ib"; next=$(( $(date +%s) + 3600 )); }
  sleep 300
done
echo "$why $(date +%H:%M) iterA ${ia:-?} iterB ${ib:-?}"
r "tail -c 1200 $D/${A}_launch.out; echo ----; tail -c 1200 $D/${B}_launch.out 2>/dev/null; echo ----; tail -5 $D/gain_ab_queue.log 2>/dev/null"
