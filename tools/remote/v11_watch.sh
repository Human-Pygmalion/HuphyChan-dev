#!/usr/bin/env bash
# Watcher for A/B v11 (both arms on the remote 3090). Run with run_in_background; its exit wakes the
# agent, so it exits ONLY on events that need judgement: an arm stopped, a phase event / DONE in either
# launcher log, or the host unreachable 3 times in a row. The hourly heartbeat is written to the
# briefing and committed HERE, by the script - no model is woken for routine progress
# (user 2026-10-02: do not spend Opus on simple repeated work).
cd "$(dirname "$0")/../.." || exit 1
H=syaro@10.8.0.14
M='~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out'
RUNS="legonly_ab_v11a_stand01_s43 legonly_ab_v11b_stand03_s43"
r() { ssh -o BatchMode=yes -o ConnectTimeout=15 $H "$@"; }
count() { r "cat $M/legonly_ab_v11[ab]_*_launch.out | grep -c 'DONE\|\[P2\] launching'"; }
alive() { r "for x in $RUNS; do pgrep -f \"[r]un_v2_scratch.py --run \$x\" >/dev/null && echo \$x; done"; }
last() { r "grep -E '\[P[12]\]' $M/$1_launch.out | tail -1"; }
fmt() {  # launcher line -> short Korean summary
  sed -E -e 's/.*\[P1\] iter ([0-9]+) stage ([0-9.]+) dwell ([0-9]+) err ([0-9.]+) base ([0-9.na]+) fell ([0-9.]+).*/1단계 \1 회 · \2 단계 · 체류 \3 · 오차 \4 · 넘어짐 \6/' \
         -e 's/.*\[P2\] iter ([0-9]+)\/([0-9]+) dr_factor ([0-9.]+).*/2단계 \1 \/ \2 회 · 무작위화 \3/'; }
heartbeat() {
  A=$(last legonly_ab_v11a_stand01_s43 | fmt); B=$(last legonly_ab_v11b_stand03_s43 | fmt)
  python3 tools/briefing/briefing.py done v11-heartbeat "A/B v11 정기 점검 ($(date +%H:%M)) — 두 팔 학습 중 (원격 3090)" \
    --finding "자동 점검(스크립트). 두 팔 모두 진행 중입니다." --number "v11a: $A · v11b: $B" >/dev/null 2>&1
  ( flock -w 300 9 && git add "docs/000.Real-time Brefing.md" docs/.briefing_state.json \
    && git commit -q -m "brief: v11 heartbeat (auto)" ) 9>/home/syaro/pyg_fea/locks/git.lock >/dev/null 2>&1
}
E0=$(count); next=$(( $(date +%s) + 3600 )); fails=0
while :; do
  a=$(alive) || { fails=$((fails+1)); [ $fails -ge 3 ] && { why="REMOTE UNREACHABLE"; break; }; sleep 120; continue; }
  fails=0
  [ "$(echo "$a" | grep -c v11)" -lt 2 ] && { why="ARM STOPPED (alive: ${a:-none})"; break; }
  [ "$(count)" != "$E0" ] && { why="PHASE EVENT"; break; }
  [ "$(date +%s)" -ge "$next" ] && { heartbeat; next=$(( $(date +%s) + 3600 )); }
  sleep 120
done
echo "$why $(date +%H:%M)"
for x in $RUNS; do last $x; done
