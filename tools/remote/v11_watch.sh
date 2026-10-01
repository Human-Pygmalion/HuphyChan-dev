#!/usr/bin/env bash
# Watcher for A/B v11 (both arms on the remote 3090). Run with run_in_background so its exit wakes
# the agent. Exits on the first of: an arm's trainer AND launcher both gone, a phase event / DONE in
# either launcher log, ssh unreachable 3 times in a row, or the 60 min heartbeat. Then prints the
# latest launcher line of both arms. `pgrep -f "[r]..."` so the check never counts itself.
H=syaro@10.8.0.14
M='~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out'
RUNS="legonly_ab_v11a_stand01_s43 legonly_ab_v11b_stand03_s43"
count() { ssh -o BatchMode=yes -o ConnectTimeout=15 $H "cat $M/legonly_ab_v11[ab]_*_launch.out | grep -c 'DONE\|\[P2\] launching'"; }
alive() { ssh -o BatchMode=yes -o ConnectTimeout=15 $H "for r in $RUNS; do pgrep -f \"[r]un_v2_scratch.py --run \$r\" >/dev/null && echo \$r; done"; }
E0=$(count); end=$(( $(date +%s) + 3600 )); why=HEARTBEAT; fails=0
while [ "$(date +%s)" -lt "$end" ]; do
  a=$(alive) || { fails=$((fails+1)); [ $fails -ge 3 ] && { why="REMOTE UNREACHABLE"; break; }; sleep 120; continue; }
  fails=0
  [ "$(echo "$a" | grep -c v11)" -lt 2 ] && { why="ARM STOPPED (alive: ${a:-none})"; break; }
  [ "$(count)" != "$E0" ] && { why="PHASE EVENT"; break; }
  sleep 120
done
echo "$why $(date +%H:%M)"
for r in $RUNS; do ssh -o BatchMode=yes -o ConnectTimeout=15 $H "grep -E '\[P[12]\]' $M/${r}_launch.out | tail -1"; done
