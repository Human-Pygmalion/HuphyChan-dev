#!/usr/bin/env bash
# One watcher for both A/B v10 arms (run with run_in_background so its exit wakes the agent).
# Exits on the first of:
#   - v10a (local) trainer gone, or its launcher printed a phase change / DONE
#   - v10b (remote 10.8.0.14) trainer gone, or "[P2] DONE"/"ALL DONE"
#   - 60 min heartbeat
# Then prints the latest line of both launchers. `pgrep -f "[t]..."` so the check never
# counts itself.
M=/home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
A_OUT=$M/analysis/out/v10a_launch.out
A0=$(grep -c "DONE\|\[P2\] launching" "$A_OUT")
end=$(( $(date +%s) + 3600 )); why=HEARTBEAT
while [ "$(date +%s)" -lt "$end" ]; do
  if ! pgrep -f "[t]rain_wandb_video.py.*v10a" >/dev/null && ! pgrep -f "[r]un_v2_scratch.py --run legonly_ab_v10a" >/dev/null; then why="v10a STOPPED"; break; fi
  [ "$(grep -c "DONE\|\[P2\] launching" "$A_OUT")" != "$A0" ] && { why="v10a PHASE EVENT"; break; }
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 syaro@10.8.0.14 \
     'grep -q "\[P2\] DONE\|ALL DONE" ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out/v10b_launch.out && echo DONE; pgrep -f "[t]rain_wandb_video.py" >/dev/null || echo GONE' 2>/dev/null) || r="SSH_FAIL"
  case "$r" in *DONE*) why="v10b DONE"; break;; *GONE*) why="v10b TRAINER GONE"; break;; esac
  sleep 120
done
echo "$why $(date +%H:%M)"
grep -E "\[P[12]\] iter" "$A_OUT" | tail -1
ssh -o BatchMode=yes -o ConnectTimeout=15 syaro@10.8.0.14 'grep -E "\[P[12]\] iter|DONE" ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out/v10b_launch.out | tail -2' 2>/dev/null || echo "v10b: ssh failed"
