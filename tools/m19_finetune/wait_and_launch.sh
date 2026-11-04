#!/bin/bash
# Wait on the 3090 until a training slot AND enough video memory are free, then start the M19 fine-tune.
# Run ON the 3090, detached:  setsid nohup bash wait_and_launch.sh <run_name> [more_iterations=2500] > logs/wait.log 2>&1 < /dev/null &
#
# Why it looks like this (user decision 2026-11-05: "비면 그때 받아서 이어서 진행"):
#  - The foot-sole session runs an overnight queue (tools/foot_sole/overnight_queue.sh, 2 training slots). That queue counts a slot as
#    "a python3 process whose environment has RUN_NAME=sole_ft_*" and starts the next arm only when a slot is free and >= 9000 MiB is free.
#  - Our run needs ~12.5 GB (2,048 envs, measured 6.4 GB at 1,024 envs). Three trainings at once ran out of memory on 2026-11-01.
#  - So: we start only when fewer than 2 such trainings run and >= NEED_MIB is free, twice in a row, and we mark our process with
#    RUN_NAME=sole_ft_m19_finetune so the queue counts it as one of its two slots (it then waits instead of piling on). Nothing of theirs is
#    stopped or edited.
set -uo pipefail
RUN="${1:?run name}"; MORE="${2:-2500}"
NEED_MIB="${NEED_MIB:-13000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
running() { local p; for p in $(pgrep -f python3); do { tr '\0' '\n' < "/proc/$p/environ"; } 2>/dev/null | grep '^RUN_NAME=sole_ft_'; done | sort -u | wc -l; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ok=0
echo "$(date '+%F %T') waiting: run=$RUN more=$MORE need=${NEED_MIB}MiB"
while true; do
  if [ "$(running)" -lt 2 ] && [ "$(free_mib)" -ge "$NEED_MIB" ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -ge 2 ] && break
  sleep 5
done
echo "$(date '+%F %T') start: running=$(running) free=$(free_mib)MiB"
export RUN_NAME=sole_ft_m19_finetune
exec bash "$HERE/launch_finetune.sh" "$RUN" 2048 "$MORE" --agent.save-interval 250 --env.sim.nconmax 210
