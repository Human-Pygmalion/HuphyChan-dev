#!/bin/bash
# Run batches of foot-sole fine-tunes on the 3090 from a plan file, one batch after the other (run ON the 3090, detached):
#   setsid nohup bash tools/foot_sole/finetune_queue.sh <plan.txt> > queue.log 2>&1 &
# Plan file: one batch per line, "<more_iterations>|<arm> <arm> ...", each arm <variant>[:VAR=value[,VAR=value]] passed to
# finetune_remote.sh; '#' lines are skipped. The first line is removed when its batch starts, so lines can be appended or
# edited while the queue waits. A batch starts only when no sole_ft_* training process is left.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PLAN="${1:?plan file}"
running() {
  local n=0 p
  for p in $(pgrep -f python3); do
    { tr '\0' '\n' < "/proc/$p/environ"; } 2>/dev/null | grep -q '^RUN_NAME=sole_ft_' && n=$((n + 1))
  done
  echo "$n"
}
while true; do
  while [ "$(running)" -gt 0 ]; do sleep 120; done
  line="$(grep -v '^\s*#' "$PLAN" | grep -v '^\s*$' | head -1)"
  [ -z "$line" ] && break
  grep -v -x -F "$line" "$PLAN" > "$PLAN.tmp"; mv "$PLAN.tmp" "$PLAN"
  more="${line%%|*}"
  echo "$(date '+%F %T') batch start (more=$more): ${line#*|}"
  for arm in ${line#*|}; do
    v="${arm%%:*}"
    envs=""
    [ "$arm" != "$v" ] && envs="${arm#*:}"
    env ${envs//,/ } bash "$HERE/finetune_remote.sh" "$v" "$more"
    sleep 60   # stagger start-up (model compile + first collection)
  done
  sleep 600    # let every arm of the batch register before polling again
done
echo "$(date '+%F %T') queue done"
