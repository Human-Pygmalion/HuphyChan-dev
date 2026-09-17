#!/usr/bin/env bash
# Bring a training run that lives on another machine back to this one.
#
# 10.8.0.14 has its own disk - the check that proved it: write a marker file here, look
# for it there, it is not there. So a run launched on it writes its logs, checkpoints,
# launcher state and hourly review snapshots to ITS disk, and nothing appears here until
# it is pulled. This does that pull.
#
#   tools/remote/pull_remote_run.sh legonly_ab_v8_standing_mid
#   tools/remote/pull_remote_run.sh legonly_ab_v8_standing_mid --with-checkpoints
#
# Without --with-checkpoints it moves only what is needed to JUDGE the run - the launcher
# log and state, the experiment note the remote review loop is appending to, and the
# tensorboard event files. Checkpoints are hundreds of megabytes and are only needed once,
# at the end, to measure the finished policy.
#
# HOST/REMOTE can be overridden for a different machine.
set -euo pipefail

RUN="${1:?usage: pull_remote_run.sh <run-name> [--with-checkpoints]}"
WITH_CKPT="${2:-}"
HOST="${HOST:-syaro@10.8.0.14}"
REMOTE="${REMOTE:-/home/syaro/MikuchanRemote/Human-Pygmalion}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[pull] $HOST:$REMOTE  ->  $HERE   run=$RUN"

# 1. launcher log + state: the run's own account of itself
rsync -a --info=name0 \
  "$HOST:$REMOTE/mujoco-sim/mjlab/analysis/out/v2_scratch_${RUN}."{log,json} \
  "$HERE/mujoco-sim/mjlab/analysis/out/" 2>/dev/null || echo "  (launcher log/state not there yet)"

# 2. the experiment note - the REMOTE review loop appends the hourly §2c snapshots to the
#    remote copy, so without this the local note silently lacks them.
NOTE=$(ssh -o BatchMode=yes "$HOST" \
  "ls $REMOTE/docs/experiments/*${RUN}*.md 2>/dev/null | head -1" || true)
if [ -n "$NOTE" ]; then
  rsync -a --info=name0 "$HOST:$NOTE" "$HERE/docs/experiments/"
  echo "  note: $(basename "$NOTE")"
else
  echo "  (no experiment note matched *${RUN}*.md on the remote)"
fi

# 3. run directories: metrics always, weights on request
for D in $(ssh -o BatchMode=yes "$HOST" \
    "ls -1d $REMOTE/mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/*${RUN}* 2>/dev/null" || true); do
  NAME=$(basename "$D")
  if [ "$WITH_CKPT" = "--with-checkpoints" ]; then
    rsync -a --info=stats0 "$HOST:$D/" \
      "$HERE/mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/$NAME/"
    echo "  run dir (with checkpoints): $NAME"
  else
    rsync -a --info=stats0 --exclude 'model_*.pt' --exclude 'videos/' "$HOST:$D/" \
      "$HERE/mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/$NAME/"
    echo "  run dir (metrics only): $NAME"
  fi
done

echo "[pull] done"
