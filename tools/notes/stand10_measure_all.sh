#!/usr/bin/env bash
# Standing-pose audit (2026-10-14): for each (run, checkpoint) roll out 6 seeds of PURE STANDING
# (12 s, command 0, no warm-up walk) on CPU and save analysis/out/stand10_<run>_m<iter>_s<seed>.npz
# (+ _model.mjb). PYG_* flags are read from each run's own repro/launch_manifest.json, exactly as
# analysis/run_v10_set.sh does, so no flag can silently differ from training.
#   cd mujoco-sim/mjlab && bash ../../tools/notes/stand10_measure_all.sh [parallel=6]
# Output is idempotent: a finished tag (npz present) is skipped.
set -u
cd "$(dirname "$0")/../../mujoco-sim/mjlab" || exit 1
P=${1:-6}
L=logs/rsl_rl/pygmalion_velocity
# label  run_dir  checkpoints
JOBS=$(cat <<'J'
v7 2026-09-16_14-25-25_legonly_ab_v7_standonly_p1 100 300 1000 2000 4100
v7 2026-09-16_18-21-14_legonly_ab_v7_standonly_p2 6000 9000
v7b 2026-09-16_22-47-17_legonly_ab_v7b_robust_stand_p1 100
v7b 2026-09-16_22-57-26_legonly_ab_v7b_robust_stand_p1r 300 1000 1999
v7b 2026-09-16_23-49-54_legonly_ab_v7b_robust_stand_p2 3000 5000 7000 8998
v9caps 2026-09-20_18-09-37_legonly_ab_v9_caps_p1 100 300 1000 1999
v9caps 2026-09-20_19-17-25_legonly_ab_v9_caps_p2 3000 5000 7000 8998
J
)
LIST=$(mktemp)
while read -r lab dir cks; do
  for n in $cks; do for s in 0 1 2 3 4 5; do
    echo "$lab $dir $n $s"
  done; done
done <<<"$JOBS" >"$LIST"
one() {
  lab=$1; dir=$2; n=$3; s=$4; D=logs/rsl_rl/pygmalion_velocity/$dir; tag=stand10_${lab}_m${n}_s${s}
  [ -f analysis/out/$tag.npz ] && return 0
  E=$(python3 -c "import json;d=json.load(open('$D/repro/launch_manifest.json'))['environment'];print(' '.join(f'{k}={v}' for k,v in sorted(d.items()) if k.startswith('PYG_')))")
  [ -n "$E" ] || { echo "NOFLAGS $tag"; return 2; }
  env $E CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python3 analysis/measure_stand.py \
    --stand-s 12 --rollout-seed $s --warmup 0 --device cpu --run-dir $D --checkpoint $D/model_$n.pt --tag $tag \
    >analysis/out/$tag.log 2>&1 && echo "OK $tag" || echo "FAIL $tag"
}
export -f one
xargs -a "$LIST" -P "$P" -L 1 bash -c 'one "$@"' _
echo ALLDONE
