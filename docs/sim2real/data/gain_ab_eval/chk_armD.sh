#!/usr/bin/env bash
cd ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
D=logs/rsl_rl/pygmalion_velocity/2026-10-17_19-39-49_rough_gain_crank60_actdelay_v11a
OUT=~/v11a_eval/gain_ab/built_delay_check_armD.txt
: > $OUT
for MODE in with_flag dropped; do
  DROP=""; [ $MODE = dropped ] && DROP="PYG_ACTION_DELAY"
  E=$(GAIN_AB_DROP="$DROP" python3 -c "import json,os;d=json.load(open(\"$D/repro/launch_manifest.json\"))[\"environment\"];drop=os.environ.get(\"GAIN_AB_DROP\",\"\").split();print(\" \".join(f\"{k}={v}\" for k,v in sorted(d.items()) if k.startswith(\"PYG_\") and k not in drop))")
  echo "##### evaluation environment: $MODE" >> $OUT
  echo "PYG env passed: $E" | tr " " "\n" | grep -E "ACTION|NOISE|GAIN_DR|^PYG env" >> $OUT
  env $E MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python3 ../../tools/sim2real/action_delay_eval_check.py 2>&1 | grep -E "^(PYG_ACTION|curriculum|action|term|max |RESULT)" >> $OUT
done
cat $OUT
