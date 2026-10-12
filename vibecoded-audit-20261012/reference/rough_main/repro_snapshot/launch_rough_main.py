"""Main rough-terrain training (2026-10-11, docs/162 step 6, design approved by the user 2026-10-11).

Start: v11a phase-2 final actor (actor-only warm start; the rough critic sees height_scan).
Changes vs the turning tests (docs/reward_research/2026-10-11_turn_in_place_recovery_literature.md §7-4):
  PYG_BOX_TERRAIN=1        box-only rough set - MuJoCo-Warp heightfield contact makes the walking foot slip
  entropy 0.002 held       0.01 blew the action std up to 0.6 and killed turning in place
  PYG_CRITIC_WARMUP_ITERS  actor held for the first 300 updates while the fresh critic learns (standing)
  PYG_DR_START_ITER=2000 / PYG_DR_END_ITER=8000
                           randomization (push, friction, mass, encoder, CoM) ramps 0 -> full; the earlier
                           rough runs inherited the parent's 15999 window, counted from 0, and trained with
                           randomization OFF (dr_factor 0 for all 3,000 iterations)
  stagnation termination + low-base-above-terrain + NaN / blow-up terminations + obs NaN guard + reward clamp 50
Rewards, gains, command box, joint ranges and the robot model are v11a's.
12,000 iterations, 4,096 envs, seed 43. Checks at 4,000 / 8,000 with analysis/matched_set.sh and terrain_eval.

Launcher body copied from launch_rough_turn_ent.py; its docstring follows.

Turning-recovery test (2026-10-11, docs/162 step 4): rough arm B + entropy coefficient 0.002.

Finding: in both rough A/B arms the policy's action noise std grew 0.13 -> 0.60 (the flat
parent never exceeded 0.28 and ended at 0.13 under an entropy anneal 0.01 -> 0.002), and the
yaw-tracking share of reward collapsed between iterations 200 and 800 as std passed 0.23 -> 0.57.
The warm start runs iterations 0..N, so the parent's anneal window (15999..29998) never starts
and the coefficient sat at 0.01 for the whole run. This arm is arm B with ONE change: the
entropy coefficient held at the parent's final 0.002. Compare with arm B at the same iteration.

Copied from the stagnation A/B launcher; the original docstring follows.

A/B (2026-10-10): stagnation termination on rough terrain, warm-started from v11a.

Both arms: task Mjlab-Velocity-Rough-Pygmalion, actor-only warm start from the v11a P2 final
checkpoint (analysis/train_actor_warmstart.py - the rough critic sees height_scan, the flat one
does not), every PYG_* flag of the v11a P2 run copied from its own repro/launch_manifest.json,
plus PYG_FRESH_STEPS=1 (curricula restart from step 0), PYG_LOW_BASE_REL=1 (the absolute
0.7 m low-base test cuts most rough-terrain episodes - a bug fix) and PYG_NAN_TERM=1 (reset an env
whose physics diverged instead of aborting the run - the first launch died at iteration 232),
all common to both arms.
Arm B adds PYG_STAGNATION=1. That is the only difference.
Review: docs/reward_research/2026-10-09_reward_termination_review.md
Run on the 3090:  .venv/bin/python3 analysis/launch_rough_stag_ab.py [--iters 3000] [--envs 4096]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

MJLAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(MJLAB, "logs", "rsl_rl", "pygmalion_velocity")
PARENT = "2026-10-02_23-52-14_legonly_ab_v11a_stand01_s43_p2"
CKPT = "model_29998.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=12000)
ap.add_argument("--arm", default="rough_box_main_v11a")
ap.add_argument("--envs", type=int, default=4096)
ap.add_argument("--gap", type=int, default=120, help="seconds between the two launches")
a = ap.parse_args()

parent_env = json.load(open(os.path.join(LOGS, PARENT, "repro", "launch_manifest.json")))["environment"]
base = {k: v for k, v in parent_env.items() if k.startswith("PYG_")}
base.update(PYG_FRESH_STEPS="1", PYG_LOW_BASE_REL="1", PYG_NAN_TERM="1",  # NaN: 1st try died at 232
            PYG_OBS_NAN_GUARD="1",  # critic NaN with finite physics: 2nd try died at 658
            PYG_BLOWUP_TERM="1", PYG_REWARD_TERM_CLAMP="50")  # 3rd try: 1e24+ rewards from blow-ups
_ENT = dict(PYG_ENTROPY_ANNEAL="1", PYG_ENTROPY_ANNEAL_START="0", PYG_ENTROPY_ANNEAL_END="0",
            PYG_ENTROPY_START_COEF="0.002", PYG_ENTROPY_END_COEF="0.002")
ALL_ARMS = {
  "rough_box_main_v11a": dict(base, PYG_STAGNATION="1", PYG_CRITIC_WARMUP_ITERS="300",
                              PYG_BOX_TERRAIN="1", PYG_DR_START_ITER="2000", PYG_DR_END_ITER="8000",
                              **_ENT),
}
arms = {a.arm: ALL_ARMS[a.arm]}


def sha(p: str) -> str:
  return hashlib.sha256(open(p, "rb").read()).hexdigest()


def snapshot(name: str, env: dict, cmd: list[str], t0: float) -> None:
  for _ in range(120):  # only a folder created by THIS launch (an older try has the same name)
    dirs = sorted(d for d in glob.glob(os.path.join(LOGS, f"*_{name}")) if os.path.getctime(d) >= t0 - 5)
    if dirs:
      break
    time.sleep(2)
  else:
    print(f"!! run dir for {name} did not appear - no repro snapshot", flush=True)
    return
  repro = os.path.join(dirs[-1], "repro")
  os.makedirs(repro, exist_ok=True)
  srcs = [os.path.abspath(__file__), os.path.join(MJLAB, "analysis", "train_actor_warmstart.py")] + [
    os.path.join(MJLAB, "src", "mjlab", "tasks", "velocity", p)
    for p in ("config/pygmalion/env_cfgs.py", "config/pygmalion/rl_cfg.py", "mdp/curriculums.py", "mdp/terminations.py",
              "mdp/velocity_command.py", "velocity_env_cfg.py")
  ] + [os.path.join(MJLAB, "src", "mjlab", "terrains", "config.py"),
       os.path.join(MJLAB, "src", "mjlab", "rl", "runner.py")
  ]
  inputs = []
  for s in srcs:
    if os.path.isfile(s):
      shutil.copy2(s, repro)
      inputs.append({"source": s, "snapshot": os.path.basename(s), "sha256": sha(s)})
  manifest = {
    "run_name": name, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    "cwd": MJLAB, "command": cmd, "environment": {k: str(v) for k, v in sorted(env.items()) if k.startswith("PYG_")},
    "parent": {"run": PARENT, "checkpoint": CKPT, "load": "actor only"}, "inputs": inputs,
  }
  json.dump(manifest, open(os.path.join(repro, "launch_manifest.json"), "w"), indent=2, ensure_ascii=False)
  print(f"[record] {name}: repro snapshot -> {repro}", flush=True)


for i, (name, flags) in enumerate(arms.items()):
  if i:
    time.sleep(a.gap)  # let the first arm allocate its GPU memory first
  cmd = [".venv/bin/python3", "analysis/train_actor_warmstart.py", "Mjlab-Velocity-Rough-Pygmalion",
         "--env.scene.num-envs", str(a.envs), "--agent.max-iterations", str(a.iters),
         "--agent.run-name", name, "--agent.seed", "43", "--agent.logger", "tensorboard",
         "--agent.resume", "True", "--agent.load-run", PARENT, "--agent.load-checkpoint", CKPT]
  env = {**{k: v for k, v in os.environ.items() if not k.startswith("PYG_")}, **flags,
         "MUJOCO_GL": "egl"}
  out = open(os.path.join(MJLAB, "analysis", "out", f"{name}_launch.out"), "w")
  t0 = time.time()
  p = subprocess.Popen(cmd, cwd=MJLAB, env=env, stdout=out, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, start_new_session=True)
  print(f"launched {name} pid {p.pid}", flush=True)
  snapshot(name, env, cmd, t0)
