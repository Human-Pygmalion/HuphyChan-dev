#!/usr/bin/env python3
"""Run foot_variant_eval.py jobs on the 3090 as checkpoints appear (run ON the 3090, detached).

Job file: one job per line, whitespace separated
  <tag> <variant> <terrain> <run> <iteration> [VAR=value ...]
  run = the training run name (sole_ft_<...>_<more>, matched as logs/.../*_<run>/model_<iteration>.pt) or ORIG (the deployed
  49999 checkpoint). VAR=value pairs are exported to the eval (e.g. PYG_FOOT_SOLREF=0.01 for runs trained with that setting).
A job runs when its checkpoint exists and its training process has saved past it; it is skipped when <out>/<tag>.json exists.
Up to --parallel jobs at once (GPU, 32 envs each). The job file is re-read every pass, so jobs can be appended.
  setsid nohup python3 tools/foot_sole/eval_queue.py <jobs.txt> <out_dir> > eval_queue.log 2>&1 &
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("jobs")
ap.add_argument("out")
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--num-envs", type=int, default=32)
args = ap.parse_args()
REPO = Path.home() / "MikuchanRemote/Human-Pygmalion"
WS = REPO / "Feet_design/Feet_Design_Workspace/mjcf_sim"
BR = WS / "union_branch_49999/mjlab_policy"
LOGS = BR / "logs/rsl_rl/pygmalion_velocity"
PY = REPO / "mujoco-sim/mjlab/.venv/bin/python3"
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
running, fails = {}, {}


def ckpt(run, it):
  if run == "ORIG":
    return BR / "checkpoints/terrain10_union_dr_49999/model_49999.pt"
  hits = sorted(LOGS.glob(f"*_{run}/model_{it}.pt"))
  if not hits or time.time() - hits[-1].stat().st_mtime < 60:  # let the save finish
    return None
  return hits[-1]


while True:
  jobs = [ln.split() for ln in Path(args.jobs).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
  for tag, p in list(running.items()):
    if p.poll() is not None:
      print(time.strftime("%F %T"), "done", tag, "rc", p.returncode, flush=True)
      if p.returncode != 0:
        fails[tag] = fails.get(tag, 0) + 1
      del running[tag]
  pending = [j for j in jobs if not (OUT / f"{j[0]}.json").exists() and j[0] not in running and fails.get(j[0], 0) < 2]
  if not pending and not running:
    break
  for j in pending:
    if len(running) >= args.parallel:
      break
    tag, variant, terrain, run, it = j[:5]
    c = ckpt(run, it)
    if c is None:
      continue
    env = dict(os.environ, PYTHONPATH=str(WS / "stock_mjlab/src"))
    env.pop("PYG_FOOT_SOLREF", None)
    env.update(kv.split("=", 1) for kv in j[5:])
    cmd = [str(PY), str(REPO / "tools/foot_sole/foot_variant_eval.py"), "--branch-root", str(BR), "--mjlab-src", str(WS / "stock_mjlab/src"),
           "--variant", variant, "--terrain", terrain, "--num-envs", str(args.num_envs), "--device", "cuda:0", "--checkpoint", str(c),
           "--tag", tag, "--out", str(OUT)]
    log = open(OUT / f"{tag}.log", "w")
    running[tag] = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
    print(time.strftime("%F %T"), "start", tag, c, " ".join(j[5:]), flush=True)
  time.sleep(30)
print(time.strftime("%F %T"), "all jobs done", flush=True)
