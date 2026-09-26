#!/usr/bin/env python3
"""Rebuild §2c training-review rows from an RSL-RL console log (for runs whose review loop
had no wandb record, e.g. the remote 3090 with --logger tensorboard).
Usage: review_rows_from_log.py <log> [<log> ...] --every 2000"""
import re, sys
args = sys.argv[1:]; every = 2000
if "--every" in args:
  k = args.index("--every"); every = int(args[k + 1]); del args[k:k + 2]
KEYS = {"reward": "Mean reward", "len": "Mean episode length", "std": "Mean action std",
        "vloss": "Mean value loss", "ent": "Mean entropy loss", "surr": "Mean surrogate loss",
        "fell": "Episode_Termination/fell_over", "low": "Episode_Termination/low_base",
        "exy": "Metrics/twist/error_vel_xy_steady", "eyaw": "Metrics/twist/error_vel_yaw",
        "dr": "Curriculum/dr_levels/dr_factor", "vx": "Curriculum/command_vel/lin_vel_x_max",
        "stage": "Curriculum/command_vel/stage", "th": "Metrics/thermal_effort_mean"}
ansi = re.compile(r"\x1b\[[0-9;]*m")
for log in args:
  it = None; cur = {}
  def flush():
    if it is not None and (it % every == 0 or it + 1 == total):
      g = lambda k, f="{:.3f}": f.format(cur[k]) if k in cur else "—"
      print(f"| {it} | {g('reward','{:.1f}')} | {g('len','{:.0f}')} | {g('std','{:.2f}')} | {g('vloss','{:.4f}')} | "
            f"{g('ent','{:.2f}')} | {g('surr','{:.4f}')} | {g('fell','{:.4f}')} / {g('low','{:.2f}')} | "
            f"{g('exy')} / {g('eyaw')} | {g('dr','{:.2f}')} / {g('vx','{:.1f}')} (단계 {g('stage','{:.0f}')}) | {g('th','{:.2f}')} |")
  total = 0
  for line in open(log, errors="replace"):
    line = ansi.sub("", line)
    m = re.search(r"Learning iteration (\d+)/(\d+)", line)
    if m:
      flush(); it = int(m.group(1)); total = int(m.group(2)); cur = {}; continue
    if ":" in line:
      k, _, v = line.strip().rpartition(":")
      for key, name in KEYS.items():
        if k.strip() == name:
          try: cur[key] = float(v.split()[0])
          except ValueError: pass
  flush()
