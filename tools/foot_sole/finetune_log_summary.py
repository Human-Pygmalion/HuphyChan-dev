#!/usr/bin/env python3
"""Parse rsl_rl console logs of the foot-sole fine-tunes (tools/foot_sole/finetune_remote.sh) into per-iteration metrics,
print window means (early / late) per run and plot the training curves.
  python3 tools/foot_sole/finetune_log_summary.py <out_dir> <name>=<train.log> [...]
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(sys.argv[1])
RUNS = dict(a.split("=", 1) for a in sys.argv[2:])
ANSI = re.compile(r"\x1b\[[0-9;]*m")
KEYS = ["Mean reward", "Mean episode length", "Episode_Termination/fell_over", "Episode_Termination/low_base",
        "Episode_Termination/time_out", "Metrics/twist/error_vel_xy", "Metrics/twist/error_vel_yaw",
        "Metrics/landing_force_mean", "Metrics/slip_velocity_mean", "Metrics/air_time_mean", "Episode_Reward/foot_slip",
        "Episode_Reward/soft_landing", "Episode_Reward/pose", "Episode_Reward/track_linear_velocity", "Curriculum/terrain_5k_unlock"]


def parse(path):
  rows, cur = [], None
  for line in Path(path).read_text(errors="replace").splitlines():
    line = ANSI.sub("", line).strip()
    m = re.match(r"Learning iteration (\d+)/\d+", line)
    if m:
      cur = {"it": int(m.group(1))}
      rows.append(cur)
      continue
    if cur is None or ":" not in line:
      continue
    k, _, v = line.rpartition(":")
    try:
      cur[k.strip()] = float(v.strip().rstrip("s"))
    except ValueError:
      pass
  return rows


data = {n: parse(p) for n, p in RUNS.items()}
last_common = min(r[-1]["it"] for r in data.values())
EARLY = (50200, 50400)  # skip the first 200: every episode restarts at the resume, so early means are reset transients
LATE = (last_common - 200, last_common)
summary = {"early_window": EARLY, "late_window": LATE, "runs": {}}
for n, rows in data.items():
  s = {}
  for k in KEYS:
    def win(a, b):
      v = [r[k] for r in rows if a <= r["it"] < b and k in r]
      return float(np.mean(v)) if v else None
    s[k] = {"early": win(*EARLY), "late": win(*LATE)}
  s["last_it"] = rows[-1]["it"]
  summary["runs"][n] = s
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "finetune_log_summary.json").write_text(json.dumps(summary, indent=1))
print(f"windows early {EARLY} late {LATE}")
for k in KEYS:
  print(f"{k:42s} " + "  ".join(f"{n}: {s[k]['early']:.4g} -> {s[k]['late']:.4g}" if s[k]["early"] is not None else f"{n}: -"
                                  for n, s in summary["runs"].items()))
plot = ["Mean reward", "Mean episode length", "Episode_Termination/fell_over", "Metrics/twist/error_vel_xy",
        "Metrics/landing_force_mean", "Metrics/slip_velocity_mean"]
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, k in zip(ax.flat, plot):
  for n, rows in data.items():
    it = np.array([r["it"] for r in rows if k in r])
    v = np.array([r[k] for r in rows if k in r])
    w = 25
    if len(v) > w:
      a.plot(it[w - 1:], np.convolve(v, np.ones(w) / w, "valid"), label=n, lw=1.2)
  a.set_title(k, fontsize=9)
  a.set_xlabel("iteration")
  a.grid(alpha=0.3)
ax[0, 0].legend()
fig.suptitle("Foot-sole fine-tunes from Union-DR 49999 (25-iteration moving mean)")
fig.tight_layout()
fig.savefig(OUT / "fig_finetune_curves.png", dpi=110)
print("wrote", OUT / "fig_finetune_curves.png")
