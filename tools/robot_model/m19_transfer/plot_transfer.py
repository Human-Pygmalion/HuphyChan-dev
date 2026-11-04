#!/usr/bin/env python3
"""Figure: policy 49999 on the training model (M10) and on newer robot models, nominal simulation, flat ground, 32 envs per bar.
Input: analyze_transfer.py json. Labels in English only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
order = [("nom_base", "M10\n23.6", "#7f7f7f"), ("nom_m16", "M16\n24.5", "#9ecae1"), ("nom_m17", "M17\n25.1", "#6baed6"),
         ("nom_m18", "M18\n25.5", "#3182bd"), ("nom_m19", "M19\n26.9", "#d62728"), ("nom_base_pl328x2", "M10 +3.28\ntorso", "#fdae6b")]
order = [o for o in order if o[0] in d]
panels = [("stand", "stance_width_cm", "stand: stance width [cm]"), ("vx0.6", "stance_width_cm", "walk 0.6 m/s: stance width [cm]"),
          ("vx0.6", "slip_p95_mm_s", "walk 0.6: foot slip P95 [mm/s]"), ("vx0.6wz0.5", "wz_err_rms", "walk+turn: yaw-rate error RMS [rad/s]"),
          ("stand", "tau_p99abs_hip_roll", "stand: hip_roll torque P99 [N m]"), ("vx0.6", "tau_p99abs_hip_roll", "walk 0.6: hip_roll torque P99 [N m]"),
          ("vx0.6", "tau_p99abs_knee", "walk 0.6: knee torque P99 [N m]"), ("vx1.5", "tau_p99abs_crank", "walk 1.5: crank torque P99 [N m]")]
fig, axs = plt.subplots(2, 4, figsize=(15, 6.4))
for ax, (seg, key, title) in zip(axs.ravel(), panels):
  vals = [d[t]["segments"][seg].get(key) for t, _, _ in order]
  ax.bar(range(len(order)), [v if v is not None else 0 for v in vals], color=[c for _, _, c in order])
  for i, v in enumerate(vals):
    if v is not None:
      ax.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=7)
  ax.set_xticks(range(len(order)))
  ax.set_xticklabels([l for _, l, _ in order], fontsize=6.5)
  ax.set_title(title, fontsize=9)
falls = ", ".join(f"{lab.split(chr(10))[0]}: {d[t]['falls']}/{d[t]['n_envs']}" for t, lab, _ in order)
dr = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
drtxt = "  |  DR on, first fall within 135 s per 64 envs: " + ", ".join(f"{k}: {dr[k]['falls']}" for k in ("dr_base", "dr_m19", "dr_base_pl328") if k in dr) if dr else ""
fig.suptitle("Policy 49999 (trained on M10) on newer models, flat ground. Nominal sim falls /32: " + falls + drtxt, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=140)
print("wrote", out)
