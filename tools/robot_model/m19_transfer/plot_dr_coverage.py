#!/usr/bin/env python3
"""Figure: where the latest model sits relative to the DR range of the training run (inputs: dr_coverage.py outputs)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

js, out = Path(sys.argv[1]), Path(sys.argv[2])
d = json.load(open(js))
S = np.load(js.with_suffix(".samples.npz"))
F, Fn, F0, names = S["F"], S["Fn"], S["F0"], list(S["names"])
fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.45, wspace=0.28)

# (a) per-link mass ratio vs DR range (right leg + base only; left is the mirror)
ax = fig.add_subplot(gs[0, :])
rows = [r for r in d["per_link"] if r["body"].startswith(("R_", "base")) and r.get("dr_ratio_range")]
order = ["base_link", "R_hip_pitch_link", "R_hip_roll_link", "R_thigh_link", "R_shin_link", "R_ankle_pitch_link", "R_foot_link", "R_crank_A", "R_crank_B", "R_rod_A", "R_rod_B"]
rows = sorted(rows, key=lambda r: order.index(r["body"]))
for i, r in enumerate(rows):
  lo, hi = r["dr_ratio_range"]
  ax.add_patch(plt.Rectangle((i - 0.35, lo), 0.7, hi - lo, color="#9ecae1", alpha=0.8, lw=0))
  ok = r["mass_inside"]
  ax.plot([i], [r["ratio"]], "o", ms=9, color="#2ca02c" if ok else "#d62728", zorder=3)
  ax.text(i, r["ratio"] * (1.07 if r["ratio"] > 1.2 else 1.0) + (0.0 if r["ratio"] > 1.2 else 0.02), f"{r['mass_new_kg']:.2f} kg", ha="center", va="bottom", fontsize=8)
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r["body"].replace("_link", "").replace("R_", "") for r in rows], rotation=20)
ax.set_ylabel("link mass: latest model / training model")
ax.set_yscale("log")
ax.set_ylim(0.8, 3.3)
ax.set_yticks([0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 2.5])
ax.set_yticklabels(["0.9", "1.0", "1.1", "1.3", "1.6", "2.0", "2.5"])
ax.set_title("(a) link mass: blue band = DR range; dot = latest model (green inside, red outside)", fontsize=11, pad=10)

# (b-d) whole-robot features
for k, (nm, lab) in enumerate([("total_mass_kg", "total mass [kg]"), ("legs_mass_kg", "both legs mass [kg]"), ("swing_leg_inertia_about_hip_kgm2", "one leg inertia about hip pitch axis [kg m^2]")]):
  ax = fig.add_subplot(gs[1, k])
  j = names.index(nm)
  ax.hist(F[:, j], bins=80, color="#9ecae1", density=True)
  ax.axvline(F0[j], color="k", ls="--", lw=1.2, label="training model")
  ax.axvline(Fn[j], color="#d62728", lw=2, label="latest model")
  ax.set_xlabel(lab)
  ax.set_yticks([])
  pct = d["whole_robot"][j]["new_percentile"]
  ax.set_title(f"({'bcd'[k]}) {pct:.0f}th percentile" + (" - outside cloud" if d["whole_robot"][j]["outside_sampled_cloud"] else ""), fontsize=10)
  if k == 0:
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle(f"Is the latest model inside the DR cloud of policy 49999?  Mahalanobis distance {d['mahalanobis']['new']:.1f} (99th percentile of DR draws: {d['mahalanobis']['dr_p99']:.1f})", fontsize=11)
fig.subplots_adjust(top=0.9)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
