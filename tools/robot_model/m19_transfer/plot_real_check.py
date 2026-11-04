#!/usr/bin/env python3
"""Figure: what the existing real logs say about the robot's weight and standing torques, next to the model variants.
Inputs: static_weigh.py json, analyze_transfer.py json (nominal runs). Labels in English only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sw = json.load(open(sys.argv[1]))
an = json.load(open(sys.argv[2]))
out = Path(sys.argv[3])
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1]))

# (a) supported mass of every still window
ax = axs[0]
x0 = 0
ticks, labels = [], []
col = {"walk1104": "#d62728", "squat1104": "#ff9896", "teleop1028": "#1f77b4", "extra2_1028": "#aec7e8"}
for lab, v in sw["logs"].items():
  for w in v["windows"]:
    ax.plot([x0, x0], [w["A"]["supported_mass_kg"], w["B"]["supported_mass_kg"]], color=col[lab], lw=1.5)
    ax.plot(x0, w["A"]["supported_mass_kg"], "o", color=col[lab], ms=5)
    ax.plot(x0, w["B"]["supported_mass_kg"], "s", color=col[lab], ms=5, mfc="white")
    x0 += 1
  ticks.append(x0 - len(v["windows"]) / 2 - 0.5)
  labels.append(lab.replace("walk1104", "walk 11-04").replace("squat1104", "squat 11-04").replace("teleop1028", "walk 10-28").replace("extra2_1028", "walk 10-28 #2"))
  x0 += 1
for m, nm, c in [(23.63, "M10 training 23.63", "k"), (24.49, "M16 24.49", "#6baed6"), (26.91, "M19 latest 26.91", "#d62728")]:
  ax.axhline(m, color=c, ls="--", lw=1)
  ax.text(x0 - 0.5, m + 0.08, nm, ha="right", fontsize=8, color=c)
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("robot mass carried by the two feet [kg]\n(from motor torque, nominal Kp)")
ax.set_title("(a) weighing the robot with its own leg motors\ncircle: leg masses of M10, square: leg masses of M19", fontsize=10)

# (b) standing torque: sim variants vs real pre-walk windows
ax = axs[1]
J = ["R_hip_roll", "L_hip_roll", "R_crank_A", "R_crank_B", "L_crank_A", "L_crank_B", "R_hip_pitch", "L_hip_pitch", "R_knee", "L_knee"]
cols = {"R_hip_roll": "right_leg/hip_roll", "L_hip_roll": "left_leg/hip_roll", "R_crank_A": "right_leg/ankle_a", "R_crank_B": "right_leg/ankle_b", "L_crank_A": "left_leg/ankle_a",
        "L_crank_B": "left_leg/ankle_b", "R_hip_pitch": "right_leg/hip_pitch", "L_hip_pitch": "left_leg/hip_pitch", "R_knee": "right_leg/knee", "L_knee": "left_leg/knee"}
pre = [w for w in sw["logs"]["walk1104"]["windows"] if w["t1"] < 30]
post = [w for w in sw["logs"]["walk1104"]["windows"] if w["t0"] > 75]
real_pre = [np.mean([w["tau_Nm"][cols[j]] for w in pre]) for j in J]
real_post = [np.mean([w["tau_Nm"][cols[j]] for w in post]) for j in J]
width = 0.15
series = [("real, before walking (0-26 s)", real_pre, "#d62728"), ("real, after walking (78-103 s)", real_post, "#ff9896"),
          ("sim M10", [an["nom_base"]["segments"]["stand"]["tau_mean_" + j] for j in J], "#7f7f7f"),
          ("sim M16", [an["nom_m16"]["segments"]["stand"]["tau_mean_" + j] for j in J], "#9ecae1"),
          ("sim M19", [an["nom_m19"]["segments"]["stand"]["tau_mean_" + j] for j in J], "#3182bd")]
for i, (nm, v, c) in enumerate(series):
  ax.bar(np.arange(len(J)) + (i - 2) * width, v, width, label=nm, color=c)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(J)))
ax.set_xticklabels(J, rotation=35, fontsize=8)
ax.set_ylabel("standing joint torque [N m] (command 0)")
ax.legend(fontsize=7.5, ncol=2)
ax.set_title("(b) standing torque: real motors (Kp x tracking error) vs simulation", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
