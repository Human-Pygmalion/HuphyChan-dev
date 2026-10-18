#!/usr/bin/env python3
"""Figure for docs/176: effect of each robustness element in the rough-terrain A/B series.

Numbers are copied from the finished-run notes (no re-measurement):
  baseline/A/C/D  docs/experiments/2026-10-17_rough_gain_crank60_actdelay_v11a.md §2-§12
  trial           docs/experiments/2026-10-14_rough_box_trial_v11a.md (evaluated with its DR ON -> not like-for-like)
Usage: python3 tools/sim2real/plot_dr_effects.py [out.png]
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs = ["base", "A", "C", "D", "trial*"]
legend = ["base = deployed rough_box_main_v11a", "A = crank 60/2.7", "C = A + PD gain DR", "D = A + command delay DR", "trial* = bundle"]
col = ["#888888", "#4c72b0", "#55a868", "#c44e52", "#8172b2"]
nan = np.nan
panels = [
    ("Survival at gain x0.5 + real timing\n(sim, runs reaching 20 s, of 8)", [0, 1, 6, 8, 8], "of 8", (0, 8.8), True),
    ("Turn-in-place achieved, median of 8 seeds\n(higher = better)", [45.3, 53.5, 48.0, 37.7, 23.4], "%", (0, 65), True),
    ("Uphill 20 deg at 0.6 m/s: falls\n(lower = better)", [6.2, 9.8, 6.6, 18.0, nan], "%", (0, 22), False),
    ("Straight walk 1.2 m/s: heading drift\n(closer to 0 = better)", [-0.66, 4.48, 1.68, 0.47, nan], "deg/s", (-1.5, 5.5), False),
]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
x = np.arange(len(runs))
for ax, (title, vals, unit, ylim, _) in zip(axs, panels):
    v = np.array(vals, float)
    bars = ax.bar(x, np.nan_to_num(v), color=col, edgecolor="black", linewidth=0.6)
    bars[-1].set_hatch("//")
    for xi, vi in zip(x, v):
        lab = "not\nmeasured" if np.isnan(vi) else (f"{vi:.0f}" if unit == "of 8" else f"{vi:+.2f}" if unit == "deg/s" else f"{vi:.1f}")
        y = 0 if np.isnan(vi) else vi
        ax.text(xi, y + (ylim[1] - ylim[0]) * 0.02 if y >= 0 else y - (ylim[1] - ylim[0]) * 0.08, lab, ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, runs, fontsize=10)
    ax.set_ylim(*ylim); ax.set_ylabel(unit); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
axs[1].axhline(43.5, color="black", ls="--", lw=1); axs[1].text(4.45, 44.5, "pass line 43.5", ha="right", fontsize=8)
fig.suptitle("Rough-terrain A/B, seed 43, 12,000 iterations, 1 training seed per run.  "
             "*trial = crank 55 + gain DR + cmd delay + soft push + encoder bias 3 deg, evaluated with its randomization ON (not like-for-like)",
             fontsize=9.5, y=0.995)
from matplotlib.patches import Patch
fig.legend([Patch(facecolor=c, edgecolor="black", hatch="//" if i == 4 else None) for i, c in enumerate(col)], legend,
           loc="lower center", ncol=5, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
out = sys.argv[1] if len(sys.argv) > 1 else "docs/mujoco/assets/176_dr_effects_rough_ab.png"
fig.savefig(out, dpi=130)
print(out)
