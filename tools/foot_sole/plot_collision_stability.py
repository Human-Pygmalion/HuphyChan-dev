#!/usr/bin/env python3
"""Figure: which foot collision model gave the steadiest policy results. Falls out of 32 environments per terrain (3090 GPU, nominal sim).
Numbers copied from docs/185, 186, 187 (same policy, collision model swapped) and study/compare_v5clean_retrain_*.txt, compare_v5fullm_seed_*.txt
(policy fine-tuned 2,500 iterations on each foot). Labels in English only.   python3 plot_collision_stability.py <out.png>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T = ["flat", "ridge", "slope\nrough", "stones\n2 cm", "stones\n1 cm", "pebbles"]
A = {"29 convex pieces": [0, 0, 23, 1, 1, 0], "v5cap (capsules)": [0, 0, 3, 1, 2, 0], "v5rim": [0, 0, 4, 1, 0, 0], "v5clean": [0, 0, 4, 0, 1, 0]}
B = {"51,000": {"29 pieces, seed 43": [0, 0, 23, 0, 1, 0], "29 pieces, seed 44": [0, 1, 8, None, None, None], "v5clean (capsules)": [17, 6, 19, 15, 14, 18]},
     "52,498": {"29 pieces, seed 43": [0, 1, 5, 7, 14, 0], "29 pieces, seed 44": [0, 0, 5, None, None, None], "v5clean (capsules)": [0, 0, 10, 32, 32, 31]}}
col = {"29 convex pieces": "#7f7f7f", "v5cap (capsules)": "#1f77b4", "v5rim": "#2ca02c", "v5clean": "#ff7f0e", "29 pieces, seed 43": "#7f7f7f",
       "29 pieces, seed 44": "#bdbdbd", "v5clean (capsules)": "#ff7f0e"}
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)


def bars(ax, d, title):
  n = len(d)
  w = 0.8 / n
  for i, (k, v) in enumerate(d.items()):
    x = np.arange(len(T)) + (i - (n - 1) / 2) * w
    vv = [np.nan if a is None else a for a in v]
    ax.bar(x, vv, w * 0.95, color=col[k], edgecolor="k", lw=0.4, label=k)
    for xi, a in zip(x, v):
      if a is not None and a > 0:
        ax.text(xi, a + 0.4, str(a), ha="center", fontsize=7)
  ax.set_xticks(range(len(T)))
  ax.set_xticklabels(T, fontsize=8)
  ax.set_title(title, fontsize=9)
  ax.legend(fontsize=7.5, loc="upper left")


bars(axs[0], A, "A. Same policy, only the collision model swapped\n(no retraining, checkpoint 51,000 of the 29-piece fine-tune)")
bars(axs[1], B["51,000"], "B1. Policy fine-tuned 2,500 iterations on each foot\ncheckpoint 51,000")
bars(axs[2], B["52,498"], "B2. Same runs, last checkpoint 52,498")
axs[0].set_ylabel("environments out of 32 that fell within 135 s")
axs[0].set_ylim(0, 36)
fig.suptitle("Foot collision model vs falls: swapping the model is quiet (A); retraining on capsules is not (B). Seed 43 unless noted, one run per bar.", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(sys.argv[1], dpi=140)
print("wrote", sys.argv[1])
