"""Figure for docs/180: net regeneration current at the battery (lossless upper bound) from measure_full runs."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "tools/power")
from regen_current import ACTUATED
O = "mujoco-sim/mjlab/analysis/out/"
runs = ["v10afinal_fcp", "v10bfinal_fcp", "v7own_fcp", "v6b_fcp"]
V = 48.0
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for r in runs:
    d = np.load(O + r + ".npz")
    bus = sum(d[f"Pmech_{s}_{j}"] for s in "LR" for j in ACTUATED)
    i = np.clip(-bus, 0, None) / V
    x = np.sort(i[i > 0])[::-1]
    ax[0].loglog(np.arange(1, len(x) + 1) / len(i) * 100, x, label=r)
ax[0].set_ylim(0.1, 40); ax[0].set_xlabel("share of time current is exceeded [%]"); ax[0].set_ylabel("net regen current at 48 V [A]")
ax[0].set_title("Net regeneration current, exceedance (50 Hz samples, lossless)"); ax[0].grid(True, which="both", alpha=.3); ax[0].legend()
res = json.load(open("docs/power/2026-10-19_regen_current_sim.json"))
keys = ["max_sample", "window_0.1s", "window_1s", "window_10s"]; lab = ["20 ms", "0.1 s", "1 s", "10 s"]
w = 0.1
for k, rr in enumerate(res):
    ax[1].bar(np.arange(4) + (k - len(res) / 2) * w, [rr["regen_current_a"][q] for q in keys], w, label=rr["file"].replace(".npz", ""))
ax[1].set_xticks(np.arange(4)); ax[1].set_xticklabels(lab); ax[1].set_xlabel("averaging window")
ax[1].set_ylabel("max net regen current [A]"); ax[1].set_title("Max of moving average (all 8 runs)"); ax[1].legend(fontsize=7, ncol=2); ax[1].grid(True, axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("docs/power/2026-10-19_regen_current_sim.png", dpi=120)
print("ok")
