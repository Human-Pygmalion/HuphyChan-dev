#!/usr/bin/env python3
"""Figure: M19 fine-tune (policy 49999 continued 2,500 iterations on model M19) against the parent and against same-length controls.
Inputs: dir with foot_variant_eval.py json outputs (tags below), dir with the foot-sole study json (terrain falls of the controls). Labels in English only.

  python3 plot_finetune.py <out_ft dir> <foot-sole study eval dir> <out.png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, SE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def falls(tag):
  p = R / f"{tag}.json"
  return json.load(open(p))["falls"] if p.exists() else None


# (label, color, {body: [tags over seeds]})
groups = [("parent 49999", "#7f7f7f", {"M10": ["dr_base", "par_dr_base_s1"], "M19": ["dr_m19", "par_dr_m19_s1", "par_dr_m19_s2"]}),
          ("control: +2,500 it, old foot,\nM10 body", "#9ecae1", {"M10": ["ctlbase_dr_base"], "M19": ["ctlbase_dr_m19"]}),
          ("control: +2,500 it, v5 foot,\nM10 body", "#6baed6", {"M10": ["ctlv5fm_dr_base"], "M19": ["ctlv5fm_dr_m19"]}),
          ("fine-tuned on M19\n(52,498)", "#d62728", {"M10": ["ft_dr_base", "ft_dr_base_s1"], "M19": ["ft_dr_m19", "ft_dr_m19_s1", "ft_dr_m19_s2"]})]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
w = 0.19
for gi, (lab, col, d) in enumerate(groups):
  for bi, body in enumerate(("M10", "M19")):
    x = bi + (gi - 1.5) * w
    v = [falls(t) for t in d[body]]
    ax.bar(x, np.mean(v), w * 0.92, color=col, hatch="//" if body == "M10" else None, edgecolor="k", lw=0.5, label=lab if bi == 0 else None)
    ax.plot([x] * len(v), v, "k.", ms=5)
    ax.text(x, max(v) + 0.4, "/".join(str(a) for a in v), ha="center", fontsize=7)
ax.set_xticks([0, 1])
ax.set_xticklabels(["evaluated on M10 (23.6 kg)", "evaluated on M19 (26.9 kg)"])
ax.set_ylabel("environments out of 64 that fell within 135 s")
ax.set_title("Domain randomization on, flat ground: bar = mean over seeds, dots = single seeds (seed 0, 1, 2)\ncontrols were run with seed 0 only", fontsize=9)
ax.legend(fontsize=7.5, loc="upper left")

ax = axs[1]
tests = [("parent 49999\nM10 body", "orig_49999", "#7f7f7f"), ("parent 49999\nM19 body", None, "#7f7f7f"), ("old foot +2,500\nM10 body", "base_52498", "#9ecae1"),
         ("v5 foot +2,500\nM10 body", "v5fullm_52498", "#6baed6"), ("fine-tuned\nM19 body", None, "#d62728")]
vals = {"slope_rough": [], "ridge": []}
for lab, key, col in tests:
  for terr in vals:
    if key:
      vals[terr].append(json.load(open(SE / f"{key}_{terr}.json"))["falls"])
    else:
      tag = ("par_" if lab.startswith("parent") else "ft_") + ("slope_m19" if terr == "slope_rough" else "ridge_m19")
      vals[terr].append(falls(tag))
x = np.arange(len(tests))
ax.bar(x - 0.2, vals["slope_rough"], 0.38, color=[c for _, _, c in tests], edgecolor="k", lw=0.5, label="rough slope")
ax.bar(x + 0.2, vals["ridge"], 0.38, color=[c for _, _, c in tests], edgecolor="k", lw=0.5, hatch="..", label="ridge")
for i in range(len(tests)):
  ax.text(i - 0.2, vals["slope_rough"][i] + 0.4, str(vals["slope_rough"][i]), ha="center", fontsize=8)
  ax.text(i + 0.2, vals["ridge"][i] + 0.4, str(vals["ridge"][i]), ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([t[0] for t in tests], fontsize=7.5)
ax.set_ylabel("environments out of 32 that fell within 135 s")
ax.set_title("Rough terrain, nominal simulation (left bar: rough slope, right bar: ridge)", fontsize=9)
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
