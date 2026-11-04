#!/usr/bin/env python3
"""Figure: mass before -> after for every part converted to aluminium (per leg) and the robot totals. English labels only.
usage: mass_figure.py <out.png>   (reads the round-2 report json)"""
import json, sys
from pathlib import Path
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_al_round2 as R2
B = R2.B
rep = json.load(open(B.ASSETS / f"{R2.OUT}_report.json"))
parts = rep["parts"]
order = sorted(parts, key=lambda k: -parts[k]["increase_g_per_leg"])
tot_leg = sum(v["increase_g_per_leg"] for v in parts.values())
mass = lambda n, xml=True: mujoco.MjModel.from_xml_path(str(B.ASSETS / f"{n}.xml")).body_mass.sum()
base_name = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455"
steps = [("M16\nfootv5+D455", mass(base_name + "_loop")), ("M17\n+plates, cam", mass(base_name + "_reinf_loop")),
         ("M18\n+cranks, mount", mass(base_name + "_reinf_alparts_loop")), ("M19 (latest)\n+round 2", mass(R2.OUT))]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1.7, 1]})
y = range(len(order))
a1.barh(list(y), [parts[k]["new_mass_g"] for k in order], color="#f21414", label="after (aluminium 6061, CAD volume x 2.70 g/cm3)")
a1.barh(list(y), [parts[k]["old_mass_g"] for k in order], color="#9aa3b0", label="before (as carried by the model: printed, or workbook Al mass for AnkleA2B)")
for i, k in enumerate(order):
    p = parts[k]
    a1.text(p["new_mass_g"] + 3, i, f"+{p['increase_g_per_leg']:.1f} g", va="center", fontsize=10)
a1.set_yticks(list(y)); a1.set_yticklabels([f"{k}  (round {parts[k]['round']})" if not k.startswith("AnkleA2B") else f"{k}  (Al before; CAD value)" for k in order], fontsize=10); a1.invert_yaxis()
a1.set_xlabel("mass per leg [g]"); a1.set_xlim(0, 380)
a1.set_title(f"Per part, one leg: +{tot_leg:.1f} g  (both legs +{2*tot_leg/1000:.3f} kg)")
a1.legend(loc="lower right", fontsize=9)
a2.bar(range(len(steps)), [s[1] for s in steps], color=["#9aa3b0", "#9aa3b0", "#d98080", "#f21414"])
for i, (n, m) in enumerate(steps):
    a2.text(i, m + 0.05, f"{m:.3f} kg", ha="center", fontsize=10)
a2.set_xticks(range(len(steps))); a2.set_xticklabels([s[0] for s in steps], fontsize=9)
a2.set_ylim(24, 28); a2.set_ylabel("robot mass, MJCF [kg]")
a2.set_title(f"Robot: {steps[-1][1] - steps[-2][1]:+.3f} kg this round\n{steps[-1][1] - steps[1][1]:+.3f} kg since the plates", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=105)
print("wrote", sys.argv[1], [round(s[1], 4) for s in steps])
