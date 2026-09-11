"""Which safety gate covers which command path. English labels only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

GATES = [
  ("NaN / infinity reject",        "A1 A2 A3", 1, 1),
  ("no-state reject",              "A4",       1, 1),
  ("joint range clip",             "A5 A6",    2, 2),
  ("speed limit (per command)",    "A7 A12",   1, 0),
  ("runaway cap (per measurement)","A8",       1, 1),
  ("resume-after-gap reseed",      "A10 A11",  1, 1),
  ("ankle solved as a pair",       "C1 C2",    1, 1),
  ("ankle joint range",            "C4",       2, 0),
  ("link-loss stop",               "D1 D2 D3", 2, 1),
  ("state freshness",              "D4",       0, 0),
  ("calibration gate on enable",   "E1 E2",    1, 1),
  ("hold-then-cut on exit",        "E4 E5",    1, 1),
  ("winding overheat cut",         "F1",       3, 0),
  ("motor fault flags",            "F2 F3",    3, 0),
  ("stuck-joint detection",        "-",        3, 0),
  ("arm token + deadman",          "-",        3, 0),
  ("IMU staleness",                "H2",       0, 0),
  ("policy output clip",           "H1",       0, 0),
]

COLOR = {0: "#c0392b", 1: "#1e8449", 2: "#d68910", 3: "#2471a3"}
LABEL = {0: "no gate anywhere", 1: "HUPHY gate, verified working",
         2: "HUPHY gate, but off or empty right now", 3: "only our adapter has it"}

fig, ax = plt.subplots(figsize=(13.0, 9.2))
ax.set_xlim(0, 10); ax.set_ylim(-1.6, len(GATES) + 2.6); ax.axis("off")

x_viewer, x_policy = 6.35, 8.35
ax.text(3.0, len(GATES) + 1.7, "safety gate", ha="center", fontsize=11.5, weight="bold")
ax.text(x_viewer, len(GATES) + 1.7, "viewer path\n(our adapter + HUPHY)",
        ha="center", fontsize=10.5, weight="bold")
ax.text(x_policy, len(GATES) + 1.7, "policy path\n(huphy-run, HUPHY only)",
        ha="center", fontsize=10.5, weight="bold")

for i, (name, codes, viewer, policy) in enumerate(GATES):
  y = len(GATES) - i
  ax.text(0.15, y, name, va="center", fontsize=10)
  ax.text(5.05, y, codes, va="center", ha="right", fontsize=8.5, color="#777")
  for x, state in ((x_viewer, viewer), (x_policy, policy)):
    ax.add_patch(Rectangle((x - 0.62, y - 0.32), 1.24, 0.64,
                           fc=COLOR[state], ec="none", alpha=0.88))
    mark = {0: "none", 1: "works", 2: "empty", 3: "adapter"}[state]
    ax.text(x, y, mark, ha="center", va="center", fontsize=8.6, color="white", weight="bold")
  ax.plot([0.1, 9.1], [y - 0.5, y - 0.5], lw=0.4, color="#e5e5e5")

handles = [Rectangle((0, 0), 1, 1, fc=COLOR[k], alpha=0.88) for k in (1, 2, 3, 0)]
ax.legend(handles, [LABEL[k] for k in (1, 2, 3, 0)], loc="lower center",
          ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("HUPHY safety gates: what each command path actually passes through\n"
             "40 adversarial scenarios run against HUPHY commit 4b30810, no hardware",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig("docs/img/huphy_safety_gate_coverage.png", dpi=140)
print("saved")
