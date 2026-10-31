#!/usr/bin/env python3
"""Contact timeline of both feet for one env and one command segment, variants stacked: normal force [BW], sole height above the
ground (lowest collision point) and foot pitch. Shows whether a stance is one continuous contact or many short touches.
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/foot_contact_timeline.py <eval_dir> <assets_dir> <out.png> [seg] [env] [t0] [len]
"""
import json
import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from foot_gait_compare_lib import foot_vertices, quat_mat  # noqa: E402

EV, AS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEG = sys.argv[4] if len(sys.argv) > 4 else "vx0.6"
ENV = int(sys.argv[5]) if len(sys.argv) > 5 else 0
T0 = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0
LEN = float(sys.argv[7]) if len(sys.argv) > 7 else 4.0
XML = {"base": "pygmalion_legonly_ab.xml", "v3full": "pygmalion_legonly_ab_v3full.xml"}
DT = 0.02
fig, ax = plt.subplots(3, 2, figsize=(15, 8.5), sharex=True)
for c, v in enumerate(XML):
  D = np.load(EV / f"{v}.npz")
  meta = json.loads((EV / f"{v}.json").read_text())
  bw = meta["robot_mass_kg"] * 9.81
  n = int(round(meta["seconds_per_cmd"] / DT))
  k = [p[0] for p in meta["program"]].index(SEG)
  a = k * n + int(T0 / DT)
  b = a + int(LEN / DT)
  t = np.arange(b - a) * DT + T0
  m = mujoco.MjModel.from_xml_path(str(AS / XML[v]))
  for fi, (f, col) in enumerate((("R", "tab:red"), ("L", "tab:blue"))):
    R = quat_mat(D["foot_quat_w"][a:b, ENV, fi])
    gv = foot_vertices(m, f"{f}_foot_link")
    low = np.min(np.einsum("tj,kj->tk", R[:, 2, :], gv), -1) + D["foot_pos_w"][a:b, ENV, fi, 2]
    fn = D["sole"][a:b, ENV, fi, 0] / bw
    ax[0, c].plot(t, fn, color=col, lw=1.3, label=f"{f} foot")
    ax[1, c].plot(t, (low - np.percentile(low, 2)) * 1000, color=col, lw=1.3)
    ax[2, c].plot(t, np.degrees(np.arcsin(np.clip(R[:, 2, 0], -1, 1))), color=col, lw=1.3)
  ax[0, c].axhline(0.1, color="k", lw=0.6, ls=":")
  ax[0, c].set_title(f"{v}: {SEG}, env {ENV}", fontsize=10)
  ax[0, c].set_ylabel("normal force [BW]")
  ax[1, c].set_ylabel("sole height [mm]")
  ax[2, c].set_ylabel("foot pitch [deg] (toe up +)")
  ax[2, c].set_xlabel("time in segment [s]")
  ax[0, c].legend(fontsize=8)
  for r in range(3):
    ax[r, c].grid(alpha=0.3)
  ax[1, c].set_ylim(-2, 60)
fig.suptitle("Contact timeline: dotted line = 0.1 BW stance threshold", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=105)
print("saved", OUT)
