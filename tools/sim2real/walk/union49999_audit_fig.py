#!/usr/bin/env python3
"""Four-panel figure for docs/sim2real/2026-10-30_union49999_dr_coverage_and_risks.md (English labels only).

  a  policy-call interval, 1028 (checkpoint 49999, --hz 50) vs 1021 (37400, --hz 200)         [log-scale histogram]
  b  summed joint-angle PSD, standing rows only, 1028 vs 1021 vib-check (5 Hz peak)
  c  yaw rate of the straight-walk segments (vx > 0.1, |wz| < 0.05, >= 2 s) with the 37400 simulator reference
  d  lateral distance between the two ankle-joint origins (MuJoCo FK of the training model) vs time, with the default width and
     the hip-roll angle at which the model's legs first touch

  CUDA_VISIBLE_DEVICES="" mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/union49999_audit_fig.py \
      --xml <pygmalion_legonly_ab.xml> --logs Expr_Data/huphy_main_90bbd1f --old Expr_Data --audit <audit json> --out <png>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_walk_analyze as rwa  # noqa: E402
import union49999_audit as ua  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--xml", required=True)
ap.add_argument("--logs", required=True)
ap.add_argument("--old", required=True)
ap.add_argument("--audit", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
A = json.load(open(a.audit, encoding="utf-8"))
logs, old = Path(a.logs), Path(a.old)
d1 = pd.read_csv(logs / "1028_teleop_ob.csv", comment="#")
d2 = pd.read_csv(logs / "1028_teleop_extra2_ob.csv", comment="#")
do = pd.read_csv(old / "vx0gain1walkob(진동확인용)1021.csv", comment="#")

fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5))
# a
x = ax[0, 0]
bins = np.arange(0, 150, 1.0)
for d, lab, c in ((d1, "49999, 1028 teleop (n=%d)" % len(d1), "C0"), (do, "37400, 1021 vib-check (n=%d)" % len(do), "C3")):
  x.hist(np.diff(d["t"].to_numpy()) * 1e3, bins=bins, histtype="stepfilled", alpha=0.55, color=c, label=lab)
x.axvline(20, color="k", lw=0.8, ls="--")
x.set_yscale("log")
x.set_xlabel("policy-call interval [ms]  (training: 20 ms)")
x.set_ylabel("count")
x.set_title("a  Policy-call interval")
x.legend(fontsize=8)
# b
x = ax[0, 1]
for d, lab, c in ((d1, "49999 standing", "C0"), (d2, "49999 standing (run 2)", "C2"), (do, "37400 vib-check", "C3")):
  vx = d["obs/cmd/vx"].to_numpy() if "obs/cmd/vx" in d else np.zeros(len(d))
  wz = d["obs/cmd/wz"].to_numpy() if "obs/cmd/wz" in d else np.zeros(len(d))
  sel = (np.abs(vx) + np.abs(wz)) < 0.1
  t = d["t"].to_numpy()[sel]
  P = d[[f"pos/{j}" for j in ua.JOINTS]].to_numpy()[sel]
  dtm = float(np.median(np.diff(t)))
  tu, Pu = rwa.uniform(t, P, dtm)
  nper = int(round(4.0 / dtm))
  f, S = signal.welch(Pu, fs=1 / dtm, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=0)
  x.semilogy(f, S.sum(1), color=c, label=lab)
x.axvspan(4.5, 5.7, color="0.9")
x.set_xlim(0.3, 12)
x.set_xlabel("frequency [Hz]")
x.set_ylabel("summed PSD of 12 joint angles [deg^2/Hz]")
x.set_title("b  Standing rows only: 5 Hz band (grey)")
x.legend(fontsize=8)
# c
x = ax[1, 0]
rows = []
for k, d in (("run 1", d1), ("run 2", d2)):
  h = A["C_heading"]["1028_teleop" if k == "run 1" else "1028_teleop_extra2"]
  for s in h["straight_segments_ge_2s"]:
    rows.append((f"{k}\n{s['t0']:.0f}-{s['t1']:.0f}s", s["yaw_rate_deg_s"], s["dur_s"]))
x.bar(range(len(rows)), [r_[1] for r_ in rows], color=["C3" if r_[1] < 0 else "C0" for r_ in rows])
x.set_xticks(range(len(rows)))
x.set_xticklabels([r_[0] for r_ in rows], fontsize=6.5)
x.axhline(0, color="k", lw=0.8)
x.axhline(-1.8, color="C1", ls="--", lw=1.2, label="37400 simulator, vx 0.5 nominal (-1.8)")
x.axhline(-2.6, color="C1", ls=":", lw=1.2, label="37400 simulator, vx 1.0 nominal (-2.6)")
x.set_ylabel("mean yaw rate [deg/s]  (negative = clockwise / right)")
x.set_title("c  Straight-walk segments (vx > 0.1, |wz| < 0.05)")
x.legend(fontsize=8)
# d
import mujoco  # noqa: E402

m = mujoco.MjModel.from_xml_path(a.xml)
dat = mujoco.MjData(m)
jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
bR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "R_ankle_pitch_link")
bL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "L_ankle_pitch_link")


def width(row):
  dat.qpos[:] = m.qpos0
  for leg, s in (("right_leg", "R"), ("left_leg", "L")):
    for jn in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
      dat.qpos[m.jnt_qposadr[jid[f"{s}_{jn}_joint"]]] = math.radians(row[f"pos/{leg}/{jn}"])
  mujoco.mj_kinematics(m, dat)
  return float(dat.xpos[bL][1] - dat.xpos[bR][1])


x = ax[1, 1]
w0 = A["G_stance"]["1028_teleop"]["default_width_m"]
onset = A["G_stance"]["1028_teleop"]["model_leg_contact_onset"]["symmetric_deg"]
w_on = w0 - onset * 0.02862 * 1.0  # width at the contact onset (both legs inward), 2.862 cm per degree
for d, lab, c in ((d1, "run 1", "C0"), (d2, "run 2", "C2")):
  idx = np.arange(0, len(d), 2)
  W = np.array([width(d.iloc[i]) for i in idx]) * 100
  x.plot(d["t"].to_numpy()[idx], W, color=c, lw=0.8, label=lab)
x.axhline(w0 * 100, color="k", ls="--", lw=1, label="default pose (hips 24.7 cm)")
x.axhline(w_on * 100, color="C3", ls=":", lw=1.2, label=f"legs touch in the model (both hip rolls {onset:.1f} deg inward)")
x.set_xlabel("time [s]")
x.set_ylabel("ankle-origin lateral distance [cm]")
x.set_title("d  Ankle-to-ankle distance (FK of logged angles)")
x.legend(fontsize=7.5)
fig.tight_layout()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=130)
print("wrote", a.out)
