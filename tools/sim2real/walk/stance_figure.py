#!/usr/bin/env python3
"""Figure of the stance-width study (English labels): a) stance width over time real vs sim, b) hip_roll abduction tgt/pos real vs sim,
c) command, d) one-variable tests (stance width and hip_roll abduction, walking 20 s segment) against the real value.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/stance_figure.py --xml <xml> --real <csv> --sim-dir <dir> --sim-name nom_L1 \
      --json <stance_report json> --out <png> [--cache dir]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stance_analyze import summarise
from stance_fk import FK


def smooth(y, n):
  k = np.ones(n) / n
  return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def abd(pos_or_tgt, side_mean=True):
  r = pos_or_tgt[:, 1]
  l = -pos_or_tgt[:, 7]
  return r, l


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True)
  ap.add_argument("--real", required=True)
  ap.add_argument("--sim-dir", required=True)
  ap.add_argument("--sim-name", default="nom_L1")
  ap.add_argument("--json", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--cache", default="")
  a = ap.parse_args()
  fk = FK(a.xml)
  cache = a.cache or None
  _, R = summarise("real", a.real, fk, cache)
  envs = sorted((Path(a.sim_dir) / "csv").glob(f"{a.sim_name}_env*.csv"))
  S = [summarise("sim", f, fk, cache)[1] for f in envs]
  J = json.loads(Path(a.json).read_text())
  n = min(len(R["t"]), min(len(s["t"]) for s in S))
  t = R["t"][:n]
  W = 25  # 0.5 s smoothing at 50 Hz
  C = {"real": "#d62728", "sim": "#1f77b4", "ink": "#222222"}
  fig = plt.figure(figsize=(14, 12))
  gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 2.2, 0.9, 0.01], hspace=0.5, wspace=0.22)
  ax1 = fig.add_subplot(gs[0, :])
  ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
  ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
  walk = R["vx"][:n] > 0.1
  for ax in (ax1, ax2, ax3):
    ax.fill_between(t, 0, 1, where=walk, color="#f2c14e", alpha=0.25, transform=ax.get_xaxis_transform(), lw=0)
  simw = np.array([s["SP"]["width"][:n] for s in S])
  simt = np.array([s["ST"]["width"][:n] for s in S])
  ax1.plot(t, smooth(R["ST"]["width"][:n], W), color=C["real"], ls="--", lw=1.0, label="real, policy target (FK of tgt)")
  ax1.plot(t, smooth(R["SP"]["width"][:n], W), color=C["real"], lw=1.6, label="real, measured (FK of pos)")
  ax1.plot(t, smooth(np.nanmean(simt, 0), W), color=C["sim"], ls="--", lw=1.0, label="simulator nominal, policy target")
  ax1.plot(t, smooth(np.nanmean(simw, 0), W), color=C["sim"], lw=1.6, label="simulator nominal, measured (mean of 8 envs)")
  ax1.axhline(0.2474, color="#777777", lw=0.8, ls=":")
  ax1.text(t[-1], 0.2474, " default pose 0.247 m", va="bottom", ha="right", fontsize=8, color="#555555")
  ax1.set_ylabel("stance width, foot centre to\nfoot centre, pelvis frame [m]")
  ax1.set_title("a) Stance width over time, same policy and same command trace (yellow = walking command, vx > 0.1 m/s)")
  ax1.legend(ncol=2, fontsize=8, loc="upper right")
  ax1.grid(alpha=0.3)
  rt, lt = abd(R["tgt"][:n]); rp, lp = abd(R["pos"][:n])
  sr_t = np.array([abd(s["tgt"][:n])[0] for s in S]).mean(0); sl_t = np.array([abd(s["tgt"][:n])[1] for s in S]).mean(0)
  sr_p = np.array([abd(s["pos"][:n])[0] for s in S]).mean(0); sl_p = np.array([abd(s["pos"][:n])[1] for s in S]).mean(0)
  m = lambda r, l: 0.5 * (r + l)  # noqa: E731
  ax2.plot(t, smooth(m(rt, lt), W), color=C["real"], ls="--", lw=1.0, label="real target")
  ax2.plot(t, smooth(m(rp, lp), W), color=C["real"], lw=1.6, label="real measured")
  ax2.plot(t, smooth(m(sr_t, sl_t), W), color=C["sim"], ls="--", lw=1.0, label="simulator target")
  ax2.plot(t, smooth(m(sr_p, sl_p), W), color=C["sim"], lw=1.6, label="simulator measured")
  ax2.axhline(0, color="#777777", lw=0.8, ls=":")
  ax2.set_ylabel("hip_roll abduction, mean of both legs\n[deg, + = foot outward]")
  ax2.set_title("b) Hip roll target and measured angle (abduction = +roll on the right leg, -roll on the left leg)")
  ax2.legend(ncol=4, fontsize=8, loc="upper right")
  ax2.grid(alpha=0.3)
  ax3.plot(t, R["vx"][:n], color=C["ink"], lw=1.0, label="vx [m/s]")
  ax3.plot(t, R["wz"][:n], color="#2ca02c", lw=1.0, label="wz [rad/s]")
  ax3.set_xlabel("time [s]"); ax3.set_ylabel("command"); ax3.legend(ncol=2, fontsize=8, loc="upper right"); ax3.grid(alpha=0.3)
  ax3.set_title("c) Teleoperation command of the real log, replayed in the simulator", fontsize=10)
  fig.subplots_adjust(bottom=0.355, top=0.96)

  # d) one-variable tests (reported angles = what the log of a real robot would show; diamond = true width from the true joint angles)
  sim = J["sim"]
  order = ["base", "payload4", "payload6", "payload8", "payload12", "kpR_0.8", "kpR_0.7", "kpR_0.6", "kdR_0.5", "kpkdR_0.7", "kpkdR_0.6", "frictR_2", "frictR_4",
           "frictAbsR_5", "encM2", "encM4", "encM-2", "imuR2", "imuR4", "imuR-2", "imuR-4", "footmu0.3", "footmu1.2", "torqR_0.7"]
  keys = []
  for v in order:
    for pre in ("ab_", "abx_"):
      if pre + v in sim:
        keys.append((v, pre + v))
  real = J["real"]["SEG"]["regimes"]["all"]
  lab, wp, wt, ab, fl = ["REAL"], [real["width_pos"]], [real["width_pos"]], [0.5 * (real["right_abd_pos"] + real["left_abd_pos"])], [0]
  for v, k in keys:
    f = sim[k]["flat"]["all"]
    lab.append(v); wp.append(f["width_pos"]); ab.append(0.5 * (f["right_abd_pos"] + f["left_abd_pos"]))
    wt.append(sim[k + "_true"]["flat"]["all"]["width_pos"] if k + "_true" in sim else f["width_pos"])
    fl.append(int(sim[k]["falls"] or 0))
  axd1 = fig.add_axes([0.07, 0.115, 0.42, 0.19])
  axd2 = fig.add_axes([0.56, 0.115, 0.42, 0.19])
  x = np.arange(len(lab))
  col = ["#d62728" if l == "REAL" else ("#7f7f7f" if l == "base" else "#1f77b4") for l in lab]
  hat = ["//" if n else "" for n in fl]
  for ax, y in ((axd1, wp), (axd2, ab)):
    ax.bar(x, y, color=col, hatch=None)
    ax.axhline(y[0], color="#d62728", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([l + (f' [fell {n}/4]' if n else '') for l, n in zip(lab, fl)], rotation=75, fontsize=7)
  axd1.plot(x, wt, "kD", ms=3, label="width from TRUE joint angles")
  axd1.legend(fontsize=7, loc="upper right")
  axd1.set_ylim(min(wt + wp) - 0.03, max(wp) + 0.02)
  axd1.set_ylabel("stance width [m]"); axd1.set_title("d) One-variable simulator tests, 20 s walking segment: stance width (from reported angles)", fontsize=9)
  axd2.set_ylabel("reported hip_roll\nabduction [deg]"); axd2.set_title("reported hip_roll abduction (mean of both legs)", fontsize=9)
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print("wrote", a.out)
