#!/usr/bin/env python3
"""Stage-6 figure: real 5 Hz burst vs the closest simulator reproduction (English labels).
  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/s6_figure.py --real "Expr_Data/vx0gain1walkob(진동확인용)1021.csv" \
      --sim "label=run.npz|condition|env" [--sim ...] --out docs/sim2real/img/2026-10-22_stage6_5hz.png
First --sim is drawn in the time-series panels, all are drawn in the spectrum panel.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from s6_real_record import load_real
from s6_metrics import ls_spec, _FR, sliding_band_amp, band_power
from s6_analyze import env_record


def spec(t, ys):
  return sum(ls_spec(t, ys[:, j]) for j in range(ys.shape[1])) / ys.shape[1]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--real", required=True)
  ap.add_argument("--sim", action="append", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--real-win", default="36,66", help="real time-series window (s)")
  a = ap.parse_args()
  r = load_real(a.real)
  rt = r["t"]
  sims = []
  for item in a.sim:
    lab, rest = item.split("=", 1)
    path, cond, k = rest.split("|")
    d = np.load(path)
    es = np.where(d["cond"] == cond)[0]
    e = int(es[int(k)]) if int(k) >= 0 else int(es[0])
    rec, fk = env_record(d, e)
    sims.append((lab, rec))
  w0, w1 = map(float, a.real_win.split(","))
  fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
  col = dict(tgt="#c0392b", pos="#1f4e79")
  # (a) real time series
  m = (rt >= w0) & (rt < w1)
  ax[0, 0].plot(rt[m], r["tgt"][m, 0] - np.mean(r["tgt"][:, 0]), color=col["tgt"], lw=1.2, label="target (policy output)")
  ax[0, 0].plot(rt[m], r["pos"][m, 0] - np.mean(r["pos"][:, 0]), color=col["pos"], lw=1.2, label="measured joint angle")
  ax[0, 0].set_ylabel("R hip_pitch [deg]")
  ax[0, 0].set_title(f"Real robot, command 0/0, gains 1.0, held by hand ({w0:.0f}-{w1:.0f} s of the log)")
  ax2 = ax[0, 0].twinx()
  gn = np.degrees(np.linalg.norm(r["gyro"], axis=1))
  ax2.plot(rt[m], gn[m], color="0.6", lw=0.8, label="body rate |gyro|")
  ax2.set_ylabel("body rate [deg/s]", color="0.4")
  ax[0, 0].legend(loc="upper left", fontsize=8)
  ax2.legend(loc="upper right", fontsize=8)
  # (b) sim time series
  lab, rec = sims[0]
  st = rec["t"]
  ax[1, 0].plot(st, rec["tgt"][:, 0] - rec["tgt"][(st > 2) & (st < 5), 0].mean(), color=col["tgt"], lw=1.2, label="target (policy output)")
  ax[1, 0].plot(st, rec["pos"][:, 0] - rec["pos"][(st > 2) & (st < 5), 0].mean(), color=col["pos"], lw=1.2, label="measured joint angle")
  ax[1, 0].axvline(5.0, color="k", ls="--", lw=0.8)
  ax[1, 0].text(5.1, ax[1, 0].get_ylim()[1] * 0.85, "impulse", fontsize=8)
  ax[1, 0].set_ylabel("R hip_pitch [deg]")
  ax[1, 0].set_xlabel("time [s]")
  ax[1, 0].set_title(f"Simulator: {lab}")
  ax[1, 0].legend(loc="lower left", fontsize=8)
  # (c) spectra
  def plot_spec(axx, t, y, **kw):
    axx.plot(_FR, spec(t, y), **kw)
  plot_spec(ax[0, 1], rt[(rt > 50) & (rt < 110)], r["act"][(rt > 50) & (rt < 110)][:, [0, 6]], color="k", lw=2, label="real burst (50-110 s)")
  plot_spec(ax[0, 1], rt[(rt > 5) & (rt < 35)], r["act"][(rt > 5) & (rt < 35)][:, [0, 6]], color="0.6", lw=1.5, label="real quiet (5-35 s)")
  cols = ["#c0392b", "#e67e22", "#2e7d32", "#6a1b9a"]
  for (lab_, rec_), c in zip(sims, cols):
    mm = (rec_["t"] > 7)
    plot_spec(ax[0, 1], rec_["t"][mm], rec_["act"][mm][:, [0, 6]], color=c, lw=1.4, label=f"sim: {lab_}")
  ax[0, 1].axvspan(3, 8, color="0.92", zorder=0)
  ax[0, 1].axvline(5.1, color="k", ls=":", lw=1)
  ax[0, 1].set_xlabel("frequency [Hz]")
  ax[0, 1].set_ylabel("Lomb-Scargle power (normalised)\nhip_pitch action, R and L mean")
  ax[0, 1].set_title("Spectrum after the disturbance (shaded: 3-8 Hz band)")
  ax[0, 1].legend(fontsize=7)
  # (d) band amplitude vs time since the disturbance (real: first gyro spike at 39.6 s of the log; sim: impulse at 5.0 s)
  cs, am = sliding_band_amp(r, "tgt", 0, rt[-1])
  ax[1, 1].plot(cs - 39.6, am, color="k", lw=1.5, label="real (disturbance = gyro spike at 39.6 s)")
  for (lab_, rec_), c in zip(sims, cols):
    cs2, am2 = sliding_band_amp(rec_, "tgt", 2.5, rec_["t"][-1])
    ax[1, 1].plot(cs2 - 5.0, am2, color=c, lw=1.4, label=f"sim: {lab_}")
  ax[1, 1].axvline(0, color="k", ls="--", lw=0.8)
  ax[1, 1].set_xlim(-6, 75)
  ax[1, 1].set_xlabel("time since the disturbance [s]")
  ax[1, 1].set_ylabel("3-8 Hz band amplitude of the targets [deg]\n(12-joint mean, 2 s windows)")
  ax[1, 1].legend(fontsize=7)
  ax[1, 1].set_title("Amplitude of the oscillation band")
  for axx in (ax[0, 1], ax[1, 1]):
    axx.grid(alpha=0.3)
  for axx in (ax[0, 0], ax[1, 0]):
    axx.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(a.out, dpi=130)
  print("wrote", a.out)


if __name__ == "__main__":
  main()
