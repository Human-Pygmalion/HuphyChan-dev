#!/usr/bin/env python3
"""Offline check of the bench step and chirp logs against a one-joint simulation. CPU only, read-only.

2026-10-16. Companion of docs/172_real_motor_pd_tuning_guide.md section 5.
It never opens a CAN socket and never moves a motor. It reads System_ID/data/pd_pmstep_* and
pd_chirp_* (host-computed PD, pure torque commands, bare motor) and does three things:

  1. step     mean step response 0 -> +10 deg of the real motor and of the simulation (figure + metrics)
  2. chirp    closed-loop frequency response |q/q*| and phase of the real motor and of the simulation
              (coherence-averaged over the repeats, log-frequency smoothing)
  3. delay    grid fit of the loop delay and the inertia scale on the chirp logs (time domain RMSE)

The simulation is a one-degree-of-freedom rotor: inertia, viscous damping, Coulomb friction,
torque clip, feedback delay and (like the host loop) a 20 ms finite-difference velocity estimate.

    python3 tools/sysid/pd_prelim_analysis.py --out /tmp/pd_prelim
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "System_ID", "data") + os.sep
DEG = np.pi / 180
# motor id -> measured output-side values (pygmalion_constants.py: RS04 / RS03 measured set)
PARAM = {4: dict(I=0.0773, c=0.0275, fr=0.519, name="RS04"), 3: dict(I=0.0638, c=0.0252, fr=0.251, name="RS03")}
FS = 1000.0


def sim(t_tgt, u_deg, kp, kd, I, c, fr, tau_max=8.0, delay=0.0005, dt=2e-4, q0_deg=None,
        v_tgt=None, vel_win=0.02):
  """tau = kp (u - q_delayed) + kd (v_tgt - v_est_delayed), clipped; Coulomb friction at velocity level."""
  n = int(t_tgt[-1] / dt) + 1
  tt = np.arange(n) * dt
  u = np.interp(tt, t_tgt, u_deg) * DEG
  vt = np.zeros(n) if v_tgt is None else np.interp(tt, t_tgt, v_tgt) * DEG
  q = np.zeros(n)
  v = np.zeros(n)
  q[0] = (u_deg[0] if q0_deg is None else q0_deg) * DEG
  d = int(round(delay / dt))
  nw = int(round(vel_win / dt)) if vel_win else 0
  for i in range(n - 1):
    j = max(i - d, 0)
    vm = v[j] if nw == 0 or j - nw < 0 else (q[j] - q[j - nw]) / (nw * dt)
    tau = max(-tau_max, min(tau_max, kp * (u[i] - q[j]) + kd * (vt[i] - vm)))
    vf = v[i] + dt * (tau - c * v[i]) / I
    df = dt * fr / I
    vn = 0.0 if abs(vf) <= df else vf - np.sign(vf) * df
    v[i + 1] = vn
    q[i + 1] = q[i] + dt * vn
  return tt, q / DEG


def load(f):
  return np.genfromtxt(f, delimiter=",", names=True)


def step_metrics(t, y, a, b, tol=0.05):
  s = (y - a) / (b - a)
  fin = s[t > t[-1] - 0.1].mean()
  t90 = t[np.argmax(s >= 0.9)]
  out = np.abs(s - 1) > tol
  return dict(t90_ms=t90 * 1e3, overshoot_pct=max(0.0, (s.max() - 1) * 100),
              t_peak_ms=t[np.argmax(s)] * 1e3, settle_ms=(t[np.where(out)[0][-1]] if out.any() else 0) * 1e3,
              final_err_deg=(1 - fin) * (b - a))


def steps(mid, out):
  P = PARAM[mid]
  grid = np.arange(0, 1.5, 0.0005)
  files = sorted(glob.glob(DATA + f"pd_pmstep_{mid}_kp20_kd1_a10_r*_pm.csv"))
  Y, S = [], []
  for fi, f in enumerate(files):
    d = load(f)
    t, u, y = d["t_s"], d["target_deg"], d["pos_deg"]
    if fi == 0:
      ts, qs = sim(t, u, 20.0, 1.0, P["I"], P["c"], P["fr"], q0_deg=y[0])
    for j in np.where(np.abs(np.diff(u)) > 1)[0]:
      a, b = u[j], u[j + 1]
      if abs(a - u[0]) < 2 and b - a > 5 and t[j + 1] + 1.49 < t[-1]:     # 0 -> +A edge
        Y.append(np.interp(t[j + 1] + grid, t, y) - a)
        if fi == 0:
          S.append(np.interp(t[j + 1] + grid, ts, qs) - a)
  Y, S = np.array(Y), np.array(S)
  A = 10.0
  return dict(real=step_metrics(grid, Y.mean(0), 0, A), sim=step_metrics(grid, S.mean(0), 0, A), n=len(Y))


def frf(pairs):
  Suu = Syy = Syu = 0
  for u, y in pairs:
    w = np.hanning(len(u))
    U = np.fft.rfft((u - u.mean()) * w)
    Y = np.fft.rfft((y - y.mean()) * w)
    Suu = Suu + np.abs(U) ** 2
    Syy = Syy + np.abs(Y) ** 2
    Syu = Syu + Y * np.conj(U)
  return np.fft.rfftfreq(len(pairs[0][0]), 1 / FS), Syu / Suu, np.abs(Syu) ** 2 / (Suu * Syy)


def resample(t, u, y):
  tg = np.arange(0, t[-1], 1 / FS)
  return np.interp(tg, t, u), np.interp(tg, t, y)


def smooth(f, H, fg, frac=0.12):
  return np.array([H[(f > x * (1 - frac)) & (f < x * (1 + frac))].mean() for x in fg])


def chirps(out):
  cases = [("RS04 kp20 kd1 amp8", "pd_chirp_4_kp20_kd1_a8_r*_b.csv", 4, 20, 1, 8.0),
           ("RS04 kp20 kd1 amp30", "pd_chirp_4_kp20_kd1_a30_r*_c.csv", 4, 20, 1, 8.0),
           ("RS04 kp220 kd6 amp5", "pd_chirp_4_kp220_kd6_a5_r*_g.csv", 4, 220, 6, 40.0),
           ("RS03 kp20 kd1 amp8", "pd_chirp_3_kp20_kd1_a8_r*_b.csv", 3, 20, 1, 8.0)]
  fg = np.geomspace(0.3, 9, 60)
  res = {}
  fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
  for ax, (name, pat, mid, kp, kd, tmax) in zip(axs.ravel(), cases):
    P = PARAM[mid]
    files = sorted(glob.glob(DATA + pat))
    runs = [load(f) for f in files]
    pairs = [resample(d["t_s"], d["target_deg"], d["pos_deg"]) for d in runs]
    n = min(len(p[0]) for p in pairs)
    f, H, coh = frf([(a[:n], b[:n]) for a, b in pairs])
    Hr, cr = smooth(f, H, fg), smooth(f, coh, fg)
    ok = (cr > 0.9) & (fg < 8)
    d0 = runs[0]
    ts, qs = sim(d0["t_s"], d0["target_deg"], kp, kd, P["I"], P["c"], P["fr"], tau_max=tmax,
                 q0_deg=d0["pos_deg"][0], v_tgt=d0["target_vel_deg_s"])
    su, sy = resample(ts, np.interp(ts, d0["t_s"], d0["target_deg"]), qs)
    fs, Hs, _ = frf([(su[:n], sy[:n])])
    Hsr = smooth(fs, Hs, fg)
    band = (fg >= 0.5) & (fg <= 6) & ok
    res[name] = dict(
      n=len(files), f_n_hz=float(np.sqrt(kp / P["I"]) / 2 / np.pi),
      zeta=float((kd + P["c"]) / (2 * np.sqrt(kp * P["I"]))),
      rms_db=float(np.sqrt(np.mean((20 * np.log10(abs(Hr / Hsr)))[band] ** 2))),
      rms_deg=float(np.sqrt(np.mean((np.angle(Hr / Hsr) * 180 / np.pi)[band] ** 2))),
      peak_real_db=float(20 * np.log10(abs(Hr[ok]).max())), f_peak_real=float(fg[ok][np.argmax(abs(Hr[ok]))]),
      peak_sim_db=float(20 * np.log10(abs(Hsr[ok]).max())), f_peak_sim=float(fg[ok][np.argmax(abs(Hsr[ok]))]))
    ax.semilogx(fg[ok], 20 * np.log10(abs(Hr))[ok], "k", lw=2, label=f"real ({len(files)} runs)")
    ax.semilogx(fg, 20 * np.log10(abs(Hsr)), "g", lw=1.5, label="simulation")
    ax.set_title(f'{name}  (f_n={res[name]["f_n_hz"]:.2f} Hz, zeta={res[name]["zeta"]:.2f})', fontsize=10)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|q/q*| [dB]")
    ax.grid(True, which="both", alpha=.3)
  axs[0, 0].legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(os.path.join(out, "chirp_bode.png"), dpi=110)
  return res


def delay_grid(out):
  res = {}
  for name, pat, mid, kp, kd, tmax in [("RS04 kp20 a8", "pd_chirp_4_kp20_kd1_a8_r*_b.csv", 4, 20, 1, 8.0),
                                       ("RS04 kp220 kd6 a5", "pd_chirp_4_kp220_kd6_a5_r*_g.csv", 4, 220, 6, 40.0)]:
    P = PARAM[mid]
    runs = [load(f) for f in sorted(glob.glob(DATA + pat))[:3]]
    grid = {}
    for vw in (None, 0.02):
      for T in (0.0, 0.5, 1.0, 2.0, 3.0):
        for sI in (0.9, 1.0, 1.1):
          e = []
          for d in runs:
            ts, qs = sim(d["t_s"], d["target_deg"], kp, kd, P["I"] * sI, P["c"], P["fr"], tau_max=tmax,
                         delay=T * 1e-3, dt=2e-4, q0_deg=d["pos_deg"][0], v_tgt=d["target_vel_deg_s"], vel_win=vw)
            sel = d["t_s"] > 0.5
            e.append(np.sqrt(np.mean((np.interp(d["t_s"], ts, qs)[sel] - d["pos_deg"][sel]) ** 2)))
          grid[f"vel_win={vw}|delay_ms={T}|I_scale={sI}"] = float(np.mean(e))
    res[name] = dict(best=min(grid, key=grid.get), best_rmse_deg=min(grid.values()), grid=grid)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="/tmp/pd_prelim")
  ap.add_argument("--skip-delay", action="store_true", help="the delay grid takes a few minutes")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  res = {"step": {PARAM[m]["name"]: steps(m, a.out) for m in (4, 3)}, "chirp": chirps(a.out)}
  if not a.skip_delay:
    res["delay_grid"] = delay_grid(a.out)
  with open(os.path.join(a.out, "pd_prelim.json"), "w") as fh:
    json.dump(res, fh, indent=1)
  print(json.dumps({k: v for k, v in res.items() if k != "delay_grid"}, indent=1))


if __name__ == "__main__":
  main()
