#!/usr/bin/env python3
"""Real-log reference for stage 6: per 10 s window, 3-8 Hz band amplitude of the target (12-joint) and the Lomb-Scargle peak of the R+L hip_pitch
action in 1-9 Hz, counted by class. Same metric code as the simulator runs (s6_metrics.py).
  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/s6_real_table.py --out docs/sim2real/data/2026-10-22_stage6_real_reference.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from s6_real_record import load_real
from s6_metrics import band_power, ls_peak, peak_fit, sliding_band_amp

RUNS = [("vib-check gain 1/1 (53 ms)", "vx0gain1walkob(진동확인용)1021.csv"), ("Kd x1.5 (53 ms)", "vx0gain1kdx1.5walkob1021.csv"),
        ("gain 0.6/1 run A (63 ms)", "1021ob061vx0.csv"), ("gain 0.6/1 run B (55 ms)", "1021vx0obgain0.61.csv"),
        ("1020 gain 1/1 (31 ms)", "policy_generalized_walk_20261020_185916.csv")]


def classify(amp, f):
  if amp < 0.5:
    return "quiet"
  if amp >= 1.0 and 4.5 <= f <= 6.0:
    return "line_5Hz"
  if amp >= 1.0 and f < 4.5:
    return "slow_1_4.5Hz"
  return "other"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", default="Expr_Data")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  out = {}
  for label, f in RUNS:
    r = load_real(f"{a.data}/{f}")
    t = r["t"]
    end = t[-1]
    wins = []
    for w0 in np.arange(0, end - 10.0 + 1e-9, 10.0):
      amp = float(np.sqrt(2 * band_power(t, r["tgt"], w0, w0 + 10)))
      pamp = float(np.sqrt(2 * band_power(t, r["pos"], w0, w0 + 10)))
      fq, h = ls_peak(t, r["act"][:, [0, 6]], w0, w0 + 10)
      g = np.degrees(np.linalg.norm(r["gyro"][(t >= w0) & (t < w0 + 10)], axis=1)).max()
      wins.append(dict(t0=float(w0), amp_tgt=amp, amp_pos=pamp, f_hp=fq, cls=classify(amp, fq), gyro_max=float(g)))
    cnt = {k: sum(w["cls"] == k for w in wins) for k in ("quiet", "line_5Hz", "slow_1_4.5Hz", "other")}
    gm = np.degrees(np.linalg.norm(r["gyro"], axis=1))
    out[label] = dict(file=f, duration_s=float(end), interval_ms=float(np.median(np.diff(t)) * 1e3), windows=wins, counts=cnt,
                      gyro_max_dps=float(gm.max()), tilt_max_deg=float(r["tilt"].max()))
    print(f"{label:30s} {end:5.0f} s  windows {len(wins)}  {cnt}  gyro max {gm.max():.0f} tilt max {r['tilt'].max():.1f}")
  # burst numbers of the vib-check run
  r = load_real(f"{a.data}/vx0gain1walkob(진동확인용)1021.csv")
  t = r["t"]
  b = {}
  for nm, (a0, a1) in {"quiet_5_35": (5, 35), "burst1_50_110": (50, 110), "burst2_135_190": (135, 190)}.items():
    b[nm] = dict(amp_tgt=float(np.sqrt(2 * band_power(t, r["tgt"], a0, a1))), amp_pos=float(np.sqrt(2 * band_power(t, r["pos"], a0, a1))),
                 power_tgt_deg2=band_power(t, r["tgt"], a0, a1), power_pos_deg2=band_power(t, r["pos"], a0, a1),
                 ls_act_hp=ls_peak(t, r["act"][:, [0, 6]], a0, a1), ls_pos_hp=ls_peak(t, r["pos"][:, [0, 6]], a0, a1),
                 ls_act_knee=ls_peak(t, r["act"][:, [3, 9]], a0, a1), ls_act_roll=ls_peak(t, r["act"][:, [1, 7]], a0, a1),
                 ls_pos_knee=ls_peak(t, r["pos"][:, [3, 9]], a0, a1), ls_pos_roll=ls_peak(t, r["pos"][:, [1, 7]], a0, a1),
                 fit_hip_pitch=peak_fit(r, a0, a1, "hip_pitch"), fit_knee=peak_fit(r, a0, a1, "knee"), fit_hip_roll=peak_fit(r, a0, a1, "hip_roll"))
  out["vib_burst_numbers"] = b
  cs, am = sliding_band_amp(r, "tgt", 0, t[-1])
  out["vib_amp_series"] = (cs.tolist(), am.tolist())
  json.dump(out, open(a.out, "w"), default=lambda o: o if not isinstance(o, np.generic) else o.item())
  print(json.dumps(b, indent=1, default=float)[:1800])


if __name__ == "__main__":
  main()
