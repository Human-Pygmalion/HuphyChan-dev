#!/usr/bin/env python3
"""Closed-loop frequency-response probe of stage 6: sinusoidal base torque (s6_oscillation_lab.py exc_*) -> response of the policy target at the same frequency.
Per environment, window 8-15 s (steady): sinusoid amplitude at f of every joint target (deg), of the base angular rate on the excited axis (deg/s, 200 Hz, true),
and of the tilt. 'tgt12' = sqrt(mean over 12 joints of amp^2) (same weighting as the band amplitude in s6_metrics.py, sqrt(2*power)).
  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/s6_exc.py --npz b3_s1.npz --json out.json [--real-csv Expr_Data/..vib.csv]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from s6_metrics import sine_fit

W0, W1 = 8.0, 15.0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True)
  ap.add_argument("--json", required=True)
  a = ap.parse_args()
  d = np.load(a.npz)
  res = {}
  tstep = np.arange(d["step_gyro"].shape[0]) * 0.005
  ms = (tstep >= W0) & (tstep < W1)
  for c in list(dict.fromkeys(d["cond"].tolist())):
    mt = re.match(r"E (pitch|roll) torque ([\d.]+) Nm (\d+) Hz", c)
    if not mt:
      continue
    ax = 1 if mt.group(1) == "pitch" else 0
    f = float(mt.group(3))
    rows = []
    for e in np.where(d["cond"] == c)[0]:
      if d["fell_k"][e] >= 0:
        continue
      n = int(d["n_calls"][e])
      t = d["t_call"][:n, e].astype(float)
      m = (t >= W0) & (t < W1)
      amps = np.array([sine_fit(t[m], np.degrees(d["tgt"][:n, e][m, j]), f)[0] for j in range(12)])
      g = np.degrees(d["step_gyro"][ms, e, ax])
      ga = sine_fit(tstep[ms], g, f)[0]
      ti = np.degrees(d["step_tilt"][ms, e])
      rows.append(dict(tgt12=float(np.sqrt((amps ** 2).mean())), tgt_hp=float(amps[[0, 6]].mean()), gyro_amp=float(ga), tilt_amp=float(sine_fit(tstep[ms], ti, f)[0])))
    res[c] = dict(f=f, axis=mt.group(1), n=len(rows), **{k: float(np.median([r[k] for r in rows])) if rows else float("nan") for k in ("tgt12", "tgt_hp", "gyro_amp", "tilt_amp")})
    res[c]["tgt_per_gyro"] = res[c]["tgt12"] / res[c]["gyro_amp"] if rows else float("nan")
    print(f"{c:30s} n {len(rows)}  tgt12 {res[c]['tgt12']:.2f} deg  hip_pitch {res[c]['tgt_hp']:.2f}  gyro {res[c]['gyro_amp']:.1f} deg/s  tilt {res[c]['tilt_amp']:.2f} deg  tgt/gyro {res[c]['tgt_per_gyro']:.3f}")
  json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
  main()
