#!/usr/bin/env python3
"""Which part of the real observation drives the deployed policy's large actions?

For every logged policy call the recorded 45-dim observation is fed to the deployed weights
(numpy, exactly as on the robot) and again with one observation group replaced by its nominal
standing value (gyro 0, gravity (0,0,-1), joint history = default pose, last action 0) or by a
sign-flipped IMU candidate. The drop in |action| tells which group drives it.

Also prints the first policy calls of each run, where the action grows before the robot has
moved much.

  python tools/sim2real/huphy_obs_attribution.py --huphy <HUPHY biped checkout>
"""
import argparse, glob, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--huphy", required=True)
ap.add_argument("--weights", default="config/policies/stand_bi.pt")
ap.add_argument("--first", type=int, default=8)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.huphy, "src"))
from huphy.control import rsl_rl, stand  # noqa: E402

model = rsl_rl.load(os.path.join(a.huphy, a.weights), spec=stand.SPEC, action_dim=12)
G = {"gyro": slice(0, 3), "grav": slice(3, 6), "q_prev": slice(6, 18), "q": slice(18, 30), "last_act": slice(30, 42)}
NOM = {"gyro": 0.0, "grav": np.array([0, 0, -1.0]), "q_prev": 0.0, "q": 0.0, "last_act": 0.0}


def run(O):
  return np.stack([model(o) for o in O])


def variants(obs):
  out = {"recorded": obs}
  for g, s in G.items():
    o = obs.copy(); o[:, s] = NOM[g]; out[f"{g}->nominal"] = o
  o = obs.copy(); o[:, [0, 1, 2, 3, 4, 5]] = [0, 0, 0, 0, 0, -1]; out["imu->nominal"] = o
  o = obs.copy(); o[:, 1:3] *= -1; out["gyro y,z flipped"] = o
  o = obs.copy(); o[:, 3] *= -1; out["grav x flipped"] = o
  o = obs.copy(); o[:, 0:2] *= -1; o[:, 3:5] *= -1; out["IMU yaw 180deg"] = o
  return out


for f in sorted(glob.glob(os.path.join(a.huphy, "archivedlogs", "*.csv"))):
  if not open(f).readline().startswith("t,obs"):
    continue
  d = np.genfromtxt(f, delimiter=",", skip_header=1)
  t, obs, act = d[:, 0], d[:, 1:46], d[:, 46:58]
  print(f"\n=== {os.path.basename(f)}  ({len(t)} calls)")
  print("first calls: t, gyro(rad/s), grav, max|q-default|deg, max|act|, argmax joint")
  for i in range(min(a.first, len(t))):
    print(f"  {t[i]-t[0]:5.2f}s gyro {np.round(obs[i,0:3],2)} grav {np.round(obs[i,3:6],2)} "
          f"|dq|max {np.degrees(np.abs(obs[i,18:30]).max()):4.1f} |act|max {np.abs(act[i]).max():5.2f} "
          f"{stand.ORDER[int(np.abs(act[i]).argmax())]}")
  for span, rows in (("first 0.5 s", t - t[0] < 0.5), ("whole run", np.ones(len(t), bool))):
    O = obs[rows]
    base = None
    line = []
    for name, o in variants(O).items():
      A = run(o)
      m = np.sqrt((A ** 2).mean())
      if base is None:
        base = m
      line.append(f"{name} {m:.2f}")
    print(f"  {span:11s} action RMS: " + " | ".join(line))
