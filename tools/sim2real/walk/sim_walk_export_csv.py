#!/usr/bin/env python3
"""Write sim_walk_record.py runs as CSV in the layout of the real HUPHY policy logs (Expr_Data/*walkob*.csv), one file
per run and env, so one analysis script reads both.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/sim_walk_export_csv.py \
      --data mujoco-sim/mjlab/analysis/out/s2r_walk [--runs R3,R6,R8] [--out <data>/csv]

Layout (checked against Expr_Data/vx0gain1walkob(...)1021.csv): first line "# limits ..." (copied), then the header
  t, obs/gyro/{x,y,z}, obs/grav/{x,y,z}, obs/q_prev/<12>, obs/q/<12>, obs/last_act/<12>, obs/cmd/{vx,wz},
  act/<12>, tgt/<12>, pos/<12>      with <12> = right_leg/{hip_pitch,hip_roll,hip_yaw,knee,ankle_a,ankle_b}, left_leg/{...}
Units: t [s] since policy start (row = one policy call); obs/gyro rad/s; obs/grav unit vector; obs/q_prev, obs/q = motor
angle minus default pose [rad]; obs/last_act, act = raw policy output; tgt = motor target [deg] (default + 0.25*act,
clipped); pos = motor angle [deg] = obs/q + default, i.e. the position the policy was given (in train-mode runs it
includes encoder bias, noise and lag exactly like the observation). ankle_a / ankle_b = crank_A / crank_B motors.
Rows after an env fell (terminated flag) are dropped.
"""
import argparse
import json
from pathlib import Path

import numpy as np

LIMITS = ("# limits right_leg/hip_pitch=-112.75:17.75 right_leg/hip_roll=-19.5:79.5 right_leg/hip_yaw=-40.5:40.5 right_leg/knee=6:114 "
          "right_leg/ankle_a=-61.88:61.88 right_leg/ankle_b=-61.88:61.88 right_leg/ankle_pitch=-40:40 right_leg/ankle_roll=-25:25 "
          "left_leg/hip_pitch=-17.75:112.75 left_leg/hip_roll=-79.5:19.5 left_leg/hip_yaw=-40.5:40.5 left_leg/knee=-114:-6 "
          "left_leg/ankle_a=-61.88:61.88 left_leg/ankle_b=-61.88:61.88 left_leg/ankle_pitch=-40:40 left_leg/ankle_roll=-25:25")
J = [f"{s}_leg/{j}" for s in ("right", "left") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")]
HEADER = (["t"] + [f"obs/gyro/{a}" for a in "xyz"] + [f"obs/grav/{a}" for a in "xyz"] + [f"obs/q_prev/{j}" for j in J]
          + [f"obs/q/{j}" for j in J] + [f"obs/last_act/{j}" for j in J] + ["obs/cmd/vx", "obs/cmd/wz"]
          + [f"act/{j}" for j in J] + [f"tgt/{j}" for j in J] + [f"pos/{j}" for j in J])


def export(npz, out_dir: Path, run: str):
  d = np.load(npz)
  timing = "n_calls" in d
  E = d["obs"].shape[1]
  default = d["default_q12"]
  for e in range(E):
    if timing:
      n = int(d["n_calls"][e])
      t, term = d["t_call"][:n, e], d["terminated"][:n, e]
    else:
      n = d["obs"].shape[0]
      t, term = d["t"], d["terminated"][:, e]
    keep = (np.cumsum(term) - term) == 0  # rows up to and including the first terminating step
    o, a, tg = d["obs"][:n, e], d["action"][:n, e], np.degrees(d["target"][:n, e])
    pos = np.degrees(o[:, 18:30] + default)
    rows = np.concatenate([t[:, None], o[:, :6], o[:, 6:18], o[:, 18:30], o[:, 30:42], o[:, 42:44], a, tg, pos], 1)[keep]
    p = out_dir / f"{run}_env{e}.csv"
    with open(p, "w") as f:
      f.write(LIMITS + "\n" + ",".join(HEADER) + "\n")
      for r in rows:
        f.write(",".join("0" if x == 0 else f"{x:.6g}" for x in r) + "\n")
  print(f"{run}: {E} files -> {out_dir}  ({len(HEADER)} columns, {int(keep.sum())} rows in last env)")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True)
  ap.add_argument("--runs", default="")
  ap.add_argument("--out", default="")
  a = ap.parse_args()
  D = Path(a.data)
  out = Path(a.out or D / "csv")
  out.mkdir(parents=True, exist_ok=True)
  runs = a.runs.split(",") if a.runs else sorted(p.stem for p in D.glob("*.npz"))
  for r in runs:
    export(D / f"{r}.npz", out, r)
