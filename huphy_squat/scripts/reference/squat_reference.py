#!/usr/bin/env python3
"""Squat reference q*(h) and default pose of robot.py, CPU MuJoCo (+ scipy).

phi0 = robot.HIP_SPREAD hip roll outward (R +, L -), feet level and pointing ahead (hip yaw,
ankle pitch / roll solved), pelvis upright. Two segments, knee stepped by 0.025 rad:
  * shin-forward balanced pose (CoM x over the sole centre) while ankle pitch <= ANKLE_CAP;
  * below that, ankle pitch held at ANKLE_CAP and the knee bent further: the CoM moves behind
    the sole centre (reported as com_dx) -- the policy has to sit back slightly.
Also: the default (action-offset) pose at hip pitch / knee of robot.DEFAULT_MOTOR_POS with the
spread (hip yaw and cranks solved), foot width.

The SQUAT_REF_* tuples in robot.py are this table linearly interpolated in h at SQUAT_REF_H,
rounded to 4 decimals; they are printed in that form and compared with robot.py.

  squat_reference.py [--out squat_reference.json]   (default: next to this script)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from huphy_squat import robot
from spread import Spread

PHI0 = robot.HIP_SPREAD
ANKLE_CAP = 0.43  # soft limit 0.454 -> 0.024 rad left for balance
OUT = Path(__file__).resolve().parent / "squat_reference.json"
# robot.py name -> table column
REF_COLUMNS = {"KNEE": "knee", "HIP_PITCH": "hip_pitch", "ANKLE_PITCH": "ankle_pitch", "HIP_YAW": "hip_yaw",
               "ANKLE_ROLL": "ankle_roll", "FOOT_WIDTH": "foot_width", "COM_DX": "com_dx"}


def row(s: Spread, phi, hp, k):
  s.set_pose(phi, hp, k)
  r = s.report(phi, k, hp)
  r["com_dx"] = float(s.d.subtree_com[s.m.body("base_link").id][0] - s.sole_centre())
  return r


def resample(table: list[dict]) -> dict[str, tuple[float, ...]]:
  """Columns of `table` interpolated at robot.SQUAT_REF_H, rounded like robot.py."""
  t = sorted(table, key=lambda r: r["h"])
  h = np.array([r["h"] for r in t])
  return {name: tuple(round(float(v), 4) for v in np.interp(robot.SQUAT_REF_H, h, [r[col] for r in t]))
          for name, col in REF_COLUMNS.items()}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", type=Path, default=OUT)
  out_path = ap.parse_args().out
  s = Spread()
  out = {"phi0": PHI0, "ankle_cap": ANKLE_CAP}
  hp0, k0 = robot.DEFAULT_MOTOR_POS["R_hip_pitch_joint"], robot.DEFAULT_MOTOR_POS["R_knee_joint"]
  # default pose
  for phi in (0.0, PHI0):
    s.set_pose(phi, hp0, k0)
    names = [f"{S}_{j}_joint" for S in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
    fy = abs(s.d.xpos[s.foot["R"]][1] - s.d.xpos[s.foot["L"]][1])
    gy = []
    for S in "LR":
      ys = [s.d.geom_xpos[g][1] + sg * s.m.geom_size[g][1] for g in s.fgeoms[S] for sg in (-1, 1)]
      gy.append((min(ys), max(ys)))
    gap = max(gy[0][0], gy[1][0]) - min(gy[0][1], gy[1][1])  # inner edges (sole boxes, level feet)
    print(f"default pose phi {phi}: foot-origin width {fy:.3f} m, inner sole gap {gap:.3f} m, h {s.d.qpos[2]:.4f}")
    print("   ", {n: round(s.q(n), 6) for n in names},
          "ankle", round(s.q("R_ankle_pitch_joint"), 4), round(s.q("R_ankle_roll_joint"), 4))
    if phi == PHI0:
      out["default_motor_pos"] = {n: s.q(n) for n in names}
      out["default_h"] = float(s.d.qpos[2])
      dev = max(abs(round(s.q(n), 6) - robot.DEFAULT_MOTOR_POS[n]) for n in names)
      print(f"    max |diff| to robot.DEFAULT_MOTOR_POS: {dev:.2e}")
  # segment 1: balanced, shin forward
  table = []
  for k in np.arange(0.35, 1.70, 0.025):
    roots = s.balanced(PHI0, float(k))
    if not roots:
      continue
    r = row(s, PHI0, roots[0], float(k))
    if r["ankle_pitch"] > ANKLE_CAP:
      break
    r["segment"] = "balanced"
    table.append(r)
  # segment 2: ankle pitch held at the cap (hip pitch solved for it), knee further
  last_k = table[-1]["knee"]

  def ankle_err(hp, k):
    s.set_pose(PHI0, hp, k)
    return s.q("R_ankle_pitch_joint") - ANKLE_CAP

  for k in np.arange(last_k + 0.025, 1.98, 0.025):
    try:
      hp = brentq(lambda x: ankle_err(x, float(k)), -1.9, 0.0, xtol=1e-9)
    except ValueError:
      break
    r = row(s, PHI0, hp, float(k))
    r["segment"] = "ankle_cap"
    table.append(r)
    if r["h"] < 0.60 or r["violations"]:
      break
  print(f"\nphi0 {PHI0}  h       knee    hip_p   ank_p   ank_r   hip_y   crankA  crankB  width  com_dx  seg viol")
  for r in table:
    print(f"        {r['h']:.4f} {r['knee']:.3f} {r['hip_pitch']:+.3f} {r['ankle_pitch']:.3f} {r['ankle_roll']:+.3f} "
          f"{r['hip_yaw']:+.3f} {r['crank_A']:+.3f} {r['crank_B']:+.3f} {r['foot_width']:.3f} {r['com_dx']:+.4f} "
          f"{r['segment'][:4]} {r['violations']}")
  out["table"] = table
  ref = resample(table)
  out["squat_ref"] = {"H": list(robot.SQUAT_REF_H), **{k: list(v) for k, v in ref.items()}}
  print(f"\nresampled at robot.SQUAT_REF_H (robot.py form):\nSQUAT_REF_H = {robot.SQUAT_REF_H}")
  worst = 0.0
  for name, vals in ref.items():
    cur = getattr(robot, f"SQUAT_REF_{name}")
    d = max(abs(a - b) for a, b in zip(vals, cur, strict=True))
    worst = max(worst, d)
    print(f"SQUAT_REF_{name} = {vals}   # max |diff| to robot.py {d:.4f}")
  print(f"max |diff| to robot.py over all SQUAT_REF_* columns: {worst:.4f}")
  out_path.write_text(json.dumps(out, indent=1))
  print("wrote", out_path)


if __name__ == "__main__":
  main()
