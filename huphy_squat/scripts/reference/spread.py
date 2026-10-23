#!/usr/bin/env python3
"""Squat height vs hip spread (hip roll outward), task MJCF, CPU MuJoCo (+ scipy).

Pose family (symmetric L/R): pelvis upright, hip roll = +-phi (outward), knee k, hip pitch hp
solved so the whole-robot CoM x is over the sole centre; hip yaw, ankle pitch, ankle roll
solved so each foot is exactly level and points straight ahead; rods + cranks from the loop.
h = pelvis height with the lowest sole corner on z = 0.

Constraint checks (90 % soft range unless noted): ankle pitch / roll (passive), cranks within
the action clip +-0.95, hip yaw, knee. Prints per phi the lowest feasible h and the pose.

  spread.py [--out spread.json]   (default: next to this script)

`Spread` is also the pose solver of squat_reference.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq, least_squares

from huphy_squat import robot

SOFT = 0.9
CRANK_CLIP = 0.95
OUT = Path(__file__).resolve().parent / "spread.json"


class Spread:
  def __init__(self):
    self.m = mujoco.MjModel.from_xml_path(str(robot.XML))
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.qa = lambda n: m.jnt_qposadr[m.joint(n).id]
    self.rng = {m.joint(i).name: m.jnt_range[i].copy() for i in range(m.njnt)}
    self.foot = {s: m.body(f"{s}_foot_link").id for s in "LR"}
    self.fgeoms = {s: [m.geom(f"{s}_foot{i}_collision").id for i in (1, 2, 3)] for s in "LR"}
    self.warm = {}

  def soft(self, n):
    lo, hi = self.rng[n]
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo) * SOFT
    return c - r, c + r

  def loop(self, S):
    free = [f"{S}_{n}" for n in ("rod_A_u1", "rod_A_u2", "rod_B_u1", "rod_B_u2", "crank_A_joint", "crank_B_joint")]
    idx = [self.qa(n) for n in free]
    sid = lambda n: self.m.site(n).id  # noqa: E731

    def f(x):
      self.d.qpos[idx] = x
      mujoco.mj_kinematics(self.m, self.d)
      p = self.d.site_xpos
      return np.concatenate([p[sid(f"{S}_rod_A_end")] - p[sid(f"{S}_ball_A")], p[sid(f"{S}_rod_B_end")] - p[sid(f"{S}_ball_B")]])

    x0 = np.zeros(6)
    r = least_squares(f, x0, bounds=(-1.6, 1.6), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    self.warm[("loop", S)] = r.x.copy()
    return float(np.abs(f(r.x)).max())

  def level_foot(self, S):
    """hip yaw, ankle pitch, ankle roll so that the foot frame == world frame (orientation)."""
    names = [f"{S}_hip_yaw_joint", f"{S}_ankle_pitch_joint", f"{S}_ankle_roll_joint"]
    idx = [self.qa(n) for n in names]
    b = self.foot[S]

    def f(x):
      self.d.qpos[idx] = x
      mujoco.mj_kinematics(self.m, self.d)
      R = self.d.xmat[b].reshape(3, 3)
      return np.array([R[0, 2], R[1, 2], R[1, 0]])  # z-axis vertical, x-axis has no y

    lo = [self.rng[n][0] for n in names]
    hi = [self.rng[n][1] for n in names]
    x0 = np.clip(self.guess[S], np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    r = least_squares(f, x0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    self.warm[("lvl", S)] = r.x.copy()
    return float(np.abs(f(r.x)).max())

  def set_pose(self, phi, hp, k):
    d = self.d
    d.qpos[:] = self.m.qpos0
    d.qpos[:3] = 0
    d.qpos[3:7] = (1, 0, 0, 0)
    for S, sg in (("R", 1.0), ("L", -1.0)):
      d.qpos[self.qa(f"{S}_hip_pitch_joint")] = sg * hp
      d.qpos[self.qa(f"{S}_knee_joint")] = sg * k
      d.qpos[self.qa(f"{S}_hip_roll_joint")] = sg * phi
      self.guess = {"R": np.array([0.0, hp + k, -phi]), "L": np.array([0.0, hp + k, phi])}
      self.level_foot(S)
    err = max(self.loop("R"), self.loop("L"))
    mujoco.mj_kinematics(self.m, d)
    d.qpos[2] = -self.zmin()
    mujoco.mj_forward(self.m, d)
    return err

  def zmin(self):
    out = []
    for S in "LR":
      for g in self.fgeoms[S]:
        c, R, h = self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3), self.m.geom_size[g]
        for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
          out.append((c + R @ (s * h))[2])
    return float(min(out))

  def sole_centre(self):
    xs, ys = [], []
    for S in "LR":
      for g in self.fgeoms[S]:
        c, h = self.d.geom_xpos[g], self.m.geom_size[g]
        xs += [c[0] - h[0], c[0] + h[0]]
        ys.append(c[1])
    return 0.5 * (min(xs) + max(xs))

  def com_dx(self, phi, hp, k):
    self.set_pose(phi, hp, k)
    return self.d.subtree_com[self.m.body("base_link").id][0] - self.sole_centre()

  def balanced(self, phi, k):
    """All hip-pitch roots of CoM x == sole centre x (there can be two: shin-forward squat and
    hip-hinged 'sit back'). Returns the list, largest hp (shin-forward branch) first."""
    lo, hi = self.soft("R_hip_pitch_joint")
    xs = np.linspace(lo, hi, 40)
    fs = [self.com_dx(phi, x, k) for x in xs]
    roots = []
    for i in range(len(xs) - 1):
      if np.sign(fs[i]) != np.sign(fs[i + 1]):
        try:
          roots.append(brentq(lambda x: self.com_dx(phi, x, k), xs[i], xs[i + 1], xtol=1e-9))
        except ValueError:
          pass
    return sorted(roots, reverse=True)

  def q(self, n):
    return float(self.d.qpos[self.qa(n)])

  def level_error(self):
    return max(abs(self.d.xmat[self.foot[S]].reshape(3, 3)[i, j]) for S in "LR" for i, j in ((0, 2), (1, 2), (1, 0)))

  def report(self, phi, k, hp):
    q = self.q
    h = float(self.d.qpos[2])
    fy = [float(self.d.xpos[self.foot[S]][1]) for S in "LR"]
    row = dict(phi=phi, knee=k, hip_pitch=hp, h=h, ankle_pitch=q("R_ankle_pitch_joint"), ankle_roll=q("R_ankle_roll_joint"),
               hip_yaw=q("R_hip_yaw_joint"), crank_A=q("R_crank_A_joint"), crank_B=q("R_crank_B_joint"),
               crank_AL=q("L_crank_A_joint"), crank_BL=q("L_crank_B_joint"), foot_width=abs(fy[0] - fy[1]))
    viol = []
    for n in ("R_ankle_pitch_joint", "R_ankle_roll_joint", "R_hip_yaw_joint", "R_knee_joint"):
      lo, hi = self.soft(n)
      if not lo <= q(n) <= hi:
        viol.append(n.split("_", 1)[1].replace("_joint", ""))
    if max(abs(q(f"{S}_crank_{c}_joint")) for S in "LR" for c in "AB") > CRANK_CLIP:
      viol.append("crank")
    if self.level_error() > 1e-6:
      viol.append("not_level")
    row["violations"] = viol
    return row


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", type=Path, default=OUT)
  out = ap.parse_args().out
  s = Spread()
  print("soft ankle pitch", s.soft("R_ankle_pitch_joint"), "ankle roll", s.soft("R_ankle_roll_joint"),
        "hip yaw", s.soft("R_hip_yaw_joint"), "knee", s.soft("R_knee_joint"), "hip roll", s.soft("R_hip_roll_joint"))
  s.set_pose(0.1, 0.0, 0.3)
  print("sign check: phi=+0.1 -> foot width", abs(s.d.xpos[s.foot["R"]][1] - s.d.xpos[s.foot["L"]][1]))
  s.set_pose(0.0, 0.0, 0.3)
  print("             phi= 0   -> foot width", abs(s.d.xpos[s.foot["R"]][1] - s.d.xpos[s.foot["L"]][1]))
  rows = []
  for phi in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
    s.warm.clear()
    for k in np.arange(0.35, 1.99, 0.05):
      for b, hp in enumerate(s.balanced(phi, float(k))):
        s.set_pose(phi, hp, float(k))
        r = s.report(phi, float(k), hp)
        r["branch"] = "shin-forward" if b == 0 else "sit-back"
        rows.append(r)
    for br in ("shin-forward", "sit-back"):
      ok = [r for r in rows if r["phi"] == phi and r["branch"] == br and not r["violations"]]
      if not ok:
        print(f"phi {phi:.2f} {br:12s}: none feasible")
        continue
      b = min(ok, key=lambda r: r["h"])
      print(f"phi {phi:.2f} ({np.degrees(phi):4.1f}°) {br:12s}: min h {b['h']:.3f}  knee {b['knee']:.2f} hip {b['hip_pitch']:+.2f} "
            f"ankle p {b['ankle_pitch']:.3f} r {b['ankle_roll']:+.3f} yaw {b['hip_yaw']:+.3f} "
            f"crank R {b['crank_A']:+.2f}/{b['crank_B']:+.2f} width {b['foot_width']:.3f}")
  out.write_text(json.dumps(rows, indent=1))
  print("wrote", out)


if __name__ == "__main__":
  main()
