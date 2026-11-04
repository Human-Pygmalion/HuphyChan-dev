#!/usr/bin/env python3
"""Weigh the real robot with its own leg motors (static equilibrium, no pelvis model needed).

For each still window of a HUPHY policy log the motor torque is tau = Kp * (tgt - pos) (Kd term is zero when still).
Per leg, virtual work on the six motor coordinates u gives   tau + J^T W + G = 0
  J  = d[foot origin position; foot rotation vector] / du   (6x6, the closed ankle loop is solved by stance_fk.FK.pose)
  G  = d(sum_i m_i g.r_i)/du                                (gravity of the leg links only, from the model's masses)
  W  = wrench the ground puts on the foot (force + moment about the foot_link origin)
so  W = -J^-T (tau + G).  The two feet carry the whole robot:  sum of the vertical forces / g  =  supported mass.
The pelvis, camera and everything above the hip never enter the equation except through the ground reaction -> the result is
independent of the pelvis mass in the model. The legs' own masses enter G (model dependent), so every window is evaluated with
the training model AND the latest model.

  mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/m19_transfer/static_weigh.py --xml-a <M10.xml> --xml-b <M19.xml> \
      --log label=path.json[,kp_scale] ... --out out.json

Caveats written into the output: Kp is the nominal one (135/135/75/130/60/60 N m/rad) -- the real torque scale was never checked with
a weight test (docs/172); friction in the motor (RS04 0.52 N m, RS03 0.25 N m) is not removed; gravity vector = IMU 'obs/grav'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim2real" / "walk"))
import stance_fk as sf  # noqa: E402

KP = np.array([135, 135, 75, 130, 60, 60] * 2, float)  # N m/rad, order of sf.COLS (right leg then left leg)
G0 = 9.81


class LegStatics:
  def __init__(self, xml):
    self.fk = sf.FK(xml)
    m = self.fk.m
    self.m = m
    self.bodies = {s: [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{s}_")] for s in "RL"}
    self.mass = {s: m.body_mass[self.bodies[s]] for s in "RL"}
    self.foot = {s: self.fk.foot_body[s] for s in "RL"}
    self.leg_mass = {s: float(self.mass[s].sum()) for s in "RL"}

  def eval(self, q12, s, gvec):
    """-> (sum m g.r, foot position, foot rotation matrix) for leg s at motor angles q12 (rad)."""
    self.fk.pose(q12)
    d = self.fk.d
    com = d.xipos[self.bodies[s]]
    gw = float(np.sum(self.mass[s][:, None] * com @ gvec))
    return gw, d.xpos[self.foot[s]].copy(), d.xmat[self.foot[s]].reshape(3, 3).copy()

  def wrench(self, q12, tau6, s, gvec, h=2e-5):
    """Ground wrench on the foot (pelvis frame, about the foot_link origin) of leg s: returns (force 3, moment 3, residual)."""
    sl = slice(0, 6) if s == "R" else slice(6, 12)
    q0 = np.array(q12, float)
    gw0, p0, R0 = self.eval(q0, s, gvec)
    J = np.zeros((6, 6))
    G = np.zeros(6)
    for k in range(6):
      qp, qm = q0.copy(), q0.copy()
      qp[sl.start + k] += h
      qm[sl.start + k] -= h
      gp, pp, Rp = self.eval(qp, s, gvec)
      gm, pm, Rm = self.eval(qm, s, gvec)
      G[k] = (gp - gm) / (2 * h)
      J[:3, k] = (pp - pm) / (2 * h)
      dR = (Rp - Rm) @ R0.T / (2 * h)  # skew(omega), world frame
      J[3:, k] = [dR[2, 1], dR[0, 2], dR[1, 0]]
    self.fk.pose(q0)
    W = -np.linalg.solve(J.T, tau6 + G)
    return W[:3], W[3:], float(self.fk.resid), J


def still_windows(a, c, win=100, tol_deg=0.6, tgt_tol_deg=0.6):
  """Indices (start, end) of windows where every measured and target joint angle moves less than tol over `win` rows."""
  pos = a[:, [c.index(f"pos/{n}") for n in sf.COLS]]
  tgt = a[:, [c.index(f"tgt/{n}") for n in sf.COLS]]
  out = []
  i = 0
  while i + win <= len(a):
    ok = np.ptp(pos[i:i + win], 0).max() < tol_deg and np.ptp(tgt[i:i + win], 0).max() < tgt_tol_deg
    if ok:
      j = i + win
      while j < len(a) and np.ptp(pos[i:j + 1], 0).max() < tol_deg and np.ptp(tgt[i:j + 1], 0).max() < tgt_tol_deg:
        j += 1
      out.append((i, j))
      i = j
    else:
      i += 10
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml-a", required=True, help="training model (M10)")
  ap.add_argument("--xml-b", required=True, help="latest model (M19)")
  ap.add_argument("--log", action="append", required=True, help="label=path[,kp_scale]")
  ap.add_argument("--out", required=True)
  ap.add_argument("--min-sec", type=float, default=2.0)
  args = ap.parse_args()
  models = {"A": LegStatics(args.xml_a), "B": LegStatics(args.xml_b)}
  res = {"models": {k: {"leg_mass_R_kg": v.leg_mass["R"], "leg_mass_L_kg": v.leg_mass["L"], "total_kg": float(v.m.body_mass.sum())} for k, v in models.items()}, "logs": {}}
  for spec in args.log:
    label, rest = spec.split("=", 1)
    path, _, kps = rest.partition(",")
    kps = float(kps) if kps else 1.0
    dct = json.load(open(path))
    c, a = dct["columns"], np.array(dct["rows"], float)
    t = a[:, c.index("t")]
    dt = float(np.median(np.diff(t)))
    win = int(round(args.min_sec / dt))
    wins = still_windows(a, c, win)
    rows = []
    for (i, j) in wins:
      pos = np.radians(a[i:j, [c.index(f"pos/{n}") for n in sf.COLS]].mean(0))
      tgt = np.radians(a[i:j, [c.index(f"tgt/{n}") for n in sf.COLS]].mean(0))
      grav = a[i:j, [c.index(f"obs/grav/{k}") for k in "xyz"]].mean(0)
      grav = grav / np.linalg.norm(grav)
      gvec = G0 * grav  # pelvis-frame gravity acceleration (points down)
      tau = KP * kps * (tgt - pos)
      r = dict(t0=float(t[i]), t1=float(t[j - 1]), cmd_vx=float(a[i:j, c.index("obs/cmd/vx")].mean()) if "obs/cmd/vx" in c else None, grav=[round(float(x), 4) for x in grav],
               tau_Nm={n: round(float(v), 2) for n, v in zip(sf.COLS, tau)})
      for key, mdl in models.items():
        Fz = {}
        for s, nm in (("R", "right"), ("L", "left")):
          F, M, resid, _ = mdl.wrench(pos, tau[0:6] if s == "R" else tau[6:12], s, gvec)
          up = -grav  # unit vector 'up' in the pelvis frame
          Fn = float(F @ up)
          Ft = F - Fn * up
          Fz[nm] = dict(F_normal_N=round(Fn, 1), F_tangent_N=[round(float(x), 1) for x in Ft], mu_needed=round(float(np.linalg.norm(Ft) / Fn), 3) if Fn > 1 else None,
                        M_Nm=[round(float(x), 1) for x in M], loop_residual=resid)
        tot = Fz["right"]["F_normal_N"] + Fz["left"]["F_normal_N"]
        r[key] = dict(feet=Fz, supported_mass_kg=round(tot / G0, 2), share_right=round(Fz["right"]["F_normal_N"] / tot, 3))
      rows.append(r)
    res["logs"][label] = dict(path=path, kp_scale=kps, n_windows=len(rows), windows=rows)
    sm = np.array([[w["A"]["supported_mass_kg"], w["B"]["supported_mass_kg"]] for w in rows]) if rows else np.zeros((0, 2))
    print(f"{label}: {len(rows)} still windows >= {args.min_sec}s; supported mass (A=M10 legs, B=M19 legs): "
          + (f"A {sm[:, 0].mean():.2f}+-{sm[:, 0].std():.2f}  B {sm[:, 1].mean():.2f}+-{sm[:, 1].std():.2f} kg" if len(rows) else "-"), flush=True)
  Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
