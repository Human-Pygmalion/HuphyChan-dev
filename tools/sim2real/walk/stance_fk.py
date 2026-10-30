#!/usr/bin/env python3
"""Stance width of the Pygmalion LegOnly-AB model by forward kinematics, from a real-layout walking CSV (real or simulator export).

For every logged row: the 12 motor angles (pos or tgt, degrees) go into the MJCF; the passive ankle pitch/roll hinges and the
rod joints are SOLVED from the crank angles by closing the four `connect` loops (Gauss-Newton, same method as
render_s2r_videos.py). The base is the pelvis frame (identity pose): x forward, y LEFT, z up.

Per row it returns, in the pelvis frame [m]: lateral position of both foot centres, of both inner foot edges (box corners of the
foot collision geoms, nearest to the midline), fore-aft position, and the foot height along the gravity-up direction.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/stance_fk.py --xml <...>/pygmalion_legonly_ab.xml --csv a.csv [--what pos|tgt]
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import mujoco
import numpy as np

MOTORS = [f"{s}_{j}_joint" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
PASSIVE = [f"{s}_ankle_{j}_joint" for s in "RL" for j in ("pitch", "roll")]
JT = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"]
COLS = [f"{s}_leg/{j}" for s in ("right", "left") for j in JT]


def load_csv(path):
  """Real-layout CSV -> (columns, array); the first line starting with # is the limits header (skipped)."""
  lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
  cols = lines[0].split(",")
  a = np.genfromtxt(io.StringIO("\n".join(lines[1:])), delimiter=",")
  return cols, a


class FK:
  def __init__(self, xml):
    spec = mujoco.MjSpec.from_file(str(xml))
    self.m = spec.compile()
    m = self.m
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_LIMIT | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    self.d = mujoco.MjData(m)
    jn = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)  # noqa: E731
    names = [jn(j) for j in range(m.njnt)]
    self.adr12 = np.array([m.jnt_qposadr[names.index(n)] for n in MOTORS])
    self.adr_pas = np.array([m.jnt_qposadr[names.index(n)] for n in PASSIVE])
    rod = [m.jnt_dofadr[j] for j, n in enumerate(names) if n and "rod" in n]
    self.free_dofs = np.array(rod + [m.jnt_dofadr[names.index(n)] for n in PASSIVE])
    self.d.qpos[:] = m.qpos0
    self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    self.foot_geoms = {}
    for s in "RL":
      self.foot_geoms[s] = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{s}_foot") and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("collision")]
    self.foot_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot_link") for s in "RL"}
    self.hipbody = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_roll_link") for s in "RL"}
    self.resid = 0.0

  def pose(self, q12):
    d, m = self.d, self.m
    d.qpos[0:3] = 0
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[self.adr12] = q12
    F = self.free_dofs
    for _ in range(40):
      mujoco.mj_forward(m, d)
      r = d.efc_pos.copy()
      if np.abs(r).max() < 1e-9:
        break
      J = d.efc_J.reshape(d.nefc, m.nv)[:, F]
      d.qpos[F + 1] += -np.linalg.solve(J.T @ J + 1e-9 * np.eye(len(F)), J.T @ r)
    mujoco.mj_forward(m, d)
    self.resid = float(np.abs(d.efc_pos).max())

  def feet(self):
    """-> dict side -> (centre xyz, inner-edge y, outer-edge y, sole z min) in the pelvis frame."""
    d, m = self.d, self.m
    out = {}
    for s in "RL":
      pts = []
      for g in self.foot_geoms[s]:
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        h = m.geom_size[g][:3]
        for sx in (-1, 1):
          for sy in (-1, 1):
            for sz in (-1, 1):
              pts.append(c + R @ (h * np.array([sx, sy, sz])))
      pts = np.array(pts)
      cen = 0.5 * (pts.min(0) + pts.max(0))  # AABB centre of all foot boxes (pelvis frame)
      sole = pts[:, 2].min()
      # sole-level points (lowest 3 cm) give the contact patch edges
      pat = pts[pts[:, 2] < sole + 0.03]
      ys = pat[:, 1]
      out[s] = dict(c=cen, y_in=(ys.min() if s == "L" else ys.max()), y_out=(ys.max() if s == "L" else ys.min()), zmin=sole, xs=(pat[:, 0].min(), pat[:, 0].max()))
    return out


def series(fk: FK, deg12: np.ndarray):
  """deg12 (N,12) motor angles in degrees (CSV layout) -> dict of arrays."""
  n = len(deg12)
  o = {k: np.full(n, np.nan) for k in ("yR", "yL", "xR", "xL", "zR", "zL", "inR", "inL", "outR", "outL", "resid")}
  rad = np.radians(deg12)
  for i in range(n):
    fk.pose(rad[i])
    f = fk.feet()
    o["yR"][i], o["yL"][i] = f["R"]["c"][1], f["L"]["c"][1]
    o["xR"][i], o["xL"][i] = f["R"]["c"][0], f["L"]["c"][0]
    o["zR"][i], o["zL"][i] = f["R"]["zmin"], f["L"]["zmin"]
    o["inR"][i], o["inL"][i] = f["R"]["y_in"], f["L"]["y_in"]
    o["outR"][i], o["outL"][i] = f["R"]["y_out"], f["L"]["y_out"]
    o["resid"][i] = fk.resid
  o["width"] = o["yL"] - o["yR"]  # centre-to-centre lateral distance
  o["gap"] = o["inL"] - o["inR"]  # inner edge to inner edge (negative = crossed)
  return o


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True)
  ap.add_argument("--csv", required=True)
  ap.add_argument("--what", default="pos", choices=["pos", "tgt"])
  ap.add_argument("--step", type=int, default=50)
  a = ap.parse_args()
  cols, arr = load_csv(a.csv)
  idx = [cols.index(f"{a.what}/{c}") for c in COLS]
  fk = FK(a.xml)
  s = series(fk, arr[:: a.step][:, idx])
  for k in ("width", "gap", "yR", "yL", "resid"):
    print(k, np.nanmean(s[k]), np.nanmin(s[k]), np.nanmax(s[k]))
