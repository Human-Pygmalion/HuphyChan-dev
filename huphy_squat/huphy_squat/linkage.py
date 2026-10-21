"""Closed-loop ankle kinematics on CPU MuJoCo, used to build `assets/linkage_table.npz`.

Each ankle is a 2-RSU linkage: two crank motors on the shin drive two rods whose lower ends
are tied to the foot by `connect` equalities. Given the two crank angles, the six passive
joints of that leg (rod_A u1/u2, rod_B u1/u2, ankle pitch, ankle roll) follow. Writing crank
angles without the matching passive angles breaks the loop (the equality then drags the
foot back during the first physics steps).

The training reset needs the passive angles for randomized crank angles on the GPU, so this
module tabulates them on a crank grid (bilinear lookup at reset), plus a 2nd-order model of the
height of the foot-sole corners, used to put the lowest sole corner on the floor at reset.

Only `numpy` + `mujoco` are needed. The table is rebuilt by `scripts/build_linkage_table.py`
and checked by `tests/test_linkage.py`.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import robot

PASSIVE = ("rod_A_u1", "rod_A_u2", "rod_B_u1", "rod_B_u2", "ankle_pitch_joint", "ankle_roll_joint")
TABLE_HALF_WIDTH = 0.08  # rad around the default crank angle (reset noise is +-0.05)
TABLE_POINTS = 17  # 0.01 rad spacing
CORNER_SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # bottom face of each foot box


def passive_names(side: str) -> list[str]:
  return [f"{side}_{n}" for n in PASSIVE]


class LoopSolver:
  """Newton solve of one leg's passive joints for given crank angles (base fixed at the origin)."""

  def __init__(self, m: mujoco.MjModel | None = None):
    self.m = m if m is not None else mujoco.MjModel.from_xml_path(str(robot.XML))
    self.d = mujoco.MjData(self.m)
    self.d.qpos[:] = self.m.qpos0
    self.qadr = {self.m.joint(i).name: self.m.jnt_qposadr[i] for i in range(self.m.njnt)}
    self.vadr = {self.m.joint(i).name: self.m.jnt_dofadr[i] for i in range(self.m.njnt)}

  def set(self, joints: dict[str, float]) -> None:
    for n, v in joints.items():
      self.d.qpos[self.qadr[n]] = v

  def residual(self, side: str) -> np.ndarray:
    mujoco.mj_kinematics(self.m, self.d)
    s = lambda n: self.d.site_xpos[self.m.site(n).id]  # noqa: E731
    return np.concatenate([s(f"{side}_rod_A_end") - s(f"{side}_ball_A"), s(f"{side}_rod_B_end") - s(f"{side}_ball_B")])

  def solve(self, side: str, x0: np.ndarray | None = None, tol: float = 1e-12, iters: int = 50) -> float:
    """Solve the 6 passive joints of `side` in place (cranks and the rest held). Returns max |residual| [m]."""
    names = passive_names(side)
    qa = [self.qadr[n] for n in names]
    va = [self.vadr[n] for n in names]
    if x0 is not None:
      self.d.qpos[qa] = x0
    sites = [(f"{side}_rod_A_end", f"{side}_ball_A"), (f"{side}_rod_B_end", f"{side}_ball_B")]
    nv = self.m.nv
    for _ in range(iters):
      r = self.residual(side)
      if np.abs(r).max() < tol:
        break
      mujoco.mj_comPos(self.m, self.d)
      J = np.zeros((6, 6))
      for k, (s1, s2) in enumerate(sites):
        j1 = np.zeros((3, nv))
        j2 = np.zeros((3, nv))
        mujoco.mj_jacSite(self.m, self.d, j1, None, self.m.site(s1).id)
        mujoco.mj_jacSite(self.m, self.d, j2, None, self.m.site(s2).id)
        J[3 * k : 3 * k + 3] = (j1 - j2)[:, va]
      dx = np.linalg.lstsq(J, -r, rcond=None)[0]
      self.d.qpos[qa] += dx
    return float(np.abs(self.residual(side)).max())

  def passive(self, side: str) -> np.ndarray:
    return np.array([self.d.qpos[self.qadr[n]] for n in passive_names(side)])


def default_pose(solver: LoopSolver | None = None) -> dict[str, float]:
  """Motor default angles (robot.DEFAULT_MOTOR_POS) + the passive angles that close both loops."""
  s = solver or LoopSolver()
  s.d.qpos[:] = s.m.qpos0
  s.set(robot.DEFAULT_MOTOR_POS)
  pose = dict(robot.DEFAULT_MOTOR_POS)
  for side in "LR":
    err = s.solve(side)
    assert err < 1e-9, (side, err)
    pose.update(zip(passive_names(side), s.passive(side).tolist(), strict=True))
  return pose


def foot_corners_z(s: LoopSolver) -> np.ndarray:
  """World z of the 4 bottom corners of every foot box (base at origin, upright). Shape (24,)."""
  mujoco.mj_kinematics(s.m, s.d)
  out = []
  for side in "LR":
    for i in (1, 2, 3):
      g = s.m.geom(f"{side}_foot{i}_collision").id
      c, R, h = s.d.geom_xpos[g], s.d.geom_xmat[g].reshape(3, 3), s.m.geom_size[g]
      for sx, sy in CORNER_SIGNS:
        out.append((c + R @ np.array([sx * h[0], sy * h[1], -h[2]]))[2])
  return np.array(out)


def build_table() -> dict[str, np.ndarray]:
  s = LoopSolver()
  pose = default_pose(s)
  tab: dict[str, np.ndarray] = {}
  grid = np.linspace(-TABLE_HALF_WIDTH, TABLE_HALF_WIDTH, TABLE_POINTS)
  max_err = 0.0
  for side in "LR":
    ca0, cb0 = pose[f"{side}_crank_A_joint"], pose[f"{side}_crank_B_joint"]
    vals = np.zeros((TABLE_POINTS, TABLE_POINTS, 6))
    for i, da in enumerate(grid):
      for j, db in enumerate(grid):
        s.set(pose)
        s.set({f"{side}_crank_A_joint": ca0 + da, f"{side}_crank_B_joint": cb0 + db})
        max_err = max(max_err, s.solve(side))
        vals[i, j] = s.passive(side)
    tab[f"{side}_grid_a"] = ca0 + grid
    tab[f"{side}_grid_b"] = cb0 + grid
    tab[f"{side}_passive"] = vals
  assert max_err < 1e-9, max_err

  # Sole-corner height vs the 12 motor angles (passive joints re-solved), 2nd order at default:
  # z(dq) ~= z0 + J dq + 0.5 dq^T H dq   (central differences).
  s.set(pose)
  z0 = foot_corners_z(s)
  motors = list(robot.MOTOR_JOINT_NAMES)
  n = len(motors)

  def corners(offsets: dict[int, float]) -> np.ndarray:
    s.set(pose)
    s.set({motors[k]: pose[motors[k]] + v for k, v in offsets.items()})
    for side in "LR":
      s.solve(side)
    return foot_corners_z(s)

  eps = 1e-3
  J = np.zeros((z0.size, n))
  H = np.zeros((z0.size, n, n))
  for i in range(n):
    zp, zm = corners({i: eps}), corners({i: -eps})
    J[:, i] = (zp - zm) / (2 * eps)
    H[:, i, i] = (zp - 2 * z0 + zm) / eps**2
    for j in range(i):
      h = (corners({i: eps, j: eps}) - corners({i: eps, j: -eps}) - corners({i: -eps, j: eps})
           + corners({i: -eps, j: -eps})) / (4 * eps**2)
      H[:, i, j] = H[:, j, i] = h
  tab["motor_names"] = np.array(motors)
  tab["corner_z0"] = z0
  tab["corner_dz_dq"] = J
  tab["corner_d2z_dq2"] = H
  tab["default_names"] = np.array(list(pose.keys()))
  tab["default_values"] = np.array(list(pose.values()))
  return tab


def load_table(path: Path | None = None) -> dict[str, np.ndarray]:
  with np.load(path or robot.LINKAGE_TABLE) as f:
    return {k: f[k] for k in f.files}
