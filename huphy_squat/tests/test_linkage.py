"""CPU MuJoCo checks of the copied MJCF and the closed-loop linkage table (no GPU needed)."""

from __future__ import annotations

import itertools

import mujoco
import numpy as np
import pytest

from huphy_squat import linkage, robot


@pytest.fixture(scope="module")
def solver():
  return linkage.LoopSolver()


@pytest.fixture(scope="module")
def table():
  return linkage.load_table()


def test_mjcf_names_and_mass(solver):
  m = solver.m
  assert abs(sum(m.body_mass) - 23.63014) < 1e-3  # rod capsules add no mass (explicit inertials)
  # foot sites sit on the foot of the same side
  for s in "LR":
    assert m.body(m.site_bodyid[m.site(f"{s}_foot_site").id]).name == f"{s}_foot_link"
  # *_ankle_a_motor sits at the crank A pivot, *_ankle_b_motor at the crank B pivot
  for s in "LR":
    for x in "ab":
      g, b = m.geom(f"{s}_ankle_{x}_motor"), m.body(f"{s}_crank_{x.upper()}")
      assert np.allclose(m.geom_pos[g.id], m.body_pos[b.id], atol=2e-4)
  assert len([m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.endswith("_rod_A_collision")]) == 2


def test_default_pose_matches_previous_task(table):
  # Same passive angles as mjlab_policy/pygmalion_mjlab/robot.py INIT_STATE (solved there independently).
  prev = {"L_rod_A_u1": 0.1479036260054202, "L_ankle_pitch_joint": 0.1750072557823706,
          "R_rod_B_u1": 0.14892320455004768, "R_ankle_pitch_joint": 0.17516470240179116,
          "R_ankle_roll_joint": -7.751934743886312e-05}
  pose = dict(zip(table["default_names"], table["default_values"], strict=True))
  for k, v in prev.items():
    assert abs(pose[k] - v) < 1e-6, k


def _interp(table, side, ca, cb):
  ga, gb, vals = table[f"{side}_grid_a"], table[f"{side}_grid_b"], table[f"{side}_passive"]
  fa = np.interp(ca, ga, np.arange(len(ga)))
  fb = np.interp(cb, gb, np.arange(len(gb)))
  i, j = min(int(fa), len(ga) - 2), min(int(fb), len(gb) - 2)
  u, v = fa - i, fb - j
  return ((1 - u) * (1 - v) * vals[i, j] + u * (1 - v) * vals[i + 1, j]
          + (1 - u) * v * vals[i, j + 1] + u * v * vals[i + 1, j + 1])


def test_table_interpolation_closes_loop(solver, table):
  """Bilinear table lookup at random crank offsets inside the reset noise leaves < 0.1 mm loop error."""
  rng = np.random.default_rng(0)
  pose = dict(zip(table["default_names"], table["default_values"], strict=True))
  worst = 0.0
  for _ in range(200):
    solver.set(pose)
    for side in "LR":
      ca = pose[f"{side}_crank_A_joint"] + rng.uniform(-0.05, 0.05)
      cb = pose[f"{side}_crank_B_joint"] + rng.uniform(-0.05, 0.05)
      solver.set({f"{side}_crank_A_joint": ca, f"{side}_crank_B_joint": cb})
      solver.set(dict(zip(linkage.passive_names(side), _interp(table, side, ca, cb), strict=True)))
      worst = max(worst, float(np.abs(solver.residual(side)).max()))
  assert worst < 1e-4, worst


def test_corner_linearization(solver, table):
  """2nd-order lowest sole corner vs exact, for random +-0.05 rad motor offsets: error < 0.5 mm."""
  rng = np.random.default_rng(1)
  pose = dict(zip(table["default_names"], table["default_values"], strict=True))
  names = [str(n) for n in table["motor_names"]]
  errs = []
  for _ in range(100):
    dq = rng.uniform(-0.05, 0.05, len(names))
    solver.set(pose)
    solver.set({n: pose[n] + d for n, d in zip(names, dq, strict=True)})
    for side in "LR":
      solver.solve(side)
    exact = linkage.foot_corners_z(solver).min()
    approx = (table["corner_z0"] + table["corner_dz_dq"] @ dq
              + 0.5 * np.einsum("cij,i,j->c", table["corner_d2z_dq2"], dq, dq)).min()
    errs.append(abs(exact - approx))
  assert max(errs) < 5e-4, max(errs)


def _contacts_with(m, d, names):
  ids = {m.geom(n).id for n in names}
  return [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[: d.ncon] if c.geom1 in ids or c.geom2 in ids]


def test_rod_capsules_free_over_ankle_rom(solver):
  """No rod-capsule contact (self or other leg) over a crank x knee grid, robot in the air."""
  m, d = solver.m, solver.d
  # emulate the task collision setup: only *_collision geoms collide
  m2 = mujoco.MjModel.from_xml_path(str(robot.XML))
  for i in range(m2.ngeom):
    if not m2.geom(i).name.endswith("_collision"):
      m2.geom_contype[i] = m2.geom_conaffinity[i] = 0
  s2 = linkage.LoopSolver(m2)
  pose = linkage.default_pose(s2)
  rods = [f"{s}_rod_{x}_collision" for s in "LR" for x in "AB"]
  bad = []
  for knee, ca, cb in itertools.product((0.1, 0.8, 1.5, 2.0), np.linspace(-0.9, 0.9, 7), np.linspace(-0.9, 0.9, 7)):
    s2.set(pose)
    for side, sg in (("R", 1), ("L", -1)):
      s2.set({f"{side}_knee_joint": sg * knee, f"{side}_crank_A_joint": sg * ca, f"{side}_crank_B_joint": sg * cb})
      if s2.solve(side) > 1e-6:
        break  # crank pair outside the reachable set
    else:
      mujoco.mj_forward(m2, s2.d)
      c = _contacts_with(m2, s2.d, rods)
      if c:
        bad.append((knee, ca, cb, c))
  assert not bad, bad[:5]
  del m, d
