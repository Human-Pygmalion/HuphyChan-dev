"""Terrain choice (2026-10-12): on a model baked with the trail (PYG_BAKE_TERRAIN) `set_terrain`
switches between the trail (heightfield + trunks collide) and flat ground (those collisions off, the
floor plane lifted to world z = 0).  Needs the trail bake, so it is skipped on a plain cache - point
PYG_VIEW_CACHE at the trail cache (~/pyg_demo/cache_trail on the 3090)."""

import mujoco
import numpy as np
import pytest

from pygviewer import CACHE_DIR
from pygviewer.contract import load_contract
from pygviewer.sim_core import SimCore

VARIANT = "LegOnly-AB"


@pytest.fixture
def core():
  c = SimCore(load_contract(CACHE_DIR, VARIANT), realtime=False)
  if not c.has_terrain:
    c.stop()
    pytest.skip("model has no trail terrain (no PYG_BAKE_TERRAIN bake in this cache)")
  c.reset("knees_bent")
  yield c
  c.stop()


def _hf(c):
  return c.m.hfield_data[c._hf_slice].copy()


def _ray_z(c, x, y):
  """World z of the first trail_ground hit by a vertical ray (None when it hits something else)."""
  d = mujoco.MjData(c.m)
  mujoco.mj_forward(c.m, d)
  gid = np.zeros(1, dtype=np.int32)
  dist = mujoco.mj_ray(c.m, d, np.array([x, y, 500.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
  if gid[0] != c._hf_gid:
    return None
  return 500.0 - dist


def _contact_geoms(c):
  names = set()
  for i in range(c.d.ncon):
    for g in (c.d.contact.geom1[i], c.d.contact.geom2[i]):
      names.add(mujoco.mj_id2name(c.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "")
  return names


def test_flat_round_trip_restores_collision_and_heights(core):
  h0 = _hf(core)
  ct0 = core.m.geom_contype[core._tree_gids].copy()
  assert (ct0 > 0).all() and h0.std() > 0.01  # rough: trunks collide, the surface is not level
  core.set_terrain("flat")
  assert (core.m.geom_contype[core._tree_gids] == 0).all()
  assert (core.m.geom_conaffinity[core._tree_gids] == 0).all()
  hits = [_ray_z(core, x, y) for x in (-40.0, 0.0, 33.0) for y in (-25.0, 5.0, 31.0)]
  assert any(z is not None for z in hits)
  assert all(abs(z) < 1e-4 for z in hits if z is not None)  # the heightfield surface is world z = 0 everywhere
  core.set_terrain("rough")
  assert (core.m.geom_contype[core._tree_gids] == ct0).all()
  assert np.array_equal(_hf(core), h0)  # byte-for-byte the baked trail again


def _load_standing_policy(c):
  import json
  from pathlib import Path

  f = Path(CACHE_DIR) / "roughmain.policy_contract.json"
  if not f.exists():
    pytest.skip("no baked policy 'roughmain' in this cache")
  pc = json.loads(f.read_text())
  c.load_policy(onnx=pc["onnx"], policy_contract=pc)
  c.mode = "policy_sim"


def test_robot_stands_on_the_chosen_ground(core):
  _load_standing_policy(core)
  core.set_gain_scale(0.5)
  core.set_terrain("flat")
  assert core.gain_scale == 0.5 and core.mode == "policy_sim"  # a switch keeps mode, policy, Action scale
  core.set_gain_scale(1.0)
  core.step_n(600)  # 3 s of the policy at zero command
  names = _contact_geoms(core)
  assert "trail_ground" in names  # the levelled heightfield carries the robot ...
  assert not any(n.startswith("trail_tree") for n in names)  # ... and no trunk is ever touched
  assert core.d.qpos[core.free_adr + 2] > 0.7  # standing on z = 0
  core.set_terrain("rough")
  core.step_n(600)
  assert "trail_ground" in _contact_geoms(core)  # the heightfield carries it again
  assert core.d.qpos[core.free_adr + 2] > 0.7


def test_bad_terrain_refused(core):
  with pytest.raises(ValueError):
    core.set_terrain("ice")
  assert core.terrain == "rough"
