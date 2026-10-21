"""Viewer env cfg: the training cfg + a throwable box, fixed (GUI-set) DR, pushes off.

Policy-facing parts are unchanged: same observations (with the training noise), same
actions, same events. Only the DR draws are pinned (see `dr.py`), the observation-lag
event takes a fixed lag, episodes never time out, and one free box per env is parked
beside the robot until thrown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg

from ..env_cfg import huphy_squat_env_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

BOX = "box"
BOX_PARK = (0.0, 3.0)  # m from the env origin, resting on the floor


def box_spec(edge: float, mass: float) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(edge / 2,) * 3, mass=mass,
                rgba=(0.85, 0.45, 0.1, 1.0), condim=3, friction=(0.6, 0.005, 0.0001))
  return spec


def fixed_obs_lag(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, lag: int = 0) -> None:
  """Viewer replacement of `sample_obs_lag`: the same lag for every env, set from the GUI."""
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device, torch.long)
  cur = getattr(env, "huphy_obs_lag", None)
  if cur is None:
    cur = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.huphy_obs_lag = cur  # type: ignore[attr-defined]
  cur[ids] = lag


def park_box(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, edge: float) -> None:
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device, torch.long)
  root = torch.zeros(len(ids), 13, device=env.device)
  root[:, :3] = env.scene.env_origins[ids]
  root[:, 0] += BOX_PARK[0]
  root[:, 1] += BOX_PARK[1]
  root[:, 2] += edge / 2
  root[:, 3] = 1.0
  env.scene[BOX].write_root_state_to_sim(root, env_ids=ids)


def view_env_cfg(num_envs: int, box_edge: float, box_mass: float) -> ManagerBasedRlEnvCfg:
  cfg = huphy_squat_env_cfg(play=False)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 1e9  # no time-out; falls still end the episode
  cfg.scene.entities[BOX] = EntityCfg(spec_fn=lambda: box_spec(box_edge, box_mass))
  cfg.events["obs_lag"] = EventTermCfg(mode="reset", func=fixed_obs_lag, params={"lag": 0})
  cfg.events["park_box"] = EventTermCfg(mode="reset", func=park_box, params={"edge": box_edge})
  return cfg


def box_body(env: ManagerBasedRlEnv) -> int:
  """Global body id of the box (same in every world)."""
  return int(env.scene[BOX].indexing.body_ids[0])
