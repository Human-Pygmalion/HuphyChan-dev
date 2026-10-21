"""Termination terms (beyond stock time_out / bad_orientation / illegal_contact / nan_detection)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def physics_blowup(env: ManagerBasedRlEnv, joint_speed_limit: float = 100.0, base_speed_limit: float = 10.0,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """End an episode whose physics left the physical range (any joint faster than
  `joint_speed_limit` rad/s, base faster than `base_speed_limit` m/s, or non-finite state)."""
  a = env.scene[asset_cfg.name]
  qv, bv = a.data.joint_vel, a.data.root_link_lin_vel_w
  bad = qv.abs().amax(dim=1) > joint_speed_limit
  bad |= bv.norm(dim=1) > base_speed_limit
  bad |= ~torch.isfinite(qv).all(dim=1) | ~torch.isfinite(bv).all(dim=1)
  return bad
