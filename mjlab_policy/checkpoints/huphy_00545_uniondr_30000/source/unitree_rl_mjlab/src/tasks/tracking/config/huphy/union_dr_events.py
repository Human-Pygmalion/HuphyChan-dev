"""Event helpers used by the HUPHY Union-DR tracking task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  return env_ids


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def add_payload_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Add an episode-fixed payload at the selected body's nominal CoM."""
  env_ids = _resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids].long()
  offsets = torch.empty(
    len(env_ids), len(body_ids), device=env.device
  ).uniform_(*mass_range)
  env.sim.model.body_mass[env_ids[:, None], body_ids] += offsets


def add_hip_roll_reset_spread(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  spread_range: tuple[float, float],
  spread_prob: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Apply the Union-DR independent outward hip-roll reset perturbation."""
  env_ids = _resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  joint_pos = asset.data.joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel = asset.data.joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  spread = torch.empty(len(env_ids), 2, device=env.device).uniform_(*spread_range)
  spread *= torch.rand(len(env_ids), 1, device=env.device) < spread_prob
  spread[:, 1] *= -1.0
  joint_pos += spread
  limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos.clamp_(limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(
    joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids
  )


class apply_horizontal_body_push:
  """Apply fixed-duration horizontal forces at randomized intervals."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._cooldown_s = cfg.params["cooldown_s"]
    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._cooldown_remaining = self._sample_uniform(
      self._cooldown_s, self._num_envs
    )

  def _sample_uniform(
    self, value_range: tuple[float, float], count: int
  ) -> torch.Tensor:
    low, high = value_range
    return torch.rand(count, device=self._device) * (high - low) + low

  def _write_force(self, env_ids: torch.Tensor, forces: torch.Tensor) -> None:
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(
      forces[:, None, :],
      torques[:, None, :],
      env_ids=env_ids,
      body_ids=self._body_ids,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    magnitude_range: tuple[float, float],
    duration_s: float | tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    del env, env_ids, cooldown_s, asset_cfg
    self._time_remaining[self._active] -= self._step_dt
    expired = self._active & (self._time_remaining <= 0.0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      self._write_force(
        expired_ids, torch.zeros((len(expired_ids), 3), device=self._device)
      )
      self._active[expired_ids] = False
      self._cooldown_remaining[expired_ids] = self._sample_uniform(
        self._cooldown_s, len(expired_ids)
      )

    inactive = ~self._active
    self._cooldown_remaining[inactive] -= self._step_dt
    triggered = inactive & (self._cooldown_remaining <= 0.0)
    if not triggered.any():
      return

    trigger_ids = triggered.nonzero(as_tuple=False).squeeze(-1)
    count = len(trigger_ids)
    magnitude = self._sample_uniform(magnitude_range, count)
    angle = self._sample_uniform((0.0, 2.0 * math.pi), count)
    forces = torch.stack(
      (
        magnitude * torch.cos(angle),
        magnitude * torch.sin(angle),
        torch.zeros_like(magnitude),
      ),
      dim=-1,
    )
    self._write_force(trigger_ids, forces)
    self._active[trigger_ids] = True
    if isinstance(duration_s, tuple):
      self._time_remaining[trigger_ids] = self._sample_uniform(duration_s, count)
    else:
      self._time_remaining[trigger_ids] = duration_s

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = (
      torch.arange(self._num_envs, device=self._device)
      if env_ids is None
      else env_ids
    )
    self._write_force(
      ids, torch.zeros((len(ids), 3), device=self._device)
    )
    self._active[ids] = False
    self._time_remaining[ids] = 0.0
    self._cooldown_remaining[ids] = self._sample_uniform(
      self._cooldown_s, len(ids)
    )
