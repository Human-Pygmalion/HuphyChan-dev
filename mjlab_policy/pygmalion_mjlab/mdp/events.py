"""Push disturbances used by the Pygmalion training environments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3_range(
  range_dict: dict[str, tuple[float, float]] | None,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  """Sample uniform ``[x, y, z, roll, pitch, yaw]`` offsets.

  ``range_dict`` maps any subset of those keys to ``(min, max)`` ranges; missing
  keys default to ``(0.0, 0.0)`` (no offset). ``None`` is treated as empty. The
  returned tensor has the requested ``shape`` whose last dimension must be 6.
  """
  range_dict = range_dict or {}
  range_list = [range_dict.get(key, (0.0, 0.0)) for key in _SE3_KEYS]
  ranges = torch.tensor(range_list, device=device)
  return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
  """Return ``env_ids`` unchanged, or all environment indices if ``None``.

  Event functions receive ``env_ids=None`` to mean "all environments" (a full
  reset, or a global-time interval term). This normalizes that sentinel to a
  concrete index tensor so the function body can assume a real ``torch.Tensor``.
  """
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  return env_ids


def _push_buffers(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """Lazily allocate (last push delta [B,6], seconds since that push [B,1])."""
  last = getattr(env, "pyg_last_push", None)
  if last is None:
    last = torch.zeros(env.num_envs, 6, device=env.device)
    env.pyg_last_push = last  # type: ignore[attr-defined]
  age = getattr(env, "pyg_push_age", None)
  if age is None:
    age = torch.zeros(env.num_envs, 1, device=env.device)
    env.pyg_push_age = age  # type: ignore[attr-defined]
  return last, age


def push_by_setting_velocity_logged(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """:func:`push_by_setting_velocity` that also RECORDS the kick it applied.

  The base function samples a root-velocity delta, adds it, and drops the sample
  on the floor - so nothing downstream can distinguish a just-pushed env from a
  quiet one. This variant is dynamically identical (same sampler, same number of
  RNG draws, same write) but stores the delta in ``env.pyg_last_push`` and resets
  ``env.pyg_push_age``, which is what the critic-side ``dr_push_state``
  observation reads (docs/103 V9).
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  delta = _sample_se3_range(velocity_range, vel_w.shape, env.device)
  vel_w = vel_w + delta
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)

  last, age = _push_buffers(env)
  last[env_ids] = delta.to(last.dtype)
  age[env_ids] = 0.0


def push_age_tracker(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  action: str = "tick",
) -> None:
  """Bookkeeping for :func:`push_by_setting_velocity_logged`.

  ``action="tick"`` (register with ``mode="step"``) ages the last push by one
  env step; ``action="clear"`` (``mode="reset"``) forgets it for the envs being
  reset, so a fresh episode never inherits the previous one's disturbance.
  """
  last, age = _push_buffers(env)
  if action == "tick":
    age += env.step_dt
  elif action == "clear":
    ids = resolve_env_ids(env, env_ids)
    last[ids] = 0.0
    age[ids] = 0.0
  else:
    raise ValueError(f"push_age_tracker: unknown action {action!r}")


def randomize_motor_strength(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  strength_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale each T-N envelope independently without scaling sub-limit torque."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  low, high = strength_range
  for actuator in asset.actuators[asset_cfg.actuator_ids]:
    if not hasattr(actuator, "strength_scale"):
      continue
    samples = torch.rand(
      (len(env_ids), actuator.strength_scale.shape[1]), device=env.device
    ) * (high - low) + low
    actuator.strength_scale[env_ids] = samples


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def add_payload_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("base_link",)),
) -> None:
  """Add an episode-fixed point payload at the selected body's nominal CoM."""
  env_ids = resolve_env_ids(env, env_ids)
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
  """Add the squat task's independent outward hip-roll reset perturbation."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  joint_pos = asset.data.joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel = asset.data.joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  spread = torch.empty(len(env_ids), 2, device=env.device).uniform_(*spread_range)
  spread *= (torch.rand(len(env_ids), 1, device=env.device) < spread_prob)
  spread[:, 1] *= -1.0
  joint_pos += spread
  limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos.clamp_(limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(
    joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids
  )


class apply_horizontal_body_push:
  """Apply fixed-duration horizontal forces with uniform magnitude and direction."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._cooldown_s = cfg.params["cooldown_s"]
    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._cooldown_remaining = self._sample_uniform(self._cooldown_s, self._num_envs)

  def _sample_uniform(self, value_range: tuple[float, float], count: int) -> torch.Tensor:
    low, high = value_range
    return torch.rand(count, device=self._device) * (high - low) + low

  def _write_force(self, env_ids: torch.Tensor, forces: torch.Tensor) -> None:
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(
      forces[:, None, :], torques[:, None, :], env_ids=env_ids, body_ids=self._body_ids
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
      self._write_force(expired_ids, torch.zeros((len(expired_ids), 3), device=self._device))
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
      (magnitude * torch.cos(angle), magnitude * torch.sin(angle), torch.zeros_like(magnitude)),
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
    self._write_force(ids, torch.zeros((len(ids), 3), device=self._device))
    self._active[ids] = False
    self._time_remaining[ids] = 0.0
    self._cooldown_remaining[ids] = self._sample_uniform(self._cooldown_s, len(ids))
