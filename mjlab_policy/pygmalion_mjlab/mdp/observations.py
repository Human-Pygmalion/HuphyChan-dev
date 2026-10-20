"""Critic-only domain-randomization state observations (not in stock mjlab). Copied verbatim from the training fork."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DR_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _live_model_field(env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
  """Live per-world model field as (num_envs, ...).

  mjlab only allocates per-world memory for fields some event asked for
  (``expand_model_fields``); everything else keeps its leading dim of 1. Expand
  the un-randomised case so callers can index uniformly.
  """
  t = getattr(env.sim.model, field)
  if t.shape[0] != env.num_envs:
    t = t.expand((env.num_envs,) + tuple(t.shape[1:]))
  return t


def _default_model_field(
  env: ManagerBasedRlEnv, field: str, idx: torch.Tensor
) -> torch.Tensor:
  """Nominal (pre-randomisation) values for ``field`` at ``idx``, as (num_envs, k, ...)."""
  d = env.sim.get_default_field(field)
  if field in env.sim.per_world_default_fields and d.shape[0] == env.num_envs:
    return d[:, idx]
  return d[idx].unsqueeze(0).expand((env.num_envs,) + tuple(d[idx].shape))


def dr_foot_friction(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DR_DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sliding friction actually in force on this env's foot geoms. Shape [B, 1].

  ``foot_friction`` uses ``shared_random=True`` so every foot geom carries the
  same coefficient; the mean is therefore that coefficient, not a summary of
  several.
  """
  asset = env.scene[asset_cfg.name]
  idx = _get_entity_indices(asset.indexing, asset_cfg, "geom", False)
  mu = _live_model_field(env, "geom_friction")[:, idx, 0]
  return mu.mean(dim=1, keepdim=True)


def dr_mass_offset(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DR_DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Total robot mass as a FRACTION off nominal. Shape [B, 1].

  Fractional rather than absolute so the scale is O(0.05) and the term stays
  comparable across model rebuilds (31.3 kg vs 35.3 kg).
  """
  asset = env.scene[asset_cfg.name]
  idx = _get_entity_indices(asset.indexing, asset_cfg, "body", False)
  live = _live_model_field(env, "body_mass")[:, idx].sum(dim=1, keepdim=True)
  nominal = _default_model_field(env, "body_mass", idx).sum(dim=1, keepdim=True)
  return (live - nominal) / nominal.clamp(min=1e-6)


def dr_com_offset(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DR_DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """CoM offset of the selected bodies from nominal, in metres. Shape [B, 3*k]."""
  asset = env.scene[asset_cfg.name]
  idx = _get_entity_indices(asset.indexing, asset_cfg, "body", False)
  live = _live_model_field(env, "body_ipos")[:, idx]
  nominal = _default_model_field(env, "body_ipos", idx)
  return (live - nominal).flatten(start_dim=1)


def dr_push_state(
  env: ManagerBasedRlEnv,
  event_name: str = "push_robot",
) -> torch.Tensor:
  """Push disturbance state. Shape [B, 8] = last kick (6) + age (1) + time-to-next (1).

  The last kick and its age are filled in by
  :func:`~mjlab.envs.mdp.events.push_by_setting_velocity_logged` plus the
  ``pyg_push_tick`` step event; time-to-next comes from the event manager's own
  interval bookkeeping. If ``push_robot`` is not an active event (phase 1, or
  play mode) every channel is zero.
  """
  last = getattr(env, "pyg_last_push", None)
  if last is None:
    last = torch.zeros(env.num_envs, 6, device=env.device)
  age = getattr(env, "pyg_push_age", None)
  if age is None:
    age = torch.zeros(env.num_envs, 1, device=env.device)

  em = env.event_manager
  time_left = torch.zeros(env.num_envs, 1, device=env.device)
  try:
    idx = em._mode_term_names["interval"].index(event_name)  # noqa: SLF001
    tl = em._interval_term_time_left[idx]  # noqa: SLF001
    if tl.numel() == env.num_envs:
      time_left = tl.to(env.device).reshape(env.num_envs, 1)
    else:  # is_global_time=True -> one scalar shared by every env
      time_left = tl.to(env.device).reshape(1, 1).expand(env.num_envs, 1)
  except (KeyError, ValueError, AttributeError, IndexError):
    pass

  return torch.cat([last, age, time_left], dim=1)


def dr_factor(
  env: ManagerBasedRlEnv,
  curriculum_name: str = "dr_levels",
) -> torch.Tensor:
  """The DR ramp scalar the ``dr_levels`` curriculum is currently applying. Shape [B, 1].

  Recomputed from the same start/end steps the curriculum uses rather than read
  out of its private state, so it cannot drift from what the events actually got.
  Returns 1.0 when there is no ramp term (DR already at full strength).
  """
  factor = 1.0
  try:
    params = env.curriculum_manager.get_term_cfg(curriculum_name).params
    start, end = int(params["start_step"]), int(params["end_step"])
    step = env.common_step_counter
    if step <= start:
      factor = 0.0
    elif step < end:
      factor = (step - start) / float(max(1, end - start))
  except (KeyError, ValueError, AttributeError):
    pass
  return torch.full((env.num_envs, 1), factor, device=env.device)

