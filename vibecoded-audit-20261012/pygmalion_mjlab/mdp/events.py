"""push_robot variant that records the applied push, for the critic observation dr_push_state. Copied verbatim from the training fork."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  sample_uniform,
)

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

