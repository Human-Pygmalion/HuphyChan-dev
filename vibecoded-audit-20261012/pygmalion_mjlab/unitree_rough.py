"""Uniform staged curriculum for mjlab's stock rough terrains."""

from __future__ import annotations

import torch


def staged_uniform_curriculum(
  env,
  env_ids: torch.Tensor | slice,
  start_iteration: int = 19_500,
  iterations_per_level: int = 2_000,
  steps_per_iteration: int = 24,
  num_levels: int = 10,
  start_max_level: int = 0,
) -> dict[str, torch.Tensor | float]:
  """Sample terrain type uniformly and unlock one difficulty row every 2k iterations."""
  terrain = env.scene.terrain
  if terrain.terrain_origins is None:
    raise RuntimeError("Unitree rough curriculum requires generated terrain")
  selected_ids = (
    torch.arange(env.num_envs, device=env.device)[env_ids]
    if isinstance(env_ids, slice)
    else env_ids
  )
  iteration = env.common_step_counter // steps_per_iteration
  unlocked_level = (
    start_max_level + max(0, iteration - start_iteration) // iterations_per_level
  )
  max_level = min(num_levels - 1, unlocked_level)
  num_types = terrain.terrain_origins.shape[1]
  levels = torch.randint(0, max_level + 1, (len(selected_ids),), device=env.device)
  types = torch.randint(0, num_types, (len(selected_ids),), device=env.device)
  terrain.terrain_levels[selected_ids] = levels
  terrain.terrain_types[selected_ids] = types
  terrain.env_origins[selected_ids] = terrain.terrain_origins[levels, types]
  return {
    "max_level": float(max_level),
    "mean_level": levels.float().mean(),
    "mean_type": types.float().mean(),
  }
