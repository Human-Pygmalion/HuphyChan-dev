"""Ten fixed walking terrains adapted from the seed-89500 bicycle curriculum."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class StripTerrainCfg(SubTerrainCfg):
  slope_deg: float = 0.0
  ridge_height_range: tuple[float, float] = (0.0, 0.0)
  ridge_spacing_range: tuple[float, float] = (0.10, 0.20)
  base_thickness: float = 1.0
  contact_solref: tuple[float, float] = (0.03, 1.0)
  contact_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.003, 0.5, 2.0)

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty
    body = spec.body("terrain")
    size_x, size_y = self.size
    center_x, center_y = size_x / 2.0, size_y / 2.0
    pitch = -math.radians(self.slope_deg)
    half_pitch = 0.5 * pitch
    thickness = self.base_thickness
    base = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(size_x / 2.0, size_y / 2.0, thickness / 2.0),
      pos=(
        center_x - 0.5 * thickness * math.sin(pitch),
        center_y,
        -0.5 * thickness * math.cos(pitch),
      ),
      quat=(math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0),
    )
    geometries = [TerrainGeometry(geom=base, color=(0.42, 0.44, 0.46, 1.0))]
    min_height, max_height = self.ridge_height_range
    if max_height > 0.0:
      x = 0.30
      while x < size_x - 0.30:
        radius = float(rng.uniform(min_height, max_height))
        z = math.tan(math.radians(self.slope_deg)) * (x - center_x)
        ridge = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_CAPSULE,
          size=(radius, (size_y - 0.10) / 2.0, 0.0),
          pos=(x, center_y, z),
          quat=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
        )
        geometries.append(TerrainGeometry(geom=ridge, color=(0.50, 0.48, 0.43, 1.0)))
        x += float(rng.uniform(*self.ridge_spacing_range))
    for geometry in geometries:
      assert geometry.geom is not None
      geometry.geom.solref = self.contact_solref
      geometry.geom.solimp = self.contact_solimp
    return TerrainOutput(origin=np.array((center_x, center_y, 0.0)), geometries=geometries)


LEVEL_SPECS = (
  ("level1_flat", 0.0, (0.0, 0.0)),
  ("level2_ridge05cm", 0.0, (0.005, 0.005)),
  ("level3_ridge10cm", 0.0, (0.010, 0.010)),
  ("level4_up3deg", 3.0, (0.0, 0.0)),
  ("level5_down3deg", -3.0, (0.0, 0.0)),
  ("level6_rough15cm", 0.0, (0.0075, 0.015)),
  ("level7_rough20cm", 0.0, (0.010, 0.020)),
  ("level8_up6deg", 6.0, (0.0, 0.0)),
  ("level9_down6deg", -6.0, (0.0, 0.0)),
  ("level10_up6deg_rough", 6.0, (0.010, 0.020)),
)


def fixed_step_curriculum(
  env,
  env_ids: torch.Tensor | slice,
  iterations_per_level: int = 5000,
  steps_per_iteration: int = 24,
  num_levels: int = 10,
  start_iteration: int = 0,
  start_max_level: int = 0,
) -> torch.Tensor:
  """Unlock levels at fixed iteration intervals, sampling unlocked levels uniformly."""
  terrain = env.scene.terrain
  if terrain.terrain_origins is None:
    raise RuntimeError("terrain10 curriculum requires generated terrain")
  selected_ids = (
    torch.arange(env.num_envs, device=env.device)[env_ids]
    if isinstance(env_ids, slice)
    else env_ids
  )
  iteration = env.common_step_counter // steps_per_iteration
  elapsed_iterations = max(0, iteration - start_iteration)
  max_level = min(
    num_levels - 1,
    start_max_level + elapsed_iterations // iterations_per_level,
  )
  levels = torch.randint(0, max_level + 1, (len(selected_ids),), device=env.device)
  terrain.terrain_levels[selected_ids] = 0
  terrain.terrain_types[selected_ids] = levels
  terrain.env_origins[selected_ids] = terrain.terrain_origins[0, levels]
  return levels.float().mean() + 1.0
