"""Long straight stairs and slopes for curriculum experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class LongStairsTerrainCfg(SubTerrainCfg):
  """A straight staircase with matching ascending and descending flights."""

  step_height: float
  num_steps: int = 30
  tread_depth: float = 23.0 / 60.0
  approach_length: float = 2.0
  base_depth: float = 0.20
  bidirectional: bool = False

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    if self.num_steps <= 0:
      raise ValueError("num_steps must be positive")
    if self.step_height <= 0.0 or self.tread_depth <= 0.0:
      raise ValueError("step_height and tread_depth must be positive")
    total_treads = 2 * self.num_steps
    profile_length = 2 * self.approach_length + total_treads * self.tread_depth
    required_length = 2 * profile_length if self.bidirectional else profile_length
    if required_length > self.size[0] + 1e-9:
      raise ValueError(
        f"{self.num_steps} up + {self.num_steps} down steps need "
        f"{required_length:.2f} m, "
        f"but terrain length is {self.size[0]:.2f} m"
      )

    body = spec.body("terrain")
    center_y = 0.5 * self.size[1]
    width = max(0.05, self.size[1] - 0.10)
    flat_color = (0.24, 0.34, 0.48, 1.0)
    geometries: list[TerrainGeometry] = []

    def add_pad(center_x: float, length: float) -> None:
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * length, 0.5 * width, 0.5 * self.base_depth),
        pos=(center_x, center_y, -0.5 * self.base_depth),
      )
      geometries.append(TerrainGeometry(geom=geom, color=flat_color))

    def add_flight(start_x: float, direction: float) -> None:
      for tread in range(total_treads):
        if tread < self.num_steps:
          top = (tread + 1) * self.step_height
        else:
          top = (total_treads - tread - 1) * self.step_height
        box_height = top + self.base_depth
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(0.5 * self.tread_depth, 0.5 * width, 0.5 * box_height),
          pos=(
            start_x + direction * (tread + 0.5) * self.tread_depth,
            center_y,
            0.5 * (top - self.base_depth),
          ),
        )
        t = top / (self.num_steps * self.step_height)
        geometries.append(
          TerrainGeometry(
            geom=geom,
            color=(0.20 + 0.35 * t, 0.42, 0.75 - 0.30 * t, 1.0),
          )
        )

    if self.bidirectional:
      center_x = 0.5 * self.size[0]
      add_pad(center_x, 2 * self.approach_length)
      add_pad(0.5 * self.approach_length, self.approach_length)
      add_pad(self.size[0] - 0.5 * self.approach_length, self.approach_length)
      add_flight(center_x + self.approach_length, 1.0)
      add_flight(center_x - self.approach_length, -1.0)
      origin_x = center_x
    else:
      add_pad(0.5 * self.approach_length, self.approach_length)
      add_pad(self.size[0] - 0.5 * self.approach_length, self.approach_length)
      add_flight(self.approach_length, 1.0)
      origin_x = 0.5 * self.approach_length

    return TerrainOutput(
      origin=np.array((origin_x, center_y, 0.0)),
      geometries=geometries,
    )


@dataclass(kw_only=True)
class LongSlopeTerrainCfg(SubTerrainCfg):
  """A straight ramp that rises and then descends at the same angle."""

  slope_deg: float
  approach_length: float = 2.0
  base_depth: float = 0.20
  bidirectional: bool = False

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    if not 0.0 < self.slope_deg < 90.0:
      raise ValueError("slope_deg must be between 0 and 90 degrees")

    angle = math.radians(self.slope_deg)
    run, width = self.size
    thickness = self.base_depth
    body = spec.body("terrain")

    def add_ramp(
      x0: float, z0: float, x1: float, z1: float
    ) -> mujoco.MjsGeom:
      theta = math.atan2(z1 - z0, x1 - x0)
      horizontal_run = x1 - x0
      length = horizontal_run / math.cos(theta)
      center_x = 0.5 * (x0 + x1) - 0.5 * thickness * math.sin(theta)
      center_z = 0.5 * (z0 + z1) - 0.5 * thickness * math.cos(theta)
      half_angle = -0.5 * theta
      return body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * length, 0.5 * max(0.05, width - 0.10), 0.5 * thickness),
        pos=(center_x, 0.5 * width, center_z),
        quat=(math.cos(half_angle), 0.0, math.sin(half_angle), 0.0),
      )

    t = self.slope_deg / 30.0
    color = (0.72, 0.48 - 0.18 * t, 0.22, 1.0)

    def add_pad(center_x: float, length: float) -> TerrainGeometry:
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * length, 0.5 * max(0.05, width - 0.10), 0.5 * thickness),
        pos=(center_x, 0.5 * width, -0.5 * thickness),
      )
      return TerrainGeometry(geom=geom, color=color)

    if self.bidirectional:
      center_x = 0.5 * run
      inner_left = center_x - self.approach_length
      inner_right = center_x + self.approach_length
      outer_left = self.approach_length
      outer_right = run - self.approach_length
      left_peak = 0.5 * (outer_left + inner_left)
      right_peak = 0.5 * (inner_right + outer_right)
      peak_height = (left_peak - outer_left) * math.tan(angle)
      geometries = [
        add_pad(center_x, 2 * self.approach_length),
        add_pad(0.5 * self.approach_length, self.approach_length),
        TerrainGeometry(geom=add_ramp(outer_left, 0.0, left_peak, peak_height), color=color),
        TerrainGeometry(geom=add_ramp(left_peak, peak_height, inner_left, 0.0), color=color),
        TerrainGeometry(geom=add_ramp(inner_right, 0.0, right_peak, peak_height), color=color),
        TerrainGeometry(geom=add_ramp(right_peak, peak_height, outer_right, 0.0), color=color),
        add_pad(run - 0.5 * self.approach_length, self.approach_length),
      ]
      origin_x = center_x
    else:
      ramp_start = self.approach_length
      ramp_end = run - self.approach_length
      if ramp_end <= ramp_start:
        raise ValueError("terrain is too short for the requested approach lengths")
      peak_x = 0.5 * (ramp_start + ramp_end)
      peak_height = (peak_x - ramp_start) * math.tan(angle)
      geometries = [
        add_pad(0.5 * self.approach_length, self.approach_length),
        TerrainGeometry(geom=add_ramp(ramp_start, 0.0, peak_x, peak_height), color=color),
        TerrainGeometry(geom=add_ramp(peak_x, peak_height, ramp_end, 0.0), color=color),
        add_pad(run - 0.5 * self.approach_length, self.approach_length),
      ]
      origin_x = 0.5 * self.approach_length
    return TerrainOutput(
      origin=np.array((origin_x, 0.5 * width, 0.0)),
      geometries=geometries,
    )


STAIR_HEIGHTS_CM = (3, 6, 9, 12, 15, 18, 21)
SLOPE_ANGLES_DEG = (5, 10, 15, 20, 25, 30)

# Twelve training stages, ordered from low to high while alternating geometry.
# The 21 cm stairs remain available for previews but are intentionally excluded.
TRAINING_LEVELS = tuple(
  item
  for slope_deg, stair_cm in zip(SLOPE_ANGLES_DEG, STAIR_HEIGHTS_CM[:-1], strict=True)
  for item in (("slope", slope_deg), ("stairs", stair_cm))
)


def stairs_slopes_terrain_cfg():
  """Return a one-row gallery config containing every requested terrain once."""
  from mjlab.terrains import TerrainGeneratorCfg

  sub_terrains: dict[str, SubTerrainCfg] = {}
  for height_cm in STAIR_HEIGHTS_CM:
    sub_terrains[f"stairs_{height_cm:02d}cm"] = LongStairsTerrainCfg(
      step_height=height_cm / 100.0,
    )
  for angle_deg in SLOPE_ANGLES_DEG:
    sub_terrains[f"slope_{angle_deg:02d}deg"] = LongSlopeTerrainCfg(
      slope_deg=float(angle_deg),
    )
  return TerrainGeneratorCfg(
    seed=42,
    size=(27.0, 9.0),
    num_rows=1,
    num_cols=len(sub_terrains),
    border_width=0.0,
    curriculum=True,
    difficulty_range=(1.0, 1.0),
    color_scheme="height",
    add_lights=True,
    sub_terrains=sub_terrains,
  )


def curriculum_terrain_cfg():
  """Return the 12-stage, extra-wide training terrain configuration."""
  from mjlab.terrains import TerrainGeneratorCfg

  sub_terrains: dict[str, SubTerrainCfg] = {}
  for kind, value in TRAINING_LEVELS:
    if kind == "slope":
      name = f"slope_{value:02d}deg"
      sub_terrains[name] = LongSlopeTerrainCfg(
        slope_deg=float(value), bidirectional=True
      )
    else:
      name = f"stairs_{value:02d}cm"
      sub_terrains[name] = LongStairsTerrainCfg(
        step_height=value / 100.0, bidirectional=True
      )
  return TerrainGeneratorCfg(
    seed=89500,
    size=(54.0, 45.0),
    num_rows=1,
    num_cols=len(sub_terrains),
    border_width=0.0,
    curriculum=True,
    difficulty_range=(1.0, 1.0),
    color_scheme="height",
    sub_terrains=sub_terrains,
  )


def progressive_uniform_curriculum(
  env,
  env_ids: torch.Tensor | slice,
  start_step: int = 1_200_048,
  iterations_per_level: int = 2_000,
  steps_per_iteration: int = 24,
  num_levels: int = 12,
) -> torch.Tensor:
  """Unlock one terrain every interval and sample all unlocked types uniformly."""
  terrain = env.scene.terrain
  if terrain.terrain_origins is None:
    raise RuntimeError("stairs/slopes curriculum requires generated terrain")
  selected_ids = (
    torch.arange(env.num_envs, device=env.device)[env_ids]
    if isinstance(env_ids, slice)
    else env_ids
  )
  elapsed_steps = max(0, env.common_step_counter - start_step)
  unlock_interval = iterations_per_level * steps_per_iteration
  max_level = min(num_levels - 1, elapsed_steps // unlock_interval)
  levels = torch.randint(
    0, max_level + 1, (len(selected_ids),), device=env.device
  )
  terrain.terrain_levels[selected_ids] = 0
  terrain.terrain_types[selected_ids] = levels
  terrain.env_origins[selected_ids] = terrain.terrain_origins[0, levels]
  return levels.float().mean() + 1.0
