"""Small deterministic obstacle courses used for interactive policy playback."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class FourStepDownSlopeCfg(SubTerrainCfg):
  """Four ascending steps, a short landing, and a downhill ramp."""

  step_height: float = 0.15
  num_steps: int = 4
  tread_depth: float = 0.38
  landing_length: float = 0.50
  downhill_deg: float = 20.0
  approach_length: float = 2.0
  exit_length: float = 2.0
  base_depth: float = 0.20
  slope_first: bool = False

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    if self.num_steps <= 0:
      raise ValueError("num_steps must be positive")
    if self.step_height <= 0.0 or self.tread_depth <= 0.0:
      raise ValueError("step dimensions must be positive")
    if not 0.0 < self.downhill_deg < 90.0:
      raise ValueError("downhill_deg must be between 0 and 90 degrees")

    width = self.size[1]
    top_height = self.num_steps * self.step_height
    ramp_run = top_height / math.tan(math.radians(self.downhill_deg))
    obstacle_length = self.num_steps * self.tread_depth + self.landing_length + ramp_run
    required_length = self.approach_length + obstacle_length + self.exit_length
    if required_length > self.size[0] + 1e-9:
      raise ValueError(
        f"course requires {required_length:.3f} m but terrain is {self.size[0]:.3f} m"
      )

    body = spec.body("terrain")
    center_y = 0.5 * width
    geometries: list[TerrainGeometry] = []

    def add_box(x0: float, x1: float, top: float, color) -> None:
      height = top + self.base_depth
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * (x1 - x0), 0.5 * width, 0.5 * height),
        pos=(0.5 * (x0 + x1), center_y, 0.5 * (top - self.base_depth)),
      )
      geometries.append(TerrainGeometry(geom=geom, color=color))

    flat_color = (0.30, 0.38, 0.46, 1.0)
    stair_color = (0.28, 0.48, 0.72, 1.0)
    ramp_color = (0.72, 0.42, 0.20, 1.0)
    def add_ramp(x0: float, x1: float, z0: float, z1: float) -> None:
      theta = math.atan2(z1 - z0, x1 - x0)
      ramp_length = (x1 - x0) / math.cos(theta)
      ramp = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * ramp_length, 0.5 * width, 0.5 * self.base_depth),
        pos=(
          0.5 * (x0 + x1) - 0.5 * self.base_depth * math.sin(theta),
          center_y,
          0.5 * (z0 + z1) - 0.5 * self.base_depth * math.cos(theta),
        ),
        quat=(math.cos(-0.5 * theta), 0.0, math.sin(-0.5 * theta), 0.0),
      )
      geometries.append(TerrainGeometry(geom=ramp, color=ramp_color))

    add_box(0.0, self.approach_length, 0.0, flat_color)
    if self.slope_first:
      ramp_start = self.approach_length
      ramp_end = ramp_start + ramp_run
      landing_end = ramp_end + self.landing_length
      stairs_end = landing_end + self.num_steps * self.tread_depth
      add_ramp(ramp_start, ramp_end, 0.0, top_height)
      add_box(ramp_end, landing_end, top_height, flat_color)
      for index in range(self.num_steps):
        x0 = landing_end + index * self.tread_depth
        top = (self.num_steps - index - 1) * self.step_height
        add_box(x0, x0 + self.tread_depth, top, stair_color)
      add_box(stairs_end, self.size[0], 0.0, flat_color)
    else:
      stairs_end = self.approach_length + self.num_steps * self.tread_depth
      landing_end = stairs_end + self.landing_length
      ramp_end = landing_end + ramp_run
      for index in range(self.num_steps):
        x0 = self.approach_length + index * self.tread_depth
        add_box(x0, x0 + self.tread_depth, (index + 1) * self.step_height, stair_color)
      add_box(stairs_end, landing_end, top_height, flat_color)
      add_ramp(landing_end, ramp_end, top_height, 0.0)
      add_box(ramp_end, self.size[0], 0.0, flat_color)

    return TerrainOutput(
      origin=np.array((0.5 * self.approach_length, center_y, 0.0)),
      geometries=geometries,
    )


def four_step_downslope_terrain_cfg(slope_first: bool = False) -> TerrainGeneratorCfg:
  """Return the requested 2 m-wide 15 cm-step / 20 degree-downhill course."""
  course = FourStepDownSlopeCfg(size=(8.0, 2.0), slope_first=slope_first)
  name = (
    "up20deg_landing50cm_downstairs15x4"
    if slope_first
    else "stairs15x4_landing50cm_down20deg"
  )
  return TerrainGeneratorCfg(
    seed=42,
    size=course.size,
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    curriculum=False,
    color_scheme="none",
    sub_terrains={name: course},
  )
