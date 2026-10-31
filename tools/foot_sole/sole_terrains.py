"""Box-built rough terrains for the foot-sole study (no height fields).

MuJoCo-Warp height-field contact misbehaves under walking feet (docs/experiments/2026-10-11_flat_vs_rough_policy_compare.md §2, §2b:
feet on a perfectly flat height field slip 3-4x faster and 77-100 % of the runs fall; the same ground built from boxes: 0 %).
The training terrain (pygmalion_mjlab/terrain10.py) is boxes and capsules only, with solref 0.03 / solimp 0.9 0.95 0.003 0.5 2.
These terrains use the same base box and the same contact settings.

ScatterBoxTerrainCfg: flat base + n random axis-yawed boxes (width, height fixed range, random yaw) over the whole tile, or only
inside a corridor along +x (corridor_half_width > 0, centred on the tile centre line) for pebbles that must be dense under the feet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class ScatterBoxTerrainCfg(SubTerrainCfg):
  n_boxes: int = 3500
  width_range: tuple[float, float] = (0.10, 0.20)
  height_range: tuple[float, float] = (0.02, 0.02)
  corridor_half_width: float = 0.0
  margin: float = 1.0
  contact_solref: tuple[float, float] = (0.03, 1.0)
  contact_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.003, 0.5, 2.0)

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty
    body = spec.body("terrain")
    sx, sy = self.size
    base = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, 0.5), pos=(sx / 2, sy / 2, -0.5))
    geoms = [TerrainGeometry(geom=base, color=(0.42, 0.44, 0.46, 1.0))]
    for _ in range(self.n_boxes):
      w = float(rng.uniform(*self.width_range))
      l = float(rng.uniform(*self.width_range))
      h = float(rng.uniform(*self.height_range))
      x = float(rng.uniform(self.margin + w, sx - self.margin - w))
      if self.corridor_half_width > 0:
        y = sy / 2 + float(rng.uniform(-self.corridor_half_width, self.corridor_half_width))
      else:
        y = float(rng.uniform(self.margin + l, sy - self.margin - l))
      yaw = float(rng.uniform(0, 2 * math.pi))
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(w / 2, l / 2, h / 2), pos=(x, y, h / 2))
      g.quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
      geoms.append(TerrainGeometry(geom=g, color=(0.50, 0.48, 0.43, 1.0)))
    for tg in geoms:
      tg.geom.solref = self.contact_solref
      tg.geom.solimp = self.contact_solimp
    return TerrainOutput(origin=np.array((sx / 2, sy / 2, 0.0)), geometries=geoms)
