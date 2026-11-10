#!/usr/bin/env python3
"""Build the M19 training MJCF with 7 sole and 5 inset upper capsules per foot."""

from __future__ import annotations

import colorsys
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "pygmalion_locomotion/assets/pygmalion_v2"
STEM = (
  "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_"
  "footv5_d455_reinf_alparts2"
)
SOURCE = ASSETS / f"{STEM}_loop.xml"
OUTPUT = ASSETS / f"{STEM}_capsule7plus5_loop.xml"
FOOT_GEOM_LINE = re.compile(r'^(\s*)<geom name="([LR])_foot\d+_collision".*?/>\s*$')


def rgba(index: int) -> str:
  red, green, blue = colorsys.hsv_to_rgb(((index - 1) * 0.61803398875) % 1.0, 0.75, 0.95)
  return f"{red:.3f} {green:.3f} {blue:.3f} 0.86"


def capsules(side: str, indent: str) -> list[str]:
  result: list[str] = []
  sole_radius = 0.0085
  sole_z = -0.057 + sole_radius
  for index in range(1, 8):
    y = (index - 4) * 0.0165
    edge_offset = max(0.0, abs(y) - 0.0275)
    end_extension = math.sqrt(max(0.0, 0.030**2 - edge_offset**2))
    x_min = -0.050 - end_extension
    x_max = 0.150 + end_extension
    result.append(
      f'{indent}<geom name="{side}_foot{index}_collision" class="collision" '
      f'type="capsule" size="{sole_radius:.4f}" '
      f'fromto="{x_min + sole_radius:.6f} {y:.6f} {sole_z:.6f} '
      f'{x_max - sole_radius:.6f} {y:.6f} {sole_z:.6f}" rgba="{rgba(index)}"/>'
    )

  # The measured visual envelope is x=[-82, 182], y=[-59.5, 59.5],
  # z=[-57, 25.1] mm. Keep these five transverse ribs inset on every axis.
  upper = (
    (-0.060, -0.030, 0.010),
    (-0.010, -0.018, 0.011),
    (0.040, -0.006, 0.012),
    (0.090, -0.010, 0.011),
    (0.140, -0.023, 0.010),
  )
  outer_half_width = 0.0555
  for index, (x, z, radius) in enumerate(upper, start=8):
    half_segment = outer_half_width - radius
    result.append(
      f'{indent}<geom name="{side}_foot{index}_collision" class="collision" '
      f'type="capsule" size="{radius:.4f}" '
      f'fromto="{x:.6f} {-half_segment:.6f} {z:.6f} '
      f'{x:.6f} {half_segment:.6f} {z:.6f}" rgba="{rgba(index)}"/>'
    )
  return result


def main() -> None:
  output: list[str] = []
  emitted: set[str] = set()
  removed = {"L": 0, "R": 0}
  for line in SOURCE.read_text().splitlines():
    match = FOOT_GEOM_LINE.match(line)
    if match is None:
      output.append(line)
      continue
    indent, side = match.group(1), match.group(2)
    removed[side] += 1
    if side not in emitted:
      output.extend(capsules(side, indent))
      emitted.add(side)
  if removed != {"L": 32, "R": 32}:
    raise RuntimeError(f"expected 32 old collisions per foot, got {removed}")
  OUTPUT.write_text("\n".join(output) + "\n")
  print(f"wrote {OUTPUT}")
  print(f"removed {removed}; added 7 sole + 5 inset upper capsules per foot")


if __name__ == "__main__":
  main()
