#!/usr/bin/env python3
"""Show the requested 30-step stairs and slopes in a Viser gallery."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import viser
from mjviser.conversions import merge_geoms

from mjlab.terrains.terrain_generator import TerrainGenerator
from pygmalion_mjlab.stairs_slopes import stairs_slopes_terrain_cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", type=int, default=8081)
  args = parser.parse_args()

  cfg = stairs_slopes_terrain_cfg()
  generator = TerrainGenerator(cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()

  terrain_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  geom_ids = [
    geom_id
    for geom_id in range(model.ngeom)
    if model.geom_bodyid[geom_id] == terrain_body_id
  ]
  mesh = merge_geoms(model, geom_ids)

  server = viser.ViserServer(port=args.port)
  server.scene.add_mesh_trimesh("/stairs_slopes", mesh)
  names = tuple(cfg.sub_terrains)
  for col, name in enumerate(names):
    origin = generator.terrain_origins[0, col]
    server.scene.add_label(
      f"/labels/{col:02d}",
      text=name.replace("_", " "),
      position=origin + np.array((0.0, 0.0, 0.35)),
    )

  server.gui.add_markdown(
    "**13 terrains:** 7 stair heights (30 up + 30 down) + "
    "6 paired up/down slopes; each patch is 27 x 9 m"
  )

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = np.array((45.0, -75.0, 65.0))
    client.camera.look_at = np.array((0.0, 0.0, 3.0))

  print(f"Generated {len(names)} terrains and {len(geom_ids)} MuJoCo geoms.")
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
