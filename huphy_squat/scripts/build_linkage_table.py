#!/usr/bin/env python3
"""Rebuild huphy_squat/assets/linkage_table.npz from the MJCF (run after any MJCF or default-pose change)."""

from __future__ import annotations

import numpy as np

from huphy_squat import linkage, robot


def main() -> None:
  tab = linkage.build_table()
  np.savez(robot.LINKAGE_TABLE, **tab)
  pose = dict(zip(tab["default_names"], tab["default_values"], strict=True))
  print(f"wrote {robot.LINKAGE_TABLE}")
  print(f"default base height (lowest sole corner on the floor): {-tab['corner_z0'].min():.5f} m")
  for k in sorted(pose):
    print(f"  {k:24s} {pose[k]: .6f}")


if __name__ == "__main__":
  main()
