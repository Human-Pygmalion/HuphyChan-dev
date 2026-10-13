#!/usr/bin/env python3
"""EBIMU frames (datasheet, user-confirmed 2026-10-13): the attitude frame (Euler/quaternion) equals the
gyro/accel frame rotated 180 deg about X (X same, Y and Z inverted; both right-handed), so its world has
z down. HUPHY's gravity_from_quat assumes the quaternion is in the gyro frame with a z-up world.
This checks, over random orientations, that HUPHY's value equals the true sensor-frame gravity with x
negated - the error seen in the real logs.

  python tools/sim2real/ebimu_frame_model.py
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

rng = np.random.default_rng(0)
Rx180 = Rot.from_euler("x", 180, degrees=True)


def huphy_gravity(w, x, y, z):  # HUPHY sensors/base.py gravity_from_quat
  return np.array([2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1])


e_flip = e_same = 0.0
for _ in range(10000):
  RS = Rot.random(random_state=rng)          # gyro/accel frame in a z-up world
  RA = Rx180 * RS * Rx180                    # attitude frame in the z-down world
  qx, qy, qz, qw = RA.as_quat()
  h = huphy_gravity(qw, qx, qy, qz)
  g = RS.inv().apply([0, 0, -1])
  e_flip = max(e_flip, np.abs(h - g * [-1, 1, 1]).max())
  e_same = max(e_same, np.abs(h - g).max())
print(f"max |HUPHY - sensor gravity with x negated| = {e_flip:.1e};  max |HUPHY - sensor gravity| = {e_same:.2f}")
