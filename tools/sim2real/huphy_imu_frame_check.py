#!/usr/bin/env python3
"""Run HUPHY's own IMU frame check (tools/imu_viewer.py analyze, unmodified) on the policy logs.

The check asks whether the policy's gyro (obs 0-2) and projected gravity (obs 3-5) obey
g_dot = -w x g, trying all 8 gyro sign combinations. (1,1,1) means they are in the same frame.

  python tools/sim2real/huphy_imu_frame_check.py --huphy <HUPHY biped checkout>
"""
import argparse, glob, importlib.util, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--huphy", required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.huphy, "src"))
spec = importlib.util.spec_from_file_location("imu_viewer", os.path.join(a.huphy, "tools/imu_viewer.py"))
iv = importlib.util.module_from_spec(spec); spec.loader.exec_module(iv)
assert iv.FIELDS[4:7] == ("gx", "gy", "gz") and iv.FIELDS[7:10] == ("ax", "ay", "az")
for f in sorted(glob.glob(os.path.join(a.huphy, "archivedlogs", "*ob*.csv"))):
  d = np.genfromtxt(f, delimiter=",", skip_header=1)
  if len(d) < 200:
    continue
  data = np.zeros((len(d), 10)); data[:, 0] = d[:, 0]; data[:, 4:7] = d[:, 1:4]; data[:, 7:10] = d[:, 4:7]
  r = iv.analyze(data)
  i, b = r["identity"], r["best"]
  print(f"{os.path.basename(f):24s} moving {r['moving']:5d} | (1,1,1) err {i[0]:.2f} corr {i[1]:+.2f} | best {b[2]} err {b[0]:.2f} corr {b[1]:+.2f}")
