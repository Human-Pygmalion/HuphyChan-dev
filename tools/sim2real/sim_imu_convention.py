#!/usr/bin/env python3
"""Sim IMU sign convention: build the task model, set the free-joint pose/velocity on the CPU MuJoCo model, mj_forward, read the IMU sensors.

  <mjlab python> tools/sim2real/sim_imu_convention.py <vibecoded-audit-20261012>
"""
import os, sys, math
sys.path.insert(0, sys.argv[1]); os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
import pygmalion_mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Pygmalion-Velocity-Flat-P2", play=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg, device="cpu"); m = env.sim.mj_model; d = mujoco.MjData(m)
sid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): i for i in range(m.nsensor)}
def read(n):
  i = sid[n]; a = m.sensor_adr[i]; return d.sensordata[a:a + m.sensor_dim[i]].copy()
s = m.sensor_objid[sid["robot/imu_upvector"]]
print("imu site:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s), "on body", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[s]),
      "site quat in body (w,x,y,z)", m.site_quat[s].round(4))
print("sensors:", list(sid))
def q_axis(ax, deg):
  h = math.radians(deg) / 2; v = [0.0, 0.0, 0.0]; v["xyz".index(ax)] = math.sin(h); return np.r_[math.cos(h), v]
gyro_name = [n for n in sid if "ang_vel" in n or "gyro" in n]
print("gyro sensors:", gyro_name)
for label, ax, deg, w in (("level", "y", 0, None), ("nose DOWN 10deg (+rotation about body y)", "y", 10, None),
                          ("right side DOWN 10deg (+rotation about body x)", "x", 10, None),
                          ("rotating nose-down at 1 rad/s", "y", 0, (0, 1, 0)), ("rotating right-side-down at 1 rad/s", "y", 0, (1, 0, 0)),
                          ("turning left (CCW from above) at 1 rad/s", "y", 0, (0, 0, 1))):
  mujoco.mj_resetData(m, d); d.qpos[2] = 1.5; d.qpos[3:7] = q_axis(ax, deg)
  if w: d.qvel[3:6] = w   # free joint angular velocity (body frame)
  mujoco.mj_forward(m, d)
  up = read("robot/imu_upvector")
  # mjlab projected_gravity_from_sensor = -upvector expressed in the body... print both raw
  g = [read(n) for n in gyro_name]
  print(f"{label:46s} upvector(sensor) {up.round(3)}  gyro {[x.round(3) for x in g]}")
env.close()
