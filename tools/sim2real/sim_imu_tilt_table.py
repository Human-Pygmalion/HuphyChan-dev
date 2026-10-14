#!/usr/bin/env python3
"""Reference table: rotate the pelvis (IMU site imu_in_base) about its own x, y, z axes in the training
model and read what the policy would get (projected gravity = -imu_upvector, gyro = imu_ang_vel), plus the
site's attitude angles. Next to it: what the real stack would show for the same pose if the EBIMU is
mounted with its gyro/accel axes = pelvis axes (datasheet: attitude frame = gyro frame rotated 180 deg
about X, world z down) and HUPHY computes gravity = R^T(0,0,-1) from that quaternion.

  <stock mjlab python> tools/sim2real/sim_imu_tilt_table.py <vibecoded-audit-20261012> --out table.csv
"""
import argparse, csv, math, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("pkg")
ap.add_argument("--angles", default="-30,-20,-10,-5,0,5,10,20,30")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.pkg); os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
import pygmalion_mjlab  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

cfg = load_env_cfg("Pygmalion-Velocity-Flat-P2", play=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg, device="cpu"); m = env.sim.mj_model; d = mujoco.MjData(m)
sid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): i for i in range(m.nsensor)}
read = lambda n: d.sensordata[m.sensor_adr[sid[n]]:m.sensor_adr[sid[n]] + m.sensor_dim[sid[n]]].copy()  # noqa: E731
site = m.sensor_objid[sid["robot/imu_upvector"]]
X180 = Rot.from_euler("x", 180, degrees=True)
rows = []
for ax in "xyz":
  for deg in [float(v) for v in a.angles.split(",")]:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 1.5
    R = Rot.from_euler(ax, deg, degrees=True)           # rotation about the pelvis' own axis from level
    qx, qy, qz, qw = R.as_quat(); d.qpos[3:7] = [qw, qx, qy, qz]
    w = np.zeros(3); w["xyz".index(ax)] = 1.0; d.qvel[3:6] = w   # 1 rad/s about the same axis (body frame)
    mujoco.mj_forward(m, d)
    grav = -read("robot/imu_upvector"); gyro = read("robot/imu_ang_vel")
    Rs = Rot.from_matrix(d.site_xmat[site].reshape(3, 3))
    roll, pitch, yaw = Rs.as_euler("xyz", degrees=False)[[0, 1, 2]]
    rpy = Rs.as_euler("ZYX", degrees=True)[::-1]          # roll(x), pitch(y), yaw(z), ZYX convention
    RE = X180 * Rs * X180                                 # EBIMU attitude frame (datasheet)
    ex, ey, ez, ew = RE.as_quat()
    if ew < 0: ew, ex, ey, ez = -ew, -ex, -ey, -ez
    ebimu_rpy = RE.as_euler("ZYX", degrees=True)[::-1]
    huphy = np.array([2 * (ew * ey - ex * ez), -2 * (ey * ez + ew * ex), 2 * (ex * ex + ey * ey) - 1])
    rows.append(dict(axis=ax, deg=deg,
                     sim_grav_x=grav[0], sim_grav_y=grav[1], sim_grav_z=grav[2],
                     sim_gyro_x=gyro[0], sim_gyro_y=gyro[1], sim_gyro_z=gyro[2],
                     sim_roll=rpy[0], sim_pitch=rpy[1], sim_yaw=rpy[2],
                     ebimu_roll=ebimu_rpy[0], ebimu_pitch=ebimu_rpy[1], ebimu_yaw=ebimu_rpy[2],
                     ebimu_qw=ew, ebimu_qx=ex, ebimu_qy=ey, ebimu_qz=ez,
                     huphy_grav_x=huphy[0], huphy_grav_y=huphy[1], huphy_grav_z=huphy[2]))
with open(a.out, "w", newline="") as f:
  wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader()
  for r in rows: wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
print("site", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, site), "quat in body", m.site_quat[site])
for r in rows:
  print(f"{r['axis']} {r['deg']:+5.0f} | sim grav {r['sim_grav_x']:+.3f} {r['sim_grav_y']:+.3f} {r['sim_grav_z']:+.3f} | sim rpy {r['sim_roll']:+6.1f} {r['sim_pitch']:+6.1f} {r['sim_yaw']:+6.1f} | gyro {r['sim_gyro_x']:+.0f} {r['sim_gyro_y']:+.0f} {r['sim_gyro_z']:+.0f} | EBIMU rpy {r['ebimu_roll']:+6.1f} {r['ebimu_pitch']:+6.1f} {r['ebimu_yaw']:+6.1f} q {r['ebimu_qw']:.3f} {r['ebimu_qx']:+.3f} {r['ebimu_qy']:+.3f} {r['ebimu_qz']:+.3f} | HUPHY grav {r['huphy_grav_x']:+.3f} {r['huphy_grav_y']:+.3f} {r['huphy_grav_z']:+.3f}")
env.close()
