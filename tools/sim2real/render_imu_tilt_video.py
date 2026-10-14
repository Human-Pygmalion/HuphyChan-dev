#!/usr/bin/env python3
"""Video: the training model's pelvis (IMU site imu_in_base) is rotated about its own x, then y, then z
axis, and the IMU values the policy would get are shown live next to it.

Left: the robot (legs at the default pose, base held in the air, no physics - a kinematic demo) with the
pelvis axes (x red forward, y green left, z blue up), the true gravity direction (orange, what the sim
gives the policy) and the gravity HUPHY would give for the same pose (magenta; datasheet frame model,
sensor axes = pelvis axes). Right: projected gravity, gyro, attitude angles, with any HUPHY value that
differs from the sim in red.

Played in real time: every rotation runs at the stated deg/s, the clock is on screen.

  <stock mjlab python> tools/sim2real/render_imu_tilt_video.py <vibecoded-audit-20261012> --out out.mp4
"""
import argparse, math, os, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("pkg")
ap.add_argument("--out", required=True)
ap.add_argument("--deg", type=float, default=20.0)
ap.add_argument("--rate", type=float, default=20.0, help="deg/s")
ap.add_argument("--hold", type=float, default=2.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--ffmpeg", default="ffmpeg")
a = ap.parse_args()
sys.path.insert(0, a.pkg); os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
import pygmalion_mjlab  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

cfg = load_env_cfg("Pygmalion-Velocity-Flat-P2", play=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg, device="cpu")
m = env.sim.mj_model; d = mujoco.MjData(m)
robot = env.scene["robot"]
q0 = robot.data.default_joint_pos[0].cpu().numpy()
jadr = [m.jnt_qposadr[m.joint(n).id] for n in [f"robot/{x}" for x in robot.joint_names]]
sid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): i for i in range(m.nsensor)}
read = lambda n: d.sensordata[m.sensor_adr[sid[n]]:m.sensor_adr[sid[n]] + m.sensor_dim[sid[n]]].copy()  # noqa: E731
site = m.sensor_objid[sid["robot/imu_upvector"]]
X180 = Rot.from_euler("x", 180, degrees=True)

# schedule: per axis 0 -> +A, hold, +A -> -A, hold, -A -> 0, hold
A, rate, hold = a.deg, a.rate, a.hold
seg = []
for ax, name in (("x", "X axis: tilt right side down (+) / left side down (-)"),
                 ("y", "Y axis: tilt forward, nose down (+) / backward (-)"),
                 ("z", "Z axis: turn left, CCW from above (+) / right (-)")):
  seg += [(ax, name, 0, 0, 1.5), (ax, name, 0, A, A / rate), (ax, name, A, A, hold), (ax, name, A, -A, 2 * A / rate),
          (ax, name, -A, -A, hold), (ax, name, -A, 0, A / rate), (ax, name, 0, 0, 1.0)]
total = sum(s[4] for s in seg)
W, H, PW = 720, 600, 560
ren = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
# per-axis view: x (roll) seen from the front, y (pitch) from the right side, z (yaw) from above
VIEW = {"x": (1.5, 180.0, -10.0), "y": (1.5, 90.0, -10.0), "z": (1.5, 180.0, -60.0)}
for gi in range(m.ngeom):  # see-through robot so the arrows at the IMU stay visible
  if m.geom_bodyid[gi] > 0 and m.geom_rgba[gi][3] > 0:
    m.geom_rgba[gi][3] = 0.35
font = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)  # noqa: E731
fontb = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)  # noqa: E731
mono = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", s)  # noqa: E731
C = dict(x=(214, 69, 65), y=(47, 154, 76), z=(47, 111, 214), sim=(222, 140, 20), pol=(214, 40, 120),
         ink=(30, 34, 40), mut=(110, 118, 130), bad=(205, 30, 30), bg=(246, 247, 249), line=(210, 214, 220))


def arrow(scene, p0, p1, rgba, width=0.012):
  if scene.ngeom >= scene.maxgeom:
    return
  g = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64))
  scene.ngeom += 1


def panel(info):
  img = Image.new("RGB", (PW, H), C["bg"]); dr = ImageDraw.Draw(img)
  y = 16
  dr.text((18, y), info["title"], font=fontb(17), fill=C["ink"]); y += 30
  dr.text((18, y), f"angle {info['ang']:+6.1f} deg    rate {info['rate']:+5.1f} deg/s    t = {info['t']:5.2f} s", font=mono(15), fill=C["ink"]); y += 34

  def block(title, vals, color, ref=None, scale=1.0, unit=""):
    nonlocal y
    dr.text((18, y), title, font=fontb(15), fill=color); y += 24
    for i, k in enumerate("xyz"):
      v = vals[i]
      bad = ref is not None and abs(v - ref[i]) > 2e-3
      col = C["bad"] if bad else C["ink"]
      dr.text((30, y), f"{k}", font=mono(15), fill=C[k])
      dr.text((52, y), f"{v:+.3f}{unit}", font=mono(15), fill=col)
      cx, wbar = 330, 190
      dr.line((cx, y + 3, cx, y + 17), fill=C["line"], width=1)
      L = max(-1, min(1, v / scale)) * wbar / 2
      dr.rectangle((min(cx, cx + L), y + 5, max(cx, cx + L), y + 15), fill=col if bad else color)
      if bad:
        dr.text((440, y), "OPPOSITE", font=fontb(13), fill=C["bad"])
      y += 22
    y += 10

  block("Sim projected gravity  (policy obs 3-5)", info["g_sim"], C["sim"])
  block("HUPHY projected gravity (predicted, obs 3-5)", info["g_hup"], C["pol"], ref=info["g_sim"])
  block("Gyro, sim and real  (policy obs 0-2, rad/s)", info["w"], C["ink"], scale=0.5)
  dr.text((18, y), "Attitude angles (deg)          roll    pitch    yaw", font=mono(14), fill=C["mut"]); y += 22
  dr.text((18, y), f"sim pelvis (x fwd,y left,z up) {info['rpy'][0]:+6.1f}  {info['rpy'][1]:+6.1f}  {info['rpy'][2]:+6.1f}", font=mono(14), fill=C["ink"]); y += 20
  dr.text((18, y), f"EBIMU display (x fwd,y right,z dn){info['erpy'][0]:+6.1f}  {info['erpy'][1]:+6.1f}  {info['erpy'][2]:+6.1f}", font=mono(14), fill=C["ink"]); y += 30
  dr.text((18, H - 66), "Kinematic demo, no physics. Real time: rotation at the rate shown.", font=font(13), fill=C["mut"])
  dr.text((18, H - 46), "HUPHY values: EBIMU datasheet frames + HUPHY gravity = R^T(0,0,-1),", font=font(13), fill=C["mut"])
  dr.text((18, H - 28), "sensor axes assumed = pelvis axes (mounting not yet verified).", font=font(13), fill=C["mut"])
  return img


tmp = os.path.splitext(a.out)[0] + "_frames"
os.makedirs(tmp, exist_ok=True)
n = int(round(total * a.fps)); k = 0
t_seg0 = 0.0
for ax, name, th0, th1, dur in seg:
  nf = int(round(dur * a.fps))
  for i in range(nf):
    u = i / max(nf, 1)
    th = th0 + (th1 - th0) * u
    rate_now = (th1 - th0) / dur if dur > 0 else 0.0
    R = Rot.from_euler(ax, th, degrees=True)
    mujoco.mj_resetData(m, d)
    for adr, v in zip(jadr, q0): d.qpos[adr] = v
    d.qpos[0:3] = [0, 0, 1.25]
    qx, qy, qz, qw = R.as_quat(); d.qpos[3:7] = [qw, qx, qy, qz]
    wv = np.zeros(3); wv["xyz".index(ax)] = math.radians(rate_now); d.qvel[3:6] = wv
    mujoco.mj_forward(m, d)
    g_sim = -read("robot/imu_upvector"); gyro = read("robot/imu_ang_vel")
    Rs = Rot.from_matrix(d.site_xmat[site].reshape(3, 3))
    RE = X180 * Rs * X180
    ex, ey, ez, ew = RE.as_quat()
    g_hup = np.array([2 * (ew * ey - ex * ez), -2 * (ey * ez + ew * ex), 2 * (ex * ex + ey * ey) - 1])
    cam.distance, cam.azimuth, cam.elevation = VIEW[ax]
    cam.lookat[:] = d.site_xpos[site] + np.array([0, 0, -0.1])
    ren.update_scene(d, cam)
    p = d.site_xpos[site].copy(); Rm = d.site_xmat[site].reshape(3, 3)
    for j, key in enumerate("xyz"):
      arrow(ren.scene, p, p + 0.4 * Rm[:, j], [*(np.array(C[key]) / 255), 1], 0.014)
    arrow(ren.scene, p, p + 0.7 * np.array([0, 0, -1.0]), [*(np.array(C["sim"]) / 255), 1.0], 0.02)
    arrow(ren.scene, p, p + 0.7 * (Rm @ g_hup), [*(np.array(C["pol"]) / 255), 1.0], 0.02)
    left = Image.fromarray(ren.render())
    dl = ImageDraw.Draw(left)
    dl.rectangle((6, 6, 470, 76), fill=C["bg"])
    dl.text((14, 54), {"x": "view: from the front", "y": "view: from the robot's right side (front = right)", "z": "view: from above, front = toward camera"}[ax], font=font(14), fill=C["mut"])
    dl.text((14, 12), "x fwd (red)  y left (green)  z up (blue)", font=font(14), fill=C["ink"])
    dl.text((14, 32), "orange: true gravity (sim)   magenta: HUPHY gravity", font=font(14), fill=C["ink"])
    frame = Image.new("RGB", (W + PW, H)); frame.paste(left, (0, 0))
    frame.paste(panel(dict(title=name, ang=th, rate=rate_now, t=k / a.fps, g_sim=g_sim, g_hup=g_hup, w=gyro,
                           rpy=Rs.as_euler("ZYX", degrees=True)[::-1], erpy=RE.as_euler("ZYX", degrees=True)[::-1])), (W, 0))
    frame.save(f"{tmp}/{k:05d}.png"); k += 1
subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
print(f"wrote {a.out}: {k} frames at {a.fps} fps = {k / a.fps:.2f} s (schedule {total:.2f} s)")
env.close()
