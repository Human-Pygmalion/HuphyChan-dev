#!/usr/bin/env python3
"""벤치 **실영상**과 **MuJoCo 의 진짜 3D 그림**을 나란히 붙인다.

2026-09-10 · 사용자가 출력축 허브에 나무 막대(인디케이터)를 붙여 주신 뒤에야 가능해졌습니다.
그 전에는 출력축이 대칭인 민무늬 원판이라 돌아도 화소가 안 바뀌어, 카메라가 회전을 아예
못 봤습니다.

★ 앞선 판 정정 — 이 판에서 바뀐 것
-----------------------------------
이전 판은 오른쪽 그림을 **제가 손으로 그린 시계 눈금**으로 냈고, 그 이유를 "이 컴퓨터에
오프스크린 OpenGL 이 없다"고 적었습니다. **그 진단이 틀렸습니다.** `MUJOCO_GL=egl` 로
MuJoCo 자체 렌더러가 정상 동작합니다(OpenGL 4.6, NVIDIA). 실패로 본 것은 **렌더가 끝난 뒤
소멸자에서 나는 뒷정리 오류**였고, 그림 자체는 잘 나왔습니다. 그래서 이 판은
`motor_scene.py` 의 **진짜 3차원 렌더**를 씁니다.

무엇을 하는가
-------------
1. 영상의 모든 프레임에서 **막대의 각도**를 읽습니다(`track_pointer.py` — 나무의 붉은기를
   쓰고, 반지름 전체를 가로지르는 것만 고릅니다).
2. 그 각도와 **기록된 실제 각도**가 가장 잘 겹치도록 **프레임률·지연·각도 눈금 셋을 한꺼번에**
   맞춥니다. 프레임률까지 찾는 이유는, 카메라 스트림이 실제로는 8.6 fps 인데 받아 적는 쪽이
   15 로 가정해 시간축이 눌려 있었기 때문입니다(이것 때문에 처음에 안 맞았습니다).
3. 왼쪽 실영상 · 오른쪽 MuJoCo 3차원 그림 · 아래 그래프를 **빈 곳 없이** 이어 붙입니다.
   그래프에는 **카메라가 잰 각도**도 같이 그립니다 — 기록이 실제 회전과 맞는다는 독립 확인입니다.

두 그림의 각도를 어떻게 맞췄나
------------------------------
카메라 각도 맞추기에서 나온 관계가 ``실제각도 = 1.308 x 그림각도 + 117.7`` 입니다. 여기서
실제각도 0 일 때 그림각도가 **-90도**, 즉 막대가 **화면에서 위**를 가리킵니다. 그리고 실제
각도가 커지면 그림에서 **시계 방향**으로 돕니다.

그래서 MuJoCo 카메라를 방위각 90도(가로축 +x 가 화면 오른쪽, +y 가 화면 위)로 두고,
관절 각도를 ``90도 - 실제각도`` 로 넣습니다. 그러면 실제각도 0 에서 막대가 위를 보고,
각도가 커지면 시계 방향으로 돌아 **실영상과 같은 방향**이 됩니다. 카메라 높이(내려보는 각)도
실영상과 비슷하게 46도로 맞췄습니다.
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import math
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motor_scene import MotorScene  # noqa: E402
from track_pointer import pointer_angle  # noqa: E402

DEG = math.pi / 180.0

XML = """
<mujoco model="one_motor">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast" timestep="{dt}">
    <flag contact="disable" gravity="disable"/>
  </option>
  <worldbody>
    <body name="rotor">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j" type="hinge" axis="0 0 1" limited="false"
             armature="{armature}" damping="{damping}" frictionloss="{frictionloss}"/>
      <geom name="pointer" type="box" size="0.10 0.012 0.006" pos="0.10 0 0"
            mass="0" rgba="0 0.66 0.47 1"/>
    </body>
  </worldbody>
  <actuator><motor name="tau" joint="j" gear="1" ctrlrange="-200 200"/></actuator>
</mujoco>
"""

BEST = {
  "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
  "RS03": dict(armature=0.0638, damping=0.0252, frictionloss=0.251),
}

# 화면 구성 (화소). 위에서부터 제목띠 / 실영상+3차원 / 그래프. 세 덩어리가 딱 붙어서
# 빈 곳이 남지 않는다. 전체 1600 x 900.
W, H_HEAD, H_TOP, H_PLOT = 1600, 46, 520, 334
COL_CMD, COL_REAL, COL_SIM = "#8f8f8f", "#111111", "#00a878"
COL_CAM = "#c0392b"


def font(sz, bold=False):
  from matplotlib import font_manager
  name = "DejaVu Sans:style=normal:weight=" + ("bold" if bold else "normal")
  return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(
      family="DejaVu Sans", weight="bold" if bold else "normal")), sz)


def load(path):
  with open(path) as f:
    r = list(csv.DictReader(f))
  off = float(r[0]["target_deg"])
  return (np.array([float(x["t_s"]) for x in r]),
          np.array([float(x["target_deg"]) for x in r]) - off,
          np.array([float(x["target_vel_deg_s"]) for x in r]),
          np.array([float(x["pos_deg"]) for x in r]) - off)


def simulate(t, tgt, tgv, kp, kd, dt, tau_max, delay_s, **prm):
  m = mujoco.MjModel.from_xml_string(XML.format(dt=dt, **prm))
  d = mujoco.MjData(m)
  n = int((t[-1] - t[0]) / dt)
  st = t[0] + np.arange(n) * dt
  qd = np.interp(st, t, tgt) * DEG
  vd = np.interp(st, t, tgv) * DEG
  lag = max(1, int(round(delay_s / dt)))
  hq, hv = [0.0] * lag, [0.0] * lag
  out = np.empty(n)
  for i in range(n):
    tau = kp * (qd[i] - hq[0]) + kd * (vd[i] - hv[0])
    d.ctrl[0] = float(np.clip(tau, -tau_max, tau_max))
    mujoco.mj_step(m, d)
    hq.append(d.qpos[0]); hq.pop(0)
    hv.append(d.qvel[0]); hv.pop(0)
    out[i] = d.qpos[0] / DEG
  return np.interp(t, st, out)


def fit_sync(ang, t, pos):
  """프레임률·지연·각도 눈금을 한꺼번에 찾는다."""
  n = len(ang)
  best = (-9.0, 15.0, 0.0, 1.0, 0.0)
  for fps in np.arange(5.0, 16.05, 0.05):
    tc = np.arange(n) / fps
    for lag in np.arange(0.0, 18.0, 0.05):
      tt = tc - lag
      m = (tt >= t[0]) & (tt <= t[-1])
      if m.sum() < 120:
        continue
      ref = np.interp(tt[m], t, pos)
      A = np.vstack([ang[m], np.ones(m.sum())]).T
      sol, *_ = np.linalg.lstsq(A, ref, rcond=None)
      r2 = 1 - np.sum((ref - A @ sol) ** 2) / np.sum((ref - ref.mean()) ** 2)
      if r2 > best[0]:
        best = (float(r2), float(fps), float(lag), float(sol[0]), float(sol[1]))
  return best


def crop_box(cx, cy, w_img, h_img, aspect, h=235.0):
  """가운데를 잡고 **패널과 같은 가로세로비**로 잘라낸다. 늘어나 보이지 않게."""
  w = h * aspect
  x0, x1 = cx - w / 2, cx + w / 2
  y0, y1 = cy - h * 0.66, cy + h * 0.34
  x0, x1 = max(0, x0), min(w_img, x1)
  y0, y1 = max(0, y0), min(h_img, y1)
  return (int(x0), int(y0), int(x1), int(y1))


def make_plot(t, tgt, pos, sim, cam_t, cam_deg, ylim):
  """그래프를 **한 번만** 그려 화소로 들고 있는다. 프레임마다 바뀌는 것은 시간 막대뿐이라,
  매번 다시 그리면 시간만 버린다."""
  fig = plt.figure(figsize=(W / 100.0, H_PLOT / 100.0), dpi=100)
  ax = fig.add_axes([0.040, 0.175, 0.950, 0.735])
  ax.plot(t, tgt, color=COL_CMD, lw=1.2, ls=(0, (6, 3)), label="command we sent")
  ax.plot(cam_t, cam_deg, color=COL_CAM, lw=1.0, alpha=0.55,
          label="angle read off the film (independent check)")
  ax.plot(t, pos, color=COL_REAL, lw=1.5, alpha=0.60, label="real motor (recorded)")
  ax.plot(t, sim, color=COL_SIM, lw=1.4, alpha=0.90, label="MuJoCo with our measured values")
  ax.set_xlim(t[0], t[-1]); ax.set_ylim(*ylim)
  ax.set_xlabel("time [s]", fontsize=10)
  ax.set_ylabel("shaft angle [deg]", fontsize=10)
  ax.grid(alpha=0.28)
  ax.legend(fontsize=9, loc="upper right", ncol=4, framealpha=0.85)
  fig.canvas.draw()
  rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
  bb = ax.get_window_extent()
  x_of = lambda tt: int(round(float(ax.transData.transform((tt, 0.0))[0])))
  rows = (int(H_PLOT - bb.y1), int(H_PLOT - bb.y0))
  plt.close(fig)
  return rgb, x_of, rows


def label(img, xy, text, sz=15, bold=False, fg=(30, 30, 34), bg=(255, 255, 255, 205)):
  d = ImageDraw.Draw(img, "RGBA")
  f = font(sz, bold)
  x, y = xy
  tb = d.textbbox((x, y), text, font=f)
  d.rectangle((tb[0] - 7, tb[1] - 5, tb[2] + 7, tb[3] + 5), fill=bg)
  d.text((x, y), text, font=f, fill=fg)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--cam", required=True)
  ap.add_argument("--csv", required=True)
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--cx", type=float, default=830); ap.add_argument("--cy", type=float, default=621)
  ap.add_argument("--r0", type=float, default=30); ap.add_argument("--r1", type=float, default=92)
  ap.add_argument("--box", default="740,470,930,700")
  ap.add_argument("--kp", type=float, default=20.0); ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--out-fps", type=int, default=20)
  ap.add_argument("--phase0", type=float, default=90.0,
                  help="실제각도 0 일 때 3차원 막대가 놓일 자리. 90 = 화면에서 위")
  ap.add_argument("--flip", type=float, default=1.0,
                  help="1 이면 실제각도가 커질 때 화면에서 시계 방향 (실영상과 같은 방향)")
  ap.add_argument("--elev", type=float, default=-46.0, help="MuJoCo 카메라가 내려보는 각")
  ap.add_argument("--azim", type=float, default=90.0)
  ap.add_argument("--dist", type=float, default=0.352)
  ap.add_argument("--crop-h", type=float, default=235.0,
                  help="실영상에서 잘라낼 세로 화소 수. 작을수록 모터가 크게 보인다")
  ap.add_argument("--only-frame", type=float, default=None,
                  help="이 시각의 한 장만 만들어 눈으로 확인한다 (영상 안 만듦)")
  ap.add_argument("--out", default="../docs/assets/motor_video.mp4")
  a = ap.parse_args(argv)

  box = tuple(int(x) for x in a.box.split(","))
  t, tgt, tgv, pos = load(a.csv)
  sim = simulate(t, tgt, tgv, a.kp, a.kd, a.dt, a.tau_max, a.delay_ms / 1e3,
                 **BEST[a.family])

  fdir = tempfile.mkdtemp(prefix="vf_")
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.cam, "-vsync", "0",
                  os.path.join(fdir, "%05d.jpg")], check=True)
  frames = sorted(globmod.glob(os.path.join(fdir, "*.jpg")))
  ang = np.array([pointer_angle(p, a.cx, a.cy, a.r0, a.r1, box=box)[0] for p in frames])
  ang = np.degrees(np.unwrap(np.radians(ang)))
  r2, fps, lag, k, b = fit_sync(ang, t, pos)
  print(f"프레임 {len(frames)}장 · 실제 프레임률 {fps:.2f} fps · 지연 {lag:.2f}초 · "
        f"설명력 {r2:.3f}")
  print(f"눈금: 실제각도 = {k:.4f} x 그림각도 + {b:.2f}  "
        f"(→ 실제각도 0 일 때 그림에서 {-b / k:+.1f}도)")
  cam_deg = k * ang + b
  cam_t = np.arange(len(frames)) / fps - lag

  half = W // 2
  lim = float(max(np.abs(pos).max(), np.abs(tgt).max())) * 1.18
  plot_rgb, x_of, (row0, row1) = make_plot(t, tgt, pos, sim, cam_t, cam_deg, (-lim, lim))
  cb = crop_box(a.cx, a.cy, 1280, 720, half / H_TOP, a.crop_h)
  scene = MotorScene(width=half, height=H_TOP, azimuth=a.azim, elevation=a.elev,
                     distance=a.dist, lookat=(0.0, 0.0, 0.012))
  p = BEST[a.family]
  head_txt = (f"RobStride {a.family}:  the bench and MuJoCo, given the same command   |   "
              f"rotor inertia {p['armature']}  ·  dry friction {p['frictionloss']} N m  ·  "
              f"viscous {p['damping']} N m s")

  def compose(tt):
    fi = int(round((tt + lag) * fps))
    c = float(np.interp(tt, t, tgt)); rr = float(np.interp(tt, t, pos))
    ss = float(np.interp(tt, t, sim))
    canvas = Image.new("RGB", (W, H_HEAD + H_TOP + H_PLOT), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, W, H_HEAD), fill=(28, 30, 36))
    d.text((16, 13), head_txt, font=font(15, True), fill=(238, 238, 242))

    cam = Image.open(frames[fi]).crop(cb).resize((half, H_TOP), Image.LANCZOS)
    label(cam, (12, 10), "the real motor, filmed  (wooden pointer taped to the shaft)", 15, True)
    canvas.paste(cam, (0, H_HEAD))

    ren = Image.fromarray(scene.render(a.phase0 - a.flip * ss,
                                       a.phase0 - a.flip * rr,
                                       a.phase0 - a.flip * c))
    label(ren, (12, 10), "MuJoCo, rendered in 3D  (same camera angle)", 15, True)
    dr = ImageDraw.Draw(ren, "RGBA")
    f2 = font(14)
    y = H_TOP - 118
    dr.rectangle((10, y - 10, 348, H_TOP - 8), fill=(255, 255, 255, 205))
    for col, name, val in ((COL_SIM, "MuJoCo (green bar)", ss),
                           (COL_REAL, "real motor (black bar)", rr),
                           (COL_CMD, "command (grey bar)", c)):
      dr.rectangle((18, y + 4, 44, y + 14), fill=col)
      dr.text((54, y), f"{name:<24}{val:+7.2f} deg", font=f2, fill=(30, 30, 34))
      y += 26
    dr.text((18, y + 2), f"MuJoCo minus real: {ss - rr:+.2f} deg", font=font(14, True),
            fill=(30, 30, 34))
    canvas.paste(ren, (half, H_HEAD))
    d.line((half, H_HEAD, half, H_HEAD + H_TOP), fill=(255, 255, 255), width=2)

    strip = plot_rgb.copy()
    xp = x_of(tt)
    strip[row0:row1, max(0, xp - 1):xp + 2] = (192, 57, 43)
    canvas.paste(Image.fromarray(strip), (0, H_HEAD + H_TOP))
    return canvas

  if a.only_frame is not None:
    out = os.path.abspath(a.out)
    compose(a.only_frame).save(out)
    print(f"한 장만: {out}")
    shutil.rmtree(fdir, ignore_errors=True)
    return 0

  out_dir = tempfile.mkdtemp(prefix="cmp_")
  n_out = 0
  for tt in np.arange(t[0], t[-1], 1.0 / a.out_fps):
    fi = int(round((tt + lag) * fps))
    if fi < 0 or fi >= len(frames):
      continue
    compose(tt).save(os.path.join(out_dir, f"{n_out:05d}.png"))
    n_out += 1
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.out_fps),
                  "-i", os.path.join(out_dir, "%05d.png"), "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
  shutil.rmtree(fdir, ignore_errors=True); shutil.rmtree(out_dir, ignore_errors=True)
  print(f"합성 {n_out}장 → 영상: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
