#!/usr/bin/env python3
"""벤치 **실영상**과 MuJoCo 를 나란히 붙인다. 축에 붙인 나무 막대로 시각까지 맞춘다.

2026-09-10 · 사용자가 출력축 허브에 나무 막대(인디케이터)를 붙여 주신 뒤에야 가능해졌습니다.
그 전에는 출력축이 대칭인 민무늬 원판이라 돌아도 화소가 안 바뀌어, 카메라가 회전을 아예
못 봤습니다.

무엇을 하는가
-------------
1. 영상의 모든 프레임에서 **막대의 각도**를 읽습니다(`track_pointer.py` — 나무의 붉은기를
   쓰고, 반지름 전체를 가로지르는 것만 고릅니다).
2. 그 각도와 **기록된 실제 각도**가 가장 잘 겹치도록 **프레임률·지연·각도 눈금 셋을 한꺼번에**
   맞춥니다. 프레임률까지 찾는 이유는, 카메라 스트림이 실제로는 8.5 fps 인데 받아 적는 쪽이
   15 로 가정해 시간축이 눌려 있었기 때문입니다(이것 때문에 처음에 안 맞았습니다).
3. 왼쪽 실영상 · 오른쪽 MuJoCo(질량 0 색 막대) · 아래 그래프로 합칩니다. 그래프에는
   **카메라가 잰 각도**도 같이 그립니다 — 기록이 실제 회전과 맞는다는 독립 확인입니다.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
      <!-- 실물에 붙인 나무 막대에 대응하는, 질량 0 인 막대. 동역학에 안 들어간다. -->
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


def dial(ax, ang_cmd, ang_real, ang_sim):
  ax.clear()
  ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
  ax.set_aspect("equal"); ax.axis("off")
  th = np.linspace(0, 2 * np.pi, 200)
  ax.plot(np.cos(th), np.sin(th), color="#cfcfcf", lw=1.2)
  for a in range(0, 360, 30):
    r = a * DEG
    ax.plot([0.93 * math.cos(r), math.cos(r)], [0.93 * math.sin(r), math.sin(r)],
            color="#e0e0e0", lw=1)
  for a, col, lw, lab, al in ((ang_cmd, "#8f8f8f", 3.2, "command", 0.9),
                              (ang_real, "#111111", 4.2, "real motor (recorded)", 0.55),
                              (ang_sim, "#00a878", 3.0, "MuJoCo", 0.85)):
    r = a * DEG
    ax.plot([0, math.cos(r)], [0, math.sin(r)], color=col, lw=lw, alpha=al,
            solid_capstyle="round", label=lab)
  ax.plot([0], [0], "o", color="#444", ms=7)
  ax.legend(fontsize=7.5, loc="lower center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, 0.015))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--cam", required=True)
  ap.add_argument("--csv", required=True)
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--cx", type=float, default=830); ap.add_argument("--cy", type=float, default=621)
  ap.add_argument("--r0", type=float, default=30); ap.add_argument("--r1", type=float, default=92)
  ap.add_argument("--box", default="740,470,930,700")
  ap.add_argument("--crop", default="700,450,1000,720")
  ap.add_argument("--kp", type=float, default=20.0); ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--out-fps", type=int, default=20)
  ap.add_argument("--gain", type=float, default=2.5)
  ap.add_argument("--out", default="../docs/assets/motor_video.mp4")
  a = ap.parse_args(argv)

  box = tuple(int(x) for x in a.box.split(","))
  crop = tuple(int(x) for x in a.crop.split(","))
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
  print(f"눈금: 실제각도 = {k:.4f} x 그림각도 + {b:.2f}")
  cam_deg = k * ang + b
  cam_t = np.arange(len(frames)) / fps - lag

  out_dir = tempfile.mkdtemp(prefix="cmp_")
  fig = plt.figure(figsize=(13.0, 7.4), dpi=100)
  gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0], width_ratios=[1.15, 1.0],
                        hspace=0.28, wspace=0.05, left=0.045, right=0.975,
                        top=0.905, bottom=0.085)
  ax_cam, ax_dial, ax_plot = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), \
      fig.add_subplot(gs[1, :])
  n_out = 0
  for tt in np.arange(t[0], t[-1], 1.0 / a.out_fps):
    fi = int(round((tt + lag) * fps))
    if fi < 0 or fi >= len(frames):
      continue
    c = float(np.interp(tt, t, tgt)); rr = float(np.interp(tt, t, pos))
    ss = float(np.interp(tt, t, sim))
    ax_cam.clear(); ax_cam.axis("off")
    ax_cam.imshow(Image.open(frames[fi]).crop(crop))
    ax_cam.set_title("the real motor, filmed  (wooden pointer on the shaft)", fontsize=10)
    dial(ax_dial, c * a.gain, rr * a.gain, ss * a.gain)
    ax_dial.set_title("MuJoCo, massless pointer on the shaft\n"
                      f"angles drawn {a.gain:g}x so they are visible", fontsize=9)
    ax_dial.text(0.5, -0.045,
                 f"command {c:+6.1f}      real {rr:+6.1f}      MuJoCo {ss:+6.1f}\n"
                 f"MuJoCo minus real:  {ss - rr:+.2f} deg", transform=ax_dial.transAxes,
                 ha="center", va="top", fontsize=10)
    ax_plot.clear()
    ax_plot.plot(t, tgt, color="#8f8f8f", lw=1.1, ls=(0, (6, 3)), label="command")
    ax_plot.plot(cam_t, cam_deg, color="#c0392b", lw=1.0, alpha=0.6,
                 label="angle read off the film")
    ax_plot.plot(t, pos, color="#111111", lw=1.4, alpha=0.55, label="real motor (recorded)")
    ax_plot.plot(t, sim, color="#00a878", lw=1.3, alpha=0.85, label="MuJoCo")
    ax_plot.axvline(tt, color="#c0392b", lw=1.2)
    ax_plot.set_xlim(t[0], t[-1]); ax_plot.set_ylim(-42, 42)
    ax_plot.set_xlabel("time [s]"); ax_plot.set_ylabel("angle [deg]")
    ax_plot.grid(alpha=0.3); ax_plot.legend(fontsize=8, loc="upper right", ncol=4)
    p = BEST[a.family]
    fig.suptitle(f"RobStride {a.family}: the bench and MuJoCo given the same command   |   "
                 f"inertia {p['armature']} · friction {p['frictionloss']} · "
                 f"viscous {p['damping']}", fontsize=11.5)
    fig.savefig(os.path.join(out_dir, f"{n_out:05d}.png"))
    n_out += 1
  plt.close(fig)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.out_fps),
                  "-i", os.path.join(out_dir, "%05d.png"), "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
  shutil.rmtree(fdir, ignore_errors=True); shutil.rmtree(out_dir, ignore_errors=True)
  print(f"합성 {n_out}장 → 영상: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
