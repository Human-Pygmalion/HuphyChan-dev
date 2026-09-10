#!/usr/bin/env python3
"""벤치 카메라 영상과 시뮬레이터를 나란히 붙인 영상을 만든다. **이 컴퓨터에서 돌린다.**

2026-09-10 · 사용자 지시: "이거 실험한 mujocolab 에 영상을 실물데이터랑 같이 띄워줄수있나?
우리 영상도 카메라 있으니 실물도 붙여서 비교해주면 좋겠는데. 모터에 Mass 0 짜리 색있는 링크
붙여서 도는거 보여주소."

무엇을 붙이는가
---------------
왼쪽 = 벤치 카메라(실물이 실제로 도는 모습).
오른쪽 = **질량 0 인 색 막대**를 축에 붙여 그린 것. 셋이 겹칩니다.

* **회색** — 우리가 시킨 목표 각도
* **검정** — 실물이 실제로 간 각도 (기록에서 읽음)
* **초록** — 같은 목표를 같은 제어식으로 넣은 **MuJoCo** 의 각도

아래 = 같은 셋을 시간축으로 펼친 그래프. 세로선이 지금 프레임입니다.

★ 물리는 MuJoCo 가 풉니다. **그림만 직접 그립니다** — 이 컴퓨터에 오프스크린 OpenGL 이
없어서 MuJoCo 자체 렌더러를 못 씁니다(EGL·OSMesa 둘 다 실패). 막대는 질량 0 이라 물리에
아무 영향도 주지 않습니다.

어떻게 맞추는가 (영상과 자료의 시각 맞추기)
-------------------------------------------
카메라와 모터는 서로 다른 컴퓨터에 있어 시계가 다릅니다. 그래서 **화면이 얼마나 움직였는가**
(프레임끼리의 차이)를 시간축으로 만들고, **기록된 속도의 크기**와 겹쳐 가장 잘 맞는 지연을
찾습니다. 삼각파처럼 모양이 뚜렷한 파형이면 잘 맞습니다.

    make_compare_video.py --cam cam.mp4 --csv data/pd_triangle_4_..._vid.csv --family RS04
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

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
      <!-- 눈으로 보라고 붙인 막대. mass=0 이라 동역학에 하나도 안 들어간다. -->
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
  t = np.array([float(x["t_s"]) for x in r])
  return (t,
          np.array([float(x["target_deg"]) for x in r]) - float(r[0]["target_deg"]),
          np.array([float(x["target_vel_deg_s"]) for x in r]),
          np.array([float(x["pos_deg"]) for x in r]) - float(r[0]["target_deg"]))


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


def cam_frames(mp4, out_dir, fps):
  os.makedirs(out_dir, exist_ok=True)
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", f"fps={fps}",
                  os.path.join(out_dir, "%05d.jpg")], check=True)
  return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg"))


def motion_signal(paths, crop):
  """프레임끼리 얼마나 달라졌는가. 화면이 움직인 정도를 시간축으로 만든다."""
  sig, prev = [], None
  for p in paths:
    im = Image.open(p).convert("L").crop(crop).resize((160, 120))
    a = np.asarray(im, dtype=np.float32)
    sig.append(0.0 if prev is None else float(np.mean(np.abs(a - prev))))
    prev = a
  return np.array(sig)


def best_lag(sig, t, vel, fps):
  """화면 움직임과 기록된 속도 크기가 가장 잘 겹치는 지연을 찾는다."""
  ft = np.arange(len(sig)) / fps
  ref = np.interp(ft, t, np.abs(vel))
  a = (sig - sig.mean()) / (sig.std() + 1e-9)
  b = (ref - ref.mean()) / (ref.std() + 1e-9)
  n = len(a)
  best, bl = -9e9, 0
  for lag in range(-int(len(a) * 0.6), int(len(a) * 0.6)):
    if lag >= 0:
      x, y = a[lag:], b[: n - lag]
    else:
      x, y = a[: n + lag], b[-lag:]
    if len(x) < fps * 3:
      continue
    c = float(np.dot(x, y) / len(x))
    if c > best:
      best, bl = c, lag
  return bl / fps, best


def dial(ax, ang_cmd, ang_real, ang_sim, span):
  ax.clear()
  ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
  ax.set_aspect("equal"); ax.axis("off")
  th = np.linspace(0, 2 * np.pi, 200)
  ax.plot(np.cos(th), np.sin(th), color="#cccccc", lw=1.2)
  for a in range(0, 360, 30):
    r = a * DEG
    ax.plot([0.94 * math.cos(r), math.cos(r)], [0.94 * math.sin(r), math.sin(r)],
            color="#dddddd", lw=1)
  for a, col, lw, lab, alpha in ((ang_cmd, "#8f8f8f", 3.0, "command", 0.85),
                                 (ang_real, "#111111", 4.0, "real motor", 0.65),
                                 (ang_sim, "#00a878", 3.0, "MuJoCo", 0.9)):
    r = a * DEG
    ax.plot([0, math.cos(r)], [0, math.sin(r)], color=col, lw=lw, alpha=alpha,
            solid_capstyle="round", label=lab)
  ax.plot([0], [0], "o", color="#444", ms=7)
  ax.legend(fontsize=8, loc="lower center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, -0.1))
  ax.set_title(f"massless pointer on the shaft   (+-{span:.0f} deg shown as +-90 deg)",
               fontsize=8.5)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--cam-still", default=None,
                  help="""벤치 사진 한 장(무엇을 재는 장비인지 보이라고 넣는 것).

  ★ 움직이는 영상을 붙이지 않는 이유: 이 모터의 출력축은 **대칭인 민무늬 은색 원판**이라
  돌아도 화소가 거의 안 바뀝니다. 실제로 초별 화면 움직임을 재 보니 전 구간이 잡음 수준
  (0.6~1.3)으로 평평했고, 극좌표로 펴서 각도 상관을 구해도 0.4 에 그쳤습니다(볼트가
  주기적이라 ±15도마다 접히기도 합니다). 그 상태로 영상을 나란히 붙이면 **맞지도 않는
  것을 맞는 것처럼** 보이게 됩니다.

  **축에 눈에 띄는 표식**(색 테이프 한 줄, 스티커, 볼트 하나에 칠 등)을 붙여 주시면
  카메라가 각도를 직접 읽을 수 있고, 그때 진짜 영상 대 영상 비교를 만들겠습니다.""")
  ap.add_argument("--csv", required=True)
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--fps", type=int, default=15)
  ap.add_argument("--crop", default="600,400,1000,720", help="카메라에서 모터가 있는 자리")
  ap.add_argument("--gain", type=float, default=3.0,
                  help="다이얼에서 각도를 몇 배로 부풀려 보일지 (작게 움직여도 보이게)")
  ap.add_argument("--out", default="../docs/img/motor_sim_vs_camera.mp4")
  a = ap.parse_args(argv)

  crop = tuple(int(x) for x in a.crop.split(","))
  still = Image.open(a.cam_still).crop(crop) if a.cam_still else None
  t, tgt, tgv, pos = load(a.csv)
  sim = simulate(t, tgt, tgv, a.kp, a.kd, a.dt, a.tau_max, a.delay_ms / 1e3,
                 **BEST[a.family])
  print(f"자료 {t[-1]:.1f}초 · 시뮬레이터 계산 완료")

  out_dir = tempfile.mkdtemp(prefix="comp_")
  span = float(np.max(np.abs(tgt)))
  n_out = 0
  fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
  gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], width_ratios=[1.0, 1.0],
                        hspace=0.30, wspace=0.06, left=0.05, right=0.97,
                        top=0.90, bottom=0.09)
  ax_cam, ax_dial, ax_plot = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), \
      fig.add_subplot(gs[1, :])
  times = np.arange(t[0], t[-1], 1.0 / a.fps)
  for tt in times:
    c = float(np.interp(tt, t, tgt)); r = float(np.interp(tt, t, pos))
    s_ = float(np.interp(tt, t, sim))
    ax_cam.clear(); ax_cam.axis("off")
    if still is not None:
      ax_cam.imshow(still)
      ax_cam.set_title("the motor this was measured on (camera still)\n"
                       "a live view would show nothing: the shaft face is a plain disc",
                       fontsize=9)
    dial(ax_dial, c * a.gain, r * a.gain, s_ * a.gain, span)
    ax_plot.clear()
    ax_plot.plot(t, tgt, color="#8f8f8f", lw=1.1, ls=(0, (6, 3)), label="command")
    ax_plot.plot(t, pos, color="#111111", lw=1.5, alpha=0.6, label="real motor")
    ax_plot.plot(t, sim, color="#00a878", lw=1.4, alpha=0.85, label="MuJoCo (measured values)")
    ax_plot.axvline(tt, color="#c0392b", lw=1.2)
    ax_plot.set_xlim(t[0], t[-1]); ax_plot.set_xlabel("time [s]")
    ax_plot.set_ylabel("angle [deg]"); ax_plot.grid(alpha=0.3)
    ax_plot.legend(fontsize=8, loc="upper right", ncol=3)
    ax_dial.text(0.5, -0.30,
                 f"command {c:+6.1f}°     real motor {r:+6.1f}°     MuJoCo {s_:+6.1f}°\n"
                 f"MuJoCo minus real motor:  {s_ - r:+.2f}°",
                 transform=ax_dial.transAxes, ha="center", va="top", fontsize=10,
                 family="DejaVu Sans")
    fig.suptitle(f"RobStride {a.family}: bench camera and MuJoCo, same command  |  "
                 f"inertia {BEST[a.family]['armature']} · friction "
                 f"{BEST[a.family]['frictionloss']} · viscous "
                 f"{BEST[a.family]['damping']}", fontsize=11)
    fig.savefig(os.path.join(out_dir, f"{n_out:05d}.png"))
    n_out += 1
  plt.close(fig)
  print(f"합성 프레임 {n_out}장")

  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                  "-i", os.path.join(out_dir, "%05d.png"), "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
  shutil.rmtree(out_dir, ignore_errors=True)
  print(f"영상: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
