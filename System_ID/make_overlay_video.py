#!/usr/bin/env python3
"""실물 사진 **위에 시뮬레이터를 같은 크기로 겹쳐** 놓고, 오른쪽에 그래프를 붙인다. 16:9.

2026-09-10 · 사용자 지시: **"Sim / Real 을 같은 스케일로, 겹쳐서 보이게 알파 줘서 시각화하고,
오른쪽에 그래프 넣어서 16:9 그래프 겸 영상으로 넣어 줘."**

앞 판은 실물과 시뮬레이터를 **나란히** 놓았습니다. 나란히 놓으면 "둘 다 비슷하네" 까지는
보이지만 **얼마나 어긋나는지**는 눈으로 못 잽니다. 겹쳐 놓으면 어긋남이 그대로 각도 차이로
보입니다.

겹치려면 세 가지가 맞아야 합니다
--------------------------------
① **투영** — 사진은 위에서 거의 수직으로 내려다본 것이라, 시뮬레이터도 **직교 투영**(멀고
   가까움에 따라 크기가 안 변하는 방식)으로 바로 위에서 그립니다. 원근 투영이면 가장자리가
   벌어져 아무리 크기를 맞춰도 안 겹칩니다.
② **크기** — 시뮬레이터 원판 반지름이 **사진 속 원판 반지름과 같은 화소 수**가 되도록 그림을
   늘립니다. 배율은 시작할 때 한 번 재서 씁니다.
③ **각도** — 카메라가 완전히 수직은 아니라 원판이 살짝 타원으로 보입니다. 그래서 사진에서
   잰 각도와 실제 각도는 곧바로 같지 않고, 둘의 관계 ``실제 = k x 사진각 + b`` 를 자료로
   맞춰 씁니다. 시뮬레이터 막대는 그 관계를 **거꾸로** 써서 놓습니다 — 즉 "시뮬레이터 각도가
   이 값이라면 사진에서는 여기 보일 것"인 자리에 그립니다.

화면 (1920 x 1080)
------------------
    제목띠 1920 x 52
    왼쪽 1080 x 1028   실물 사진 + 겹친 시뮬레이터 (원판은 옅게, 막대는 진하게)
    오른쪽 840 x 1028  위 = 각도, 아래 = 실물과의 차이, 맨 아래 = 지금 값
"""

from __future__ import annotations

import argparse
import csv
import json
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
from mjlab_scene import MjlabJointScene  # noqa: E402

DEG = math.pi / 180.0
W, H = 1920, 1080
H_HEAD = 52
LEFT = 1080
RIGHT = W - LEFT
H_BODY = H - H_HEAD
H_LEG = 104                 # 왼쪽 칸 아래 설명띠
H_CAM = H_BODY - H_LEG      # 사진이 차지하는 높이

COL_CMD, COL_REAL, COL_SIM, COL_CAM = "#e8a020", "#111111", "#00cc8c", "#c0392b"

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
    </body>
  </worldbody>
  <actuator><motor name="tau" joint="j" gear="1" ctrlrange="-200 200"/></actuator>
</mujoco>
"""

BEST = {
  "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
  "RS03": dict(armature=0.0638, damping=0.0252, frictionloss=0.251),
}


def font(sz, bold=False):
  from matplotlib import font_manager
  return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(
      family="DejaVu Sans", weight="bold" if bold else "normal")), sz)


def load_motor(path):
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


def load_frames(fdir, meta_path):
  """찍은 그림과 **그 시각**. 시각은 커널이 프레임마다 찍어 준 값이고, 모터 기록과 같은
  시계다. 그래서 모터가 시작한 시각을 빼면 그것이 곧 모터 자료의 시간축이다."""
  t0 = float(json.load(open(meta_path))["t0_monotonic"])
  with open(os.path.join(fdir, "frames.csv")) as f:
    rows = list(csv.DictReader(f))
  paths, ts = [], []
  for r in rows:
    p = os.path.join(fdir, f"color_{int(r['idx']):05d}.jpg")
    if os.path.exists(p):
      paths.append(p)
      ts.append(float(r["t_mono_s"]) - t0)
  return paths, np.array(ts), t0


def load_angles(path, t0):
  with open(path) as f:
    rows = list(csv.DictReader(f))
  ang = np.degrees(np.unwrap(np.radians(
      np.array([float(r["angle_img_deg"]) for r in rows]))))
  return ang, np.array([float(r["t_mono_s"]) for r in rows]) - t0


def fit_scale(ang, ang_t, t, pos):
  """시각은 이미 정확하므로 **각도 눈금만** 맞춘다 (기울기와 치우침 둘)."""
  m = (ang_t >= t[0]) & (ang_t <= t[-1])
  ref = np.interp(ang_t[m], t, pos)
  A = np.vstack([ang[m], np.ones(int(m.sum()))]).T
  sol, *_ = np.linalg.lstsq(A, ref, rcond=None)
  r2 = 1 - np.sum((ref - A @ sol) ** 2) / np.sum((ref - ref.mean()) ** 2)
  return float(sol[0]), float(sol[1]), float(r2)


def make_plots(t, tgt, pos, sim, cam_t, cam_deg, family, prm):
  """오른쪽 그래프를 **한 번만** 그려 화소로 들고 있는다. 프레임마다 바뀌는 것은 시간 막대
  뿐이라, 매번 다시 그리면 시간만 버린다."""
  fig = plt.figure(figsize=(RIGHT / 100.0, H_BODY / 100.0), dpi=100)
  ax1 = fig.add_axes([0.115, 0.560, 0.865, 0.395])
  ax2 = fig.add_axes([0.115, 0.085, 0.865, 0.350])
  ax1.plot(t, tgt, color=COL_CMD, lw=1.3, ls=(0, (6, 3)), label="command we sent")
  ax1.plot(cam_t, cam_deg, color=COL_CAM, lw=1.0, alpha=0.55,
           label="angle read off the film")
  ax1.plot(t, pos, color=COL_REAL, lw=1.6, alpha=0.60, label="real motor (recorded)")
  ax1.plot(t, sim, color=COL_SIM, lw=1.5, alpha=0.95, label="MuJoCo, measured values")
  ax1.set_ylabel("shaft angle [deg]", fontsize=10)
  ax1.set_title(f"RobStride {family}: same command into the bench and into MuJoCo",
                fontsize=11)
  ax1.legend(fontsize=8.5, loc="upper right", ncol=2, framealpha=0.9)

  ax2.axhline(0, color="#999", lw=0.9)
  ax2.plot(t, sim - pos, color=COL_SIM, lw=1.4, alpha=0.95, label="MuJoCo minus real motor")
  ax2.plot(cam_t, cam_deg - np.interp(cam_t, t, pos), color=COL_CAM, lw=1.0, alpha=0.55,
           label="film minus real motor (how well the camera reads it)")
  ax2.set_xlabel("time [s]", fontsize=10)
  ax2.set_ylabel("difference [deg]", fontsize=10)
  ax2.set_title("how far apart they are", fontsize=11)
  ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
  for ax in (ax1, ax2):
    ax.set_xlim(t[0], t[-1]); ax.grid(alpha=0.28)
  lim = float(max(np.abs(pos).max(), np.abs(tgt).max())) * 1.20
  ax1.set_ylim(-lim, lim)
  d = float(np.nanmax(np.abs(sim - pos))) * 1.5 + 0.5
  ax2.set_ylim(-d, d)

  fig.canvas.draw()
  rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
  hh = rgb.shape[0]
  spans = []
  for ax in (ax1, ax2):
    bb = ax.get_window_extent()
    spans.append((int(hh - bb.y1), int(hh - bb.y0)))
  x_of = lambda tt: int(round(float(ax1.transData.transform((tt, 0.0))[0])))  # noqa: E731
  plt.close(fig)
  return rgb, x_of, spans


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--frames-dir", required=True, help="crops_to_jpg.py 가 만든 폴더")
  ap.add_argument("--angles", required=True, help="파이가 읽어 둔 angles.csv")
  ap.add_argument("--csv", required=True, help="모터 기록")
  ap.add_argument("--meta", default=None)
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--xml", default=("../mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/"
                                    "xmls/SemiFullDoF_prototype-tempmass-motormeasured-armfix"
                                    "_v30_proxyfix_loop_bench20260910.xml"),
                  help="**학습에 쓰는 그 로봇 파일**. 여기서 관절과 모터를 그대로 꺼내 그린다")
  ap.add_argument("--motor", default="4")
  # 사진 속 기하 (원본 프레임 좌표계)
  ap.add_argument("--hub", required=True, help="축 중심 x,y (원본 프레임 좌표)")
  ap.add_argument("--disc-r", type=float, required=True, help="사진 속 원판 반지름 [화소]")
  ap.add_argument("--ptr-reach", type=float, default=1.45,
                  help="실물 막대가 원판 반지름의 몇 배까지 뻗는가")
  ap.add_argument("--crop", required=True, help="잘라 낸 영역 x0,y0,x1,y1 (원본 프레임 좌표)")
  ap.add_argument("--alpha-disc", type=float, default=0.20)
  ap.add_argument("--alpha-ptr", type=float, default=0.70)
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--out-fps", type=int, default=25)
  ap.add_argument("--preview", type=float, default=None,
                  help="이 시각의 한 장만 만들어 눈으로 확인한다 (영상 안 만듦)")
  ap.add_argument("--out", default="../docs/assets/overlay.mp4")
  a = ap.parse_args(argv)

  hx, hy = (float(v) for v in a.hub.split(","))
  cx0, cy0, cx1, cy1 = (int(v) for v in a.crop.split(","))
  meta = a.meta or a.csv.replace(".csv", ".meta.json")
  prm = BEST[a.family]

  t, tgt, tgv, pos = load_motor(a.csv)
  sim = simulate(t, tgt, tgv, a.kp, a.kd, a.dt, a.tau_max, a.delay_ms / 1e3, **prm)
  frames, cam_t, t0 = load_frames(a.frames_dir, meta)
  ang, ang_t = load_angles(a.angles, t0)
  k, b, r2 = fit_scale(ang, ang_t, t, pos)
  cam_deg = k * ang + b
  print(f"프레임 {len(frames)}장 · 실제 초당 "
        f"{len(frames) / max(1e-9, cam_t[-1] - cam_t[0]):.2f}장 · "
        f"각도 눈금 실제 = {k:.4f} x 사진각 + {b:.2f} · 설명력 {r2:.3f}")
  print(f"  → 실제각도 0 일 때 사진에서 {-b / k:+.1f}도")

  # 사진 -> 왼쪽 칸으로 옮길 때의 배율과 자리.
  # **칸을 꽉 채우도록** 늘린 뒤 넘치는 만큼 가운데를 남기고 잘라 낸다. 비율을 그대로 두고
  # 넣으면 위아래(또는 좌우)에 검은 띠가 남는데, 사용자가 "빈 곳 없게" 라고 하셨다.
  cw, ch = cx1 - cx0, cy1 - cy0
  s = max(LEFT / cw, H_CAM / ch)                  # 사진 화소 -> 칸 화소 (덮기)
  fw, fh = int(round(cw * s)), int(round(ch * s))
  offx, offy = (fw - LEFT) // 2, (fh - H_CAM) // 2
  hub_px = ((hx - cx0) * s - offx, (hy - cy0) * s - offy)
  disc_r_px = a.disc_r * s

  scene = MjlabJointScene(a.xml, a.family, size=900, ptr_reach=a.ptr_reach,
                          cmd_reach=a.ptr_reach * 1.12)
  # 크기 맞추기: **모델의 모터 원통 반지름**을 사진 속 **도는 원판 반지름**에 맞춘다.
  # 모델의 모터는 원통 하나로 단순화돼 있어서 실물과 꼭 같은 부품은 아니므로, "도는 면끼리
  # 겹친다"를 규칙으로 정하고 그대로 적어 둔다.
  ov_scale = disc_r_px / scene.motor_px
  ov_size = int(round(scene.size * ov_scale))
  print(f"겹치기: 모델 모터 반지름 {scene.motor_r:.3f} m = {scene.motor_px:.0f}화소 → "
        f"사진 속 원판 {a.disc_r:.0f}화소(칸에서 {disc_r_px:.0f}) · {ov_scale:.3f}배")
  print(f"  장면-그림 각도 눈금: 장면 0도 → 그림 {scene.img_of_scene[0]:+.1f}도, "
        f"부호 {scene.img_of_scene[1]:+.0f}")

  def img_deg(true_deg):
    """실제 각도 -> **사진에서 보일 각도**. ``실제 = k x 사진각 + b`` 를 거꾸로 푼다."""
    return (true_deg - b) / k

  plot_rgb, x_of, spans = make_plots(t, tgt, pos, sim, ang_t, cam_deg, a.family, prm)
  head = (f"RobStride {a.family} (motor {a.motor})   |   mjlab's own 3D render, drawn on top "
          f"of the real motor at the same size   |   rotor inertia {prm['armature']}  ·  "
          f"friction {prm['frictionloss']} N m  ·  viscous {prm['damping']} N m s")

  def compose(tt):
    fi = int(np.argmin(np.abs(cam_t - tt)))
    c = float(np.interp(tt, t, tgt)); rr = float(np.interp(tt, t, pos))
    ss = float(np.interp(tt, t, sim))
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, W, H_HEAD), fill=(26, 28, 34))
    d.text((16, 16), head, font=font(15, True), fill=(238, 238, 242))

    # --- 왼쪽: 사진 위에 시뮬레이터를 얹는다 ---
    cam = (Image.open(frames[fi]).crop((0, 0, cw, ch)).resize((fw, fh), Image.LANCZOS)
           .crop((offx, offy, offx + LEFT, offy + H_CAM)))
    base = np.asarray(cam, dtype=np.float32)
    rgb, m_disc, m_ptr = scene.render(img_deg(ss), img_deg(rr), img_deg(c))
    ov = np.asarray(Image.fromarray(rgb).resize((ov_size, ov_size), Image.LANCZOS),
                    dtype=np.float32)
    md = np.asarray(Image.fromarray(m_disc.astype(np.uint8) * 255).resize(
        (ov_size, ov_size), Image.NEAREST)) > 127
    mp = np.asarray(Image.fromarray(m_ptr.astype(np.uint8) * 255).resize(
        (ov_size, ov_size), Image.NEAREST)) > 127
    ox = int(round(hub_px[0] - ov_size / 2)); oy = int(round(hub_px[1] - ov_size / 2))
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    ww = min(ov_size - sx0, LEFT - dx0); hh2 = min(ov_size - sy0, H_CAM - dy0)
    if ww > 0 and hh2 > 0:
      sub = (slice(sy0, sy0 + hh2), slice(sx0, sx0 + ww))
      dst = (slice(dy0, dy0 + hh2), slice(dx0, dx0 + ww))
      # 원판은 옅게, 막대는 진하게 얹는다. 원판까지 진하게 얹으면 정작 봐야 할 실물이 가려진다.
      for msk, al in ((md[sub], a.alpha_disc), (mp[sub], a.alpha_ptr)):
        blk = base[dst]
        blk[msk] = (1.0 - al) * blk[msk] + al * ov[sub][msk]
        base[dst] = blk
    left = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dl = ImageDraw.Draw(left, "RGBA")
    dl.ellipse((hub_px[0] - disc_r_px, hub_px[1] - disc_r_px,
                hub_px[0] + disc_r_px, hub_px[1] + disc_r_px),
               outline=(255, 255, 255, 150), width=2)
    canvas.paste(left, (0, H_HEAD))

    # 왼쪽 칸 아래 설명
    y = H_HEAD + H_CAM + 10
    dc = ImageDraw.Draw(canvas, "RGBA")
    dc.rectangle((0, H_HEAD + H_CAM, LEFT, H), fill=(20, 22, 28))
    f2 = font(15, True)
    for i, (col, lab) in enumerate((((0, 204, 140), "mjlab model, our measured numbers"),
                                    ((235, 235, 240), "the real motor (recorded angle)"),
                                    ((255, 184, 26), "the command"))):
      dc.rectangle((22, y + i * 26 + 4, 52, y + i * 26 + 16), fill=col)
      dc.text((64, y + i * 26), lab, font=f2, fill=(232, 232, 238))
    dc.text((520, y), f"MuJoCo   {ss:+7.2f} deg", font=f2, fill=(0, 220, 150))
    dc.text((520, y + 26), f"real     {rr:+7.2f} deg", font=f2, fill=(238, 238, 244))
    dc.text((520, y + 52), f"apart    {ss - rr:+7.2f} deg", font=font(16, True),
            fill=(255, 210, 120))
    dc.text((790, y + 26), f"t = {tt:5.2f} s", font=f2, fill=(180, 184, 196))

    # --- 오른쪽: 그래프 + 지금 시각 막대 ---
    strip = plot_rgb.copy()
    xp = x_of(tt)
    for (r0, r1) in spans:
      strip[r0:r1, max(0, xp - 1):xp + 2] = (192, 57, 43)
    canvas.paste(Image.fromarray(strip), (LEFT, H_HEAD))
    return canvas

  if a.preview is not None:
    out = os.path.abspath(a.out)
    compose(a.preview).save(out)
    print(f"한 장만: {out}")
    return 0

  out_dir = tempfile.mkdtemp(prefix="ov_")
  n = 0
  for tt in np.arange(max(t[0], cam_t[0]), min(t[-1], cam_t[-1]), 1.0 / a.out_fps):
    compose(tt).save(os.path.join(out_dir, f"{n:05d}.png"))
    n += 1
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.out_fps),
                  "-i", os.path.join(out_dir, "%05d.png"), "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
  shutil.rmtree(out_dir, ignore_errors=True)
  print(f"합성 {n}장 → 영상: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
