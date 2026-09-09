#!/usr/bin/env python3
"""똑같은 위치 지령을 실물과 시뮬레이터에 넣고 응답을 겹친다. **이 컴퓨터에서 돌린다.**

2026-09-09. 사용자 지시: "실제로 모터를 그 파라미터로 만들어서, 똑같은 지령 넣어서 응답
비교해 줘. 계단 위치 입력(위아래 역전), 처프. 정지/동 마찰을 넣고 안 넣고도 비교하라."

무엇을 똑같이 맞추는가
----------------------
실물은 **호스트에서 PD 를 계산해 순수 토크로** 내보냈습니다(모터에 kp 를 실어 보내는 길이
지금 고장 나 있습니다 — `pd_profiles.py` 설명 참조). 시뮬레이터도 매 물리 걸음마다 같은
식을 계산합니다.

    tau = kp x (목표각 − 지금각) + kd x (목표속도 − 지금속도),  그 뒤 ±토크상한으로 자름

목표는 실물이 실제로 받은 것과 **같은 시각의 같은 값**을 기록에서 읽어 씁니다. 그러니 두
응답이 갈리면 원인은 **관성·마찰·점성**뿐입니다.

★ 되먹임 지연도 맞춥니다. 실물 고리는 명령을 보내고 응답이 오기까지 약 0.5 밀리초 걸리고,
그만큼 **낡은 상태**로 PD 를 계산합니다. 시뮬레이터에도 같은 지연을 넣습니다. 안 넣으면
시뮬레이터가 부당하게 유리합니다.

겹치는 네 벌
------------
1. **실물**
2. **지금 모형의 값** — armature 0.016333 / frictionloss 0.269456 / damping 0.009492
3. **오늘 실측** — armature 0.0773 / frictionloss 0.519 / damping 0.0275
4. **실측 관성인데 마찰을 뺀 것** — 마찰 항이 무엇을 하는지 보려고
5. **실측인데 마찰에 정지마찰 값(0.901)을 넣은 것** — MuJoCo 는 정지·동 마찰을 나눠 담을
   자리가 없어서, 둘 중 어느 값을 넣느냐가 응답을 얼마나 바꾸는지 봅니다
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

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
    </body>
  </worldbody>
  <actuator><motor name="tau" joint="j" gear="1" ctrlrange="-120 120"/></actuator>
</mujoco>
"""

VARIANTS = {
  "RS04": [
    ("model today",          "#c0392b", "--",
     dict(armature=0.016333, frictionloss=0.269456, damping=0.009492)),
    ("measured (all three)", "#16a085", "-",
     dict(armature=0.0773,   frictionloss=0.519,    damping=0.0275)),
    ("measured, friction removed", "#8e44ad", ":",
     dict(armature=0.0773,   frictionloss=0.0,      damping=0.0)),
    ("measured, but stiction value 0.901", "#e67e22", "-.",
     dict(armature=0.0773,   frictionloss=0.901,    damping=0.0275)),
  ],
  "RS03": [
    ("model today",          "#c0392b", "--",
     dict(armature=0.015265, frictionloss=0.285370, damping=0.022342)),
    ("measured (all three)", "#16a085", "-",
     dict(armature=0.0638,   frictionloss=0.400,    damping=0.0025)),
    ("measured, friction removed", "#8e44ad", ":",
     dict(armature=0.0638,   frictionloss=0.0,      damping=0.0)),
    ("measured armature, old friction", "#e67e22", "-.",
     dict(armature=0.0638,   frictionloss=0.285370, damping=0.022342)),
  ],
}


def load(path):
  with open(path) as f:
    r = list(csv.DictReader(f))
  t = np.array([float(x["t_s"]) for x in r])
  tgt = np.array([float(x["target_deg"]) for x in r])
  tgv = np.array([float(x["target_vel_deg_s"]) for x in r])
  pos = np.array([float(x["pos_deg"]) for x in r])
  return t, tgt, tgv, pos


def simulate(t, tgt_deg, tgv_deg, kp, kd, dt, p0_deg, tau_max, delay_s, **prm):
  m = mujoco.MjModel.from_xml_string(XML.format(dt=dt, **prm))
  d = mujoco.MjData(m)
  d.qpos[0] = p0_deg * DEG
  n = int((t[-1] - t[0]) / dt)
  ts = t[0] + np.arange(n) * dt
  qd = np.interp(ts, t, tgt_deg) * DEG
  vd = np.interp(ts, t, tgv_deg) * DEG
  lag = max(1, int(round(delay_s / dt)))
  hist_q = [d.qpos[0]] * lag
  hist_v = [0.0] * lag
  out = np.empty(n)
  for i in range(n):
    # 낡은 상태로 계산한다 (실물이 그렇게 하므로)
    tau = kp * (qd[i] - hist_q[0]) + kd * (vd[i] - hist_v[0])
    d.ctrl[0] = float(np.clip(tau, -tau_max, tau_max))
    mujoco.mj_step(m, d)
    hist_q.append(d.qpos[0]); hist_q.pop(0)
    hist_v.append(d.qvel[0]); hist_v.pop(0)
    out[i] = d.qpos[0] / DEG
  return ts, out


def gain_envelope(t, y, tgt, n_bins=55):
  edges = np.linspace(t[0], t[-1], n_bins + 1)
  fc, g = [], []
  for i in range(n_bins):
    s = (t >= edges[i]) & (t < edges[i + 1])
    if s.sum() < 20:
      continue
    a_in = (tgt[s].max() - tgt[s].min()) / 2
    if a_in < 1e-3:
      continue
    fc.append((edges[i] + edges[i + 1]) / 2)
    g.append((y[s].max() - y[s].min()) / 2 / a_in)
  return np.array(fc), np.array(g)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--step-csv", required=True)
  ap.add_argument("--chirp-csv", required=True)
  ap.add_argument("--family", default="RS04")
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--f0", type=float, default=0.2)
  ap.add_argument("--f1", type=float, default=10.0)
  ap.add_argument("--dt", type=float, default=0.0002)
  ap.add_argument("--zoom-s", type=float, default=3.0, help="계단 확대 구간 길이")
  ap.add_argument("--out", default="../docs/img/pd_sim_vs_real_rs04.png")
  a = ap.parse_args(argv)

  kp, kd = a.kp, a.kd            # 호스트에서 계산하므로 양자화 없음
  variants = VARIANTS[a.family]
  print(f"{a.family} · kp={kp} kd={kd} · 토크 상한 {a.tau_max} · 되먹임 지연 {a.delay_ms} ms")
  for lab, _c, _s, p in variants:
    fn = math.sqrt(kp / p["armature"]) / (2 * math.pi)
    z = (kd + p["damping"]) / (2 * math.sqrt(kp * p["armature"]))
    print(f"  {lab:36}: 고유진동수 {fn:5.2f} Hz · 감쇠비 {z:5.2f}")

  fig, axs = plt.subplots(2, 2, figsize=(16, 9.5))

  # --- 계단 --------------------------------------------------------------
  t, tgt, tgv, pos = load(a.step_csv)
  z0 = t[0] + (t[-1] - t[0]) * 0.45
  zs = (t >= z0) & (t <= z0 + a.zoom_s)
  axs[0][0].plot(t[zs], tgt[zs], "-", color="#7f8c8d", lw=1.6, label="commanded")
  axs[0][0].plot(t[zs], pos[zs], "-", color="#2c3e50", lw=2.0, label="real motor")
  sims = {}
  for lab, col, ls, p in variants:
    ts, y = simulate(t, tgt, tgv, kp, kd, a.dt, pos[0], a.tau_max, a.delay_ms / 1e3, **p)
    sims[lab] = (ts, y)
    zz = (ts >= z0) & (ts <= z0 + a.zoom_s)
    axs[0][0].plot(ts[zz], y[zz], ls, color=col, lw=1.6, label=f"sim: {lab}")
  axs[0][0].set_xlabel("time [s]"); axs[0][0].set_ylabel("angle [deg]")
  axs[0][0].set_title(f"STEP, alternating +-{abs(tgt).max()-abs(tgt).min():.0f} deg "
                      f"(zoom)  -  kp {kp:g} / kd {kd:g}")
  axs[0][0].legend(fontsize=8); axs[0][0].grid(alpha=0.3)

  # 계단마다의 정착 오차 (마찰이 남기는 것)
  edges = np.where(np.diff(tgt) != 0)[0]
  rows = []
  for lab, (ts, y) in [("real motor", (t, pos))] + [(k, v) for k, v in sims.items()]:
    errs = []
    for e in edges[:-1]:
      te = t[e]
      s = (ts > te + 0.9) & (ts < te + 1.4)          # 계단 후반 = 자리잡은 뒤
      if s.sum() > 5:
        errs.append(np.median(y[s] - np.interp(ts[s], t, tgt)))
    if errs:
      rows.append((lab, float(np.mean(np.abs(errs))), float(np.max(np.abs(errs)))))
  bar_col = {"real motor": "#2c3e50", **{lab: c for lab, c, _s, _p in variants}}
  axs[0][1].barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1],
                 color=[bar_col.get(r[0], "#888") for r in rows][::-1])
  for i, r in enumerate(rows[::-1]):
    axs[0][1].text(r[1], i, f"  {r[1]:.2f} deg", va="center", fontsize=9)
  axs[0][1].set_xlabel("steady-state error left at the end of each step [deg]")
  axs[0][1].set_title("STEP: what friction leaves behind\n"
                      "the offset the controller cannot close")
  axs[0][1].grid(alpha=0.3, axis="x")

  # --- 처프 --------------------------------------------------------------
  t2, tgt2, tgv2, pos2 = load(a.chirp_csv)
  D = t2[-1] - t2[0]
  k = math.log(a.f1 / a.f0) / D
  freq = lambda tt: a.f0 * math.exp(k * (tt - t2[0]))     # noqa: E731
  z1 = t2[0] + D * 0.55
  zs2 = (t2 >= z1) & (t2 <= z1 + 2.0)
  axs[1][0].plot(t2[zs2], tgt2[zs2], "-", color="#7f8c8d", lw=1.4, label="commanded")
  axs[1][0].plot(t2[zs2], pos2[zs2], "-", color="#2c3e50", lw=1.8, label="real motor")
  fc, g = gain_envelope(t2, pos2, tgt2)
  axs[1][1].semilogx([freq(x) for x in fc], g, "o-", color="#2c3e50", lw=2.0, ms=4,
                     label="real motor")
  for lab, col, ls, p in variants:
    ts, y = simulate(t2, tgt2, tgv2, kp, kd, a.dt, pos2[0], a.tau_max, a.delay_ms / 1e3, **p)
    zz = (ts >= z1) & (ts <= z1 + 2.0)
    axs[1][0].plot(ts[zz], y[zz], ls, color=col, lw=1.3, label=f"sim: {lab}")
    fc2, g2 = gain_envelope(ts, y, np.interp(ts, t2, tgt2))
    axs[1][1].semilogx([freq(x) for x in fc2], g2, ls, color=col, lw=1.8, label=f"sim: {lab}")
    axs[1][1].axvline(math.sqrt(kp / p["armature"]) / (2 * math.pi), color=col, ls=":",
                      lw=1.0, alpha=0.5)
  axs[1][1].axhline(1.0, color="#95a5a6", ls=":", lw=1.0)
  axs[1][0].set_xlabel("time [s]"); axs[1][0].set_ylabel("angle [deg]")
  axs[1][0].set_title(f"CHIRP {a.f0:g} to {a.f1:g} Hz (zoom near "
                      f"{freq(z1):.1f} Hz)")
  axs[1][0].legend(fontsize=8, ncol=2); axs[1][0].grid(alpha=0.3)
  axs[1][1].set_xlabel("frequency [Hz]")
  axs[1][1].set_ylabel("amplitude out / amplitude in")
  axs[1][1].set_title("CHIRP: how much of the command survives\n"
                      "dotted verticals = each model's natural frequency")
  axs[1][1].legend(fontsize=8); axs[1][1].grid(alpha=0.3, which="both")

  fig.suptitle(f"{a.family}: the same position command to the real motor and to a "
               f"one-joint MuJoCo model (host-side PD, {a.delay_ms:g} ms feedback delay)",
               fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print("\n계단 정착 오차 (작을수록 실물에 가까움):")
  for lab, mean, mx in rows:
    print(f"  {lab:36} 평균 {mean:5.2f}도 · 최대 {mx:5.2f}도")
  print(f"\n그림: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
