#!/usr/bin/env python3
"""가장 믿을 수 있는 값으로 **시킨 것 / 실물 / 시뮬레이터** 셋을 겹친다. 2×2, 모터마다 한 장.

2026-09-10 · 사용자 지시: "가장 정확하다고 생각하는 Inertia, Damping, frictionloss 값 활용해
Sim / 지령과 비교한 Plot 을 파형별로, 모터별로 2x2 로 그려줘."

한 칸에 셋이 겹칩니다.

* **회색 점선** — 우리가 시킨 목표
* **검정** — 실물이 실제로 간 곳 (10회 평균. 옅은 선이 개별 10회)
* **초록** — 같은 목표를 같은 제어식으로 넣은 시뮬레이터

쓰는 값 (2026-09-09/10 재측정, `docs/131`):

    RS04  관성 0.0773 · 점성 0.0275 · 마찰 0.519
    RS03  관성 0.0638 · 점성 0.0252 · 마찰 0.251

시뮬레이터에도 **같은 목표·같은 이득·같은 토크 상한·같은 되먹임 지연(0.5 밀리초)** 을 줍니다.
그래야 남는 차이가 이 세 값 때문이라고 말할 수 있습니다.
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
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
  <actuator><motor name="tau" joint="j" gear="1" ctrlrange="-200 200"/></actuator>
</mujoco>
"""

BEST = {
  "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
  "RS03": dict(armature=0.0638, damping=0.0252, frictionloss=0.251),
}


def load_runs(pattern):
  out = []
  for p in sorted(globmod.glob(pattern)):
    with open(p) as f:
      r = list(csv.DictReader(f))
    if len(r) < 200:
      continue
    out.append((np.array([float(x["t_s"]) for x in r]),
                np.array([float(x["target_deg"]) for x in r]),
                np.array([float(x["target_vel_deg_s"]) for x in r]),
                np.array([float(x["pos_deg"]) for x in r])))
  return out


def common_grid(runs, dt=0.001):
  tmax = min(r[0][-1] for r in runs)
  ts = np.arange(0, tmax, dt)
  P, T, V = [], None, None
  for t, tgt, tgv, pos in runs:
    off = tgt[0]                    # 회차마다 출발 자리가 다르므로 목표 원점을 뺀다
    P.append(np.interp(ts, t, pos - off))
    if T is None:
      T = np.interp(ts, t, tgt - off)
      V = np.interp(ts, t, tgv)
  return ts, T, V, np.array(P)


def simulate(ts, tgt, tgv, kp, kd, dt, tau_max, delay_s, **prm):
  m = mujoco.MjModel.from_xml_string(XML.format(dt=dt, **prm))
  d = mujoco.MjData(m)
  n = int(ts[-1] / dt)
  st = np.arange(n) * dt
  qd = np.interp(st, ts, tgt) * DEG
  vd = np.interp(st, ts, tgv) * DEG
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
  return np.interp(ts, st, out)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--tag", default="c")
  ap.add_argument("--amp", type=float, default=30.0)
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0002)
  ap.add_argument("--f0", type=float, default=0.1)
  ap.add_argument("--f1", type=float, default=10.0)
  ap.add_argument("--out", default="../docs/img/pd_2x2_rs04.png")
  a = ap.parse_args(argv)

  A, T = a.amp, a.tag
  # 2x2 로 넣을 네 파형. 각각 다른 값에 민감하도록 고른 것들이다.
  PANELS = [
    (f"STEP  +-{A:g} deg     (big move: inertia + friction together)",
     f"pd_step_{a.motor}_kp{a.kp:g}_kd{a.kd:g}_a{A:g}_r*_{T}.csv", (2.4, 5.4)),
    ("STEP  +-1.5 deg     (small move: inside the friction dead zone)",
     f"pd_step_{a.motor}_kp{a.kp:g}_kd{a.kd:g}_a1.5_r*_{T}.csv", (2.4, 5.4)),
    (f"CHIRP  {a.f0:g} to {a.f1:g} Hz     (slow to fast: inertia)",
     f"pd_chirp_{a.motor}_kp{a.kp:g}_kd{a.kd:g}_a{A:g}_r*_{T}.csv", None),
    ("TRIANGLE  steady speed     (constant speed: friction)",
     f"pd_triangle_{a.motor}_kp{a.kp:g}_kd{a.kd:g}_a{A:g}_r*_{T}.csv", (0.0, 9.5)),
  ]

  prm = BEST[a.family]
  fig, axs = plt.subplots(2, 2, figsize=(15.5, 9.0))
  print(f"{a.family} · 관성 {prm['armature']} · 점성 {prm['damping']} · "
        f"마찰 {prm['frictionloss']} · 이득 kp={a.kp} kd={a.kd}")

  for ax, (title, pat, zoom) in zip(axs.ravel(), PANELS):
    runs = load_runs(os.path.join(a.data, pat))
    if not runs:
      ax.text(0.5, 0.5, "no data:\n" + pat, ha="center", va="center",
              transform=ax.transAxes, fontsize=8, color="#999")
      ax.set_title(title, fontsize=10)
      continue
    ts, tgt, tgv, P = common_grid(runs)
    y = simulate(ts, tgt, tgv, a.kp, a.kd, a.dt, a.tau_max, a.delay_ms / 1e3, **prm)
    lo, hi = zoom if zoom else (ts[-1] * 0.78, ts[-1])
    z = (ts >= lo) & (ts <= hi)
    mean = P.mean(axis=0)

    ax.plot(ts[z], tgt[z], color="#8f8f8f", lw=1.1, ls=(0, (6, 3)), alpha=0.75,
            label="what we asked for (the command)")
    for k, prun in enumerate(P):
      ax.plot(ts[z], prun[z], color="#111111", lw=0.5, alpha=0.12,
              label=f"real motor - each of the {len(P)} runs" if k == 0 else None)
    ax.plot(ts[z], mean[z], color="#111111", lw=1.7, alpha=0.55, zorder=6,
            label=f"real motor - average of {len(P)}")
    ax.plot(ts[z], y[z], color="#00a878", lw=1.5, alpha=0.75, zorder=5,
            label="simulator with the measured values")

    err_r = float(np.mean(np.abs(mean[z] - tgt[z])))
    err_s = float(np.mean(np.abs(y[z] - tgt[z])))
    gap = float(np.mean(np.abs(y[z] - mean[z])))
    ax.text(0.015, 0.03,
            f"average distance from the command:  real {err_r:.2f} deg   "
            f"sim {err_s:.2f} deg\naverage gap between sim and real:  {gap:.2f} deg",
            transform=ax.transAxes, fontsize=8.2, va="bottom",
            bbox=dict(fc="white", ec="#ccc", alpha=0.9))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time [s]"); ax.set_ylabel("angle [deg]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.9)
    print(f"  {title.split('(')[0].strip():34} 실물-지령 {err_r:6.2f}도 · "
          f"시뮬-지령 {err_s:6.2f}도 · 시뮬-실물 {gap:6.2f}도")

  fig.suptitle(
    f"RobStride {a.family} (motor {a.motor}) - command, real motor, and the simulator "
    f"using the measured numbers\n"
    f"rotor inertia {prm['armature']}  ·  friction {prm['frictionloss']}  ·  "
    f"viscous drag {prm['damping']}   |   gains kp {a.kp:g} / kd {a.kd:g}, "
    f"same {a.delay_ms:g} ms feedback delay on both sides", fontsize=11.5)
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"그림: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
