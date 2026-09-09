#!/usr/bin/env python3
"""시뮬레이터에 **모터 한 대만** 세우고 실물과 같은 토크를 주어 나란히 본다.

2026-09-09. 사용자 지시: "Sim 에서도 그 파라미터를 갖고 mjlab 에서 모터'만'을 만들고,
그 응답과 비교하라."

무엇을 견주는가
---------------
벤치의 실물은 축에 아무것도 안 달린 맨 모터입니다. 그래서 모형도 경첩 하나 + 토크 하나로만
세웁니다. 다리도 중력도 접촉도 없습니다. 두 쪽에 **똑같은 토크**를 주고 속도가 어떻게
올라가는지 겹쳐 봅니다.

세 벌을 겹칩니다.

1. **실물** (라즈베리파이에서 잰 것)
2. **지금 모형의 값** — armature 0.0097 / frictionloss 0.2695 / damping 0.0095
3. **실측한 값** — armature 0.0773 / frictionloss 0.523 / damping 0.0276

2번이 실물에서 얼마나 벗어나 있는지, 3번이 실물 위에 얹히는지가 이 그림의 전부입니다.

    .venv/bin/python3 sim_vs_real_accel.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

DEG = np.pi / 180.0

XML = """
<mujoco model="one_motor">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast" timestep="{dt}">
    <flag contact="disable" gravity="disable"/>
  </option>
  <worldbody>
    <body name="rotor">
      <!-- 도는 것은 회전자뿐이므로 몸체 관성은 없는 셈 치고 armature 가 전부 맡는다. -->
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j" type="hinge" axis="0 0 1" limited="false"
             armature="{armature}" damping="{damping}" frictionloss="{frictionloss}"/>
    </body>
  </worldbody>
  <actuator><motor name="tau" joint="j" gear="1"/></actuator>
</mujoco>
"""

SETS = {
  "model today":   dict(armature=0.0097, frictionloss=0.2695, damping=0.0095),
  "measured":      dict(armature=0.0773, frictionloss=0.523,  damping=0.0276),
}


def run_sim(tau: float, seconds: float, dt: float, **p):
  m = mujoco.MjModel.from_xml_string(XML.format(dt=dt, **p))
  d = mujoco.MjData(m)
  d.ctrl[0] = tau
  n = int(seconds / dt)
  t = np.arange(n) * dt
  v = np.empty(n)
  for i in range(n):
    mujoco.mj_step(m, d)
    v[i] = d.qvel[0]
  return t, v / DEG


def load(p):
  with open(p) as f:
    r = [(float(x["t_s"]), float(x["vel_deg_s"])) for x in csv.DictReader(f)]
  return np.array([x[0] for x in r]), np.array([abs(x[1]) for x in r])


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--taus", default="1.2,1.8,2.4,3.0,3.6")
  ap.add_argument("--dt", type=float, default=0.0002)
  ap.add_argument("--out", default="../docs/img/motor_sim_vs_real_accel.png")
  a = ap.parse_args(argv)

  taus = [float(x) for x in a.taus.split(",") if x.strip()]
  fig, axs = plt.subplots(1, len(taus), figsize=(3.5 * len(taus), 4.2), sharey=True)
  if len(taus) == 1:
    axs = [axs]

  for ax, tau in zip(axs, taus):
    real = sorted(glob.glob(os.path.join(a.data, f"accel_{a.motor}_{tau:g}_p_*.csv")))
    tmax = 0.0
    for p in real:
      t, v = load(p)
      ax.plot(t * 1e3, v, color="#2c3e50", lw=1.2, alpha=0.65,
              label="real (bench)" if p == real[0] else None)
      tmax = max(tmax, t[-1])
    tmax = max(tmax, 0.15)
    for (lab, prm), col, ls in zip(SETS.items(), ("#c0392b", "#16a085"), ("--", "-")):
      ts, vs = run_sim(tau, tmax, a.dt, **prm)
      ax.plot(ts * 1e3, vs, ls, color=col, lw=1.8, alpha=0.9, label=f"sim: {lab}")
    ax.set_title(f"{tau:g} Nm", fontsize=11)
    ax.set_xlabel("time [ms]")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 800)
  axs[0].set_ylabel("speed [deg/s]")
  axs[0].legend(fontsize=8.5, loc="lower right")
  fig.suptitle("One motor, nothing on the shaft: same torque step to the real motor and to "
               "the simulated one\n"
               "model today: armature 0.0097 / friction 0.2695 / damping 0.0095    |    "
               "measured: armature 0.0773 / friction 0.523 / damping 0.0276",
               fontsize=10.5)
  fig.tight_layout(rect=(0, 0, 1, 0.88))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"그림: {a.out}")

  # 숫자로도 한 줄
  print("\n같은 토크에서 150~350 도/s 를 지나는 가속도 [라디안/s^2]")
  print(f"{'토크':>6} {'실물':>9} {'지금 모형':>11} {'실측값':>9}")
  for tau in taus:
    row = [f"{tau:5.1f}"]
    reals = []
    for p in sorted(glob.glob(os.path.join(a.data, f"accel_{a.motor}_{tau:g}_p_*.csv"))):
      t, v = load(p)
      i1 = int(np.argmax(v >= 150)); i2 = int(np.argmax(v >= 350))
      if i2 > i1:
        reals.append((v[i2] - v[i1]) * DEG / (t[i2] - t[i1]))
    row.append(f"{np.mean(reals):9.2f}" if reals else "        -")
    for lab, prm in SETS.items():
      ts, vs = run_sim(tau, 2.0, a.dt, **prm)
      i1 = int(np.argmax(vs >= 150)); i2 = int(np.argmax(vs >= 350))
      row.append(f"{(vs[i2]-vs[i1])*DEG/(ts[i2]-ts[i1]):11.2f}" if i2 > i1 else "          -")
    print(" ".join(row))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
