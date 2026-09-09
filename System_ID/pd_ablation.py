#!/usr/bin/env python3
"""세 값을 **하나씩만** 바꿔 가며 응답이 어떻게 달라지는지 본다. **이 컴퓨터에서 돌린다.**

2026-09-09. 사용자 지시: "좀 더 다양한 명령을 넣어줘. 대조군으로 각 armature / damping /
frictionloss 값들이 너무 크거나 아예 없을 때랑도 비교해줘. 샘플 수 10번으로 늘리고,
백그라운드에 알파 빼서 넣고, 평균값을 비교해서 보여줘."

왜 하나씩만 바꾸는가
--------------------
셋을 한꺼번에 바꾸면 어느 값이 무엇을 했는지 알 수 없습니다. **둘은 실측값에 고정하고 하나만**
없음 → 지금 모형 → 실측 → 과다(4배)로 옮깁니다. 그러면 그 값이 **어느 명령에서 얼마나**
드러나는지가 그대로 보입니다.

다섯 가지 명령을 쓰는 이유도 같습니다 — 값마다 드러나는 자리가 다릅니다.

===========  ====================================================================
큰 계단      뜨는 속도(관성)와 끝내 못 지운 오차(마찰)를 한 번에
작은 계단    마찰의 죽은구간 안에서만 논다. **frictionloss 에 가장 민감**
처프         주파수 응답. **armature 에 가장 민감** (고유진동수가 √관성에 반비례)
느린 삼각파  일정 속도 20 도/s. 점성은 무시할 크기라 **쿨롱 마찰만** 보임
빠른 삼각파  일정 속도 240 도/s. **여기서 점성이 드러남**
===========  ====================================================================

10번씩 받은 실물 자료는 **개별을 옅게, 평균을 진하게** 그립니다. 개별이 평균 언저리에
몰려 있으면 그 평균을 믿어도 됩니다.
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
  <actuator><motor name="tau" joint="j" gear="1" ctrlrange="-120 120"/></actuator>
</mujoco>
"""

BASE = dict(armature=0.0773, frictionloss=0.519, damping=0.0275)
TODAY = dict(armature=0.016333, frictionloss=0.269456, damping=0.009492)

SWEEP = {
  # 값 이름 -> [(설명, 값), ...]  나머지 둘은 실측값 고정
  "armature": [("almost none (1/20)", 0.0773 / 20), ("model today", 0.016333),
               ("MEASURED", 0.0773), ("4x too big", 0.0773 * 4)],
  "frictionloss": [("none (0)", 0.0), ("model today", 0.269456),
                   ("MEASURED", 0.519), ("4x too big", 0.519 * 4)],
  "damping": [("none (0)", 0.0), ("model today", 0.009492),
              ("MEASURED", 0.0275), ("4x too big", 0.0275 * 4)],
}
"""armature 만 0 을 못 쓴다 — 관절에 관성이 하나도 없으면 계산이 발산한다. 1/20 로 대신한다."""

COLS = ["#8e44ad", "#c0392b", "#16a085", "#e67e22"]


def load_runs(pattern):
  runs = []
  for p in sorted(globmod.glob(pattern)):
    with open(p) as f:
      r = list(csv.DictReader(f))
    if len(r) < 200:
      continue
    runs.append((np.array([float(x["t_s"]) for x in r]),
                 np.array([float(x["target_deg"]) for x in r]),
                 np.array([float(x["target_vel_deg_s"]) for x in r]),
                 np.array([float(x["pos_deg"]) for x in r])))
  return runs


def common_grid(runs, dt=0.001):
  """모든 회차를 같은 시각 격자에 올린다. 각 회차의 출발 위치는 다르므로 **뺀다**."""
  tmax = min(r[0][-1] for r in runs)
  ts = np.arange(0, tmax, dt)
  P, T, V = [], None, None
  for t, tgt, tgv, pos in runs:
    off = tgt[0]                       # 각 회차의 목표 원점 = 그 회차의 시작 자리
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
  return st, np.interp(ts, st, out)


# --------------------------------------------------------------------- 지표
def metrics(mode, ts, tgt, tgv, pos, f0=0.3, f1=10.0):
  e = pos - tgt
  if mode == "chirp":
    D = ts[-1] - ts[0]
    k = math.log(f1 / f0) / D
    fc, g = [], []
    edges = np.linspace(ts[0], ts[-1], 50)
    for i in range(len(edges) - 1):
      s = (ts >= edges[i]) & (ts < edges[i + 1])
      if s.sum() < 20:
        continue
      ain = (tgt[s].max() - tgt[s].min()) / 2
      if ain < 1e-3:
        continue
      fc.append(f0 * math.exp(k * ((edges[i] + edges[i + 1]) / 2 - ts[0])))
      g.append((pos[s].max() - pos[s].min()) / 2 / ain)
    fc, g = np.array(fc), np.array(g)
    # 봉우리는 1.5 Hz 위에서만 찾는다. 한 토막이 0.4초라 그 아래 주파수는 사이클이 반도
    # 안 들어가서 진폭 추정이 못 믿을 값이 된다.
    hi = fc > 1.5
    return {"gain at 10 Hz": float(np.interp(9.0, fc, g)),
            "peak gain": float(g[hi].max()) if hi.any() else float("nan"),
            "peak freq [Hz]": float(fc[hi][g[hi].argmax()]) if hi.any() else float("nan")}
  if mode.startswith("step"):
    # ★ 모서리는 **크게 뛴 곳**으로만 잡는다. 시각 격자에 다시 올릴 때 계단이 한 칸
    # 번지므로 "달라진 곳"으로 잡으면 모서리가 수백 개가 되어 지표가 통째로 망가진다
    # (넘어감이 195%로 나오는 것을 보고 알아챘다).
    span = float(tgt.max() - tgt.min())
    edges = np.where(np.abs(np.diff(tgt)) > 0.4 * span)[0]
    ss, ov, rise = [], [], []
    for i, ed in enumerate(edges[:-1]):
      te, tn = ts[ed], ts[edges[i + 1]]
      lo, hi = float(tgt[ed]), float(tgt[ed + 1])
      d = hi - lo
      if abs(d) < 1e-6:
        continue
      sgn = 1.0 if d > 0 else -1.0
      # 자리잡은 뒤(다음 계단 직전 0.5초)에 남아 있는 오차
      s1 = (ts > tn - 0.5) & (ts < tn)
      if s1.sum() > 5:
        ss.append(abs(float(np.median(e[s1]))))
      # 계단 직후 0.6초: 목표를 **가는 방향으로** 얼마나 지나쳤는가
      s2 = (ts > te) & (ts < te + 0.6)
      if s2.sum() > 5:
        beyond = float(np.max((pos[s2] - hi) * sgn))
        ov.append(max(0.0, beyond) / abs(d) * 100)
        # 목표의 90%까지 가는 데 걸린 시간
        reached = np.where((pos[s2] - (lo + 0.9 * d)) * sgn >= 0)[0]
        if len(reached):
          rise.append((float(ts[s2][reached[0]]) - te) * 1e3)
    return {"left-over error [deg]": float(np.mean(ss)) if ss else float("nan"),
            "overshoot [%]": float(np.mean(ov)) if ov else float("nan"),
            "rise to 90% [ms]": float(np.mean(rise)) if rise else float("nan")}
  # triangle
  moving = np.abs(tgv) > 0.5 * np.abs(tgv).max()
  vel = np.gradient(pos, ts)
  stuck = moving & (np.abs(vel) < 0.15 * np.abs(tgv).max())
  return {"tracking error [deg]": float(np.mean(np.abs(e[moving]))),
          "stuck at reversal [%]": float(stuck.sum() / max(1, moving.sum()) * 100),
          "lag behind command [deg]": float(np.mean(e[moving] * -np.sign(tgv[moving])))}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0002)
  ap.add_argument("--out-prefix", default="../docs/img/pd_ablation_rs04")
  a = ap.parse_args(argv)

  CMDS = [
    ("big step +-8 deg",     "step",     f"pd_step_{a.motor}_kp20_kd1_a8_r*_b.csv"),
    ("small step +-1.5 deg", "step",     f"pd_step_{a.motor}_kp20_kd1_a1.5_r*_b.csv"),
    ("chirp 0.3-10 Hz",      "chirp",    f"pd_chirp_{a.motor}_kp20_kd1_a8_r*_b.csv"),
    ("slow triangle 20 deg/s",  "triangle", f"pd_triangle_{a.motor}_kp20_kd1_a20_r*_b.csv"),
    ("fast triangle 240 deg/s", "triangle", f"pd_triangle_{a.motor}_kp20_kd1_a60_r*_bf.csv"),
  ]

  loaded = []
  for name, mode, pat in CMDS:
    runs = load_runs(os.path.join(a.data, pat))
    if not runs:
      print(f"  자료 없음: {pat}")
      continue
    ts, tgt, tgv, P = common_grid(runs)
    loaded.append((name, mode, ts, tgt, tgv, P))
    print(f"{name:26} {len(runs):2d}회 · {ts[-1]:.1f}초 · 회차 사이 흩어짐(제곱평균) "
          f"{np.sqrt(np.mean(P.var(axis=0))):.3f}도")

  # ---------- 그림 1: 실물 10회 (개별 옅게 + 평균 진하게) ----------------
  fig, axs = plt.subplots(len(loaded), 1, figsize=(13, 2.6 * len(loaded)))
  for ax, (name, mode, ts, tgt, tgv, P) in zip(np.atleast_1d(axs), loaded):
    w = min(ts[-1], 6.0)
    s = ts <= w
    for p in P:
      ax.plot(ts[s], p[s], color="#2c3e50", lw=0.8, alpha=0.22)
    ax.plot(ts[s], P.mean(axis=0)[s], color="#2c3e50", lw=2.0,
            label=f"real motor, mean of {len(P)}")
    ax.plot(ts[s], tgt[s], color="#7f8c8d", lw=1.3, label="commanded")
    ax.set_title(f"{name}   (faint = each of the {len(P)} runs, bold = their mean)",
                 fontsize=10)
    ax.set_ylabel("angle [deg]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
  np.atleast_1d(axs)[-1].set_xlabel("time [s]")
  fig.suptitle(f"RobStride motor {a.motor}: five different commands, "
               f"{len(loaded[0][5])} repeats each", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(f"{a.out_prefix}_runs.png", dpi=125)
  print(f"\n그림: {a.out_prefix}_runs.png")

  # ---------- 그림 2: 값 하나씩 쓸어보기 --------------------------------
  real_m = {name: metrics(mode, ts, tgt, tgv, P.mean(axis=0))
            for name, mode, ts, tgt, tgv, P in loaded}
  keys = {name: list(real_m[name].keys()) for name in real_m}
  nrow = len(loaded)
  fig2, axs2 = plt.subplots(nrow, 3, figsize=(15.5, 2.9 * nrow), squeeze=False)
  table = {}
  for ci, (pname, options) in enumerate(SWEEP.items()):
    for ri, (name, mode, ts, tgt, tgv, P) in enumerate(loaded):
      ax = axs2[ri][ci]
      key = keys[name][0]
      xs, ys, labs = [], [], []
      for (lab, val), col in zip(options, COLS):
        prm = dict(BASE); prm[pname] = val
        _st, y = simulate(ts, tgt, tgv, a.kp, a.kd, a.dt, a.tau_max,
                          a.delay_ms / 1e3, **prm)
        mm = metrics(mode, ts, tgt, tgv, y)
        table[(name, pname, lab)] = mm
        xs.append(val); ys.append(mm[key]); labs.append(lab)
      ax.plot(xs, ys, "o-", color="#34495e", lw=1.5, ms=7, zorder=2)
      for x, y, lab, col in zip(xs, ys, labs, COLS):
        ax.plot([x], [y], "o", color=col, ms=10, zorder=3)
      ax.axhline(real_m[name][key], color="#c0392b", ls="--", lw=1.8, zorder=1)
      ax.text(0.02, 0.94, f"real motor = {real_m[name][key]:.3g}", color="#c0392b",
              transform=ax.transAxes, fontsize=8.5, va="top")
      ax.set_xscale("symlog", linthresh=max(1e-4, min(v for _l, v in options if v > 0) / 3))
      ax.grid(alpha=0.3)
      if ri == 0:
        ax.set_title(f"sweeping {pname}\n(other two held at measured)", fontsize=10)
      if ci == 0:
        ax.set_ylabel(f"{name}\n{key}", fontsize=8.5)
      if ri == nrow - 1:
        ax.set_xlabel(f"{pname} value")
  h = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
       for c, l in zip(COLS, [o[0] for o in SWEEP["frictionloss"]])]
  h.append(plt.Line2D([], [], ls="--", color="#c0392b", label="real motor (mean of 10)"))
  fig2.legend(handles=h, loc="lower center", ncol=5, fontsize=9)
  fig2.suptitle(f"RobStride motor {a.motor}: change ONE parameter at a time - "
                f"none / today / measured / 4x too big", fontsize=12)
  fig2.tight_layout(rect=(0, 0.035, 1, 0.97))
  fig2.savefig(f"{a.out_prefix}_sweep.png", dpi=125)
  print(f"그림: {a.out_prefix}_sweep.png")

  # ---------- 표 -------------------------------------------------------
  print("\n" + "=" * 108)
  for name, mode, ts, tgt, tgv, P in loaded:
    print(f"\n■ {name}")
    ks = keys[name]
    print(f"  {'설정':38}" + "".join(f"{k:>23}" for k in ks))
    print(f"  {'실물 (10회 평균)':36}" + "".join(f"{real_m[name][k]:23.4g}" for k in ks))
    for pname, options in SWEEP.items():
      for lab, val in options:
        mm = table[(name, pname, lab)]
        mark = " ←" if lab == "MEASURED" else "  "
        print(f"  {pname:14} {lab:20} {val:<6.4g}"[:38].ljust(38)
              + "".join(f"{mm[k]:23.4g}" for k in ks) + mark)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
