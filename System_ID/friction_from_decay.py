#!/usr/bin/env python3
"""속도가 변하는 구간 **전체**에서 마찰을 뽑는다. **이 컴퓨터에서 돌린다.**

2026-09-09. 자리잡은 속도를 기다리는 방식은 낮은 토크에서 너무 느립니다 — 시간상수가 약
3.7초라 0.60 뉴턴미터짜리는 14초를 기다려도 35% 더 느려질 참이었습니다. 기다리지 않아도
됩니다. **회전자 관성을 이미 정확히 알기 때문입니다** (0.0773 kg·m², 두 방법이 합치).

    시킨 토크 − 관성 x (속도 변화율) = 마찰(속도)

왼쪽은 전부 아는 값입니다. 그러니 속도가 변하는 **매 순간**이 그 속도에서의 마찰 한 점을
줍니다. 한 번 감속시키면 그 속도 범위 전체의 마찰 곡선이 한꺼번에 나옵니다.

미분은 여전히 안 합니다
-----------------------
짧은 창(기본 0.06초)마다 **적분 꼴**로 계산합니다.

    마찰 = 시킨 토크 − 관성 x (창 끝 속도 − 창 처음 속도) / (창 길이)
    그때의 속도 = (창 동안 돈 각도) / (창 길이)      ← 위치 차이라 정확하다

속도 자체도 보고 값을 안 쓰고 **위치를 이어 붙여** 구합니다 (보고 속도는 859 도/s 에서
잘립니다).

    friction_from_decay.py --glob 'data/coastmax_*.csv'
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEG = math.pi / 180.0
J_KGM2 = 0.0773
"""회전자 관성. 2026-09-09 실측 (타성 감속 73~81, 마찰 소거법 77.26 +- 0.64 g·m^2)."""
OLD_C, OLD_B = 0.523, 0.0276


def unwrap(raw):
  out = [0.0]
  for i in range(1, len(raw)):
    d = raw[i] - raw[i - 1]
    while d > 720.0:
      d -= 1440.0
    while d < -720.0:
      d += 1440.0
    out.append(out[-1] + d)
  return out


def points(path, win_s, tau_col, skip_s):
  """한 벌에서 (속도 라디안/s, 마찰 뉴턴미터, 온도, 방향) 점들을 뽑는다."""
  with open(path) as f:
    r = list(csv.DictReader(f))
  if len(r) < 200:
    return []
  t = [float(x["t_s"]) for x in r]
  pos = unwrap([float(x["pos_deg"]) for x in r])
  tc = [float(x["temp_c"]) for x in r]
  tau = ([float(x[tau_col]) for x in r] if tau_col and tau_col in r[0]
         else [0.0] * len(r))
  t0 = t[0] + skip_s
  out, i = [], 0
  while i < len(t):
    j = i
    while j < len(t) and t[j] - t[i] < win_s:
      j += 1
    if j >= len(t) - 1:
      break
    dt = t[j] - t[i]
    if dt <= 0 or t[i] < t0:
      i = j
      continue
    k = max(3, (j - i) // 4)
    if i - k < 0 or j + k >= len(t):
      i = j
      continue
    v1 = (pos[i + k] - pos[i - k]) / (t[i + k] - t[i - k]) * DEG
    v2 = (pos[j + k] - pos[j - k]) / (t[j + k] - t[j - k]) * DEG
    wbar = (pos[j] - pos[i]) / dt * DEG
    if abs(wbar) < 0.05:
      i = j
      continue
    s = 1.0 if wbar > 0 else -1.0
    fric = (sum(tau[i:j]) / (j - i)) * s - J_KGM2 * (v2 - v1) / dt * s
    out.append((abs(wbar), fric, sum(tc[i:j]) / (j - i), s))
    i = j
  return out


def linfit(xs, ys):
  n = len(xs); sx = sum(xs); sy = sum(ys)
  sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
  syy = sum(y * y for y in ys); den = n * sxx - sx * sx
  m = (n * sxy - sx * sy) / den
  b = (sy - m * sx) / n
  r2 = (n * sxy - sx * sy) ** 2 / (den * (n * syy - sy * sy))
  res = [y - (m * x + b) for x, y in zip(xs, ys)]
  sd = (sum(e * e for e in res) / max(1, n - 2)) ** 0.5
  return m, b, r2, sd * (n / den) ** 0.5, sd


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--glob", default="data/coast_4_*_warm.csv")
  ap.add_argument("--tau-col", default="cmd_tau_nm")
  ap.add_argument("--win-s", type=float, default=0.06)
  ap.add_argument("--skip-s", type=float, default=0.15)
  ap.add_argument("--fit-lo", type=float, default=1.0)
  ap.add_argument("--inertia", type=float, default=J_KGM2,
                  help="이 모터의 회전자 관성. RS04 0.0773 · RS03 0.0638 (2026-09-09 실측)")
  ap.add_argument("--out", default="../docs/img/motor_friction_curve.png")
  ap.add_argument("--title", default="")
  a = ap.parse_args(argv)

  globals()["J_KGM2"] = a.inertia
  pts = []
  for p in sorted(glob.glob(a.glob)):
    pts += points(p, a.win_s, a.tau_col, a.skip_s)
  if len(pts) < 50:
    print(f"점이 부족합니다 ({len(pts)})"); return 1

  fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0))
  for s, col, lab in ((1.0, "#2c3e50", "forward"), (-1.0, "#c0392b", "reverse")):
    d = [(w, f) for w, f, _, sg in pts if sg == s]
    if len(d) < 20:
      continue
    axs[0].scatter([x[0] for x in d], [x[1] for x in d], s=3, c=col, alpha=0.25, label=lab)
    hi = [x for x in d if x[0] >= a.fit_lo]
    if len(hi) > 20:
      m, b, r2, se, _ = linfit([x[0] for x in hi], [x[1] for x in hi])
      print(f"{lab:>8}: 마찰 {b:6.3f} Nm · 점성 {m:7.4f} +- {se:6.4f} · "
            f"결정계수 {r2:.4f} · {len(hi)}점")

  hi = [(w, f) for w, f, _, _ in pts if w >= a.fit_lo]
  m, b, r2, se, sd = linfit([x[0] for x in hi], [x[1] for x in hi])
  xs = [0, max(x[0] for x in pts) * 1.02]
  axs[0].plot(xs, [m * x + b for x in xs], "-", color="#16a085", lw=2.4, zorder=5,
              label=f"fit above {a.fit_lo:g} rad/s: {b:.3f} + {m:.4f} w")
  axs[0].plot(xs, [OLD_B * x + OLD_C for x in xs], "--", color="#f39c12", lw=2.0, zorder=5,
              label=f"yesterday: {OLD_C} + {OLD_B} w")
  axs[0].set_xlabel("speed [rad/s]"); axs[0].set_ylabel("friction torque [Nm]")
  axs[0].set_title("Friction across the whole speed range\n"
                   "commanded torque minus inertia x acceleration")
  axs[0].legend(fontsize=8.5, loc="upper left"); axs[0].grid(alpha=0.3)
  axs[0].set_xlim(left=0); axs[0].set_ylim(bottom=0)

  lo = [(w, f, sg) for w, f, _, sg in pts if w <= max(3.0, a.fit_lo * 3)]
  for s, col, lab in ((1.0, "#2c3e50", "forward"), (-1.0, "#c0392b", "reverse")):
    d = [(w, f) for w, f, sg in lo if sg == s]
    if d:
      axs[1].scatter([x[0] for x in d], [x[1] for x in d], s=8, c=col, alpha=0.35, label=lab)
  if lo:
    step = max(0.05, max(x[0] for x in lo) / 25)
    bx, by = [], []
    k = 0
    while k * step < max(x[0] for x in lo):
      w = [x[1] for x in lo if k * step <= x[0] < (k + 1) * step]
      if len(w) >= 5:
        w.sort(); bx.append((k + 0.5) * step); by.append(w[len(w) // 2])
      k += 1
    axs[1].plot(bx, by, "o-", color="#8e44ad", lw=2, ms=4, label="median per speed bin")
  xs2 = [0, max(x[0] for x in lo) if lo else 3.0]
  axs[1].plot(xs2, [m * x + b for x in xs2], "-", color="#16a085", lw=2,
              label="the straight fit, extended down")
  axs[1].set_xlabel("speed [rad/s]"); axs[1].set_ylabel("friction torque [Nm]")
  axs[1].set_title("Close-up at low speed\n"
                   "points above the line = the Stribeck drop (stiction giving way)")
  axs[1].legend(fontsize=8.5); axs[1].grid(alpha=0.3)

  temps = [t for _, _, t, _ in pts]
  fig.suptitle((a.title or "RobStride motor 4, friction remeasured on the Raspberry Pi")
               + f"   |   {min(temps):.0f}-{max(temps):.0f} C, {len(pts)} windows",
               fontsize=11.5)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"\n  양쪽: 마찰 {b:.3f} Nm · 점성 {m:.4f} +- {se:.4f} · 결정계수 {r2:.4f} · {len(hi)}점")
  print(f"  어제: 마찰 {OLD_C} · 점성 {OLD_B}")
  print(f"그림: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
