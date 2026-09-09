#!/usr/bin/env python3
"""다시 잰 마찰을 그린다. **이 컴퓨터에서 돌린다.**

왼쪽 — **토크 대 자리잡은 속도**. 토크를 하나 정해 두면 모터는 마찰과 맞비기는 속도에서
멈춥니다. 가속도가 0 이라 관성이 식에 안 들어옵니다. 세로 절편이 움직일 때 마찰,
기울기가 점성입니다.

오른쪽 — 어제 값과 겹쳐 보기. 어제는 속도를 붙잡고 **모터가 보고하는 토크**를 읽었는데,
그 값이 속도가 붙으면 못 믿는다는 것을 오늘 알았습니다(회전 방향을 따라 부호가 바뀌는
+0.3 치우침). 이번에는 **시킨 토크**만 씁니다.
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

DEG = 3.141592653589793 / 180.0
OLD_C, OLD_B = 0.523, 0.0276          # 어제 (속도 붙잡고 보고 토크 읽기)


def linfit(xs, ys):
  n = len(xs); sx = sum(xs); sy = sum(ys)
  sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
  syy = sum(y * y for y in ys); den = n * sxx - sx * sx
  m = (n * sxy - sx * sy) / den
  b = (sy - m * sx) / n
  r2 = (n * sxy - sx * sy) ** 2 / (den * (n * syy - sy * sy))
  return m, b, r2


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--read-s", type=float, default=2.5)
  ap.add_argument("--out", default="../docs/img/motor_friction_retest.png")
  a = ap.parse_args(argv)

  pat = re.compile(rf"fcurve_{a.motor}_([0-9.]+)_(p|n)_")
  pts = []
  for p in sorted(glob.glob(os.path.join(a.data, f"fcurve_{a.motor}_*.csv"))):
    mm = pat.search(os.path.basename(p))
    if not mm:
      continue
    with open(p) as f:
      r = [(float(x["t_s"]), float(x["vel_true_deg_s"])) for x in csv.DictReader(f)]
    if len(r) < 200:
      continue
    tail = [x for x in r if x[0] > r[-1][0] - a.read_s]
    vs = [abs(x[1]) for x in tail]
    if not vs:
      continue
    v = sum(vs) / len(vs)
    if max(vs) - min(vs) > max(12.0, 0.03 * v) or v < 5.0:
      continue
    pts.append((mm.group(2), float(mm.group(1)), v * DEG))
  if not pts:
    print("자료 없음"); return 1

  fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0))
  for sgn, col, lab in (("p", "#2c3e50", "forward"), ("n", "#c0392b", "reverse")):
    d = [(w, t) for s, t, w in pts if s == sgn]
    if len(d) < 3:
      continue
    axs[0].scatter([x[0] for x in d], [x[1] for x in d], c=col, s=34, zorder=3, label=lab)
    b, c, r2 = linfit([x[0] for x in d], [x[1] for x in d])
    xs = [0, max(x[0] for x in d) * 1.05]
    axs[0].plot(xs, [b * x + c for x in xs], "-", color=col, lw=1.4, alpha=0.8)
    print(f"{lab}: 마찰 {c:.3f} Nm · 점성 {b:.4f} · 결정계수 {r2:.4f} · {len(d)}점")

  allp = [(w, t) for _, t, w in pts]
  b, c, r2 = linfit([x[0] for x in allp], [x[1] for x in allp])
  xs = [0, max(x[0] for x in allp) * 1.05]
  axs[0].plot(xs, [b * x + c for x in xs], "-", color="#16a085", lw=2.4, zorder=2,
              label=f"both: {c:.3f} + {b:.4f} w")
  axs[0].plot(xs, [OLD_B * x + OLD_C for x in xs], "--", color="#7f8c8d", lw=1.8,
              label=f"yesterday: {OLD_C} + {OLD_B} w")
  axs[0].set_xlabel("settled speed [rad/s]")
  axs[0].set_ylabel("commanded torque [Nm]")
  axs[0].set_title("Friction, remeasured\ncommanded torque vs the speed it settles at")
  axs[0].legend(fontsize=8.5, loc="upper left"); axs[0].grid(alpha=0.3)
  axs[0].set_xlim(left=0)

  for p in sorted(glob.glob(os.path.join(a.data, f"fcurve_{a.motor}_*_p_0*.csv"))):
    with open(p) as f:
      r = [(float(x["t_s"]), abs(float(x["vel_true_deg_s"])), abs(float(x["vel_rep_deg_s"])))
           for x in csv.DictReader(f)]
    if len(r) < 200:
      continue
    t0 = r[0][0]
    axs[1].plot([x[0] - t0 for x in r], [x[1] for x in r], lw=0.9, alpha=0.75)
  axs[1].axhline(859.4, color="#c0392b", ls="--", lw=1.4)
  axs[1].text(0.98, 859.4, " reported speed saturates here (15.0 rad/s)", color="#c0392b",
              fontsize=8.5, va="bottom", ha="right", transform=axs[1].get_yaxis_transform())
  axs[1].set_xlabel("time [s]")
  axs[1].set_ylabel("true speed from position [deg/s]")
  axs[1].set_title("How each torque settles\n(spun up first, then dropped to the target)")
  axs[1].grid(alpha=0.3)

  fig.suptitle(f"RobStride motor {a.motor}: friction remeasured on the Raspberry Pi "
               f"(1800+ exchanges/s), commanded torque only", fontsize=11.5)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"\n그림: {a.out}")
  print(f"양쪽 합침: 마찰 {c:.3f} Nm · 점성 {b:.4f} · 결정계수 {r2:.4f} · {len(allp)}점")
  print(f"어제 값   : 마찰 {OLD_C} · 점성 {OLD_B}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
