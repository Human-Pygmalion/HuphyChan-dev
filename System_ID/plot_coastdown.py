#!/usr/bin/env python3
"""타성 감속 자료를 그림으로 그린다. **이 컴퓨터에서 돌린다** (그림 도구가 여기 있음).

세 장을 그립니다.

1. **속도가 떨어지는 모습** — 시작 속도별로 겹쳐 그립니다. 마찰만으로 서는 것이므로 거의
   직선이어야 합니다 (점성이 섞인 만큼만 살짝 휩니다).
2. **속도 변화율 대 속도** — 직선이면 모형이 맞는 것입니다. 기울기가 -점성/관성,
   세로 절편이 -마찰/관성 입니다.
3. **놓은 뒤 모터가 보고한 토크** — 0 근처여야 합니다. 0 이 아니면 모터가 몰래 힘을 쓰고
   있다는 뜻이고, 그러면 이 측정 전체가 무효입니다. **이 그림이 검산입니다.**

그림의 글자는 영어로 씁니다 (한글은 그림 도구에서 네모로 깨집니다).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEG = 3.141592653589793 / 180.0
TAU_C_NM = 0.523
B_NM_S_RAD = 0.0276


def load(path):
  with open(path) as f:
    r = csv.DictReader(f)
    return [(float(x["t_s"]), float(x["pos_deg"]), float(x["vel_deg_s"]),
             float(x["tau_rep_nm"]), float(x["temp_c"])) for x in r]


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--glob", default="data/coast_*.csv")
  ap.add_argument("--out", default="../docs/img/motor_coastdown.png")
  ap.add_argument("--lo-frac", type=float, default=0.25)
  ap.add_argument("--hi-frac", type=float, default=0.92)
  a = ap.parse_args(argv)

  paths = sorted(glob.glob(a.glob))
  if not paths:
    print(f"자료가 없습니다: {a.glob}")
    return 1

  fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
  cmap = plt.get_cmap("viridis")
  speeds = sorted({os.path.basename(p).split("_")[2] for p in paths})

  all_pts = []
  for p in paths:
    rows = load(p)
    if len(rows) < 30:
      continue
    sp = os.path.basename(p).split("_")[2]
    c = cmap(speeds.index(sp) / max(1, len(speeds) - 1))
    t = [r[0] * 1e3 for r in rows]
    v = [abs(r[2]) for r in rows]
    axs[0].plot(t, v, color=c, lw=1.0, alpha=0.8)
    axs[2].plot(t, [r[3] for r in rows], color=c, lw=0.8, alpha=0.7)

    v0 = max(v[:10])
    i1 = next((i for i, x in enumerate(v) if x <= a.hi_frac * v0), None)
    i2 = next((i for i, x in enumerate(v) if x <= a.lo_frac * v0), None)
    if i1 is None or i2 is None or i2 - i1 < 20:
      continue
    k = max(3, (i2 - i1) // 12)
    for s in range(i1, i2 - k, k):
      e = s + k
      dt = rows[e][0] - rows[s][0]
      if dt <= 0:
        continue
      acc = (abs(rows[e][2]) - abs(rows[s][2])) * DEG / dt
      vm = (abs(rows[e][2]) + abs(rows[s][2])) / 2 * DEG
      all_pts.append((vm, acc, c))

  if all_pts:
    axs[1].scatter([p[0] for p in all_pts], [p[1] for p in all_pts],
                   c=[p[2] for p in all_pts], s=14, alpha=0.75)
    n = len(all_pts)
    sx = sum(p[0] for p in all_pts); sy = sum(p[1] for p in all_pts)
    sxx = sum(p[0] ** 2 for p in all_pts); sxy = sum(p[0] * p[1] for p in all_pts)
    den = n * sxx - sx * sx
    if abs(den) > 1e-12:
      m = (n * sxy - sx * sy) / den
      b = (sy - m * sx) / n
      xs = [min(p[0] for p in all_pts), max(p[0] for p in all_pts)]
      axs[1].plot(xs, [m * x + b for x in xs], "r-", lw=1.6,
                  label=f"fit: slope {m:.3f}, intercept {b:.3f}\n"
                        f"J from intercept = {TAU_C_NM/abs(b)*1e3:.2f} g m^2\n"
                        f"J from slope     = {B_NM_S_RAD/abs(m)*1e3:.2f} g m^2")
      axs[1].legend(fontsize=8, loc="lower right")

  axs[0].set_xlabel("time after torque released [ms]")
  axs[0].set_ylabel("speed [deg/s]")
  axs[0].set_title("Coast-down: speed decay (friction only)")
  axs[0].grid(alpha=0.3)

  axs[1].set_xlabel("speed [rad/s]")
  axs[1].set_ylabel("deceleration [rad/s^2]")
  axs[1].set_title("Deceleration vs speed\n(slope = -b/J, intercept = -tau_c/J)")
  axs[1].grid(alpha=0.3)

  axs[2].axhline(0, color="k", lw=0.8)
  axs[2].set_xlabel("time after torque released [ms]")
  axs[2].set_ylabel("reported torque [Nm]")
  axs[2].set_title("SANITY CHECK: motor must report ~0 Nm\n(if not, it is still pushing)")
  axs[2].grid(alpha=0.3)

  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"그림: {a.out}  ({len(paths)}개 자료)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
