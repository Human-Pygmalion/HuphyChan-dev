#!/usr/bin/env python3
"""회전자 관성 측정 결과를 그림으로 그린다. **이 컴퓨터에서 돌린다.**

두 방법을 나란히 놓습니다.

* 왼쪽 — **타성 감속**. 토크를 0 으로 놓고 마찰만으로 서게 두었을 때의 속도. 마찰값을
  안다는 전제 위에 서는 방법입니다.
* 가운데 — **토크 대 가속도**. 같은 속도 구간을 서로 다른 토크로 지나게 하면 마찰이
  식에서 통째로 빠집니다. **기울기가 곧 관성**이고, 세로 절편은 그 구간의 마찰입니다.
* 오른쪽 — 놓은 뒤 모터가 보고한 토크. 0 근처여야 정상입니다.

그림 글자는 영어입니다 (그림 도구가 한글을 네모로 그림).
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
TAU_C_NM, B_NM_S_RAD = 0.523, 0.0276
MODEL_ARM = {"4": 0.0097, "3": 0.0049}   # biped_cfg.py 의 armature (감속기 뒤 기준)


def load(p):
  with open(p) as f:
    return [(float(x["t_s"]), float(x["pos_deg"]), float(x["vel_deg_s"]),
             float(x["tau_rep_nm"]), float(x["temp_c"])) for x in csv.DictReader(f)]


def alpha_of(rows, v_lo=150.0, v_hi=350.0):
  v = [abs(r[2]) for r in rows]
  i1 = next((i for i, x in enumerate(v) if x >= v_lo), None)
  if i1 is None:
    return None
  i2 = next((i for i, x in enumerate(v[i1:], i1) if x >= v_hi), None)
  if i2 is None or i2 - i1 < 8:
    return None
  w = min(5, (i2 - i1) // 3)
  dt = rows[i2][0] - rows[i1][0]
  if dt <= 0:
    return None
  return (sum(v[i2 - w:i2]) / w - sum(v[i1:i1 + w]) / w) * DEG / dt


def linfit(xs, ys):
  n = len(xs)
  sx, sy = sum(xs), sum(ys)
  sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
  syy = sum(y * y for y in ys)
  den = n * sxx - sx * sx
  m = (n * sxy - sx * sy) / den
  b = (sy - m * sx) / n
  r2 = (n * sxy - sx * sy) ** 2 / (den * (n * syy - sy * sy))
  return m, b, r2


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor", default="4")
  ap.add_argument("--data", default="data")
  ap.add_argument("--out", default="../docs/img/motor_inertia_rs04.png")
  a = ap.parse_args(argv)

  fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.1))

  # --- 1) 타성 감속 -------------------------------------------------------
  coasts = sorted(glob.glob(os.path.join(a.data, f"coast_{a.motor}_*.csv")))
  for p in coasts:
    rows = load(p)
    if len(rows) < 30:
      continue
    neg = "warmneg" in p
    axs[0].plot([r[0] * 1e3 for r in rows], [abs(r[2]) for r in rows],
                color="#c0392b" if neg else "#2c3e50", lw=0.9, alpha=0.55)
    axs[2].plot([r[0] * 1e3 for r in rows], [r[3] for r in rows],
                color="#c0392b" if neg else "#2c3e50", lw=0.6, alpha=0.4)
  axs[0].plot([], [], color="#2c3e50", label="forward")
  axs[0].plot([], [], color="#c0392b", label="reverse")
  axs[0].set_xlabel("time after torque released [ms]")
  axs[0].set_ylabel("speed [deg/s]")
  axs[0].set_title("Method 1 - coast-down\n(torque released; only friction stops it)")
  axs[0].legend(fontsize=9); axs[0].grid(alpha=0.3)

  # --- 2) 토크 대 가속도 ---------------------------------------------------
  pat = re.compile(rf"accel_{a.motor}_([0-9.]+)_(p|n)_")
  xs, ys, cs = [], [], []
  for p in sorted(glob.glob(os.path.join(a.data, f"accel_{a.motor}_*.csv"))):
    mm = pat.search(os.path.basename(p))
    if not mm:
      continue
    rows = load(p)
    if len(rows) < 20:
      continue
    al = alpha_of(rows)
    if al is None or al <= 0:
      continue
    xs.append(al); ys.append(float(mm.group(1)))
    cs.append("#2c3e50" if mm.group(2) == "p" else "#c0392b")
  if xs:
    axs[1].scatter(xs, ys, c=cs, s=34, zorder=3, alpha=0.85)
    J, f0, r2 = linfit(xs, ys)
    lo, hi = min(xs), max(xs)
    axs[1].plot([0, hi * 1.05], [f0, J * hi * 1.05 + f0], "-", color="#16a085", lw=2, zorder=2)
    pred = TAU_C_NM + B_NM_S_RAD * (250.0 * DEG)
    axs[1].text(0.03, 0.96,
                f"slope = INERTIA = {J*1e3:.2f} g m$^2$\n"
                f"intercept = friction = {f0:.3f} Nm\n"
                f"R$^2$ = {r2:.4f}   (n={len(xs)})\n"
                f"model armature = {MODEL_ARM.get(a.motor, 0)*1e3:.1f} g m$^2$"
                f"  ->  {J/MODEL_ARM.get(a.motor, 1):.1f}x too low"
                + (f"\nfriction predicted from separate fit: {pred:.3f} Nm"
                   if a.motor == "4" else ""),
                transform=axs[1].transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="#bbb", alpha=0.92))
  axs[1].set_xlabel("acceleration through 150-350 deg/s window [rad/s$^2$]")
  axs[1].set_ylabel("commanded torque [Nm]")
  axs[1].set_title("Method 2 - torque vs acceleration\nfriction cancels: slope IS the inertia")
  axs[1].grid(alpha=0.3); axs[1].set_xlim(left=0)

  axs[2].axhline(0, color="k", lw=0.8)
  axs[2].set_xlabel("time after torque released [ms]")
  axs[2].set_ylabel("reported torque [Nm]")
  axs[2].set_title("Sanity check: reported torque while coasting")
  axs[2].text(0.5, 0.02,
              "mean +0.32 Nm forward, -0.40 Nm reverse: follows the motion.\n"
              "Not a real shaft torque - both methods would then disagree by 1.8x.\n"
              "They agree, so this is an artifact of the estimate at speed.",
              transform=axs[2].transAxes, ha="center", va="bottom", fontsize=8.5,
              bbox=dict(fc="white", ec="#bbb", alpha=0.92))
  axs[2].grid(alpha=0.3)

  fig.suptitle(f"RobStride motor {a.motor} - rotor inertia at the output shaft "
               f"(bare shaft, warm)", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"그림: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
