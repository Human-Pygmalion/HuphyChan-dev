#!/usr/bin/env python3
"""한 바퀴를 아주 느리게 왕복해서 **코깅을 마찰과 갈라낸다.**

2026-09-10 · 사용자 지시: **"한 바퀴 촘촘히 훑어서 코깅 맞는지 확인해줘."**

무엇이 문제인가
---------------
±10도 실험에서 **방향에 따라 멈추는 자리가 다르다**는 것을 봤고, 그 쏠림이 무릎에서는 축의
자리에 따라 부호까지 바뀌었습니다. 코깅(자석과 철심이 만드는 톱니 같은 힘)의 모양이지만,
다섯 자리만 봤으므로 **단정할 수 없었습니다.** 코깅이라면 **한 바퀴에 톱니가 몇 개**인지가
정해져 있고, 그 개수는 극 수·슬롯 수와 감속비로 예측됩니다. 그러니 한 바퀴를 촘촘히 훑어
**주기를 세면** 확인이 됩니다.

어떻게 가르는가 — 마찰을 지우는 뺄셈
------------------------------------
**아주 느린 일정 속도**로 돌리면 관성이 안 끼고 점성도 미미합니다. 그러면 제어기가 내는 힘이
곧 그 자리에서 필요한 힘입니다.

    필요한 힘(θ) = 마찰 x 진행방향 + 코깅힘(θ)

마찰은 **방향을 따라 부호가 뒤집히고**, 코깅은 자리의 함수라 **안 뒤집힙니다.** 그래서 같은
자리를 양방향으로 지나가 두 값을 더하고 빼면 둘이 깔끔하게 갈립니다.

    코깅(θ) = (갈 때 + 올 때) ÷ 2        마찰(θ) = (갈 때 − 올 때) ÷ 2

이 뺄셈은 우리가 회전자 무게감을 잴 때 쓴 것과 **같은 수법**입니다 — 두 상황을 만들어 원하지
않는 항을 소거하는 것.

톱니를 어떻게 세는가
--------------------
자리별로 정리한 코깅힘을 **주파수로 분해**합니다(푸리에). 봉우리가 서는 자리가 곧 "한 바퀴에
톱니 몇 개"입니다. 출력축은 감속기를 거치므로, **회전자 기준 톱니 수 = 출력축 톱니 수 ÷ 감속비
(9)** 입니다. 회전자 기준 수가 극 수·슬롯 수와 어울리는 정수로 떨어지면 코깅이 맞습니다.

    pd_cogging.py --motor 4 --family RS04
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
import numpy as np  # noqa: E402

GEAR = 9.0            # 두 모터 모두 9:1 (데이터시트 확인, 2026-09-10)
FRICTION = {"RS04": 0.519, "RS03": 0.251}


def load(pattern):
  out = []
  for p in sorted(globmod.glob(pattern)):
    with open(p) as f:
      r = list(csv.DictReader(f))
    if len(r) < 2000:
      continue
    out.append(dict(
        t=np.array([float(x["t_s"]) for x in r]),
        tgv=np.array([float(x["target_vel_deg_s"]) for x in r]),
        pos=np.array([float(x["pos_deg"]) for x in r]),
        vel=np.array([float(x["vel_deg_s"]) for x in r]),
        tau=np.array([float(x["tau_cmd_nm"]) for x in r])))
  return out


def bin_by_position(runs, lo, hi, step, want_sign):
  """한 방향으로 지나간 구간만 골라 **자리별로 힘을 평균**한다.

  방향이 바뀌는 자리(±180도 근처)는 가속이 끼므로 뺍니다 — 목표 속도가 최고치의 90%에
  못 미치는 구간을 버립니다."""
  edges = np.arange(lo, hi + step, step)
  acc = np.zeros(len(edges) - 1)
  cnt = np.zeros(len(edges) - 1)
  for r in runs:
    vmax = np.abs(r["tgv"]).max()
    m = (np.sign(r["tgv"]) == want_sign) & (np.abs(r["tgv"]) > 0.9 * vmax)
    if m.sum() < 100:
      continue
    idx = np.digitize(r["pos"][m], edges) - 1
    ok = (idx >= 0) & (idx < len(acc))
    np.add.at(acc, idx[ok], r["tau"][m][ok])
    np.add.at(cnt, idx[ok], 1.0)
  centres = 0.5 * (edges[:-1] + edges[1:])
  with np.errstate(invalid="ignore", divide="ignore"):
    mean = acc / cnt
  return centres, mean, cnt


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--family", default="RS04", choices=list(FRICTION))
  ap.add_argument("--amp", type=float, default=180.0)
  ap.add_argument("--tag", default="cog")
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--step", type=float, default=0.25, help="자리를 몇 도 단위로 묶을까")
  ap.add_argument("--margin", type=float, default=8.0, help="양 끝에서 버릴 각도")
  ap.add_argument("--out-dir", default="../docs/img")
  a = ap.parse_args(argv)

  pat = (f"pd_triangle_{a.motor}_kp{a.kp:g}_kd{a.kd:g}_a{a.amp:g}_r*_{a.tag}.csv")
  runs = load(os.path.join(a.data, pat))
  if not runs:
    print(f"자료가 없습니다: {pat}")
    return 1
  lo, hi = -a.amp + a.margin, a.amp - a.margin
  cen, f_fwd, n_fwd = bin_by_position(runs, lo, hi, a.step, +1)
  _, f_bwd, n_bwd = bin_by_position(runs, lo, hi, a.step, -1)
  ok = (n_fwd > 5) & (n_bwd > 5) & np.isfinite(f_fwd) & np.isfinite(f_bwd)
  cen, f_fwd, f_bwd = cen[ok], f_fwd[ok], f_bwd[ok]
  cog = 0.5 * (f_fwd + f_bwd)
  fric = 0.5 * (f_fwd - f_bwd)
  cog0 = cog - cog.mean()          # 일정한 치우침은 따로 뺀다 (그건 코깅이 아니다)

  print(f"{a.family} (모터 {a.motor}) · {len(runs)}회 · 자리 {cen[0]:.0f}~{cen[-1]:.0f}도 · "
        f"{len(cen)}칸({a.step}도) · 칸마다 {int(np.median(n_fwd[n_fwd>0]))}/"
        f"{int(np.median(n_bwd[n_bwd>0]))} 표본")
  print(f"  마찰(자리 평균) {fric.mean():+.4f} 뉴턴미터 "
        f"(따로 잰 값 {FRICTION[a.family]:.3f}) · 자리에 따른 흔들림 ±{fric.std():.4f}")
  print(f"  일정한 치우침 {cog.mean():+.4f} 뉴턴미터 · "
        f"자리에 따라 변하는 몫 ±{cog0.std():.4f} (최대 {np.abs(cog0).max():.4f})")

  # ---------------------------------------------------------------- 주파수
  # 자리에 대한 함수이므로, 푸리에의 "주파수"는 **한 바퀴에 톱니 몇 개**가 된다.
  span = cen[-1] - cen[0]
  y = cog0 - cog0.mean()
  win = np.hanning(len(y))
  sp = np.abs(np.fft.rfft(y * win)) / (len(y) / 2) * 2.0
  # 인덱스 k = 잰 구간(span도)에 톱니 k 개 → 한 바퀴(360도)로 환산
  per_rev = np.arange(len(sp)) * (360.0 / span)
  keep = (per_rev > 2) & (per_rev < 0.5 * 360.0 / a.step)
  top = np.argsort(sp[keep])[::-1][:6]
  idxs = np.nonzero(keep)[0][top]
  print("  가장 센 주기들:")
  for i in idxs:
    n_out = per_rev[i]
    print(f"    출력축 한 바퀴에 {n_out:7.1f}개 (주기 {360.0/n_out:6.2f}도) · "
          f"세기 {sp[i]:.4f} 뉴턴미터 · 회전자 기준 {n_out/GEAR:6.2f}개")

  # ---------------------------------------------------------------- 그림
  fig, axs = plt.subplots(3, 1, figsize=(14.0, 11.0))
  ax = axs[0]
  ax.plot(cen, f_fwd, lw=1.0, alpha=0.75, label="torque needed going one way")
  ax.plot(cen, f_bwd, lw=1.0, alpha=0.75, label="torque needed going the other way")
  ax.axhline(0, color="#999", lw=0.8)
  ax.set_ylabel("torque [N m]"); ax.grid(alpha=0.28)
  ax.set_title(f"RobStride {a.family} (motor {a.motor}) - one slow turn each way, "
               f"{len(runs)} runs\nthe gap between the two curves is friction; what they share "
               f"is the position-dependent part", fontsize=11)
  ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

  ax = axs[1]
  ax.plot(cen, cog0, color="#7b3fbf", lw=1.0, alpha=0.9,
          label="(one way + other way) / 2, mean removed - this is the cogging-like part")
  ax.plot(cen, fric - fric.mean(), color="#00a878", lw=1.0, alpha=0.7,
          label="(one way - other way) / 2, mean removed - this is friction, and it barely moves")
  ax.axhline(0, color="#999", lw=0.8)
  ax.set_xlabel("where on the output shaft [deg]"); ax.set_ylabel("torque [N m]")
  ax.grid(alpha=0.28); ax.legend(fontsize=9, framealpha=0.9)
  ax.set_title("friction cancels in the sum, and what is left is a function of position",
               fontsize=11)

  ax = axs[2]
  ax.plot(per_rev[keep], sp[keep], color="#111111", lw=1.0)
  for i in idxs[:4]:
    ax.annotate(f"{per_rev[i]:.0f} per turn\n({360.0/per_rev[i]:.2f} deg)\n"
                f"rotor: {per_rev[i]/GEAR:.1f}",
                (per_rev[i], sp[i]), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
  ax.set_xlabel("bumps per turn of the output shaft"); ax.set_ylabel("size [N m]")
  ax.grid(alpha=0.28)
  ax.set_title("how many bumps per turn - cogging has to land on a whole number "
               f"of bumps per turn of the rotor (output number / {GEAR:g})", fontsize=11)
  fig.tight_layout()
  os.makedirs(os.path.abspath(a.out_dir), exist_ok=True)
  out = os.path.join(a.out_dir, f"pd_cogging_{a.family.lower()}.png")
  fig.savefig(out, dpi=125); plt.close(fig)
  print(f"그림: {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
