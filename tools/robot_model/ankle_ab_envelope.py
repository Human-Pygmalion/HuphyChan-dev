#!/usr/bin/env python3
"""발목 크랭크 두 개가 **함께** 만들 수 있는 자리를 표로 만든다 (AB 공간).

2026-09-13 · 사용자 설계 지시(`docs/139`)의 "Table" 이 이 파일이 만드는 것입니다.

## 왜 표가 필요한가

발목은 크랭크 A·B 두 모터가 발판 하나를 미는 닫힌 기구입니다. 그래서 **A 가 얼마인지에
따라 B 가 갈 수 있는 범위가 달라집니다.** A 와 B 를 따로따로 자르면(지금 HUPHY 가
그럼) 각자는 범위 안인데 **조합이 발판을 ROM 밖으로** 보냅니다.

    A 가 +50 일 때 B 는 -12 ~ +31 까지만  (예시)
    A 가   0 일 때 B 는 -47 ~ +47 까지     (예시)

이 "A 마다의 B 구간" 을 미리 계산해 두면, 제어 주기 안에서는 **표를 읽어 자르기만**
하면 됩니다. 기구학을 매 주기 여러 번 푸는 것보다 싸고, 무엇보다 **결과가 예측
가능**합니다.

## 무엇을 계산하는가

크랭크 각 (a, b) 격자마다:

    1. `solve_fk(a, b)` 가 풀리는가        <- 로드가 닿지 않으면 애초에 불가
    2. 나온 (숙임, 기울임) 이 ROM 안인가   <- 발판이 갈 수 있는 자세인가

둘 다 참인 칸이 **허용 영역**입니다. 그 영역을

    b_lo[a], b_hi[a]   A 를 고정했을 때 B 가 갈 수 있는 최소·최대
    a_lo[b], a_hi[b]   반대 방향

두 벌로 저장합니다. 두 벌인 이유는 **어느 쪽이 먼저 한계에 걸렸느냐에 따라 읽는
표가 다르기** 때문입니다 (docs/139 2절).

## 구멍이 있는지 반드시 본다

"A 를 고정했을 때 갈 수 있는 B" 가 **하나의 이어진 구간**이어야 `[lo, hi]` 두 숫자로
적을 수 있습니다. 중간에 끊긴 곳(구멍)이 있으면 그 표는 거짓말을 합니다. 그래서
계산할 때마다 **끊김을 세어 보고 보고**합니다.

    ankle_ab_envelope.py --leg left --out tools/robot_model/ankle_ab_envelope_left.json
    ankle_ab_envelope.py --leg left --plot docs/img/ankle_ab_envelope.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HUPHY_SRC = Path("/home/syaro/external_repos/HUPHY/src")
if HUPHY_SRC.exists():
  sys.path.insert(0, str(HUPHY_SRC))

ROM_PITCH = (-50.0, 30.0)
ROM_ROLL = (-20.0, 20.0)
"""발판이 갈 수 있는 자세. **시뮬레이터 모델(MJCF)에서 읽은 값**입니다 —
`ankle_pitch` `[-0.872665, 0.523599] rad`, `ankle_roll` `[-0.349066, 0.349066] rad`."""

CRANK_LIMIT = 68.75
"""크랭크 모터 자체의 한계 (캘리브레이션 `limits_deg`). 격자는 이 안에서만 봅니다."""


def build(leg: str, step: float, rom_pitch, rom_roll, crank_limit: float):
  from huphy.kinematics.ankle import AnkleGeometry, AnkleKinematics

  geom = AnkleGeometry()
  if leg == "left":
    geom = geom.mirrored()
  kin = AnkleKinematics(geom)

  n = int(round(2 * crank_limit / step)) + 1
  axis = [round(-crank_limit + i * step, 4) for i in range(n)]

  ok = [[False] * n for _ in range(n)]      # ok[i][j] : a=axis[i], b=axis[j]
  pitch = [[float("nan")] * n for _ in range(n)]
  roll = [[float("nan")] * n for _ in range(n)]
  unsolvable = 0
  for i, a in enumerate(axis):
    for j, b in enumerate(axis):
      try:
        p, r = kin.solve_fk(a, b)
      except Exception:                                     # noqa: BLE001
        unsolvable += 1
        continue
      # ★ 되짚어 확인합니다. 뉴턴 반복은 추정이 멀면 **다른 자세로 수렴**하고,
      # 그 값을 그대로 표에 적으면 표가 "갈 수 있다" 고 거짓말을 합니다
      # (2026-09-13: 이 확인 없이 구운 표 때문에 클리퍼가 4/2000 를 놓쳤음).
      if abs(p) > 90.0 or abs(r) > 90.0:
        # 다중회전 답 (+5840도 같은). IK 로 되짚으면 360 배수라 통과해 버립니다.
        unsolvable += 1
        continue
      try:
        ba, bb = kin.solve_ik(p, r, enforce_envelope=False)
      except Exception:                                     # noqa: BLE001
        unsolvable += 1
        continue
      if abs(ba - a) > 0.05 or abs(bb - b) > 0.05:
        unsolvable += 1
        continue
      pitch[i][j], roll[i][j] = p, r
      ok[i][j] = (rom_pitch[0] <= p <= rom_pitch[1]
                  and rom_roll[0] <= r <= rom_roll[1])

  def spans(mask_rows):
    """행마다 참인 칸의 [처음, 끝] 과 **끊긴 횟수**."""
    lo, hi, breaks = [], [], 0
    for row in mask_rows:
      idx = [k for k, v in enumerate(row) if v]
      if not idx:
        lo.append(None)
        hi.append(None)
        continue
      lo.append(axis[idx[0]])
      hi.append(axis[idx[-1]])
      # 이어져 있는가: 처음과 끝 사이에 거짓이 있으면 구멍
      breaks += sum(1 for k in range(idx[0], idx[-1]) if not row[k])
    return lo, hi, breaks

  b_lo, b_hi, holes_b = spans(ok)                                   # a 고정
  cols = [[ok[i][j] for i in range(n)] for j in range(n)]
  a_lo, a_hi, holes_a = spans(cols)                                 # b 고정

  inside = sum(sum(1 for v in row if v) for row in ok)
  return dict(
    leg=leg, step=step, axis=axis, crank_limit=crank_limit,
    rom_pitch=list(rom_pitch), rom_roll=list(rom_roll),
    b_lo=b_lo, b_hi=b_hi, a_lo=a_lo, a_hi=a_hi,
    stats=dict(cells=n * n, inside=inside, unsolvable=unsolvable,
               holes_row=holes_b, holes_col=holes_a),
    _ok=ok, _pitch=pitch, _roll=roll,
  )


def draw(table, out_png: str) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  axis = np.array(table["axis"])
  ok = np.array(table["_ok"], dtype=bool)
  pitch = np.array(table["_pitch"], dtype=float)
  roll = np.array(table["_roll"], dtype=float)
  lim = table["crank_limit"]

  fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.8))

  # -- (1) 허용 영역과 경계표 -------------------------------------------
  # ok[i][j] : i=a, j=b  ->  imshow 는 (row=y, col=x) 이므로 전치해서 x=a, y=b
  ax[0].imshow(ok.T, origin="lower", cmap="Greens", alpha=0.55,
               extent=[-lim, lim, -lim, lim], aspect="equal", vmin=0, vmax=1.4)
  a_ax = axis
  b_lo = np.array([np.nan if v is None else v for v in table["b_lo"]], dtype=float)
  b_hi = np.array([np.nan if v is None else v for v in table["b_hi"]], dtype=float)
  ax[0].plot(a_ax, b_lo, lw=1.6, color="tab:green", label="b_lo(a) / b_hi(a)  — the table")
  ax[0].plot(a_ax, b_hi, lw=1.6, color="tab:green")
  ax[0].add_patch(plt.Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False,
                                ec="0.35", lw=1.2, ls="--"))
  ax[0].text(-lim + 2, lim - 6, "motor limit box  ±68.75°", color="0.35", fontsize=9)

  # 예시 하나: A 가 상한에 걸리고 B 가 밖으로 나간 명령
  a_cmd, b_cmd = 62.0, 40.0
  i = int(np.argmin(np.abs(a_ax - min(a_cmd, lim))))
  a_c = float(a_ax[i])
  b_c = float(np.clip(b_cmd, b_lo[i], b_hi[i]))
  ax[0].plot([a_cmd], [b_cmd], "x", ms=11, mew=2.4, color="tab:red")
  ax[0].annotate("", xy=(a_c, b_c), xytext=(a_cmd, b_cmd),
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
  ax[0].plot([a_c], [b_c], "o", ms=7, color="tab:red")
  ax[0].annotate(f"commanded ({a_cmd:.0f}, {b_cmd:.0f})", (a_cmd, b_cmd),
                 textcoords="offset points", xytext=(-6, 10), color="tab:red", fontsize=9,
                 ha="right")
  ax[0].annotate(f"applied ({a_c:.1f}, {b_c:.1f})", (a_c, b_c),
                 textcoords="offset points", xytext=(8, -14), color="tab:red", fontsize=9)
  # 복귀 방향은 열려 있다는 표시
  ax[0].annotate("", xy=(a_c, b_c - 26), xytext=(a_c, b_c - 2),
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6))
  ax[0].text(a_c - 1, b_c - 30, "return direction\nstays free", color="tab:blue",
             fontsize=9, ha="right")

  ax[0].set_xlabel("crank A  [deg]")
  ax[0].set_ylabel("crank B  [deg]")
  ax[0].set_title("Reachable pairs in AB space, and the clip that respects them\n"
                  f"green = FK solves AND foot inside ROM (pitch {table['rom_pitch']}, "
                  f"roll {table['rom_roll']})", fontsize=10)
  ax[0].legend(loc="lower left", fontsize=8)
  ax[0].grid(alpha=.25)

  # -- (2) 왜 축마다 자르면 모자란가 -------------------------------------
  # ★ 기구학이 못 푼 칸은 말도 안 되는 값을 냅니다 (수만 도). 그대로 그리면 색이
  # 전부 한쪽으로 몰려 아무것도 안 보입니다. 가릴 것은 가리고 그립니다.
  shown = np.where(np.abs(pitch) <= 120.0, pitch, np.nan)
  cs = ax[1].contourf(axis, axis, shown.T, levels=np.linspace(-90, 90, 19),
                      cmap="RdBu_r", alpha=.85, extend="both")
  fig.colorbar(cs, ax=ax[1], label="foot pitch  [deg]")
  shown_roll = np.where(np.abs(roll) <= 120.0, roll, np.nan)
  ax[1].contour(axis, axis, shown.T, levels=[table["rom_pitch"][0], table["rom_pitch"][1]],
                colors="k", linewidths=1.6)
  ax[1].contour(axis, axis, shown_roll.T,
                levels=[table["rom_roll"][0], table["rom_roll"][1]],
                colors="k", linewidths=1.0, linestyles=":")
  ax[1].add_patch(plt.Rectangle((-22.75, -22.75), 45.5, 45.5, fill=False,
                                ec="tab:purple", lw=1.6))
  ax[1].text(-22.0, 24.0, "our per-axis trim ±22.75°", color="tab:purple", fontsize=9)
  ax[1].plot([53.8], [-53.8], "x", ms=11, mew=2.4, color="k")
  ax[1].annotate("both axes 'legal' at ±53.8°,\nfoot pitches 67°  (measured)",
                 (53.8, -53.8), textcoords="offset points", xytext=(-10, 14),
                 fontsize=9, ha="right")
  ax[1].set_xlabel("crank A  [deg]")
  ax[1].set_ylabel("crank B  [deg]")
  ax[1].set_title("Per-axis limits cannot express this: the foot pose is a function\n"
                  "of BOTH cranks (solid = pitch ROM edge, dotted = roll ROM edge)",
                  fontsize=10)
  ax[1].set_aspect("equal")
  ax[1].grid(alpha=.25)

  fig.suptitle(f"Ankle AB envelope — {table['leg']} leg, real 2-RSU kinematics, "
               f"{table['step']}° grid", fontsize=11)
  fig.tight_layout()
  Path(out_png).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_png, dpi=140)
  print(f"  그림 -> {out_png}")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--leg", choices=("left", "right"), default="left")
  ap.add_argument("--step", type=float, default=0.5, help="격자 간격 (도)")
  ap.add_argument("--crank-limit", type=float, default=CRANK_LIMIT)
  ap.add_argument("--out", default=None, help="표를 쓸 json")
  ap.add_argument("--plot", default=None, help="그림을 쓸 png")
  a = ap.parse_args(argv)

  t = build(a.leg, a.step, ROM_PITCH, ROM_ROLL, a.crank_limit)
  s = t["stats"]
  print(f"  {a.leg} 다리 · 격자 {a.step}도 · 칸 {s['cells']}개")
  print(f"    허용 {s['inside']} ({100.0 * s['inside'] / s['cells']:.1f}%) · "
        f"기구학이 못 푼 칸 {s['unsolvable']}")
  print(f"    구간이 끊긴 칸: A 고정 {s['holes_row']} · B 고정 {s['holes_col']} "
        f"({'표로 쓸 수 있음' if s['holes_row'] == s['holes_col'] == 0 else '★ 표가 거짓말함'})")

  if a.out:
    slim = {k: v for k, v in t.items() if not k.startswith("_")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    print(f"  표 -> {a.out}")
  if a.plot:
    draw(t, a.plot)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
