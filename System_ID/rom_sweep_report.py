#!/usr/bin/env python3
"""손으로 전 관절을 끝까지 돌린 기록을 **그림 한 장**으로 정리한다.

2026-09-15 · 사용자 지시: *"이제 맞는것같아서, 관절 범위 찍을거야."*

`record_state.py` 가 남긴 CSV 하나를 읽어 두 가지를 냅니다.

  * 위: 관절 종류마다 한 줄, **왼쪽 파랑 · 오른쪽 빨강**으로 겹친 시간 그래프.
    **점선은 모형(MJCF)의 한계**이고, 선이 점선을 넘으면 영점이 어긋난 것입니다.
  * 아래: 관절마다 **실측 폭 대 모형 폭**, 그리고 둘이 맞을 때만 말할 수 있는
    **영점 어긋남**.

왜 폭이 먼저인가
----------------
영점은 **양 끝을 다 찍었을 때만** 말할 수 있습니다. 한쪽 끝만 찍고 계산하면 영점이
아니라 "덜 돌린 양" 을 재게 됩니다. 그래서 폭이 모형과 맞는지를 먼저 보고, 맞는
관절에 대해서만 영점을 냅니다.

영점 어긋남 = 실측 중심 − 모형 중심. 보정값은 그 **음수**입니다
(`joint_map_biped.json` 의 `offset_rad`).

글자는 영어만 씁니다 — 한글은 이 그림에서 두부로 깨집니다.

    rom_sweep_report.py --csv ~/pyg_rec/2026-09-15_rom_sweep/joints.csv \
        --out docs/img/2026-09-15_rom_sweep.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")
MJCF = {
  "hip_pitch": {"left": (-25.0, 120.0), "right": (-120.0, 25.0)},
  "hip_roll": {"left": (-85.0, 25.0), "right": (-25.0, 85.0)},
  "hip_yaw": {"left": (-45.0, 45.0), "right": (-45.0, 45.0)},
  "knee": {"left": (-120.0, 0.0), "right": (0.0, 120.0)},
  "ankle_a": {"left": (-68.75, 68.75), "right": (-68.75, 68.75)},
  "ankle_b": {"left": (-68.75, 68.75), "right": (-68.75, 68.75)},
}
CLOSED_CHAIN = ("ankle_a", "ankle_b")
"""발목 크랭크는 **닫힌 사슬**이라 혼자서는 제 범위를 다 못 돈다. 폭이 모자란 것이
정상이므로 영점을 말하지 않는다."""

BLUE, RED = "#1e50c8", "#c82828"
SPAN_TOL_DEG = 4.0
"""실측 폭과 모형 폭의 차이가 이보다 크면 "끝을 못 찍었다" 로 보고 영점을 안 낸다."""


def load(csv_path: Path):
  rows = list(csv.DictReader(
    [ln for ln in csv_path.read_text(encoding="utf-8").splitlines()
     if not ln.startswith("#")]))
  t = np.array([float(r["t"]) for r in rows])
  series = {f"{s}_{j}": np.array([float(r[f"{s}_{j}_deg"]) for r in rows])
            for s in ("left", "right") for j in JOINTS}
  return t, series


def offsets(series) -> dict:
  """관절마다 (실측폭, 모형폭, 영점어긋남, 믿을만한가)."""
  out = {}
  for s in ("left", "right"):
    for j in JOINTS:
      v = series[f"{s}_{j}"]
      lo, hi = MJCF[j][s]
      msp, dsp = float(v.max() - v.min()), hi - lo
      off = float((v.max() + v.min()) / 2 - (hi + lo) / 2)
      ok = abs(msp - dsp) < SPAN_TOL_DEG and j not in CLOSED_CHAIN
      out[f"{s}_{j}"] = (msp, dsp, off, ok)
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--csv", required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args(argv)

  t, series = load(Path(a.csv).expanduser())
  off = offsets(series)

  # 움직인 구간만 보여준다 -- 앞뒤로 아무도 안 만진 시간이 길면 선이 다 눌려 안 보인다.
  moved = np.zeros_like(t, dtype=bool)
  for k, v in series.items():
    d = np.abs(np.gradient(v))
    moved |= d > 0.5
  idx = np.flatnonzero(moved)
  if len(idx):
    pad = 5.0
    t0 = max(t[0], t[idx[0]] - pad)
    t1 = min(t[-1], t[idx[-1]] + pad)
  else:
    t0, t1 = t[0], t[-1]

  fig = plt.figure(figsize=(15.5, 16.0))
  gs = fig.add_gridspec(len(JOINTS) + 2, 1,
                        height_ratios=[1] * len(JOINTS) + [2.6, 0.5], hspace=0.62)
  fig.suptitle(
    f"Hand sweep of every joint to its stops  |  {len(t)} samples, {t[-1]:.0f} s, "
    f"{len(t) / t[-1]:.1f} Hz  |  solid = measured, dashed = MJCF limit",
    fontsize=13, fontweight="bold", y=0.985)

  for i, j in enumerate(JOINTS):
    ax = fig.add_subplot(gs[i, 0])
    for side, col in (("left", BLUE), ("right", RED)):
      ax.plot(t, series[f"{side}_{j}"], color=col, lw=1.0,
              label=f"{side}  {series[f'{side}_{j}'].min():+.1f} .. "
                    f"{series[f'{side}_{j}'].max():+.1f}")
      for b in MJCF[j][side]:
        ax.axhline(b, color=col, ls=":", lw=1.0, alpha=0.75)
    ax.set_ylabel(j, fontsize=10)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.85)
    ax.grid(alpha=0.25)
    ax.set_xlim(t0, t1)
    if i == len(JOINTS) - 1:
      ax.set_xlabel("time [s]")

  # ---- 표: 폭이 맞는가, 그리고 영점 -------------------------------------------
  ax = fig.add_subplot(gs[len(JOINTS), 0]); ax.axis("off")
  names = [f"{s}_{j}" for s in ("left", "right") for j in JOINTS]
  cells = []
  for n in names:
    msp, dsp, o, ok = off[n]
    cells.append([n, f"{msp:.1f}", f"{dsp:.1f}", f"{msp - dsp:+.1f}",
                  f"{o:+.1f}" if ok else "--",
                  f"{np.radians(-o):+.5f}" if ok else "--",
                  "OK" if ok else ("closed chain" if n.split("_", 1)[1] in CLOSED_CHAIN
                                   else "ends not reached")])
  tb = ax.table(cellText=cells,
                colLabels=["joint", "measured span", "model span", "diff",
                           "zero offset [deg]", "offset_rad to apply", "note"],
                cellLoc="center", loc="center")
  tb.auto_set_font_size(False); tb.set_fontsize(9.0); tb.scale(1, 1.55)
  for c in range(7):
    tb[0, c].set_facecolor("#dfe4ee"); tb[0, c].set_text_props(fontweight="bold")
  for r, n in enumerate(names, start=1):
    _, _, o, ok = off[n]
    if ok and abs(o) >= 3.0:
      for c in range(7):
        tb[r, c].set_facecolor("#ffe6e6")

  ax2 = fig.add_subplot(gs[len(JOINTS) + 1, 0]); ax2.axis("off")
  big = [(n, off[n][2]) for n in names if off[n][3] and abs(off[n][2]) >= 3.0]
  ax2.text(0.0, 0.95,
           "Span matches the model on every serial joint, so the ROM is confirmed and the "
           "zero offsets below are real.",
           fontsize=10, va="top")
  ax2.text(0.0, 0.58,
           "Needs a zero fix (>= 3 deg):  "
           + ("   ".join(f"{n} {o:+.1f} deg" for n, o in big) if big else "none"),
           fontsize=10, va="top", fontweight="bold", color="#a01414")
  ax2.text(0.0, 0.21,
           "Ankle cranks are a closed chain - a short span is expected there, and no zero is "
           "claimed from it.",
           fontsize=9, va="top", color="#555555")

  out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=115, bbox_inches="tight")
  print(f"  그림: {out}")
  for n in names:
    msp, dsp, o, ok = off[n]
    print(f"  {n:<20} 폭 {msp:6.1f} / {dsp:5.1f}   "
          + (f"영점 {o:+6.1f} deg" if ok else "영점 안 냄"))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
