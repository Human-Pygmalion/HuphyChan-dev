#!/usr/bin/env python3
"""**ROM 을 벗어나는 명령 전 영역**을 훑어서, 자르는 로직이 정말 막는지 영상으로 만든다.

2026-09-13 · 사용자 지시: "ROM 벗어나는 전 영역에 대해 Clipping 로직이 잘 막는지
영상으로 정리."

    mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/ankle_clip_sweep_video.py \
        --step 3 --out docs/img/2026-09-13_ankle_clip_sweep.mp4

## 무엇을 보여 주나

크랭크 A·B 명령을 ±90도 네모 전체에 격자로 깔고, 한 줄씩 넣습니다. 점 색은 결과입니다.

    회색   명령이 이미 허용 영역 안 -- 손댈 것이 없었음
    파랑   영역 밖이었고 **잘라서 안으로 들어왔음**
    빨강   영역 밖인데 **못 막았음**

왼쪽은 크랭크 공간(명령이 들어온 자리), 오른쪽은 발판 자세 공간(잘린 결과가 실제로
세우는 자세)입니다. 방식마다 한 장면씩, 마지막에 나란히 비교합니다.

글자는 그림 위 띠에 한글로 올립니다 -- 그림 안 라벨은 영어입니다(한글은 두부가 됨).

## 주의

허용 영역의 **기울임 축은 지금 틀렸습니다** (`docs/141` 3절: 우리 기구학이 실제보다
약 1.6배 작게 봅니다). 이 영상은 "로직이 **자기가 믿는 영역**을 잘 지키는가" 를
보여 주는 것이고, 그 영역 자체가 맞는지는 별 문제입니다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from ankle_clip import AnkleClipper                                    # noqa: E402
import ankle_clip_strategies as S                                      # noqa: E402

KOREAN = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
KOREAN_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

GREY, BLUE, RED = "#9aa0a6", "#1a73e8", "#d93025"
HEAD_H = 118
FPS = 12


def sweep(strategy: S.Strategy, axis: np.ndarray):
  """격자 전체를 넣고, **줄 단위로** 결과를 모은다. (행별 결과 목록)"""
  rows = []
  for a in axis:
    row = []
    for b in axis:
      r = strategy.clip(float(a), float(b))
      if isinstance(strategy, S.BoosterTorque):
        strategy.accept(r.a_out, r.b_out)
      kind = "grey" if not r.changed and r.inside else ("blue" if r.inside else "red")
      row.append((float(a), float(b), r.a_out, r.b_out, r.pitch, r.roll, kind))
    rows.append(row)
  return rows


def draw(rows, upto: int, clipper: AnkleClipper, title_en: str, counters):
  """0..upto 줄까지 그린 그림 하나를 RGB 배열로."""
  fig, ax = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=110)
  lim = 95.0

  # 왼쪽: 크랭크 공간
  ax[0].set_title(f"crank space - commanded  ({title_en})", fontsize=10)
  ax[0].set_xlabel("crank A [deg]"); ax[0].set_ylabel("crank B [deg]")
  ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim); ax[0].set_aspect("equal")
  ax[0].grid(alpha=0.25, lw=0.4)
  if clipper.table:
    axis = np.array(clipper.table["axis"], dtype=float)
    lo = np.array([np.nan if v is None else v for v in clipper.table["b_lo"]], dtype=float)
    hi = np.array([np.nan if v is None else v for v in clipper.table["b_hi"]], dtype=float)
    ok = ~np.isnan(lo)
    ax[0].fill_between(axis[ok], lo[ok], hi[ok], color="#34a853", alpha=0.18,
                       label="reachable (in ROM)")
    ax[0].legend(fontsize=8, loc="upper left")

  # 오른쪽: 발판 자세 공간
  ax[1].set_title("foot pose space - after clipping", fontsize=10)
  ax[1].set_xlabel("foot pitch [deg]"); ax[1].set_ylabel("foot roll [deg]")
  ax[1].set_xlim(-70, 50); ax[1].set_ylim(-45, 45)
  ax[1].grid(alpha=0.25, lw=0.4)
  p0, p1 = clipper.rom_pitch
  r0, r1 = clipper.rom_roll
  ax[1].add_patch(plt.Rectangle((p0, r0), p1 - p0, r1 - r0, facecolor="#34a853",
                                alpha=0.18, edgecolor="#34a853", lw=1.0))
  ax[1].text(p0 + 1, r1 + 2, "ROM box", fontsize=8, color="#188038")

  for key, col, sz in (("grey", GREY, 3), ("blue", BLUE, 4), ("red", RED, 14)):
    xs = [c[0] for row in rows[:upto] for c in row if c[6] == key]
    ys = [c[1] for row in rows[:upto] for c in row if c[6] == key]
    ax[0].scatter(xs, ys, s=sz, c=col, marker="s" if key != "red" else "x",
                  linewidths=1.0 if key == "red" else 0)
    ps = [c[4] for row in rows[:upto] for c in row if c[6] == key and c[4] == c[4]]
    rs = [c[5] for row in rows[:upto] for c in row if c[6] == key and c[5] == c[5]]
    ax[1].scatter(ps, rs, s=sz, c=col, marker="s" if key != "red" else "x",
                  linewidths=1.0 if key == "red" else 0)

  n = sum(len(r) for r in rows[:upto])
  ng = counters["grey"]; nb = counters["blue"]; nr = counters["red"]
  ax[1].text(0.98, 0.02,
             f"commands {n}\nuntouched {ng}\nclipped in {nb}\nESCAPED {nr}",
             transform=ax[1].transAxes, ha="right", va="bottom", fontsize=9,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d0d0d0"))
  fig.tight_layout()
  fig.canvas.draw()
  buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  return buf


def compose(frame: np.ndarray, caption: str, sub: str, tag: str):
  from PIL import Image, ImageDraw, ImageFont
  h, w = frame.shape[:2]
  canvas = Image.new("RGB", (w, h + HEAD_H), (24, 24, 27))
  canvas.paste(Image.fromarray(frame), (0, HEAD_H))
  d = ImageDraw.Draw(canvas)
  big = ImageFont.truetype(KOREAN_BOLD, 27)
  mid = ImageFont.truetype(KOREAN, 19)
  small = ImageFont.truetype(KOREAN, 16)
  d.text((16, 10), "발목 크랭크 명령 자르기 — 허용 영역을 벗어나는 전 영역 훑기",
         font=small, fill=(150, 150, 155))
  d.text((16, 34), caption, font=big, fill=(255, 236, 150))
  d.text((16, 72), sub, font=mid, fill=(232, 232, 234))
  d.text((w - 240, 12), tag, font=small, fill=(150, 150, 155))
  return np.asarray(canvas)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--step", type=float, default=3.0, help="격자 간격 [도]")
  ap.add_argument("--span", type=float, default=90.0, help="명령 범위 ±이 값 [도]")
  ap.add_argument("--leg", default="left", choices=("left", "right"))
  ap.add_argument("--out", default="docs/img/2026-09-13_ankle_clip_sweep.mp4")
  ap.add_argument("--strategies", nargs="*", default=["ours_pocs", "unitree_pr",
                                                      "booster_torque"])
  args = ap.parse_args(argv)

  axis = np.arange(-args.span, args.span + 1e-9, args.step)
  clipper = AnkleClipper(args.leg)
  print(f"  격자 {len(axis)}x{len(axis)} = {len(axis)**2}개 명령 · 방식 {len(args.strategies)}개")

  results = {}
  for key in args.strategies:
    st = S.make(key, clipper)
    st.reset()
    rows = sweep(st, axis)
    results[key] = rows
    tot = {"grey": 0, "blue": 0, "red": 0}
    for row in rows:
      for c in row:
        tot[c[6]] += 1
    print(f"  {key:<16} 손안댐 {tot['grey']:>5} · 잘라서 안 {tot['blue']:>5} · "
          f"★ 못막음 {tot['red']:>5}  ({100*tot['red']/max(1,sum(tot.values())):.2f}%)")

  frames = []
  with tempfile.TemporaryDirectory() as td:
    idx = 0
    for key in args.strategies:
      rows = results[key]
      counters = {"grey": 0, "blue": 0, "red": 0}
      for upto in range(1, len(rows) + 1):
        for c in rows[upto - 1]:
          counters[c[6]] += 1
        if upto % 2 and upto != len(rows):
          continue                                   # 두 줄에 한 장만 -- 영상 길이
        img = compose(draw(rows, upto, clipper, key, counters),
                      S.NAMES[key],
                      f"명령 {sum(len(r) for r in rows[:upto])}개 넣음 · "
                      f"잘라서 안으로 {counters['blue']} · 못 막음 {counters['red']}",
                      f"crank A = {rows[upto-1][0][0]:+.0f}deg")
        p = Path(td) / f"f{idx:05d}.png"
        PILImage.fromarray(img).save(p)
        frames.append(p); idx += 1
      # 각 방식 끝에서 잠깐 멈춤
      last = frames[-1]
      for _ in range(FPS):
        p = Path(td) / f"f{idx:05d}.png"
        p.write_bytes(last.read_bytes()); frames.append(p); idx += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-pattern_type", "glob",
           "-i", str(Path(td) / "f*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
      print(r.stderr[-1500:]); return 1
  print(f"  영상: {out}  ({len(frames)}장 · {len(frames)/FPS:.1f}초)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
