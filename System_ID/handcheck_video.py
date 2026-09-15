#!/usr/bin/env python3
"""손으로 돌린 기록을 **영상 + 관절 그림**으로 묶는다.

2026-09-15 · 사용자 지시: *"로봇 동작범위 내에서 손으로 돌려볼테니 데이터 기록해.
영상도 찍고."*

`record_cam.py` 가 남긴 사진과 `record_state.py` 가 남긴 CSV 를 **벽시계로 맞춰**
한 화면에 붙입니다. 왼쪽이 실물 카메라, 오른쪽이 그 순간의 관절 각도입니다.

  * 관절 종류마다 한 줄, **왼쪽은 파랑 · 오른쪽은 빨강**으로 겹칩니다
  * **점선은 모형(MJCF)의 한계** — 실물이 그 선을 넘으면 영점이 어긋난 것입니다
  * 지금 값을 숫자로도 적고, 한계를 넘은 값은 **빨간 글씨**로 표시합니다

영상은 **실시간**입니다 (찍은 초당 장수 그대로). 빨리 감지 않습니다.

    handcheck_video.py --dir ~/pyg_rec/2026-09-15_handcheck \
        --out docs/video/2026-09-15_handcheck.mp4 --from-s 450
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")
MJCF = {
  "hip_pitch": {"left": (-25, 120), "right": (-120, 25)},
  "hip_roll": {"left": (-85, 25), "right": (-25, 85)},
  "hip_yaw": {"left": (-45, 45), "right": (-45, 45)},
  "knee": {"left": (-120, 0), "right": (0, 120)},
  "ankle_a": {"left": (-68.75, 68.75), "right": (-68.75, 68.75)},
  "ankle_b": {"left": (-68.75, 68.75), "right": (-68.75, 68.75)},
}
BLUE, RED, GRAY, BG = (30, 80, 200), (200, 40, 40), (150, 150, 150), (255, 255, 255)
PANEL_W = 560
WIN_S = 20.0   # 그림에 보이는 시간 창


def _font(sz):
  try:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
  except Exception:                                         # noqa: BLE001
    return ImageFont.load_default()


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dir", required=True, help="record_cam/record_state 가 쓴 폴더")
  ap.add_argument("--out", required=True)
  ap.add_argument("--from-s", type=float, default=0.0, help="이 초 이전은 버린다")
  ap.add_argument("--height", type=int, default=540)
  a = ap.parse_args(argv)

  root = Path(a.dir).expanduser()
  rows = list(csv.DictReader(
    [ln for ln in (root / "joints.csv").read_text(encoding="utf-8").splitlines()
     if not ln.startswith("#")]))
  ct = np.array([float(r["t"]) for r in rows])
  cw = np.array([float(r["wall"]) for r in rows])
  series = {f"{s}_{j}": np.array([float(r[f"{s}_{j}_deg"]) for r in rows])
            for s in ("left", "right") for j in JOINTS}

  times = {}
  for ln in (root / "cam" / "times.txt").read_text(encoding="utf-8").splitlines():
    i, w = ln.split()
    times[int(i)] = float(w)
  idx = sorted(times)
  t0_wall = cw[0]
  keep = [i for i in idx if times[i] - t0_wall >= a.from_s]
  print(f"  사진 {len(idx)}장 중 {len(keep)}장 사용 (앞 {a.from_s:.0f}초 버림)")
  if not keep:
    print("  쓸 사진이 없습니다"); return 1
  fps = round(len(keep) / max(times[keep[-1]] - times[keep[0]], 1e-6))
  print(f"  실시간 초당 {fps}장으로 묶습니다")

  f_lab, f_sm = _font(13), _font(11)
  h = a.height
  head_h, row_h = 26, (h - 26) // len(JOINTS)

  with tempfile.TemporaryDirectory() as td:
    for n, fi in enumerate(keep):
      wall = times[fi]
      k = int(np.searchsorted(cw, wall))
      k = min(max(k, 0), len(ct) - 1)
      cam = Image.open(root / "cam" / f"f{fi:06d}.jpg")
      cam = cam.resize((int(cam.width * h / cam.height), h))
      panel = Image.new("RGB", (PANEL_W, h), BG)
      dr = ImageDraw.Draw(panel)
      dr.text((6, 4), f"t = {ct[k]:7.1f} s   |   solid = measured, dashed = MJCF limit",
              fill=(0, 0, 0), font=f_sm)
      dr.text((PANEL_W - 96, 4), "LEFT", fill=BLUE, font=f_sm)
      dr.text((PANEL_W - 52, 4), "RIGHT", fill=RED, font=f_sm)
      lo_k = int(np.searchsorted(ct, ct[k] - WIN_S))
      for i, j in enumerate(JOINTS):
        y0, y1 = head_h + i * row_h + 13, head_h + (i + 1) * row_h - 3
        lo = min(MJCF[j][s][0] for s in ("left", "right"))
        hi = max(MJCF[j][s][1] for s in ("left", "right"))
        pad = 0.10 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def px(v, y0=y0, y1=y1, lo=lo, hi=hi):
          return y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)

        dr.rectangle([54, y0, PANEL_W - 6, y1], outline=(215, 215, 215))
        dr.text((5, y0 - 12), j, fill=(0, 0, 0), font=f_lab)
        xs = np.linspace(54, PANEL_W - 6, max(k - lo_k + 1, 2))
        txt = []
        for side, col in (("left", BLUE), ("right", RED)):
          v = series[f"{side}_{j}"]
          seg = v[lo_k:k + 1]
          if len(seg) > 1:
            dr.line(list(zip(xs[:len(seg)], px(seg), strict=False)), fill=col, width=2)
          a_, b_ = MJCF[j][side]
          for bound in (a_, b_):
            yy = px(bound)
            for x in range(54, PANEL_W - 6, 8):
              dr.line([x, yy, x + 4, yy], fill=col, width=1)
          now = float(v[k])
          over = now < a_ - 0.5 or now > b_ + 0.5
          txt.append((f"{side[0].upper()} {now:+7.1f}", RED if over else col, over))
        x = 60
        for s_, c_, over in txt:
          dr.text((x, y0 - 12), s_ + ("  OVER" if over else ""), fill=c_, font=f_sm)
          x += 110
      out = Image.new("RGB", (cam.width + PANEL_W, h), (24, 24, 27))
      out.paste(cam, (0, 0)); out.paste(panel, (cam.width, 0))
      out.save(Path(td) / f"g{n:06d}.png")
      if n % 500 == 0:
        print(f"    {n}/{len(keep)}장", flush=True)
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
      ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob",
       "-i", str(Path(td) / "g*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-crf", "26", str(dst)], capture_output=True, text=True)
    if r.returncode:
      print(r.stderr[-1200:]); return 1
  print(f"  영상: {dst}  ({len(keep)}장 · 초당 {fps}장 · {len(keep) / fps:.0f}초)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
