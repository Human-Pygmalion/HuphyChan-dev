#!/usr/bin/env python3
"""녹화한 사진들에 **그때 무슨 일이 있었는지**를 얹어 영상으로 만든다.

2026-09-12. 모터를 돌릴 때는 늘 녹화하는데, 사진만 쌓아 두면 나중에 아무도 안
봅니다. 무엇을 명령했고 관절이 어떻게 답했는지를 **그림 위에 같이** 얹어야 봅니다.

무엇을 얹는가

* 왼쪽 위: 그 순간의 각도·목표·토크 (제어 기록 csv 에서)
* 아래: 각도와 토크 그래프, 지금 어디인지 세로선
* 맨 아래: 사람이 쓴 자막 — 구간마다 한 줄

시각 맞추기
-----------
사진에는 받은 시각(`times.txt`, 유닉스 시각)이 있고, 제어 기록에는 프로그램
시작부터의 초가 있습니다. 둘을 잇는 기준점 하나(`--epoch0`)를 줘야 합니다 —
기록을 남긴 프로그램이 시작한 유닉스 시각입니다. 파일 수정시각에서 되짚어
구하면 1초 안쪽으로 맞습니다.

    make_caption_video.py --frames ~/pyg_rec/2026-09-12_knee \
        --csv System_ID/data/2026-09-12_knee.csv --motor 4 \
        --epoch0 1789210891 --from-t 205 --to-t 345 \
        --captions caps.json --out docs/assets/hw/knee.mp4

`caps.json` 은 `[[시작초, "자막"], ...]` 입니다.

★ 이 파일은 그림만 만듭니다. 모터로 나가는 길이 없습니다.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KOREAN_FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
KOREAN_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"


def load_rows(path: str, motor: int):
  """(시각, 각도, 목표, 토크) 목록. 시각은 기록 프로그램 시작부터의 초."""
  out = []
  with open(path, newline="") as fh:
    for r in csvmod.DictReader(fh):
      try:
        out.append((float(r["t"]), float(r[f"{motor}_deg"]),
                    float(r[f"{motor}_tgt"]), float(r[f"{motor}_nm"])))
      except (KeyError, TypeError, ValueError):
        continue
  return out


def strip_chart(rows, t_from, t_to, width, height, png):
  """각도와 토크 그래프를 한 장 그려 둔다. 프레임마다 세로선만 옮겨 그린다."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  seg = [r for r in rows if t_from <= r[0] <= t_to]
  t = [r[0] for r in seg]
  fig, ax = plt.subplots(2, 1, figsize=(width / 100.0, height / 100.0),
                         sharex=True, dpi=100)
  ax[0].plot(t, [r[2] for r in seg], lw=1.1, color="tab:orange", label="target")
  ax[0].plot(t, [r[1] for r in seg], lw=1.6, color="tab:blue", label="measured")
  ax[0].set_ylabel("deg", fontsize=8)
  ax[0].legend(fontsize=7, loc="upper left", ncol=2)
  ax[1].plot(t, [r[3] for r in seg], lw=1.1, color="tab:green")
  ax[1].axhline(0, lw=0.6, color="0.6")
  ax[1].set_ylabel("N·m", fontsize=8)
  ax[1].set_xlabel("time [s]", fontsize=8)
  for a in ax:
    a.grid(alpha=0.3)
    a.tick_params(labelsize=7)
  fig.tight_layout(pad=0.4)
  fig.savefig(png, dpi=100)
  plt.close(fig)
  # 데이터 좌표 -> 그림 픽셀. 세로선을 옳은 자리에 그으려면 이게 필요합니다.
  box = ax[0].get_position()
  return (box.x0 * width, box.x1 * width)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--frames", required=True, help="fNNNNNN.jpg 와 times.txt 가 있는 곳")
  ap.add_argument("--csv", required=True)
  ap.add_argument("--motor", type=int, required=True)
  ap.add_argument("--epoch0", type=float, required=True,
                  help="제어 기록의 0초에 해당하는 유닉스 시각")
  ap.add_argument("--from-t", type=float, required=True)
  ap.add_argument("--to-t", type=float, required=True)
  ap.add_argument("--captions", required=True, help='[[초, "자막"], ...] 인 json')
  ap.add_argument("--title", default="")
  ap.add_argument("--fps", type=float, default=10.0)
  ap.add_argument("--out", required=True)
  a = ap.parse_args(argv)

  from PIL import Image, ImageDraw, ImageFont

  root = Path(a.frames).expanduser()
  times = {}
  for line in (root / "times.txt").read_text().splitlines():
    try:
      n, t = line.split()
      times[int(n)] = float(t)
    except ValueError:
      continue
  if not times:
    print("times.txt 가 비어 있습니다", file=sys.stderr)
    return 1

  rows = load_rows(a.csv, a.motor)
  if not rows:
    print(f"{a.csv} 에 모터 {a.motor} 칸이 없습니다", file=sys.stderr)
    return 1
  caps = [(float(t), str(s)) for t, s in json.loads(Path(a.captions).read_text())]
  caps.sort()

  tmp = Path(tempfile.mkdtemp(prefix="capvid_"))
  chart_png = tmp / "chart.png"
  W = 960
  chart_h = 260
  strip_chart(rows, a.from_t, a.to_t, W, chart_h, chart_png)
  chart = Image.open(chart_png).convert("RGB")

  big = ImageFont.truetype(KOREAN_BOLD, 26)
  mid = ImageFont.truetype(KOREAN_FONT, 20)
  small = ImageFont.truetype(KOREAN_FONT, 17)

  wanted = sorted(n for n, ts in times.items()
                  if a.from_t <= ts - a.epoch0 <= a.to_t)
  if not wanted:
    print("그 구간에 해당하는 사진이 없습니다", file=sys.stderr)
    return 1
  print(f"  사진 {len(wanted)}장 ({a.from_t:g}~{a.to_t:g}초)", flush=True)

  cam_h = 720
  head_h = 92
  canvas_h = head_h + cam_h + chart_h
  ri = 0
  made = 0
  for k, n in enumerate(wanted):
    t = times[n] - a.epoch0
    while ri + 1 < len(rows) and rows[ri + 1][0] <= t:
      ri += 1
    _, deg, tgt, nm = rows[ri]
    cap = ""
    for ct, cs in caps:
      if t >= ct:
        cap = cs
    frame = Image.open(root / f"f{n:06d}.jpg").convert("RGB").resize((W, cam_h))
    canvas = Image.new("RGB", (W, canvas_h), (18, 18, 20))
    canvas.paste(frame, (0, head_h))
    canvas.paste(chart, (0, head_h + cam_h))
    d = ImageDraw.Draw(canvas)
    if a.title:
      d.text((14, 8), a.title, font=mid, fill=(150, 150, 155))
    d.text((14, 34), cap, font=big, fill=(255, 236, 150))
    # 숫자는 그림 위에 얹는다 -- 눈이 한 군데만 보면 되게.
    panel = (f"{t - a.from_t:6.1f}초   각도 {deg:+7.2f}°   "
             f"목표 {tgt:+7.2f}°   토크 {nm:+5.2f} N·m")
    d.rectangle([0, head_h, W, head_h + 34], fill=(0, 0, 0))
    d.text((14, head_h + 6), panel, font=mid, fill=(240, 240, 240))
    # 그래프 위 세로선
    frac = (t - a.from_t) / max(1e-6, (a.to_t - a.from_t))
    x0, x1 = 78, W - 22
    x = x0 + frac * (x1 - x0)
    d.line([(x, head_h + cam_h + 6), (x, canvas_h - 34)], fill=(220, 60, 60), width=2)
    d.text((W - 250, 12), "왼 무릎 · 실시간 속도", font=small, fill=(150, 150, 155))
    canvas.save(tmp / f"g{k:06d}.png")
    made += 1
    if made % 200 == 0:
      print(f"    {made}/{len(wanted)}장", flush=True)

  out = Path(a.out).expanduser()
  out.parent.mkdir(parents=True, exist_ok=True)
  cmd = ["ffmpeg", "-y", "-framerate", f"{a.fps:g}", "-i", str(tmp / "g%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(out)]
  r = subprocess.run(cmd, capture_output=True, text=True)
  if r.returncode != 0:
    print(r.stderr[-1500:], file=sys.stderr)
    return 1
  shutil.rmtree(tmp, ignore_errors=True)
  size = out.stat().st_size / 1e6
  print(f"  {made}장 -> {out} ({size:.1f} MB, {made / a.fps:.0f}초, 실시간)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
