#!/usr/bin/env python3
"""파이 위에서 **원본 영상을 다룬다. 파이썬 기본 기능만** 쓴다.

2026-09-10. 파이에는 numpy 도 Pillow 도 ffmpeg 도 없고, 인터넷도 pip 도 없어서 설치할 수도
없습니다. 그런데 원본을 그대로 가져오기엔 큽니다 — 1920x1080 으로 30초 찍으면 3.7 기가바이트
이고, 여기 연결은 초당 10.6 메가바이트라 6분이 걸립니다.

그래서 **무거운 것은 파이에서 줄이고, 줄인 것만 가져옵니다.**

    track   막대 각도를 파이에서 읽어 숫자만 남긴다 (파일이 아주 작아진다)
    crop    영상에 쓸 장만 골라 **모터 주변만 잘라** 남긴다

색을 어떻게 보는가
------------------
YUYV 는 밝기(Y)와 색(U, V)을 따로 담습니다. 가로로 붙은 화소 두 개가 4 바이트를 나눠 씁니다.

    [Y0][U][Y1][V]      U = 파랑기, V = 붉은기 (둘 다 128 이 가운데)

앞서 웹캠에서는 "빨강 − 파랑"으로 나무를 찾았는데, 여기서는 **V − U** 가 바로 그 값입니다.
RGB 로 바꿀 필요가 없습니다 — 나무 막대는 붉은/노란기가 있어 V 가 크고 U 가 작습니다.

    yuyv_tools.py track --raw ~/System_ID/data/cam1 --cx 700 --cy 210 --r0 40 --r1 120
    yuyv_tools.py crop  --raw ~/System_ID/data/cam1 --box 380,40,1340,664 --every 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct


def load_meta(d):
  m = json.load(open(os.path.join(d, "meta.json")))
  rows = []
  with open(os.path.join(d, "frames.csv")) as f:
    next(f)
    for line in f:
      i, t, seq, off = line.strip().split(",")
      rows.append((int(i), float(t), int(seq), int(off)))
  return m, rows


def build_table(w, cx, cy, r0, r1, nang, nr):
  """각도마다 반지름을 따라 **V 와 U 가 놓인 바이트 자리**를 미리 계산해 둔다.

  프레임마다 다시 계산하면 파이썬에서 너무 느리다. 자리만 미리 잡아 두면 나중에는
  바이트를 꺼내 빼기만 하면 된다."""
  tab = []
  for a in range(nang):
    th = 2 * math.pi * a / nang
    ca, sa = math.cos(th), math.sin(th)
    col = []
    for k in range(nr):
      r = r0 + (r1 - r0) * k / (nr - 1)
      x, y = int(cx + r * ca), int(cy + r * sa)
      base = y * w * 2 + (x >> 1) * 4
      col.append(base)
    tab.append(col)
  return tab


def track(a) -> int:
  m, rows = load_meta(a.raw)
  w, h, sz = m["width"], m["height"], m["sizeimage"]
  nang, nr = a.nang, a.nr
  tab = build_table(w, a.cx, a.cy, a.r0, a.r1, nang, nr)
  q = max(0, min(nr - 1, int(round(0.30 * (nr - 1)))))   # 반지름 방향 하위 30%
  th = [2 * math.pi * i / nang for i in range(nang)]
  out = open(os.path.join(a.raw, "angles.csv"), "w")
  out.write("idx,t_mono_s,angle_img_deg,strength\n")
  with open(os.path.join(a.raw, "frames.raw"), "rb") as f:
    for idx, t, seq, off in rows:
      f.seek(off)
      buf = f.read(sz)
      if len(buf) < sz:
        break
      prof = []
      for col in tab:
        # 막대는 고리를 통째로 가로지르므로 **모든 반지름에서** 값이 크다. 한 반지름에만
        # 있는 얼룩(테이프 조각 등)은 하위 30% 를 쓰면 걸러진다.
        vals = sorted(buf[b + 3] - buf[b + 1] for b in col)
        prof.append(vals[q])
      k = max(range(nang), key=prof.__getitem__)
      mean = sum(prof) / nang
      vx = vy = 0.0
      for d in range(-40, 41):
        i = (k + d) % nang
        wgt = prof[i] - mean
        if wgt > 0:
          vx += wgt * math.cos(th[i]); vy += wgt * math.sin(th[i])
      ang = math.degrees(math.atan2(vy, vx))
      out.write(f"{idx},{t:.6f},{ang:.3f},{prof[k] - mean:.2f}\n")
  out.close()
  print(f"각도 {len(rows)}줄 → {os.path.join(a.raw, 'angles.csv')}")
  return 0


def crop(a) -> int:
  """영상에 쓸 장만 골라 **모터 주변만** 잘라 낸다. 자를 때 가로 시작과 폭은 짝수여야 한다
  — YUYV 는 가로로 붙은 화소 둘이 색을 나눠 쓰기 때문이다."""
  m, rows = load_meta(a.raw)
  w, h, sz = m["width"], m["height"], m["sizeimage"]
  x0, y0, x1, y1 = (int(v) for v in a.box.split(","))
  x0 -= x0 % 2
  x1 -= x1 % 2
  cw, ch = x1 - x0, y1 - y0
  pick = rows[::a.every]
  outp = os.path.join(a.raw, "crops.bin")
  with open(os.path.join(a.raw, "frames.raw"), "rb") as f, open(outp, "wb") as g:
    g.write(struct.pack("=4sIIII", b"CRP1", cw, ch, len(pick), 0))
    for idx, t, seq, off in pick:
      g.write(struct.pack("=Id", idx, t))
    for idx, t, seq, off in pick:
      for yy in range(y0, y1):
        f.seek(off + yy * w * 2 + x0 * 2)
        g.write(f.read(cw * 2))
  print(f"자른 장수 {len(pick)} · {cw}x{ch} · "
        f"{os.path.getsize(outp) / 1e6:.0f} MB → {outp}")
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  t = sub.add_parser("track", help="막대 각도를 읽는다")
  t.add_argument("--raw", required=True)
  t.add_argument("--cx", type=float, required=True)
  t.add_argument("--cy", type=float, required=True)
  t.add_argument("--r0", type=float, default=40.0)
  t.add_argument("--r1", type=float, default=120.0)
  t.add_argument("--nang", type=int, default=720)
  t.add_argument("--nr", type=int, default=24)
  t.set_defaults(fn=track)
  c = sub.add_parser("crop", help="영상에 쓸 장만 잘라 낸다")
  c.add_argument("--raw", required=True)
  c.add_argument("--box", required=True, help="x0,y0,x1,y1")
  c.add_argument("--every", type=int, default=3, help="몇 장에 한 장씩 고를까")
  c.set_defaults(fn=crop)
  a = ap.parse_args(argv)
  return a.fn(a)


if __name__ == "__main__":
  raise SystemExit(main())
