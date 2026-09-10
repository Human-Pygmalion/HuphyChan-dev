#!/usr/bin/env python3
"""파이가 잘라 보낸 **원본 조각**(`crops.bin`)을 그림 파일로 바꾼다. 받는 쪽에서 돌린다.

2026-09-10. 파이에는 그림을 압축할 도구가 아예 없습니다(numpy·Pillow·ffmpeg 없음, 인터넷도
pip 도 없어 설치 불가). 그래서 파이는 **자르기만** 하고, 색을 풀어 그림으로 만드는 일은
이 컴퓨터에서 합니다.

    crops.bin 의 생김새
      머리말  "CRP1" · 가로 · 세로 · 장수 · 예비        (=4sIIII)
      차례    장마다 (번호, 시각)                        (=Id)
      본문    장마다 가로 x 세로 x 2 바이트 (YUYV 원본)

YUYV 는 가로로 붙은 화소 둘이 색을 나눠 씁니다: `[Y0][U][Y1][V]`. 밝기는 화소마다 있고
색은 둘이 함께 씁니다. 그래서 색만 옆으로 두 배 늘려 놓고 정해진 식으로 빨강·초록·파랑을
만듭니다.

결과는 `make_video2.py --frames-dir` 가 그대로 읽는 모양(`color_00000.jpg` + `frames.csv`)
으로 남깁니다.
"""

from __future__ import annotations

import argparse
import os
import struct

import numpy as np
from PIL import Image

HEAD = "=4sIIII"
ENTRY = "=Id"


def yuyv_to_rgb(buf, w, h):
  d = np.frombuffer(buf, dtype=np.uint8).reshape(h, w // 2, 4)
  Y = np.empty((h, w), np.float32)
  Y[:, 0::2] = d[:, :, 0]
  Y[:, 1::2] = d[:, :, 2]
  U = np.repeat(d[:, :, 1], 2, axis=1).astype(np.float32) - 128.0
  V = np.repeat(d[:, :, 3], 2, axis=1).astype(np.float32) - 128.0
  rgb = np.stack([Y + 1.402 * V,
                  Y - 0.344136 * U - 0.714136 * V,
                  Y + 1.772 * U], axis=-1)
  return np.clip(rgb, 0, 255).astype(np.uint8)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--crops", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--quality", type=int, default=92)
  a = ap.parse_args(argv)

  os.makedirs(a.out, exist_ok=True)
  with open(a.crops, "rb") as f:
    magic, w, h, n, _ = struct.unpack(HEAD, f.read(struct.calcsize(HEAD)))
    if magic != b"CRP1":
      raise SystemExit(f"머리말이 다릅니다: {magic!r}")
    idx_t = [struct.unpack(ENTRY, f.read(struct.calcsize(ENTRY))) for _ in range(n)]
    with open(os.path.join(a.out, "frames.csv"), "w") as g:
      g.write("idx,t_mono_s,src_idx\n")
      for i, (src, t) in enumerate(idx_t):
        buf = f.read(w * h * 2)
        if len(buf) < w * h * 2:
          print(f"자료가 {i}장에서 끊겼습니다 — 거기까지만 씁니다")
          break
        Image.fromarray(yuyv_to_rgb(buf, w, h)).save(
            os.path.join(a.out, f"color_{i:05d}.jpg"), quality=a.quality)
        g.write(f"{i},{t:.6f},{src}\n")
  print(f"{n}장 · {w}x{h} → {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
