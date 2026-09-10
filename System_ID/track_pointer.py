#!/usr/bin/env python3
"""카메라 그림에서 **축에 붙인 나무 막대의 각도**를 읽는다.

2026-09-10. 사용자가 출력축 허브에 나무 막대를 붙여 주셔서, 카메라가 회전을 직접 볼 수
있게 됐습니다. (그 전에는 출력축이 대칭인 민무늬 원판이라 돌아도 화소가 안 바뀌었습니다.)

어떻게 찾는가
-------------
나무는 **붉은기**가 있고 은색 원판은 빨강과 파랑이 비슷합니다. 그래서 `빨강 − 파랑` 을
"나무다움" 점수로 씁니다. 그 다음 허브를 중심으로 한 **고리 모양 영역**에서 각도마다
점수를 평균 내고, 가장 높은 각도를 막대 방향으로 봅니다. 막대는 고리 전체를 가로지르므로
책상이나 테이프 같은 얼룩보다 훨씬 강하게 나옵니다.

카메라가 비스듬히 보므로 원판이 타원으로 보입니다. 그래서 **그림에서 잰 각도와 실제 각도는
곧바로 같지 않습니다.** 알려진 각도 몇 개로 그 관계를 맞춰 씁니다(`--calib`).
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math

import numpy as np
from PIL import Image


def pointer_angle(path, cx, cy, r0, r1, nang=720, box=None):
  im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
  wood = im[:, :, 0] - im[:, :, 2]
  if box:
    x0, y0, x1, y1 = box
    mask = np.zeros_like(wood, dtype=bool)
    mask[y0:y1, x0:x1] = True
    wood = np.where(mask, wood, -1e3)
  th = np.linspace(0, 2 * math.pi, nang, endpoint=False)
  rs = np.linspace(r0, r1, 24)
  R, T = np.meshgrid(rs, th, indexing="ij")
  X = np.clip((cx + R * np.cos(T)).astype(int), 0, wood.shape[1] - 1)
  Y = np.clip((cy + R * np.sin(T)).astype(int), 0, wood.shape[0] - 1)
  # 반지름 방향 **하위 30% 값**을 쓴다. 막대는 고리를 통째로 가로지르므로 모든 반지름에서
  # 값이 높지만, 테이프 조각이나 책상처럼 한 반지름에만 있는 얼룩은 여기서 걸러진다.
  # (평균을 썼더니 옆에 붙은 노란 테이프를 막대로 잘못 잡았다.)
  prof = np.percentile(wood[Y, X], 30, axis=0)
  # 가장 높은 각도 언저리를 무게중심으로 다듬는다 (한 칸 해상도보다 곱게)
  k = int(np.argmax(prof))
  win = 40
  idx = [(k + d) % nang for d in range(-win, win + 1)]
  w = np.clip(prof[idx] - prof.mean(), 0, None)
  ang = th[idx]
  vx = float(np.sum(w * np.cos(ang))); vy = float(np.sum(w * np.sin(ang)))
  return math.degrees(math.atan2(vy, vx)), float(prof[k] - prof.mean())


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--glob", required=True)
  ap.add_argument("--cx", type=float, required=True)
  ap.add_argument("--cy", type=float, required=True)
  ap.add_argument("--r0", type=float, default=30)
  ap.add_argument("--r1", type=float, default=95)
  ap.add_argument("--box", default=None, help="x0,y0,x1,y1 로 찾을 영역을 좁힌다")
  ap.add_argument("--out", default=None)
  a = ap.parse_args(argv)
  box = tuple(int(x) for x in a.box.split(",")) if a.box else None
  res = []
  for p in sorted(globmod.glob(a.glob)):
    ang, strength = pointer_angle(p, a.cx, a.cy, a.r0, a.r1, box=box)
    res.append(dict(path=p, angle_img_deg=ang, strength=strength))
    print(f"{p.split('/')[-1]:>16}  각도 {ang:+8.2f}도  세기 {strength:6.1f}")
  if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
