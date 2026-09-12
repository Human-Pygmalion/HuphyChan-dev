#!/usr/bin/env python3
"""벤치 카메라를 **사진 한 장씩 받아 쌓는다.** 모터가 움직일 때는 항상 켠다.

2026-09-12. 사용자 지시입니다 — "모터에 힘이 들어가는 모든 순간을 녹화." 순서도
정해져 있습니다:

    1) 녹화를 켠다
    2) 사진 수가 **실제로 느는지** 확인한다
    3) 사람이 없는지 확인한다
    4) 그 다음에 명령을 보낸다

파이에서 도는 `cam_stream.py` 가 `snapshot.jpg` 를 내주고, 여기서는 그걸 되풀이해
받아 `fNNNNNN.jpg` 로 저장합니다. 받은 시각은 `times.txt` 에 한 줄씩 남습니다 —
나중에 영상으로 묶을 때 **실제 시간과 맞추려면** 그 시각이 필요합니다.

★ 이 파일은 **받기만 합니다.** 모터로 나가는 길이 없습니다.

    record_cam.py --out ~/pyg_rec/2026-09-12_kp20 --fps 10
    record_cam.py --out ... --seconds 600        # 저절로 멈추게
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--url", default="http://10.8.0.17:8099/snapshot.jpg")
  ap.add_argument("--out", required=True)
  ap.add_argument("--fps", type=float, default=10.0)
  ap.add_argument("--seconds", type=float, default=3600.0)
  ap.add_argument("--quiet-every", type=float, default=10.0,
                  help="몇 초마다 몇 장 받았는지 한 줄 찍을지")
  a = ap.parse_args(argv)

  out = Path(a.out).expanduser()
  out.mkdir(parents=True, exist_ok=True)
  times = (out / "times.txt").open("a")
  period = 1.0 / max(a.fps, 0.1)
  t0 = time.time()
  n, fails, said = 0, 0, t0
  print(f"  {a.url} -> {out} (초당 {a.fps:g}장)", flush=True)
  try:
    while time.time() - t0 < a.seconds:
      tick = time.time()
      try:
        with urllib.request.urlopen(a.url, timeout=3.0) as r:
          raw = r.read()
        if raw:
          (out / f"f{n:06d}.jpg").write_bytes(raw)
          times.write(f"{n} {tick:.3f}\n")
          n += 1
      except Exception as e:                                # noqa: BLE001
        fails += 1
        if fails in (1, 10, 100):
          print(f"  못 받음 {fails}회: {e}", flush=True)
      if time.time() - said >= a.quiet_every:
        said = time.time()
        times.flush()
        print(f"  {n}장 ({time.time() - t0:.0f}초, 못 받음 {fails}회)", flush=True)
      time.sleep(max(0.0, period - (time.time() - tick)))
  except KeyboardInterrupt:
    print("\n  사람이 멈춤")
  finally:
    times.close()
  print(f"  모두 {n}장 -> {out}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
