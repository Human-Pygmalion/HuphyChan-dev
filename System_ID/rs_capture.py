#!/usr/bin/env python3
"""RealSense D435i 로 벤치를 찍는다. **모터와 같은 시계**로 시각을 찍어 남긴다.

2026-09-10 · 사용자: "Raspberry pi 에 Realsense d435i 연결해뒀어. 이 카메라 써서 더 고해상도,
고 프레임으로 비교해봐." (`lsusb` 확인: `8086:0b07 Intel Corp. RealSense D435`)

왜 이 카메라가 나은가 — 화소보다 **시각**이 더 큰 이득입니다
--------------------------------------------------------------
앞의 웹캠 촬영에서는 프레임이 언제 찍혔는지 알 수가 없었습니다. 받아 적는 쪽이 초당 15장이라
가정했는데 실제로는 **초당 8.6장**이라, 영상 시계가 1.75배 빨리 갔습니다. 그래서 영상과 자료를
맞추려고 **프레임률·지연·각도 눈금 셋을 한꺼번에 맞추는 수색**을 해야 했습니다.

이 카메라는 프레임마다 시각을 같이 줍니다. 그리고 지금은 **카메라와 모터가 같은 컴퓨터에**
붙어 있으므로, 둘 다 그 컴퓨터의 같은 시계(`time.monotonic`)로 찍으면 **맞출 것이 없습니다.**
수색해서 얻은 0.974 짜리 맞춤 대신, 그냥 정확한 시각을 씁니다.

덤으로 초당 장수도 8.6 에서 **60** 으로 올라갑니다. 10 헤르츠까지 올리는 처프에서 앞의 촬영은
한 주기에 한 장도 못 건졌지만, 60 장이면 한 주기에 6 장입니다.

무엇을 남기는가
---------------
    <폴더>/color_00001.jpg ...   빛 그림 (막대 각도를 읽는 데 쓴다)
    <폴더>/frames.csv            한 줄에 한 장: 번호 · **컴퓨터 시계** · 카메라 시계 · 프레임번호
    <폴더>/meta.json             해상도·초당 장수·놓친 장수 등

★ 먼저 `--probe` 로 돌리세요. 이 카메라가 **실제로 내주는** 해상도·장수 목록만 출력하고
끝납니다. 촬영도 안 하고 모터도 안 건드립니다. 되지 않는 조합을 적어 놓고 실험 중에
실패하는 일을 막기 위한 것입니다.

    rs_capture.py --probe
    rs_capture.py --out ~/System_ID/data/cam_rs1 --seconds 30 --width 1280 --height 720 --fps 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time

try:
  import numpy as np
  import pyrealsense2 as rs
except ImportError as exc:                                  # noqa: BLE001
  print(f"필요한 꾸러미가 없습니다: {exc}\n"
        "  라즈베리파이: python3 -m pip install --break-system-packages pyrealsense2 numpy\n"
        "  (안 되면 librealsense 를 소스로 빌드해야 합니다)", file=sys.stderr)
  raise SystemExit(2)

from PIL import Image  # noqa: E402


def probe() -> int:
  """이 카메라가 실제로 내주는 조합만 출력한다. 촬영하지 않는다."""
  ctx = rs.context()
  devs = list(ctx.query_devices())
  if not devs:
    print("RealSense 를 못 찾았습니다. USB 3 포트에 꽂혀 있는지 보세요", file=sys.stderr)
    return 1
  for d in devs:
    print(f"기기: {d.get_info(rs.camera_info.name)} · 일련번호 "
          f"{d.get_info(rs.camera_info.serial_number)} · 펌웨어 "
          f"{d.get_info(rs.camera_info.firmware_version)}")
    for s in d.sensors:
      seen = set()
      rows = []
      for p in s.get_stream_profiles():
        try:
          v = p.as_video_stream_profile()
        except Exception:                                   # noqa: BLE001
          continue
        key = (str(p.stream_type()), str(p.format()), v.width(), v.height(), p.fps())
        if key in seen:
          continue
        seen.add(key)
        rows.append(key)
      if not rows:
        continue
      print(f"  [{s.get_info(rs.camera_info.name)}]")
      for st, fmt, w, h, f in sorted(rows, key=lambda r: (r[0], -r[2], -r[4])):
        print(f"    {st:24} {fmt:14} {w:5d} x {h:<5d} {f:4d} 장/초")
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--probe", action="store_true",
                  help="내줄 수 있는 조합만 출력하고 끝낸다. 촬영 안 함")
  ap.add_argument("--out", default=os.path.expanduser("~/System_ID/data/cam_rs"))
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--fps", type=int, default=60)
  ap.add_argument("--quality", type=int, default=88, help="JPEG 품질")
  ap.add_argument("--writers", type=int, default=3,
                  help="그림을 파일로 쓰는 일꾼 수. 찍는 일을 막지 않게 따로 돌린다")
  ap.add_argument("--warmup", type=float, default=1.5,
                  help="자동 노출이 자리잡을 때까지 버리는 시간")
  a = ap.parse_args(argv)

  if a.probe:
    return probe()

  os.makedirs(a.out, exist_ok=True)
  q: queue.Queue = queue.Queue(maxsize=180)
  dropped = [0]

  def writer():
    while True:
      item = q.get()
      if item is None:
        q.task_done()
        return
      idx, img = item
      Image.fromarray(img[:, :, ::-1]).save(          # bgr8 로 받으므로 뒤집어 저장
          os.path.join(a.out, f"color_{idx:05d}.jpg"), quality=a.quality)
      q.task_done()

  threads = [threading.Thread(target=writer, daemon=True) for _ in range(a.writers)]
  for t in threads:
    t.start()

  pipe = rs.pipeline()
  cfg = rs.config()
  cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, a.fps)
  try:
    prof = pipe.start(cfg)
  except RuntimeError as exc:
    print(f"이 조합으로는 못 켭니다: {exc}\n"
          "  `--probe` 로 실제로 되는 조합을 먼저 확인하세요", file=sys.stderr)
    return 1

  # 자동 노출·자동 화이트밸런스를 **고정**한다. 나무 막대를 '붉은기'로 찾기 때문에, 촬영 중에
  # 화이트밸런스가 움직이면 그 붉은기가 프레임마다 달라져 각도 읽기가 흔들린다.
  try:
    cs = prof.get_device().first_color_sensor()
    for _ in range(int(a.warmup * a.fps)):
      pipe.wait_for_frames()
    if cs.supports(rs.option.enable_auto_exposure):
      cs.set_option(rs.option.enable_auto_exposure, 0)
    if cs.supports(rs.option.enable_auto_white_balance):
      cs.set_option(rs.option.enable_auto_white_balance, 0)
    print("노출·화이트밸런스 고정 완료")
  except Exception as exc:                                  # noqa: BLE001
    print(f"노출 고정을 못 했습니다({exc}) — 그대로 진행합니다")

  rows = []
  t0 = time.monotonic()
  idx = 0
  print(f"촬영 시작 · {a.width}x{a.height} · 초당 {a.fps}장 · {a.seconds:.0f}초 · {a.out}",
        flush=True)
  try:
    while time.monotonic() - t0 < a.seconds:
      fs = pipe.wait_for_frames()
      c = fs.get_color_frame()
      if not c:
        continue
      t_mono = time.monotonic()
      img = np.asanyarray(c.get_data()).copy()
      rows.append((idx, f"{t_mono:.6f}", f"{c.get_timestamp():.3f}", c.get_frame_number()))
      try:
        q.put_nowait((idx, img))
      except queue.Full:
        dropped[0] += 1               # 디스크가 못 따라오면 **찍는 쪽을 막지 않고** 버린다
        rows.pop()
      idx += 1
  except KeyboardInterrupt:
    print("사람이 중단")
  finally:
    pipe.stop()
    for _ in threads:
      q.put(None)
    q.join()

  with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx", "t_mono_s", "camera_timestamp_ms", "frame_number"])
    w.writerows(rows)
  el = time.monotonic() - t0
  meta = dict(width=a.width, height=a.height, fps_asked=a.fps, seconds=el,
              frames_kept=len(rows), frames_dropped=dropped[0],
              fps_actual=(len(rows) / el if el > 0 else 0.0), t0_monotonic=t0)
  json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
  print(f"남긴 장수 {len(rows)} · 버린 장수 {dropped[0]} · 실제 초당 "
        f"{meta['fps_actual']:.2f}장 · {a.out}")
  if dropped[0] > len(rows) * 0.02:
    print("★ 2% 넘게 버렸습니다 — 해상도나 초당 장수를 낮추거나 --writers 를 늘리세요")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
