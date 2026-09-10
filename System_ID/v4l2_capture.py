#!/usr/bin/env python3
"""카메라를 **커널 장치로 직접** 열어 찍는다. 프레임마다 **커널이 찍은 시각**을 남긴다.

2026-09-10 · 사용자가 라즈베리파이에 RealSense D435 를 달아 주셨습니다.

왜 인텔 라이브러리를 안 쓰는가
------------------------------
파이에는 `pyrealsense2` 도 `ffmpeg` 도 `numpy` 도 없습니다(파이썬 3.13 용 인텔 꾸러미는
아직 없습니다). 그런데 이 카메라는 **평범한 USB 영상 장치**로도 보입니다.

    /dev/video0   깊이
    /dev/video2   적외선
    /dev/video4   ★ 빛 그림 (YUYV, 1920x1080 까지)

그래서 커널 영상 장치(V4L2)를 직접 엽니다. 설치할 것이 없고, **더 중요한 이득**이 있습니다.

★ 커널은 프레임을 건네줄 때 **그 프레임이 잡힌 시각**을 같이 줍니다. USB 카메라에서 그 시각은
CLOCK_MONOTONIC 이고, 파이썬의 `time.monotonic()` 과 **같은 시계**입니다. 모터 기록도 같은
컴퓨터에서 같은 시계로 찍히므로, 둘을 맞추는 일이 **뺄셈 하나**가 됩니다.

앞의 웹캠 촬영에서는 이게 없어서, 초당 장수·지연·각도 눈금 셋을 한꺼번에 수색해야 했습니다.
초당 15장인 줄 알았는데 실제로는 8.6장이라 영상 시계가 1.75배 빨리 갔던 것입니다.

무엇을 남기는가
---------------
    <폴더>/frames.raw     YUYV 원본을 이어 붙인 것 (한 장 = 가로 x 세로 x 2 바이트)
    <폴더>/frames.csv     한 줄에 한 장: 번호 · **커널이 찍은 시각** · 순번 · 파일 안 위치
    <폴더>/meta.json      가로·세로·초당 장수·놓친 장수

원본 그대로 쓰는 이유는 파이에 그림을 압축할 도구가 없기 때문입니다. 대신 **읽는 쪽**
(`yuyv_to_jpeg.py`)에서 필요한 장만 골라 그림으로 바꿉니다.

    v4l2_capture.py --probe                       # 되는 조합만 출력, 촬영 안 함
    v4l2_capture.py --out ~/System_ID/data/cam1 --seconds 30 --width 1280 --height 720 --fps 60
"""

from __future__ import annotations

import argparse
import fcntl
import json
import mmap
import os
import select
import struct
import sys
import time

# --- 커널에 명령을 거는 번호 만들기 (리눅스 ioctl 규칙) ---------------------------------
_NONE, _WRITE, _READ = 0, 1, 2


def _ioc(d, t, nr, size):
  return (d << 30) | (size << 16) | (ord(t) << 8) | nr


CAP_FMT = "=16s32s32sIII3I"          # v4l2_capability
PIX_FMT = "=8I4I"                    # v4l2_pix_format (48 바이트)
FMT_FMT = "=I4x" + "48s" + "152x"    # v4l2_format (208 바이트)
REQ_FMT = "=IIII4x"                  # v4l2_requestbuffers (20 바이트)
BUF_FMT = ("=IIIII4x"                # index type bytesused flags field (+정렬 4)
           "qq"                      # timestamp: 초, 마이크로초
           "II4B4s"                  # timecode
           "II"                      # sequence, memory
           "Q"                       # m (offset 등)
           "II"                      # length, reserved2
           "i4x")                    # request_fd (+정렬 4)
PARM_FMT = "=I4x" + "6I" + "16x" + "156x"   # v4l2_streamparm (204 바이트)

VIDIOC_QUERYCAP = _ioc(_READ, "V", 0, struct.calcsize(CAP_FMT))
VIDIOC_S_FMT = _ioc(_READ | _WRITE, "V", 5, 208)
VIDIOC_REQBUFS = _ioc(_READ | _WRITE, "V", 8, 20)
VIDIOC_QUERYBUF = _ioc(_READ | _WRITE, "V", 9, 88)
VIDIOC_QBUF = _ioc(_READ | _WRITE, "V", 15, 88)
VIDIOC_DQBUF = _ioc(_READ | _WRITE, "V", 17, 88)
VIDIOC_STREAMON = _ioc(_WRITE, "V", 18, 4)
VIDIOC_STREAMOFF = _ioc(_WRITE, "V", 19, 4)
VIDIOC_S_PARM = _ioc(_READ | _WRITE, "V", 22, 204)

BUF_TYPE_VIDEO_CAPTURE = 1
MEMORY_MMAP = 1
PIX_FMT_YUYV = 0x56595559            # 'YUYV'
FIELD_NONE = 1


# v4l2_buffer 의 값 순서. 손으로 0 을 스무 개 늘어놓으면 자리를 밀기 쉽고, 밀어도 오류가
# 안 나고 **조용히 엉뚱한 칸**을 건드린다. 그래서 채우고 읽는 일을 함수로 묶는다.
_B_SEQUENCE, _B_OFFSET, _B_LENGTH = 14, 16, 17
_B_TV_SEC, _B_TV_USEC, _B_BYTESUSED, _B_INDEX = 5, 6, 2, 0


def pack_buf(index=0):
  """비어 있는 v4l2_buffer 한 개. 커널에 건네 주거나 되받을 때 쓴다."""
  return struct.pack(BUF_FMT,
                     index, BUF_TYPE_VIDEO_CAPTURE, 0, 0, 0,   # index type bytesused flags field
                     0, 0,                                     # timestamp 초, 마이크로초
                     0, 0, 0, 0, 0, 0, b"\0" * 4,              # timecode
                     0, MEMORY_MMAP,                           # sequence, memory
                     0,                                        # m (offset)
                     0, 0,                                     # length, reserved2
                     0)                                        # request_fd


def unpack_buf(raw):
  return struct.unpack(BUF_FMT, raw)


def _check_sizes():
  """구조체 크기를 커널이 기대하는 값과 맞춰 둔다. 하나라도 어긋나면 ioctl 이 조용히
  엉뚱한 자리를 읽으므로, **여기서 먼저 멈추는 편이 낫다.**"""
  assert struct.calcsize(PIX_FMT) == 48, struct.calcsize(PIX_FMT)
  assert struct.calcsize(FMT_FMT) == 208, struct.calcsize(FMT_FMT)
  assert struct.calcsize(REQ_FMT) == 20, struct.calcsize(REQ_FMT)
  assert struct.calcsize(BUF_FMT) == 88, struct.calcsize(BUF_FMT)
  assert struct.calcsize(PARM_FMT) == 204, struct.calcsize(PARM_FMT)


def set_format(fd, w, h, fps):
  pix = struct.pack(PIX_FMT, w, h, PIX_FMT_YUYV, FIELD_NONE, 0, 0, 0, 0, 0, 0, 0, 0)
  out = fcntl.ioctl(fd, VIDIOC_S_FMT, struct.pack(FMT_FMT, BUF_TYPE_VIDEO_CAPTURE, pix))
  got = struct.unpack(PIX_FMT, struct.unpack(FMT_FMT, out)[1])
  # 초당 장수는 **부탁**이다. 카메라가 못 내주면 조용히 다른 값이 되므로 끝에서 실측한다.
  parm = struct.pack(PARM_FMT, BUF_TYPE_VIDEO_CAPTURE, 0, 0, 1, int(fps), 0, 0)
  try:
    fcntl.ioctl(fd, VIDIOC_S_PARM, parm)
  except OSError as exc:
    print(f"초당 장수를 못 정했습니다({exc}) — 카메라 기본값으로 진행", file=sys.stderr)
  return got[0], got[1], got[5]           # 가로, 세로, 한 장 바이트 수


def probe(dev) -> int:
  """이 장치가 무엇인지, 어떤 크기가 되는지만 보고 끝낸다. 찍지 않는다."""
  fd = os.open(dev, os.O_RDWR)
  try:
    cap = struct.unpack(CAP_FMT,
                        fcntl.ioctl(fd, VIDIOC_QUERYCAP, b"\0" * struct.calcsize(CAP_FMT)))
    z = lambda b: b.rstrip(b"\0").decode(errors="replace")     # noqa: E731
    print(f"장치 {dev}\n  드라이버 {z(cap[0])}\n  이름 {z(cap[1])}\n  연결 {z(cap[2])}")
    for w, h in ((1920, 1080), (1280, 720), (960, 540), (848, 480), (640, 480)):
      gw, gh, sz = set_format(fd, w, h, 60)
      ok = "" if (gw, gh) == (w, h) else "   ← 부탁과 다름"
      print(f"  {w:5d}x{h:<5d} → 실제 {gw}x{gh} · 한 장 {sz / 1e6:5.2f} MB{ok}")
  finally:
    os.close(fd)
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dev", default="/dev/video4", help="빛 그림 장치. RealSense 는 보통 video4")
  ap.add_argument("--probe", action="store_true", help="무엇이 되는지만 보고 끝낸다")
  ap.add_argument("--out", default=os.path.expanduser("~/System_ID/data/cam1"))
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--fps", type=int, default=60)
  ap.add_argument("--nbuf", type=int, default=8, help="커널과 주고받는 칸 수")
  ap.add_argument("--warmup", type=float, default=1.0, help="노출이 자리잡을 때까지 버릴 시간")
  a = ap.parse_args(argv)

  _check_sizes()
  if a.probe:
    return probe(a.dev)

  os.makedirs(a.out, exist_ok=True)
  fd = os.open(a.dev, os.O_RDWR)
  w, h, sizeimage = set_format(fd, a.width, a.height, a.fps)
  print(f"{a.dev} · {w}x{h} · 한 장 {sizeimage} 바이트 · 초당 {a.fps}장 부탁")

  fcntl.ioctl(fd, VIDIOC_REQBUFS, struct.pack(REQ_FMT, a.nbuf, BUF_TYPE_VIDEO_CAPTURE,
                                              MEMORY_MMAP, 0))
  maps = []
  for i in range(a.nbuf):
    b = unpack_buf(fcntl.ioctl(fd, VIDIOC_QUERYBUF, pack_buf(i)))
    offset, length = b[_B_OFFSET], b[_B_LENGTH]
    maps.append(mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ, offset=offset))
    fcntl.ioctl(fd, VIDIOC_QBUF, pack_buf(i))

  fcntl.ioctl(fd, VIDIOC_STREAMON, struct.pack("=I", BUF_TYPE_VIDEO_CAPTURE))
  raw = open(os.path.join(a.out, "frames.raw"), "wb", buffering=1024 * 1024 * 8)
  rows = []
  t_begin = time.monotonic()
  kept, dropped = 0, 0
  try:
    while time.monotonic() - t_begin < a.seconds + a.warmup:
      r, _, _ = select.select([fd], [], [], 1.0)
      if not r:
        continue
      b = unpack_buf(fcntl.ioctl(fd, VIDIOC_DQBUF, pack_buf()))
      idx, used = b[_B_INDEX], b[_B_BYTESUSED]
      tv_sec, tv_usec, seq = b[_B_TV_SEC], b[_B_TV_USEC], b[_B_SEQUENCE]
      t_frame = tv_sec + tv_usec * 1e-6          # ★ 커널이 찍은 시각 (time.monotonic 과 같은 시계)
      if time.monotonic() - t_begin >= a.warmup and used >= sizeimage:
        raw.write(maps[idx][:sizeimage])
        rows.append((kept, f"{t_frame:.6f}", seq, kept * sizeimage))
        kept += 1
      else:
        dropped += 1
      fcntl.ioctl(fd, VIDIOC_QBUF, pack_buf(idx))
  except KeyboardInterrupt:
    print("사람이 중단")
  finally:
    fcntl.ioctl(fd, VIDIOC_STREAMOFF, struct.pack("=I", BUF_TYPE_VIDEO_CAPTURE))
    raw.close()
    for m in maps:
      m.close()
    os.close(fd)

  with open(os.path.join(a.out, "frames.csv"), "w") as f:
    f.write("idx,t_mono_s,sequence,byte_offset\n")
    for r in rows:
      f.write(",".join(str(x) for x in r) + "\n")
  span = (float(rows[-1][1]) - float(rows[0][1])) if len(rows) > 1 else 0.0
  meta = dict(width=w, height=h, sizeimage=sizeimage, fps_asked=a.fps,
              frames=kept, warmup_skipped=dropped, span_s=span,
              fps_actual=(kept - 1) / span if span > 0 else 0.0, device=a.dev,
              pixelformat="YUYV")
  json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
  print(f"찍은 장수 {kept} · {span:.2f}초 · **실제 초당 {meta['fps_actual']:.2f}장** · "
        f"{os.path.getsize(os.path.join(a.out, 'frames.raw')) / 1e6:.0f} MB · {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
