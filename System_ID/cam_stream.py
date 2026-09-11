#!/usr/bin/env python3
"""벤치 카메라를 **웹으로 실시간 중계**한다. 파이에 아무것도 설치하지 않는다.

2026-09-11 · 사용자가 라즈베리파이에 USB 카메라를 달아 주셨습니다. 로봇 오른다리가
같이 붙어 있어서, **버스에 무엇을 보내기 전에 사람 손이 치워졌는지 눈으로 확인**할
길이 필요합니다. 그 확인이 이 프로그램의 첫 번째 쓸모입니다.

왜 이렇게 만드는가
------------------
파이에는 `ffmpeg` 도 `numpy` 도 `pip` 도 없고 인터넷도 없습니다. 그런데 이 카메라는
**움직이는 JPEG(MJPG)** 를 그대로 내보냅니다. 즉 커널이 건네주는 한 칸이 **그 자체로
완성된 그림 파일**입니다.

그래서 다시 압축할 일이 없습니다 — 받은 것을 그대로 웹으로 흘려보내면 됩니다. 파이는
바이트를 옮기기만 하므로 부담이 거의 없고, 파이썬에 원래 들어 있는 기능만 씁니다.

디스크에 아무것도 쓰지 않습니다. 2026-09-10 에 SD 카드가 초당 124 메가바이트를 못 받아
촬영이 조용히 멈춘 적이 있어서, 이번에는 저장 자체를 하지 않습니다.

    cam_stream.py                          # 1280x720, 초당 15장, 8099 번
    cam_stream.py --width 1920 --height 1080 --fps 10
    cam_stream.py --probe                  # 무엇이 되는지만 보고 끝

브라우저에서

    http://<파이 주소>:8099/               보기 화면
    http://<파이 주소>:8099/stream.mjpg    영상만
    http://<파이 주소>:8099/snapshot.jpg   지금 한 장
    http://<파이 주소>:8099/status.json    초당 장수·크기·놓친 장수
"""

from __future__ import annotations

import argparse
import fcntl
import mmap
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# 커널 영상 장치(V4L2) 를 직접 부르기 위한 자료 구조.
# 2026-09-10 에 실물로 검증된 값 그대로임 (`v4l2_capture.py`). 한 칸이라도 어긋나면
# ioctl 이 **오류 없이 엉뚱한 자리**를 읽으므로 크기를 시작할 때 확인한다.
# ---------------------------------------------------------------------------
_NONE, _WRITE, _READ = 0, 1, 2


def _ioc(d, t, nr, size):
  return (d << 30) | (size << 16) | (ord(t) << 8) | nr


PIX_FMT = "=8I4I"                    # v4l2_pix_format (48 바이트)
FMT_FMT = "=I4x" + "48s" + "152x"    # v4l2_format (208 바이트)
REQ_FMT = "=IIII4x"                  # v4l2_requestbuffers (20 바이트)
BUF_FMT = ("=IIIII4x"                # index type bytesused flags field
           "qq"                      # timestamp: 초, 마이크로초
           "II4B4s"                  # timecode
           "II"                      # sequence, memory
           "Q"                       # m (offset 등)
           "II"                      # length, reserved2
           "i4x")                    # request_fd
PARM_FMT = "=I4x" + "6I" + "16x" + "156x"   # v4l2_streamparm (204 바이트)

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
PIX_FMT_MJPG = 0x47504A4D            # 'MJPG'
FIELD_NONE = 1

_B_INDEX, _B_BYTESUSED = 0, 2
_B_TV_SEC, _B_TV_USEC = 5, 6
_B_SEQUENCE, _B_OFFSET, _B_LENGTH = 14, 16, 17

JPEG_EOI = b"\xff\xd9"


def pack_buf(index=0):
  return struct.pack(BUF_FMT,
                     index, BUF_TYPE_VIDEO_CAPTURE, 0, 0, 0,
                     0, 0,
                     0, 0, 0, 0, 0, 0, b"\0" * 4,
                     0, MEMORY_MMAP,
                     0,
                     0, 0,
                     0)


def check_sizes():
  for name, fmt, want in (("PIX_FMT", PIX_FMT, 48), ("FMT_FMT", FMT_FMT, 208),
                          ("REQ_FMT", REQ_FMT, 20), ("BUF_FMT", BUF_FMT, 88),
                          ("PARM_FMT", PARM_FMT, 204)):
    got = struct.calcsize(fmt)
    if got != want:
      raise SystemExit(f"{name} 크기가 {got} 인데 {want} 여야 합니다. 이대로 두면 "
                       f"ioctl 이 조용히 엉뚱한 자리를 읽습니다")


def set_format(fd, w, h, fps):
  """원하는 크기·초당 장수를 **부탁**한다. 카메라가 못 맞추면 조용히 다른 값이 되므로
  돌려받은 값을 그대로 쓰고, 실제 초당 장수는 나중에 세어서 확인한다."""
  pix = struct.pack(PIX_FMT, w, h, PIX_FMT_MJPG, FIELD_NONE, 0, 0, 0, 0, 0, 0, 0, 0)
  out = fcntl.ioctl(fd, VIDIOC_S_FMT, struct.pack(FMT_FMT, BUF_TYPE_VIDEO_CAPTURE, pix))
  got = struct.unpack(PIX_FMT, struct.unpack(FMT_FMT, out)[1])
  if got[2] != PIX_FMT_MJPG:
    raise SystemExit("카메라가 움직이는 JPEG 를 안 내줍니다. 이 프로그램은 그것만 다룹니다")
  parm = struct.pack(PARM_FMT, BUF_TYPE_VIDEO_CAPTURE, 0, 0, 1, int(fps), 0, 0)
  try:
    fcntl.ioctl(fd, VIDIOC_S_PARM, parm)
  except OSError as exc:
    print(f"초당 장수를 못 정했습니다({exc}) — 카메라 기본값으로 진행", file=sys.stderr)
  return got[0], got[1]


class Grabber:
  """카메라에서 계속 한 장씩 받아 **가장 최근 것 하나만** 들고 있는다.

  보는 사람이 여럿이어도 카메라는 하나만 읽습니다. 늦게 들어온 사람은 밀린 옛 장면이
  아니라 **지금 장면**부터 봅니다 — 손이 치워졌는지 확인하는 용도라 그게 맞습니다.
  """

  def __init__(self, device, width, height, fps, nbuf=4):
    self.device, self.want = device, (width, height, fps)
    self.nbuf = nbuf
    self.frame = None            # 가장 최근 JPEG
    self.stamp = 0.0             # 커널이 그 장을 잡은 시각 (time.monotonic 과 같은 시계)
    self.seq = -1
    self.count = 0
    self.dropped = 0             # 커널이 센 놓친 장수 (순번이 건너뛴 만큼)
    self.started = time.monotonic()
    self.error = None
    self._cond = threading.Condition()
    self._stop = threading.Event()
    self.width = self.height = 0

  def start(self):
    t = threading.Thread(target=self._run, daemon=True)
    t.start()
    # 첫 장이 올 때까지 잠깐 기다린다. 여기서 실패하면 웹을 띄우기 전에 알 수 있다.
    with self._cond:
      self._cond.wait_for(lambda: self.frame is not None or self.error is not None,
                          timeout=8.0)
    if self.error:
      raise SystemExit(f"카메라를 열지 못했습니다: {self.error}")
    if self.frame is None:
      raise SystemExit("카메라가 8초 안에 한 장도 안 줬습니다")
    return self

  def stop(self):
    self._stop.set()

  def latest(self, after_seq=-1, timeout=5.0):
    """가장 최근 장. `after_seq` 를 주면 그보다 새 장이 올 때까지 기다린다."""
    with self._cond:
      ok = self._cond.wait_for(lambda: self.frame is not None and self.seq != after_seq,
                               timeout=timeout)
      return (self.frame, self.seq, self.stamp) if ok else (None, after_seq, 0.0)

  def status(self):
    up = max(1e-6, time.monotonic() - self.started)
    return dict(width=self.width, height=self.height, frames=self.count,
                dropped=self.dropped, fps=round(self.count / up, 2),
                uptime_s=round(up, 1),
                bytes_last=len(self.frame) if self.frame else 0)

  def _run(self):
    fd = maps = None
    try:
      fd = os.open(self.device, os.O_RDWR)
      w, h, fps = self.want
      self.width, self.height = set_format(fd, w, h, fps)

      fcntl.ioctl(fd, VIDIOC_REQBUFS,
                  struct.pack(REQ_FMT, self.nbuf, BUF_TYPE_VIDEO_CAPTURE, MEMORY_MMAP, 0))
      maps = []
      for i in range(self.nbuf):
        b = struct.unpack(BUF_FMT, fcntl.ioctl(fd, VIDIOC_QUERYBUF, pack_buf(i)))
        maps.append(mmap.mmap(fd, b[_B_LENGTH], mmap.MAP_SHARED,
                              mmap.PROT_READ, offset=b[_B_OFFSET]))
        fcntl.ioctl(fd, VIDIOC_QBUF, pack_buf(i))
      fcntl.ioctl(fd, VIDIOC_STREAMON, struct.pack("=I", BUF_TYPE_VIDEO_CAPTURE))

      last_seq = None
      while not self._stop.is_set():
        b = struct.unpack(BUF_FMT, fcntl.ioctl(fd, VIDIOC_DQBUF, pack_buf()))
        idx, used = b[_B_INDEX], b[_B_BYTESUSED]
        seq = b[_B_SEQUENCE]
        stamp = b[_B_TV_SEC] + b[_B_TV_USEC] / 1e6
        data = maps[idx][:used]
        fcntl.ioctl(fd, VIDIOC_QBUF, pack_buf(idx))

        # 어떤 카메라는 JPEG 끝 표시 뒤에 쓰레기를 붙여 보낸다. 잘라 둔다.
        end = data.rfind(JPEG_EOI)
        if end != -1:
          data = data[:end + 2]

        if last_seq is not None and seq > last_seq + 1:
          self.dropped += seq - last_seq - 1
        last_seq = seq

        with self._cond:
          self.frame, self.seq, self.stamp = data, seq, stamp
          self.count += 1
          self._cond.notify_all()
    except Exception as exc:                                    # noqa: BLE001
      with self._cond:
        self.error = f"{type(exc).__name__}: {exc}"
        self._cond.notify_all()
    finally:
      try:
        if fd is not None:
          fcntl.ioctl(fd, VIDIOC_STREAMOFF, struct.pack("=I", BUF_TYPE_VIDEO_CAPTURE))
      except OSError:
        pass
      for m in maps or []:
        m.close()
      if fd is not None:
        os.close(fd)


PAGE = """<!doctype html><meta charset="utf-8"><title>벤치 카메라</title>
<style>
 body{margin:0;background:#14161a;color:#e8e8e8;font:14px/1.5 system-ui,sans-serif}
 header{padding:10px 14px;background:#1c1f26;display:flex;gap:18px;align-items:baseline;
        flex-wrap:wrap;border-bottom:1px solid #2a2e37}
 h1{font-size:15px;margin:0;font-weight:600}
 .n{color:#8ea9c9;font-variant-numeric:tabular-nums}
 img{display:block;width:100%;height:auto;background:#000}
 footer{padding:8px 14px;color:#7b8494;font-size:12px}
</style>
<header>
  <h1>벤치 카메라 &mdash; 실시간</h1>
  <span class="n" id="s">연결 중&hellip;</span>
</header>
<img src="/stream.mjpg" alt="벤치 실시간 화면">
<footer>이 화면은 저장되지 않습니다. 한 장만 받으려면
  <code>/snapshot.jpg</code>, 숫자만 보려면 <code>/status.json</code>.</footer>
<script>
 async function tick(){
   try{
     const r = await fetch('/status.json', {cache:'no-store'});
     const j = await r.json();
     document.getElementById('s').textContent =
       `${j.width}\\u00d7${j.height} \\u00b7 \\ucd08\\ub2f9 ${j.fps}\\uc7a5 \\u00b7 ` +
       `\\ubc1b\\uc740 ${j.frames}\\uc7a5 \\u00b7 \\ub193\\uce5c ${j.dropped}\\uc7a5`;
   }catch(e){ document.getElementById('s').textContent = '\\uc0c1\\ud0dc\\ub97c \\ubabb \\uc77d\\uc74c'; }
 }
 tick(); setInterval(tick, 2000);
</script>
"""


def make_handler(grab):
  class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                                  # 조용히
      pass

    def _send(self, code, ctype, body, extra=()):
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      for k, v in extra:
        self.send_header(k, v)
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      path = self.path.split("?")[0]
      if path in ("/", "/index.html"):
        self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
      elif path == "/status.json":
        import json
        self._send(200, "application/json", json.dumps(grab.status()).encode())
      elif path == "/snapshot.jpg":
        frame, _, _ = grab.latest()
        if frame is None:
          self._send(503, "text/plain; charset=utf-8", "아직 한 장도 못 받았습니다".encode())
        else:
          self._send(200, "image/jpeg", frame)
      elif path == "/stream.mjpg":
        self._stream()
      else:
        self._send(404, "text/plain; charset=utf-8", "없는 주소".encode())

    def _stream(self):
      self.send_response(200)
      self.send_header("Content-Type",
                       "multipart/x-mixed-replace; boundary=huphyframe")
      self.send_header("Cache-Control", "no-store")
      self.send_header("Connection", "close")
      self.end_headers()
      seq = -1
      try:
        while True:
          frame, seq, _ = grab.latest(after_seq=seq, timeout=10.0)
          if frame is None:
            break
          self.wfile.write(b"--huphyframe\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
          self.wfile.write(frame)
          self.wfile.write(b"\r\n")
      except (BrokenPipeError, ConnectionResetError):
        pass          # 보던 사람이 창을 닫은 것. 정상임
      self.close_connection = True

  return Handler


def probe(device):
  """무엇이 되는지만 본다. 카메라를 잡고 있지 않는다."""
  os.system(f"v4l2-ctl -d {device} --list-formats-ext 2>/dev/null | head -60")
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--device", default="/dev/video0")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--fps", type=int, default=15)
  ap.add_argument("--port", type=int, default=8099)
  ap.add_argument("--bind", default="0.0.0.0")
  ap.add_argument("--probe", action="store_true", help="되는 조합만 보고 끝")
  a = ap.parse_args(argv)

  if a.probe:
    return probe(a.device)

  check_sizes()
  grab = Grabber(a.device, a.width, a.height, a.fps).start()
  print(f"카메라 {a.device} · {grab.width}x{grab.height} · 초당 {a.fps}장 부탁함")

  srv = ThreadingHTTPServer((a.bind, a.port), make_handler(grab))
  srv.daemon_threads = True
  print(f"스트리밍 주소 http://{a.bind}:{a.port}/   (멈추려면 Ctrl-C)")
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    print("\n멈춥니다")
  finally:
    grab.stop()
    srv.server_close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
