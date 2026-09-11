#!/usr/bin/env python3
"""열두 관절을 **웹에서 실시간으로** 보며, 손으로 움직여 방향과 양을 확인한다.

2026-09-11 · 목적은 둘입니다.

  1. **통신이 정상인가** — 열두 개가 빠짐없이, 끊김 없이 답하는가
  2. **각도 방향이 맞는가** — 실물 관절을 손으로 굽혔을 때 숫자가 시뮬에서 의도한
     쪽으로 가는가

★ 이 프로그램은 모터를 움직일 수 없습니다
------------------------------------------
보내는 것은 **세기 0 · 감쇠 0 · 토크 0** 인 프레임 하나뿐입니다. 켜기(0xFC)·
끄기(0xFD)·영점(0xFE)·힘이 들어가는 어떤 값도 **이 파일 안에 존재하지 않습니다.**
화면의 '지금을 0으로' 단추도 **보는 사람의 브라우저 안에서만** 기준을 옮깁니다 —
모터에는 아무것도 쓰지 않습니다.

주의: 어떤 모터에 이미 힘이 들어가 자세를 붙잡고 있었다면, 세기 0 인 이 프레임이
그 힘을 놓게 만듭니다. 화면의 '상태' 칸이 `힘 켜짐` 으로 보이면 사람을 부르세요.

왜 별도로 만드는가
------------------
뷰어에는 명령을 보내는 길이 있습니다. 지금은 **실수로도 모터가 움직이면 안 되는**
단계라, 명령을 만들 방법 자체가 없는 프로그램을 따로 둡니다.

뷰어로도 같이 보낸다
--------------------
`--telemetry <주소>:<포트>` 를 주면 같은 값을 HUPHY 형식 UDP 로 내보냅니다. 뷰어의
3D 화면이 그것을 받아 실물 자세를 그대로 따라 그립니다.

**읽는 곳은 한 군데뿐입니다.** 두 프로그램이 같은 통신선을 따로 읽으면 서로의 응답을
가로채 엉뚱한 값을 집습니다. 그래서 웹 화면과 뷰어 송신이 **같은 읽기 결과**를 씁니다.

보내기만 하고 받지 않습니다 — 뷰어가 내리는 명령을 받는 통로가 이 프로그램에 없습니다.

    joint_web.py --port 8098
    joint_web.py --port 8098 --telemetry 192.168.20.177:9871
    http://<파이 주소>:8098/
"""

from __future__ import annotations

import argparse
import json
import re
import math
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAN_FRAME = "=IB3x8s"
ZERO_FRAME = bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF])
"""목표각 0 · 목표속도 0 · 세기 0 · 감쇠 0 · 토크 0. 이 파일이 만드는 유일한 내용."""

MAP_PATH = os.path.expanduser("~/System_ID/can_map.json")
CONV_PATH = os.path.expanduser("~/System_ID/sim_joint_conv.json")

RANGE = {"RS03": dict(pmax=12.57, vmax=20.0, tmax=60.0),
         "RS04": dict(pmax=12.57, vmax=15.0, tmax=120.0)}

# 모터 번호 -> (다리, 관절, 모델, 한글이름, 시뮬 관절이름)
LAYOUT = {
  1:  ("left",  "hip_pitch", "RS04", "엉덩이 앞뒤", "L_hip_pitch_joint"),
  2:  ("left",  "hip_roll",  "RS04", "엉덩이 좌우", "L_hip_roll_joint"),
  3:  ("left",  "hip_yaw",   "RS03", "엉덩이 회전", "L_hip_yaw_joint"),
  4:  ("left",  "knee",      "RS04", "무릎",        "L_knee_joint"),
  5:  ("left",  "ankle_a",   "RS03", "발목 A",      "L_crank_A_joint"),
  6:  ("left",  "ankle_b",   "RS03", "발목 B",      "L_crank_B_joint"),
  7:  ("right", "hip_pitch", "RS04", "엉덩이 앞뒤", "R_hip_pitch_joint"),
  8:  ("right", "hip_roll",  "RS04", "엉덩이 좌우", "R_hip_roll_joint"),
  9:  ("right", "hip_yaw",   "RS03", "엉덩이 회전", "R_hip_yaw_joint"),
  10: ("right", "knee",      "RS04", "무릎",        "R_knee_joint"),
  11: ("right", "ankle_a",   "RS03", "발목 A",      "R_crank_A_joint"),
  12: ("right", "ankle_b",   "RS03", "발목 B",      "R_crank_B_joint"),
}
MODE_NAME = {0: "힘 꺼짐", 1: "보정중", 2: "힘 켜짐"}


def uint_to_float(x, lo, hi, bits):
  return float(x) / ((1 << bits) - 1) * (hi - lo) + lo


def decode(data: bytes, rng: dict) -> dict:
  q = (data[1] << 8) | data[2]
  dq = (data[3] << 4) | (data[4] >> 4)
  tau = ((data[4] & 0x0F) << 8) | data[5]
  temp = ((data[6] & 0x0F) << 8) | data[7]
  b6 = data[6]
  return dict(
    deg=round(math.degrees(uint_to_float(q, -rng["pmax"], rng["pmax"], 16)), 2),
    vel=round(math.degrees(uint_to_float(dq, -rng["vmax"], rng["vmax"], 12)), 1),
    nm=round(uint_to_float(tau, -rng["tmax"], rng["tmax"], 12), 2),
    temp=round(temp / 10.0, 1),
    mode=(b6 >> 6) & 0x03, fault=bool(b6 & 0x20), warning=bool(b6 & 0x10),
  )


class Poller:
  """열두 모터를 계속 읽어 최신 값을 들고 있다. 버스마다 스레드 하나.

  **버스로 나가는 곳은 여기 한 군데뿐이다.** 다른 프로그램이 따로 보내면 세기 0 인
  프레임과 세기가 있는 프레임이 번갈아 나가 힘이 반으로 깎이고 떨린다. 그래서
  움직이는 명령도 여기서 함께 내보내되, **무엇을 보낼지는 `cmd_guard` 가 정한다.**
  """

  def __init__(self, buses, hz=20.0, guard=None):
    self.buses = buses                      # [{"iface":..,"leg":..,"motors":[..]}]
    self.guard = guard                      # None 이면 읽기만 한다
    self.period = 1.0 / hz
    self.lock = threading.Lock()
    self.state = {}                         # 모터번호 -> 값
    self.bus_stat = {b["iface"]: dict(ok=0, miss=0, leg=b["leg"]) for b in buses}
    self._last = {}
    self._stop = threading.Event()

  def start(self):
    for b in self.buses:
      threading.Thread(target=self._run, args=(b,), daemon=True).start()
    time.sleep(0.5)
    return self

  def stop(self):
    self._stop.set()

  def snapshot(self):
    now = time.monotonic()
    with self.lock:
      out = {}
      for mid, v in self.state.items():
        row = dict(v)
        row["age_s"] = round(now - row.pop("_at"), 2)
        out[mid] = row
      stat = {k: dict(v) for k, v in self.bus_stat.items()}
    return out, stat

  def _run(self, bus):
    iface, ids = bus["iface"], bus["motors"]
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((iface,))
    while not self._stop.is_set():
      for mid in ids:
        if self._stop.is_set():
          break
        model = LAYOUT.get(mid, (None, None, "RS04"))[2]
        sock.settimeout(0.0)
        try:
          while True:
            sock.recv(16)
        except (BlockingIOError, TimeoutError, socket.timeout):
          pass
        # 이번 주기에 이 모터로 보낼 것. 가드가 없거나 할 말이 없으면 세기 0.
        data, last = ZERO_FRAME, self._last.get(mid, 0.0)
        now_t = time.monotonic()
        if self.guard is not None:
          with self.lock:
            v = self.state.get(mid)
          got = self.guard.frame_for(
            mid, v["deg"] if v else 0.0, v["temp"] if v else 0.0,
            now_t - last if last else self.period,
            mode=v["mode"] if v else None)
          if isinstance(got, tuple):
            data = got[1]
          elif got is not None:
            data = got
        self._last[mid] = now_t
        try:
          sock.send(struct.pack(CAN_FRAME, mid, 8, data))
        except OSError:
          time.sleep(0.2)
          continue
        got = None
        end = time.monotonic() + 0.02
        while time.monotonic() < end:
          sock.settimeout(max(0.0, end - time.monotonic()))
          try:
            raw = sock.recv(16)
          except (TimeoutError, socket.timeout):
            break
          _, dlc, data = struct.unpack(CAN_FRAME, raw)
          if dlc >= 8 and data[0] == mid:
            got = decode(data[:8], RANGE[model])
            break
        with self.lock:
          if got is None:
            self.bus_stat[iface]["miss"] += 1
          else:
            self.bus_stat[iface]["ok"] += 1
            got["_at"] = time.monotonic()
            self.state[mid] = got
      time.sleep(self.period)
    sock.close()


PROG_READ_ONLY = 4
PROG_SLIDER_DRY = 5
PROG_SLIDER_LIVE = 6
"""이 프로그램이 뷰어에 대는 이름표. `pygviewer/scenario.py` 의 `PROGRAMS` 와
**같은 번호여야 합니다** -- 한쪽만 고치면 뷰어가 조용히 `unknown` 으로 돌아갑니다.

  4  읽기 전용        세기 0 프레임만. 힘이 **나갈 수 없음**
  5  슬라이더 시험    명령을 계산만 하고 보내지 않음. 힘이 **안 나감**
  6  슬라이더 실행    단추를 누르면 힘이 **나감**
"""


class TelemetrySender:
  """읽은 값을 HUPHY 형식 UDP 로 내보낸다. **보내기만 한다.**

      {"t": 초, "loop_dt": 밀리초,
       "left_leg/knee/pos": 도, ".../tgt": 도, ".../err": 도,
       ".../vel": 도/초, ".../tau": 뉴턴미터, ".../temp": 도, ...}

  힘을 넣지 않으므로 `tgt` 는 언제나 `pos` 와 같고 `err` 는 0 입니다. 뷰어에 보이는
  움직임은 **전부 사람이 손으로 만든 것**입니다.
  """

  def __init__(self, poller, target: str, hz: float = 50.0, guard=None):
    host, _, port = target.partition(":")
    self.addr = (host, int(port or 9871))
    self.poller, self.period = poller, 1.0 / hz
    self.guard = guard
    """무엇이 돌고 있는지 뷰어에 알려 주려고 들고 있는다 (아래 `prog`).

    뷰어(`pygviewer/scenario.py`)는 "지금 로봇에서 무엇이 도는가"를 **추측하지
    않습니다** -- 로봇이 스스로 말해 주지 않으면 `unknown` 으로 두고, 이름이 붙는
    조합도 만들지 않습니다. 그래서 이 프로그램도 자기 이름을 말해야 합니다.

    셋 중 하나입니다. 힘이 나가는지 안 나가는지가 갈리므로 **번호를 따로 씁니다** --
    한 번호에 묶으면 뷰어가 "힘 꺼짐" 이라고 적어 둔 채 실제로는 나갈 수 있습니다."""
    self.sent = 0
    self._stop = threading.Event()
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

  def start(self):
    threading.Thread(target=self._run, daemon=True).start()
    return self

  def stop(self):
    self._stop.set()

  def _run(self):
    t0 = last = time.monotonic()
    while not self._stop.is_set():
      now = time.monotonic()
      state, _ = self.poller.snapshot()
      pkt = {"t": round(now - t0, 4), "loop_dt": round((now - last) * 1000.0, 3)}
      last = now
      prog = PROG_READ_ONLY
      if self.guard is not None:
        prog = PROG_SLIDER_DRY if self.guard.dry_run else PROG_SLIDER_LIVE
      for mid, (leg, motor, _m, _ko, _sim) in LAYOUT.items():
        v = state.get(mid)
        if v is None:
          continue
        k = f"{leg}_leg/{motor}"
        pkt[f"{k}/pos"] = v["deg"]
        pkt[f"{k}/tgt"] = v["deg"]          # 힘을 안 넣으므로 목표 = 지금 자리
        pkt[f"{k}/err"] = 0.0
        pkt[f"{k}/vel"] = v["vel"]
        pkt[f"{k}/tau"] = v["nm"]
        pkt[f"{k}/temp"] = v["temp"]
        pkt[f"{k}/age"] = round(v["age_s"] * 1000.0, 1)
        pkt[f"{k}/ack"] = 1.0 if v["age_s"] < 0.5 else 0.0
        pkt[f"{k}/miss"] = 0.0
        pkt[f"{k}/prog"] = prog
      try:
        self.sock.sendto(json.dumps(pkt).encode(), self.addr)
        self.sent += 1
      except OSError:
        pass
      time.sleep(max(0.0, self.period - (time.monotonic() - now)))


class CamWatch:
  """카메라 서버에 **우리가 대신 물어봐 준다.**

  왜 화면이 직접 안 묻는가: 관절 화면(8098)과 카메라(8099)는 포트가 달라 브라우저가
  서로 묻는 것을 막습니다. 서버끼리 물으면 그 제약이 없습니다.

  왜 필요한가: 벤치가 가만히 있으면 **멎은 화면과 살아 있는 화면이 똑같이 보입니다.**
  숫자가 올라가는 것을 보여 주는 것이 유일하게 확실한 증거입니다.
  """

  def __init__(self, url: str, period: float = 1.0):
    self.url, self.period = url, period
    self.data = {"reachable": False}
    self._stop = threading.Event()

  def start(self):
    if self.url:
      threading.Thread(target=self._run, daemon=True).start()
    return self

  def stop(self):
    self._stop.set()

  def _run(self):
    import urllib.request
    while not self._stop.is_set():
      try:
        with urllib.request.urlopen(self.url, timeout=2.0) as r:
          d = json.loads(r.read())
        d["reachable"] = True
        self.data = d
      except Exception as exc:                                  # noqa: BLE001
        self.data = {"reachable": False, "why": f"{type(exc).__name__}"}
      self._stop.wait(self.period)


NEEDS_LIMIT = ("L_knee_joint", "R_knee_joint",
               "L_hip_pitch_joint", "R_hip_pitch_joint",
               "L_hip_roll_joint", "R_hip_roll_joint")
"""절대 한계 없이는 열지 않는 관절.

무릎은 **0도가 곧 하드스톱**이라 상대 구간만으로는 스톱 너머로 갈 수 있습니다
(2026-09-11 에 실제로 9.5도 넘겼습니다). 엉덩이 앞뒤·좌우는 **다리 전체를 드는**
관절이라 약한 세기로는 처지고, 처지다가 스톱에 부딪히는 것을 막을 방법이
한계뿐입니다."""
"""한쪽 끝이 하드스톱이라 **상대 구간만으로는 열 수 없는** 관절.

무릎은 완전히 편 자세가 곧 끝이고 그 너머는 없습니다. "지금 자리에서 ±N도" 만
걸면 그 끝을 넘어 밀 수 있습니다 — 2026-09-12 에 실제로 그렇게 열었고,
하드스톱 쪽으로 9.5도까지 밀 수 있는 상태였습니다."""


def _split_pairs(text: str):
  """`이름=값` 을 **쉼표로든 공백으로든** 나눈다.

  2026-09-12: 실행 스크립트에서 공백으로 적었다가 값이 통째로 하나로 읽혀
  `float("8 R_hip_pitch_joint=8 ...")` 로 죽었다. 어느 쪽으로 적어도 되게 한다 --
  구분자를 기억하게 만드는 것은 사람 몫이 아니다.
  """
  for part in re.split(r"[,\s]+", (text or "").strip()):
    part = part.strip()
    if part:
      yield part


def _kv2(text: str) -> dict:
  """`이름=낮은쪽:높은쪽` 을 사전으로."""
  out = {}
  for part in _split_pairs(text):
    name, _, rng = part.partition("=")
    a, _, b = rng.partition(":")
    out[name.strip()] = (float(a), float(b))
  return out


def _kv(text: str) -> dict:
  """`이름=값,이름=값` 을 사전으로. 비어 있으면 빈 사전."""
  out = {}
  for part in _split_pairs(text):
    name, _, val = part.partition("=")
    out[name.strip()] = float(val)
  return out


def build_rows(conv):
  rows = []
  for mid, (leg, motor, model, ko, sim) in sorted(LAYOUT.items()):
    c = conv.get(sim) or {}
    rows.append(dict(id=mid, leg=leg, motor=motor, model=model, ko=ko, sim=sim,
                     lo=c.get("lo"), hi=c.get("hi"), axis=c.get("axis")))
  return rows


PAGE_HEAD = """<!doctype html><meta charset="utf-8"><title>관절 확인</title>
<style>
 :root{--bg:#14161a;--pan:#1c1f26;--line:#2a2e37;--tx:#e8e8e8;--dim:#8b93a1;
       --ok:#2f9e68;--warn:#c98a15;--bad:#c0392b;--acc:#5b9bd5}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 system-ui,sans-serif}
 header{padding:10px 14px;background:var(--pan);border-bottom:1px solid var(--line);
        display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:15px;margin:0}
 .dim{color:var(--dim)}
 .wrap{display:grid;grid-template-columns:minmax(420px,1fr) minmax(520px,1.1fr);gap:12px;padding:12px}
 @media(max-width:1100px){.wrap{grid-template-columns:1fr}}
 .card{background:var(--pan);border:1px solid var(--line);border-radius:8px;overflow:hidden}
 .card h2{font-size:13px;margin:0;padding:8px 12px;border-bottom:1px solid var(--line);
          color:var(--dim);font-weight:600;letter-spacing:.02em}
 table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #23262e}
 th{color:var(--dim);font-weight:500;font-size:12px}
 td.l,th.l{text-align:left}
 .big{font-size:16px;font-weight:600}
 .bar{position:relative;height:8px;background:#23262e;border-radius:4px;min-width:110px}
 .bar i{position:absolute;top:0;bottom:0;background:var(--acc);border-radius:4px}
 .bar u{position:absolute;top:-2px;bottom:-2px;width:2px;background:#fff;opacity:.5}
 .pill{padding:1px 6px;border-radius:10px;font-size:11px}
 .pill.ok{background:#17331f;color:#7fd6a0}
 .pill.bad{background:#3a1414;color:#ff9c8f}
 .pill.warn{background:#3a2d10;color:#ffcf70}
 img{display:block;width:100%;background:#000}
 button{background:#2a2e37;color:var(--tx);border:1px solid #3a3f4a;border-radius:8px;
        padding:9px 14px;cursor:pointer;font:inherit;min-height:44px;touch-action:manipulation}
 button:hover{background:#333844}
 button.on{background:#7f1d1d;border-color:#a33;color:#fff;font-weight:700}
 button:disabled{opacity:.38;cursor:not-allowed}
 button:disabled:hover{background:#2a2e37}
 button#lock{background:#14532d;border-color:#2f7d4f;color:#d7f5e1;font-weight:700}
 button#lock.open{background:#7a5c10;border-color:#c9a227;color:#fff}
 #ctlbody.locked input[type=range]{opacity:.38;pointer-events:none}
 #lockbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
          border:1px solid #3a3f4a;border-radius:8px;padding:9px 11px;margin-bottom:10px}
 input[type=range]{height:38px;accent-color:var(--acc);touch-action:manipulation}
 @media(max-width:700px){
   header{gap:8px} h1{font-size:14px}
   .wrap{padding:8px;gap:8px}
   th,td{padding:4px 5px;font-size:12px}
 }
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#444;
      margin-left:6px;vertical-align:middle}
 .dot.beat{background:#7fd6a0;box-shadow:0 0 0 0 rgba(127,214,160,.7);animation:b .45s ease-out}
 .dot.dead{background:#ff6b5b}
 @keyframes b{from{box-shadow:0 0 0 0 rgba(127,214,160,.8)}to{box-shadow:0 0 0 9px rgba(127,214,160,0)}}
 #banner{display:none;padding:9px 14px;background:var(--bad);font-weight:600}
 #banner.on{display:block}
</style>
<header>
  <h1 id="title">실물 관절 확인</h1>
  <span class="dim" id="subtitle">불러오는 중…</span>
  <span class="dim" id="rate"></span>
  <button onclick="zeroAll()">지금 자세를 0으로</button>
  <button onclick="clearZero()">기준 지우기</button>
</header>
<div style="padding:7px 14px;background:#3a2d10;color:#ffcf70;font-size:13px">
  &#9888; 줄 이름의 <b>왼/오른</b>은 설정 파일에 적힌 짝이지 실물에서 확인한 것이
  아닙니다. 확인하려면 한쪽 무릎을 손으로 굽혀 보고, <b>오른 무릎(10번)</b>이 움직이면
  설정이 맞고 <b>왼 무릎(4번)</b>이 움직이면 좌우가 뒤바뀐 것입니다.
</div>
<div id="banner"></div>
"""


def make_handler(poller, rows, cam_url, cam_watch, viser_url, guard=None):
  body_rows = json.dumps(rows, ensure_ascii=False)

  class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
      pass

    def _send(self, code, ctype, body):
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      p = self.path.split("?")[0]
      if p in ("/", "/index.html"):
        self._send(200, "text/html; charset=utf-8", self._page().encode("utf-8"))
      elif p in ("/cam.mjpg", "/cam.jpg"):
        # 카메라를 **이 서버가 대신 받아 넘긴다.** 화면 안에 내부 주소를 박아 두면
        # 바깥(터널)에서 열었을 때 그림만 안 나온다 - 2026-09-11 실제로 그랬다.
        self._proxy_cam(single=(p == "/cam.jpg"))
      elif p == "/joints.json":
        state, stat = poller.snapshot()
        self._send(200, "application/json",
                   json.dumps(dict(motors=state, buses=stat, cam=cam_watch.data,
                                   cmd=(guard.status() if guard else None),
                                   t=round(time.time(), 2))).encode())
      else:
        self._send(404, "text/plain; charset=utf-8", "없는 주소".encode())

    def do_POST(self):
      """화면에서 오는 목표 하나. **누르고 있는 동안만** 들어온다.

      손을 떼면 이 요청이 끊기고, 0.3초 뒤 가드가 힘을 뺀다 (`deadman_s`).
      화면을 닫든 인터넷이 끊기든 같은 일이 일어난다 - 원격에서 쓰기에 이 편이
      '무장해 두면 계속 나가는' 방식보다 안전하다.
      """
      if self.path.split("?")[0] == "/release" and guard is not None:
        # 사람이 멈춤을 눌렀다. **데드맨을 기다리지 않고 바로 끊는다.**
        try:
          n = int(self.headers.get("Content-Length", 0))
          body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, TypeError):
          body = {}
        r = guard.release_by_name(str(body.get("name", "")), str(body.get("token", "")))
        self._send(200 if r.get("ok") else 403, "application/json",
                   json.dumps(r, ensure_ascii=False).encode())
        return
      if self.path.split("?")[0] != "/cmd" or guard is None:
        self._send(404, "text/plain; charset=utf-8", "없는 주소".encode())
        return
      try:
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
      except (ValueError, TypeError):
        self._send(400, "application/json",
                   json.dumps(dict(ok=False, why="못 읽음"), ensure_ascii=False).encode())
        return
      r = guard.local_target(str(body.get("name", "")), float(body.get("deg", 0.0)),
                             str(body.get("token", "")))
      self._send(200 if r.get("ok") else 403, "application/json",
                 json.dumps(r, ensure_ascii=False).encode())

    def _proxy_cam(self, single: bool) -> None:
      import urllib.request
      src = cam_url.rsplit("/", 1)[0] + ("/snapshot.jpg" if single else "/stream.mjpg")
      try:
        up = urllib.request.urlopen(src, timeout=5.0)
      except Exception as exc:                                  # noqa: BLE001
        self._send(502, "text/plain; charset=utf-8",
                   f"카메라 서버에 못 닿음: {type(exc).__name__}".encode())
        return
      if single:
        self._send(200, "image/jpeg", up.read())
        return
      self.send_response(200)
      self.send_header("Content-Type", up.headers.get(
        "Content-Type", "multipart/x-mixed-replace; boundary=huphyframe"))
      self.send_header("Cache-Control", "no-store")
      self.send_header("Connection", "close")
      self.end_headers()
      try:
        while True:
          chunk = up.read(32768)
          if not chunk:
            break
          self.wfile.write(chunk)
      except (BrokenPipeError, ConnectionResetError, OSError):
        pass                                  # 보던 사람이 창을 닫음. 정상임
      finally:
        up.close()
      self.close_connection = True

    def _page(self):
      return PAGE_HEAD + f"""
<div class="wrap">
  <div>
    <div class="card"><h2>벤치 화면 <span id="camlive"></span></h2>
      <img src="/cam.mjpg" alt="벤치">
      <table id="cam"></table></div>
    <div class="card" style="margin-top:12px"><h2>통신 상태</h2>
      <table id="bus"></table></div>
    <div class="card" style="margin-top:12px"><h2>3D 화면</h2>
      <div style="padding:8px 12px"><a href="{viser_url}" target="_blank"
        style="color:#5b9bd5">{viser_url} 에서 열기</a>
        <div class="dim" style="margin-top:4px">실물 관절이 그대로 반영됩니다.
        이 화면과 같은 값을 씁니다.</div></div></div>
  </div>
  <div class="card" id="ctlcard" style="display:none;margin-bottom:12px">
    <h2>관절 움직이기 <span id="ctlmode"></span></h2>
    <div style="padding:10px 12px">
      <div id="tokrow" style="margin-bottom:8px;display:none">
        <label class="dim">암호</label>
        <input id="tok" type="password" style="width:170px;background:#12141a;color:#e8e8e8;
          border:1px solid #3a3f4a;border-radius:6px;padding:8px 9px" placeholder="arm token">
      </div>
      <div id="lockbar">
        <button id="lock" aria-pressed="true">\U0001F512 \uC7A0\uAE40</button>
        <span class="dim" id="locknote"></span>
      </div>
      <div class="dim" style="margin-bottom:8px">
        단추를 켜면 움직이고, 끄면 멈춥니다. <b>화면을 닫거나 다른 앱으로 가면
        0.3초 안에 힘이 빠집니다.</b>
      </div>
      <div id="ctlbody" class="locked">
      <div style="margin-bottom:10px">
        <button id="allzero" style="font-weight:700">전부 0\u00B0 로</button>
        <button id="allstop">전부 멈춤 (자세는 잡고 있음)</button>
        <button id="allfree">전부 힘 빼기</button>
        <span class="dim" id="allnote" style="margin-left:8px"></span>
      </div>
      <div id="ctl"></div>
      </div>
      <div id="ctlstat" class="dim" style="margin-top:8px"></div>
    </div>
  </div>
  <div class="card"><h2>관절 &mdash; 손으로 움직여 보세요</h2>
    <table id="j"><thead><tr>
      <th class="l">관절</th><th>지금(도)</th><th>0 기준 변화</th>
      <th class="l">시뮬에서의 범위</th><th class="l">상태</th>
    </tr></thead><tbody></tbody></table></div>
</div>
<script>
const ROWS = {body_rows};

// ------------------------------------------------------------------ 잠금
// 화면을 열면 **언제나 잠긴 채로** 시작한다. 새로고침이 잠금을 푸는 일이 있으면
// 안 되므로 저장하지 않는다 -- 주머니 속에서 눌리거나 남이 열었을 때,
// "지난번에 풀어 뒀다" 가 이유가 되면 안 된다.
//
// ★ '전부 멈춤' 은 **절대 잠그지 않는다.** 멈추는 길이 잠겨 있으면 그건 잠금이
//   아니라 덫이다. 잠그는 행위 자체도 돌고 있던 관절을 전부 멈춘다.
let LOCKED = true;
function setLocked(on, why){{
  LOCKED = !!on;
  const b = document.getElementById('lock'), body = document.getElementById('ctlbody');
  if (!b || !body) return;
  b.textContent = LOCKED ? '\U0001F512 \uC7A0\uAE40' : '\U0001F513 \uD480\uB9BC';
  b.classList.toggle('open', !LOCKED);
  b.setAttribute('aria-pressed', LOCKED ? 'true' : 'false');
  body.classList.toggle('locked', LOCKED);
  document.getElementById('locknote').textContent = LOCKED
    ? '\uC7A0\uACA8 \uC788\uC2B5\uB2C8\uB2E4 \u2014 \uD480\uC5B4\uC57C '
      + '\uAD00\uC808\uC744 \uC6C0\uC9C1\uC77C \uC218 \uC788\uC2B5\uB2C8\uB2E4'
      + (why ? ' (' + why + ')' : '')
    : '\uD480\uB824 \uC788\uC2B5\uB2C8\uB2E4 \u2014 \uB2E8\uCD94\uAC00 '
      + '\uBC14\uB85C \uB4E3\uC2B5\uB2C8\uB2E4';
  for (const el of document.querySelectorAll('#ctlbody button')) {{
    if (el.id === 'allstop' || el.id === 'allfree') continue;   // 멈추고 힘 빼는 길은 늘 열려 있다
    el.disabled = LOCKED;
  }}
  for (const el of document.querySelectorAll('#ctlbody input[type=range]'))
    el.disabled = LOCKED;
  if (LOCKED && window.__HOOKS)
    for (const h of Object.values(window.__HOOKS))
      h.stop('\uC7A0\uAE08');   // 잠그면 보내기만 멈춘다. 힘은 그대로 -- 목표 위치를 잡고 있어야 한다
}}
let base = JSON.parse(localStorage.getItem('base') || '{{}}');
function zeroAll(){{ fetch('/joints.json',{{cache:'no-store'}}).then(r=>r.json()).then(j=>{{
    base = {{}}; for (const k in j.motors) base[k] = j.motors[k].deg;
    localStorage.setItem('base', JSON.stringify(base)); }}); }}
function clearZero(){{ base = {{}}; localStorage.removeItem('base'); }}
function bar(v, lo, hi){{
  if (lo==null||hi==null) return '<div class="bar"></div>';
  const f = Math.max(0, Math.min(1, (v-lo)/(hi-lo)));
  const z = Math.max(0, Math.min(1, (0-lo)/(hi-lo)));
  return `<div class="bar"><i style="left:0;width:${{(f*100).toFixed(1)}}%"></i>`
       + `<u style="left:${{(z*100).toFixed(1)}}%"></u></div>`;
}}
async function tick(){{
  let j; try{{ j = await (await fetch('/joints.json',{{cache:'no-store'}})).json(); }}
  catch(e){{ document.getElementById('banner').className='on';
             document.getElementById('banner').textContent='\\u2716 \\uc11c\\ubc84 \\uc5f0\\uacb0 \\ub04a\\uae40'; return; }}
  let hot = [], live = 0;
  const tb = document.querySelector('#j tbody'); tb.innerHTML = '';
  for (const r of ROWS){{
    const m = j.motors[r.id];
    let cur='&mdash;', dlt='', rng='', st='<span class="pill bad">응답 없음</span>';
    if (m){{
      live++;
      const d = (base[r.id]!=null) ? (m.deg - base[r.id]) : null;
      cur = `<span class="big">${{m.deg.toFixed(2)}}</span>`;
      dlt = d==null ? '<span class="dim">기준 없음</span>'
                    : `<span class="big">${{d>0?'+':''}}${{d.toFixed(2)}}</span>`;
      rng = (r.lo==null) ? '<span class="dim">모름</span>'
          : `${{bar(m.deg, r.lo, r.hi)}}<span class="dim">${{r.lo}} ~ ${{r.hi}}\\uB3C4</span>`;
      const mode = m.mode===2 ? '<span class="pill bad">힘 켜짐</span>'
                 : m.mode===0 ? '<span class="pill ok">힘 꺼짐</span>'
                 : '<span class="pill warn">보정중</span>';
      st = mode + (m.fault?' <span class="pill bad">고장</span>':'')
                + (m.warning?' <span class="pill warn">경고</span>':'')
                + (m.age_s>1 ? ` <span class="pill warn">${{m.age_s}}초 전</span>` : '')
                + ` <span class="dim">${{m.temp}}\\uB3C4</span>`;
      if (m.temp >= 45) hot.push(`${{r.ko}} ${{m.temp}}\\uB3C4`);
    }}
    tb.insertAdjacentHTML('beforeend',
      `<tr><td class="l">${{r.leg==='left'?'왼':'오른'}} ${{r.ko}}`
      + ` <span class="dim">${{r.id}}번 · ${{r.sim}}</span></td>`
      + `<td>${{cur}}</td><td>${{dlt}}</td><td class="l">${{rng}}</td><td class="l">${{st}}</td></tr>`);
  }}
  // ---- 카메라가 정말 새 장면을 받고 있는지 ----------------------------
  const c = j.cam || {{}}, CD = document.getElementById('camlive'),
        CT = document.getElementById('cam');
  if (!c.reachable){{
    CD.innerHTML = '<span class="dot dead"></span>';
    CT.innerHTML = '<tr><td class="l"><span class="pill bad">카메라 서버에 못 물어봄</span>'
      + ` <span class="dim">${{c.why||''}}</span></td></tr>`;
    hot.push('\uCE74\uBA54\uB77C \uC11C\uBC84 \uC751\uB2F5 \uC5C6\uC74C');
  }} else {{
    const grew = (window._camFrames != null) && (c.frames > window._camFrames);
    window._camFrames = c.frames;
    if (grew){{                       // 새 장면이 올 때마다 점을 한 번 뛰게 한다
      CD.innerHTML = '<span class="dot"></span>';
      const d = CD.firstChild; void d.offsetWidth; d.className = 'dot beat';
    }} else if (!CD.innerHTML) CD.innerHTML = '<span class="dot"></span>';
    if (c.stalled){{
      CD.innerHTML = '<span class="dot dead"></span>';
      hot.push(`\uD654\uBA74\uC774 \uBA48\uCCA0\uC74C (${{c.age_s}}\uCD08 \uC804)`);
    }}
    CT.innerHTML =
      `<tr><th class="l">받은 장수</th><td class="l"><span class="big">${{c.frames}}</span>`
      + ` <span class="dim">\uACC4\uC18D \uC62C\uB77C\uAC00\uBA74 \uC0B4\uC544\uC788\uB294 \uAC83</span></td></tr>`
      + `<tr><th class="l">초당 장수</th><td class="l">${{c.fps}}</td></tr>`
      + `<tr><th class="l">마지막 장면</th><td class="l">`
      + (c.stalled ? `<span class="pill bad">${{c.age_s}}초 전</span>`
                   : `<span class="pill ok">${{c.age_s}}초 전</span>`) + `</td></tr>`
      + `<tr><th class="l">놓친 장수</th><td class="l">${{c.dropped}}`
      + (c.restarts ? ` <span class="pill warn">재시작 ${{c.restarts}}회</span>` : '') + `</td></tr>`;
  }}

  // ---- 움직이기 칸 ----------------------------------------------------
  const K = j.cmd;
  if (K){{
    document.getElementById('ctlcard').style.display = '';
    document.getElementById('tokrow').style.display = K.needs_token ? '' : 'none';
    document.getElementById('title').textContent = K.dry_run
      ? '\uC2E4\uBB3C \uAD00\uC808 \u2014 \uC2DC\uD5D8 \uBAA8\uB4DC'
      : '\uC2E4\uBB3C \uAD00\uC808 \u2014 \u26A0 \uC9C4\uC9DC\uB85C \uC6C0\uC9C1\uC785\uB2C8\uB2E4';
    document.getElementById('subtitle').textContent = K.dry_run
      ? '\uBA85\uB839\uC744 \uACC4\uC0B0\uB9CC \uD558\uACE0 \uBCF4\uB0B4\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4'
      : '\uB2E8\uCD94\uB97C \uB204\uB974\uBA74 \uBAA8\uD130\uC5D0 \uD798\uC774 \uB4E4\uC5B4\uAC11\uB2C8\uB2E4';
    document.getElementById('ctlmode').innerHTML = K.dry_run
      ? '<span class="pill warn">시험 모드 &mdash; 힘이 안 들어감</span>'
      : '<span class="pill bad">힘이 들어갑니다</span>';
    if (!window._ctlBuilt){{
      window._ctlBuilt = true;
      const box = document.getElementById('ctl');
      for (const [mid, m] of Object.entries(K.motors)){{
        box.insertAdjacentHTML('beforeend',
          `<div style="margin:7px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap">`
          + `<b style="min-width:104px">${{m.ko}}</b>`
          + `<span class="dim">${{mid}}번</span>`
          + `<input type="range" id="s${{mid}}" min="${{m.lo}}" max="${{m.hi}}"`
          + ` step="0.5" value="${{m.home}}" style="width:210px">`
          + `<span id="v${{mid}}" class="big" style="min-width:62px">+0.0&deg;</span>`
          + `<button id="z${{mid}}" title="이 관절을 영자세로">0\u00B0 \uB85C</button>`
          + `<button id="b${{mid}}">움직이기</button>`
          + `<span id="n${{mid}}" class="dim"></span></div>`);
      }}
      window.__HOME = {{}};
      for (const [mid, m] of Object.entries(K.motors)) window.__HOME[mid] = m.home;
      window.__HOOKS = {{}};
      for (const mid of Object.keys(K.motors)) hookToggle(mid);
      // 전부 한꺼번에. 관절 하나짜리 단추와 **같은 길**을 부른다 -- 따로 만들면
      // 한쪽만 고쳐지는 일이 생긴다.
      document.getElementById('allzero').addEventListener('click', () => {{
        for (const h of Object.values(window.__HOOKS)) h.toZero();
        document.getElementById('allnote').textContent =
          '\uC804\uBD80 \uC601\uC790\uC138\uB85C \uAC00\uB294 \uC911';
      }});
      document.getElementById('allstop').addEventListener('click', () => {{
        for (const h of Object.values(window.__HOOKS)) h.stop('\uC804\uBD80 \uBA48\uCDA4');
        document.getElementById('allnote').textContent =
          '\uBCF4\uB0B4\uAE30\uB9CC \uBA48\uCDA4 \u2014 \uC790\uC138\uB294 \uC7A1\uACE0 \uC788\uC2B5\uB2C8\uB2E4';
      }});
      document.getElementById('allfree').addEventListener('click', () => {{
        for (const h of Object.values(window.__HOOKS)) h.free('\uC804\uBD80 \uD798 \uBE7C\uAE30');
        document.getElementById('allnote').textContent =
          '\uD798\uC744 \uBE84\uC2B5\uB2C8\uB2E4 \u2014 \uC911\uB825\uC73C\uB85C \uCC98\uC9D1\uB2C8\uB2E4';
      }});
      const lb = document.getElementById('lock');
      if (!lb.dataset.hooked) {{
        lb.dataset.hooked = '1';
        lb.addEventListener('click', () => setLocked(!LOCKED));
      }}
      setLocked(LOCKED);        // 새로 만든 단추·슬라이더에도 지금 상태를 입힌다
    }}
    const rows = Object.entries(K.motors).map(([mid, m]) =>
      `${{m.ko}}: ${{m.torque_on ? '<span class="pill bad">힘 켜짐</span>' : '<span class="pill ok">힘 꺼짐</span>'}}`
      + (m.q0 != null ? ` 기준 ${{m.q0.toFixed(2)}}\u00B0` : '')
      + (m.cmd != null ? ` \u2192 목표 ${{m.cmd.toFixed(2)}}\u00B0` : '')).join(' &nbsp;|&nbsp; ');
    document.getElementById('ctlstat').innerHTML = rows
      + `<br>세기 ${{K.kp}}/${{K.kd}} · 구간 \u00B1${{K.span_deg}}\u00B0 · 속도 ${{K.max_vel_deg_s}}\u00B0/s`
      + ` · 마지막 명령 ${{K.rx_age_s == null ? '없음' : K.rx_age_s + '초 전'}}`
      + ` · 잘림 ${{K.stats.clipped}} · 암호틀림 ${{K.stats.bad_token}}`;
    // 힘이 켜진 것은 **경고가 아니다** -- 사람이 그렇게 시킨 것이고, 단추가 이미
    // '멈추기' 로 바뀌어 있다. 여기에 넣었더니 관절 12개가 붉은 띠로 화면 위를
    // 덮어서, 정작 봐야 할 과열·통신두절이 그 안에 묻혔다 (2026-09-12 사용자 지적).
    // 띠에는 **사람이 시키지 않았는데 잘못된 것**만 올린다.
    for (const [mid, m] of Object.entries(K.motors))
      if (m.temp_cut) hot.push(`${{m.ko}} \\uB108\\uBB34 \\uB728\\uAC70\\uC6CC \\uB04A\\uC74C`);
  }}

  else {{
    document.getElementById('title').textContent =
      '\uC2E4\uBB3C \uAD00\uC808 \uD655\uC778 \u2014 \uC77D\uAE30 \uC804\uC6A9';
    document.getElementById('subtitle').textContent =
      '\uC774 \uD654\uBA74\uC740 \uBAA8\uD130\uB97C \uC6C0\uC9C1\uC77C \uC218 '
      + '\uC5C6\uC2B5\uB2C8\uB2E4 (\uC138\uAE30 0 \u00B7 \uD1A0\uD06C 0 \uC778 '
      + '\uD504\uB808\uC784\uB9CC \uBCF4\uB0C4)';
  }}

  const bt = document.getElementById('bus');
  bt.innerHTML = '<tr><th class="l">선</th><th class="l">다리</th><th>응답</th><th>무응답</th></tr>'
    + Object.entries(j.buses).map(([k,v])=>
      `<tr><td class="l">${{k}}</td><td class="l">${{v.leg||'?'}}</td>`
      + `<td>${{v.ok}}</td><td>${{v.miss>0?'<span class="pill bad">'+v.miss+'</span>':0}}</td></tr>`).join('');
  document.getElementById('rate').textContent = `\\uC751\\uB2F5 ${{live}}/12`;
  const B = document.getElementById('banner');
  if (hot.length){{ B.className='on'; B.textContent='\\u2716 ' + hot.join(' \\u00B7 '); }}
  else B.className='';
}}
function hookToggle(mid){{
  const sl = document.getElementById('s'+mid), vv = document.getElementById('v'+mid),
        bt2 = document.getElementById('b'+mid), nn = document.getElementById('n'+mid),
        zb = document.getElementById('z'+mid);
  const NAME = {{}};
  for (const r of ROWS) NAME[r.id] = r.sim;
  const HOME = window.__HOME || {{}};
  let timer = null, base = 0, started = false;
  // 2026-09-12 사용자 지시: *"모터 지멋대로 계속 꺼지잖아. 내가 풀면 꺼지도록해.
  // 니멋대로 끄지마."*
  //
  // 그래서 **스스로 끄는 것을 전부 없앴다.** 예전에는 넷이 있었다:
  //   1. 120초가 지나면 꺼짐            -> 없앰
  //   2. 화면이 뒤로 가면(blur) 꺼짐     -> 없앰. 다른 창을 누르는 것은 손을 뗀 것이 아니다
  //   3. 화면이 가려지면 꺼짐            -> 없앰
  //   4. 명령이 0.3초 끊기면 꺼짐        -> 남기되 **넉넉하게**. 아래 참고
  //
  // 남은 것은 셋뿐이고 전부 **사람이 한 일이거나, 사람이 사라진 것**이다:
  //   * 멈춤을 누름 / 자물쇠를 잠금  -> `/release` 로 **바로** 끊는다
  //   * 화면이 실제로 닫힘(pagehide) -> 같은 길
  //   * 화면이 죽어 명령이 끊김       -> 데드맨. 이건 꺼야 한다 -- 안 그러면 아무도
  //     안 보는 로봇에 힘이 계속 들어가 있다
  const show = () => vv.textContent = (sl.value >= 0 ? '+' : '') + (+sl.value).toFixed(1) + '\u00B0';
  sl.addEventListener('input', show); show();

  async function beat(){{
    if (!timer && started) return;        // 이미 멈췄으면 아무것도 쓰지 않는다
    const el = document.getElementById('tok');
    try{{
      const r = await fetch('/cmd', {{method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{name: NAME[mid], deg: (+sl.value),
                               token: el ? el.value : ''}})}});
      const j2 = await r.json();
      // 기다리는 동안 멈췄을 수 있다. 늦게 온 답이 멈춘 관절을 "도는 중" 으로
      // 되돌려 쓰면 화면이 거짓말을 한다.
      if (!timer) return;
      nn.textContent = !j2.ok ? ('거부: ' + (j2.why||''))
        : j2.homing
          ? '\uC601\uC790\uC138\uB85C \uB3CC\uC544\uAC00\uB294 \uC911'
          : `\uBAA9\uD45C ${{(+sl.value).toFixed(1)}}\u00B0 \uC720\uC9C0 \uC911`;
      if (!j2.ok) stop('거부됨');
    }}catch(e){{ stop('연결 끊김'); }}
  }}
  function start(){{
    // 켜는 순간의 자리를 기준으로 삼지 않는다 -- 로봇 쪽 가드가 **먼저 영자세로**
    // 데려가고, 도착한 뒤에야 슬라이더 값을 쓴다. 그래서 여기서 읽을 것이 없다.
    //
    // 잠금은 여기서도 본다. 단추를 못 누르게 하는 것만으로는 부족하다 -- 이 함수는
    // '전부 0도로' 처럼 다른 코드에서도 불리므로, 길목 자체를 막아야 한다.
    if (LOCKED) {{ nn.textContent = '\uC7A0\uACA8 \uC788\uC2B5\uB2C8\uB2E4'; return; }}
    if (timer) return;
    started = true;
    timer = setInterval(beat, 100); beat();
    bt2.classList.add('on'); bt2.textContent = '멈추기';
  }}
  // 멈춤 = **보내기를 멈춘다.** 힘은 그대로 둔다.
  //
  // 슬라이더는 목표 위치이므로, 힘을 빼면 중력이 관절을 끌어내리고 화면과 실물이
  // 어긋난다 (2026-09-12 사용자 지적). 그래서 멈춤은 "마지막 목표를 잡은 채로
  // 손을 뗀다" 는 뜻이다. 힘을 빼는 것은 `free()` 하나뿐이고, 사람이 따로 누른다.
  function stop(why){{
    if (!timer) return;
    clearInterval(timer); timer = null;
    bt2.classList.remove('on'); bt2.textContent = '움직이기';
    nn.textContent = (why || '멈춤') + ' \u2014 \uC790\uC138\uB294 \uC7A1\uACE0 \uC788\uC2B5\uB2C8\uB2E4';
  }}
  function free(why){{
    if (timer) {{ clearInterval(timer); timer = null; }}
    bt2.classList.remove('on'); bt2.textContent = '움직이기';
    nn.textContent = (why || '\uD798 \uBE7C\uAE30') + ' \u2014 \uD798\uC774 \uBE60\uC84C\uC2B5\uB2C8\uB2E4';
    const el = document.getElementById('tok');
    try{{
      fetch('/release', {{method:'POST', headers:{{'Content-Type':'application/json'}},
        keepalive: true,
        body: JSON.stringify({{name: NAME[mid], token: el ? el.value : ''}})}});
    }}catch(e){{}}
  }}
  bt2.addEventListener('click', () => timer ? stop('멈춤') : start());
  // 영자세로. 슬라이더를 그 관절의 영자세에 놓고 켠다 -- 켜면 가드가 어차피 먼저
  // 영자세로 데려가므로, 이 단추는 **그 뒤에 어디로도 가지 않게** 하는 것이다.
  function toZero(){{
    sl.value = HOME[mid]; show();
    if (!timer) start();
  }}
  zb.addEventListener('click', toZero);
  window.__HOOKS[mid] = {{toZero, stop, start, free}};
  // 화면이 **실제로 닫힐 때만** 멈춘다. 다른 창을 누르거나(blur) 잠깐 가려지는 것은
  // 손을 뗀 것이 아니다 -- 그것들로 끄던 것이 "지멋대로 꺼진다" 의 정체였다.
  window.addEventListener('pagehide', () => stop('화면이 닫힘'));
}}
tick(); setInterval(tick, 200);
</script>"""

  return H


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--port", type=int, default=8098)
  ap.add_argument("--bind", default="0.0.0.0")
  ap.add_argument("--hz", type=float, default=20.0)
  ap.add_argument("--cam", default="http://10.8.0.17:8099/stream.mjpg")
  ap.add_argument("--cam-status", default=None,
                  help="카메라 상태 주소. 안 주면 --cam 에서 만들어 씀")
  ap.add_argument("--viser", default="http://10.8.0.16:8094/", help="3D 화면 주소")
  # ---- 움직이는 명령 (기본은 꺼져 있음) --------------------------------
  ap.add_argument("--command", action="store_true",
                  help="화면 슬라이더로 관절을 움직일 수 있게 함. 이것 없이는 읽기만 함")
  ap.add_argument("--allow", nargs="+", default=["L_hip_yaw_joint", "R_hip_yaw_joint"],
                  help="움직여도 되는 시뮬 관절 이름. 여기 없는 것은 무시함")
  ap.add_argument("--arm-token", default="", help="이 암호가 맞는 꾸러미만 받음")
  ap.add_argument("--cmd-port", type=int, default=9872)
  ap.add_argument("--kp", type=float, default=10.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--span-deg", type=float, default=10.0,
                  help="힘을 넣은 자리에서 좌우로 이만큼까지만 움직임")
  ap.add_argument("--max-vel-deg-s", type=float, default=20.0)
  ap.add_argument("--deadman-s", type=float, default=0.3)
  ap.add_argument("--max-temp-c", type=float, default=50.0)
  ap.add_argument("--kp-map", default="",
                  help="관절마다 다른 세기. 예: L_knee_joint=12,R_knee_joint=12")
  ap.add_argument("--span-map", default="",
                  help="관절마다 다른 구간(도). 예: L_knee_joint=15,R_knee_joint=15")
  ap.add_argument("--limit-map", default="",
                  help="모터 각도 기준의 고정 한계. 예: L_knee_joint=0:120 "
                       "(한쪽 끝이 하드스톱인 관절은 이것 없이는 못 엶)")
  ap.add_argument("--live", action="store_true",
                  help="**정말로 힘을 넣음.** 없으면 계산만 하고 힘은 안 넣음(시험)")
  ap.add_argument("--telemetry", default=None,
                  help="뷰어로도 같은 값을 보냄. 예: 192.168.20.177:9871")
  a = ap.parse_args(argv)

  if not os.path.exists(MAP_PATH):
    raise SystemExit(f"{MAP_PATH} 가 없습니다. 먼저 can_up.py 를 돌리세요")
  buses = [b for b in json.load(open(MAP_PATH))["buses"] if b.get("motors")]
  conv = json.load(open(CONV_PATH)) if os.path.exists(CONV_PATH) else {}
  if not conv:
    print("시뮬 규약 파일이 없습니다 — 범위 칸이 비어 보입니다", flush=True)

  for b in buses:
    print(f"  {b['iface']}  {b['leg']}  모터 {b['motors']}")
  guard = None
  if a.command:
    from cmd_guard import CommandGuard
    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ankle_pair_mask.json")
    pair_mask = json.load(open(mp)) if os.path.exists(mp) else None
    if pair_mask is None and any("crank" in n for n in a.allow):
      raise SystemExit("발목을 열려면 ankle_pair_mask.json 이 있어야 합니다 "
                       "(크랭크 두 개를 한 쌍으로 자르는 지도)")
    if not a.arm_token:
      print("  ** 암호 없이 돕니다 — 이 화면에 닿는 사람은 누구나 움직일 수 있습니다. **")
      print("     바깥에 열어 둔 주소(터널 등)가 있으면 먼저 닫을 것.", flush=True)
    guard = CommandGuard(
      allow=a.allow, arm_token=a.arm_token, port=a.cmd_port,
      kp=a.kp, kd=a.kd, span_deg=a.span_deg, max_vel_deg_s=a.max_vel_deg_s,
      deadman_s=a.deadman_s, max_temp_c=a.max_temp_c, dry_run=not a.live,
      pair_mask=pair_mask,
      kp_by=_kv(a.kp_map), span_by=_kv(a.span_map),
      limit_by=_kv2(a.limit_map), needs_limit=NEEDS_LIMIT).start()
    st = guard.status()
    print(f"  ** 움직이는 명령 {'켜짐 — 실제로 힘이 들어갑니다' if a.live else '시험 모드 (힘 안 넣음)'} **")
    print(f"     허락한 관절 {a.allow} · 세기 {st['kp']:g}/{st['kd']:g} · "
          f"구간 ±{st['span_deg']:g}도 · 속도 {st['max_vel_deg_s']:g}도/초")
    print(f"     손 떼면 {st['deadman_s']:g}초 안에 멈춤 · {st['max_temp_c']:g}도에서 끊음 · "
          f"받는 포트 {st['port']}", flush=True)

  poller = Poller(buses, hz=a.hz, guard=guard).start()
  sender = None
  if a.telemetry:
    sender = TelemetrySender(poller, a.telemetry, guard=guard).start()
    print(f"  뷰어로도 보냅니다 -> {a.telemetry} (보내기만 함)", flush=True)

  cam_status = a.cam_status or a.cam.rsplit("/", 1)[0] + "/status.json"
  cam_watch = CamWatch(cam_status).start()
  print(f"  카메라 상태를 물어봅니다 -> {cam_status}", flush=True)

  srv = ThreadingHTTPServer(
    (a.bind, a.port),
    make_handler(poller, build_rows(conv), a.cam, cam_watch, a.viser, guard))
  srv.daemon_threads = True
  print(f"관절 화면 http://{a.bind}:{a.port}/   (Ctrl-C 로 멈춤)", flush=True)
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    # ★ 무슨 일로 끝나든 힘부터 끊는다. 끄기 프레임이 실제로 나가도록 잠깐 기다린다.
    if guard is not None:
      for mid in guard.allow:
        guard.release(mid)
      time.sleep(0.3)
      guard.stop()
      print("  힘을 끊었습니다", flush=True)
    if sender is not None:
      sender.stop()
    cam_watch.stop()
    poller.stop()
    srv.server_close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
