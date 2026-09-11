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

    joint_web.py --port 8098
    http://<파이 주소>:8098/
"""

from __future__ import annotations

import argparse
import json
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
  """열두 모터를 계속 읽어 최신 값을 들고 있는다. 버스마다 스레드 하나."""

  def __init__(self, buses, hz=20.0):
    self.buses = buses                      # [{"iface":..,"leg":..,"motors":[..]}]
    self.period = 1.0 / hz
    self.lock = threading.Lock()
    self.state = {}                         # 모터번호 -> 값
    self.bus_stat = {b["iface"]: dict(ok=0, miss=0, leg=b["leg"]) for b in buses}
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
        try:
          sock.send(struct.pack(CAN_FRAME, mid, 8, ZERO_FRAME))
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
 button{background:#2a2e37;color:var(--tx);border:1px solid #3a3f4a;border-radius:6px;
        padding:5px 10px;cursor:pointer;font:inherit}
 button:hover{background:#333844}
 #banner{display:none;padding:9px 14px;background:var(--bad);font-weight:600}
 #banner.on{display:block}
</style>
<header>
  <h1>실물 관절 확인 &mdash; 읽기 전용</h1>
  <span class="dim">이 화면은 모터를 움직일 수 없습니다 (세기 0 &middot; 토크 0 인 프레임만 보냄)</span>
  <span class="dim" id="rate"></span>
  <button onclick="zeroAll()">지금 자세를 0으로</button>
  <button onclick="clearZero()">기준 지우기</button>
</header>
<div id="banner"></div>
"""


def make_handler(poller, rows, cam_url):
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
      elif p == "/joints.json":
        state, stat = poller.snapshot()
        self._send(200, "application/json",
                   json.dumps(dict(motors=state, buses=stat,
                                   t=round(time.time(), 2))).encode())
      else:
        self._send(404, "text/plain; charset=utf-8", "없는 주소".encode())

    def _page(self):
      return PAGE_HEAD + f"""
<div class="wrap">
  <div>
    <div class="card"><h2>벤치 화면</h2><img src="{cam_url}" alt="벤치"></div>
    <div class="card" style="margin-top:12px"><h2>통신 상태</h2>
      <table id="bus"></table></div>
  </div>
  <div class="card"><h2>관절 &mdash; 손으로 움직여 보세요</h2>
    <table id="j"><thead><tr>
      <th class="l">관절</th><th>지금(도)</th><th>0 기준 변화</th>
      <th class="l">시뮬에서의 범위</th><th class="l">상태</th>
    </tr></thead><tbody></tbody></table></div>
</div>
<script>
const ROWS = {body_rows};
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
      if (m.mode===2) hot.push(`${{r.ko}} \\uD798\\uC774 \\uCF1C\\uC838 \\uC788\\uC74C`);
    }}
    tb.insertAdjacentHTML('beforeend',
      `<tr><td class="l">${{r.leg==='left'?'왼':'오른'}} ${{r.ko}}`
      + ` <span class="dim">${{r.id}}번 · ${{r.sim}}</span></td>`
      + `<td>${{cur}}</td><td>${{dlt}}</td><td class="l">${{rng}}</td><td class="l">${{st}}</td></tr>`);
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
  a = ap.parse_args(argv)

  if not os.path.exists(MAP_PATH):
    raise SystemExit(f"{MAP_PATH} 가 없습니다. 먼저 can_up.py 를 돌리세요")
  buses = [b for b in json.load(open(MAP_PATH))["buses"] if b.get("motors")]
  conv = json.load(open(CONV_PATH)) if os.path.exists(CONV_PATH) else {}
  if not conv:
    print("시뮬 규약 파일이 없습니다 — 범위 칸이 비어 보입니다", flush=True)

  for b in buses:
    print(f"  {b['iface']}  {b['leg']}  모터 {b['motors']}")
  poller = Poller(buses, hz=a.hz).start()

  srv = ThreadingHTTPServer((a.bind, a.port), make_handler(poller, build_rows(conv), a.cam))
  srv.daemon_threads = True
  print(f"관절 화면 http://{a.bind}:{a.port}/   (Ctrl-C 로 멈춤)", flush=True)
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    poller.stop()
    srv.server_close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
