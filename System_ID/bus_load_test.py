#!/usr/bin/env python3
"""모터 6대를 한 버스에 물리고 초당 200번 제어할 수 있는가. 라즈베리파이에서 돌린다.

2026-09-09. 사용자 정정: **"모터 총 6개가 하나의 버스에 연결되는데, 최소한 200Hz 는
안정적으로, 패킷로스나 응답지연 없이 나와야 해."**

앞서 제가 "여유 5.5배"라고 한 것은 **평균 처리량**으로 따진 것이라 틀렸습니다. 제어는
평균이 아니라 **마감**입니다 — 초당 200번이면 주기가 5 밀리초이고, 그 5 밀리초 안에
**6대 전부**에게 명령을 주고 6대 전부의 응답을 받아야 합니다. 한 주기라도 늦으면 그
주기는 실패입니다.

어떻게 6대를 흉내내는가
-----------------------
벤치에는 모터가 2대(번호 3, 4)뿐입니다. 그래서 **선 위의 짐**을 실제와 같게 만듭니다.

* 명령 6개를 보냅니다 (번호 1~6). 3번과 4번만 진짜로 답합니다.
* 없는 4대의 **응답이 차지했을 자리**를 호스트가 채움 프레임 4개로 메웁니다.
* 그러면 선 위에는 한 주기에 프레임 12개가 흐릅니다 — 진짜 6대일 때와 같은 수입니다.

★ 다르게 남는 점: 없는 4대의 응답은 모터가 아니라 호스트가 냅니다. 그래서 **버스 중재**
(누가 먼저 말하는가)의 양상은 완전히 같지 않습니다. 프레임 수와 시간 점유는 같습니다.

무엇을 재는가
-------------
매 주기마다 **고정된 시각 격자**(진짜 제어기처럼)에 맞춰 시작하고,

* 그 주기의 응답이 **다 왔는가** (패킷 손실)
* 응답이 다 오기까지 **얼마나 걸렸는가** (여유 시간)
* 격자에서 **얼마나 밀렸는가** (지터)

를 기록하고, **최악값**까지 봅니다. 평균이 아니라 최악이 제어를 망가뜨립니다.

    bus_load_test.py --rates 200,300,400,500 --motors 6 --seconds 20
    sudo chrt -f 80 python3 bus_load_test.py ...   # 실시간 우선순위로 올려서 다시
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rscodec import mit, tables  # noqa: E402

FMT = "=IB3x8s"
FILLER_ID = 0x7A0
"""아무 모터도 안 쓰는 번호. 선을 채우기만 하고 아무도 답하지 않는다."""


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", default="can0")
  ap.add_argument("--real-ids", default="3,4", help="진짜로 답하는 모터")
  ap.add_argument("--motors", type=int, default=6, help="흉내낼 총 대수")
  ap.add_argument("--rates", default="200,300,400,500")
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--model", default="RS04")
  a = ap.parse_args(argv)

  real = [int(x) for x in a.real_ids.split(",")]
  ids = sorted(set(real) | set(range(1, a.motors + 1)))[:a.motors]
  missing = [i for i in real if i not in ids]
  assert not missing, (f"진짜 모터 {missing} 가 명령 목록에 없다 — --motors 를 늘리거나 "
                       f"--real-ids 를 줄여라 (안 그러면 오지 않을 응답을 기다린다)")
  enc = tables.MIT_ENCODING[tables.Model(a.model)]
  data = mit.pack_command(position_deg=0.0, velocity_deg_s=0.0, kp=0.0, kd=0.0,
                          torque_nm=0.0, enc=enc)
  cmd = {i: struct.pack(FMT, i, 8, data) for i in ids}
  filler = struct.pack(FMT, FILLER_ID, 8, data)
  n_filler = a.motors - len(real)          # 없는 모터의 응답 자리를 메운다

  s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  s.bind((a.channel,))

  print(f"모터 {a.motors}대 흉내 (진짜 {real}, 채움 {n_filler}개) · "
        f"한 주기에 프레임 {a.motors * 2}개\n")
  print(f"{'Hz':>5} {'주기':>8} {'완주율':>8} {'중앙':>9} {'95%':>8} {'99%':>8} "
        f"{'99.9%':>8} {'최악':>8}   1ms/2ms/3ms 넘은 주기 수")

  for rate in [float(x) for x in a.rates.split(",")]:
    T = 1.0 / rate
    n_cycles = int(a.seconds * rate)
    used, jit, ok = [], [], 0
    t_start = time.monotonic()
    for k in range(n_cycles):
      deadline = t_start + (k + 1) * T
      due = t_start + k * T
      # 격자까지 기다린다 (바쁜 대기 - 잠들면 스케줄러가 늦게 깨운다)
      while time.monotonic() < due:
        pass
      c0 = time.monotonic()
      jit.append((c0 - due) * 1e3)
      s.setblocking(False)
      for i in ids:
        try:
          s.send(cmd[i])
        except OSError:
          pass
      for _ in range(n_filler):
        try:
          s.send(filler)
        except OSError:
          pass
      got = 0
      while got < len(real) and time.monotonic() < deadline:
        try:
          raw = s.recv(16)
        except OSError:
          continue
        cid, dlc, d = struct.unpack(FMT, raw)
        if dlc >= 8:
          try:
            mid, *_ = mit.decode_state(d[:dlc], enc=enc)
          except (ValueError, KeyError):
            continue
          if mid in real:
            got += 1
      used.append((time.monotonic() - c0) * 1e3)
      if got == len(real):
        ok += 1
      # 남은 응답이 다음 주기로 새지 않게 비운다
      while True:
        try:
          s.recv(16)
        except OSError:
          break
    used.sort(); jit.sort()
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    over = [sum(1 for u in used if u > x) for x in (1.0, 2.0, 3.0)]
    print(f"{rate:5.0f} {T*1e3:6.2f}ms {ok/n_cycles*100:7.2f}% {q(used,0.5):7.3f}ms "
          f"{q(used,0.95):6.3f}ms {q(used,0.99):6.3f}ms {q(used,0.999):6.3f}ms "
          f"{used[-1]:6.3f}ms   {over[0]:5d} / {over[1]:5d} / {over[2]:5d}  "
          f"(총 {n_cycles})")
  s.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
