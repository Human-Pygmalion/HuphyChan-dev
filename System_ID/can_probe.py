#!/usr/bin/env python3
"""버스에 어떤 모터가 붙어 있는지 **축을 못 움직이는 명령만으로** 알아본다.

2026-09-11 · 실물 로봇(HUPHY 오른다리/양다리)이 크레인에 매달린 채 붙어 있습니다.
사용자 지시는 **어떠한 움직임 명령도 넣지 말 것**입니다.

무엇을 보내는가 — 딱 한 종류뿐입니다
------------------------------------
**고장값 조회** (RobStride 명령 5의 조회 형태):

    CAN 번호 = 모터 번호,  내용 = FF FF FF FF FF FF 00 FB

이것은 모터의 **기록을 읽어 달라는 질의**입니다. 제어 경로(목표각·세기·토크)를 전혀
지나지 않으므로, 모터에 힘이 들어가 있든 꺼져 있든 **축이 움직일 수 없습니다.**

일부러 안 쓰는 것들 — 이 파일에는 만들 방법 자체가 없습니다
-----------------------------------------------------------
    0xFC 켜기        힘이 들어감
    0xFD 끄기        힘이 빠짐 (붙잡고 있던 관절이 풀림)
    0xFE 영점        제어 기준이 옮겨가 수백 도 이동 지시가 될 수 있음 (2026-09-11 확인)
    MIT 동작 프레임   세기·토크가 0이어도 형식상 '움직여라' 이고,
                     붙잡고 있던 모터를 풀어 버림

돌아오는 것
-----------
답한 모터는 8바이트를 돌려줍니다. 첫 칸이 모터 번호, 그다음 네 칸이 고장값이며
**작은 자리가 먼저** 옵니다. 0이면 고장 없음입니다.

    can_probe.py                      # 1~12번을 훑음
    can_probe.py --ids 1 2 3          # 지정한 번호만
    can_probe.py --listen 5           # 아무것도 안 보내고 5초 듣기만
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

CAN_FRAME = "=IB3x8s"                 # struct can_frame (16 바이트)
CMD_FAULT_QUERY = 0xFB                # 명령 5
F_CMD_QUERY = 0x00                    # 0xFF 가 아니면 '조회'

# 켜기·끄기·영점·동작 프레임은 이 파일에 **의도적으로 없습니다.**
# 실수로도 만들어지지 않게, 내용 바이트를 만드는 함수가 이것 하나뿐입니다.
FAULT_QUERY_DATA = bytes([0xFF] * 6 + [F_CMD_QUERY, CMD_FAULT_QUERY])

FAULT_BITS = {
  0: "과열", 1: "전류 이상", 2: "전압 낮음", 3: "전압 높음",
  4: "B상 전류", 5: "과속", 6: "엔코더 미보정", 7: "A상 전류",
  14: "과부하", 15: "자석 이상", 16: "권선 과열", 17: "드라이버 이상",
}


def open_bus(channel: str) -> socket.socket:
  s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  s.bind((channel,))
  return s


def drain(sock: socket.socket, seconds: float):
  """그 시간 동안 들어온 것을 모은다. 아무것도 보내지 않는다."""
  out = []
  end = time.monotonic() + seconds
  while True:
    left = end - time.monotonic()
    if left <= 0:
      break
    sock.settimeout(left)
    try:
      raw = sock.recv(16)
    except (TimeoutError, socket.timeout):
      break
    can_id, dlc, data = struct.unpack(CAN_FRAME, raw)
    out.append((can_id & 0x7FF, data[:dlc]))
  return out


def describe_fault(word: int) -> str:
  if word == 0:
    return "고장 없음"
  on = [name for bit, name in FAULT_BITS.items() if word & (1 << bit)]
  return f"0x{word:08X}  " + (", ".join(on) if on else "(뜻을 모르는 비트)")


def flush(sock: socket.socket) -> None:
  """큐에 남은 묵은 프레임을 버린다. 고장 응답은 보통 상태 프레임과 CAN 번호가
  같아서, 묵은 것이 남아 있으면 그걸 고장값으로 잘못 읽는다."""
  sock.settimeout(0.0)
  try:
    while True:
      sock.recv(16)
  except (BlockingIOError, TimeoutError, socket.timeout):
    pass


def probe(sock: socket.socket, motor_id: int, wait_s: float):
  flush(sock)
  sock.send(struct.pack(CAN_FRAME, motor_id, 8, FAULT_QUERY_DATA))
  for can_id, data in drain(sock, wait_s):
    if len(data) >= 5 and data[0] == motor_id:
      word = data[1] | (data[2] << 8) | (data[3] << 16) | (data[4] << 24)
      return can_id, word, data
  return None


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", default="can0")
  ap.add_argument("--ids", type=int, nargs="*", default=list(range(1, 13)))
  ap.add_argument("--wait-ms", type=float, default=60.0, help="한 번호당 답을 기다리는 시간")
  ap.add_argument("--listen", type=float, help="아무것도 안 보내고 이 초만큼 듣기만 함")
  a = ap.parse_args(argv)

  sock = open_bus(a.channel)

  if a.listen is not None:
    print(f"{a.channel} 를 {a.listen:g}초 동안 듣기만 합니다 (보내는 것 없음)")
    got = drain(sock, a.listen)
    if not got:
      print("  아무것도 안 들어왔습니다 — 지금 버스에서 말하는 사람이 없습니다")
    for can_id, data in got[:40]:
      print(f"  0x{can_id:03X}  {data.hex(' ')}")
    print(f"  총 {len(got)}개")
    return 0

  print(f"{a.channel} · 보내는 것은 **고장값 조회뿐**입니다 "
        f"({FAULT_QUERY_DATA.hex(' ')}) — 축이 움직일 수 없습니다\n")
  print(f"  {'번호':>4}  {'응답':<6} {'돌아온 CAN 번호':<16} 고장 상태")
  print("  " + "-" * 62)
  found = []
  for motor_id in a.ids:
    got = probe(sock, motor_id, a.wait_ms / 1000.0)
    if got is None:
      print(f"  {motor_id:>4}  {'없음':<6}")
      continue
    can_id, word, data = got
    found.append((motor_id, word))
    print(f"  {motor_id:>4}  {'있음':<6} 0x{can_id:03X}{'':<12} {describe_fault(word)}")
  print("  " + "-" * 62)
  print(f"  버스에 있는 모터 {len(found)}개: {[m for m, _ in found]}")
  bad = [m for m, w in found if w]
  if bad:
    print(f"  ** 고장 기록이 있는 모터: {bad} **")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
