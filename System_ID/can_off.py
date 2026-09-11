#!/usr/bin/env python3
"""모터의 **힘을 뺀다.** 이 파일이 하는 일은 그것 하나뿐이다.

2026-09-12. 화면을 거쳐 힘을 빼려다 열둘 중 다섯이 안 꺼진 일이 있었다. 화면이
죽었거나, 보내는 길이 막혔거나, 프로그램이 먼저 내려가면 끄기 명령이 못 나간다.
**끄는 길은 아무것에도 기대지 않아야 한다.**

보내는 내용은 끄기 프레임 하나뿐이다.

    FF FF FF FF FF FF 00 FD     힘 빼기 (0xFD)

★ 이 파일에는 **켜기(0xFC)도 영점(0xFE)도 없고, 각도나 세기를 담은 프레임을 만드는
  길도 없다.** 즉 이 파일로는 축을 돌릴 수 없다.

힘을 빼면 관절이 **중력으로 처진다.** 매달린 다리는 그냥 늘어지지만, 무언가를 받치고
있었다면 그것이 떨어진다. 사람과 손이 치워졌는지 먼저 눈으로 볼 것.

    can_off.py --channel can0 --ids 1 2 3 4 5 6
    can_off.py --channel can0 --ids 1 2 3 4 5 6 --verify
"""

from __future__ import annotations

import argparse
import socket
import struct
import time

CAN_FRAME = "=IB3x8s"

STOP_FRAME = bytes([0xFF] * 6 + [0xFF, 0xFD])
"""힘 빼기. **이 파일이 만드는 유일한 내용이다.**

★ 일곱 번째 바이트(`F_CMD`)가 **반드시 `0xFF`** 여야 한다. 2026-09-12 에 여기를
`0x00` 으로 적었다가 큰일 날 뻔했다 — 같은 명령 바이트 `0xFD` 가 `F_CMD` 에 따라
뜻이 완전히 달라진다:

    FF ... FF FD    힘 빼기
    FF ... 00 FD    **통신 프로토콜을 0(PRIVATE)으로 바꿈** (전원 재투입 후 적용)

즉 끄려던 명령이 "다음에 전원을 넣으면 다른 말로 이야기해라" 가 된다. 실제로 열두
개 전부에 세 번씩 보냈고, 알아챈 것은 **모터 하나가 안 꺼져서** 였다. 다른 명령도
같은 구조이니 (`0xFC` 는 켜기/제어모드, `0xFB` 는 고장클리어/고장조회) 이 파일
바깥에서도 `F_CMD` 를 반드시 눈으로 확인할 것."""

SET_PROTOCOL_MIT = bytes([0xFF] * 6 + [0x02, 0xFD])
"""통신 프로토콜을 MIT(2) 로 되돌린다. `--restore-protocol` 일 때만 보낸다.

위 실수를 되돌리는 용도다. 지금 쓰는 방식이 MIT 이므로, 이것을 보내는 것은 원래
상태로 못 박는 것이지 새로 바꾸는 것이 아니다."""

ZERO_FRAME = bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF])
"""읽기용. 세기·감쇠·토크가 전부 0 이라 이것으로도 축은 안 돈다.

`--verify` 일 때만 쓴다 -- 정말 꺼졌는지 확인하려면 물어봐야 하는데, 물어보는
방법이 프레임을 보내는 것뿐이기 때문이다 (MIT 방식에는 읽기 전용 명령이 없다).
"""

MODE_NAME = {0: "힘 꺼짐", 1: "보정중", 2: "힘 켜짐"}


def read_mode(sock, motor_id: int, wait_s: float = 0.08):
  """이 모터가 지금 켜져 있는가. 못 읽으면 `None`."""
  sock.settimeout(0.0)
  try:
    while True:
      sock.recv(16)
  except (BlockingIOError, TimeoutError, socket.timeout):
    pass
  sock.send(struct.pack(CAN_FRAME, motor_id, 8, ZERO_FRAME))
  end = time.monotonic() + wait_s
  while True:
    left = end - time.monotonic()
    if left <= 0:
      return None
    sock.settimeout(left)
    try:
      raw = sock.recv(16)
    except (TimeoutError, socket.timeout):
      return None
    _, dlc, data = struct.unpack(CAN_FRAME, raw)
    if dlc >= 8 and data[0] == motor_id:
      return (data[6] >> 6) & 0x03


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", required=True)
  ap.add_argument("--ids", type=int, nargs="+", required=True,
                  help="끌 모터 번호. 실수로 전체가 나가지 않게 반드시 적어야 함")
  ap.add_argument("--repeat", type=int, default=3,
                  help="끄기 프레임을 몇 번 보낼지. 한 장이 흘려질 수 있어 되풀이한다")
  ap.add_argument("--verify", action="store_true", help="끝나고 정말 꺼졌는지 확인")
  ap.add_argument("--restore-protocol", action="store_true",
                  help="통신 방식을 MIT 로 못 박는다 (F_CMD 를 잘못 보낸 것을 되돌릴 때)")
  a = ap.parse_args(argv)

  sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  sock.bind((a.channel,))

  print(f"{a.channel} · 끄기 프레임만 보냅니다 ({STOP_FRAME.hex(' ')})")
  print(f"  관절이 중력으로 처집니다. 사람과 손이 치워졌는지 확인하십시오.\n")

  if a.restore_protocol:
    for motor_id in a.ids:
      sock.send(struct.pack(CAN_FRAME, motor_id, 8, SET_PROTOCOL_MIT))
      time.sleep(0.02)
    print(f"  통신 방식을 MIT 로 못 박았습니다: {a.ids}")
    time.sleep(0.2)

  for _ in range(max(1, a.repeat)):
    for motor_id in a.ids:
      sock.send(struct.pack(CAN_FRAME, motor_id, 8, STOP_FRAME))
      time.sleep(0.01)
    time.sleep(0.05)
  print(f"  {len(a.ids)}개에 끄기를 {a.repeat}번씩 보냈습니다: {a.ids}")

  if not a.verify:
    return 0

  time.sleep(0.3)
  print(f"\n  {'번호':>4} {'상태':<10}")
  bad = []
  for motor_id in a.ids:
    mode = read_mode(sock, motor_id)
    name = "응답 없음" if mode is None else MODE_NAME.get(mode, f"모름({mode})")
    print(f"  {motor_id:>4} {name:<10}")
    if mode != 0:
      bad.append(motor_id)
  if bad:
    print(f"\n  ** 아직 안 꺼진 것: {bad} ** 다시 돌리거나 전원을 보십시오.")
    return 1
  print("\n  전부 꺼졌습니다.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
