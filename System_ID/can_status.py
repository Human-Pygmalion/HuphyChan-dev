#!/usr/bin/env python3
"""모터의 **지금 상태**를 읽는다 — 각도·속도·토크·온도·고장표시·힘이 켜졌는지.

2026-09-11 · 실물 로봇이 크레인에 매달린 채 붙어 있는 상태에서 씁니다.

읽으려면 보내야 한다
--------------------
MIT 방식에는 **읽기 전용 프레임이 없습니다.** 모터는 동작 명령을 받았을 때만 상태로
답합니다. 그래서 **세기 0 · 감쇠 0 · 토크 0** 인 프레임을 보내고 그 답을 받습니다.

    7F FF 7F F0 00 00 07 FF     목표각 0 · 목표속도 0 · 세기 0 · 감쇠 0 · 토크 0

토크가 0이므로 이 프레임으로는 축이 돌지 않습니다.

★ 딱 하나 주의할 것
-------------------
어떤 모터에 **이미 힘이 들어가 자세를 붙잡고 있었다면**, 세기 0 인 이 프레임이 그
힘을 0으로 만들어 **관절이 풀립니다.** 축을 돌리는 것이 아니라 붙잡던 것을 놓는
것입니다. 그래서:

  * 사람 손이 치워졌는지 먼저 눈으로 확인하고
  * **한 번에 한 개씩**, 풀려도 영향이 작은 관절(발목)부터 확인합니다.

답에 들어 있는 `동작모드` 가 그것을 알려 줍니다 — `0`이면 힘이 꺼져 있던 것이라
아무 일도 일어나지 않았다는 뜻이고, `2`면 켜져 있었다는 뜻입니다.

    can_status.py --ids 11              # 발목 A 한 개만
    can_status.py --ids 7 8 9 10 11 12  # 여섯 개
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import time

CAN_FRAME = "=IB3x8s"

# 세기·감쇠·토크가 모두 0 인 프레임. 이 파일은 **이것 말고 다른 내용을 만들지
# 않습니다** — 켜기(0xFC)·끄기(0xFD)·영점(0xFE) 은 아예 들어 있지 않습니다.
ZERO_FRAME = bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF])

# 모델별 값 변환 범위 (RobStride 매뉴얼 260713 판, MIT 절).
RANGE = {
  "RS03": dict(pmax=12.57, vmax=20.0, tmax=60.0),
  "RS04": dict(pmax=12.57, vmax=15.0, tmax=120.0),
}
# 양다리 배치 (config/robot_v1.0.yaml 과 같음). 번호가 곧 어느 관절인지 정합니다.
# 모델을 함께 적는 이유: 각도 범위는 넷 다 같지만 **속도와 토크 눈금이 다릅니다**
# (RS03 은 초당 20 라디안·60 Nm, RS04 는 15 라디안·120 Nm). 틀리면 조용히 다른
# 숫자가 나옵니다.
LAYOUT = {
  # 왼다리 — can1
  1: ("왼 엉덩이앞뒤", "RS04"), 2: ("왼 엉덩이좌우", "RS04"), 3: ("왼 엉덩이회전", "RS03"),
  4: ("왼 무릎", "RS04"), 5: ("왼 발목 A", "RS03"), 6: ("왼 발목 B", "RS03"),
  # 오른다리 — can0
  7: ("오른 엉덩이앞뒤", "RS04"), 8: ("오른 엉덩이좌우", "RS04"), 9: ("오른 엉덩이회전", "RS03"),
  10: ("오른 무릎", "RS04"), 11: ("오른 발목 A", "RS03"), 12: ("오른 발목 B", "RS03"),
}
MODE_NAME = {0: "힘 꺼짐", 1: "보정중", 2: "힘 켜짐"}


def uint_to_float(x, lo, hi, bits):
  return float(x) / ((1 << bits) - 1) * (hi - lo) + lo


def decode(data: bytes, rng: dict):
  """상태 프레임 8바이트를 사람이 읽는 값으로. 온도는 12비트이고 위 4비트는 표시등."""
  motor_id = data[0]
  q = (data[1] << 8) | data[2]
  dq = (data[3] << 4) | (data[4] >> 4)
  tau = ((data[4] & 0x0F) << 8) | data[5]
  temp = ((data[6] & 0x0F) << 8) | data[7]
  b6 = data[6]
  return dict(
    motor_id=motor_id,
    deg=math.degrees(uint_to_float(q, -rng["pmax"], rng["pmax"], 16)),
    vel=math.degrees(uint_to_float(dq, -rng["vmax"], rng["vmax"], 12)),
    nm=uint_to_float(tau, -rng["tmax"], rng["tmax"], 12),
    temp_c=temp / 10.0,
    mode=(b6 >> 6) & 0x03,
    fault=bool(b6 & 0x20),
    warning=bool(b6 & 0x10),
  )


def flush(sock):
  sock.settimeout(0.0)
  try:
    while True:
      sock.recv(16)
  except (BlockingIOError, TimeoutError, socket.timeout):
    pass


def ask(sock, motor_id, rng, wait_s=0.05):
  flush(sock)
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
      return decode(data[:8], rng)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", default="can0")
  ap.add_argument("--ids", type=int, nargs="+", required=True,
                  help="읽을 모터 번호. 실수로 전체가 나가지 않게 반드시 적어야 함")
  ap.add_argument("--repeat", type=int, default=1, help="몇 번 읽을지 (떨림 확인용)")
  a = ap.parse_args(argv)

  sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  sock.bind((a.channel,))

  print(f"{a.channel} · 세기 0 · 감쇠 0 · 토크 0 인 프레임만 보냅니다 "
        f"({ZERO_FRAME.hex(' ')})\n")
  head = f"  {'번호':>4} {'관절':<14} {'각도(도)':>9} {'속도(도/초)':>11} " \
         f"{'토크(Nm)':>9} {'온도(도)':>8}  상태"
  print(head)
  print("  " + "-" * (len(head) + 6))
  for _ in range(a.repeat):
    for motor_id in a.ids:
      name, model = LAYOUT.get(motor_id, (f"{motor_id}번", "RS04"))
      st = ask(sock, motor_id, RANGE[model])
      if st is None:
        print(f"  {motor_id:>4} {name:<14} {'응답 없음':>9}")
        continue
      flags = MODE_NAME.get(st["mode"], f"모드{st['mode']}")
      if st["fault"]:
        flags += " · **고장**"
      if st["warning"]:
        flags += " · 경고"
      print(f"  {motor_id:>4} {name:<14} {st['deg']:>9.2f} {st['vel']:>11.2f} "
            f"{st['nm']:>9.2f} {st['temp_c']:>8.1f}  {flags}")
    if a.repeat > 1:
      print()
      time.sleep(0.2)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
