#!/usr/bin/env python3
"""모터 **한 개**에 켜기를 보내고, 켜졌는지 눈으로 따라간다.

2026-09-12. "전부 0도로" 를 눌렀는데 열둘 중 다섯만 켜졌다. 안 켜진 일곱은
RS03 여섯 개 전부와 RS04 한 개(8번)였다. 고장 표시는 열둘 다 없었고, 켜기를
154번 다시 보내도 그대로였다.

여러 개를 한꺼번에 돌리면 무엇이 원인인지 안 보인다. **한 개만** 놓고, 보내는
것과 돌아오는 것을 한 줄씩 찍는다.

보내는 것은 셋뿐이다.

    FF FF FF FF FF FF FF FC     켜기
    FF FF FF FF FF FF FF FD     끄기
    세기·감쇠·토크 0 인 MIT 프레임    (상태를 물어보려면 보내야 한다)

★ 각도나 세기가 든 프레임은 **만들지 않는다.** 즉 이 파일로는 축을 돌릴 수 없다.
  다만 켜기를 보내므로, 힘은 들어갈 수 있다 (목표·세기가 0 이라 버티기만 한다).

    enable_probe.py --channel can0 --id 3
    enable_probe.py --channel can0 --id 3 --kp 10     세기를 넣어 보며
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import time

CAN_FRAME = "=IB3x8s"

ENABLE = bytes([0xFF] * 7 + [0xFC])
STOP = bytes([0xFF] * 7 + [0xFD])
ZERO = bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF])

MODE_NAME = {0: "꺼짐", 1: "보정중", 2: "켜짐", 3: "모름3"}
ENC = {"RS03": dict(pmax=12.57, vmax=20.0, tmax=60.0, kp_max=5000.0, kd_max=100.0),
       "RS04": dict(pmax=12.57, vmax=15.0, tmax=120.0, kp_max=5000.0, kd_max=100.0)}


def _f2u(x, lo, hi, bits):
  x = max(lo, min(hi, float(x)))
  span = hi - lo
  return int(((x - lo) / span if span > 0 else 0.0) * ((1 << bits) - 1))


def mit(pos_deg, kp, kd, enc):
  q = _f2u(math.radians(pos_deg), -enc["pmax"], enc["pmax"], 16)
  dq = _f2u(0.0, -enc["vmax"], enc["vmax"], 12)
  kp_u = _f2u(kp, 0.0, enc["kp_max"], 12)
  kd_u = _f2u(kd, 0.0, enc["kd_max"], 12)
  tau = _f2u(0.0, -enc["tmax"], enc["tmax"], 12)
  return bytes([(q >> 8) & 0xFF, q & 0xFF,
                (dq >> 4) & 0xFF, ((dq & 0x0F) << 4) | ((kp_u >> 8) & 0x0F), kp_u & 0xFF,
                (kd_u >> 4) & 0xFF, ((kd_u & 0x0F) << 4) | ((tau >> 8) & 0x0F), tau & 0xFF])


def ask(sock, motor_id, payload, wait_s=0.05):
  """보내고, 그 모터가 답한 것을 돌려준다. `(원시8바이트, 보낸ID)` 또는 `None`."""
  sock.settimeout(0.0)
  try:
    while True:
      sock.recv(16)
  except (BlockingIOError, TimeoutError, socket.timeout):
    pass
  sock.send(struct.pack(CAN_FRAME, motor_id, 8, payload))
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
    can_id, dlc, data = struct.unpack(CAN_FRAME, raw)
    if dlc >= 8 and data[0] == motor_id:
      return data[:8], can_id


def show(tag, got, enc):
  if got is None:
    print(f"  {tag:<22} 응답 없음")
    return None
  data, can_id = got
  b6 = data[6]
  mode = (b6 >> 6) & 0x03
  q = (data[1] << 8) | data[2]
  deg = math.degrees(q / 65535.0 * (2 * enc["pmax"]) - enc["pmax"])
  print(f"  {tag:<22} {data.hex(' ')}  CAN 0x{can_id:03X}  "
        f"{MODE_NAME.get(mode, mode):<5} {deg:+7.2f}도  "
        f"고장{'있음' if b6 & 0x20 else '없음'} 경고{'있음' if b6 & 0x10 else '없음'}")
  return mode


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", required=True)
  ap.add_argument("--id", type=int, required=True)
  ap.add_argument("--model", default="RS03", choices=sorted(ENC))
  ap.add_argument("--kp", type=float, default=0.0)
  ap.add_argument("--kd", type=float, default=0.0)
  ap.add_argument("--cycles", type=int, default=10)
  a = ap.parse_args(argv)
  enc = ENC[a.model]

  sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  sock.bind((a.channel,))
  print(f"\n{a.channel} · 모터 {a.id} ({a.model}) · 세기 {a.kp:g} 감쇠 {a.kd:g}\n")

  show("먼저 상태만 물음", ask(sock, a.id, ZERO), enc)
  show("끄기 FF..FD", ask(sock, a.id, STOP), enc)
  time.sleep(0.1)
  show("상태", ask(sock, a.id, ZERO), enc)

  print()
  show("켜기 FF..FC", ask(sock, a.id, ENABLE), enc)
  time.sleep(0.1)
  modes = []
  for k in range(a.cycles):
    frame = mit(0.0, a.kp, a.kd, enc) if (a.kp or a.kd) else ZERO
    m = show(f"켠 뒤 {k + 1}회째", ask(sock, a.id, frame), enc)
    modes.append(m)
    time.sleep(0.05)

  print()
  show("끄기 FF..FD", ask(sock, a.id, STOP), enc)
  time.sleep(0.1)
  show("마지막 상태", ask(sock, a.id, ZERO), enc)

  on = sum(1 for m in modes if m == 2)
  print(f"\n  켜진 것으로 보고한 주기 {on}/{len(modes)}")
  if on == 0:
    print("  ** 켜기를 받아도 켜지지 않습니다. **")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
