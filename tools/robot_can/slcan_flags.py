#!/usr/bin/env python3
"""CANable 어댑터에게 **직접** 물어본다 — 펌웨어 판과 버스 오류 깃발.

2026-09-21 · 열두 모터가 MIT 에도 private 에도 전 대역 무응답이다. 커널이 보는 slcan 은
오류를 세지 않으므로(`ip -s link` 의 오류 계수가 늘 0이다) **어댑터에게 직접 물어야** 한다.

slcan 명령 몇 개만 쓴다.

    V\r   펌웨어 판
    C\r   버스 닫기
    S8\r  초당 100만 비트
    O\r   버스 열기
    F\r   상태 깃발 (한 바이트, 16진 두 자리)

깃발 비트 뜻 (slcan 표준):

    bit0 받기칸 참   bit1 보내기칸 참   bit2 오류 경고   bit3 넘침
    bit5 오류 수동(error passive)   bit6 중재 짐   bit7 버스 오류

**버스에 아무도 없으면** 보낸 프레임이 확인(ACK)을 못 받아 bit5·bit7 이 선다. 그게
"선은 멀쩡한데 상대가 없다" 의 증거다.

프레임은 **고장값 조회 하나만** 보낸다 (`ff ff ff ff ff ff 00 fb`) — 축을 못 돌린다.
"""

from __future__ import annotations

import argparse
import time


def talk(f, cmd: str, wait: float = 0.25) -> str:
  f.write(cmd.encode()); f.flush()
  time.sleep(wait)
  try:
    raw = f.read(256)
  except Exception:
    raw = b""
  if raw is None:
    raw = b""
  # \r 은 "알았다", \x07(BEL) 은 "그 명령 못 한다" 이다. 잘라 내면 둘이 똑같이 보인다.
  note = {b"\r": "  <- 알았다(CR)", b"\x07": "  <- 거절(BEL)", b"": "  <- 아무 답 없음"}
  return repr(raw) + note.get(raw, "")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dev", required=True, help="예: /dev/ttyACM2")
  ap.add_argument("--id", type=int, default=1, help="고장조회를 보낼 모터 번호")
  a = ap.parse_args(argv)

  import os
  import termios
  import tty
  fd = os.open(a.dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
  # ★ raw 로 맞추지 않으면 줄 단위 처리가 \r 을 건드려 답이 안 보인다.
  # 실제로 2026-09-21 에 이것 때문에 "어댑터가 답을 안 한다" 로 잘못 읽을 뻔했다.
  try:
    tty.setraw(fd)
    at = termios.tcgetattr(fd)
    at[4] = at[5] = termios.B115200          # ispeed / ospeed (CDC-ACM 은 무시하지만 맞춰 둔다)
    termios.tcsetattr(fd, termios.TCSANOW, at)
    termios.tcflush(fd, termios.TCIOFLUSH)
  except Exception as exc:
    print(f"  (raw 설정 실패: {exc})")
  f = os.fdopen(fd, "r+b", buffering=0)

  CR = "\r"
  print(f"{a.dev}")
  for label, cmd in (("펌웨어 V", "V"), ("닫기 C", "C"), ("속도 S8", "S8"),
                     ("열기 O", "O"), ("깃발 F (보내기 전)", "F")):
    print(f"  {label:<18} {talk(f, cmd + CR)}")
  # 고장값 조회: 확장 번호 = 모터번호, 내용 ff ff ff ff ff ff 00 fb
  frame = "T%08X8FFFFFFFFFFFF00FB%s" % (a.id, CR)
  print(f"  보냄            {frame.strip()}")
  print(f"                  {talk(f, frame)}")
  time.sleep(0.3)
  print(f"  깃발 F (보낸 뒤)   {talk(f, 'F' + CR)}")
  print(f"  닫기 C             {talk(f, 'C' + CR)}")
  print("""
  깃발 읽는 법: 'F' 뒤 두 자리 16진수. 00 이면 깨끗.
    20(bit5) 오류 수동 · 80(bit7) 버스 오류 → 보냈는데 아무도 확인 안 해 줌 = 상대가 없음
    04(bit2) 오류 경고 → 오류가 쌓이는 중""")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
