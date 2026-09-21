#!/usr/bin/env python3
"""모터가 **private(공장 기본) 프로토콜**로 말하고 있는지 읽기만 해서 알아본다.

2026-09-21 · 열두 개가 MIT 프레임(고장조회·켜기·상태)에 전부 무응답인데 전원·배선·종단은
정상이다. `docs/123` 09-04 에 같은 일이 있었고 원인은 **모터가 private 프로토콜 상태**였다.
HUPHY 는 MIT 만 할 줄 알아서(`codec/` 에 `mit.py` 하나뿐) 그때는 이 상태를 못 봤다.

보내는 것은 **Type 0 핑 하나뿐이다** — `commission_private_to_mit.py` 가 read-only 로 분류한
바로 그 프레임이고, 내용이 전부 0 이라 축을 돌릴 방법이 없다.

    확장 CAN 번호 = (종류<<24) | (00<<16) | (호스트 0xFD<<8) | 모터번호
    Type 0 핑      0000FD<번호>#0000000000000000

`python-can` 없이 리눅스 기본 소켓만 쓴다(라즈베리파이에 아무것도 안 깔려 있어도 돌게).
`slcand` 가 올려 둔 `can0`·`can1` 에 그대로 붙는다.

    private_ping.py --channel can0 --ids 1 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import socket
import struct
import time

CAN_FRAME = "=IB3x8s"
CAN_EFF_FLAG = 0x80000000
HOST = 0xFD


def ping(sock, mid: int, wait: float) -> list[tuple[int, bytes]]:
  arb = (0 << 24) | (0x00 << 16) | (HOST << 8) | mid
  sock.send(struct.pack(CAN_FRAME, arb | CAN_EFF_FLAG, 8, bytes(8)))
  got, t0 = [], time.time()
  while time.time() - t0 < wait:
    sock.settimeout(max(0.0, wait - (time.time() - t0)))
    try:
      raw = sock.recv(16)
    except (socket.timeout, TimeoutError):
      break
    cid, dlc, data = struct.unpack(CAN_FRAME, raw)
    got.append((cid & 0x1FFFFFFF, data[:dlc]))
  return got


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", required=True)
  ap.add_argument("--ids", type=int, nargs="+", default=list(range(1, 128)))
  ap.add_argument("--wait-ms", type=float, default=120.0)
  a = ap.parse_args(argv)

  sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  sock.bind((a.channel,))
  print(f"{a.channel} · private Type 0 핑만 보냅니다 (내용 전부 0) — 축이 움직일 수 없습니다\n")
  print(f"  {'번호':>5} {'응답':<8} {'돌아온 번호':<14} 내용")
  print("  " + "-" * 62)
  alive = []
  for mid in a.ids:
    got = ping(sock, mid, a.wait_ms / 1000.0)
    if got:
      alive.append(mid)
      for cid, data in got:
        print(f"  {mid:5d} {'있음':<8} 0x{cid:08X}   {data.hex(' ')}")
    else:
      print(f"  {mid:5d} {'없음':<8}")
  print("  " + "-" * 62)
  print(f"  private 로 답한 모터 {len(alive)}개: {alive}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
