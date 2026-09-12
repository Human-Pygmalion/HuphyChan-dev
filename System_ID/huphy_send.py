#!/usr/bin/env python3
"""제어 루프(`huphy_live`)에 목표를 **천천히** 보내고, 따라오는지 찍는다.

2026-09-12. 화면 대신 쓰는 손 도구입니다. 화면이 하는 일과 똑같습니다 — 한 줄짜리
JSON 을 UDP 로 보내는 것뿐이고, 한계·속도·프레임은 전부 로봇 프로그램이 합니다.

**왜 천천히 보내는가.** 목표를 한 번에 멀리 주면 그 자체는 가드가 자르지만, 자른
뒤에도 관절은 **한 주기 이동량 상한만큼 계속** 밀립니다. 세기 10에서 10도 오차는
약 2뉴턴미터라, 매달린 관절이 홱 돕니다. 사람이 보면서 멈출 수 있으려면 목표가
사람 눈으로 따라갈 만큼 천천히 가야 합니다.

    huphy_send.py --to 127.0.0.1:9872 --ramp hip_yaw=0 --seconds 6
    huphy_send.py --to 127.0.0.1:9872 --ramp ankle_a=10 ankle_b=-10 --seconds 6
    huphy_send.py --to 127.0.0.1:9872 --release hip_yaw

`--from-state` 로 상태 포트를 주면 지금 각도에서 출발합니다. 안 주면 `--start` 로
사람이 적어 줘야 합니다 — 어디서 출발하는지 모르는 채로 보내지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Dict, Optional


def parse_pairs(items) -> Dict[str, float]:
  out = {}
  for one in items or []:
    name, _, value = str(one).partition("=")
    if not value:
      raise SystemExit(f"'{one}' 은 이름=값 모양이어야 합니다 (예: hip_yaw=0)")
    out[name.strip()] = float(value)
  return out


def read_state(port: int, wait_s: float = 3.0) -> Optional[Dict[str, dict]]:
  """제어 루프가 뿌리는 상태를 한 장 받는다. 못 받으면 `None`."""
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  try:
    sock.bind(("0.0.0.0", int(port)))
  except OSError:
    return None
  sock.settimeout(wait_s)
  try:
    raw, _ = sock.recvfrom(65535)
    return json.loads(raw).get("motors") or {}
  except (TimeoutError, socket.timeout, ValueError, TypeError):
    return None
  finally:
    sock.close()


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--to", default="127.0.0.1:9872")
  ap.add_argument("--ramp", nargs="*", default=[], metavar="이름=도",
                  help="여기까지 천천히 간다")
  ap.add_argument("--start", nargs="*", default=[], metavar="이름=도",
                  help="여기서 출발한다. --from-state 가 있으면 안 줘도 된다")
  ap.add_argument("--from-state", type=int, default=None, metavar="포트",
                  help="제어 루프가 뿌리는 상태를 받아 지금 각도에서 출발")
  ap.add_argument("--release", nargs="*", default=[], metavar="이름",
                  help="그 관절 목표를 지금 자리로 묶는다 (힘은 그대로)")
  ap.add_argument("--seconds", type=float, default=6.0)
  ap.add_argument("--hz", type=float, default=20.0)
  ap.add_argument("--token", default="")
  a = ap.parse_args(argv)

  host, _, port = a.to.partition(":")
  addr = (host, int(port or 9872))
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

  if a.release:
    for name in a.release:
      sock.sendto(json.dumps({"release": name}).encode(), addr)
      print(f"  {name} 를 지금 자리에 묶으라고 보냈습니다")
    return 0

  goal = parse_pairs(a.ramp)
  if not goal:
    raise SystemExit("--ramp 로 갈 곳을 적어 주세요 (예: hip_yaw=0)")

  start = parse_pairs(a.start)
  if a.from_state is not None:
    rows = read_state(a.from_state)
    if rows is None:
      raise SystemExit(f"상태를 못 받았습니다 (포트 {a.from_state}). "
                       f"제어 루프가 --state-to 로 이쪽에 보내고 있나요?")
    by_joint = {r["joint"].split("/")[-1]: r for r in rows.values()}
    for name in goal:
      row = by_joint.get(name)
      if row is None:
        raise SystemExit(f"상태에 {name} 이 없습니다. 있는 것: {sorted(by_joint)}")
      start.setdefault(name, float(row["deg"]))
  missing = [n for n in goal if n not in start]
  if missing:
    raise SystemExit(f"{missing} 의 출발 각도를 모릅니다. --from-state 나 --start 로 "
                     f"알려 주세요 — 어디서 출발하는지 모른 채로는 안 보냅니다.")

  print(f"\n  {addr[0]}:{addr[1]} 로 {a.seconds:g}초에 걸쳐 보냅니다 "
        f"(초당 {a.hz:g}번)")
  for name in sorted(goal):
    print(f"    {name:12} {start[name]:+8.2f} -> {goal[name]:+8.2f}도  "
          f"({goal[name] - start[name]:+.2f}도, 평균 초당 "
          f"{abs(goal[name] - start[name]) / max(a.seconds, 1e-9):.1f}도)")
  print()

  period = 1.0 / a.hz
  t0 = time.monotonic()
  sent = 0
  while True:
    t = time.monotonic() - t0
    frac = min(1.0, t / a.seconds) if a.seconds > 0 else 1.0
    # 부드럽게: 시작과 끝에서 속도가 0 인 곡선. 계단이 안 생긴다.
    ease = frac * frac * (3.0 - 2.0 * frac)
    targets = {n: start[n] + (goal[n] - start[n]) * ease for n in goal}
    payload = {"targets": targets}
    if a.token:
      payload["token"] = a.token
    sock.sendto(json.dumps(payload).encode(), addr)
    sent += 1
    if frac >= 1.0:
      break
    time.sleep(period)

  # 도착한 목표를 몇 번 더 되풀이해 둔다 -- UDP 라 한 장이 흘려질 수 있다.
  for _ in range(5):
    sock.sendto(json.dumps({"targets": dict(goal)}).encode(), addr)
    time.sleep(period)
  print(f"  {sent + 5}번 보냈습니다. 마지막 목표 "
        + ", ".join(f"{n} {v:+.2f}도" for n, v in sorted(goal.items())))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
