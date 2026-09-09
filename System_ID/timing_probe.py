#!/usr/bin/env python3
"""명령 주기가 지금 얼마이고, 무엇이 그것을 정하는지 가른다. 라즈베리파이에서 돌린다.

2026-09-09. "지금 명령 주기는 얼마인가? 이론적 상한인가?"

이론적 상한이 무엇인지부터
--------------------------
CAN 선의 초당 100만 비트에서, 자료 8바이트짜리 표준 프레임 하나는

    시작 1 + 번호 11 + 3 + 길이 4 + 자료 64 + 검사 15 + 1 + 응답 1 + 1 + 끝 7 + 사이 3
    = 111 비트  (연속된 같은 비트를 끊어 넣는 채움 비트까지 세면 최대 130)

이므로 **111~130 마이크로초**입니다. 우리가 한 번 주고받는 것은 **명령 1 + 응답 1 = 2개**
이므로 선 위에서만 따져도 **222~260 마이크로초**, 곧 **초당 3,800~4,500번**이 선의 천장입니다.

무엇을 가르는가
---------------
지금 방식은 "하나 보내고 답을 기다렸다가 다음을 보낸다"입니다. 이러면 **선이 노는 동안**
USB 를 왕복하는 시간이 통째로 더해집니다. 그래서 이렇게 잽니다.

``depth=1``   하나 보내고 기다림 (지금 방식)
``depth=N``   N 개를 먼저 보내 놓고 N 개를 받음 (겹쳐 보내기)
``blast``     답을 아예 안 받고 보내기만

* 겹쳐 보냈을 때 **빨라지면** → 지금은 **왕복 시간**에 묶여 있다. 선에는 여유가 있다.
  펌웨어를 바꾸면(글자 방식 → 이진 방식) 이득이 있다.
* 겹쳐도 **안 빨라지면** → 이미 **선의 천장**이다. 펌웨어를 바꿔도 소용없다.

안전
----
이득도 토크도 0 인 명령만 보냅니다. 모터에 힘이 들어가지 않습니다.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402


def run(m: Motor, depth: int, seconds: float):
  """depth 개를 먼저 보내 놓고 그만큼 받는다. depth=1 이면 지금 방식."""
  n_sent = n_got = 0
  lat = []
  t0 = time.monotonic()
  while time.monotonic() - t0 < seconds:
    marks = []
    for _ in range(depth):
      marks.append(time.monotonic())
      m.send()
      n_sent += 1
    for k in range(depth):
      s = m.recv(timeout_s=0.02)
      if s is None:
        continue          # 하나 빠져도 나머지를 마저 읽는다 (break 하면 다음 묶음과 어긋난다)
      n_got += 1
      lat.append((time.monotonic() - marks[k]) * 1e3)
  el = time.monotonic() - t0
  return n_got / el, n_sent / el, lat


def blast(m: Motor, seconds: float):
  """답을 안 받고 보내기만. 커널 대기줄이 흡수하는 양까지 같이 보인다."""
  n = 0
  t0 = time.monotonic()
  while time.monotonic() - t0 < seconds:
    m.send()
    n += 1
  return n / (time.monotonic() - t0)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--seconds", type=float, default=3.0)
  ap.add_argument("--depths", default="1,2,3,4,6,8,12,16")
  a = ap.parse_args(argv)

  depths = [int(x) for x in a.depths.split(",") if x.strip()]
  print("선 위에서만 따진 천장: 프레임 111~130 마이크로초 x 2개 = 초당 3,800~4,500번\n")
  print(f"{'겹침':>5} {'주고받기/초':>12} {'주기':>10} {'왕복 중앙':>10} "
        f"{'5%':>7} {'95%':>7} {'최대':>7}  (밀리초)")
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    if m.read() is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    best = (0.0, 0)
    for d in depths:
      rate, srate, lat = run(m, d, a.seconds)
      if not lat:
        print(f"{d:5d}  응답 없음")
        continue
      lat.sort()
      q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
      print(f"{d:5d} {srate:10.1f} {rate:10.1f} {(1-rate/srate)*100:6.1f}% "
            f"{1e6/rate:8.0f} us {statistics.median(lat):9.3f} {q(0.95):7.3f}")
      if rate > best[0]:
        best = (rate, d)
    br = blast(m, a.seconds)
    print(f"\n답을 안 받고 보내기만: 초당 {br:.0f}개 "
          f"(선의 천장은 8바이트 프레임 기준 초당 약 9,000개 — 이보다 크면 대기줄에 쌓인 것)")
    print(f"\n가장 빠른 설정: 겹침 {best[1]}, 초당 {best[0]:.0f}번 "
          f"= 선 천장(4,000)의 {best[0]/4000*100:.0f}%")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
