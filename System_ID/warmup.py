#!/usr/bin/env python3
"""모터를 돌려 데운다. **로봇 위에서 돌린다.**

2026-09-09, 사용자 지시: "5분정도 최고속도로 기동해서 웜업후 다시 해봐."

마찰은 온도에 크게 좌우됩니다. 차가운 윤활유는 뻑뻑하고, 데워지면 묽어집니다. 그래서 차가운
상태에서 잰 값 하나로 "이 모터의 마찰은 얼마"라고 말할 수 없습니다. **같은 측정을 데운 뒤에
다시 해서 두 값을 나란히 놓아야** 합니다.

두 가지 방식
------------
``--mode speed`` (기본) — 속도를 붙잡는 명령. 다만 **속도 명령 칸의 부호화 상한이
15.0 라디안/s = 859 도/s** 라, 이 방식으로는 그 위로 못 올라갑니다. 비례 제어의 처짐까지
빼면 실제 상한은 약 825 도/s 입니다.

``--mode torque`` — **속도를 시키지 않고 토크만 줍니다.** 그러면 모터는 마찰과 맞비길 때까지
계속 빨라지므로 **진짜 최고 속도**가 나옵니다. 2026-09-09 실측으로 1.5 뉴턴미터에서
1059 도/s(18.5 라디안/s)까지 갔습니다. 사용자 지시("최고속도로 웜업")에 맞는 쪽은 이쪽입니다.

  ★ 이 영역에서는 **모터가 보고하는 속도를 믿으면 안 됩니다.** 보고 값도 같은 15.0 라디안/s
  에서 잘려서, 실제 1059 도/s 인데 859 라고 답합니다. 그래서 이 방식에서는 **위치를 미분해서**
  진짜 속도를 계산합니다(되감김을 풀어서).

어떻게 도는가
-------------
속도를 붙잡는 명령(kp=0, kd만)으로 돌립니다. **방향을 정해진 간격마다 뒤집습니다** — 한
방향으로만 계속 돌면 회전이 한없이 쌓이는데, 뒤집으면 제자리 근처에 머뭅니다. 방향을 바꿀 때
가속·감속이 생기므로 데우는 효과도 더 좋습니다.

목표 속도는 **천천히 올립니다**. 처음부터 최고 속도를 시키면 속도 차이가 커서 순간 토크가
크게 튑니다.

멈추는 조건 (하나라도 걸리면 즉시)
----------------------------------
* 온도가 정한 값에 닿으면 — 기본 45도 (모터를 끊는 값 50도보다 낮게)
* 토크가 정한 값을 넘어 이어지면
* 모터가 고장을 보고하면
* 응답이 끊기면
* 정해진 시간이 끝나면

어느 경우든 속도 명령을 0 으로 되돌린 뒤 토크를 끕니다. 갑자기 힘만 빼면 관성으로 계속
돕니다.

    warmup.py --seconds 300 --speed 600 --motor-id 4 --model RS04
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
  # 라즈베리파이: 파이썬 기본 기능만 쓰는 구동부. 아무것도 설치할 필요가 없다.
  from rs_direct_socket import Motor  # noqa: E402
except ImportError:                    # 로봇 가상머신: python-can 이 이미 깔려 있다
  from rs_direct import Motor  # noqa: E402


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--seconds", type=float, default=300.0)
  ap.add_argument("--mode", choices=("speed", "torque"), default="speed")
  ap.add_argument("--speed", type=float, default=600.0, help="속도 방식일 때 목표 속도 도/s")
  ap.add_argument("--torque", type=float, default=1.6,
                  help="토크 방식일 때 줄 토크. 1.5 에서 약 1059 도/s 가 나왔다")
  ap.add_argument("--v-cap", type=float, default=1250.0,
                  help="위치에서 구한 진짜 속도가 여기 닿으면 즉시 중단")
  ap.add_argument("--kd", type=float, default=2.0)
  ap.add_argument("--flip-s", type=float, default=1e9,
                  help="이 간격마다 방향을 뒤집는다. **기본은 뒤집지 않음** — 600 도/s 에서 "
                       "뒤집으면 가속에 전원이 주저앉아 저전압 고장이 나고, 감속에 모터가 "
                       "발전기가 되어 과전압 경고가 뜬다(2026-09-09 실측). 한 방향으로 꾸준히 "
                       "도는 것이 전원에 훨씬 순하다")
  ap.add_argument("--decel-s", type=float, default=6.0, help="끝낼 때 속도를 0 까지 내리는 시간")
  ap.add_argument("--ramp-s", type=float, default=3.0, help="목표 속도까지 올리는 시간")
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--torque-stop", type=float, default=20.0)
  ap.add_argument("--report-s", type=float, default=20.0)
  ap.add_argument("--out", default="/home/syaro/System_ID/data/warmup.csv")
  a = ap.parse_args(argv)

  rows = []
  stop = None
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    t_start_c = st.temp_c
    print(f"모터 {a.motor_id}({a.model}) · 시작 {st.position_deg:.1f}도 · {t_start_c:.0f}도")
    tgt = (f"토크 {a.torque:.2f} 뉴턴미터 (속도는 마찰이 정함)" if a.mode == "torque"
           else f"목표 {a.speed:.0f} 도/s")
    print(f"{tgt} · {a.seconds:.0f}초 · {a.temp_stop:.0f}도에서 중단 · "
          f"진짜 속도 상한 {a.v_cap:.0f} 도/s\n")
    # 고장 지우기는 **조건 없이** 보낸다. 2026-09-09: 모터 3 은 고장 표시가 False 인데도
    # 이걸 먼저 안 보내면 켜기가 안 먹어서, 토크를 줘도 전류가 0 인 채 가만히 있었다.
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)

    t0 = time.monotonic()
    next_report = a.report_s
    hot_streak = 0
    unwrapped, last_raw, trail = None, 0.0, []
    fast_streak = 0
    try:
      while True:
        t = time.monotonic() - t0
        if t >= a.seconds:
          stop = "정해진 시간 끝"
          break
        # 방향: flip_s 마다 뒤집는다. 목표 속도는 ramp_s 동안 올린다.
        sign = 1.0 if int(t / a.flip_s) % 2 == 0 else -1.0
        phase = t - int(t / a.flip_s) * a.flip_s      # 이번 구간에서 얼마나 지났나
        scale = min(1.0, phase / a.ramp_s) if a.ramp_s > 0 else 1.0
        if a.mode == "torque":
          target = sign * a.torque * scale
          m.torque(target)
        else:
          target = sign * a.speed * scale
          m.send(position_deg=0.0, velocity_deg_s=target, kp=0.0, kd=a.kd)
        s2 = m.recv(timeout_s=0.01)
        if s2 is None:
          continue
        # 위치를 이어 붙여 진짜 속도를 구한다. 보고 속도는 859 도/s 에서 잘린다.
        if unwrapped is None:
          unwrapped, last_raw = 0.0, s2.position_deg
        d = s2.position_deg - last_raw
        while d > 720.0:
          d -= 1440.0
        while d < -720.0:
          d += 1440.0
        unwrapped += d
        last_raw = s2.position_deg
        trail.append((t, unwrapped))
        while len(trail) > 1 and t - trail[0][0] > 0.25:
          trail.pop(0)
        v_true = ((trail[-1][1] - trail[0][1]) / (trail[-1][0] - trail[0][0])
                  if len(trail) > 20 and trail[-1][0] > trail[0][0] else s2.velocity_deg_s)
        rows.append((t, target, s2.position_deg, s2.velocity_deg_s, s2.torque_nm, s2.temp_c,
                     v_true))
        # 상한 판정은 **이어질 때만**. 표본이 한 번 빠지면(58 ms 구멍을 실제로 겪었다)
        # 위치 되감김이 어긋나 한 점만 치솟는데, 그걸로 멈추면 안 된다.
        fast_streak = fast_streak + 1 if abs(v_true) >= a.v_cap else 0
        if fast_streak > 300:
          stop = f"진짜 속도가 상한 {a.v_cap:.0f} 도/s 에 닿음"
          break
        if s2.temp_c >= a.temp_stop:
          stop = f"온도 {s2.temp_c:.0f}도 도달"
          break
        if s2.fault:
          stop = "모터가 고장을 보고함"
          break
        hot_streak = hot_streak + 1 if abs(s2.torque_nm) > a.torque_stop else 0
        if hot_streak > 50:                    # 잠깐 튄 것이 아니라 이어질 때만
          stop = f"토크가 {a.torque_stop} 뉴턴미터를 넘어 이어짐"
          break
        if t >= next_report:
          recent = [r for r in rows if r[0] > t - a.report_s]
          vs = [abs(r[6]) for r in recent] or [0]
          ts = [abs(r[4]) for r in recent] or [0]
          print(f"  {t:5.0f}초 · 온도 {s2.temp_c:5.1f}도 (+{s2.temp_c-t_start_c:.1f}) · "
                f"진짜 속도 평균 {sum(vs)/len(vs):6.0f} 최고 {max(vs):6.0f} 도/s "
                f"(보고 {abs(s2.velocity_deg_s):5.0f}) · "
                f"토크 평균 {sum(ts)/len(ts):5.2f} 최고 {max(ts):5.2f} 뉴턴미터")
          next_report += a.report_s
    except KeyboardInterrupt:
      stop = "사람이 중단"
    finally:
      # 속도를 **천천히** 0 으로 데려온다. 갑자기 0 을 시키면 모터가 급제동하며 발전기가
      # 되어 전압을 되밀고, 그러면 과전압 경고가 뜬다(2026-09-09 실제로 겪음 - 방향을
      # 15초마다 뒤집다가 저전압 고장 + 과전압 경고로 멈췄다).
      t_dec = time.monotonic()
      while time.monotonic() - t_dec < a.decel_s:
        frac = max(0.0, 1.0 - (time.monotonic() - t_dec) / a.decel_s)
        if a.mode == "torque":
          # 토크를 0 으로 줄여 가면 마찰이 세운다. 음의 토크를 주면 제동이 되어 발전기가 된다.
          m.torque(a.torque * frac)
        else:
          m.send(position_deg=0.0, velocity_deg_s=a.speed * frac, kp=0.0, kd=a.kd)
        m.recv(timeout_s=0.005)
      for _ in range(15):
        m.torque(0.0); m.recv(timeout_s=0.002)

    st = m.read(tries=8)
    print(f"\n끝난 이유: {stop}")
    if st:
      print(f"끝 위치 {st.position_deg:.1f}도 · 온도 {st.temp_c:.0f}도 "
            f"(시작 {t_start_c:.0f}도, +{st.temp_c - t_start_c:.0f}) · 고장 {st.fault}")

  if rows:
    with open(a.out, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(["t_s", "cmd", "pos_deg", "vel_deg_s", "tau_est_nm", "temp_c",
                  "vel_true_deg_s"])
      w.writerows(rows)
    print(f"기록: {a.out} ({len(rows)}줄)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
