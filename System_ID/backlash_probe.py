#!/usr/bin/env python3
"""**감속기의 헐거움(백래시)을 잰다.** 모터가 말하는 각도와 실제 원판 각도를 함께 본다.

2026-09-11 · 사용자 지시: 벤치 A3.

왜 카메라가 필요한가
--------------------
헐거움은 **모터 안쪽과 출력축 사이**에 있습니다. 그래서 모터가 말하는 각도만 보면 안 보일
수 있습니다 — 엔코더가 회전자 쪽에 있으면 헐거운 만큼은 엔코더에 안 잡히기 때문입니다.
카메라는 **출력축에 붙은 원판**을 직접 봅니다. 둘을 같이 보면 그 사이가 드러납니다.

★ 덤: 둘이 완벽히 같이 움직이면, 그건 **엔코더가 사실상 출력축을 보고 있다**는 뜻이고
제어에는 좋은 소식입니다. 그것도 알아 둘 값입니다.

어떻게 재는가
-------------
자리를 붙잡아 둔 채(위치 제어), **얹는 힘을 −3 에서 +3 뉴턴미터까지 아주 천천히** 오갑니다.
축은 힘÷세기 만큼(약 ±9도) 움직이고, 그 사이 톱니는 한쪽 면에서 반대 면으로 옮겨 붙습니다.
힘을 되돌릴 때 같은 자리로 안 돌아오면 그 차이가 **헐거움 + 비틀림**입니다.

느리게 하는 것이 중요합니다 — 빠르면 관성이 끼어 헐거움과 구분이 안 됩니다.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_profiles import home  # noqa: E402
from rs_direct_socket import Motor  # noqa: E402

DEG = math.pi / 180.0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--bias", type=float, default=3.0, help="얹는 힘의 최대치 [뉴턴미터]")
  ap.add_argument("--period-s", type=float, default=12.0, help="한 왕복에 걸리는 시간")
  ap.add_argument("--cycles", type=int, default=2)
  ap.add_argument("--bound-deg", type=float, default=30.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--out", default=os.path.expanduser("~/System_ID/data/backlash.csv"))
  a = ap.parse_args(argv)

  rows = []
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    s = m.read()
    if s is None:
      print("모터가 응답하지 않습니다", file=sys.stderr); return 1
    print(f"모터 {a.motor_id}({a.model}) · {s.temp_c:.0f}도")
    m.clear_fault(); time.sleep(0.1); m.enable(); time.sleep(0.1)
    p0 = home(m, target_deg=s.position_deg, tol_deg=6.0, runaway_deg=300.0)
    m.clear_fault(); time.sleep(0.05); m.enable(); time.sleep(0.05)
    print(f"붙잡을 자리 {p0:.2f}도 · 얹는 힘 ±{a.bias} 뉴턴미터 · 한 왕복 {a.period_s}초 x "
          f"{a.cycles}\n")

    secs = a.period_s * a.cycles
    acc, last_raw, trail = p0, None, []
    t_start = time.monotonic()
    tau = 0.0
    why = "끝까지"
    while True:
      t = time.monotonic() - t_start
      if t >= secs:
        break
      m.torque(tau)
      st = m.recv(timeout_s=0.01)
      if st is None:
        continue
      if last_raw is None:
        last_raw = st.position_deg
      d = st.position_deg - last_raw
      while d > 720.0:
        d -= 1440.0
      while d < -720.0:
        d += 1440.0
      acc += d
      last_raw = st.position_deg
      trail.append((t, acc))
      while len(trail) > 2 and t - trail[0][0] > 0.02:
        trail.pop(0)
      vel = ((trail[-1][1] - trail[0][1]) / (trail[-1][0] - trail[0][0])
             if len(trail) > 4 and trail[-1][0] > trail[0][0] else st.velocity_deg_s)
      # 삼각파로 오가는 얹는 힘
      ph = (t % a.period_s) / a.period_s
      bias = a.bias * (4 * ph - 1 if ph < 0.5 else 3 - 4 * ph)
      tau = a.kp * (p0 - acc) * DEG + a.kd * (0.0 - vel) * DEG + bias
      tau = max(-a.tau_max, min(a.tau_max, tau))
      rows.append((t, bias, tau, acc, vel, st.torque_nm, st.temp_c))
      if abs(acc - p0) > a.bound_deg:
        why = f"경계 {a.bound_deg:.0f}도 넘음"
        break
      if st.fault:
        why = "모터가 고장을 보고함"
        break
    for _ in range(30):
      m.torque(0.0); m.recv(timeout_s=0.002)
    st = m.read(tries=10)
    print(f"끝난 이유: {why} · {len(rows)}줄 · 마지막 {st.position_deg:.2f}도 · "
          f"{st.temp_c:.0f}도 · 고장 {st.fault}")

  with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "bias_nm", "tau_cmd_nm", "pos_deg", "vel_deg_s", "tau_rep_nm", "temp_c"])
    w.writerows(rows)
  print(f"기록: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
