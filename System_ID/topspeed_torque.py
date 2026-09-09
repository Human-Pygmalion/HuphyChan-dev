#!/usr/bin/env python3
"""순수 토크로 이 모터의 **진짜** 최고 속도를 찾는다. 라즈베리파이에서 돌린다.

2026-09-09. 속도 명령으로 재니 825 도/s 에서 딱 멈췄는데, 토크는 0.84 뉴턴미터밖에 안 쓰고
있었습니다. 모터가 힘이 부친 게 아니라 **명령어가 먼저 바닥난 것**이었습니다 — MIT 방식
명령의 속도 칸이 담을 수 있는 최대가 15.0 라디안/s = **859 도/s** 이고, 비례 제어의 처짐
34 도/s 를 빼면 정확히 825 가 됩니다.

그래서 속도를 시키지 않고 **토크만 줍니다.** 그러면 모터는 마찰과 맞비길 때까지 계속
빨라지고, 그 자리가 그 토크에서의 최고 속도입니다. 토크를 여러 값으로 주면 무부하 근처의
토크-속도 곡선이 그대로 나옵니다.

안전
----
* 속도 상한을 걸고, 닿으면 즉시 토크를 0 으로 내립니다.
* 끝낼 때는 토크를 0 으로 놓고 **마찰이 세우게** 둡니다 — 제동을 걸지 않으므로 모터가
  발전기가 되지 않습니다(급제동은 과전압을 냅니다).
* 방향은 한쪽으로만. 도는 중에 뒤집지 않습니다.

    topspeed_torque.py --motor-id 4 --model RS04 --torques 0.9,1.1,1.3,1.5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402


def coast_to_rest(m: Motor, timeout_s: float = 8.0) -> None:
  t0, n = time.monotonic(), 0
  while time.monotonic() - t0 < timeout_s:
    m.torque(0.0)
    s = m.recv(timeout_s=0.01)
    if s is None:
      continue
    n = n + 1 if abs(s.velocity_deg_s) < 10.0 else 0
    if n > 100:
      return


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--torques", default="0.9,1.1,1.3,1.5")
  ap.add_argument("--v-cap", type=float, default=1250.0)
  ap.add_argument("--max-s", type=float, default=10.0)
  ap.add_argument("--settle-window-s", type=float, default=1.0,
                  help="이 시간 동안 속도가 거의 안 변하면 자리잡은 것으로 본다")
  ap.add_argument("--settle-deg-s", type=float, default=6.0)
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  a = ap.parse_args(argv)

  taus = [float(x) for x in a.torques.split(",") if x.strip()]
  best = 0.0
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · {st.temp_c:.0f}도 · 속도 상한 {a.v_cap:.0f} 도/s\n")
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)
    try:
      for tau in taus:
        coast_to_rest(m)
        rows, why, t0 = [], "시간 상한", time.monotonic()
        while True:
          t = time.monotonic() - t0
          if t >= a.max_s:
            break
          m.torque(tau)
          s = m.recv(timeout_s=0.01)
          if s is None:
            continue
          rows.append((t, s.position_deg, s.velocity_deg_s, s.torque_nm, s.temp_c))
          if s.fault:
            why = "모터가 고장을 보고함"; break
          if abs(s.velocity_deg_s) >= a.v_cap:
            why = "속도 상한 도달"; break
          if s.temp_c >= a.temp_stop:
            why = f"온도 {s.temp_c:.0f}도"; break
          w = [r for r in rows if r[0] > t - a.settle_window_s]
          if t > 2.0 and len(w) > 200:
            vv = [abs(r[2]) for r in w]
            if max(vv) - min(vv) < a.settle_deg_s:
              why = "자리잡음"; break
        for _ in range(10):
          m.torque(0.0); m.recv(timeout_s=0.002)
        with open(os.path.join(a.out_dir, f"topspeed_{a.motor_id}_{tau:g}.csv"),
                  "w", newline="") as f:
          w = csv.writer(f)
          w.writerow(["t_s", "pos_deg", "vel_deg_s", "tau_rep_nm", "temp_c"])
          w.writerows(rows)
        tail = [abs(r[2]) for r in rows[-300:]] or [0.0]
        v = sum(tail) / len(tail)
        best = max(best, v)
        print(f"  토크 {tau:4.2f} 뉴턴미터 → 최고 {v:6.0f} 도/s "
              f"({v*3.14159/180:5.2f} 라디안/s) · {rows[-1][0]:4.1f}초 · "
              f"{rows[-1][4]:.0f}도 · {why}")
        if why in ("모터가 고장을 보고함",) or (rows and rows[-1][4] >= a.temp_stop):
          break
      coast_to_rest(m)
    except KeyboardInterrupt:
      pass
    finally:
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)
  print(f"\n순수 토크로 낸 최고 속도: {best:.0f} 도/s "
        f"(속도 명령으로는 859 도/s 가 부호화 상한, 실제 825)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
