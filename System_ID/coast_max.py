#!/usr/bin/env python3
"""최고 속도에서 **순수 타성 감속**을 받는다 — 마찰 곡선 전 구간용. 라즈베리파이에서 돌린다.

2026-09-09. 사용자 지시로 마찰을 다시 재는 중입니다. 지금까지 방법들은 **시킨 토크**를
식에 넣었는데, 이 방법은 토크가 **정확히 0** 이라 그것조차 안 들어갑니다. 남는 것은

    마찰(속도) = 관성 x (감속률)

뿐이고, 관성은 오늘 두 방법으로 확정했습니다(0.0773 kg·m²).

한 번 놓으면 최고 속도에서 0 까지 훑고 지나가므로, **속도 구간 전체의 마찰이 한 번에**
나옵니다. 특히 아주 느린 구간(붙어 있던 힘이 풀리는 자리)은 다른 방법으로는 못 봅니다.

    coast_max.py --motor-id 4 --model RS04 --repeats 4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402


def cool_to(m: Motor, target_c: float, max_s: float) -> float:
  """토크를 끈 채 온도가 내려가길 기다린다. 측정 자체가 모터를 데우기 때문이다."""
  t0 = time.monotonic()
  last = None
  while time.monotonic() - t0 < max_s:
    m.torque(0.0)
    s = m.recv(timeout_s=0.01)
    if s is None:
      continue
    last = s.temp_c
    if s.temp_c <= target_c and time.monotonic() - t0 > 1.0:
      return last
  return last if last is not None else -1.0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--spin-tau", type=float, default=2.0)
  ap.add_argument("--spin-s", type=float, default=6.0)
  ap.add_argument("--coast-s", type=float, default=4.0)
  ap.add_argument("--repeats", type=int, default=4)
  ap.add_argument("--cool-to", type=float, default=42.0)
  ap.add_argument("--cool-max-s", type=float, default=180.0)
  ap.add_argument("--temp-stop", type=float, default=47.0)
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--tag", default="")
  a = ap.parse_args(argv)

  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · {st.temp_c:.0f}도 · "
          f"{a.cool_to:.0f}도까지 식힌 뒤 시작합니다\n")
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)
    try:
      for rep in range(a.repeats):
        for sign in (+1.0, -1.0):
          tc = cool_to(m, a.cool_to, a.cool_max_s)
          # 올리기 (기록 안 함)
          t0 = time.monotonic()
          while time.monotonic() - t0 < a.spin_s:
            m.torque(sign * a.spin_tau)
            m.recv(timeout_s=0.01)
          # 놓기 (여기부터 기록)
          rows, t1, n_slow = [], time.monotonic(), 0
          why = "시간 상한"
          while time.monotonic() - t1 < a.coast_s:
            m.torque(0.0)
            s = m.recv(timeout_s=0.01)
            if s is None:
              continue
            rows.append((time.monotonic() - t1, s.position_deg, s.velocity_deg_s,
                         s.torque_nm, s.temp_c))
            if s.fault:
              why = "모터가 고장을 보고함"; break
            n_slow = n_slow + 1 if abs(s.velocity_deg_s) < 8.0 else 0
            if n_slow > 200:
              why = "멈춤"; break
          name = (f"coastmax_{a.motor_id}_{'p' if sign > 0 else 'n'}_{rep}"
                  f"{('_' + a.tag) if a.tag else ''}.csv")
          with open(os.path.join(a.out_dir, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "pos_deg", "vel_deg_s", "tau_rep_nm", "temp_c"])
            w.writerows(rows)
          v0 = max(abs(r[2]) for r in rows[:20]) if rows else 0
          tend = rows[-1][4] if rows else -1
          print(f"  #{rep} {'+' if sign > 0 else '-'} : 식힘 {tc:.0f}도 → "
                f"놓은 순간 보고속도 {v0:5.0f} 도/s · {len(rows):5d}점 · "
                f"{rows[-1][0] if rows else 0:.2f}초 · 끝 {tend:.0f}도 · {why}")
          if tend >= a.temp_stop or why == "모터가 고장을 보고함":
            print(f"    -> 중단"); raise KeyboardInterrupt
    except KeyboardInterrupt:
      pass
    finally:
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
