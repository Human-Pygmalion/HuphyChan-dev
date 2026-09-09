#!/usr/bin/env python3
"""마찰을 다시 잰다 — **시킨 토크**와 **위치에서 구한 속도**만 쓴다. 라즈베리파이용.

2026-09-09. 사용자 지시: "이거 지금 셋업에서 다시 재봐. 이제 지연 없잖아."

예전 방법의 두 약점
-------------------
어제는 **속도를 붙잡아 놓고 모터가 보고하는 토크를 읽었습니다.** 오늘 그 두 가지가 다 흔들린다는
것을 알았습니다.

1. **보고 토크는 속도가 붙으면 못 믿습니다.** 토크를 0 으로 놓고 타성으로 도는 중에도
   +0.32(정방향) / −0.40(역방향)을 보고했습니다. 회전 방향을 따라 부호가 뒤집히는,
   실제 힘이 아닌 겉보기 값입니다.
2. **보고 속도는 859 도/s 에서 잘립니다** (부호화 상한 15.0 라디안/s). 실제로 1250 도/s 로
   돌고 있어도 859 라고 답합니다.

이번 방법
---------
**시킨 토크**를 그대로 쓰고(멈춰 있을 때 눈금이 0.995~1.053 로 확인된 값입니다),
속도는 **위치를 이어 붙여 미분**해서 구합니다. 둘 다 흔들리지 않습니다.

토크를 하나 정해 놓고 그대로 두면, 모터는 **마찰과 맞비기는 속도**에서 멈춥니다. 그 자리에서

    시킨 토크 = (움직일 때 마찰) + (점성) x 속도

이므로, 토크를 여러 값으로 주고 **토크 대 자리잡은 속도** 직선을 맞추면 세로 절편이
움직일 때 마찰, 기울기가 점성입니다. 가속도가 0 인 자리라 **관성이 식에 안 들어옵니다.**

붙어 있는 힘은 어떻게 피하나
----------------------------
정지 마찰(0.88 뉴턴미터)이 움직일 때 마찰(0.52)보다 크므로, 낮은 토크로는 **아예 출발을
못 합니다.** 그래서 매번 **높은 토크로 먼저 돌려 놓고**(1.4 뉴턴미터, 2.5초) 목표 토크로
낮춰서 위에서 아래로 자리잡게 합니다.

안전
----
* 진짜 속도에 상한을 겁니다. 닿으면 즉시 토크 0.
* 끝낼 때는 토크를 0 으로 놓고 마찰이 세웁니다. 제동을 걸지 않으므로 발전기가 되지 않습니다.
* 방향 전환은 완전히 선 뒤에만 합니다.

    friction_curve.py --motor-id 4 --model RS04 --torques 0.60,0.70,0.80,0.90,1.00
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402

DEG = 3.141592653589793 / 180.0


class TrueSpeed:
  """보고 속도 대신 **위치를 이어 붙여** 진짜 속도를 낸다 (되감김을 푼다)."""

  def __init__(self, window_s: float = 0.30):
    self.window_s = window_s
    self.acc = 0.0
    self.last = None
    self.trail = []

  def push(self, t: float, pos_deg: float) -> float:
    if self.last is None:
      self.last = pos_deg
    d = pos_deg - self.last
    while d > 720.0:
      d -= 1440.0
    while d < -720.0:
      d += 1440.0
    self.acc += d
    self.last = pos_deg
    self.trail.append((t, self.acc))
    while len(self.trail) > 2 and t - self.trail[0][0] > self.window_s:
      self.trail.pop(0)
    if len(self.trail) < 20 or self.trail[-1][0] <= self.trail[0][0]:
      return 0.0
    return (self.trail[-1][1] - self.trail[0][1]) / (self.trail[-1][0] - self.trail[0][0])


def hold(m: Motor, tau: float, seconds: float, v_cap: float, ts: TrueSpeed, t0: float,
         rows: list, temp_stop: float):
  """토크 tau 를 seconds 동안 준다. 매 순간의 진짜 속도를 rows 에 남긴다."""
  end = time.monotonic() + seconds
  why = ""
  while time.monotonic() < end:
    m.torque(tau)
    s = m.recv(timeout_s=0.01)
    if s is None:
      continue
    t = time.monotonic() - t0
    v = ts.push(t, s.position_deg)
    rows.append((t, tau, s.position_deg, s.velocity_deg_s, v, s.torque_nm, s.temp_c))
    if s.fault:
      return "모터가 고장을 보고함"
    if abs(v) >= v_cap:
      return f"진짜 속도 상한 {v_cap:.0f} 도/s"
    if s.temp_c >= temp_stop:
      return f"온도 {s.temp_c:.0f}도"
  return why


def coast_to_rest(m: Motor, timeout_s: float = 10.0) -> None:
  t0, n = time.monotonic(), 0
  while time.monotonic() - t0 < timeout_s:
    m.torque(0.0)
    s = m.recv(timeout_s=0.01)
    if s is None:
      continue
    n = n + 1 if abs(s.velocity_deg_s) < 8.0 else 0
    if n > 150:
      return


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--torques", default="0.58,0.62,0.66,0.70,0.75,0.80,0.85,0.90,0.95,1.00")
  ap.add_argument("--repeats", type=int, default=2)
  ap.add_argument("--spin-tau", type=float, default=1.4, help="출발용 토크 (붙어 있는 힘 넘기기)")
  ap.add_argument("--spin-s", type=float, default=2.5)
  ap.add_argument("--settle-s", type=float, default=14.0,
                  help="자리잡기를 기다리는 시간. 시간상수(관성/점성)가 약 2.8초라 그 5배")
  ap.add_argument("--read-s", type=float, default=2.5, help="끝의 이만큼을 평균해 읽는다")
  ap.add_argument("--v-cap", type=float, default=1400.0)
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--tag", default="")
  a = ap.parse_args(argv)

  taus = [float(x) for x in a.torques.split(",") if x.strip()]
  pts = []
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · {st.temp_c:.0f}도")
    print(f"출발 {a.spin_tau} 뉴턴미터 {a.spin_s}초 → 목표 토크로 {a.settle_s}초 자리잡기 → "
          f"끝 {a.read_s}초 평균\n")
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)
    t0 = time.monotonic()
    try:
      for rep in range(a.repeats):
        for sign in (+1.0, -1.0):
          for tau in taus:
            coast_to_rest(m)
            ts = TrueSpeed()
            rows = []
            why = hold(m, sign * a.spin_tau, a.spin_s, a.v_cap, ts, t0, rows, a.temp_stop)
            if not why:
              why = hold(m, sign * tau, a.settle_s, a.v_cap, ts, t0, rows, a.temp_stop)
            for _ in range(10):
              m.torque(0.0); m.recv(timeout_s=0.002)
            with open(os.path.join(a.out_dir,
                      f"fcurve_{a.motor_id}_{tau:g}_{'p' if sign > 0 else 'n'}_{rep}"
                      f"{('_' + a.tag) if a.tag else ''}.csv"), "w", newline="") as f:
              w = csv.writer(f)
              w.writerow(["t_s", "cmd_tau_nm", "pos_deg", "vel_rep_deg_s", "vel_true_deg_s",
                          "tau_rep_nm", "temp_c"])
              w.writerows(rows)
            tail = [r for r in rows if r[0] > rows[-1][0] - a.read_s]
            if len(tail) < 100:
              print(f"  {sign*tau:+5.2f} Nm #{rep}: 자료 부족 ({why})")
              continue
            vs = [abs(r[4]) for r in tail]
            v = sum(vs) / len(vs)
            spread = max(vs) - min(vs)
            tc = tail[-1][6]
            settled = spread < max(12.0, 0.03 * v)
            if settled and v > 5.0:
              pts.append((sign, tau, v * DEG))
            print(f"  {sign*tau:+5.2f} Nm #{rep}: 자리잡은 속도 {v:7.1f} 도/s "
                  f"({v*DEG:5.2f} 라디안/s) · 흔들림 {spread:5.1f} · "
                  f"{'자리잡음' if settled else '★ 아직 안 잡힘'} · {tc:.0f}도"
                  + (f" · {why}" if why else ""))
            if why in ("모터가 고장을 보고함",) or tc >= a.temp_stop:
              raise KeyboardInterrupt
      coast_to_rest(m)
    except KeyboardInterrupt:
      pass
    finally:
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)

  print()
  for label, sel in (("+ 방향", lambda s: s > 0), ("- 방향", lambda s: s < 0),
                     ("양쪽 합침", lambda s: True)):
    d = [(w, t) for s, t, w in pts if sel(s)]
    if len(d) < 3:
      continue
    n = len(d)
    sx = sum(p[0] for p in d); sy = sum(p[1] for p in d)
    sxx = sum(p[0] ** 2 for p in d); sxy = sum(p[0] * p[1] for p in d)
    syy = sum(p[1] ** 2 for p in d)
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
      continue
    b = (n * sxy - sx * sy) / den
    c = (sy - b * sx) / n
    r2 = (n * sxy - sx * sy) ** 2 / (den * (n * syy - sy * sy))
    res = [p[1] - (b * p[0] + c) for p in d]
    sd = (sum(r * r for r in res) / max(1, n - 2)) ** 0.5
    print(f"{label:>9}: 움직일 때 마찰 {c:6.3f} Nm · 점성 {b:7.4f} +- "
          f"{sd*(n/den)**0.5:6.4f} · 결정계수 {r2:.4f} · {n}점")
  print("\n어제 값(속도 붙잡고 보고 토크 읽기): 마찰 0.523 · 점성 0.0276")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
