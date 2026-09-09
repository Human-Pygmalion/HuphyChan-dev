#!/usr/bin/env python3
"""회전자 관성을 **마찰을 몰라도** 잰다. 두 번째 방법. 라즈베리파이에서 돌린다.

2026-09-09. 타성 감속(`inertia_id.py`)으로 74~81 g·m^2 이 나왔지만, 그 답은 **어제 잰 마찰
값이 맞다는 전제** 위에 서 있습니다. 마찰이 20% 틀리면 관성도 20% 틀립니다. 그래서 마찰이
**식에서 아예 사라지는** 방법으로 다시 잽니다.

어떻게 사라지는가
-----------------
같은 속도 구간에서 **서로 다른 두 토크**로 가속시킵니다. 속도가 같으면 마찰도 같습니다.

    관성 x 가속도1 = 토크1 - 마찰
    관성 x 가속도2 = 토크2 - 마찰
    ------------------------------ 빼면
    관성 x (가속도2 - 가속도1) = 토크2 - 토크1

마찰이 통째로 빠졌습니다. **모터가 몰래 더하는 치우침이 있어도 그것까지 같이 빠집니다.**
(타성 감속에서 놓은 뒤 보고 토크가 0 이 아니라 +0.3 이었던 그 값입니다.)

실제로는 토크를 다섯 단계로 주고 **토크 대 가속도** 직선을 맞춥니다. 기울기가 관성이고,
세로 절편이 그 속도 구간의 마찰입니다 — 즉 **어제 잰 마찰이 맞는지도 덤으로 검산됩니다.**

가속도는 어떻게 구하나
----------------------
미분하지 않습니다. 정해 둔 속도 구간(기본 150~350 도/s)을 지나는 데 **걸린 시간**을 재고
속도차를 그 시간으로 나눕니다. 양 끝 속도는 여러 점을 평균해서 잡음을 줄입니다.

안전
----
* 속도가 상한(기본 700 도/s)에 닿으면 **그 자리에서 토크를 0** 으로 놓습니다. 그 뒤는
  마찰이 알아서 세웁니다 — 제동을 걸지 않으므로 모터가 발전기가 되지 않습니다.
* 한 번 미는 시간이 0.3초를 넘지 않게 묶습니다.
* 출력축에 아무것도 안 달려 있어야 합니다.
* 양방향을 번갈아 하므로 제자리 근처에 머뭅니다. 다만 방향 전환은 **반드시 멈춘 뒤에**
  합니다 (도는 중에 뒤집으면 저전압 고장이 납니다 - 2026-09-09 실측).

    accel_id.py --motor-id 4 --model RS04 --torques 1.0,1.5,2.0,2.5,3.0 --repeats 3
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


def settle(m: Motor, timeout_s: float = 4.0) -> None:
  """토크 0 으로 두고 완전히 설 때까지 기다린다."""
  t0 = time.monotonic()
  n = 0
  while time.monotonic() - t0 < timeout_s:
    m.torque(0.0)
    st = m.recv(timeout_s=0.01)
    if st is None:
      continue
    n = n + 1 if abs(st.velocity_deg_s) < 8.0 else 0
    if n > 100:
      return


def burst(m: Motor, tau: float, v_cap: float, max_s: float):
  """토크 tau 를 주고 속도가 상한에 닿을 때까지 기록한다. 끝나면 곧바로 토크 0."""
  rows = []
  t0 = time.monotonic()
  why = "상한 도달"
  while True:
    t = time.monotonic() - t0
    if t >= max_s:
      why = "시간 상한"
      break
    m.torque(tau)
    st = m.recv(timeout_s=0.01)
    if st is None:
      continue
    rows.append((t, st.position_deg, st.velocity_deg_s, st.torque_nm, st.temp_c))
    if st.fault:
      why = "모터 고장 보고"
      break
    if abs(st.velocity_deg_s) >= v_cap:
      break
  for _ in range(10):
    m.torque(0.0); m.recv(timeout_s=0.002)
  return rows, why


def alpha_of(rows, v_lo: float, v_hi: float):
  """정해 둔 속도 구간을 지나는 평균 가속도 [라디안/s^2]."""
  v = [abs(r[2]) for r in rows]
  i1 = next((i for i, x in enumerate(v) if x >= v_lo), None)
  if i1 is None:
    return None
  i2 = next((i for i, x in enumerate(v[i1:], i1) if x >= v_hi), None)
  if i2 is None or i2 - i1 < 8:
    return None
  w = min(5, (i2 - i1) // 3)
  v1 = sum(v[i1:i1 + w]) / w * DEG
  v2 = sum(v[i2 - w:i2]) / w * DEG
  dt = rows[i2][0] - rows[i1][0]
  if dt <= 0 or v2 <= v1:
    return None
  return (v2 - v1) / dt, dt, i2 - i1


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--torques", default="1.0,1.5,2.0,2.5,3.0")
  ap.add_argument("--repeats", type=int, default=3)
  ap.add_argument("--v-lo", type=float, default=150.0, help="가속도를 재기 시작하는 속도 도/s")
  ap.add_argument("--v-hi", type=float, default=350.0, help="다 재는 속도 도/s")
  ap.add_argument("--v-cap", type=float, default=700.0, help="여기 닿으면 즉시 토크 0")
  ap.add_argument("--max-s", type=float, default=0.30, help="한 번 미는 시간 상한")
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--tag", default="")
  a = ap.parse_args(argv)

  taus = [float(x) for x in a.torques.split(",") if x.strip()]
  os.makedirs(a.out_dir, exist_ok=True)
  pts = []                      # (부호, 토크, 가속도)

  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · 시작 {st.position_deg:.1f}도 · {st.temp_c:.0f}도")
    print(f"속도 {a.v_lo:.0f}~{a.v_hi:.0f} 도/s 구간에서 가속도를 잽니다 "
          f"(상한 {a.v_cap:.0f}, 한 번 {a.max_s*1e3:.0f} 밀리초 이내)\n")
    # 고장 지우기는 **조건 없이** 보낸다. 2026-09-09: 모터 3 은 고장 표시가 False 인데도
    # 이걸 먼저 안 보내면 켜기가 안 먹어서, 토크를 줘도 전류가 0 인 채 가만히 있었다.
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)

    try:
      for rep in range(a.repeats):
        for tau in taus:
          for sign in (+1.0, -1.0):
            settle(m)                       # 방향 전환은 반드시 멈춘 뒤에
            rows, why = burst(m, sign * tau, a.v_cap, a.max_s)
            name = (f"accel_{a.motor_id}_{tau:g}_{'p' if sign > 0 else 'n'}_{rep}"
                    f"{('_' + a.tag) if a.tag else ''}.csv")
            with open(os.path.join(a.out_dir, name), "w", newline="") as f:
              wr = csv.writer(f)
              wr.writerow(["t_s", "pos_deg", "vel_deg_s", "tau_rep_nm", "temp_c"])
              wr.writerows(rows)
            tc = rows[-1][4] if rows else float("nan")
            r = alpha_of(rows, a.v_lo, a.v_hi)
            if r is None:
              print(f"  {sign*tau:+5.2f} Nm #{rep}: 구간을 못 지남 ({why}, {len(rows)}줄)")
            else:
              al, dt, n = r
              pts.append((sign, tau, al))
              print(f"  {sign*tau:+5.2f} Nm #{rep}: 가속도 {al:8.2f} 라디안/s^2 "
                    f"({dt*1e3:5.1f} ms, {n:4d}점) · {tc:.0f}도 · {why}")
            if tc >= a.temp_stop or why == "모터 고장 보고":
              print(f"    -> 중단: {why} / {tc:.0f}도")
              raise KeyboardInterrupt
      settle(m)
    except KeyboardInterrupt:
      pass
    finally:
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)

  # 토크 대 가속도 직선 맞춤. 방향별로 따로 (마찰이 방향에 따라 다르므로) + 합쳐서도.
  print()
  for label, sel in (("+ 방향", lambda s: s > 0), ("- 방향", lambda s: s < 0),
                     ("양쪽 합침", lambda s: True)):
    d = [(al, tau) for s, tau, al in pts if sel(s)]
    if len(d) < 4:
      continue
    n = len(d)
    sx = sum(p[0] for p in d); sy = sum(p[1] for p in d)
    sxx = sum(p[0] ** 2 for p in d); sxy = sum(p[0] * p[1] for p in d)
    syy = sum(p[1] ** 2 for p in d)
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
      continue
    J = (n * sxy - sx * sy) / den            # 기울기 = 관성
    f = (sy - J * sx) / n                    # 절편 = 그 속도 구간의 마찰
    ss = (n * sxy - sx * sy) ** 2 / (den * (n * syy - sy * sy)) if n * syy - sy * sy > 0 else 0
    res = [p[1] - (J * p[0] + f) for p in d]
    sd = (sum(r * r for r in res) / max(1, n - 2)) ** 0.5
    seJ = sd * (n / den) ** 0.5
    print(f"{label:>9}: 관성 {J*1e3:7.2f} +- {seJ*1e3:5.2f} g·m^2 · "
          f"마찰 절편 {f:5.3f} Nm · 결정계수 {ss:.4f} · {n}점")
  print("\n비교: 타성 감속으로는 74~81 g·m^2 · 화면 속 모형은 9.7 g·m^2")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
