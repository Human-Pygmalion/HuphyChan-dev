#!/usr/bin/env python3
"""회전자 관성을 잰다. **라즈베리파이에서 돌린다.**

2026-09-09. 세 값 중 마지막 하나입니다. 마찰(0.523 뉴턴미터)과 점성(0.0276)은 이미 쟀고,
관성만 남았습니다. 관성은 **빠른 속도 변화에서만** 드러나서 가상머신 경로(초당 230회)로는
못 쟀는데, 라즈베리파이에서 초당 1814회가 나와 이제 됩니다.

어떻게 재는가 — 타성 감속
--------------------------
모터를 한 방향으로 돌린 뒤 **토크를 정확히 0 으로 놓고 손을 뗍니다.** 그러면 모터를 세우는
힘은 마찰밖에 없습니다. 뉴턴의 식은 이렇게 됩니다:

    관성 x (속도 변화율) = -(움직일 때 마찰) - (점성) x 속도

여기서 마찰과 점성은 **이미 아는 값**입니다. 그래서 미지수는 관성 하나뿐입니다.

미분하지 않는다
---------------
"속도 변화율"을 그대로 구하려면 속도를 미분해야 하는데, 모터가 보고하는 속도는 잡음이
±6.5 도/s 나 됩니다. 미분하면 그 잡음이 몇 배로 커집니다. 그래서 **반대로 적분**합니다.
위 식을 속도가 v1 에서 v2 로 떨어지는 구간에서 시간으로 적분하면:

    관성 x (v1 - v2) = 마찰 x (걸린 시간) + 점성 x (그동안 돈 각도)

오른쪽은 전부 **직접 잰 값**입니다 — 걸린 시간은 시계로, 돈 각도는 위치 차이로. 미분이
한 번도 안 들어갑니다. 정리하면

    관성 = (마찰 x 걸린 시간 + 점성 x 돈 각도) / (v1 - v2)

교차 확인으로 **속도 변화율 대 속도**의 직선 맞춤도 함께 합니다(기울기 = -점성/관성,
세로 절편 = -마찰/관성). 두 길이 같은 답을 주면 믿을 수 있습니다.

왜 끝부분을 버리는가
--------------------
아주 느려지면 **붙어 있는 힘**(0.88 뉴턴미터)이 움직일 때 마찰(0.523)을 대신하기 시작해서
식이 달라집니다. 그래서 맨 끝(기본 25% 아래)은 맞춤에서 뺍니다.

안전
----
* 감속은 **마찰이 알아서** 시킵니다. 우리가 제동을 걸지 않으므로 모터가 발전기가 되지
  않습니다 (2026-09-09 에 급제동으로 저전압 고장을 낸 적이 있습니다).
* 방향을 뒤집지 않습니다. 같은 이유입니다.
* 출력축에 아무것도 안 달려 있어야 합니다. 카메라로 확인했습니다.
* 온도가 정한 값(기본 45도)에 닿으면 즉시 중단합니다.

    inertia_id.py --motor-id 4 --model RS04 --speeds 200,400,600 --repeats 3
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

# 2026-09-09 실측 (docs/experiments/2026-09-09_motor_sysid_attempt.md, 데운 상태)
TAU_C_NM = 0.523
"""움직이는 중의 마찰. 방향별로 + 0.490 / - 0.557 이었고 그 가운데 값."""
B_NM_S_RAD = 0.0276
"""점성 계수. 토크 = 마찰 + 이 값 x 속도(라디안/초)."""


def coast_once(m: Motor, speed_deg_s: float, kd: float, ramp_s: float, hold_s: float,
               max_coast_s: float, temp_stop: float) -> tuple[list, str]:
  """한 번: 속도를 올리고 - 잠시 유지하고 - 토크를 0 으로 놓고 멈출 때까지 기록."""
  rows = []
  t0 = time.monotonic()
  # 올리기: 속도 명령을 ramp_s 동안 0 에서 목표까지
  while True:
    t = time.monotonic() - t0
    if t >= ramp_s + hold_s:
      break
    tgt = speed_deg_s * min(1.0, t / ramp_s) if ramp_s > 0 else speed_deg_s
    m.send(position_deg=0.0, velocity_deg_s=tgt, kp=0.0, kd=kd)
    st = m.recv(timeout_s=0.01)
    if st is None:
      continue
    if st.temp_c >= temp_stop:
      return rows, f"온도 {st.temp_c:.0f}도"
    if st.fault:
      return rows, "모터 고장 보고"

  # 놓기: 여기서부터가 실제 측정 구간
  t_release = time.monotonic()
  n_slow = 0
  while True:
    t = time.monotonic() - t_release
    if t >= max_coast_s:
      return rows, "시간 초과 (안 멈춤)"
    m.torque(0.0)                       # kp=kd=0, 토크 0 -> 진짜로 아무 힘도 안 준다
    st = m.recv(timeout_s=0.01)
    if st is None:
      continue
    rows.append((t, st.position_deg, st.velocity_deg_s, st.torque_nm, st.temp_c))
    if st.fault:
      return rows, "모터 고장 보고"
    # 멈춤 판정: 속도가 잡음 수준(±6.5 도/s) 안에 얼마간 이어질 때
    n_slow = n_slow + 1 if abs(st.velocity_deg_s) < 8.0 else 0
    if n_slow > 60:
      return rows, "멈춤"


def fit(rows, sign: float, lo_frac: float, hi_frac: float):
  """적분법으로 관성을 구한다. rows = [(t, pos_deg, vel_deg_s, ...)]"""
  if len(rows) < 30:
    return None
  v0 = max(abs(r[2]) for r in rows[:10])
  v_hi, v_lo = hi_frac * v0, lo_frac * v0

  # 속도가 v_hi 아래로 처음 내려간 곳 ~ v_lo 아래로 처음 내려간 곳
  i1 = next((i for i, r in enumerate(rows) if abs(r[2]) <= v_hi), None)
  i2 = next((i for i, r in enumerate(rows) if abs(r[2]) <= v_lo), None)
  if i1 is None or i2 is None or i2 - i1 < 20:
    return None

  # 양 끝 속도는 평균으로 (한 점을 쓰면 잡음이 그대로 들어간다)
  w = 5
  v1 = sum(abs(r[2]) for r in rows[i1:i1 + w]) / w * DEG
  v2 = sum(abs(r[2]) for r in rows[i2 - w:i2]) / w * DEG
  dt = rows[i2][0] - rows[i1][0]
  dth = abs(rows[i2][1] - rows[i1][1]) * DEG
  if v1 - v2 <= 0 or dt <= 0:
    return None
  J = (TAU_C_NM * dt + B_NM_S_RAD * dth) / (v1 - v2)

  # 교차 확인: 속도 변화율 대 속도의 직선 맞춤 (구간을 10 토막 내어 각 토막의 기울기)
  pts = []
  k = max(3, (i2 - i1) // 10)
  for a in range(i1, i2 - k, k):
    b = a + k
    ddt = rows[b][0] - rows[a][0]
    if ddt <= 0:
      continue
    acc = (rows[b][2] - rows[a][2]) * DEG / ddt
    vm = (rows[b][2] + rows[a][2]) / 2 * DEG
    pts.append((vm, acc))
  J2 = None
  if len(pts) >= 4:
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if abs(den) > 1e-12:
      slope = (n * sxy - sx * sy) / den          # = -점성/관성
      inter = (sy - slope * sx) / n              # = -마찰/관성 (부호는 회전 방향)
      if abs(inter) > 1e-9:
        J2 = TAU_C_NM / abs(inter)
  return dict(J=J, J_cross=J2, v1=v1, v2=v2, dt=dt, dth=dth,
              n=i2 - i1, tau_rep=sum(abs(r[3]) for r in rows[i1:i2]) / (i2 - i1))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--speeds", default="200,400,600", help="타성 감속 시작 속도 도/s")
  ap.add_argument("--repeats", type=int, default=3)
  ap.add_argument("--kd", type=float, default=2.0)
  ap.add_argument("--ramp-s", type=float, default=2.0)
  ap.add_argument("--hold-s", type=float, default=1.5, help="목표 속도에서 안정될 때까지")
  ap.add_argument("--max-coast-s", type=float, default=3.0)
  ap.add_argument("--lo-frac", type=float, default=0.25, help="이 아래는 붙는 힘 영역이라 버림")
  ap.add_argument("--hi-frac", type=float, default=0.92, help="놓은 직후 과도구간을 피함")
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--tag", default="")
  a = ap.parse_args(argv)

  speeds = [float(x) for x in a.speeds.split(",") if x.strip()]
  os.makedirs(a.out_dir, exist_ok=True)
  results = []

  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · 시작 {st.position_deg:.1f}도 · {st.temp_c:.0f}도")
    print(f"쓰는 값: 마찰 {TAU_C_NM} 뉴턴미터 · 점성 {B_NM_S_RAD} (2026-09-09 실측)\n")
    # 고장 지우기는 **조건 없이** 보낸다. 2026-09-09: 모터 3 은 고장 표시가 False 인데도
    # 이걸 먼저 안 보내면 켜기가 안 먹어서, 토크를 줘도 전류가 0 인 채 가만히 있었다.
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)

    try:
      for sp in speeds:
        for rep in range(a.repeats):
          rows, why = coast_once(m, sp, a.kd, a.ramp_s, a.hold_s, a.max_coast_s, a.temp_stop)
          name = f"coast_{a.motor_id}_{int(sp)}_{rep}{('_' + a.tag) if a.tag else ''}.csv"
          path = os.path.join(a.out_dir, name)
          with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t_s", "pos_deg", "vel_deg_s", "tau_rep_nm", "temp_c"])
            wr.writerows(rows)
          r = fit(rows, 1.0, a.lo_frac, a.hi_frac)
          tc = rows[-1][4] if rows else float("nan")
          if r is None:
            print(f"  {sp:4.0f} 도/s #{rep}: 맞춤 실패 ({why}, {len(rows)}줄)")
          else:
            results.append((sp, rep, r, tc))
            cross = f"{r['J_cross']*1e3:6.2f}" if r["J_cross"] else "  --  "
            print(f"  {sp:4.0f} 도/s #{rep}: 관성 {r['J']*1e3:6.2f} g·m^2 "
                  f"(교차 {cross}) · {r['v1']:5.2f}->{r['v2']:5.2f} 라디안/s · "
                  f"{r['dt']*1e3:5.0f} ms · {r['dth']:5.2f} 라디안 · {r['n']:4d}점 · "
                  f"놓은 뒤 보고토크 평균 {r['tau_rep']:.3f} · {tc:.0f}도 · {why}")
          if why not in ("멈춤",):
            print(f"    -> 중단: {why}")
            raise KeyboardInterrupt
          time.sleep(0.4)
    except KeyboardInterrupt:
      pass
    finally:
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)

  if results:
    Js = sorted(r["J"] for _, _, r, _ in results)
    n = len(Js)
    med = Js[n // 2] if n % 2 else (Js[n // 2 - 1] + Js[n // 2]) / 2
    mean = sum(Js) / n
    sd = (sum((j - mean) ** 2 for j in Js) / max(1, n - 1)) ** 0.5
    print(f"\n{'='*62}")
    print(f"회전자 관성 = {mean*1e3:.3f} g·m^2  (중앙 {med*1e3:.3f}, 흩어짐 {sd*1e3:.3f}, {n}회)")
    print(f"          = {mean:.6f} kg·m^2")
    print(f"{'='*62}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
