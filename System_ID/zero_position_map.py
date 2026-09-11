#!/usr/bin/env python3
"""**모터 안쪽 위치 되먹임이 축의 자리에 따라 달라지는가.** 실물 전에 닫아야 할 마지막 조각.

2026-09-11. 지금까지 알아낸 것:

* 축이 **영점 근처(0.1도)** 일 때는 정상입니다 — 목표를 보고각과 같게 주면 토크가 −0.03
  뉴턴미터, 목표를 옮기면 올바른 방향으로 반응합니다.
* 그런데 축이 영점에서 **멀리 가 있을 때**는 같은 명령에 **50~70 뉴턴미터**가 나왔습니다.

그래서 이 프로그램은 축을 여러 자리로 데려가 **같은 시험을 반복**합니다. 자리에 따라
달라지면, 실물에서 지켜야 할 규칙이 생깁니다 — "쓰기 전에 영점을 잡아라" 또는 "이 범위
밖에서는 안쪽 되먹임을 쓰지 마라".

안전
----
* 자리 옮기기는 **우리가 계산한 순수 토크**로만 합니다 (하루 종일 정확했던 경로).
* 시험은 **3 밀리초**만 힘을 줍니다. 65 뉴턴미터라도 축은 0.01도쯤 움직입니다.
* 시험 도중 토크가 커지면 그 자리에서 끊습니다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_profiles import home  # noqa: E402
from rs_direct_socket import Motor  # noqa: E402

KP_STEP = 5000.0 / 4095


def peek(m: Motor, p_des_deg: float, kp: float, ms: float = 3.0):
  taus, poss = [], []
  t0 = time.monotonic()
  while (time.monotonic() - t0) * 1e3 < ms:
    m.send(position_deg=p_des_deg, velocity_deg_s=0.0, kp=kp, kd=0.0)
    st = m.recv(timeout_s=0.004)
    if st is not None:
      taus.append(st.torque_nm); poss.append(st.position_deg)
  for _ in range(30):
    m.torque(0.0); m.recv(timeout_s=0.002)
  time.sleep(0.3)
  pk = max(taus, key=abs) if taus else 0.0
  mv = (poss[-1] - poss[0]) if len(poss) > 1 else 0.0
  return pk, mv


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--kp", type=float, default=5.0)
  ap.add_argument("--places", default="0,60,120,240,360,-120,-360,0")
  a = ap.parse_args(argv)
  kp_eff = int(a.kp / 5000.0 * 4095) * KP_STEP

  with Motor(motor_id=a.motor_id, model=a.model) as m:
    s = m.read()
    print(f"모터 {a.motor_id}({a.model}) · 시작 {s.position_deg:.2f}도 · {s.temp_c:.0f}도")
    print(f"kp {a.kp:g} → 실제 {kp_eff:.3f} · 목표는 늘 **그 자리의 보고각과 같게** 줍니다 "
          f"(제대로면 토크 0)\n")
    m.clear_fault(); time.sleep(0.1); m.enable(); time.sleep(0.1)
    print(f"  {'데려간 자리':>12}{'실제 보고각':>14}{'가장 큰 토크':>14}"
          f"{'안쪽 오차[라디안]':>18}{'움직임':>10}")
    for tgt in (float(x) for x in a.places.split(",")):
      p = home(m, target_deg=tgt, tol_deg=6.0, runaway_deg=500.0, timeout_s=25.0)
      m.clear_fault(); time.sleep(0.05); m.enable(); time.sleep(0.05)
      s = m.read(tries=10)
      pk, mv = peek(m, s.position_deg, a.kp)
      print(f"  {tgt:12.0f}{s.position_deg:14.2f}{pk:14.2f}{pk/kp_eff:18.2f}{mv:10.2f}")
    home(m, target_deg=0.0, tol_deg=6.0, runaway_deg=500.0)
    s = m.read(tries=10)
    print(f"\n끝 · {s.position_deg:.2f}도 · {s.temp_c:.0f}도 · 고장 {s.fault}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
