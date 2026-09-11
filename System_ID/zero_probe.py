#!/usr/bin/env python3
"""**영점을 다시 잡으면 모터 안쪽의 큰 오차가 사라지는가.** 아주 짧게, 아주 조심스럽게.

2026-09-11. 앞선 시험에서 확인된 것: 목표를 **지금 보고된 각도와 똑같이** 주어도, 세기 5
에서 모터가 **50~70 뉴턴미터**를 냅니다. 제대로라면 0 이어야 합니다. 그 토크가 뜻하는 안쪽
오차는 10~14 라디안 — 부호화 한계(12.57)와 같은 크기입니다.

남은 물음: **영점 재설정으로 고쳐지는가.** 앞 시험은 축이 되감김 경계까지 밀려가 판독이
섞였습니다. 그래서 여기서는

* 힘을 주는 시간을 **3 밀리초**로 줄입니다. 65 뉴턴미터라도 축은 0.01도밖에 안 움직입니다.
* 판단은 **최대 토크**로 합니다 (평균은 짧은 버스트에서 의미가 없습니다).
* 한 번 잴 때마다 힘을 끄고 자리를 읽습니다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402

KP_STEP = 5000.0 / 4095            # kp 칸 한 눈금. 이보다 작으면 **잘려서 0** 이 된다.


def peek(m: Motor, p_des_deg: float, kp: float, ms: float = 3.0):
  """3 밀리초만 힘을 주고 끈다. 가장 큰 토크와 움직인 각도를 돌려준다."""
  taus, poss = [], []
  t0 = time.monotonic()
  while (time.monotonic() - t0) * 1e3 < ms:
    m.send(position_deg=p_des_deg, velocity_deg_s=0.0, kp=kp, kd=0.0)
    st = m.recv(timeout_s=0.004)
    if st is not None:
      taus.append(st.torque_nm); poss.append(st.position_deg)
  for _ in range(30):
    m.torque(0.0); m.recv(timeout_s=0.002)
  time.sleep(0.35)
  pk = max((abs(x) for x in taus), default=0.0)
  sgn = max(taus, key=abs) if taus else 0.0
  mv = (poss[-1] - poss[0]) if len(poss) > 1 else 0.0
  return sgn, pk, mv, len(taus)


def round_trip(m: Motor, kp: float, label: str):
  kp_eff = int(kp / 5000.0 * 4095) * KP_STEP
  s = m.read(tries=10)
  p = s.position_deg
  t0, pk0, mv0, n0 = peek(m, p, kp)
  s = m.read(tries=10)
  t1, pk1, mv1, n1 = peek(m, s.position_deg + 0.2 / 0.0174533, kp)   # 목표를 0.2 라디안 옆으로
  print(f"  {label:<12} 보고각 {p:+9.2f}도 · kp {kp:g}→{kp_eff:.3f}")
  print(f"     목표 = 보고각        가장 큰 토크 {t0:+8.2f} 뉴턴미터 → 안쪽 오차 "
        f"{t0/kp_eff:+7.2f} 라디안 · 움직임 {mv0:+6.2f}도 · 표본 {n0}")
  print(f"     목표 +0.2 라디안     가장 큰 토크 {t1:+8.2f} → 바뀐 양 {t1-t0:+7.2f} "
        f"(기대 {kp_eff*0.2:+.2f}) · 움직임 {mv1:+6.2f}도")
  return t0 / kp_eff if kp_eff else float("nan")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--kp", type=float, default=5.0)
  a = ap.parse_args(argv)

  with Motor(motor_id=a.motor_id, model=a.model) as m:
    s = m.read()
    if s is None:
      print("모터가 응답하지 않습니다", file=sys.stderr); return 1
    print(f"모터 {a.motor_id}({a.model}) · 시작 {s.position_deg:.2f}도 · {s.temp_c:.0f}도 · "
          f"고장 {s.fault}\n")
    m.clear_fault(); time.sleep(0.1); m.enable(); time.sleep(0.1)

    before = round_trip(m, a.kp, "영점 전")

    print("\n  영점 재설정 (명령 4, CMD_SET_ZERO) — 힘을 끄고 보냅니다")
    for _ in range(6):
      m.disable(); time.sleep(0.03)
    m._cmd(0xFE)
    time.sleep(0.6)
    s = m.read(tries=10)
    print(f"  영점 직후 보고각 {s.position_deg:+9.2f}도 "
          f"({'0 근처 = 영점이 먹음' if abs(s.position_deg) < 5 else '★ 안 먹음'})\n")
    m.clear_fault(); time.sleep(0.1); m.enable(); time.sleep(0.1)

    after = round_trip(m, a.kp, "영점 후")

    print(f"\n  → 안쪽 오차 {before:+.2f} → {after:+.2f} 라디안")
    if abs(after) < 0.3 * abs(before):
      print("  → **영점 재설정이 고칩니다.** 실물에서도 쓰기 전에 영점을 잡아야 합니다.")
    else:
      print("  → 영점으로는 안 고쳐집니다. 다른 원인입니다.")
    s = m.read(tries=10)
    print(f"\n끝 · {s.position_deg:.2f}도 · {s.temp_c:.0f}도 · 고장 {s.fault}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
