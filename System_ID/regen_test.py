#!/usr/bin/env python3
"""회생 전력을 전원이 견디는지 시험한다. 라즈베리파이에서 돌린다.

2026-09-09. 사용자가 **과전압 흡수·배출 장치**를 붙이고 "있고 없고 응답 차이를 확인해
달라"고 했습니다.

무엇이 문제였나 (장치 없을 때, 같은 날 낮에 실측)
------------------------------------------------
속도를 15초마다 뒤집으며 데우려 했더니 **17초 만에 모터가 고장을 보고**했습니다.
고장값 4 = 비트2 = **저전압**(매뉴얼 15쪽, 기능코드 0x3022), 경고는 비트3 = 과전압.

기록을 보면 방향이 바뀌는 순간 **465 도/s 에서 제동 토크 −9.17 뉴턴미터**였습니다. 곧
**약 74 와트를 전원 쪽으로 되민** 것입니다. 모터가 감속할 때는 발전기가 되어 전압을 밀어
올리고(과전압), 곧바로 반대로 가속할 때는 전원이 주저앉습니다(저전압).

이 시험이 하는 일
-----------------
**그때와 똑같은 조건부터** 시작합니다(600 도/s, 15초마다 전환). 그게 버티면 장치가 일한
것입니다. 그 다음 조건을 단계적으로 세게 합니다 — 더 자주 뒤집고, 더 빠르게 — 새 한계가
어디인지 찾습니다.

매 순간 기록합니다: 시킨 속도, 진짜 속도(위치에서), 토크, **되민 전력**(토크 x 속도가
음수인 구간), 온도, 고장·경고 깃발.

안전
----
* **사람이 벤치에서 손을 뗀 뒤에** 시작해야 합니다. 축이 맨 상태로 빠르게 돕니다.
* 온도 상한, 고장 즉시 중단, 끝낼 때는 천천히 감속.
* 단계는 약한 것부터. 고장이 나면 그 자리에서 멈추고 더 세게 가지 않습니다.

    regen_test.py --tag with_device
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402

DEG = math.pi / 180.0


class TrueSpeed:
  """보고 속도는 859 도/s 에서 잘리므로 위치를 이어 붙여 쓴다."""

  def __init__(self, window_s: float = 0.05):
    self.w = window_s
    self.acc = 0.0
    self.last = None
    self.trail = []

  def push(self, t, pos):
    if self.last is None:
      self.last = pos
    d = pos - self.last
    while d > 720.0:
      d -= 1440.0
    while d < -720.0:
      d += 1440.0
    self.acc += d
    self.last = pos
    self.trail.append((t, self.acc))
    while len(self.trail) > 2 and t - self.trail[0][0] > self.w:
      self.trail.pop(0)
    if len(self.trail) < 8 or self.trail[-1][0] <= self.trail[0][0]:
      return 0.0
    return (self.trail[-1][1] - self.trail[0][1]) / (self.trail[-1][0] - self.trail[0][0])


def stage(m: Motor, speed: float, flip_s: float, seconds: float, kd: float,
          ramp_s: float, temp_stop: float, out_path: str):
  rows = []
  ts = TrueSpeed()
  why = "완주"
  t0 = time.monotonic()
  while True:
    t = time.monotonic() - t0
    if t >= seconds:
      break
    sign = 1.0 if int(t / flip_s) % 2 == 0 else -1.0
    scale = min(1.0, t / ramp_s) if ramp_s > 0 else 1.0
    tgt = sign * speed * scale
    m.send(position_deg=0.0, velocity_deg_s=tgt, kp=0.0, kd=kd)
    s = m.recv(timeout_s=0.01)
    if s is None:
      continue
    v = ts.push(t, s.position_deg)
    p = s.torque_nm * v * DEG          # 음수 = 전원 쪽으로 되밀고 있음
    rows.append((t, tgt, s.position_deg, s.velocity_deg_s, v, s.torque_nm, p,
                 s.temp_c, int(s.fault), int(s.warn)))
    if s.fault:
      why = "★ 모터가 고장을 보고함"
      break
    if s.temp_c >= temp_stop:
      why = f"온도 {s.temp_c:.0f}도"
      break
  # 천천히 세운다 (급제동은 그 자체가 회생이다)
  last = rows[-1][4] if rows else 0.0
  td = time.monotonic()
  while time.monotonic() - td < 5.0:
    f = max(0.0, 1.0 - (time.monotonic() - td) / 5.0)
    m.send(position_deg=0.0, velocity_deg_s=last * f, kp=0.0, kd=kd)
    m.recv(timeout_s=0.005)
  for _ in range(25):
    m.torque(0.0); m.recv(timeout_s=0.002)

  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "cmd_vel", "pos_deg", "vel_rep", "vel_true", "tau_rep_nm",
                "power_w", "temp_c", "fault", "warn"])
    w.writerows(rows)
  return rows, why


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--stages", default="600:15,600:8,600:4,800:4,800:2",
                  help="속도:전환주기 목록. 약한 것부터")
  ap.add_argument("--dwell-s", type=float, default=60.0)
  ap.add_argument("--kd", type=float, default=2.0)
  ap.add_argument("--ramp-s", type=float, default=3.0)
  ap.add_argument("--temp-stop", type=float, default=47.0)
  ap.add_argument("--cool-to", type=float, default=42.0)
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--tag", default="with_device")
  a = ap.parse_args(argv)

  stages = []
  for s in a.stages.split(","):
    sp, fl = s.split(":")
    stages.append((float(sp), float(fl)))

  print("장치 없을 때(같은 날 낮): 600 도/s · 15초 전환 → **17초 만에 저전압 고장**")
  print(f"이번: {a.tag}\n")
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · {st.temp_c:.0f}도")
    m.clear_fault(); time.sleep(0.15)
    m.enable(); time.sleep(0.15)
    try:
      for sp, fl in stages:
        # 식히기
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120.0:
          m.torque(0.0)
          s = m.recv(timeout_s=0.01)
          if s is not None and s.temp_c <= a.cool_to and time.monotonic() - t0 > 1.0:
            break
        name = f"regen_{int(sp)}_{fl:g}_{a.tag}.csv"
        rows, why = stage(m, sp, fl, a.dwell_s, a.kd, a.ramp_s, a.temp_stop,
                          os.path.join(a.out_dir, name))
        if not rows:
          print(f"  {sp:.0f} 도/s · {fl:g}초 전환: 자료 없음")
          continue
        brake = [r for r in rows if r[6] < 0]
        pk = min((r[6] for r in brake), default=0.0)
        pk_tau = min((r[5] for r in rows), default=0.0)
        warn_t = next((r[0] for r in rows if r[9]), None)
        print(f"  {sp:4.0f} 도/s · {fl:g}초 전환 · {rows[-1][0]:5.1f}초 · "
              f"되민 전력 최대 {abs(pk):6.1f} 와트 · 제동 토크 최대 {abs(pk_tau):5.2f} 뉴턴미터 · "
              f"최고 속도 {max(abs(r[4]) for r in rows):6.0f} 도/s · {rows[-1][7]:.0f}도 · "
              f"경고 {'없음' if warn_t is None else f'{warn_t:.1f}초'} · {why}")
        if "고장" in why:
          print("    -> 여기서 멈춥니다. 더 세게 가지 않습니다.")
          break
    except KeyboardInterrupt:
      print("사람이 중단")
    finally:
      for _ in range(25):
        m.torque(0.0); m.recv(timeout_s=0.002)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
