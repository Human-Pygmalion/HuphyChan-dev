#!/usr/bin/env python3
"""이 모터가 실제로 낼 수 있는 최고 속도를 찾는다. 라즈베리파이에서 돌린다.

2026-09-09. 웜업을 600 도/s 로 했는데 사용자가 "이게 최고속도인가"라고 물었습니다. 아닙니다 —
사양상 무부하는 대략 1140 도/s 입니다. 그런데 **사양의 무부하 속도는 토크가 0 일 때의 값**이고,
실제로는 마찰을 이기는 만큼 토크를 써야 하므로 그보다 낮은 데서 멈춥니다. 그 자리를 찾습니다.

왜 그냥 큰 값을 시키면 안 되는가
--------------------------------
속도 명령은 모터 안에서 ``토크 = kd x (시킨 속도 - 지금 속도)`` 로 바뀝니다. 모터가 못 따라
오면 그 차이가 계속 벌어지고, **토크가 그만큼 커집니다.** 높은 속도에서 큰 토크는 곧 큰
전력이고, 이 전원은 74 와트에서 이미 주저앉은 적이 있습니다(2026-09-09 저전압 고장).

그래서 이렇게 합니다
--------------------
명령을 조금씩 올리되, **실제 속도가 명령을 따라오지 못하기 시작하면 거기서 올리기를 멈춥니다.**
따라오는 한 계속 올리고, 벌어지기 시작하면 그 자리가 상한입니다. 토크도 따로 묶습니다.

    speed_probe.py --motor-id 4 --model RS04
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--start", type=float, default=400.0, help="여기서 시작 도/s")
  ap.add_argument("--step", type=float, default=50.0, help="한 계단 도/s")
  ap.add_argument("--dwell-s", type=float, default=4.0, help="한 계단에 머무는 시간")
  ap.add_argument("--kd", type=float, default=1.5)
  ap.add_argument("--gain-stop", type=float, default=0.4,
                  help="""명령을 한 계단 올렸을 때 **실제 속도가 그 계단의 이 비율만큼도 안
올라가면** 상한으로 본다.

  앞서 '시킨 값과 실제의 차이'로 판정했더니 400 도/s 에서 바로 걸렸는데, 그 차이(69 도/s)는
  포화가 아니라 **비례 제어의 당연한 처짐**이었다 - 마찰을 이길 토크 0.65 를 내려면 반드시
  0.65/kd 만큼 오차가 있어야 한다. 처짐은 속도와 거의 무관하게 일정하므로, 명령을 올릴 때
  실제도 같이 올라가는 한 아직 여유가 있는 것이다.""")
  ap.add_argument("--torque-stop", type=float, default=3.0)
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--max-cmd", type=float, default=1300.0)
  ap.add_argument("--decel-s", type=float, default=8.0)
  ap.add_argument("--out", default=os.path.expanduser("~/System_ID/data/speed_probe.csv"))
  a = ap.parse_args(argv)

  rows, stop, best = [], None, 0.0
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · {st.temp_c:.0f}도 · kd={a.kd} "
          f"(못 따라오면 토크가 치솟지 않게 작게 둠)\n")
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)

    cmd = a.start
    prev_v = None
    t_all = time.monotonic()
    try:
      while cmd <= a.max_cmd:
        t0 = time.monotonic()
        vs, ts, tc = [], [], 0.0
        while time.monotonic() - t0 < a.dwell_s:
          m.send(position_deg=0.0, velocity_deg_s=cmd, kp=0.0, kd=a.kd)
          s = m.recv(timeout_s=0.01)
          if s is None:
            continue
          rows.append((time.monotonic() - t_all, cmd, s.position_deg, s.velocity_deg_s,
                       s.torque_nm, s.temp_c))
          if time.monotonic() - t0 > a.dwell_s * 0.5:   # 뒤쪽 절반만 (자리잡은 뒤)
            vs.append(s.velocity_deg_s); ts.append(s.torque_nm)
          tc = s.temp_c
          if s.fault:
            stop = "모터가 고장을 보고함"
            break
        if stop or not vs:
          break
        v = sum(vs) / len(vs)
        tq = sum(ts) / len(ts)
        lag = cmd - v
        best = max(best, v)
        print(f"  시킨 {cmd:6.0f} → 실제 {v:6.1f} 도/s (차이 {lag:5.1f}) · "
              f"토크 {tq:5.2f} 뉴턴미터 · 전력 약 {abs(tq*v*3.14159/180):5.1f} 와트 · {tc:.0f}도"
              + (f" · 앞 계단 대비 +{v - prev_v:5.1f}" if prev_v is not None else ""))
        if prev_v is not None and (v - prev_v) < a.gain_stop * a.step:
          stop = (f"명령을 {a.step:.0f} 올렸는데 실제는 {v - prev_v:.0f} 만 올라감 = 포화")
          break
        prev_v = v
        if abs(tq) > a.torque_stop:
          stop = f"토크 {tq:.1f} 뉴턴미터 초과"
          break
        if tc >= a.temp_stop:
          stop = f"온도 {tc:.0f}도"
          break
        cmd += a.step
      else:
        stop = "명령 상한까지 다 따라옴"
    except KeyboardInterrupt:
      stop = "사람이 중단"
    finally:
      # 천천히 내린다. 급제동하면 모터가 발전기가 되어 과전압이 난다.
      last = best if best > 0 else a.start
      td = time.monotonic()
      while time.monotonic() - td < a.decel_s:
        f = 1.0 - (time.monotonic() - td) / a.decel_s
        m.send(position_deg=0.0, velocity_deg_s=last * max(0.0, f), kp=0.0, kd=a.kd)
        m.recv(timeout_s=0.005)
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)

  print(f"\n끝난 이유: {stop}")
  print(f"실제로 낸 최고 속도: {best:.0f} 도/s")
  if rows:
    with open(a.out, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(["t_s", "cmd_vel_deg_s", "pos_deg", "vel_deg_s", "tau_est_nm", "temp_c"])
      w.writerows(rows)
    print(f"기록: {a.out} ({len(rows)}줄)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
