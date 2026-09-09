#!/usr/bin/env python3
"""마찰을 잰다. **로봇 위에서 돌린다.**

2026-09-09. 어댑터가 초당 224회에서 막혀 있어 **회전자 관성**은 아직 못 잽니다(빠른 흔들림이
필요함). 그런데 **마찰은 느린 현상이라 지금 속도로도 정확히 잴 수 있습니다.** 셋 중 둘을
지금 확정할 수 있다는 뜻입니다.

두 가지를 잽니다
----------------
``breakaway`` — **정지 마찰**. 토크를 0부터 아주 천천히 올리다가 **움직이기 시작하는 순간**의
토크를 기록합니다. 그 값이 곧 붙어 있는 힘입니다. 움직임이 감지되면 그 자리에서 즉시 멈추므로
거의 움직이지 않습니다.

  덤으로 **보고 토크의 눈금**도 여기서 나옵니다. 움직이기 직전에는 모터가 멈춰 있으므로
  실제 토크 = 시킨 토크이고, 그때 모터가 뭐라고 보고하는지 보면 눈금이 맞는지 알 수 있습니다.
  (앞선 흔들기 자료에서 보고 토크가 시킨 값의 0.27~0.37배로 나왔는데, 그건 15 밀리초 지연과
  가속이 섞인 값이라 눈금이라고 부를 수 없었습니다.)

``viscous`` — **속도에 비례하는 마찰**. 속도를 여러 값으로 붙잡아 두고 각각 필요한 토크를
읽습니다. 토크 = 정지마찰 + 계수 × 속도 이므로, 직선으로 맞추면 **세로 절편이 정지 마찰,
기울기가 점성 계수**입니다.

왜 이건 224 Hz 로도 되는가
--------------------------
둘 다 **정상 상태**를 재는 일입니다. 흔드는 속도가 아니라 머무는 값을 보므로, 초당 몇 번
주고받느냐가 결과를 바꾸지 않습니다. 관성은 반대입니다 — 빠른 방향 전환에서만 드러나므로
어댑터를 고쳐야 합니다.

안전
----
* ``breakaway`` 는 움직임이 감지되면 **그 즉시** 토크를 0으로 놓습니다. 움직이는 양이 몇 도를
  넘지 않습니다.
* ``viscous`` 는 모터가 **계속 돕니다**. 출력축에 아무것도 안 달려 있어야 합니다(카메라로
  확인했음). 총 회전량과 시간을 모두 제한하고, 속도가 정한 값을 넘으면 중단합니다.
* 어느 쪽이든 끝나면 토크 0 → 토크 끄기.

    friction_id.py breakaway --motor-id 4 --model RS04 --max-torque 1.5 --rate 0.15
    friction_id.py viscous   --motor-id 4 --model RS04
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
  # 라즈베리파이: 파이썬 기본 기능만 쓰는 구동부. 아무것도 설치할 필요가 없다.
  from rs_direct_socket import Motor  # noqa: E402
except ImportError:                    # 로봇 가상머신: python-can 이 이미 깔려 있다
  from rs_direct import Motor  # noqa: E402

MOVE_DEG = 0.20
"""이만큼 위치가 바뀌면 '움직이기 시작했다'고 본다.

**속도로 판정하면 안 된다** (2026-09-09 실측). 토크를 0 으로 두고 가만히 둔 상태에서도
모터가 보고하는 속도가 최대 6.5 도/s 까지 튄다(흩어짐 1.98). 그래서 속도 5 도/s 를 문턱으로
썼더니 **0.000 뉴턴미터에서 '움직였다'** 고 잡혔고, 정작 위치는 0.00도 그대로였다.

같은 상태에서 **위치는 0.022도밖에 안 흔들린다.** 0.20도는 그 아홉 배라 잡음에 안 걸리고,
움직임이 시작되면 곧바로 넘는다. 판정이 늦어 그만큼 더 밀리는 양은 0.2도로 묶인다."""

MOVE_DEG_S = 30.0
"""속도는 **보조**로만 쓴다. 잡음 최대치(6.5)의 네 배 이상이라 오검출하지 않는다."""

SETTLE_S = 0.8


def run_breakaway(m, a, log):
  """토크를 천천히 올리며 움직이기 시작하는 지점을 찾는다. 양방향으로 여러 번."""
  results = []
  for trial in range(a.trials):
    for sign in (+1, -1):
      # 자리가 멎을 때까지 기다린다. 앞 시행에서 밀린 관절이 천천히 되돌아오는데, 그걸
      # 그대로 시작하면 **0 뉴턴미터에서 '움직였다'** 고 잡힌다(2026-09-09 실제로 그랬음:
      # - 방향 네 번 모두 0.000 에서 걸리고 위치는 -0.22~-0.31도 흘렀음).
      settled = None
      t_wait = time.monotonic()
      while time.monotonic() - t_wait < 6.0:
        a1 = m.read(tries=3)
        time.sleep(0.4)
        a2 = m.read(tries=3)
        if a1 and a2 and abs(a2.position_deg - a1.position_deg) < 0.05:
          settled = a2
          break
      st = settled or m.read()
      if st is None:
        raise RuntimeError("모터가 응답하지 않습니다")
      if settled is None:
        print("    (자리가 6초 동안 안 멎었습니다 - 그대로 진행합니다)")

      # 톱니 사이 놀음을 **올릴 방향과 같은 쪽으로** 미리 없앤다.
      # 안 그러면 경사를 올릴 때 처음 0.2도가 놀음을 지나가는 데 쓰이고, 그걸 '움직이기
      # 시작했다'고 잡아 버린다 - 2026-09-09 에 - 방향이 매번 0.000 뉴턴미터에서 걸린 이유다.
      # 미는 힘은 정지 마찰보다 작아야 넘어가지 않는다.
      if a.preseat > 0:
        base = st.position_deg
        t_ps = time.monotonic()
        while time.monotonic() - t_ps < a.preseat_s:
          m.torque(sign * a.preseat)
          p = m.recv(timeout_s=0.005)
          if p and abs(p.position_deg - base) > a.bound_deg:
            break                      # 예상보다 많이 밀리면 그만둔다
        for _ in range(8):
          m.torque(0.0); m.recv(timeout_s=0.002)
        time.sleep(0.5)
        st = m.read(tries=6) or st

      q0 = st.position_deg
      tau = 0.0
      t0 = time.monotonic()
      found = None
      while tau < a.max_torque:
        t = time.monotonic() - t0
        tau = a.rate * t                      # 초당 a.rate 뉴턴미터씩 올린다
        m.torque(sign * tau)
        st = m.recv(timeout_s=0.01)
        if st is None:
          continue
        log.append((t, sign * tau, st.position_deg, st.velocity_deg_s, st.torque_nm, st.temp_c))
        moved = abs(st.position_deg - q0)
        if moved > a.move_deg or abs(st.velocity_deg_s) > MOVE_DEG_S or moved > a.bound_deg:
          found = (tau, abs(st.torque_nm), st.position_deg - q0)
          break
      # 무슨 일이 있어도 바로 힘을 뺀다
      for _ in range(10):
        m.torque(0.0)
        m.recv(timeout_s=0.002)
      time.sleep(0.3)
      if found is None:
        print(f"  {trial+1}회 {'+' if sign>0 else '-'}: {a.max_torque} 뉴턴미터까지 안 움직임")
      else:
        results.append(found)
        print(f"  {trial+1}회 {'+' if sign>0 else '-'}: 움직이기 시작 "
              f"**{found[0]:.3f} 뉴턴미터** · 그때 모터 보고 {found[1]:.3f} · "
              f"움직인 양 {found[2]:+.2f}도")
  return results


def run_viscous(m, a, log):
  """속도를 붙잡아 두고 필요한 토크를 읽는다. 토크 = 정지마찰 + 계수 x 속도."""
  out = []
  total_deg = 0.0
  st = m.read()
  last = st.position_deg if st else 0.0
  for v in a.speeds:
    for sign in (+1, -1):
      target = sign * v
      taus, vels = [], []
      t0 = time.monotonic()
      while time.monotonic() - t0 < a.dwell_s:
        # kp 는 0, kd 만 준다 = 속도를 붙잡는 명령. 위치는 신경쓰지 않는다.
        m.send(position_deg=0.0, velocity_deg_s=target, kp=0.0, kd=a.kd)
        st = m.recv(timeout_s=0.01)
        if st is None:
          continue
        d = abs(st.position_deg - last)
        total_deg += min(d, 30.0)        # 한 바퀴 넘어갈 때의 튀는 값은 잘라 센다
        last = st.position_deg
        log.append((time.monotonic()-t0, target, st.position_deg, st.velocity_deg_s,
                    st.torque_nm, st.temp_c))
        if time.monotonic() - t0 > a.dwell_s * 0.4:   # 앞 40% 는 아직 붙잡히는 중
          taus.append(st.torque_nm)
          vels.append(st.velocity_deg_s)
        if st.temp_c >= a.temp_stop:
          raise RuntimeError(f"온도 {st.temp_c} 도 도달")
        if total_deg > a.spin_budget_deg:
          raise RuntimeError(f"총 회전량 {total_deg:.0f}도 초과 - 정한 예산을 넘었습니다")
      # 다음 설정으로 넘어가기 전에 **천천히** 세운다. 곧바로 반대 방향을 시키면 모터가
      # 급제동하며 발전기가 되어 전압을 되민다 - 2026-09-09 웜업에서 그렇게 저전압 고장이
      # 났다(속도 465 도/s 에서 제동 토크 -9.17 뉴턴미터 = 약 74 와트를 전원으로 되밀었음).
      t_d = time.monotonic()
      while time.monotonic() - t_d < 1.5:
        frac = 1.0 - (time.monotonic() - t_d) / 1.5
        m.send(position_deg=0.0, velocity_deg_s=target * max(0.0, frac), kp=0.0, kd=a.kd)
        m.recv(timeout_s=0.005)
      for _ in range(6):
        m.torque(0.0); m.recv(timeout_s=0.002)
      if taus:
        import statistics as s
        vm = s.mean(vels); tm = s.mean(taus)
        out.append((vm, tm))
        print(f"  목표 {target:+7.1f} 도/s -> 실제 {vm:+7.1f} 도/s, 필요한 토크 {tm:+.4f} 뉴턴미터")
      time.sleep(0.25)
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("mode", choices=["breakaway", "viscous"])
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--channel", default="can0")
  # breakaway
  ap.add_argument("--max-torque", type=float, default=1.5)
  ap.add_argument("--rate", type=float, default=0.15, help="초당 몇 뉴턴미터씩 올릴지")
  ap.add_argument("--trials", type=int, default=3)
  ap.add_argument("--preseat", type=float, default=0.30,
                  help="경사를 올리기 전에 같은 방향으로 이만큼 밀어 톱니 놀음을 없앤다. "
                       "정지 마찰보다 작아야 한다 (0 이면 안 함)")
  ap.add_argument("--preseat-s", type=float, default=0.6)
  ap.add_argument("--bound-deg", type=float, default=8.0)
  ap.add_argument("--move-deg", type=float, default=MOVE_DEG,
                  help="이만큼 움직이면 '시작했다'고 본다. 기본 0.2도는 잡음(0.022도)의 아홉 "
                       "배지만, 톱니 사이 놀음(백래시)이 그보다 크면 놀음을 지나가는 것을 "
                       "마찰로 오해한다 - 그때는 크게 준다")
  # viscous
  ap.add_argument("--speeds", type=float, nargs="+",
                  default=[20, 40, 80, 140, 220])
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--dwell-s", type=float, default=1.2)
  ap.add_argument("--spin-budget-deg", type=float, default=4000.0,
                  help="이만큼 돌면 중단. 축에 아무것도 안 달려 있어야 한다")
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--out", default=None)
  a = ap.parse_args(argv)

  log = []
  with Motor(channel=a.channel, motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · 시작 {st.position_deg:.2f}도 · "
          f"{st.temp_c:.0f}도 · 고장 {st.fault}")
    if st.fault:
      m.clear_fault(); time.sleep(0.05)
    m.enable(); time.sleep(0.05)
    try:
      if a.mode == "breakaway":
        print(f"\n토크를 초당 {a.rate} 뉴턴미터씩 올리며 움직이는 순간을 찾습니다 "
              f"(최대 {a.max_torque}):")
        res = run_breakaway(m, a, log)
        if res:
          import statistics as s
          brk = [r[0] for r in res]; rep = [r[1] for r in res]
          print(f"\n★ 정지 마찰 = {s.mean(brk):.3f} 뉴턴미터 "
                f"(중앙값 {s.median(brk):.3f}, 흩어짐 {s.pstdev(brk):.3f}, {len(brk)}회)")
          scale = s.mean([r/b for b, r in zip(brk, rep) if b > 0.05])
          print(f"★ 모터 보고 토크 / 시킨 토크 = {scale:.3f} "
                f"(1.0 이면 눈금이 맞는 것)")
          print(f"   비교: 지금 화면 속 모형이 쓰는 값 0.2695 뉴턴미터")
      else:
        print(f"\n속도를 붙잡아 두고 필요한 토크를 읽습니다 (kd={a.kd}):")
        pts = run_viscous(m, a, log)
        if len(pts) >= 3:
          import numpy as np
          v = np.array([abs(p[0]) for p in pts]); t = np.array([abs(p[1]) for p in pts])
          A = np.vstack([v, np.ones_like(v)]).T
          slope, icept = np.linalg.lstsq(A, t, rcond=None)[0]
          pred = A @ np.array([slope, icept])
          r2 = 1 - ((t-pred)**2).sum()/((t-t.mean())**2).sum()
          # 도/s 를 라디안/s 로 바꿔 모형과 같은 단위로
          slope_rad = slope * 180.0 / 3.141592653589793
          print(f"\n★ 직선 맞춤: 토크 = {icept:.4f} + {slope:.6f} x 속도(도/s)   결정계수 {r2:.4f}")
          print(f"★ 정지 마찰(절편) = {icept:.4f} 뉴턴미터   "
                f"(모형 값 0.2695)")
          print(f"★ 점성 계수 = {slope_rad:.5f} 뉴턴미터/(라디안/s)   "
                f"(모형 값 0.0095)")
    finally:
      for _ in range(10):
        m.torque(0.0); m.recv(timeout_s=0.002)

  if log:
    out = a.out or f"/home/syaro/System_ID/data/friction_{a.mode}_{a.motor_id}_{time.strftime('%H%M%S')}.csv"
    with open(out, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(["t_s", "cmd", "pos_deg", "vel_deg_s", "tau_est_nm", "temp_c"])
      w.writerows(log)
    print(f"\n기록: {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
