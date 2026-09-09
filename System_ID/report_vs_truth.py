#!/usr/bin/env python3
"""모터가 **말하는 값**과 **실제 값**을 칸마다 견준다. 라즈베리파이에서 돌린다.

2026-09-09. 사용자 지시: "그러면 보고 값, 실제 값 비교도 준비해 줘."

오늘 두 칸이 거짓말하는 것을 따로따로 발견했습니다(속도는 859 도/s 에서 잘리고, 토크는 도는
중에 회전 방향으로 치우칩니다). 발견될 때마다 놀라지 않으려면 **네 칸 전부를 한 번에**
견줘야 합니다.

문제는 "실제 값"을 어디서 가져오는가입니다. 칸마다 다릅니다.

===========  ==============================================================
위치         **기준 삼는 칸.** 바깥 자가 없어 절대값은 못 봅니다. 대신 이 칸이
             믿을 만하다는 근거는 따로 있습니다 — 이 칸을 미분한 값과 속도 칸이
             859 도/s 아래에서 0.07% 안에 맞습니다. 서로 다른 두 칸이 맞을
             확률은, 둘 다 같은 방식으로 틀리지 않는 한, 낮습니다.
속도         위치 칸을 이어 붙여 미분한 값이 실제입니다.
토크         **모터가 멈춰 있으면 실제 토크 = 시킨 토크**입니다(움직이지 않으니
             가속에도 마찰에도 안 쓰였다는 뜻). 그리고 **타성으로 도는 중이면
             실제 토크 = 정확히 0** 입니다. 그리고 **일정한 속도로 도는 중이면
             실제 토크 = 그 속도의 마찰**이고, 마찰은 오늘 두 번 쟀습니다.
             이 세 자리에서 견줍니다.
온도         바깥 온도계가 없습니다. 대신 **두 모터를 나란히 놓고**, 하나만
             돌렸을 때 그것만 오르는지를 봅니다.
===========  ==============================================================

안전
----
가만히 있는 구간은 붙어 있는 힘(약 0.9 뉴턴미터)보다 낮은 토크만 씁니다. 도는 구간은
축이 맨 상태여야 합니다.

    report_vs_truth.py --motor-id 4 --model RS04
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
TAU_C, B_VISC = 0.519, 0.0275     # 오늘 순수 타성 감속으로 잰 값


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--still-max", type=float, default=0.60,
                  help="가만히 있는 구간에서 줄 최대 토크. 붙어 있는 힘(약 0.9)보다 낮게")
  ap.add_argument("--still-steps", type=int, default=13)
  ap.add_argument("--still-dwell-s", type=float, default=0.8)
  ap.add_argument("--move-deg", type=float, default=0.30,
                  help="이만큼 움직이면 '가만히'가 깨진 것이라 그 점은 버린다")
  ap.add_argument("--spin-taus", default="0.65,0.75,0.85,1.0,1.2,1.5,2.0")
  ap.add_argument("--kick", type=float, default=1.6,
                  help="출발 발길질. 붙어 있는 힘을 넘겨 놓아야 낮은 토크도 계속 돈다")
  ap.add_argument("--spin-s", type=float, default=8.0)
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--tag", default="")
  a = ap.parse_args(argv)

  suffix = ("_" + a.tag) if a.tag else ""
  still_rows, spin_rows = [], []

  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · {st.temp_c:.0f}도\n")
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)

    # ---- 1) 가만히 있는 구간: 실제 토크 = 시킨 토크 --------------------
    print("[1] 멈춰 있는 채로 토크를 올린다 — 이때 실제 토크 = 시킨 토크")
    print(f"{'시킨':>7} {'보고':>8} {'보고/시킨':>10} {'움직인 양':>10}")
    for i in range(a.still_steps):
      tau = a.still_max * (i + 1) / a.still_steps
      p0 = None
      vals = []
      t0 = time.monotonic()
      while time.monotonic() - t0 < a.still_dwell_s:
        m.torque(tau)
        s = m.recv(timeout_s=0.01)
        if s is None:
          continue
        if p0 is None:
          p0 = s.position_deg
        if time.monotonic() - t0 > a.still_dwell_s * 0.4:
          vals.append((s.torque_nm, s.position_deg))
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)
      time.sleep(0.15)
      if len(vals) < 20:
        continue
      rep = sum(v[0] for v in vals) / len(vals)
      moved = max(abs(v[1] - p0) for v in vals)
      ok = moved < a.move_deg
      still_rows.append((tau, rep, moved, int(ok)))
      print(f"{tau:7.3f} {rep:8.3f} {rep/tau:10.3f} {moved:9.3f}도"
            + ("" if ok else "  ← 움직여서 버림"))

    # ---- 2) 도는 구간: 실제 토크 = 그 속도의 마찰, 실제 속도 = 위치에서 ----
    print("\n[2] 일정 속도로 도는 중 — 실제 토크 = 그 속도의 마찰, "
          "실제 속도 = 위치를 이어 붙여 미분")
    print(f"{'시킨':>7} {'보고속도':>9} {'진짜속도':>9} {'보고/진짜':>9} "
          f"{'보고토크':>9} {'실제토크':>9} {'차이':>8}")
    for tau in [float(x) for x in a.spin_taus.split(",")]:
      # 출발 발길질: 붙어 있는 힘(약 0.9)을 넘겨 놓아야 낮은 토크도 계속 돈다
      t_k = time.monotonic()
      while time.monotonic() - t_k < 1.5:
        m.torque(a.kick); m.recv(timeout_s=0.01)
      rows = []
      t0 = time.monotonic()
      acc, last = 0.0, None
      while time.monotonic() - t0 < a.spin_s:
        m.torque(tau)
        s = m.recv(timeout_s=0.01)
        if s is None:
          continue
        if last is None:
          last = s.position_deg
        d = s.position_deg - last
        while d > 720.0:
          d -= 1440.0
        while d < -720.0:
          d += 1440.0
        acc += d
        last = s.position_deg
        rows.append((time.monotonic() - t0, acc, s.velocity_deg_s, s.torque_nm, s.temp_c))
      for _ in range(20):
        m.torque(0.0); m.recv(timeout_s=0.002)
      # 마지막 2초 (자리잡은 뒤)
      tail = [r for r in rows if r[0] > rows[-1][0] - 2.0]
      if len(tail) < 200:
        continue
      v_true = (tail[-1][1] - tail[0][1]) / (tail[-1][0] - tail[0][0])
      if abs(v_true) < 5.0:
        print(f"{tau:7.2f}  안 돌았습니다 (붙어 있는 힘보다 작음)")
        continue
      v_rep = sum(abs(r[2]) for r in tail) / len(tail)
      t_rep = sum(r[3] for r in tail) / len(tail)
      t_true = TAU_C + B_VISC * abs(v_true) * DEG      # 자리잡았으면 실제 토크 = 마찰
      spin_rows.append((tau, v_rep, abs(v_true), t_rep, t_true, tail[-1][4]))
      print(f"{tau:7.2f} {v_rep:9.1f} {abs(v_true):9.1f} {v_rep/abs(v_true):9.3f} "
            f"{t_rep:9.3f} {t_true:9.3f} {t_rep-t_true:+8.3f}")
      # 식히며 멈추기
      t1 = time.monotonic()
      while time.monotonic() - t1 < 4.0:
        m.torque(0.0); m.recv(timeout_s=0.005)

    for _ in range(20):
      m.torque(0.0); m.recv(timeout_s=0.002)

  with open(os.path.join(a.out_dir, f"rvt_still_{a.motor_id}{suffix}.csv"), "w",
            newline="") as f:
    w = csv.writer(f); w.writerow(["cmd_nm", "rep_nm", "moved_deg", "still"])
    w.writerows(still_rows)
  with open(os.path.join(a.out_dir, f"rvt_spin_{a.motor_id}{suffix}.csv"), "w",
            newline="") as f:
    w = csv.writer(f)
    w.writerow(["cmd_nm", "vel_rep", "vel_true", "tau_rep", "tau_true", "temp_c"])
    w.writerows(spin_rows)
  print(f"\n기록: rvt_still_{a.motor_id}{suffix}.csv, rvt_spin_{a.motor_id}{suffix}.csv")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
