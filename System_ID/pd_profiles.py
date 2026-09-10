#!/usr/bin/env python3
"""위치 PD 로 계단·처프(sine sweep) 응답을 받는다. 라즈베리파이에서 돌린다.

2026-09-09. 사용자 지시: "실제로 모터를 그 파라미터로 만들어서, 똑같은 지령 넣어서
(Kp/Kd/armature,damping,...) 응답 비교해 줘. 계단 위치 입력(위아래 역전), 처프 패턴으로."

모터 안에서 도는 식은 이것입니다 (우리가 kp, kd 를 실어 보냅니다):

    tau = kp x (목표각 - 지금각) + kd x (목표속도 - 지금속도)

시뮬레이터에서도 **똑같은 식을 똑같은 이득으로** 돌려 겹쳐 봅니다. 다른 것은 관성·마찰·점성
뿐이므로, 두 응답이 갈리면 그 셋 때문입니다.

★★ 위치 명령의 함정 (2026-09-09 에 실제로 당했습니다)
------------------------------------------------------
**모터 내부 위치는 여러 바퀴를 누적하는데, 보고되는 칸은 ±720도에서 되감깁니다.** 오늘
모터를 많이 돌려 보고 값이 −626.69도(되감김 가장자리)일 때 "보고 위치 + 1도"를 목표로
줬더니, 모터가 보는 오차는 1도가 아니라 **721도**였고 500~860도를 돌아 버렸습니다.

그래서 이 프로그램은 두 겹으로 막습니다.

1. **먼저 원점 근처로 데려옵니다** (``home``). 이때는 **속도 명령**만 씁니다 — 속도 명령은
   부호화 범위 안에 갇혀 있어 폭주할 수가 없습니다.
2. **모든 위치 명령에 하드 경계**를 겁니다. 목표에서 정해진 각도 이상 벗어나면 그 자리에서
   토크를 0 으로 놓습니다. 초당 3300번 보므로 폭주해도 1도 안쪽에서 잡힙니다.

    pd_profiles.py gain  --motor-id 4 --model RS04
    pd_profiles.py step  --motor-id 4 --model RS04 --kp 20 --kd 1.0 --amp 8
    pd_profiles.py chirp --motor-id 4 --model RS04 --kp 20 --kd 1.0 --amp 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402

DEG = math.pi / 180.0


def home(m: Motor, target_deg: float = 0.0, tol_deg: float = 3.0, kp: float = 25.0,
         kd: float = 1.2, tau_max: float = 4.0, timeout_s: float = 20.0,
         runaway_deg: float = 200.0) -> float:
  """정해진 자리로 데려온다. **호스트에서 PD 를 계산해 순수 토크로** 보낸다.

  ★ 2026-09-10 정정 — 앞의 판은 모터에 **속도 명령**(kp=0, kd만)을 보냈는데, 그게 전혀 안
  들었습니다. 훑기 실험 로그에서 목표가 −60·−30·0 인데 결과가 전부 −25 근처로 나와서
  알아챘습니다. 모터의 **내부 위치 되먹임이 우리 명령을 안 쓰는 것과 같은 문제**로 보이고,
  내부 속도 되먹임도 마찬가지입니다(원인 미규명, 별도 조사 필요).

  그래서 이 판은 **하루 종일 정확했던 토크 명령만** 씁니다. 계산은 여기서 합니다.

      토크 = kp x (목표각 − 지금각) + kd x (0 − 지금속도)

  안전
  ----
  토크로 몰면 속도 부호화 범위라는 자연스러운 상한이 사라지므로, 대신 **직접** 막습니다.
  * 토크를 ``tau_max`` (기본 4 뉴턴미터) 로 자릅니다 — 모터 한계 120 의 3% 입니다.
  * 시작 자리에서 ``runaway_deg`` 이상 벗어나면 즉시 토크를 끄고 나옵니다.
  * 정해진 시간이 지나면 나옵니다.
  어느 경우든 나가기 전에 토크를 0 으로 놓습니다.

  마찰 때문에 목표에 정확히는 못 섭니다. 남는 오차는 **마찰 ÷ kp** 로, kp=25 면 약 1.2도
  입니다. 그래서 기본 허용오차가 3도입니다.
  """
  acc, last_raw, start = None, None, None
  trail = []
  t0 = time.monotonic()
  tau = 0.0
  while time.monotonic() - t0 < timeout_s:
    m.torque(tau)
    st = m.recv(timeout_s=0.01)
    if st is None:
      continue
    if last_raw is None:
      last_raw, acc = st.position_deg, st.position_deg
      start = acc
    d = st.position_deg - last_raw
    while d > 720.0:
      d -= 1440.0
    while d < -720.0:
      d += 1440.0
    acc += d
    last_raw = st.position_deg
    trail.append((time.monotonic() - t0, acc))
    while len(trail) > 2 and trail[-1][0] - trail[0][0] > 0.02:
      trail.pop(0)
    vel = ((trail[-1][1] - trail[0][1]) / (trail[-1][0] - trail[0][0])
           if len(trail) > 4 and trail[-1][0] > trail[0][0] else st.velocity_deg_s)
    err = target_deg - acc
    if abs(err) < tol_deg and abs(vel) < 20.0:
      break
    if abs(acc - start) > runaway_deg:
      print(f"[home] 시작 자리에서 {runaway_deg:.0f}도 넘게 벗어남 — 멈춥니다", flush=True)
      break
    tau = max(-tau_max, min(tau_max, kp * err * DEG + kd * (0.0 - vel) * DEG))
  for _ in range(25):
    m.torque(0.0); m.recv(timeout_s=0.002)
  time.sleep(0.4)
  st = m.read(tries=10)
  return st.position_deg if st else (acc if acc is not None else float("nan"))


def run_profile(m: Motor, profile, seconds: float, kp: float, kd: float, p0: float,
                bound_deg: float, temp_stop: float, tau_max: float = 8.0):
  """profile(t) -> (목표 오프셋 도, 목표 속도 도/s).

  ★ PD 를 **여기서** 계산해 **순수 토크**로 내보낸다. 모터에 kp 를 실어 보내는 길은 지금
  못 쓴다 — 목표를 지금 위치와 똑같이 주어 오차가 0.000도인데도 kp 칸이 0 이 아니게 되는
  순간 1.5 밀리초 만에 토크가 −90 뉴턴미터로 터진다(가속도로 역산해도 약 100 뉴턴미터가
  실제로 걸린다). 목표를 어긋나게 줘도 토크가 비례하지 않는다 = **모터가 우리가 보낸
  목표각을 쓰고 있지 않다.** 원인 미규명, 별도 조사 필요.

  토크 명령은 오늘 하루 종일 정확했으므로 검증된 것만 쓴다. 시뮬레이터 쪽도 사실 같은
  구조다(TnPdActuatorCfg 가 매 물리 걸음마다 파이썬 PD 를 계산하고 토크-속도로 자른다).

  위치는 되감김을 풀어 쓰고, 속도는 보고 값 대신 위치에서 만든다(보고 속도는 859 도/s
  에서 잘린다).
  """
  rows = []
  why = "끝까지"
  acc, last_raw = 0.0, None
  trail = []
  t0 = time.monotonic()          # ★ 이 값을 돌려준다 — 카메라 프레임과 같은 시계다
  tau = 0.0
  while True:
    t = time.monotonic() - t0
    if t >= seconds:
      break
    m.torque(tau)
    st = m.recv(timeout_s=0.01)
    if st is None:
      continue
    if last_raw is None:
      last_raw, acc = st.position_deg, p0
    d = st.position_deg - last_raw
    while d > 720.0:
      d -= 1440.0
    while d < -720.0:
      d += 1440.0
    acc += d
    last_raw = st.position_deg
    trail.append((t, acc))
    while len(trail) > 2 and t - trail[0][0] > 0.02:
      trail.pop(0)
    vel = ((trail[-1][1] - trail[0][1]) / (trail[-1][0] - trail[0][0])
           if len(trail) > 4 and trail[-1][0] > trail[0][0] else st.velocity_deg_s)
    q, qd = profile(t)
    tgt = p0 + q
    tau = kp * (tgt - acc) * DEG + kd * (qd - vel) * DEG
    tau = max(-tau_max, min(tau_max, tau))
    rows.append((t, tgt, qd, acc, vel, tau, st.torque_nm, st.temp_c))
    if abs(acc - tgt) > bound_deg:
      why = f"경계 {bound_deg:.0f}도 넘음 ({acc - tgt:+.1f})"
      break
    if st.fault:
      why = "모터가 고장을 보고함"
      break
    if st.temp_c >= temp_stop:
      why = f"온도 {st.temp_c:.0f}도"
      break
  for _ in range(25):
    m.torque(0.0); m.recv(timeout_s=0.002)
  return rows, why, t0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("mode", choices=["gain", "step", "chirp", "triangle", "videosync"])
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--amp", type=float, default=8.0, help="계단·처프 진폭 도")
  ap.add_argument("--period-s", type=float, default=1.5, help="계단 한 쪽에 머무는 시간")
  ap.add_argument("--cycles", type=int, default=6)
  ap.add_argument("--repeats", type=int, default=1,
                  help="같은 명령을 몇 번 되풀이해 받을지. 매번 원점 복귀부터 다시 한다")
  ap.add_argument("--f0", type=float, default=0.2)
  ap.add_argument("--f1", type=float, default=10.0)
  ap.add_argument("--chirp-s", type=float, default=40.0)
  ap.add_argument("--tau-max", type=float, default=8.0, help="호스트 PD 토크 상한")
  ap.add_argument("--bound-deg", type=float, default=25.0,
                  help="목표에서 이만큼 벗어나면 즉시 토크 0")
  ap.add_argument("--temp-stop", type=float, default=47.0)
  ap.add_argument("--home-deg", type=float, default=0.0,
                  help="★계단을 **어느 자리에서** 칠 것인가. 0 이 아니면 그 각도로 데려간 뒤 "
                       "거기를 중심으로 오르내린다. 같은 크기의 계단을 축의 서로 다른 자리에서 "
                       "쳐 보면, 마찰이 자리에 따라 다른지(코깅) 알 수 있다")
  ap.add_argument("--home-tol", type=float, default=5.0,
                  help="원점 복귀를 여기 안에 들어오면 끝낸 것으로 본다. 자리를 바꿔 가며 "
                       "실험할 때는 자리들이 서로 구분되도록 좁혀야 한다")
  ap.add_argument("--cool-to", type=float, default=None,
                  help="회차를 시작하기 전에 온도가 이 값 아래로 내려갈 때까지 **힘을 끄고** "
                       "기다린다. 훑기처럼 회차가 많은 실험에서 중간에 온도 상한에 걸려 "
                       "자료를 잃는 일을 막는다")
  ap.add_argument("--out-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--tag", default="")
  a = ap.parse_args(argv)

  suffix = ("_" + a.tag) if a.tag else ""
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    print(f"모터 {a.motor_id}({a.model}) · 보고 위치 {st.position_deg:.1f}도 · {st.temp_c:.0f}도")
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)

    p0 = home(m, target_deg=a.home_deg, tol_deg=a.home_tol)
    miss = abs(p0 - a.home_deg)
    print(f"시작 자리로 복귀 후 {p0:.2f}도 (목표 {a.home_deg:+.0f}도, 차 {miss:.2f}도, "
          f"되감김 가장자리 ±720 에서 멀어야 안전)\n")
    if abs(p0) > 300.0:
      print("되감김 가장자리에 너무 가깝습니다 — 중단합니다", file=sys.stderr)
      return 1
    # ★ 2026-09-10: 앞의 판은 복귀가 **전혀 안 되는데도** 조용히 진행했고, 자리별 실험
    # 30벌이 통째로 못 쓰게 됐습니다(목표 −60·−30·0 이 전부 −25 근처였음). 그래서 이제
    # 목표를 크게 벗어나면 **여기서 멈춥니다.**
    if miss > max(3.0 * a.home_tol, 10.0):
      print(f"복귀가 목표에서 {miss:.1f}도나 벗어났습니다 (허용 {a.home_tol:.0f}도) — "
            f"중단합니다", file=sys.stderr)
      return 1

    if a.mode == "gain":
      # 이득 단위 확인: 붙어 있는 힘보다 작은 토크만 나오게 오차를 고른다.
      print("이득 단위 확인 — 멈춘 채로 오차를 주고 토크를 본다")
      print(f"{'kp':>5} {'오차(도)':>9} {'보고토크':>10} {'kp x 오차[라디안]':>18} "
            f"{'움직임':>9}")
      for kp, dd in ((10, 1.0), (10, 2.0), (20, 1.0), (20, 2.0), (30, 1.0), (30, 1.5)):
        rows, why, _ = run_profile(m, lambda t: (dd, 0.0), 1.0, kp, 0.2, p0,
                                   a.bound_deg, a.temp_stop, a.tau_max)
        if len(rows) < 100:
          print(f"{kp:5.0f} {dd:9.2f}  자료 부족 ({why})")
          continue
        tail = rows[len(rows) // 2:]
        tq = statistics.mean(r[6] for r in tail)
        moved = statistics.mean(r[3] for r in tail) - p0
        err = dd - moved
        print(f"{kp:5.0f} {dd:9.2f} {tq:10.3f} {kp*err*DEG:18.3f} {moved:8.3f}도"
              + ("" if why == "끝까지" else f"  ({why})"))
        time.sleep(0.4)
      for _ in range(25):
        m.torque(0.0); m.recv(timeout_s=0.002)
      return 0

    A = a.amp
    if a.mode == "step":
      T = a.period_s
      prof = lambda t: (A if int(t / T) % 2 == 0 else -A, 0.0)      # noqa: E731
      secs = T * 2 * a.cycles
    elif a.mode == "videosync":
      # 카메라 영상과 자료의 시각을 맞추기 위한 파형.
      #   0~1.5초  빠른 흔들기 (초당 2번, ±20도) — 카메라가 확실히 보는 동기 신호
      #   1.5~4초  정지
      #   4초~    느린 삼각파 (일정 속도) — 실제로 견줄 파형
      # 카메라로 회전각을 직접 재려 했으나 화각·볼트 주기 때문에 상관이 0.4 에 그쳤다.
      # 큰 움직임 한 번이면 프레임 차이만으로 시각을 확실히 맞출 수 있다.
      AJ, FJ, T0, T1 = 20.0, 2.0, 1.5, 4.0
      TT = a.period_s * 2
      v = 4.0 * A / TT
      def prof(t):                                   # noqa: E306
        if t < T0:
          w = 2 * math.pi * FJ
          return AJ * math.sin(w * t), AJ * w * math.cos(w * t)
        if t < T1:
          return 0.0, 0.0
        ph = ((t - T1) % TT) / TT
        if ph < 0.25:
          return 4 * A * ph, v
        if ph < 0.75:
          return A - 4 * A * (ph - 0.25), -v
        return -A + 4 * A * (ph - 0.75), v
      secs = T1 + TT * a.cycles
    elif a.mode == "triangle":
      # 일정한 속도로 오르내린다. 속도가 일정하므로 **점성**이 드러나고, 방향이 바뀌는
      # 자리에서 **정지 마찰**이 만드는 평평한 구간이 보인다.
      T = a.period_s * 2
      v = 4.0 * A / T
      def prof(t):                                   # noqa: E306
        ph = (t % T) / T
        if ph < 0.25:
          return 4 * A * ph, v
        if ph < 0.75:
          return A - 4 * A * (ph - 0.25), -v
        return -A + 4 * A * (ph - 0.75), v
      secs = T * a.cycles
    else:
      f0, f1, D = a.f0, a.f1, a.chirp_s
      k = math.log(f1 / f0) / D                      # 로그 처프: 낮은 쪽에 시간을 더 준다
      def prof(t):                                   # noqa: E306
        ph = 2 * math.pi * f0 * (math.exp(k * t) - 1.0) / k
        f = f0 * math.exp(k * t)
        return A * math.sin(ph), A * 2 * math.pi * f * math.cos(ph)
      secs = D

    print(f"{a.mode} · kp={a.kp} kd={a.kd} 진폭 {A}도 · 한 번에 {secs:.0f}초 · "
          f"{a.repeats}회 · 경계 {a.bound_deg:.0f}도")
    for rep in range(a.repeats):
      if rep:
        p0 = home(m, target_deg=a.home_deg, tol_deg=a.home_tol)   # 매번 같은 자리에서
        if abs(p0 - a.home_deg) > max(3.0 * a.home_tol, 10.0):
          print(f"    복귀 실패 ({p0:.1f}도, 목표 {a.home_deg:+.0f}도) — 중단", file=sys.stderr)
          break
      if a.cool_to is not None:
        # 힘을 끈 채로 식힌다. 계단 크기를 훑을 때는 회차가 많아 열이 쌓이는데, 온도 상한에
        # 걸려 중간에 끊기면 그 크기의 자료를 통째로 잃는다.
        t_wait = time.monotonic()
        while True:
          stc = m.read(tries=8)
          if stc is None or stc.temp_c <= a.cool_to:
            break
          if time.monotonic() - t_wait > 600.0:
            print(f"    (600초를 기다려도 {a.cool_to:.0f}도 아래로 안 내려감 — 그냥 진행)")
            break
          print(f"    식히는 중 {stc.temp_c:.1f}도 → {a.cool_to:.0f}도 "
                f"({time.monotonic() - t_wait:.0f}초)", flush=True)
          time.sleep(10.0)
      rows, why, t_start = run_profile(m, prof, secs, a.kp, a.kd, p0, a.bound_deg,
                                       a.temp_stop, a.tau_max)
      hs = "" if a.home_deg == 0.0 else f"_h{a.home_deg:+g}"
      name = (f"pd_{a.mode}_{a.motor_id}_kp{a.kp:g}_kd{a.kd:g}_a{A:g}{hs}"
              f"_r{rep}{suffix}.csv")
      with open(os.path.join(a.out_dir, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "target_deg", "target_vel_deg_s", "pos_deg", "vel_deg_s",
                    "tau_cmd_nm", "tau_rep_nm", "temp_c"])
        w.writerows(rows)
      # 카메라와 시각을 맞추기 위한 쪽지. **같은 컴퓨터의 같은 시계**(time.monotonic)로 찍힌
      # 값이라, 카메라의 frames.csv 와 그대로 빼면 맞습니다 — 예전처럼 프레임률과 지연을
      # 수색해서 맞출 필요가 없습니다.
      with open(os.path.join(a.out_dir, name.replace(".csv", ".meta.json")), "w") as f:
        json.dump(dict(t0_monotonic=t_start, mode=a.mode, motor_id=a.motor_id,
                       model=a.model, kp=a.kp, kd=a.kd, amp=A, home_deg=a.home_deg,
                       tau_max=a.tau_max, seconds=secs, why=why, rows=len(rows)), f, indent=1)
      if rows:
        errs = [abs(r[3] - r[1]) for r in rows]
        print(f"  #{rep}: {len(rows):6d}줄 · {rows[-1][0]:5.1f}초 · 추종 오차 중앙 "
              f"{statistics.median(errs):5.2f}도 · 토크 최대 "
              f"{max(abs(r[5]) for r in rows):5.2f} Nm · {rows[-1][7]:.0f}도 · {why}")
      if why != "끝까지":
        print(f"    -> 중단: {why}")
        break
    home(m)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
