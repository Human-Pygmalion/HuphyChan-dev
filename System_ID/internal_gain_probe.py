#!/usr/bin/env python3
"""**모터 안의 위치 되먹임이 우리가 보낸 목표를 쓰는가.** 실물에 붙이기 전에 닫아야 할 물음.

2026-09-11 · 사용자 지시: 벤치에서 A1~A3 진행.

왜 이걸 재는가
--------------
벤치 측정은 전부 **계산을 우리 쪽에서 하고 순수 토크만** 보내는 방식이었습니다. 그런데
화면(뷰어)과 로봇 프로그램은 모터에게 **"이 각도로 가라 + 세기"** 를 보냅니다. 그 경로는
2026-09-09 에 **오차가 0.000도인데 1.5 밀리초 만에 −90 뉴턴미터**를 냈습니다. 실물 열두
관절을 그 경로로 몰면 관절이 전력으로 튀어나갑니다. 원인 미규명 상태입니다.

의심 하나 — 예전 시험이 **포화**했을 수 있습니다
-------------------------------------------------
그때 세기(kp)는 8이었습니다. 만약 모터 안쪽이 보는 오차가 크다면(예: 12 라디안),
8 x 12 = 96 뉴턴미터라 **모터 한계에 붙습니다.** 붙어 있으면 목표를 조금 바꿔도 토크가
안 변합니다 — "모터가 우리 목표를 안 쓴다"로 보이지만, 사실은 "이미 한계라 더 못 낸다"일
수 있습니다.

그래서 이 시험은 **가장 작은 세기**로 합니다. kp 칸은 12비트라 한 눈금이 5000/4095 =
**1.221** 입니다. 이 값이면 오차가 최대치(12.57 라디안)라도 토크는 **15.3 뉴턴미터** —
한계(120)에 한참 못 미쳐 **포화하지 않습니다.** 그러면 토크가 곧 오차를 말해 줍니다.

    모터 안쪽이 보는 오차 [라디안] = 나온 토크 ÷ kp

무엇을 보는가
-------------
1. **목표 = 지금 보고된 각도** 로 주었을 때 토크가 0 근처인가
2. 목표를 **1 라디안 옮기면** 토크가 kp x 1 만큼 변하는가 (변하면 우리 목표를 쓰는 것)
3. 1이 크게 어긋나면 **영점을 다시 잡고** 반복

안전
----
* 가장 작은 세기만 씁니다. 최악의 토크가 15.3 뉴턴미터입니다.
* 한 번에 **15 밀리초만** 힘을 주고 바로 끕니다. 그 사이 축은 2도쯤 움직입니다.
* 매번 끄고 위치를 읽어, 시작 자리에서 정한 각도 이상 벗어나면 **즉시 중단**합니다.
* 속도 되먹임(kd)은 작게 켭니다 — 빨라지면 스스로 제동이 걸립니다.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs_direct_socket import Motor  # noqa: E402


def burst(m: Motor, p_des_deg: float, kp: float, kd: float, ms: float = 15.0,
          tau_abort: float = 20.0, move_abort: float = 12.0):
  """아주 짧게 힘을 주고 끈다. (평균 토크, 처음 토크들, 움직인 각도)

  ★ 버스트 **안에서도** 끊습니다. 모터가 안쪽 오차를 크게 보면 토크가 수십 뉴턴미터로
  튀는데, 그대로 정해진 시간을 다 채우면 축이 엄청난 속도까지 갑니다(65 뉴턴미터면
  0.4초에 초당 300 라디안). 버스트가 끝나기를 기다렸다 판단하면 늦습니다."""
  taus, poss = [], []
  t0 = time.monotonic()
  stopped = ""
  while (time.monotonic() - t0) * 1e3 < ms:
    m.send(position_deg=p_des_deg, velocity_deg_s=0.0, kp=kp, kd=kd)
    st = m.recv(timeout_s=0.005)
    if st is not None:
      taus.append(st.torque_nm)
      poss.append(st.position_deg)
      if len(taus) >= 3 and abs(sum(taus[-3:]) / 3) > tau_abort:
        stopped = f"토크가 {tau_abort:.0f} 넘음"
        break
      if len(poss) > 1 and abs(poss[-1] - poss[0]) > move_abort:
        stopped = f"{move_abort:.0f}도 넘게 움직임"
        break
  if stopped:
    print(f"      (버스트 도중 중단: {stopped})")
  for _ in range(25):
    m.torque(0.0); m.recv(timeout_s=0.002)
  time.sleep(0.25)
  if not taus:
    return float("nan"), [], 0.0
  early = taus[: max(3, len(taus) // 4)]
  moved = (poss[-1] - poss[0]) if len(poss) > 1 else 0.0
  return statistics.mean(early), taus, moved


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--kp", type=float, default=1.221, help="kp 칸의 한 눈금 (5000/4095)")
  ap.add_argument("--kd", type=float, default=0.5)
  ap.add_argument("--bound-deg", type=float, default=25.0)
  ap.add_argument("--set-zero", action="store_true",
                  help="3단계: 영점을 다시 잡고 1단계를 반복한다")
  a = ap.parse_args(argv)

  DEG = 3.141592653589793 / 180.0
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    start = st.position_deg
    print(f"모터 {a.motor_id}({a.model}) · 보고 위치 {start:.2f}도 · {st.temp_c:.0f}도 · "
          f"kp {a.kp:.3f} (한 눈금) · kd {a.kd}")
    print(f"이 세기에서 최악의 토크 = kp x 12.57 라디안 = {a.kp*12.57:.1f} 뉴턴미터 "
          f"(한계 {'120' if a.model=='RS04' else '60'} 에 한참 못 미침 → 포화 없음)\n")
    m.clear_fault(); time.sleep(0.1)
    m.enable(); time.sleep(0.1)

    def guard():
      s = m.read(tries=8)
      if s and abs(s.position_deg - start) > a.bound_deg:
        print(f"★ 시작 자리에서 {s.position_deg - start:+.1f}도 벗어남 — 중단", file=sys.stderr)
        return False
      return True

    # --- 0단계: 세기 0 (대조) --------------------------------------------------
    s0 = m.read(tries=8)
    t, _, mv = burst(m, s0.position_deg, kp=0.0, kd=0.0)
    print(f"0단계 대조 (kp=0)            토크 {t:+7.3f} 뉴턴미터 · 움직임 {mv:+5.2f}도")

    # --- 1단계: 목표 = 지금 보고된 각도 ----------------------------------------
    if not guard():
      return 1
    s1 = m.read(tries=8)
    t1, _, mv1 = burst(m, s1.position_deg, a.kp, a.kd)
    err1 = t1 / a.kp if a.kp else float("nan")
    print(f"1단계 목표=보고각 {s1.position_deg:+8.2f}도   토크 {t1:+7.3f} → "
          f"모터가 보는 오차 {err1:+7.3f} 라디안 ({err1/DEG:+8.1f}도) · 움직임 {mv1:+5.2f}도")

    # --- 2단계: 목표를 ±1 라디안 옮긴다 ----------------------------------------
    print("\n2단계 목표를 옮기면 토크가 따라 변하는가 (변하면 우리 목표를 쓰는 것)")
    print(f"  기대: 1 라디안 옮길 때마다 토크가 {a.kp:.3f} 뉴턴미터씩")
    base = None
    for d_rad in (-1.0, -0.5, 0.0, +0.5, +1.0):
      if not guard():
        return 1
      s = m.read(tries=8)
      t, _, mv = burst(m, s.position_deg + d_rad / DEG, a.kp, a.kd)
      if d_rad == 0.0:
        base = t
      print(f"  목표 {d_rad:+5.2f} 라디안 어긋나게   토크 {t:+7.3f} · 움직임 {mv:+5.2f}도")
    print()

    # --- 2b단계: 세기를 사다리로 올려 본다 --------------------------------------
    # 1·2단계에서 토크가 잡음 수준으로 나왔습니다. 그 세기에서는 **토크 보고 자체가
    # 못 믿을 값**입니다(한 번의 응답은 실제가 0이어도 ±0.7 뉴턴미터로 퍼집니다). 그래서
    # 세기를 올려 가며 **움직임**으로 봅니다 — 움직임은 애매하지 않습니다.
    #
    # 안쪽 오차가 크다면(예: 12 라디안) 세기를 올릴수록 토크가 그에 비례해 커져야 하고,
    # 작다면 계속 0 근처여야 합니다.
    print("2b단계 세기를 올리며 (목표=보고각) — 토크와 움직임")
    print(f"  {'kp':>7}{'평균 토크':>12}{'최대 토크':>12}{'움직임':>10}{'안쪽 오차가 12라디안이면':>24}")
    for kp in (1.221, 5.0, 10.0, 20.0, 40.0):
      if not guard():
        return 1
      s = m.read(tries=8)
      t, taus, mv = burst(m, s.position_deg, kp, a.kd, ms=30.0)
      pk = max((abs(x) for x in taus), default=0.0)
      print(f"  {kp:7.3f}{t:12.3f}{pk:12.3f}{mv:10.2f}{kp*12.57:24.1f}")
      if pk > 25.0 or abs(mv) > 6.0:
        print("  ★ 토크나 움직임이 커졌습니다 — 사다리를 멈춥니다")
        break
    print()

    # --- 2c단계: 목표를 10도 옮기면 정말 10도 가는가 ----------------------------
    print("2c단계 목표를 +10도 옮기면 그만큼 가는가 (이것이 진짜 기능 시험)")
    for kp in (5.0, 10.0, 20.0):
      if not guard():
        return 1
      s = m.read(tries=8)
      p0 = s.position_deg
      t, taus, _ = burst(m, p0 + 10.0, kp, a.kd, ms=400.0)
      s2 = m.read(tries=8)
      moved = s2.position_deg - p0
      pk = max((abs(x) for x in taus), default=0.0)
      print(f"  kp {kp:5.1f} · 목표 +10.0도 → 실제 {moved:+7.2f}도 · "
            f"토크 평균 {t:+6.2f} 최대 {pk:6.2f} 뉴턴미터")
      if pk > 30.0:
        print("  ★ 토크가 커졌습니다 — 멈춥니다")
        break
    print()

    # --- 3단계: 영점을 다시 잡고 같은 것을 잰다 ---------------------------------
    def implied_error(tag, kp=5.0, ms=10.0):
      """목표를 보고각과 같게 주고, 나온 토크로 **모터 안쪽이 보는 오차**를 역산한다.

      토크 한 번의 보고는 못 믿지만(±0.7), 여기 토크는 수십 뉴턴미터라 잡음보다 훨씬 큽니다.
      kp 는 12비트로 **잘라내기** 때문에 실제로 실린 값은 아래처럼 다시 계산합니다."""
      s = m.read(tries=8)
      kp_eff = int(kp / 5000.0 * 4095) * 5000.0 / 4095
      t, taus, mv = burst(m, s.position_deg, kp, a.kd, ms=ms)
      pk = max((abs(x) for x in taus), default=0.0)
      err = (t / kp_eff) if kp_eff else float("nan")
      print(f"  {tag:<18} 보고각 {s.position_deg:+8.2f}도 · kp 요청 {kp:g} → 실제 "
            f"{kp_eff:.3f} · 토크 평균 {t:+8.2f} 최대 {pk:7.2f} · "
            f"**안쪽 오차 {err:+7.2f} 라디안 ({err*57.2958:+8.0f}도)** · 움직임 {mv:+6.2f}도")
      return err

    if a.set_zero:
      if not guard():
        return 1
      print("3단계 영점을 다시 잡으면 안쪽 오차가 사라지는가")
      before = implied_error("영점 전")
      for _ in range(5):
        m.disable(); time.sleep(0.02)
      m._cmd(0xFE)                      # CMD_SET_ZERO (명령 4)
      time.sleep(0.4)
      m.clear_fault(); time.sleep(0.1)
      m.enable(); time.sleep(0.1)
      after = implied_error("영점 후")
      if before == before and after == after:
        print(f"\n  → 안쪽 오차 {before:+.2f} → {after:+.2f} 라디안 "
              f"({'사라짐 = 가설 확인' if abs(after) < 0.3*abs(before) else '그대로 = 가설 기각'})")

    s = m.read(tries=8)
    print(f"\n끝 · 위치 {s.position_deg:.2f}도 (시작 {start:.2f}) · 온도 {s.temp_c:.0f}도 · "
          f"고장 {s.fault}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
