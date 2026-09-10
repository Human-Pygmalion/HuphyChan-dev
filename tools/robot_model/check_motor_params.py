#!/usr/bin/env python3
"""학습을 걸기 전에 **모터 값이 정말 그대로 들어갔는지** 확인한다.

2026-09-11. 이 프로그램이 있는 이유는 실제로 조용히 어긋난 적이 있기 때문입니다.
2026-09-10 에 `MOTOR_MEAS` 를 새 측정값으로 고쳤는데, 발목 값을 만드는 프로그램
(`ankle_rp_envelope.py`)이 같은 숫자를 **손으로 들고 있어서** 따라오지 않았습니다. 그래서
엉덩이 회전은 새 값, 같은 종류 모터인 발목은 옛 값 — 한 모형 안에서 4.2배 어긋난 채였고,
**아무 오류도 안 났습니다.** 둘을 대조하는 곳이 없었기 때문입니다.

여기가 그 대조하는 곳입니다.

무엇을 보는가
-------------
1. 발목 값 파일이 `MOTOR_MEAS` 와 같은 모터 값에서 나왔는가
2. 각 발목 방식에서 **액추에이터가 관절에 실제로 실어 주는 값**이 무엇인가
3. 마찰·점성이 빠진 관절이 있는가 (있으면 그 관절만 다른 물리로 학습된다)

    check_motor_params.py                # 기본(직렬)
    PYG_ANKLE_MODE=AB check_motor_params.py
"""

from __future__ import annotations

import json
import os
import sys

REPO = "/home/syaro/MikuchanRemote/Human-Pygmalion"
sys.path.insert(0, f"{REPO}/mujoco-sim/mjlab/src")


def main() -> int:
  from mjlab.asset_zoo.robots.pygmalion import pygmalion_constants as C

  mode = os.environ.get("PYG_ANKLE_MODE", "").strip() or "(기본, 직렬)"
  print(f"발목 방식: {mode}")
  print(f"측정값 (MOTOR_MEAS, 하나뿐인 출처):")
  for fam, m in C.MOTOR_MEAS.items():
    print(f"  {fam}  회전자 관성 {m['armature']:<8} 마찰 {m['frictionloss']:<8} "
          f"점성 {m['damping']}")

  bad = []

  # --- 1. 발목 포락표가 같은 값에서 나왔는가 -------------------------------------
  env = C.ANKLE_RP_ENVELOPE
  if env.exists():
    mot = json.load(open(env))["motor"]
    want = C.MOTOR_MEAS[mot["name"]]
    got = (mot["J"], mot["b"], mot["tc"])
    exp = (want["armature"], want["damping"], want["frictionloss"])
    ok = all(abs(a - b) < 1e-9 for a, b in zip(got, exp))
    print(f"\n발목 포락표 {env.name}: 모터 {mot['name']} "
          f"J={mot['J']} b={mot['b']} tc={mot['tc']}  → {'맞음' if ok else '★ 어긋남'}")
    if not ok:
      bad.append(f"발목 포락표가 MOTOR_MEAS 와 다릅니다: {got} 대 {exp}. "
                 f"tools/robot_model/ankle_rp_envelope.py 를 다시 돌리세요")
  else:
    bad.append(f"발목 포락표가 없습니다: {env}")

  # --- 2. 관절에 실제로 실리는 값 ------------------------------------------------
  print("\n액추에이터가 관절에 실어 주는 값 (MJCF 의 관절값을 **덮어씁니다**)")
  print(f"  {'대상 관절':<46}{'관성':>10}{'마찰':>10}{'점성':>10}")
  for act in C.PYG_ARTICULATION.actuators:
    names = str(act.target_names_expr)
    arm = getattr(act, "armature", None)
    if arm is None:
      print(f"  {names:<46}{'(포락표에서 반사)':>30}")
      continue
    fr = getattr(act, "frictionloss", None)
    vd = getattr(act, "viscous_damping", None)
    flag = ""
    if fr is None or vd is None:
      flag = "   ★ 마찰/점성 없음"
      bad.append(f"{names} 에 마찰 또는 점성이 없습니다 — 이 관절만 다른 물리로 학습됩니다")
    print(f"  {names:<46}{arm:>10.5g}{str(fr):>10}{str(vd):>10}{flag}")

  # --- 3. 판정 -------------------------------------------------------------------
  print()
  if bad:
    print("★★ 그대로 학습하면 안 됩니다:")
    for b in bad:
      print(f"   - {b}")
    return 1
  print("모든 관절이 같은 출처의 값을 씁니다. 학습을 걸어도 됩니다.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
