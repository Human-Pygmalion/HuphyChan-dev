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
import re
import sys
from pathlib import Path

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

  # --- 3. 모델 파일들이 같은 값을 들고 있는가 (2026-09-15) ------------------------
  #
  # 사용자 지시: *"모든 설정파일은 같은 값을 가져야 한다고. 혹은 단 하나만 True 값을
  # 가지도록 하라고."* 여기가 그 "같은 값인가" 를 보는 곳입니다. 출처는 위 `MOTOR_MEAS`
  # 하나뿐이고, 아래는 **디스크에 있는 파일이 그 값을 정말 들고 있는지** 봅니다 --
  # mjlab 이 불러올 때 덮어쓰므로 학습은 맞더라도, 파일만 다른 도구로 가져가면
  # 안 맞을 수 있기 때문입니다 (실제로 09-15 까지 MJCF 는 전부 0 이었습니다).
  print("\n모델 파일이 들고 있는 값 (mjlab 없이 그 파일만 열었을 때)")
  bad += _check_model_files(C)

  # --- 4. 판정 -------------------------------------------------------------------
  print()
  if bad:
    print("★★ 그대로 학습하면 안 됩니다:")
    for b in bad:
      print(f"   - {b}")
    return 1
  print("모든 관절이 같은 출처의 값을 씁니다. 학습을 걸어도 됩니다.")
  return 0


def _check_model_files(C) -> list[str]:
  """MJCF 와 URDF 를 **그대로 열어** 관절마다 세 값을 읽고 `MOTOR_MEAS` 와 대조한다.

  mjlab 을 거치지 않습니다 -- 거치면 덮어쓴 값을 보게 되어 검사가 뜻이 없습니다.
  """
  import sys as _sys
  _sys.path.insert(0, f"{REPO}/tools/robot_model")
  import write_motor_params_xml as W

  meas = W.SETS["bench-2026-09-10"]
  if meas != C.MOTOR_MEAS:
    return [f"write_motor_params_xml 이 읽은 값이 MOTOR_MEAS 와 다릅니다: "
            f"{meas} 대 {C.MOTOR_MEAS}"]

  import mujoco
  bad: list[str] = []
  xmls = sorted(Path(f"{REPO}/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls")
                .glob("*v30_proxyfix*.xml"))
  xmls = [x for x in xmls if "bench" not in x.name]
  for x in xmls:
    m = mujoco.MjModel.from_xml_path(str(x))
    miss = []
    for i in range(m.njnt):
      n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
      fam = W.motor_for(n)
      if fam is None:
        continue                    # 수동 관절 -- 모터 값이 있으면 안 된다
      d = m.jnt_dofadr[i]
      want = meas[fam]
      got = (m.dof_armature[d], m.dof_damping[d], m.dof_frictionloss[d])
      exp = (want["armature"], want["damping"], want["frictionloss"])
      if any(abs(a - b) > 1e-9 for a, b in zip(got, exp)):
        miss.append(f"{n} {tuple(round(float(v), 5) for v in got)} 대 {exp}")
    mark = "맞음" if not miss else f"★ 어긋남 {len(miss)}개"
    print(f"  {x.name:<62} {mark}")
    if miss:
      bad.append(f"{x.name}: {miss[0]}" + (f" 외 {len(miss) - 1}개" if len(miss) > 1 else ""))

  urdfs = sorted(Path(f"{REPO}/pygmalion_locomotion/assets/pygmalion_v2").glob("*v30_proxyfix*.urdf"))
  for u in urdfs:
    txt = u.read_text(encoding="utf-8")
    miss = []
    for jn, fam in ((m_.group(1), W.motor_for(m_.group(1)))
                    for m_ in re.finditer(r'<joint name="([^"]+)" type="revolute">', txt)):
      if fam is None:
        continue
      blk = txt.split(f'<joint name="{jn}" type="revolute">', 1)[1].split("</joint>", 1)[0]
      dyn = re.search(r'<dynamics damping="([-\d.eE]+)" friction="([-\d.eE]+)"', blk)
      want = meas[fam]
      if dyn is None:
        miss.append(f"{jn} <dynamics> 없음")
        continue
      got = (float(dyn.group(1)), float(dyn.group(2)))
      exp = (want["damping"], want["frictionloss"])
      if any(abs(a - b) > 1e-9 for a, b in zip(got, exp)):
        miss.append(f"{jn} {got} 대 {exp}")
    mark = "맞음" if not miss else f"★ 어긋남 {len(miss)}개"
    print(f"  {u.name:<62} {mark}")
    if miss:
      bad.append(f"{u.name}: {miss[0]}" + (f" 외 {len(miss) - 1}개" if len(miss) > 1 else ""))
  return bad


if __name__ == "__main__":
  raise SystemExit(main())
