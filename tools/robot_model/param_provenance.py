#!/usr/bin/env python3
"""모델 파라미터마다 **파일·줄·변수명·지금 값**을 찾아서 표로 낸다.

2026-09-15 · 사용자 지시: *"이렇게 추상적으로 주면 너도 놓치잖아. 파일위치, 정의 위치,
변수명, 출처도 일자, 방법, 실제 수치(codeblock) 확인해"*

왜 손으로 안 쓰는가
-------------------
"Fusion CAD 에서 왔습니다" 같은 문장은 **확인할 수가 없습니다.** 값이 바뀌어도 문장은
그대로 남고, 읽는 사람은 그걸 모릅니다. 그래서 이 프로그램은

  * 변수 이름을 **파일에서 찾아** `파일:줄` 을 적고,
  * **지금 그 파일에 적힌 값**을 읽어 같이 적습니다.

값이 바뀌면 표가 바뀌고, 변수가 사라지면 **못 찾았다고 적습니다**(조용히 넘어가지 않음).
출처의 날짜·방법·근거 문서만 사람이 씁니다 — 그건 파일에서 읽을 수 없기 때문입니다.

    python3 tools/robot_model/param_provenance.py            # 표를 찍는다
    python3 tools/robot_model/param_provenance.py --insert docs/145_model_parameter_provenance.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONST = ("mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py")
BUILD = "tools/robot_model/build_robot.py"
WRITER = "tools/robot_model/write_motor_params_xml.py"
MASSP = "tools/robot_model/massprops_fusion.py"
ENVCFG = "mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py"

MARK_BEGIN = "<!-- PARAM-PROVENANCE:BEGIN (tools/robot_model/param_provenance.py) -->"
MARK_END = "<!-- PARAM-PROVENANCE:END -->"

#: (무엇, 파일, 찾을 변수(정규식), 언제·어떻게 잰 값인가, 근거 문서)
ENTRIES: list[tuple[str, str, str, str, str]] = [
  ("회전자 관성·점성·마른 마찰 (실측 정본)", CONST, r"^MOTOR_MEAS = \{",
   "2026-09-10 · 라즈베리파이에 CAN 어댑터 직결(제어 루프 약 4 kHz). 관성은 **두 경로가 "
   "합치는 것**으로 확인 — ① 토크를 끊고 타성으로 감속 ② 토크 대 가속의 기울기(마찰이 "
   "뺄셈에서 지워짐). 모델 없이 공진으로 재확인: kp=20 에서 2.48 Hz = sqrt(kp/J)/2π. "
   "마찰·점성은 순수 타성 감속, 삼각파 지연(마찰/kp)과 1 % 안에서 일치",
   "docs/130(방법) · docs/131(인수인계·감사)"),
  ("같은 값의 두 번째 사본 (URDF 주입용)", WRITER, r'^\s*"bench-2026-09-10": \{',
   "위와 같은 측정. `check_motor_params.py` 가 둘이 갈라지지 않는지 봄",
   "docs/144"),
  ("옛 값 (재현 전용, 학습 금지)", CONST, r"^if _MOTOR_MEAS_LEGACY:",
   "2026-09-12 · e3c67da2 이전 정책을 **자기 시절 모터**에 태워 재생할 때만. "
   "★그 값 자체가 옮겨 적기 오류였음(벤치 실제값 0.030605/0.026008/0.526084)",
   "docs/131"),
  ("카탈로그 관성 (PYG_MOTOR_MEAS=0 일 때만)", CONST, r"^ARMATURE_04 = ",
   "RobStride 설명서. 실측 이전의 값이라 지금 학습에는 안 쓰임", "—"),
  ("최대 토크·속도 한계 (카탈로그)", CONST, r"^ACTUATOR_04 = ElectricActuator\(",
   "RobStride 설명서. 파일 주석에 출처 링크가 적혀 있음. "
   "**속도 한계는 `PYG_TN=0` 일 때만 쓰임** — 켜져 있으면 실측 곡선이 이김",
   "pygmalion_constants.py 의 주석 링크"),
  ("실측 토크-속도 곡선", CONST, r"^def tn_curve\(",
   "`Motor_Spec/RS0*_TN_curve_48V.csv` (48 V 실측). 곡선은 **최고 토크가 꺾이기 "
   "시작하는 속도**부터 적혀 있고, 끝점으로 공식 무부하 속도를 붙임",
   "Motor_Spec/"),
  ("공식 무부하 속도", CONST, r"^NO_LOAD_RPM = ",
   "카탈로그 값(둘 다 200 rpm = 20.94 rad/s). 곡선의 토크 0 끝점으로만 쓰임 — "
   "**실측 최고점은 이보다 낮음**(RS04 190 rpm @ 11.2 N·m)", "Motor_Spec/"),
  ("엉덩이·무릎 위치제어 게인", CONST, r"^PYG_ACTUATOR_03_HIP_YAW = _pd\(",
   "2026-07-06 사용자 지시 · G1(100/2)·K-bot(100/10, 같은 RS04)을 질량비 51.5/35=1.47 "
   "로 환산 → Kp 150. 무릎만 220. Kd 는 G1 환산 3 ~ K-bot 환산 15 사이에서 6",
   "pygmalion_constants.py 주석"),
  ("발목 관절 게인 (2-RSU)", CONST, r"^ANKLE_2RSU_KP = ",
   "G1 급 발목 강성에 맞춤. 최대 토크는 `mech_design_eval.py` 의 야코비 전치 계산 "
   "(RS03 두 개 공동구동, 크랭크 약 30 mm)", "docs/87 §5"),
  ("발목 최대 토크 (2-RSU 공동구동)", CONST, r"^ANKLE_2RSU_EFF_PITCH = ",
   "`mech_design_eval.py` 의 야코비 전치 계산. 숙임은 모터 두 개가 같이 밀어 관절측 "
   "약 120 이 되는 것을 연속 정격으로 90 으로 낮춤", "docs/87 §5"),
  ("크랭크 게인 (발목에서 유도)", CONST, r"^CRANK_KP = ",
   "발목 관절 게인을 **실측 지렛대비 1.25(숙임)** 로 반사하고 모터 두 개로 나눔: "
   "28.5 × 1.25² / 2. 손으로 고른 숫자가 아니라 유도값", "docs/87 §5"),
  ("관절 가동범위", BUILD, r"^JOINT = \{",
   "CAD 자기충돌 스윕(`rom_measured.json`) + 설계캡. 2026-09-02 에 사용자가 뷰어에서 "
   "관절을 **손으로 다 돌려 보고** 부호 방향이 뒤집힌 것을 찾아 축과 범위를 같이 정정",
   "docs/107 (전 관절 ROM 표)"),
  ("설계캡 (범위의 상한)", BUILD, r"^DESIGN_CAP = \{",
   "관절마다 기구가 허용하는 끝값. hip_roll 내전 +25° 는 2026-07-13 에 "
   "gen21 앵커의 밀기 회복 스텝이 +25 에 포화하는 것을 보고 **유지**로 확정",
   "docs/experiments/2026-07-13_gen21_bent_p2.md §11"),
  ("크랭크 범위", BUILD, r"^CRANK_RANGE = ",
   "발목이 필요로 하는 것보다 넓게 두고 스윕으로 검증", "docs/107"),
  ("모터 질량 (저울 실측)", MASSP, r"^RS04_MEASURED_KG = ",
   "사용자가 저울로 잼. RS03 은 카탈로그 0.880 kg 보다 무거운 0.9195 kg",
   "URDF 머리말 주석에 출처 문장이 그대로 들어감"),
  ("링크 질량·무게중심·관성의 원천 파일", BUILD, r"^\s*massprops\s+~/pyg_fea",
   "Fusion CAD 내보내기(`massprops_step.py` 가 STEP 에서 계산). 모터는 위의 저울 "
   "실측 질량으로 바꿔치기함", "docs/62 계통"),
  ("질량 무작위화 범위", ENVCFG, r"_mass_dr = Path\(",
   "`tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_*.json`. 부품별 "
   "공차·체결구 50 % 가정에서 나옴", "런의 launch_manifest.json 에 경로가 박힘"),
  ("액션 자르기 = soft 한계 공식", CONST, r"def safe_target_clip\(",
   "중심 ± 0.5 × 범위 × 계수(0.9). mjlab `Entity.soft_joint_pos_limits` 와 **같은 식** "
   "이라 정책이 통과하는 클램프와 시뮬레이터가 강제하는 클램프가 하나의 계약이 됨. "
   "경계에 계수를 곱하는 것이 아님(비대칭 관절에서 갈림)", "런 노트 §1b-3"),
  ("리셋·목표 웅크림 자세", CONST, r"^def _crouch|^CROUCH|_INIT_MID",
   "★2026-09-14: 여기 발목 값이 **깊은 웅크림 것**이라 발바닥이 10.6도 들려 있었음. "
   "얕은 웅크림은 발목 = 무릎 − 엉덩이 로 유도해야 맞음. v6 을 중단시킨 원인",
   "docs/experiments/ABORTED_2026-09-14_19-37-39_legonly_ab_v6_standing_p1.md"),
]


def find(rel: str, pattern: str) -> tuple[str, str] | None:
  """파일에서 변수를 찾아 (`파일:줄`, 그 줄부터의 값 덩어리) 를 낸다."""
  p = REPO / rel
  if not p.exists():
    return None
  lines = p.read_text(encoding="utf-8").splitlines()
  rx = re.compile(pattern)
  for i, ln in enumerate(lines):
    if rx.search(ln):
      block = [ln]
      # 열린 괄호가 닫힐 때까지, 아니면 한 줄짜리
      depth = ln.count("{") + ln.count("(") - ln.count("}") - ln.count(")")
      j = i
      while depth > 0 and j + 1 < len(lines) and j - i < 24:
        j += 1
        block.append(lines[j])
        depth += (lines[j].count("{") + lines[j].count("(")
                  - lines[j].count("}") - lines[j].count(")"))
      return f"{rel}:{i + 1}", "\n".join(block)
  return None


def render() -> str:
  out = [MARK_BEGIN, "",
         "> 이 절은 `tools/robot_model/param_provenance.py` 가 **파일에서 직접 읽어** 씁니다.",
         "> 값이 바뀌면 여기도 바뀌고, 변수가 사라지면 못 찾았다고 적힙니다. 손으로 고치지 마십시오.",
         ""]
  missing = []
  for name, rel, pat, how, doc in ENTRIES:
    got = find(rel, pat)
    out.append(f"### {name}")
    out.append("")
    if got is None:
      out.append(f"❌ **못 찾았습니다** — `{rel}` 에서 `{pat}` 를 찾지 못했습니다. "
                 "옮겨졌거나 이름이 바뀌었습니다.")
      missing.append(name)
      out.append("")
      continue
    where, block = got
    out.append(f"* **정의 위치** `{where}`")
    out.append(f"* **언제·어떻게** {how}")
    out.append(f"* **근거 문서** {doc}")
    out.append("")
    out.append("```python")
    out.append(block)
    out.append("```")
    out.append("")
  if missing:
    out.append(f"**★ 못 찾은 항목 {len(missing)}개: {', '.join(missing)}**")
    out.append("")
  out.append(MARK_END)
  return "\n".join(out)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--insert", help="이 노트의 표식 사이를 갈아 끼운다 (없으면 끝에 붙임)")
  a = ap.parse_args(argv)
  body = render()
  if not a.insert:
    print(body)
    return 0
  p = REPO / a.insert if not Path(a.insert).is_absolute() else Path(a.insert)
  txt = p.read_text(encoding="utf-8")
  if MARK_BEGIN in txt and MARK_END in txt:
    head = txt[: txt.index(MARK_BEGIN)]
    tail = txt[txt.index(MARK_END) + len(MARK_END):]
    txt = head + body + tail
  else:
    txt = txt.rstrip() + "\n\n" + body + "\n"
  p.write_text(txt, encoding="utf-8")
  print(f"{p.name}: ok ({len(ENTRIES)}개 항목)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
