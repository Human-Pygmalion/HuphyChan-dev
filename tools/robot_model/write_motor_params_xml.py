#!/usr/bin/env python3
"""실측한 모터 값을 **MJCF 파일 안에 직접 써 넣은 수정본**을 만든다.

2026-09-10 · 사용자 지시: "우리 최신 urdf(mjcf) 에 armature, damping, frictionloss 바뀌어야
할거같은데, 수정한 버전을 추가하고."

왜 필요한가
-----------
지금 모델 파일(XML)에는 주 관절의 `armature`·`damping`·`frictionloss` 가 **적혀 있지
않습니다.** mjlab 이 불러올 때 액추에이터 설정(`pygmalion_constants.py` 의 `MOTOR_MEAS`)에서
집어넣습니다. 그래서 **XML 만 열어 보면 그 값이 무엇인지 알 수 없고**, mjlab 이 아닌 다른
도구(순수 MuJoCo, 다른 학습 코드)로 이 파일을 불러오면 **그 값들이 아예 안 들어갑니다.**

이 도구는 값이 **파일 안에 박힌** 사본을 만듭니다. 원본은 건드리지 않습니다.

어느 관절이 어느 모터인가
-------------------------
액추에이터 설정과 같은 대응을 씁니다.

===========================  ======  ==============================================
관절 이름                     모터    비고
===========================  ======  ==============================================
`*_hip_pitch_joint`          RS04
`*_hip_roll_joint`           RS04
`*_knee_joint`               RS04
`*_waist_yaw_joint`          RS04    상체(선택)
`*_hip_yaw_joint`            RS03
`*_crank_A_joint` / `_B_`    RS03    발목 폐루프의 구동 크랭크
`*_shoulder_*_joint`         RS03    상체(선택)
`*_ankle_pitch/roll_joint`   —       AB(폐루프) 모델에서는 **수동 힌지**. 건드리지 않음
`*_rod_*`                    —       수동 링크. 건드리지 않음
`*_toe_joint`                —       수동 스프링. 건드리지 않음
===========================  ======  ==============================================

    write_motor_params_xml.py --set bench2026-09-10 \
        --xml .../pygmalion_v4_printed_loop.xml
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- 값 묶음
SETS = {
  # 2026-09-09/10 라즈베리파이 직결 재측정. 관성은 두 방법(타성 감속 · 마찰 소거 뺄셈)이
  # 합치고 공진 자리로 다시 확인했으며, 마찰·점성은 순수 타성 감속에서 나왔습니다.
  # docs/130 (방법) · docs/131 (인수인계).
  "bench-2026-09-10": {
    "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
    "RS03": dict(armature=0.0638, damping=0.0252, frictionloss=0.251),
  },
}

JOINT_MOTOR = [
  (re.compile(r".*_hip_pitch_joint$"), "RS04"),
  (re.compile(r".*_hip_roll_joint$"), "RS04"),
  (re.compile(r".*_knee_joint$"), "RS04"),
  (re.compile(r".*waist_yaw_joint$"), "RS04"),
  (re.compile(r".*_hip_yaw_joint$"), "RS03"),
  (re.compile(r".*_crank_[AB]_joint$"), "RS03"),
  (re.compile(r".*_shoulder_.*_joint$"), "RS03"),
]

JOINT_RE = re.compile(r'<joint\b([^>]*?)/?>')
NAME_RE = re.compile(r'name="([^"]+)"')


def motor_for(name: str) -> str | None:
  for pat, fam in JOINT_MOTOR:
    if pat.match(name):
      return fam
  return None


def patch(text: str, values: dict) -> tuple[str, list]:
  """`<joint ...>` 마다 이름을 보고 세 속성을 넣거나 갈아 끼운다."""
  touched = []

  def one(m):
    attrs = m.group(1)
    nm = NAME_RE.search(attrs)
    if not nm:
      return m.group(0)
    fam = motor_for(nm.group(1))
    if fam is None:
      return m.group(0)
    v = values[fam]
    out = attrs
    for k in ("armature", "damping", "frictionloss"):
      new = f'{k}="{v[k]:.6g}"'
      if re.search(rf'\b{k}="[^"]*"', out):
        out = re.sub(rf'\b{k}="[^"]*"', new, out)
      else:
        out = out.rstrip() + " " + new
    touched.append((nm.group(1), fam))
    closing = "/>" if m.group(0).rstrip().endswith("/>") else ">"
    return f"<joint{out if out.startswith(' ') else ' ' + out}{closing}"

  return JOINT_RE.sub(one, text), touched


URDF_JOINT_RE = re.compile(r'(<joint\b[^>]*?>)(.*?)(</joint>)', re.S)
DYN_RE = re.compile(r'\s*<dynamics\b[^>]*/>')
ARM_COMMENT_RE = re.compile(r'\s*<!--\s*armature \(reflected rotor inertia\).*?-->', re.S)
"""두 번 돌려도 같은 결과가 나오게 **직전에 쓴 것부터 걷어낸다.** 안 그러면 주석이
돌릴 때마다 한 줄씩 쌓인다 (2026-09-15 실제로 그랬음)."""


def patch_urdf(text: str, values: dict) -> tuple[str, list]:
  """URDF 의 `<joint>` 마다 `<dynamics damping= friction=/>` 를 넣거나 갈아 끼운다.

  2026-09-15 · 사용자 지시: *"최신 mjcf 에 있는 설정 중에 urdf 에도 쓸수있지만 override
  하도록 설정된 값들이 있나? 그런게 있으면 urdf 에도 업데이트해줘. 실제로 나중에 로봇을
  다른 플랫폼 시뮬쪽에 달면 문제가 될거같아."*

  ★ **armature 는 URDF 규격에 자리가 없습니다.** 점성(`damping`)과 쿨롱 마찰(`friction`)만
  `<dynamics>` 로 표준에 맞게 넣고, armature 는 관절마다 **주석**으로 적고 옆에 sidecar
  JSON 을 따로 냅니다. 규격에 없는 태그를 지어내면 엄격한 파서가 파일 전체를 거부할 수
  있어서, 읽는 쪽이 골라 쓸 수 있게 둡니다.

  URDF 의 `friction` 은 MuJoCo 의 `frictionloss` 와 같은 뜻입니다(움직임을 막는 방향의
  일정한 힘). `damping` 도 양쪽 다 속도에 비례하는 감쇠라 그대로 옮겨집니다.
  """
  touched = []

  def one(m):
    head, body, tail = m.group(1), m.group(2), m.group(3)
    nm = NAME_RE.search(head)
    if not nm:
      return m.group(0)
    fam = motor_for(nm.group(1))
    if fam is None:
      return m.group(0)
    v = values[fam]
    body = ARM_COMMENT_RE.sub("", DYN_RE.sub("", body))     # 있던 것은 걷어낸다
    body = body.rstrip() + (
      f'\n    <dynamics damping="{v["damping"]:.6g}" friction="{v["frictionloss"]:.6g}"/>'
      f'\n    <!-- armature (reflected rotor inertia) = {v["armature"]:.6g} kg*m^2 : '
      f'no standard URDF field; see the sidecar .motor_params.json -->\n  ')
    touched.append((nm.group(1), fam))
    return head + body + tail

  return URDF_JOINT_RE.sub(one, text), touched


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--xml", nargs="+", required=True,
                  help="고칠 파일들. `.urdf` 면 URDF 규칙으로, 아니면 MJCF 규칙으로 고친다")
  ap.add_argument("--set", dest="which", default="bench-2026-09-10", choices=list(SETS))
  ap.add_argument("--suffix", default=None,
                  help="사본 이름에 붙일 꼬리. 기본은 값 묶음 이름")
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--in-place", action="store_true",
                  help="사본을 만들지 않고 **원본을 고친다**. URDF 처럼 '다른 플랫폼이 "
                       "그대로 집어 가는 파일' 은 값이 그 안에 있어야 해서 필요하다")
  ap.add_argument("--sidecar", action="store_true",
                  help="URDF 옆에 `<이름>.motor_params.json` 을 같이 낸다 "
                       "(armature 는 URDF 규격에 자리가 없다)")
  a = ap.parse_args(argv)

  values = SETS[a.which]
  suffix = a.suffix or a.which.replace("-", "")
  print(f"값 묶음: {a.which}")
  for fam, v in values.items():
    print(f"  {fam}: armature {v['armature']:.6g} · damping {v['damping']:.6g} · "
          f"frictionloss {v['frictionloss']:.6g}")
  print()

  for p in (Path(x) for x in a.xml):
    if not p.exists():
      print(f"  없음: {p}", file=sys.stderr)
      continue
    is_urdf = p.suffix.lower() == ".urdf"
    out, touched = (patch_urdf if is_urdf else patch)(p.read_text(), values)
    dst = p if a.in_place else p.with_name(f"{p.stem}_{suffix}{p.suffix}")
    counts = {}
    for _n, fam in touched:
      counts[fam] = counts.get(fam, 0) + 1
    if a.dry_run:
      print(f"  (연습) {p.name} → {dst.name} · 고칠 관절 "
            + ", ".join(f"{k} {v}개" for k, v in sorted(counts.items())))
      continue
    dst.write_text(out)
    if is_urdf and a.sidecar:
      import json
      side = dst.with_suffix(".motor_params.json")
      side.write_text(json.dumps({
        "_comment": ("armature (reflected rotor inertia, kg*m^2) has no standard URDF "
                     "field. Apply these per joint when loading this URDF into a "
                     "simulator that supports it (MuJoCo: <joint armature>, Isaac: "
                     "ArticulationRootProperties/joint armature)."),
        "source": a.which,
        "joints": {n: {"motor": fam, **values[fam]} for n, fam in touched},
      }, indent=2, ensure_ascii=False) + "\n")
      print(f"  {side.name} 같이 씀")
    print(f"  {p.name} → {dst.name} · 고친 관절 "
          + ", ".join(f"{k} {v}개" for k, v in sorted(counts.items())))
    print("      " + ", ".join(n for n, _f in touched))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
