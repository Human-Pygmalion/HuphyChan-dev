#!/usr/bin/env python3
"""모델 파일의 **좌우 이름표만** 맞바꾼다. 기하는 한 글자도 안 건드린다.

2026-09-12. 사용자가 찾은 결함입니다 — 모델에서 `L_` 이 붙은 사슬이 **물리적으로는
오른다리**입니다.

    L_hip_pitch_link  pos = (0, **-0.1237**, 0)      <- -y
    R_hip_pitch_link  pos = (0, **+0.1237**, 0)      <- +y

앞이 어느 쪽인지는 기구로 확인했습니다: 무릎을 60도 굽히면 발이 **-x 로 0.41m**
움직입니다. 무릎은 뒤로 접히므로 **+x 가 앞**이고, z 가 위인 오른손 좌표계에서
**+y 가 왼쪽**입니다. 따라서 `L_` 은 오른다리에 붙어 있었습니다.

이 모델은 좌우가 **미러 부호**(왼 무릎 `[0,+120]` / 오른 무릎 `[-120,0]`)라, 이름이
뒤바뀌면 **부호가 뒤집힌 것과 똑같은 증상**이 납니다. 실제로 실물 무릎이 "세기를
올려도 안 움직인다" 로 나타났습니다 (`docs/134`).

## 무엇을 바꾸고 무엇을 안 바꾸나

바꿈: 이름과 **이름을 가리키는 참조** — `name=`, `mesh=`, `body1/2=`, `site1/2=`,
`joint=`, `joint1/2=`, `class=`, URDF 의 `<parent link=>` / `<child link=>`.

**안 바꿈**: `file=` 과 `filename=` (실제 STL 파일 이름). 그래서 기하가 그대로
따라옵니다 — 예전에 `name="L_thigh" file="L_thigh.stl"` 이던 자산은
`name="R_thigh" file="L_thigh.stl"` 이 되고, -y 사슬의 geom 은 `mesh="R_thigh"` 로
그 자산을 가리키므로 **같은 STL** 을 씁니다. 파일 이름의 글자는 이제 뜻이 없고
**자산 이름이 기준**입니다.

위치·질량·관성·축·범위는 **한 글자도 안 건드립니다.** 바뀌는 것은 이름표뿐입니다.

## 확인

`--check` 로 무엇이 바뀔지 먼저 봅니다. 바꾼 뒤에는 MuJoCo 로 전후를 불러
**바디 번호별 위치·질량과 관절 번호별 축·범위가 같은지** 비교합니다 (`--verify`).
같지 않으면 되돌려야 합니다.

    swap_leg_names.py --check  <파일...>
    swap_leg_names.py --apply  <파일...>
    swap_leg_names.py --verify <원본> <바꾼것>      # mjlab .venv 파이썬 필요
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_ATTRS = ("file", "filename")
"""이 속성의 값은 건드리지 않습니다. 실제 파일 이름이라 바꾸면 기하가 바뀝니다."""

ATTR = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')
SIDE = re.compile(r'\b([LR])_')


def swap_text(value: str) -> str:
  """값 안의 `L_` <-> `R_` 를 **동시에** 맞바꾼다."""
  return SIDE.sub(lambda m: ("R_" if m.group(1) == "L" else "L_"), value)


def swap_file(text: str) -> tuple[str, int]:
  n = 0

  def one(m: re.Match) -> str:
    nonlocal n
    attr, val = m.group(1), m.group(2)
    if attr in SKIP_ATTRS or not SIDE.search(val):
      return m.group(0)
    n += 1
    return f'{attr}="{swap_text(val)}"'

  return ATTR.sub(one, text), n


def check(path: Path) -> int:
  text = path.read_text(encoding="utf-8")
  new, n = swap_file(text)
  names = sorted(set(re.findall(r'name="([LR]_[^"]*)"', text)))
  print(f"  {path.name:<64} 바꿀 자리 {n:>4}개 · 이름 {len(names)}개")
  return n


HEADER = """<!-- 2026-09-12: 좌우 이름표를 맞바꿨습니다 (tools/robot_model/swap_leg_names.py).
     그 전까지 `L_` 이 붙은 사슬이 **물리적으로는 오른다리**였습니다 (L_hip_pitch_link
     가 -y 에 있었음). 앞은 +x 이고(무릎을 굽히면 발이 -x 로 감), z 가 위인 오른손
     좌표계라 +y 가 왼쪽입니다. 자세한 것은 docs/136.
     기하·질량·관성·축·범위는 한 글자도 안 바꿨습니다. 이름과 그 참조만 바꿨습니다.
     ★ STL 파일 이름은 그대로 둬서, 지금은 `<mesh name="L_..." file="R_....stl">` 처럼
     어긋나 보입니다. **자산 이름(name)이 기준**이고 파일 이름의 글자는 뜻이 없습니다. -->
"""


def apply(path: Path) -> int:
  text = path.read_text(encoding="utf-8")
  if "swap_leg_names.py" in text:
    print(f"  {path.name:<64} 이미 바꿈 (머리말 있음)")
    return 0
  new, n = swap_file(text)
  if n:
    # 머리말은 첫 줄(선언) 바로 뒤에 넣습니다.
    lines = new.split("\n")
    at = 1 if lines and lines[0].lstrip().startswith("<?xml") else 0
    lines.insert(at, HEADER.rstrip("\n"))
    path.write_text("\n".join(lines), encoding="utf-8")
  print(f"  {path.name:<64} {n:>4}자리 바꿈" + ("" if n else " (건드릴 것 없음)"))
  return n


def verify(before: Path, after: Path) -> int:
  """전후 모델이 **물리적으로 같은지** 본다. 이름만 달라야 한다."""
  import mujoco
  import numpy as np

  a = mujoco.MjModel.from_xml_path(str(before))
  b = mujoco.MjModel.from_xml_path(str(after))
  bad = []
  if (a.nbody, a.njnt, a.ngeom) != (b.nbody, b.njnt, b.ngeom):
    print(f"  개수가 다릅니다: {a.nbody, a.njnt, a.ngeom} vs {b.nbody, b.njnt, b.ngeom}")
    return 1
  for i in range(a.nbody):
    if not np.allclose(a.body_pos[i], b.body_pos[i]) or \
       not np.allclose(a.body_mass[i], b.body_mass[i]) or \
       not np.allclose(a.body_inertia[i], b.body_inertia[i]):
      bad.append(f"바디 {i}")
  for i in range(a.njnt):
    if not np.allclose(a.jnt_axis[i], b.jnt_axis[i]) or \
       not np.allclose(a.jnt_range[i], b.jnt_range[i]) or \
       not np.allclose(a.jnt_pos[i], b.jnt_pos[i]):
      bad.append(f"관절 {i}")
  for i in range(a.ngeom):
    if not np.allclose(a.geom_pos[i], b.geom_pos[i]) or \
       not np.allclose(a.geom_size[i], b.geom_size[i]):
      bad.append(f"geom {i}")
  swapped = 0
  for i in range(a.njnt):
    na = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
    nb = mujoco.mj_id2name(b, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
    if na and nb and na != nb:
      swapped += (swap_text(na) == nb)
  if bad:
    print(f"  ★ 물리가 달라졌습니다: {bad[:8]}{' ...' if len(bad) > 8 else ''}")
    return 1
  print(f"  기하·질량·관성·축·범위 전부 같음 · 이름만 바뀐 관절 {swapped}개")
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("--check", action="store_true")
  g.add_argument("--apply", action="store_true")
  g.add_argument("--verify", action="store_true", help="원본 바뀐것 두 개를 준다")
  ap.add_argument("files", nargs="+")
  a = ap.parse_args(argv)

  paths = [Path(p) for p in a.files]
  if a.verify:
    if len(paths) != 2:
      raise SystemExit("--verify 는 파일 두 개(원본, 바꾼 것)를 받습니다")
    return verify(paths[0], paths[1])
  total = 0
  for p in paths:
    if not p.exists():
      print(f"  없음: {p}", file=sys.stderr)
      continue
    total += (check(p) if a.check else apply(p))
  print(f"\n  모두 {total}자리")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
