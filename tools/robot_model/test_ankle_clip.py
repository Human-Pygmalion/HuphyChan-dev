"""`ankle_clip.py` 의 여섯 가지 성질을 **하드웨어 없이** 확인한다.

2026-09-13 · `docs/139` 8절의 검사 목록 그대로입니다.

    mujoco-sim/mjlab/.venv/bin/python3 -m pytest tools/robot_model/test_ankle_clip.py -q

기구학(HUPHY)이 있어야 돌아갑니다. 없으면 통째로 건너뜁니다 — 이 파일은 개발
기계에서 도는 것이 목적이고, 로봇에서 돌릴 것이 아닙니다.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ankle_clip = pytest.importorskip("ankle_clip")
pytest.importorskip("huphy.kinematics.ankle")


@pytest.fixture(scope="module")
def clipper():
  return ankle_clip.AnkleClipper("left")


# 1. 영역 안 명령은 한 글자도 안 바뀐다
@pytest.mark.parametrize("a,b", [(0.0, 0.0), (5.0, -3.0), (-10.0, 12.0), (20.0, 20.0)])
def test_inside_commands_pass_untouched(clipper, a, b):
  if not clipper.inside(a, b):
    pytest.skip(f"({a},{b}) 는 영역 밖이라 이 검사 대상이 아님")
  r = clipper.clip(a, b)
  assert r.a_out == pytest.approx(a) and r.b_out == pytest.approx(b)
  assert r.stage == "none" and r.inside


# 2. 영역 밖 명령은 **항상** 영역 안으로 나온다
def test_every_outside_command_lands_inside(clipper):
  rng = random.Random(7)
  escaped = []
  for _ in range(400):
    a, b = rng.uniform(-95, 95), rng.uniform(-95, 95)
    r = clipper.clip(a, b)
    if not r.inside:
      escaped.append((round(a, 2), round(b, 2)))
  assert not escaped, f"빠져나간 명령: {escaped[:5]}"


# 3. 경계에 붙어 있어도 **안쪽으로는 그대로 간다** (복귀 보장)
def test_the_way_back_is_never_blocked(clipper):
  edge = clipper.clip(62.0, 40.0)                 # 경계로 밀려난 자리
  assert edge.inside
  inward = clipper.clip(edge.a_out * 0.5, edge.b_out * 0.5)   # 안쪽으로
  assert inward.stage == "none", "안쪽 명령이 잘렸습니다 — 복귀가 막힙니다"
  assert inward.a_out == pytest.approx(edge.a_out * 0.5)


# 4. 표와 정확 기구학이 어긋나지 않는다
def test_table_agrees_with_kinematics(clipper):
  if not clipper.table:
    pytest.skip("표가 없음")
  rng = random.Random(11)
  bad = 0
  for _ in range(120):
    a = rng.uniform(-50, 50)
    span = clipper.table_span(a)
    if span is None:
      continue
    mid = 0.5 * (span[0] + span[1])
    if not clipper.inside(a, mid):                # 표가 "안" 이라는데 기구학은?
      bad += 1
  assert bad == 0, f"표와 기구학이 어긋난 자리 {bad}곳"


# 5. 표에 끊긴 구간이 없다 (두 숫자로 적어도 되는 근거)
def test_table_spans_are_single_intervals(clipper):
  if not clipper.table:
    pytest.skip("표가 없음")
  s = clipper.table["stats"]
  assert s["holes_row"] == 0 and s["holes_col"] == 0


# 6. 기구학이 실패해도 조용히 넘어가지 않는다
def test_kinematics_failure_falls_back_and_is_counted():
  class Broken:
    def solve_fk(self, *a, **k):
      raise RuntimeError("일부러 실패")

    def solve_ik(self, *a, **k):
      raise RuntimeError("일부러 실패")

  c = ankle_clip.AnkleClipper("left", kinematics=Broken())
  r = c.clip(60.0, 60.0)
  assert r.stage in ("table_fallback", "hold_last")
  assert not r.inside                        # 확인할 방법이 없으면 안전을 주장하지 않는다
  assert any("표" in n or "직전" in n for n in r.notes)


# 다중회전 답을 거르는지 (2026-09-13 에 실제로 뚫렸던 자리)
def test_multi_turn_solutions_are_rejected(clipper):
  f = clipper.fk(-36.66, 21.90, guess=(60.0, -10.0))
  assert f is None or (abs(f[0]) <= ankle_clip.POSE_SANE_DEG
                       and abs(f[1]) <= ankle_clip.POSE_SANE_DEG)


# 좌우 표가 거울상인지 -- 한쪽만 구우면 반대 다리가 조용히 틀린다
def test_left_and_right_tables_are_mirrors():
  import json
  root = HERE
  paths = {s: root / f"ankle_ab_envelope_{s}.json" for s in ("left", "right")}
  if not all(p.exists() for p in paths.values()):
    pytest.skip("표 두 개가 다 있어야 함")
  t = {s: json.loads(p.read_text(encoding="utf-8")) for s, p in paths.items()}
  span = {}
  for s, tab in t.items():
    have = [tab["axis"][i] for i, v in enumerate(tab["b_lo"]) if v is not None]
    span[s] = (have[0], have[-1])
  assert span["left"][0] == pytest.approx(-span["right"][1])
  assert span["left"][1] == pytest.approx(-span["right"][0])
