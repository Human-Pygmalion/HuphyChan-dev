"""`bridge/guards.py` — HUPHY 에 없는 안전장치 셋을 **하드웨어 없이** 시험한다.

2026-09-13. 이 셋은 `System_ID/huphy_live.py` 에서 옮겨왔고, 거기서는 실물 경로
검사(`tools/huphy_audit/e2e_live.py`)로만 확인하고 있었습니다. 그 검사는 HUPHY 와
가짜 통신선이 있어야 돌아갑니다. 여기서는 **가짜 다리 하나**로 같은 성질을 봅니다 --
개발 기계에서 pytest 한 번에 도는 것이 있어야 옮기다 깨진 것을 바로 압니다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygviewer.bridge import guards  # noqa: E402


# ---------------------------------------------------------------- 가짜 로봇
@dataclass
class FakeGains:
  kp: float = 10.0
  kd: float = 1.0


@dataclass
class FakeMotor:
  id: int
  model: str = "RS04"
  gains: FakeGains = field(default_factory=FakeGains)
  limits_deg: tuple = (-120.0, 120.0)


@dataclass
class FakeConfig:
  motors: dict


class FakeKinematics:
  """발판 각도를 크랭크 각의 **절반 합/차** 로 흉내낸다. 진짜 2-RSU 가 아니다.

  쌍 검사가 "조합을 본다" 는 성질만 보면 되므로, 단조롭고 뒤집기 쉬운 식이면 됩니다.
  """

  def solve_fk(self, a: float, b: float):
    return (0.5 * (a + b), 0.5 * (a - b))


class FakePart:
  def __init__(self, pid="left_leg"):
    self.id = pid
    self.config = FakeConfig(motors={
      "hip_yaw": FakeMotor(3),
      "knee": FakeMotor(4),
      "ankle_a": FakeMotor(5, model="RS03"),
      "ankle_b": FakeMotor(6, model="RS03"),
    })
    self.kinematics = FakeKinematics()


class FakeRobot:
  def __init__(self, *parts):
    self.parts = list(parts)


class Echo:
  """받은 목표를 그대로 돌려주는 `Motion`. 감싸는 쪽만 시험하기 위한 것."""

  def __init__(self, targets):
    self.targets = dict(targets)

  def __call__(self, t, observation):
    return dict(self.targets)


# 과열 차단 시험은 여기 없습니다 — 브리지의 `motor_fault.ThermalCutoff` 가 원본이고
# `tests/test_fault_visibility.py` 가 이미 덮습니다. 2026-09-13 에 제가 그것을
# `guards.py` 로 옮겼다가, 목적지에 이미 있는 것을 확인하고 되물렸습니다.

# ---------------------------------------------------------------- 발목 쌍
class TestAnklePairGuard:
  def _guard(self, inner, **kw):
    return guards.AnklePairGuard(inner, FakeRobot(FakePart()), **kw)

  def test_inside_passes(self):
    inner = Echo({"left_leg/ankle_a": 5.0, "left_leg/ankle_b": -5.0})
    g = self._guard(inner)
    out = g(0.0, {})
    assert out["left_leg/ankle_a"] == 5.0 and out["left_leg/ankle_b"] == -5.0
    assert g.axis_clipped == 0 and g.pair_clipped == 0

  def test_axis_clip(self):
    """축 하나만 봐도 걸리는 경우."""
    inner = Echo({"left_leg/ankle_a": 999.0, "left_leg/ankle_b": 0.0})
    g = self._guard(inner)
    out = g(0.0, {})
    assert abs(out["left_leg/ankle_a"]) <= guards.ANKLE_AXIS_DEG + 1e-6
    assert g.axis_clipped == 1

  def test_pair_clip_when_each_axis_is_legal(self):
    """축마다는 범위 안인데 **조합**이 발판을 ROM 밖으로 보내는 경우.

    이것이 이 가드의 존재 이유입니다 -- HUPHY 는 모터 공간에서 두 각을 따로만
    자르므로 이 경우를 못 잡습니다.
    """
    # 가짜 기구학에서 숙임 = (a+b)/2. 둘 다 축 한계(22.75) 안이지만 합이 ROM 밖.
    inner = Echo({"left_leg/ankle_a": 22.0, "left_leg/ankle_b": 22.0})
    g = self._guard(inner, rom_pitch=(-5.0, 5.0), rom_roll=(-90.0, 90.0))
    out = g(0.0, {})
    pitch, _ = FakeKinematics().solve_fk(out["left_leg/ankle_a"],
                                         out["left_leg/ankle_b"])
    assert pitch <= 5.0 + 1e-6
    assert g.pair_clipped == 1

  def test_both_cranks_retreat_together(self):
    """한쪽만 물러나면 두 로드가 서로 다른 자세를 요구해 기구가 비틀립니다."""
    inner = Echo({"left_leg/ankle_a": 20.0, "left_leg/ankle_b": 20.0})
    g = self._guard(inner, rom_pitch=(-1.0, 1.0), rom_roll=(-90.0, 90.0))
    out = g(0.0, {})
    assert out["left_leg/ankle_a"] == pytest.approx(out["left_leg/ankle_b"])

  def test_a_leg_without_kinematics_is_left_alone(self):
    inner = Echo({"left_leg/ankle_a": 5.0})
    part = FakePart()
    part.kinematics = None
    g = guards.AnklePairGuard(inner, FakeRobot(part))
    assert g(0.0, {})["left_leg/ankle_a"] == 5.0


# ---------------------------------------------------------------- 게인
class TestGainControl:
  def _gc(self, **kw):
    kw.setdefault("kp_cap", 20.0)
    kw.setdefault("kd_cap", 5.0)
    return guards.GainControl(FakeRobot(FakePart()), **kw)

  def test_set_and_read_back(self):
    g = self._gc()
    r = g.set("left_leg/knee", kp=12.0, kd=2.0)
    assert r["ok"] and r["kp"] == 12.0
    assert g.current()["left_leg/knee"]["kp"] == 12.0

  def test_cap_clips_and_says_so(self):
    g = self._gc()
    r = g.set("left_leg/knee", kp=500.0)
    assert r["kp"] == 20.0
    assert any("상한" in n for n in r["notes"])

  def test_wire_value_is_reported(self):
    """12비트 눈금으로 잘린 **실제로 나가는 값**을 같이 알려야 합니다 -- 15 를
    넣으면 모터는 14.65 를 받습니다."""
    g = self._gc()
    r = g.set("left_leg/knee", kp=15.0)
    assert r["kp_on_wire"] < 15.0

  def test_the_zero_trap(self):
    """눈금(약 1.221)보다 작은 세기는 통신선에서 **정확히 0** 이 됩니다.
    약하게 둔 줄 알지만 힘이 아예 안 들어갑니다."""
    g = self._gc()
    r = g.set("left_leg/knee", kp=0.5)
    assert r["kp_on_wire"] == 0.0
    assert any("0" in n for n in r["notes"])

  def test_unknown_joint_is_refused(self):
    assert self._gc().set("left_leg/elbow", kp=1.0)["ok"] is False

  def test_bare_name_resolves_when_unambiguous(self):
    g = self._gc()
    assert g.set("knee", kp=11.0)["ok"] is True

  def test_bare_name_is_refused_when_ambiguous(self):
    g = guards.GainControl(FakeRobot(FakePart("left_leg"), FakePart("right_leg")),
                           kp_cap=20.0, kd_cap=5.0)
    assert g.set("knee", kp=11.0)["ok"] is False
