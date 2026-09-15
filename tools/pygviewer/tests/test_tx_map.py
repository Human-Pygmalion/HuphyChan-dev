"""docs/123 plan A, item 2: the sim-rad -> HUPHY-cal-deg conversion (``bridge/tx_map.py``)
must be the exact numeric inverse of ``bridge/huphy_udp.py``'s receive-side conversion, and
must round-trip both knees' physical +30 deg synthetic case from ``test_bridge_huphy.py`` -
this is the SAME sign/travel_sign check, run backwards, so a future edit that breaks
symmetry between the two directions cannot pass one file's tests while failing the other's.
"""

import math

import pytest

from pygviewer import CACHE_DIR
from pygviewer.bridge.huphy_udp import DEFAULT_MAP_PATH, JointMap, huphy_deg_to_sim_rad
from pygviewer.bridge.tx_map import (
  JointTargetMapper,
  UnknownSimJointError,
  clamp_gain,
  sim_rad_s_to_cal_deg_s,
  sim_rad_to_cal_deg,
)
from pygviewer.contract import load_contract

VARIANT = "LegOnly-AB"


def _contract():
  try:
    return load_contract(CACHE_DIR, VARIANT)
  except FileNotFoundError:
    pytest.skip(f"no baked contract for {VARIANT}")


# --------------------------------------------------------------------------- pure conversion
def test_sim_rad_to_cal_deg_is_the_exact_inverse_of_huphy_deg_to_sim_rad():
  for sign, offset_rad, travel_sign, deg in (
    (1, 0.0, 1.0, 17.3),
    (1, 0.0, -1.0, 17.3),
    (-1, 0.0, 1.0, -42.0),
    (1, math.radians(2.5), -1.0, 8.125),
    (-1, math.radians(-1.1), -1.0, -60.0),
  ):
    rad = huphy_deg_to_sim_rad(deg, sign=sign, offset_rad=offset_rad, travel_sign=travel_sign)
    back_deg = sim_rad_to_cal_deg(rad, sign=sign, offset_rad=offset_rad, travel_sign=travel_sign)
    assert back_deg == pytest.approx(deg, abs=1e-9)


def test_both_knees_plus_30_deg_round_trips_through_sim_rad_and_back():
  """The exact synthetic case ``test_bridge_huphy.py`` fixes forward: physical +30 deg knee
  flexion on both legs -> sim L_knee +0.5235988 / R_knee -0.5235988 rad (map default
  sign=1/offset=0, contract travel_sign +1 / -1). Converting those two rad values back must
  recover +30.0 deg for BOTH legs - if the sign convention were wrong in one direction, one
  leg's degrees would come back as -30.0 instead."""
  l_rad = huphy_deg_to_sim_rad(30.0, sign=1, offset_rad=0.0, travel_sign=1.0)
  r_rad = huphy_deg_to_sim_rad(30.0, sign=1, offset_rad=0.0, travel_sign=-1.0)
  assert l_rad == pytest.approx(0.5235988, abs=1e-6)
  assert r_rad == pytest.approx(-0.5235988, abs=1e-6)

  l_deg = sim_rad_to_cal_deg(l_rad, sign=1, offset_rad=0.0, travel_sign=1.0)
  r_deg = sim_rad_to_cal_deg(r_rad, sign=1, offset_rad=0.0, travel_sign=-1.0)
  assert l_deg == pytest.approx(30.0, abs=1e-6)
  assert r_deg == pytest.approx(30.0, abs=1e-6)


def test_rate_conversion_matches_position_sign_convention():
  deg_s = 90.0
  rad_s = math.radians(deg_s)  # forward is the same formula as position minus the offset
  back = sim_rad_s_to_cal_deg_s(rad_s, sign=1, travel_sign=-1.0)
  assert back == pytest.approx(-deg_s, abs=1e-9)


# --------------------------------------------------------------------------- clamp_gain
def test_clamp_gain_passes_through_in_range_values_with_no_warning():
  v, warns = clamp_gain(3.0, cap=5.0, name="kp")
  assert v == 3.0
  assert warns == []


def test_clamp_gain_clamps_above_cap_with_a_warning():
  v, warns = clamp_gain(12.0, cap=5.0, name="kp")
  assert v == 5.0
  assert len(warns) == 1
  assert "kp" in warns[0] and "5" in warns[0]


def test_clamp_gain_clamps_negative_to_zero():
  v, warns = clamp_gain(-1.0, cap=5.0, name="kd")
  assert v == 0.0
  assert warns


# --------------------------------------------------------------------------- JointTargetMapper
def test_mapper_known_sim_joints_has_all_twelve_motor_rows():
  c = _contract()
  mapper = JointTargetMapper(c)
  known = mapper.known_sim_joints()
  assert len(known) == 12
  assert "L_knee_joint" in known and "R_crank_A_joint" in known


def test_mapper_to_motor_targets_both_knees_plus_30_deg():
  """Biped structure migration (2026-09-04): ``JointTargetMapper()`` with no explicit ``jmap``
  uses ``DEFAULT_MAP_PATH``, which is now ``joint_map_biped.json`` - so the limb keys this
  groups by are ``left_leg``/``right_leg``, matching HUPHY biped's own ``Leg.id`` vocabulary."""
  c = _contract()
  mapper = JointTargetMapper(c)
  # 2026-09-15: the wire is mirrored, same as the model, so a 30 deg flexion of each knee
  # is -30 on the left and +30 on the right. The old expectation (+30 on both, from
  # L_knee = +0.5236 rad) asked L_knee to hold an angle outside its own range of -120..0
  # deg, and only looked right because travel_sign negated it on the way out.
  out = mapper.to_motor_targets(
    ["L_knee_joint", "R_knee_joint"], [-0.5235988, 0.5235988]
  )
  assert out["left_leg"]["knee"] == pytest.approx(-30.0, abs=1e-4)
  assert out["right_leg"]["knee"] == pytest.approx(30.0, abs=1e-4)


def test_mapper_groups_multiple_motors_per_limb():
  c = _contract()
  mapper = JointTargetMapper(c)
  out = mapper.to_motor_targets(
    ["L_hip_pitch_joint", "L_knee_joint"], [0.1, 0.2]
  )
  assert set(out.keys()) == {"left_leg"}
  assert set(out["left_leg"].keys()) == {"hip_pitch", "knee"}


def test_mapper_crank_ab_maps_to_ankle_a_ankle_b_motor_names():
  c = _contract()
  mapper = JointTargetMapper(c)
  out = mapper.to_motor_targets(["L_crank_A_joint", "L_crank_B_joint"], [0.0, 0.0])
  assert set(out["left_leg"].keys()) == {"ankle_a", "ankle_b"}


def test_mapper_rejects_unknown_joint_name_hard_not_a_guess():
  c = _contract()
  mapper = JointTargetMapper(c)
  with pytest.raises(UnknownSimJointError):
    mapper.to_motor_targets(["not_a_real_joint"], [0.0])


def test_mapper_uses_the_same_map_file_as_the_receive_side():
  c = _contract()
  jmap = JointMap(DEFAULT_MAP_PATH)
  mapper = JointTargetMapper(c, jmap=jmap)
  assert mapper.jmap is jmap


def test_mapper_still_works_against_the_legacy_left_right_map():
  """``joint_map_huphy.json`` (pre-biped, bare left/right) is not the default any more but
  must keep grouping correctly when passed explicitly - the mapper hardcodes no vocabulary,
  it reads whatever the map file says."""
  from pygviewer.bridge.huphy_udp import LEGACY_MAP_PATH

  c = _contract()
  mapper = JointTargetMapper(c, jmap=JointMap(LEGACY_MAP_PATH))
  out = mapper.to_motor_targets(["L_knee_joint"], [0.5235988])
  assert out["left"]["knee"] == pytest.approx(30.0, abs=1e-4)
