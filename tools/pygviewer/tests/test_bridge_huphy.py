"""P3 item 8: the HUPHY UDP adapter's unit/sign/name conversion, checked against the model
contract's own ``travel_sign`` - the exact synthetic case named in the task brief:
a physical +30 deg knee flexion on BOTH legs must land as sim ``L_knee +0.5236`` /
``R_knee -0.5236`` rad (contract travel_sign +1 / -1), to 1e-6.

Biped structure migration (2026-09-04, docs/121 section 12 / docs/123 section 11): HUPHY's
``biped`` branch prefixes limb names with ``Leg.id`` (``left_leg``/``right_leg``, matching
``robot.yaml``'s own ``limbs`` keys), not the bare ``left``/``right`` the pre-biped fork used.
``DEFAULT_MAP_PATH`` now points at ``joint_map_biped.json`` (that new vocabulary); the legacy
``joint_map_huphy.json`` (bare ``left``/``right``) is unchanged on disk and still loadable via
``LEGACY_MAP_PATH`` - most of the tests below exercise the new default, and a dedicated section
near the end re-runs the same shape of check against the legacy map to prove it still works.
"""

import math

import pytest

from pygviewer import CACHE_DIR
from pygviewer.bridge.huphy_udp import (
  DEFAULT_MAP_PATH,
  LEGACY_MAP_PATH,
  HuphyBridge,
  JointMap,
  huphy_deg_s_to_sim_rad_s,
  huphy_deg_to_sim_rad,
  huphy_torque_to_sim,
)
from pygviewer.contract import load_contract

VARIANT = "LegOnly-AB"


def _contract():
  try:
    return load_contract(CACHE_DIR, VARIANT)
  except FileNotFoundError:
    pytest.skip(f"no baked contract for {VARIANT}")


def _legacy_bridge(c):
  """A bridge on the pre-biped ``left``/``right`` vocabulary - used by the legacy-map section
  at the bottom of this file."""
  return HuphyBridge(c, jmap=JointMap(LEGACY_MAP_PATH))


# ----------------------------------------------------------------- biped default (left_leg/right_leg)
def test_joint_map_has_exactly_twelve_motor_rows_and_is_unverified_by_default():
  jmap = JointMap(DEFAULT_MAP_PATH)
  assert len(jmap.motors) == 12
  assert jmap.side_mapping_verified is False
  for limb in ("left_leg", "right_leg"):
    for motor in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"):
      row = jmap.sim_joint(limb, motor)
      assert row["sim_joint"].endswith("_joint")


def test_unknown_limb_motor_pair_is_a_hard_failure_not_a_guess():
  jmap = JointMap(DEFAULT_MAP_PATH)
  with pytest.raises(KeyError):
    jmap.sim_joint("left_leg", "elbow")
  with pytest.raises(KeyError):
    jmap.sim_joint("middle", "knee")
  # the PRE-biped vocabulary is no longer valid against the new default - a caller that still
  # sends bare "left" gets the same hard failure as any other unregistered pair, never a guess.
  with pytest.raises(KeyError):
    jmap.sim_joint("left", "knee")


def test_pure_conversion_matches_the_task_briefs_synthetic_case():
  """Both knees physically flexed +30 deg; map default sign=1/offset=0; travel_sign from the
  contract is +1 (L) / -1 (R) (verified directly against the contract in the test below)."""
  l_rad = huphy_deg_to_sim_rad(30.0, sign=1, offset_rad=0.0, travel_sign=1.0)
  r_rad = huphy_deg_to_sim_rad(30.0, sign=1, offset_rad=0.0, travel_sign=-1.0)
  assert l_rad == pytest.approx(0.5235988, abs=1e-6)
  assert r_rad == pytest.approx(-0.5235988, abs=1e-6)


def test_bridge_end_to_end_knees_in_the_wires_own_mirrored_convention():
  """2026-09-15: the wire is ALREADY mirrored, so the bridge must pass it straight through.

  The robot's own limits are per-side opposite (`left_leg/knee: -117..-3`,
  `right_leg/knee: +3..+117`), matching the model's `L_knee (-120, 0)` /
  `R_knee (0, +120)`. A knee flexed 30 deg therefore arrives as -30 on the left and
  +30 on the right, and each must land unchanged in its own leg's range. Applying the
  contract's travel_sign on top of that negated three joints - see
  `WIRE_IS_ALREADY_MIRRORED`.
  """
  c = _contract()
  bridge = HuphyBridge(c)  # DEFAULT_MAP_PATH -> joint_map_biped.json (left_leg/right_leg)
  assert bridge.travel_sign["L_knee_joint"] == pytest.approx(1.0)
  assert bridge.travel_sign["R_knee_joint"] == pytest.approx(1.0)

  payload = {"t": 0.0, "loop_dt": 10.0, "left_leg/knee/pos": -30.0, "right_leg/knee/pos": 30.0}
  msg = bridge.parse_fast(payload)
  assert msg is not None
  by_name = dict(zip(msg.joint_names, msg.q))
  assert by_name["L_knee_joint"] == pytest.approx(-0.5235988, abs=1e-6)
  assert by_name["R_knee_joint"] == pytest.approx(0.5235988, abs=1e-6)
  # every other joint in this packet is untouched -> null, never a guessed 0.0
  for n, v in by_name.items():
    if n not in ("L_knee_joint", "R_knee_joint"):
      assert v is None


def test_rom_deg_clips_pos_and_tgt_before_conversion_when_set():
  """ROM clip task (2026-09-04): an optional `rom_deg` on a joint-map row clips the incoming
  HUPHY cal-space degrees BEFORE the sim-rad conversion. Both shipped joint maps have
  rom_deg: null on every row (neither rig is commissioned) so this only exercises the
  mechanism via a mutated in-memory JointMap, not a change to the checked-in JSON."""
  c = _contract()
  jmap = JointMap(DEFAULT_MAP_PATH)
  jmap.motors[("left_leg", "knee")]["rom_deg"] = [-10.0, 10.0]
  bridge = HuphyBridge(c, jmap=jmap)
  payload = {"t": 0.0, "left_leg/knee/pos": 30.0, "left_leg/knee/tgt": -30.0}
  msg = bridge.parse_fast(payload)
  by_name = dict(zip(msg.joint_names, msg.q))
  by_name_tgt = dict(zip(msg.joint_names, msg.target))
  assert by_name["L_knee_joint"] == pytest.approx(math.radians(10.0), abs=1e-6)
  assert by_name_tgt["L_knee_joint"] == pytest.approx(math.radians(-10.0), abs=1e-6)
  assert bridge.rom_clamp_count["L_knee_joint"] == 2
  assert any("rom_deg" in w for w in bridge.warnings)


def test_rom_deg_null_is_a_no_op_on_the_shipped_maps():
  """The actual, checked-in behaviour today: rom_deg is null on every row of both maps, so
  this is byte-identical to the pre-rom_deg conversion path."""
  c = _contract()
  bridge = HuphyBridge(c)  # DEFAULT_MAP_PATH, rom_deg: null everywhere
  payload = {"t": 0.0, "left_leg/knee/pos": 1000.0}  # absurd, uncalibrated multi-turn value
  msg = bridge.parse_fast(payload)
  by_name = dict(zip(msg.joint_names, msg.q))
  assert by_name["L_knee_joint"] == pytest.approx(math.radians(1000.0), abs=1e-6)
  assert bridge.rom_clamp_count == {}


def test_bridge_converts_velocity_and_torque_with_the_same_sign():
  c = _contract()
  bridge = HuphyBridge(c)
  payload = {"t": 0.0, "left_leg/knee/vel": 90.0, "right_leg/knee/vel": 90.0,
             "left_leg/knee/tau": 5.0, "right_leg/knee/tau": 5.0}
  msg = bridge.parse_fast(payload)
  by_name_qd = dict(zip(msg.joint_names, msg.qd))
  by_name_tau = dict(zip(msg.joint_names, msg.tau_est))
  # 2026-09-15: rate and torque follow position - the wire is already mirrored, so both
  # legs pass straight through (WIRE_IS_ALREADY_MIRRORED). The point of this test is that
  # vel/tau get the SAME treatment as pos, not that the two legs disagree.
  assert by_name_qd["L_knee_joint"] == pytest.approx(math.radians(90.0), abs=1e-6)
  assert by_name_qd["R_knee_joint"] == pytest.approx(math.radians(90.0), abs=1e-6)
  assert by_name_tau["L_knee_joint"] == pytest.approx(5.0, abs=1e-6)
  assert by_name_tau["R_knee_joint"] == pytest.approx(5.0, abs=1e-6)


def test_minus_one_sentinel_becomes_null_and_warns_after_three():
  c = _contract()
  bridge = HuphyBridge(c)
  for _ in range(3):
    msg = bridge.parse_fast({"t": 0.0, "left_leg/knee/pos": -1})
  assert dict(zip(msg.joint_names, msg.q))["L_knee_joint"] is None
  assert any("3 packets in a row" in w for w in bridge.warnings)


def test_ankle_derived_is_separate_from_the_canonical_actuated_joints():
  c = _contract()
  bridge = HuphyBridge(c)
  payload = {"t": 0.0, "left_leg/ankle_pitch/pos": 10.0, "left_leg/ankle_roll/pos": -5.0}
  msg = bridge.parse_fast(payload)
  assert msg.ankle_derived is not None
  assert msg.ankle_derived["L"]["pitch"] == pytest.approx(math.radians(10.0), abs=1e-6)
  assert msg.ankle_derived["L"]["roll"] == pytest.approx(math.radians(-5.0), abs=1e-6)
  assert "L_crank_A_joint" in msg.joint_names
  assert all(v is None for v in msg.q)  # the FAST packet only carried ankle-joint fields


def test_guard_and_can_fields_are_still_ignored_not_hard_failures():
  """``guard/*``/``can/*`` (and the flat ``missing`` global) are not per-motor DIAG fields
  either - genuinely nothing this bridge interprets. Motor health task (2026-09-04): a
  DIAG_MOTOR_FIELDS key (``temp`` here) is now recognised - see
  test_diag_motor_fields_are_parsed_into_the_joint_state below - so this test narrows to
  only the fields that remain untouched. These key prefixes never reach a jmap lookup at all
  ("guard"/"can" are not in KNOWN_MOTORS/KNOWN_ANKLE_JOINTS), so this is unaffected by which
  joint map is the default."""
  c = _contract()
  bridge = HuphyBridge(c)
  payload = {"t": 0.0, "left_leg/guard/clip_limit": 1.0, "left_leg/can/tx_errors": 0.0, "missing": 0.0}
  assert bridge.parse_fast(payload) is None  # nothing recognised as a fast/diag joint field


def test_diag_motor_fields_are_parsed_into_the_joint_state():
  """Motor health task (2026-09-04): DIAG_MOTOR_FIELDS (temp/age/ack/miss) are now parsed
  the SAME way FAST fields are - accumulated into the persistent per-joint buffer, emitted
  on JointState.temp_c/motor_age_ms/ack/miss (never sign/offset-corrected, they are not
  angles). A negative value (HUPHY's -1 "no data"/"not commanded" sentinel) becomes None,
  same as every other field on this wire, but WITHOUT the 3-in-a-row warning tracking
  ``_sentinel`` applies to pos/tgt/vel/tau (an idle motor legitimately reports age=-1/
  ack=-1 forever - that is not a flapping-connection warning)."""
  c = _contract()
  bridge = HuphyBridge(c)
  payload = {
    "t": 0.0,
    "left_leg/knee/temp": 42.0, "left_leg/knee/age": 3.5, "left_leg/knee/ack": 1.0,
    "left_leg/knee/miss": 0.0,
  }
  msg = bridge.parse_fast(payload)
  assert msg is not None
  i = msg.joint_names.index("L_knee_joint")
  assert msg.temp_c[i] == pytest.approx(42.0)
  assert msg.motor_age_ms[i] == pytest.approx(3.5)
  assert msg.ack[i] == pytest.approx(1.0)
  assert msg.miss[i] == pytest.approx(0.0)
  # every other joint's diag fields stay None - only L_knee_joint's row was in the payload
  j = msg.joint_names.index("R_knee_joint")
  assert msg.temp_c[j] is None
  assert msg.motor_age_ms[j] is None


def test_diag_negative_sentinel_becomes_none_without_a_warning():
  c = _contract()
  bridge = HuphyBridge(c)
  msg = bridge.parse_fast({"t": 0.0, "left_leg/knee/age": -1.0, "left_leg/knee/ack": -1.0})
  assert msg is not None
  i = msg.joint_names.index("L_knee_joint")
  assert msg.motor_age_ms[i] is None
  assert msg.ack[i] is None
  assert not bridge.warnings, "an idle motor's -1 age/ack is a normal steady state, not a warning"


def test_diag_only_packet_still_emits_last_known_fast_values():
  """A diag-only payload (real HUPHY's split-packet design) must still emit a full
  JointState - the persistent per-joint buffer means the LAST fast reading rides along
  with the freshly-updated diag fields, exactly as FAST-only accumulation already worked."""
  c = _contract()
  bridge = HuphyBridge(c)
  bridge.parse_fast({"t": 0.0, "left_leg/knee/pos": 30.0})
  msg = bridge.parse_fast({"t": 0.1, "left_leg/knee/temp": 41.0})
  assert msg is not None
  i = msg.joint_names.index("L_knee_joint")
  assert msg.temp_c[i] == pytest.approx(41.0)
  assert msg.q[i] is not None, "the earlier fast pos reading must still be carried"


def test_imu_packet_prefers_gravity_over_reconstructing_a_quaternion():
  c = _contract()
  bridge = HuphyBridge(c)
  payload = {
    "t": 0.0, "imu/base/gx": 1.0, "imu/base/gy": 0.0, "imu/base/gz": 0.0,
    "imu/base/ax": 0.0, "imu/base/ay": 0.0, "imu/base/az": -9.81,
    "imu/base/grav_x": 0.0, "imu/base/grav_y": 0.0, "imu/base/grav_z": -1.0,
    "imu/base/age": 5.0,
  }
  msg = bridge.parse_imu(payload)
  assert msg is not None
  assert msg.quat_wxyz is None
  assert msg.gravity_b == pytest.approx([0.0, 0.0, -1.0])
  assert msg.gyro_rad_s[0] == pytest.approx(math.radians(1.0))
  assert msg.age_s == pytest.approx(0.005)


def test_huphy_rate_and_torque_helpers_apply_travel_sign_only():
  assert huphy_deg_s_to_sim_rad_s(180.0, sign=1, travel_sign=-1.0) == pytest.approx(-math.pi)
  assert huphy_torque_to_sim(10.0, sign=-1, travel_sign=1.0) == pytest.approx(-10.0)


# ----------------------------------------------------------------- legacy map (left/right)
def test_legacy_map_still_has_twelve_rows_with_the_old_left_right_vocabulary():
  """``joint_map_huphy.json`` predates biped and is kept unchanged on disk - it must keep
  working exactly as before, loaded explicitly (never the default any more)."""
  jmap = JointMap(LEGACY_MAP_PATH)
  assert len(jmap.motors) == 12
  assert jmap.side_mapping_verified is False
  for limb in ("left", "right"):
    for motor in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"):
      row = jmap.sim_joint(limb, motor)
      assert row["sim_joint"].endswith("_joint")
  # and the NEW biped vocabulary is correctly rejected by the OLD map - each map is its own
  # closed, explicit table, never a union of every vocabulary anyone has ever used.
  with pytest.raises(KeyError):
    jmap.sim_joint("left_leg", "knee")


def test_legacy_bridge_end_to_end_knees_in_the_wires_own_mirrored_convention():
  """Same as the non-legacy case above, through the bare left/right map."""
  c = _contract()
  bridge = _legacy_bridge(c)
  payload = {"t": 0.0, "left/knee/pos": -30.0, "right/knee/pos": 30.0}
  msg = bridge.parse_fast(payload)
  assert msg is not None
  by_name = dict(zip(msg.joint_names, msg.q))
  assert by_name["L_knee_joint"] == pytest.approx(-0.5235988, abs=1e-6)
  assert by_name["R_knee_joint"] == pytest.approx(0.5235988, abs=1e-6)
