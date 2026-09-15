"""Sim -> HUPHY direction: canonical sim-joint radians -> HUPHY limb/motor cal-space degrees.

This is the EXACT inverse of ``huphy_udp.py``'s telemetry conversion, and deliberately reuses
its ``JointMap`` (same map file, same 12-row hard-failure-on-unknown table, same
``DEFAULT_MAP_PATH`` - biped's ``joint_map_biped.json`` since 2026-09-04, docs/123 section 11)
so the two directions of the bridge can never independently drift.  It has NO import of
``huphy`` and never touches a socket - it is pure conversion math plus the joint table, so it
is unit-testable on any machine, including this one, without HUPHY installed (docs/123
section 4: the robot host is remote and this module has to be provably correct before it ever
runs there).  Nothing here hardcodes a limb name - ``limb`` is whatever the map file's own
``motors[i]["limb"]`` says (``left_leg``/``right_leg`` for the biped default, ``left``/
``right``/``bench`` for the legacy maps).

Units, once more because getting this wrong is exactly how a sim2real project loses a day:
sim side is ALWAYS rad / rad-s / N*m (schema.py's rule).  HUPHY side is ALWAYS degrees / deg-s
/ N*m, in "cal space" (``cal = sign*raw + offset``, already zero- and sign-corrected for the
physical joint - see ``huphy_udp.py``'s module docstring for the full picture including why
``vel``/``tau`` need the same sign fixup as ``pos`` even though HUPHY's own calibration layer
does not apply it to them).

``travel_sign`` is NEVER guessed here - same as ``huphy_udp.py``, it is read once from the
model contract's ``joint_contract[name]["travel_sign"]`` and only ever looked up.
"""

from __future__ import annotations

import math

from ..contract import ModelContract
from .huphy_udp import (  # noqa: F401  (re-export)
  DEFAULT_MAP_PATH,
  WIRE_IS_ALREADY_MIRRORED,
  JointMap,
  huphy_torque_to_sim,
)

__all__ = [
  "sim_rad_to_cal_deg",
  "sim_rad_s_to_cal_deg_s",
  "sim_torque_to_cal",
  "clamp_gain",
  "JointTargetMapper",
  "UnknownSimJointError",
]


def sim_rad_to_cal_deg(rad: float, sign: int, offset_rad: float, travel_sign: float) -> float:
  """Inverse of ``huphy_udp.huphy_deg_to_sim_rad``.

  Forward:  ``rad = travel_sign * sign * radians(deg) + offset_rad``
  Inverse:  ``deg = degrees((rad - offset_rad) / (travel_sign * sign))``

  ``sign`` and ``travel_sign`` are each +-1, so dividing by their product is the same as
  multiplying by it (``1/x == x`` for ``x in {-1, 1}``) - written as a multiply below so a
  future ``sign``/``travel_sign`` of 0 (a configuration bug, not a valid calibration) raises
  nothing silently wrong instead of a ``ZeroDivisionError`` that looks like a crash in this
  function rather than in the config that produced it.  Verified round-trip with
  ``huphy_deg_to_sim_rad`` to 1e-9 in ``test_tx_map.py``.
  """
  denom = float(travel_sign) * float(sign)
  return math.degrees((rad - float(offset_rad)) * denom)


def sim_rad_s_to_cal_deg_s(rad_s: float, sign: int, travel_sign: float) -> float:
  """Same convention as position, no offset - a rate has no zero-point to correct."""
  return math.degrees(rad_s * float(travel_sign) * float(sign))


# Torque's sign fixup is literally the same multiply-by-(travel_sign*sign) operation in both
# directions (sign, travel_sign in {-1, 1} makes the map its own inverse) - re-exported under
# a name that reads correctly at each call site rather than duplicating the function body.
sim_torque_to_cal = huphy_torque_to_sim


def clamp_gain(value: float, cap: float, *, name: str = "gain") -> tuple[float, list[str]]:
  """Clamp a kp/kd value into ``[0, cap]``.  Returns ``(clamped_value, warnings)``.

  Shared by every place in this bridge that has to enforce ``--kp-max``/``--kd-max``
  (``tx_client.py``, ``huphy_remote_motion.py``, ``dummy_rx.py``) so the three never drift
  into different clamp behaviour.  A negative gain is clamped to 0, not treated as "disable" -
  MIT-mode firmware has no notion of a negative spring/damper, so passing one through would be
  a config bug wearing a control decision's clothes.
  """
  warnings: list[str] = []
  v = float(value)
  if v < 0.0:
    warnings.append(f"{name}={v:g} < 0, clamped to 0")
    v = 0.0
  if v > cap:
    warnings.append(f"{name}={v:g} > cap {cap:g}, clamped to {cap:g}")
    v = cap
  return v, warnings


class UnknownSimJointError(KeyError):
  """A ``JointTarget.joint_names`` entry is not one of the 12 rows in the joint map (default: ``joint_map_biped.json``).

  Hard failure, never a guess - same policy as ``JointMap.sim_joint`` on the receive side."""


class JointTargetMapper:
  """The one place that turns a wire-format ``JointTarget`` (sim names, rad) into
  HUPHY-side ``{limb: {motor_name: cal_deg}}`` - the mirror of ``HuphyBridge`` on the receive
  side, built from the SAME ``JointMap`` and the SAME contract ``travel_sign`` table.

  For the AB (loop-ankle) build, ``L_crank_A_joint``/``L_crank_B_joint`` map straight to the
  ``ankle_a``/``ankle_b`` MOTOR rows in the joint map (default: ``joint_map_biped.json``) - no kinematics here, because
  at this table's level a crank angle already IS the thing the motor reports 1:1 (same as
  ``huphy_udp.py``'s receive side, docs/123 section 4 item 2).  Turning those two motor
  targets into the ``ankle_pitch``/``ankle_roll`` action HUPHY's ``Leg.build_commands``
  actually accepts requires HUPHY's own ``AnkleKinematics.solve_fk`` - that step is
  deliberately NOT in this module (it would force a ``huphy`` import here, breaking "testable
  without HUPHY installed"); it lives in ``huphy_remote_motion.py``, which already needs
  ``huphy`` for everything else.
  """

  def __init__(self, contract: ModelContract, jmap: JointMap | None = None):
    self.jmap = jmap or JointMap(DEFAULT_MAP_PATH)
    # 2026-09-15: 받는 쪽과 **같은 관문**을 쓴다. 전에는 이 줄이 계약을 직접 읽어서,
    # 받기만 고치고 보내기는 안 고친 상태가 될 수 있었다 -- 그러면 명령이 거꾸로 나간다.
    # 로봇 교정(`config/calibration/*_leg_v1.0.json`)이 12개 모터 전부
    # `sign=1.0, offset_deg=0.0` 이고 `limits_deg` 가 MJCF 좌우 범위와 같으므로,
    # 전선과 모형 사이에 곱할 것이 애초에 없다.
    self.travel_sign: dict[str, float] = {
      n: (1.0 if WIRE_IS_ALREADY_MIRRORED
          else float(contract.raw["joint_contract"][n]["travel_sign"]))
      for n in contract.action_joint_names
    }
    self._by_sim_joint: dict[str, tuple[str, str, dict]] = {
      row["sim_joint"]: (limb, motor, row) for (limb, motor), row in self.jmap.motors.items()
    }

  def known_sim_joints(self) -> set[str]:
    return set(self._by_sim_joint)

  def motor_row(self, sim_joint: str) -> tuple[str, str, dict]:
    """``(limb, motor_name, joint_map_row)`` for one sim joint name.  Public accessor used by
    ``dummy_rx.py`` to build OUTGOING telemetry (pos/vel/tau) with the same table and the same
    ``travel_sign`` this class uses for targets - the wire's pos/vel/tau fields live in the
    same cal-deg space as a command, so the conversion function is identical either way."""
    row = self._by_sim_joint.get(sim_joint)
    if row is None:
      raise UnknownSimJointError(f"{sim_joint!r} not in {self.jmap.path}'s motors table")
    return row

  def to_motor_targets(
    self, joint_names: list[str], q_target: list[float]
  ) -> dict[str, dict[str, float]]:
    """``{limb: {motor_name: cal_deg}}``, grouped by limb.  Raises ``UnknownSimJointError``
    for any name not in the joint map's 12 motor rows - a caller that wants to
    accept a partial, filtered set should pre-filter with ``known_sim_joints()`` first
    (exactly what ``huphy_remote_motion.py``'s ``--enable`` list does)."""
    unknown = [n for n in joint_names if n not in self._by_sim_joint]
    if unknown:
      raise UnknownSimJointError(
        f"{unknown} not in {self.jmap.path}'s motors table (known: {sorted(self._by_sim_joint)})"
      )
    out: dict[str, dict[str, float]] = {}
    for name, q in zip(joint_names, q_target):
      limb, motor, row = self._by_sim_joint[name]
      ts = self.travel_sign.get(name)
      if ts is None:
        raise KeyError(f"{name}: no travel_sign in the model contract's joint_contract")
      out.setdefault(limb, {})[motor] = sim_rad_to_cal_deg(q, row["sign"], row["offset_rad"], ts)
    return out
