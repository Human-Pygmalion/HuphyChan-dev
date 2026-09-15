"""HUPHY's 100 Hz UDP JSON telemetry -> canonical ``JointState``/``ImuState``.

HUPHY splits its telemetry into up to three UDP packets per cycle (``telemetry/snapshot.py``
module docstring): FAST (``pos tgt err vel tau`` per motor, every cycle), DIAG (temperature,
CAN health, slow), and IMU.  This module reassembles the fast stream into full canonical
joint vectors and turns the IMU stream into ``ImuState`` - it never touches CAN, never
originates a command, and never guesses a joint it does not recognise.

Units on the HUPHY wire: motor ``pos``/``tgt`` are DEGREES in "cal space" (already sign- and
zero-corrected for the physical joint, per ``leg.py get_observation``), ``vel`` is deg/s and
``tau`` is N*m - but ``vel``/``tau`` are passed straight from the motor driver WITHOUT going
through the same calibration as ``pos`` (``leg.py:370-372``), so the sign fixup this bridge
applies from the joint map must be applied to vel and tau too, not only to pos.  ``-1`` is
HUPHY's "no data" sentinel on every field; it becomes ``null``, and three of the same field
in a row raises a warning (docs/121 section 3).

The mapping table (default: ``joint_map_biped.json``) has exactly 12 rows (2 legs x 6 motors)
plus 4 ankle-joint rows (2 legs x pitch/roll, the FK-derived values HUPHY reports alongside
the crank motors).  A ``(limb, motor)`` pair on the wire that is not one of those rows is a
HARD FAILURE - counted, logged, surfaced in ``Status`` - never a regex match or a silent
default.

Biped structure migration (2026-09-04, docs/121 section 12 / docs/123 section 11): HUPHY's
``biped`` branch prefixes every joint/motor name with the OWNING LIMB's ``Leg.id`` (``robots/
biped.py``: "관절 이름에 팔다리 이름이 붙음" - "right_leg/knee" not "right/knee"), and that
``Leg.id`` comes straight from ``robot.yaml``'s ``limbs`` key. ``joint_map_biped.json`` uses
that vocabulary (``left_leg``/``right_leg``) and is now ``DEFAULT_MAP_PATH``.  Nothing in this
module hardcodes a limb name anywhere - ``JointMap`` reads whatever vocabulary the map FILE
uses, so the wire-format parsing below is unchanged either way.  ``joint_map_huphy.json`` (the
pre-biped fork's bare ``left``/``right`` vocabulary) is kept on disk unchanged and stays
loadable - explicitly, via ``--map`` or ``LEGACY_MAP_PATH`` - because it is still what some
tests and ``joint_map_bench.json`` (its ``bench``-limb variant) reference.

ROM clip task (2026-09-04): each row MAY also carry an optional ``rom_deg: [lo, hi]`` field -
the real hardware's own calibrated ROM (HUPHY ``Motor.limits_deg``), in the same
already-calibrated cal-space degrees ``pos``/``tgt`` arrive in.  ``null`` (both maps ship
with it null on every row, since neither rig is commissioned yet) means "no clip here".
This is defense-in-depth ONLY, layered in FRONT of ``sim_core.py``'s own hard model-range
clip at the qpos-snap point, which stays the actual safety backstop regardless.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from ..schema import ImuState, JointState

DEFAULT_PORT = 9871
DEFAULT_MAP_PATH = Path(__file__).with_name("joint_map_biped.json")
LEGACY_MAP_PATH = Path(__file__).with_name("joint_map_huphy.json")
"""The pre-biped fork's map: same 12+4 rows, bare ``left``/``right`` limb vocabulary. Kept for
anything that still names it explicitly (``joint_map_bench.json``'s own ``bench``-limb variant
predates biped too, and is unaffected by this DEFAULT_MAP_PATH switch either way)."""

FAST_MOTOR_FIELDS = ("pos", "tgt", "err", "vel", "tau")
ANKLE_JOINT_FIELDS = ("pos", "tgt", "err", "vel")
KNOWN_MOTORS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")
KNOWN_ANKLE_JOINTS = ("ankle_pitch", "ankle_roll")

# Motor health task (2026-09-04): HUPHY's own DIAG fields (telemetry/snapshot.py
# DIAG_MOTOR_FIELDS - temp/age/ack/miss) - split into their own, slower UDP packet on real
# HUPHY (module docstring), but a simple single-packet sender (today's bench_telemetry.py)
# may fold them into the SAME payload as the FAST fields. Both shapes work unchanged here:
# they are handled in the SAME per-field dispatch loop as FAST_MOTOR_FIELDS below, so a
# diag-only packet (real HUPHY) or a combined one (bench) both set `touched=True` and
# refresh whichever of temp/age/ack/miss it carries in the SAME persistent per-joint buffer
# FAST already accumulates into - see the class docstring.
DIAG_MOTOR_FIELDS = (
  "temp", "age", "ack", "miss", "stuck", "fault_le", "fault_be", "temp_valid", "cutoff",
  # +-180 fold guard (docs/125 round 3): wrap_blocked says this joint is refusing commands,
  # wrap_margin says how many degrees are left before the fold.
  "wrap_blocked", "wrap_margin",
  # Which robot-side program is driving (scenario.py). Same value on every joint it drives.
  "prog",
)
"""Fault visibility (2026-09-05, docs/121/docs/124) added `stuck`/`fault_le`/`fault_be` to the
original temp/age/ack/miss set - handled the SAME way (never travel-sign/offset corrected,
since none of these is a joint ANGLE; a negative value is treated as "no data" the same as any
other DIAG field, though none of these three is ever legitimately negative in practice).
Unlike temp/age/ack/miss (which come from HUPHY's own Telemetry, split into the slower DIAG
packet), these three are sent by huphy_remote_motion.py's OWN small supplementary UDP sender
(RemoteMotion._send_fault_telemetry) - a SEPARATE datagram to the same destination, since
HUPHY's Telemetry class (never edited, hard project constraint) has no extension point for a
field it does not know about. Both shapes land in the SAME per-field dispatch loop below
either way, so it makes no difference to this bridge which sender a given tick's payload
came from."""


class JointMap:
  """The explicit limb/motor -> sim-joint table.  No fallback: an unlisted pair raises."""

  def __init__(self, path: str | Path = DEFAULT_MAP_PATH):
    self.path = Path(path)
    raw = json.loads(self.path.read_text())
    self.side_mapping_verified = bool(raw.get("side_mapping_verified"))
    self.motors = {(m["limb"], m["motor"]): m for m in raw["motors"]}
    if len(self.motors) != 12:
      raise ValueError(f"{self.path}: expected 12 motor rows (2 legs x 6 motors), found {len(self.motors)}")
    self.ankle_joints = {(a["limb"], a["joint"]): a for a in raw["ankle_joints"]}

  def sim_joint(self, limb: str, motor: str) -> dict:
    row = self.motors.get((limb, motor))
    if row is None:
      raise KeyError(
        f"(limb={limb!r}, motor={motor!r}) is not one of the 12 rows in {self.path} - "
        "hard failure, not a guess (docs/121 section 3)"
      )
    return row

  def ankle_joint(self, limb: str, joint: str) -> dict:
    row = self.ankle_joints.get((limb, joint))
    if row is None:
      raise KeyError(f"(limb={limb!r}, joint={joint!r}) is not in {self.path}'s ankle_joints table")
    return row


WIRE_IS_ALREADY_MIRRORED = True
"""실물이 보내는 각도가 **이미 모형과 같은 좌우 규약**인가? (2026-09-15: 그렇다)

`travel_sign` 은 "이 관절은 어느 쪽으로 길게 도는가"(`test_bake_contract.py` — 범위의
긴 쪽이 음수면 −1)이고, 이 다리의 **거울 규약**을 뜻합니다. 이것을 곱하는 설계는
*실물이 좌우 구분 없이 같은 부호로 보낸다*(양다리 굽힘 = 둘 다 +30) 는 **가정** 위에
서 있었습니다. 그 가정이 틀렸습니다.

실물은 **좌우가 이미 뒤집힌 채로** 보냅니다. 두 가지 독립된 증거:

1. 로봇 제 설정(`robot.yaml` → 텔레메트리 `lo`/`hi`)이 좌우 반대입니다 —
   `left_leg/knee: -117 ~ -3`, `right_leg/knee: +3 ~ +117`. 모형의
   `L_knee (-120, 0)` / `R_knee (0, +120)` 과 **쪽까지 그대로** 맞습니다.
2. 09-15 손 검사 23,660줄: 왼 무릎 −110.9…+9.7, 오른 무릎 −0.5…+120.5.
   좌우가 반대 부호이고, 각각 제 쪽 모형 범위 안입니다.

그래서 `travel_sign` 을 한 번 더 곱하면 **두 번 뒤집는 것**이 됩니다.
실제로 그랬습니다 (실물 생중계, 2026-09-15):

    L_knee       전선 -12.7 도  ->  화면 +0.0 도   (뒤집힌 뒤 위 한계에 걸려 죽음)
    R_hip_pitch  전선  -3.7 도  ->  화면 +3.7 도   (거꾸로)
    L_hip_roll   전선  +7.2 도  ->  화면 -7.2 도   (거꾸로)

`True` 면 이 곱셈을 하지 않습니다. 보내는 쪽(`sim_rad_to_cal_deg`)도 **같은**
`travel_sign` 표를 읽으므로, 여기 한 곳만 바꾸면 받기/보내기가 같이 맞습니다 —
한쪽만 바뀌어 명령이 거꾸로 나가는 일이 없습니다.

되돌리려면 `False` 로 두십시오. 실물의 영점 규약이 바뀌면 이 값을 다시 판단해야 합니다.
"""


def huphy_deg_to_sim_rad(deg: float, sign: int, offset_rad: float, travel_sign: float) -> float:
  """The one conversion this whole bridge exists to get right: HUPHY physical degrees, in
  its own "cal space" convention, into the sim's canonical, travel-signed radians.

  ``sign``/``offset_rad`` come from the joint map (hardware-side calibration); ``travel_sign``
  comes from the MODEL CONTRACT (docs/reward_research/2026-09-03_stiff_knee_root_cause.md) -
  it is never guessed here, only looked up.  Verified: a physical +30 deg knee flexion on
  both legs (map default sign=1, offset=0) converts to sim L_knee +0.5236 / R_knee -0.5236
  rad, matching the contract's travel_sign (+1 / -1) - see test_bridge_huphy.py.
  """
  return travel_sign * float(sign) * math.radians(deg) + float(offset_rad)


def huphy_deg_s_to_sim_rad_s(deg_s: float, sign: int, travel_sign: float) -> float:
  """Same sign convention as position, no offset - a rate has no zero-point to correct."""
  return travel_sign * float(sign) * math.radians(deg_s)


def huphy_torque_to_sim(nm: float, sign: int, travel_sign: float) -> float:
  """Torque already arrives in N*m; only the sign convention needs correcting."""
  return travel_sign * float(sign) * float(nm)


class HuphyBridge:
  """Stateful packet parser: accumulates FAST fields AND DIAG motor-health fields (temp/age/
  ack/miss, 2026-09-04) across packets into a persistent per-joint buffer (HUPHY sends one
  packet per limb, and fast/diag/imu are split further), and emits a full 12-joint
  ``JointState`` whenever a FAST-or-DIAG packet updates anything - a diag-only packet still
  triggers an emission, carrying the LAST KNOWN pos/tgt/vel/tau for a joint alongside its
  freshly-updated temp/age/ack/miss, same accumulation spirit as FAST always had."""

  def __init__(self, contract, jmap: JointMap | None = None):
    self.jmap = jmap or JointMap()
    self.act_names = list(contract.action_joint_names)
    self.travel_sign = {
      n: (1.0 if WIRE_IS_ALREADY_MIRRORED
          else float(contract.raw["joint_contract"][n]["travel_sign"]))
      for n in self.act_names
    }
    self._buf = {
      n: dict(
        q=None, target=None, qd=None, tau=None, temp=None, age=None, ack=None, miss=None,
        stuck=None, fault_le=None, fault_be=None, temp_valid=None, cutoff=None, prog=None,
      )
      for n in self.act_names
    }
    self._ankle_buf: dict[str, dict[str, float]] = {}
    self._seq = 0
    self._minus_one_streak: dict[str, int] = {}
    self.warnings: deque[str] = deque(maxlen=20)
    self.packets_parsed = 0
    self._imu_seq = 0
    # ROM clip task (2026-09-04): a joint-map row MAY carry an optional `rom_deg: [lo, hi]`
    # field - the real hardware's OWN calibrated ROM (HUPHY `Motor.limits_deg`, filled in
    # once the leg has been through `commission sweep`), in the same already-calibrated
    # cal-space degrees `pos`/`tgt` arrive in on this wire. `None` (both joint maps ship
    # with `rom_deg: null` on every row today, since neither rig has been commissioned yet)
    # means "no clip here" - defense-in-depth ONLY, layered in FRONT of SimCore's own hard
    # model-range clip at the qpos-snap point (sim_core.py `_update_replay_targets`), which
    # is the actual safety backstop regardless of whether any bridge sets this.
    self.rom_clamp_count: dict[str, int] = {}
    # Command-link counters as the RECEIVER counts them (docs/127 section 3-1). Latched: the
    # robot sends them at a low rate, and the last reported value stays current until a newer
    # one arrives - a JointState built between two reports must still carry them, or the
    # viewer's command-path trace would blink between "accepted 1722" and "not reported".
    self._link_stats: dict[str, float] = {}
    self.imu_quat_rejects = 0

  def _clip_rom_deg(self, sim_joint: str, row: dict, value: float) -> float:
    rom = row.get("rom_deg")
    if rom is None:
      return value
    lo, hi = float(rom[0]), float(rom[1])
    clipped = min(max(value, lo), hi)
    if clipped != value:
      n = self.rom_clamp_count.get(sim_joint, 0) + 1
      self.rom_clamp_count[sim_joint] = n
      if n == 1:
        self.warnings.append(f"{sim_joint}: rom_deg {value:.2f} -> {clipped:.2f} deg")
    return clipped

  # ---------------------------------------------------------------------- link
  LINK_COUNTERS = (
    "accepted", "rejected_seq", "rejected_arm_token", "rejected_contract",
    "parse_errors", "seq_restarts",
  )
  """What the command receiver counts, per ``bridge/remote_target.py``'s ``LatestOnly.stats``.

  These answer the question that has needed a terminal on the robot every single time so far:
  did my commands even arrive, and if they were dropped, which gate dropped them? The robot
  has always kept them and printed them to its own log; nothing carried them here (docs/127
  section 3-1). ``rejected_seq`` in particular is the counter that read 5823 against 843
  accepted while a whole measurement run silently did nothing.
  """

  def parse_link(self, payload: dict) -> bool:
    """Absorb ``link/<counter>`` keys. Returns whether any were present.

    Deliberately a separate two-segment namespace rather than a per-motor diag field: these
    are properties of the LINK, not of a joint, and smuggling them onto one motor's row would
    make them look joint-specific and break the moment that motor is not in the enable list.
    ``parse_fast`` ignores them for free - it skips every key that is not ``limb/motor/field``.
    """
    seen = False
    for key, value in payload.items():
      parts = key.split("/")
      if len(parts) != 2 or parts[0] != "link":
        continue
      if parts[1] not in self.LINK_COUNTERS:
        continue          # an unknown counter is ignored, never guessed at
      if value is None:
        continue
      self._link_stats[parts[1]] = float(value)
      seen = True
    return seen

  # ---------------------------------------------------------------------- fast
  def parse_fast(self, payload: dict) -> JointState | None:
    self.parse_link(payload)   # link counters may ride along in any packet
    touched = False
    for key, value in payload.items():
      parts = key.split("/")
      if len(parts) != 3:
        continue
      limb, motor, field = parts
      if limb == "imu":
        continue  # handled by parse_imu - not a joint value
      if motor in KNOWN_MOTORS:
        if field == "err":
          continue  # err is target-minus-pos, derivable; not carried on the canonical wire
        if field in DIAG_MOTOR_FIELDS:
          row = self.jmap.sim_joint(limb, motor)  # KeyError propagates: hard failure, not a guess
          sim_joint = row["sim_joint"]
          touched = True
          # Motor health (2026-09-04): temp/age/ack/miss are NEVER travel-sign/offset
          # corrected (they are not joint ANGLES) and never routed through `_sentinel`'s
          # 3-in-a-row WARNING tracking - unlike a missing pos/tgt sample, "age=-1" (never
          # responded) or "ack=-1" (not commanded) are legitimate, common steady states for
          # an idle/unconnected motor, not evidence of a flapping connection worth a warning
          # log line every time. Just the uniform "-1 (or any negative) means null" wire
          # convention (schema.py's rule), silently.
          self._buf[sim_joint][field] = None if value is None or value < 0 else float(value)
          continue
        if field not in FAST_MOTOR_FIELDS:
          continue  # guard/*, can/* etc - genuinely not interpreted here
        row = self.jmap.sim_joint(limb, motor)  # KeyError propagates: hard failure, not a guess
        sim_joint = row["sim_joint"]
        touched = True
        if self._sentinel(f"{limb}/{motor}/{field}", value):
          continue
        ts = self.travel_sign[sim_joint]
        if field == "pos":
          value = self._clip_rom_deg(sim_joint, row, value)
          self._buf[sim_joint]["q"] = huphy_deg_to_sim_rad(value, row["sign"], row["offset_rad"], ts)
        elif field == "tgt":
          value = self._clip_rom_deg(sim_joint, row, value)
          self._buf[sim_joint]["target"] = huphy_deg_to_sim_rad(value, row["sign"], row["offset_rad"], ts)
        elif field == "vel":
          self._buf[sim_joint]["qd"] = huphy_deg_s_to_sim_rad_s(value, row["sign"], ts)
        elif field == "tau":
          self._buf[sim_joint]["tau"] = huphy_torque_to_sim(value, row["sign"], ts)
      elif motor in KNOWN_ANKLE_JOINTS:
        if field != "pos":
          continue  # only the FK-derived angle is carried in ankle_derived
        row = self.jmap.ankle_joint(limb, motor)
        touched = True
        if self._sentinel(f"{limb}/{motor}/{field}", value):
          continue
        side_buf = self._ankle_buf.setdefault(row["side"], {})
        side_buf[row["field"]] = float(row["sign"]) * math.radians(value) + float(row["offset_rad"])
      # anything else (diag: temp/age/ack/miss, guard/*, can/*) is intentionally not
      # interpreted here - it is not a canonical joint value, so there is nothing to convert
    if not touched:
      return None
    self.packets_parsed += 1
    return self._emit_joint_state()

  def _emit_joint_state(self) -> JointState:
    self._seq += 1
    return JointState(
      t_ns=time.monotonic_ns(),
      seq=self._seq,
      src="real",
      contract_hash=None,  # hardware has no notion of a baked sim contract
      joint_names=list(self.act_names),
      q=[self._buf[n]["q"] for n in self.act_names],
      qd=[self._buf[n]["qd"] for n in self.act_names],
      tau_est=[self._buf[n]["tau"] for n in self.act_names],
      target=[self._buf[n]["target"] for n in self.act_names],
      temp_c=[self._buf[n]["temp"] for n in self.act_names],
      motor_age_ms=[self._buf[n]["age"] for n in self.act_names],
      ack=[self._buf[n]["ack"] for n in self.act_names],
      miss=[self._buf[n]["miss"] for n in self.act_names],
      ankle_derived=({s: dict(v) for s, v in self._ankle_buf.items() if v} or None),
      stuck=[self._buf[n]["stuck"] for n in self.act_names],
      fault_le=[self._buf[n]["fault_le"] for n in self.act_names],
      fault_be=[self._buf[n]["fault_be"] for n in self.act_names],
      temp_valid=[self._buf[n]["temp_valid"] for n in self.act_names],
      cutoff=[self._buf[n]["cutoff"] for n in self.act_names],
      prog=[self._buf[n]["prog"] for n in self.act_names],
      link_stats=(dict(self._link_stats) or None),
    )

  # ---------------------------------------------------------------------- imu
  def parse_imu(self, payload: dict) -> ImuState | None:
    """``-1`` is HUPHY's "missing" sentinel only on ``age``/``sensor_dt`` fields
    (``IMU_MISSING_IS_MINUS_ONE`` in ``telemetry/snapshot.py``) - everything else (gyro,
    accel, ``grav_*``) is blanked to 0.0 when unknown, which is indistinguishable from a
    real zero on HUPHY's own wire and not something this bridge can fix.  Applying the -1
    check to ``grav_z`` too would be a real bug: -1.0 is exactly what an upright, standing
    robot reports there, and this bridge nulled out the single most common IMU reading
    before this was caught by ``test_bridge_huphy.py``."""
    touched = False
    gx = gy = gz = ax = ay = az = gvx = gvy = gvz = age_ms = None
    qw = qx = qy = qz = None
    for key, value in payload.items():
      parts = key.split("/")
      if len(parts) != 3 or parts[0] != "imu":
        continue
      _, _name, field = parts
      touched = True
      if field == "age":
        if self._sentinel(key, value):
          continue
        age_ms = value
        continue
      if field == "gx":
        gx = math.radians(value)
      elif field == "gy":
        gy = math.radians(value)
      elif field == "gz":
        gz = math.radians(value)
      elif field == "ax":
        ax = value
      elif field == "ay":
        ay = value
      elif field == "az":
        az = value
      elif field == "grav_x":
        gvx = value
      elif field == "grav_y":
        gvy = value
      elif field == "grav_z":
        gvz = value
      elif field == "qw":
        qw = value
      elif field == "qx":
        qx = value
      elif field == "qy":
        qy = value
      elif field == "qz":
        qz = value
    if not touched:
      return None
    self._seq += 1
    return ImuState(
      t_ns=time.monotonic_ns(),
      seq=self._seq,
      src="real",
      contract_hash=None,
      # The quaternion, which used to be dropped here (2026-09-08).
      #
      # The old note said HUPHY has already reordered its sensor's native (z,y,x,w)
      # quaternion, so re-deriving one here would risk a second, independent reordering bug,
      # and left this null "unless a future need requires it". The need arrived: the
      # dashboard's real-IMU axes need a full attitude, and gravity alone fixes only two of
      # the three degrees of freedom - there is no yaw in it, so the axes cannot be drawn.
      #
      # The risk the note worried about is handled by CHECKING rather than by trusting.
      # `_imu_quat_ok` re-derives gravity from these very components using HUPHY's own
      # formula (`sensors/base.py::gravity_from_quat`) and compares it against the grav_*
      # fields HUPHY computed itself. Agreement means the ordering is right; disagreement
      # nulls the quaternion and warns, so a reordering bug can never reach the screen
      # silently. Verified against 40 live packets from this bench: worst component
      # difference 0.008, which is the wire's own two-decimal rounding.
      quat_wxyz=self._imu_quat_ok(qw, qx, qy, qz, gvx, gvy, gvz),
      gyro_rad_s=([gx, gy, gz] if None not in (gx, gy, gz) else None),
      acc_m_s2=([ax, ay, az] if None not in (ax, ay, az) else None),
      gravity_b=([gvx, gvy, gvz] if None not in (gvx, gvy, gvz) else None),
      age_s=(age_ms / 1e3 if age_ms not in (None, -1) else None),
    )

  IMU_QUAT_GRAVITY_TOL = 0.05
  """How far the quaternion-derived gravity may sit from HUPHY's own before the quaternion is
  rejected. The wire carries two decimals, so agreement lands near 0.008 on this bench; 0.05
  passes that comfortably while catching any component-ordering or sign error, all of which
  move a component by far more."""

  def _imu_quat_ok(self, qw, qx, qy, qz, gvx, gvy, gvz) -> list[float] | None:
    """The quaternion, but only if it agrees with the gravity HUPHY computed from it.

    HUPHY's own ``sensors/base.py::gravity_from_quat`` is ``g = -(third ROW of R)`` for the
    standard (w, x, y, z) body-to-world rotation:

        g = ( 2(wy - xz),  -2(yz + wx),  2(x^2 + y^2) - 1 )

    Re-deriving it here and comparing against the grav_* fields on the same packet turns "is
    the component order right?" from a thing to be trusted into a thing that is measured, on
    every packet. Getting this wrong is not hypothetical: a first attempt at this check used
    the third COLUMN instead of the third row and disagreed on X by 0.59, which reads exactly
    like a sensor sign fault and is not one.

    ``None`` (and a warning) when the check fails or a field is missing, so a wrong attitude
    is never drawn as if it were right.
    """
    if None in (qw, qx, qy, qz):
      return None
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if not (0.9 < n < 1.1):
      self.warnings.append(f"imu quaternion is not a unit quaternion (|q| = {n:.3f})")
      return None
    if None in (gvx, gvy, gvz):
      return [w, x, y, z]      # nothing to check against; the fields are named, so take them
    derived = (2.0*(w*y - x*z), -2.0*(y*z + w*x), 2.0*(x*x + y*y) - 1.0)
    worst = max(abs(a - b) for a, b in zip(derived, (gvx, gvy, gvz)))
    if worst > self.IMU_QUAT_GRAVITY_TOL:
      if self.imu_quat_rejects == 0:
        self.warnings.append(
          f"imu quaternion disagrees with the reported gravity by {worst:.3f} "
          f"(derived {tuple(round(v,3) for v in derived)} vs sent "
          f"{(round(gvx,3), round(gvy,3), round(gvz,3))}) - attitude not shown"
        )
      self.imu_quat_rejects += 1
      return None
    return [w, x, y, z]

  # ---------------------------------------------------------------------- sentinel tracking
  def _sentinel(self, field: str, value) -> bool:
    """True if ``value`` is HUPHY's -1 "no data" sentinel; tracks 3-in-a-row for a warning."""
    is_missing = value == -1
    n = self._minus_one_streak.get(field, 0) + 1 if is_missing else 0
    self._minus_one_streak[field] = n
    if n == 3:
      self.warnings.append(f"{field}: -1 (missing) for 3 packets in a row")
    return is_missing


class HuphyUdpReceiver:
  """Owns the UDP socket and a background thread.  Never raises into the caller: a hard
  mapping failure or a malformed packet is counted (``core.real.note_bridge_error``) and the
  next packet is still processed - one bad line from the robot must not take the bridge
  down, the same "send and forget, never block" spirit as HUPHY's own ``UdpSink``."""

  def __init__(
    self,
    core,
    port: int = DEFAULT_PORT,
    host: str = "0.0.0.0",
    jmap_path: str | Path | None = None,
  ):
    self.core = core
    self.bridge = HuphyBridge(core.c, JointMap(jmap_path) if jmap_path else JointMap())
    self.port = port
    self.host = host
    self._sock: socket.socket | None = None
    self._thread: threading.Thread | None = None
    self._running = False
    self.packets = 0
    self.hard_failures = 0
    self.last_error: str | None = None

  def start(self) -> None:
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._sock.bind((self.host, self.port))
    self._sock.settimeout(0.5)
    self._running = True
    self._thread = threading.Thread(target=self._loop, name="pygviewer-huphy-udp", daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._running = False
    if self._thread is not None:
      self._thread.join(timeout=2.0)
      self._thread = None
    if self._sock is not None:
      self._sock.close()
      self._sock = None

  def _loop(self) -> None:
    while self._running:
      try:
        data, _addr = self._sock.recvfrom(4096)
      except socket.timeout:
        continue
      except OSError:
        break
      self.packets += 1
      try:
        payload = json.loads(data.decode("utf-8"))
      except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        self.hard_failures += 1
        self.last_error = f"bad packet: {exc}"
        self.core.real.note_bridge_error(self.last_error)
        continue
      try:
        if any(k.startswith("imu/") for k in payload):
          msg = self.bridge.parse_imu(payload)
          if msg is not None:
            self.core.real.ingest_imu_state(msg)
        else:
          msg = self.bridge.parse_fast(payload)
          if msg is not None:
            self.core.real.ingest_joint_state(msg)
      except KeyError as exc:
        self.hard_failures += 1
        self.last_error = str(exc)
        self.core.real.note_bridge_error(self.last_error)


def main(argv: list[str] | None = None) -> int:
  """Standalone launcher: ``run.py bridge huphy --variant ... [--port 9871]``.  Runs the
  receiver against a live SimCore (headless, no viser/API) until Ctrl-C."""
  import argparse

  from .. import CACHE_DIR, VARIANTS
  from ..contract import load_contract
  from ..sim_core import SimCore

  ap = argparse.ArgumentParser(prog="pygviewer bridge huphy")
  ap.add_argument("--variant", default="LegOnly-AB", choices=list(VARIANTS))
  ap.add_argument("--cache", default=CACHE_DIR)
  ap.add_argument("--port", type=int, default=DEFAULT_PORT)
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--map", default=None)
  a = ap.parse_args(argv)

  c = load_contract(a.cache, a.variant)
  core = SimCore(c, realtime=True)
  core.start()
  recv = HuphyUdpReceiver(core, port=a.port, host=a.host, jmap_path=a.map)
  recv.start()
  print(f"HUPHY UDP bridge listening on {a.host}:{a.port} for {a.variant}, feeding SimCore")
  try:
    while True:
      time.sleep(1.0)
  except KeyboardInterrupt:
    pass
  finally:
    recv.stop()
    core.stop()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
