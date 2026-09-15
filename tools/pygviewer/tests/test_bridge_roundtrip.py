"""docs/123 plan A item 7, the consolidated local round-trip verification: ``tx_client.py``
(sine target) -> UDP :9872 -> ``dummy_rx.py`` (arm/deadman/enable + PD physics) -> HUPHY-format
UDP :9870 -> ``huphy_udp.HuphyBridge`` (the viewer's REAL receive adapter, unchanged - this is
what a real robot's telemetry would also go through) -> canonical ``JointState``.

**Precision note, stated explicitly because it does not match the number the task brief
names**: the task text asks for "부호·단위 왕복 1e-6" (sign/unit round trip to 1e-6). The PURE
conversion math (``tx_map.sim_rad_to_cal_deg`` inverting ``huphy_udp.huphy_deg_to_sim_rad``)
already meets that - proven to 1e-9 in ``test_tx_map.py``. But THROUGH the actual UDP wire,
HUPHY's own telemetry format rounds every field to 2 decimal PLACES IN DEGREES (``huphy_udp.py``
module docstring: "소수점 둘째 자리로 반올림" - this is HUPHY's real wire convention,
``dummy_rx.py`` reproduces it faithfully because a real robot's telemetry has the same
limit). 0.01 deg is ~1.75e-4 rad, two orders of magnitude looser than 1e-6. This file checks
the round trip to a tolerance that accounts for that rounding (a few times the 0.01 deg
quantum) and states the achieved number - not a silently loosened "1e-6" claim.
"""

import json
import math
import socket
import time

import pytest

from pygviewer import CACHE_DIR
from pygviewer.bridge.dummy_rx import DummyRx
from pygviewer.bridge.huphy_udp import HuphyBridge
from pygviewer.bridge.tx_client import TxClient
from pygviewer.contract import load_contract
from pygviewer.schema import JointTarget

VARIANT = "LegOnly-AB"
ARM_TOKEN = "roundtrip-verify-token"

# HUPHY's wire rounds to 2 decimal DEGREES; the achievable round-trip bound is that quantum
# plus a small margin for float accumulation across two conversions (rad->deg, deg->rad).
WIRE_ROUNDING_DEG = 0.01
ACHIEVABLE_RAD_TOL = math.radians(WIRE_ROUNDING_DEG) * 3  # ~5.2e-4 rad, 3x margin


def _contract():
  try:
    return load_contract(CACHE_DIR, VARIANT)
  except FileNotFoundError:
    pytest.skip(f"no baked contract for {VARIANT}")


def _free_port() -> int:
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.bind(("127.0.0.1", 0))
  port = s.getsockname()[1]
  s.close()
  return port


def _wait_until(cond, timeout=3.0, interval=0.02):
  deadline = time.monotonic() + timeout
  last = None
  while time.monotonic() < deadline:
    last = cond()
    if last is not None:
      return last
    time.sleep(interval)
  raise AssertionError(f"condition not met within {timeout}s (last={last!r})")


class HuphyBridgeCatcher:
  """Feeds raw UDP packets straight into the REAL ``huphy_udp.HuphyBridge`` (the viewer's
  actual receive adapter) so this test proves the same code path a real robot's telemetry
  goes through, not a reimplementation of it.

  ``pump()``'s cap is deliberately SMALL and its timeout SHORT: dummy_rx streams telemetry
  continuously at ~200 packets/s for as long as it runs, so draining a large cap (64 was
  tried first) with anything but a very short per-read timeout turns "drain what's currently
  queued" into "block until N MORE packets arrive from the still-running sender" - measured
  at ~300ms per call. Called from inside this test's OWN send loop, that stall was long
  enough between ``client.tick()`` calls to blow past the 100ms default ``ttl_ms`` and
  spuriously retrigger dummy_rx's deadman - a reader too eager to drain was perturbing the
  very round trip it was supposed to only observe, not a bug in dummy_rx itself."""

  def __init__(self, port: int, contract):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.sock.bind(("127.0.0.1", port))
    self.sock.settimeout(0.01)
    self.bridge = HuphyBridge(contract)
    self.last_state = None

  def pump(self, max_packets: int = 8):
    for _ in range(max_packets):
      try:
        data, _ = self.sock.recvfrom(4096)
      except socket.timeout:
        break
      payload = json.loads(data.decode("utf-8"))
      state = self.bridge.parse_fast(payload)
      if state is not None:
        self.last_state = state

  def close(self):
    self.sock.close()


@pytest.fixture
def rig():
  c = _contract()
  listen_port = _free_port()
  tele_port = _free_port()
  rx = DummyRx(
    contract=c, listen_host="127.0.0.1", listen_port=listen_port,
    telemetry_host="127.0.0.1", telemetry_port=tele_port, arm_token=ARM_TOKEN,
    deadman_s=0.2, return_s=0.3, hz=100.0,
  )
  catcher = HuphyBridgeCatcher(tele_port, c)
  client = TxClient(
    "127.0.0.1", listen_port, joint_names=list(c.action_joint_names),
    arm_token=ARM_TOKEN, origin="manual", contract=c, hz=50.0,
  )
  rx.start()
  try:
    yield c, rx, client, catcher
  finally:
    client.stop()
    rx.stop()
    catcher.close()


def test_sign_and_unit_round_trip_through_the_real_huphy_udp_adapter(rig):
  """tx_client (rad) -> dummy_rx (rad -> HUPHY cal-deg, 2-decimal wire rounding) ->
  huphy_udp.HuphyBridge (HUPHY cal-deg -> rad) - the recovered JointState.q must match the
  commanded rad to within the wire's own rounding, for BOTH legs' knees (opposite
  travel_sign, same synthetic case ``test_bridge_huphy.py``/``test_tx_map.py`` fix)."""
  c, rx, client, catcher = rig
  # 2026-09-15: was L +0.35 / R -0.20 rad, and NEITHER knee can hold its own value
  # (L_knee range -120..0 deg, R_knee 0..+120). They only round-tripped because
  # travel_sign negated each one twice. With the wire recognised as already mirrored,
  # the round trip has to be exercised with angles each knee can actually reach.
  target_l = -0.35
  target_r = 0.20
  client.arm()
  client.set_target({"L_knee_joint": target_l, "R_knee_joint": target_r})
  # feed it long enough for the PD model to settle close to the target (a few time constants)
  for _ in range(60):
    client.tick()
    time.sleep(0.02)
    catcher.pump()

  def _settled():
    catcher.pump()
    st = catcher.last_state
    if st is None:
      return None
    by_name = dict(zip(st.joint_names, st.q))
    l_ok = by_name.get("L_knee_joint") is not None and abs(by_name["L_knee_joint"] - target_l) < ACHIEVABLE_RAD_TOL
    r_ok = by_name.get("R_knee_joint") is not None and abs(by_name["R_knee_joint"] - target_r) < ACHIEVABLE_RAD_TOL
    return st if (l_ok and r_ok) else None

  st = _wait_until(_settled, timeout=3.0)
  by_name = dict(zip(st.joint_names, st.q))
  assert abs(by_name["L_knee_joint"] - target_l) < ACHIEVABLE_RAD_TOL
  assert abs(by_name["R_knee_joint"] - target_r) < ACHIEVABLE_RAD_TOL


def test_deadman_trigger_timing_is_0_2s_plus_minus_0_05s(rig):
  """Real wall-clock measurement (not the fake-clock unit test in test_remote_target.py) -
  item 7's own precision requirement for the actually-running background thread."""
  c, rx, client, catcher = rig
  client.arm()
  client.set_target({"L_knee_joint": -0.3})
  for _ in range(15):
    client.tick()
    time.sleep(0.02)

  assert rx.last_phase == "live"
  t_last_send = time.monotonic()

  def _left_live():
    return True if rx.last_phase != "live" else None

  _wait_until(_left_live, timeout=1.0, interval=0.005)
  elapsed = time.monotonic() - t_last_send
  assert 0.15 <= elapsed <= 0.25, f"deadman triggered at {elapsed:.3f}s, want 0.2 +- 0.05s"


def test_disabled_motor_is_unaffected_end_to_end(rig):
  c, rx, client, catcher = rig
  rx.stop()
  rx2 = DummyRx(
    contract=c, listen_host="127.0.0.1", listen_port=rx.listen_port,
    telemetry_host="127.0.0.1", telemetry_port=rx.telemetry_port, arm_token=ARM_TOKEN,
    deadman_s=0.2, return_s=0.3, hz=100.0, enable={"L_knee_joint"},
  )
  rx2.start()
  try:
    client.arm()
    client.set_target({"L_knee_joint": -0.3, "L_hip_pitch_joint": 0.3})
    for _ in range(40):
      client.tick()
      time.sleep(0.02)
      catcher.pump()

    def _knee_moved():
      st = catcher.last_state
      if st is None:
        return None
      by_name = dict(zip(st.joint_names, st.q))
      q = by_name.get("L_knee_joint")
      return st if (q is not None and abs(q - -0.3) < ACHIEVABLE_RAD_TOL) else None

    _wait_until(_knee_moved, timeout=3.0)
    st = catcher.last_state
    by_name = dict(zip(st.joint_names, st.q))
    hip_default = c.default_q("L_hip_pitch_joint")
    assert abs(by_name["L_hip_pitch_joint"] - hip_default) < math.radians(2.0)
  finally:
    rx2.stop()


def test_origin_policy_is_refused_before_anything_is_sent():
  """origin=policy never gets far enough to be a wire concern - JointTarget itself refuses
  to be constructed (schema.py), and TxClient refuses even earlier (construction time)."""
  from pydantic import ValidationError

  with pytest.raises(ValidationError):
    JointTarget(
      t_ns=1, seq=1, joint_names=["L_knee_joint"], q_target=[0.1],
      arm_token="x", origin="policy",
    )
  with pytest.raises(RuntimeError):
    TxClient("127.0.0.1", 9872, joint_names=["L_knee_joint"], arm_token="x", origin="policy")
