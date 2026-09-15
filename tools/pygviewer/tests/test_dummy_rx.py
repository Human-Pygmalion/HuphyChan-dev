"""docs/123 plan A item 4: ``bridge/dummy_rx.py`` end to end over REAL UDP sockets (loopback,
ephemeral ports) - no ``huphy`` needed. ``tx_client.py`` is the sender (also real code, not a
stub), so this exercises the actual wire format both ends will use with a real robot.

Timing note: the physics/telemetry loop runs at 100 Hz by default; these tests poll with a
generous timeout rather than sleeping a fixed guess, so they are not flaky on a loaded CI box
but still fail fast when the behaviour is actually wrong.

Biped structure migration (2026-09-04): ``DummyRx`` builds its ``JointTargetMapper`` with no
explicit joint map, so it picks up whichever ``DEFAULT_MAP_PATH`` the bridge defaults to - now
``joint_map_biped.json`` (``left_leg``/``right_leg``), not the pre-biped ``left``/``right``.
The telemetry key prefixes these tests match against were updated accordingly; nothing in
``dummy_rx.py`` itself needed to change, since it already reads the limb name from the map
rather than hardcoding one.
"""

import json
import math
import socket
import time

import pytest

from pygviewer import CACHE_DIR
from pygviewer.bridge.dummy_rx import DummyRx
from pygviewer.bridge.tx_client import TxClient
from pygviewer.bridge.tx_map import JointTargetMapper, sim_rad_to_cal_deg
from pygviewer.contract import load_contract

VARIANT = "LegOnly-AB"
ARM_TOKEN = "bench-test-token"


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


def _default_deg(c, sim_joint: str) -> float:
  """The same conversion dummy_rx itself uses for its outgoing telemetry - the contract's
  default_q (rad) for ``sim_joint`` re-expressed in HUPHY cal-deg, so a test's "did it return
  to default" check compares against the REAL default (LegOnly-AB uses a bent-knee/bent-hip
  keyframe, not a straight/zero pose - discovered while debugging this file, see docs/123
  section 5) rather than an assumed 0."""
  return _target_deg(c, sim_joint, c.default_q(sim_joint))


def _target_deg(c, sim_joint: str, rad: float) -> float:
  """Same conversion, for an arbitrary commanded rad value rather than the contract default -
  lets a test assert "moved to near the COMMANDED value" instead of a threshold like ">5 deg"
  that a nonzero default pose can already satisfy before anything was ever commanded."""
  mapper = JointTargetMapper(c)
  _limb, _motor, row = mapper.motor_row(sim_joint)
  ts = mapper.travel_sign[sim_joint]
  return sim_rad_to_cal_deg(rad, row["sign"], row["offset_rad"], ts)


def _wait_until(cond, timeout=3.0, interval=0.02):
  deadline = time.monotonic() + timeout
  last = None
  while time.monotonic() < deadline:
    last = cond()
    if last:
      return last
    time.sleep(interval)
  raise AssertionError(f"condition not met within {timeout}s (last={last!r})")


class TelemetryCatcher:
  """Raw UDP listener standing in for a real HUPHY UDP consumer, so a test can inspect exactly
  what dummy_rx put on the wire without depending on huphy_udp.py (that integration is its own
  test file, test_bridge_roundtrip.py)."""

  def __init__(self, port: int):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.sock.bind(("127.0.0.1", port))
    self.sock.settimeout(0.01)

  def latest(self, key_prefix: str) -> dict | None:
    """Drains up to ``_MAX_DRAIN`` currently-queued packets and returns the last one
    containing ``key_prefix``.  Capped, not "until a timeout" - dummy_rx's physics thread
    streams telemetry continuously at 100 Hz for as long as it runs, so an uncapped
    "read until the socket goes quiet" loop never sees the socket go quiet and livelocks.

    ``_MAX_DRAIN`` is deliberately SMALL: dummy_rx sends ~200 packets/s, so draining a large
    cap when the queue is nearly empty turns "drain what's queued" into "block until N MORE
    packets arrive from the still-running sender" - measured at max=64 with a 50ms timeout,
    this cost ~300ms PER CALL (found while debugging test_bridge_roundtrip.py, where the same
    pattern stalled that test's own send loop long enough between ticks to blow past the
    100ms default ttl_ms and spuriously retrigger the deadman - a reader that is too eager to
    drain perturbs the very system it is trying to only observe)."""
    out = None
    for _ in range(self._MAX_DRAIN):
      try:
        data, _ = self.sock.recvfrom(4096)
      except socket.timeout:
        break
      obj = json.loads(data.decode("utf-8"))
      if any(k.startswith(key_prefix) for k in obj):
        out = obj
    return out

  _MAX_DRAIN = 8

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
    # hold_s=0.0: this fixture's tests care about deadman-trip and return-to-default timing,
    # not the (now separate, docs/123 section 5) flat-hold phase - 0 recovers the old
    # "slew starts the instant the deadman trips" behaviour these tests were written against.
    deadman_s=0.15, hold_s=0.0, return_s=0.3, hz=100.0,
  )
  catcher = TelemetryCatcher(tele_port)
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


def test_live_target_moves_the_motor_toward_it(rig):
  c, rx, client, catcher = rig
  target_deg = _target_deg(c, "L_knee_joint", -0.4)
  client.arm()
  client.set_target({"L_knee_joint": -0.4})
  # keep feeding fresh packets faster than the 0.15s deadman for a bit, so it stays "live"
  for _ in range(30):
    client.tick()
    time.sleep(0.02)

  # LegOnly-AB's default pose is already a nonzero bent-knee angle, so ">5 deg" alone would
  # pass even for an untouched joint - compare against the actual COMMANDED degrees instead.
  def _moved():
    pkt = catcher.latest("left_leg/knee/")
    if pkt is None:
      return None
    return pkt if abs(pkt.get("left_leg/knee/pos", 1e9) - target_deg) < 2.0 else None

  pkt = _wait_until(_moved, timeout=3.0)
  assert abs(pkt["left_leg/knee/pos"] - target_deg) < 2.0


def test_deadman_then_return_to_default(rig):
  c, rx, client, catcher = rig
  default_deg = _default_deg(c, "L_knee_joint")  # LegOnly-AB's default is a bent-knee pose,
  # NOT 0 - discovered while writing this test (docs/123 section 5); comparing against a
  # hand-picked 0 here would silently pass for the wrong reason (or fail for the wrong one).
  client.arm()
  client.set_target({"L_knee_joint": -0.5})
  for _ in range(20):
    client.tick()
    time.sleep(0.02)

  # stop sending; the deadman (0.15s) then the return-to-default (0.3s) should land the
  # physics model back within a few degrees of `default_deg`, well before the 2s timeout.
  #
  # `catcher`'s UDP socket was never read during the 0.4s sending loop above, so it is
  # sitting on a backlog of ~80 EARLY packets (dummy_rx streams continuously regardless of
  # whether anything is reading). A bounded `latest()` drains the OLDEST of those first - and
  # the very earliest ones, from right after `set_target`, still show a position close to
  # `default_deg` simply because the PD model had not had time to move away from it yet. That
  # is a coincidence of stale data, not a sign the deadman actually fired - so the predicate
  # ALSO requires `rx.last_phase` to have actually left "live", which only becomes true once
  # the catcher has caught up to now.
  def _back_near_default():
    pkt = catcher.latest("left_leg/knee/")
    if pkt is None or rx.last_phase == "live":
      return None
    return pkt if abs(pkt.get("left_leg/knee/pos", 1e9) - default_deg) < 2.0 else None

  pkt = _wait_until(_back_near_default, timeout=2.0)
  assert abs(pkt["left_leg/knee/pos"] - default_deg) < 2.0
  assert rx.last_phase in ("returning", "default", "hold")


def test_disabled_joint_never_moves(rig):
  c, rx, client, catcher = rig
  # rebuild rx with an enable filter that excludes hip_pitch
  rx.stop()
  rx2 = DummyRx(
    contract=c, listen_host="127.0.0.1", listen_port=rx.listen_port,
    telemetry_host="127.0.0.1", telemetry_port=rx.telemetry_port, arm_token=ARM_TOKEN,
    deadman_s=0.15, return_s=0.3, hz=100.0, enable={"L_knee_joint"},
  )
  rx2.start()
  knee_target_deg = _target_deg(c, "L_knee_joint", -0.4)
  try:
    client.arm()
    client.set_target({"L_knee_joint": -0.4, "L_hip_pitch_joint": 0.4})
    for _ in range(30):
      client.tick()
      time.sleep(0.02)

    def _knee_moved():
      pkt = catcher.latest("left_leg/knee/")
      return pkt if pkt and abs(pkt.get("left_leg/knee/pos", 1e9) - knee_target_deg) < 2.0 else None

    _wait_until(_knee_moved, timeout=3.0)
    hip_pkt = catcher.latest("left_leg/hip_pitch/")
    assert hip_pkt is not None
    hip_default_deg = _default_deg(c, "L_hip_pitch_joint")
    # never commanded (not in --enable) - stayed at its own default, not driven toward the
    # 0.4 rad the message asked for.
    assert abs(hip_pkt["left_leg/hip_pitch/pos"] - hip_default_deg) < 1.0
  finally:
    rx2.stop()


def test_wrong_arm_token_is_rejected_and_does_not_move_the_motor(rig):
  c, rx, client, catcher = rig
  bad_client = TxClient(
    "127.0.0.1", rx.listen_port, joint_names=["L_knee_joint"],
    arm_token="wrong-token", origin="manual", hz=50.0,
  )
  bad_client.arm()
  bad_client.set_target({"L_knee_joint": -0.9})
  try:
    for _ in range(10):
      bad_client.tick()
      time.sleep(0.02)
    time.sleep(0.2)
    assert rx.latest.stats.rejected_arm_token >= 1
    knee_default_deg = _default_deg(c, "L_knee_joint")
    pkt = catcher.latest("left_leg/knee/")
    assert pkt is None or abs(pkt.get("left_leg/knee/pos", knee_default_deg) - knee_default_deg) < 1.0
  finally:
    bad_client.stop()


def test_unknown_joint_name_is_rejected_not_crashed_on():
  c = _contract()
  listen_port = _free_port()
  tele_port = _free_port()
  rx = DummyRx(
    contract=c, listen_host="127.0.0.1", listen_port=listen_port,
    telemetry_host="127.0.0.1", telemetry_port=tele_port, arm_token=ARM_TOKEN,
  )
  rx.start()
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    from pygviewer.schema import to_jsonl

    # hand-build past TxClient's own joint validation to prove the RECEIVER rejects it too
    from pygviewer.schema import JointTarget

    bad = JointTarget(
      t_ns=1, seq=1, joint_names=["totally_made_up_joint"], q_target=[0.1],
      arm_token=ARM_TOKEN, origin="manual",
    )
    sock.sendto(to_jsonl(bad).strip().encode(), ("127.0.0.1", listen_port))
    time.sleep(0.3)
    assert rx.latest.stats.parse_errors >= 1
    assert rx.latest.stats.accepted == 0
  finally:
    sock.close()
    rx.stop()
