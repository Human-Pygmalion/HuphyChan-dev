"""Sync-before-arm gate (``hw_sync.py``, docs/123 section 10.2, 2026-09-04).

The near-miss this fixes, verbatim: the real L_knee sat at 27.8 deg while the Joints tab's
manual target - left over from a previous drag, or just the sim's default pose - showed
66.4 deg. Nothing stopped an ARM at that moment from sending that 38.6 deg jump as the very
first TX packet. ``POST /sync_from_real`` (dashboard: "0. sync from hardware") is the missing
step, and ``POST /tx/arm`` now refuses (409) until it has been run and is still valid.

Six scenarios (task item 5):
  (a) no real telemetry has EVER arrived on this process -> 409, no silent success.
  (b) some joints have real data, most don't (the bench today: 1 of 12) -> synced/skipped
      split correctly, never conflated.
  (c) a real value outside the contract's safe_clip (the bench's -44 deg case) -> recorded in
      ``clipped`` with the raw/applied/range, AND the sim's own manual target actually lands
      inside the safe range - never silently pretending sim can reproduce an out-of-range pose.
  (d) arm without ever syncing -> 409 naming the joint(s) that need a sync.
  (e) sync, then arm -> succeeds.
  (f) sync, then the synced joint's telemetry goes stale -> the sync is invalidated -> a
      later arm is refused again, even though nothing else changed.

**2026-09-04 bench-verified bug fix** (scenarios (h)-(k) below): the FIRST live bench run of
this gate found it structurally unarmable. The real bench sat at L_hip_yaw_joint=54.0 deg and
L_knee_joint=171.2 deg, both outside this model's safe_clip range (L_hip_yaw [-40.5, 40.5],
L_knee [6.0, 114.0] deg) - the bench's actual default pose, not an edge case. ``POST
/sync_from_real`` clipped both honestly (applied 40.5/114.0 deg). But the VERY NEXT ``POST
/tx/arm`` was refused 409 "real hardware moved ... 57.2 deg since sync" - **the hardware had
not moved at all**; it sat at 171.2 deg both before and after. The drift check compared live
real telemetry against ``self.synced`` (the CLIPPED/applied value, 114.0), not the real value
that was actually measured (171.2) - a permanent, structural gap for any out-of-range joint
that has nothing to do with the hardware moving. Fixed by storing the RAW real value at sync
time (``HwSyncState.real_at_sync``) and comparing real-vs-real, never real-vs-clipped-target
(see ``hw_sync.py``'s :meth:`HwSyncState.check_arm_ready` docstring). A second, related bug in
the same bench run: dead (physically disconnected) motors still occupy a slot in every HUPHY
JointState frame and report ``q=0.0`` - indistinguishable from a real reading by value alone -
so four unconnected joints got synced to a false "0 deg" target; fixed by consulting the same
``GET /health`` ack/miss/motor_age_ms verdict ``POST /sync_from_real`` already has access to.

  (h) a clipped joint whose real hardware sits perfectly still -> arm now SUCCEEDS (was a
      structural 409 before the fix), and the response's ``sync.clip_warnings`` names the
      real position, the model range, and the travel arming will cause, in deg.
  (i) a joint that ACTUALLY moves on the real hardware after sync (clipped or not) -> arm is
      still refused, and the message's before/after values are the RAW real readings, never
      the clipped target.
  (j) a joint with diag fields (ack/miss/motor_age_ms) verdicting ``dead`` while its position
      field is still fresh (the phantom-0.0 case) -> skipped as "no real data (motor not
      responding)", never synced.
  (k) a joint genuinely never touched by any message at all keeps the plain "no real data"
      reason unchanged - the dead-motor distinction never fires for it.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pygviewer import CACHE_DIR
from pygviewer.api import build_app
from pygviewer.contract import load_contract
from pygviewer.hw_sync import STALE_INVALIDATE_S
from pygviewer.schema import JointState
from pygviewer.sim_core import SimCore

VARIANT = "LegOnly-AB"
STATIC_DIR = Path(__file__).resolve().parents[1] / "pygviewer" / "static"


def _contract():
  try:
    return load_contract(CACHE_DIR, VARIANT)
  except FileNotFoundError:
    pytest.skip(f"no baked contract for {VARIANT}")


def _fresh_core() -> SimCore:
  core = SimCore(_contract(), realtime=False)
  core.reset("knees_bent")
  core.mode = "manual"
  core.step_n(1)
  return core


@pytest.fixture(scope="module")
def core():
  core = _fresh_core()
  yield core
  core.stop()


@pytest.fixture(scope="module")
def client(core):
  return TestClient(build_app(core, core.c.freshness()))


def _ingest(core: SimCore, values: dict[str, float], seq: int = 1) -> None:
  """Feed real telemetry the same way ``WS /ws/in`` would, for exactly the named joints -
  every other actuated joint is left with no data at all, same as the bench today."""
  names = list(values)
  core.real.ingest_joint_state(
    JointState(
      t_ns=time.monotonic_ns(), seq=seq, src="dummy", joint_names=names,
      q=[values[n] for n in names],
    )
  )


def _ingest_diag(
  core: SimCore, joint: str, *, q: float,
  ack: float | None = None, miss: float | None = None, motor_age_ms: float | None = None,
  seq: int = 1,
) -> None:
  """Like :func:`_ingest` but for a single joint carrying HUPHY diag fields - the shape a
  bench-verified "dead motor reporting a phantom position" scenario needs (scenarios (j)/(k)
  below), matching ``test_health.py``'s own ``_feed`` helper's field convention."""
  core.real.ingest_joint_state(
    JointState(
      t_ns=time.monotonic_ns(), seq=seq, src="dummy", joint_names=[joint],
      q=[q],
      ack=([ack] if ack is not None else None),
      miss=([miss] if miss is not None else None),
      motor_age_ms=([motor_age_ms] if motor_age_ms is not None else None),
    )
  )


def _fresh_core_client() -> tuple[SimCore, TestClient]:
  """An isolated core+client pair (unlike the module-scoped ``core``/``client`` fixtures
  scenarios (a)-(g) above share) - scenarios (h)-(k) below need precise control over exactly
  which joints have carried which fields, which is easiest to reason about starting from a
  guaranteed-empty ``RealState``."""
  core = _fresh_core()
  return core, TestClient(build_app(core, core.c.freshness()))


def _configure_tx(client: TestClient, enable: list[str]) -> None:
  """Mode -> config -> enable, in that order (docs/123 section 10.2's own numbering).

  Until 2026-09-07 the order was also forced by ``POST /tx/config`` invalidating any earlier
  sync unconditionally; it now only does so when the host/port actually change (see that
  endpoint's comment and ``test_tx_form_truth.py``), but this remains the natural order."""
  assert client.post("/mode", json={"mode": "manual"}).status_code == 200
  r = client.post("/tx/config", json={"host": "127.0.0.1", "port": 9, "enable": enable})
  assert r.status_code == 200, r.text
  assert client.post("/tx/enable", json={"on": True}).status_code == 200


# ------------------------------------------------------------------------------------- (a)
def test_sync_with_no_telemetry_ever_is_409():
  """A totally fresh process that has never received one ``/ws/in`` message - the 409 case is
  specifically "no stream exists at all", not "most joints have no data" (scenario (b) below,
  the bench's actual normal state, must succeed)."""
  core = _fresh_core()
  client = TestClient(build_app(core, core.c.freshness()))
  try:
    r = client.post("/sync_from_real")
    assert r.status_code == 409, r.text
    assert "real telemetry" in r.json()["detail"].lower()
  finally:
    core.stop()


# ------------------------------------------------------------------------------------- (b)
def test_partial_sync_reports_synced_and_skipped_separately(core, client):
  # 2026-09-15: was +0.5 rad, which L_knee (range -120..0 deg) cannot hold at all - it only
  # survived because the bridge's travel_sign negated it on the way in. With that double
  # negation removed (WIRE_IS_ALREADY_MIRRORED) the value has to be one this joint can
  # actually reach, so the test now uses a real left-knee flexion angle.
  _ingest(core, {"L_knee_joint": -0.5})
  r = client.post("/sync_from_real")
  assert r.status_code == 200, r.text
  body = r.json()
  assert body["synced"] == {"L_knee_joint": pytest.approx(-0.5)}
  assert body["clipped"] == {}
  assert len(body["skipped"]) == len(core.act_names) - 1
  assert body["skipped"]["R_knee_joint"] == "no real data"
  assert all(v == "no real data" for v in body["skipped"].values())
  assert body["sync_token"]
  assert body["max_delta_before"] >= 0.0
  assert isinstance(body["t"], float)


# ------------------------------------------------------------------------------------- (c)
def test_out_of_rom_real_value_is_clipped_and_target_stays_in_range(core, client):
  lo, hi = core.c.clip("L_knee_joint")
  out_of_range = lo - 0.5  # e.g. the bench's real -44 deg vs. a safe_clip floor near +6 deg
  _ingest(core, {"L_knee_joint": out_of_range}, seq=2)
  r = client.post("/sync_from_real")
  assert r.status_code == 200, r.text
  body = r.json()
  c = body["clipped"]["L_knee_joint"]
  assert c["real"] == pytest.approx(out_of_range)
  assert c["applied"] == pytest.approx(lo)
  assert c["range"] == pytest.approx([lo, hi])
  assert body["synced"]["L_knee_joint"] == pytest.approx(lo)  # never handed a value sim can't hold

  core.step_n(core.decimation)  # let the queued /target-equivalent command actually apply
  s = core.snapshot()
  applied_target = dict(zip(s["act_names"], s["target"]))["L_knee_joint"]
  assert lo - 1e-9 <= applied_target <= hi + 1e-9


# ------------------------------------------------------------------------------------- (d)
def test_arm_without_sync_is_409_naming_the_joint(core, client):
  assert client.post("/mode", json={"mode": "manual"}).status_code == 200
  r = client.post("/tx/config", json={"host": "127.0.0.1", "port": 9, "enable": ["L_knee_joint"]})
  assert r.status_code == 200, r.text
  assert client.post("/tx/enable", json={"on": True}).status_code == 200
  # POST /tx/config (just above) invalidates any earlier sync (docs/123 section 10.2) - this
  # process is now guaranteed to be in the "never synced (for this config)" state without
  # depending on test execution order.
  r = client.post("/tx/arm")
  assert r.status_code == 409, r.text
  assert "L_knee_joint" in r.json()["detail"]
  assert client.get("/tx/status").json()["sync"]["valid"] is False


# ------------------------------------------------------------------------------------- (e)
def test_sync_then_arm_succeeds(core, client):
  _ingest(core, {"L_knee_joint": 0.4}, seq=3)
  r = client.post("/sync_from_real")
  assert r.status_code == 200, r.text
  # `SimCore.submit` only queues; the live sim applies the new target on its next CONTROL tick
  # (one per `decimation` physics steps). Without this the target is still the pre-sync value,
  # which is the very race the arm-time jump check exists to catch (2026-09-07).
  core.step_n(core.decimation)
  assert client.get("/tx/status").json()["sync"]["valid"] is True
  r = client.post("/tx/arm")
  assert r.status_code == 200, r.text
  assert r.json()["armed"] is True
  st = client.get("/tx/status").json()
  assert st["sync"]["valid"] is True
  assert st["sync"]["synced_joints"] == ["L_knee_joint"]


# ------------------------------------------------------------------------------------- (f)
def test_stale_telemetry_after_sync_invalidates_and_blocks_arm(core, client):
  assert client.post("/tx/disarm").status_code == 200
  _ingest(core, {"L_knee_joint": 0.35}, seq=4)
  r = client.post("/sync_from_real")
  assert r.status_code == 200, r.text
  assert client.get("/tx/status").json()["sync"]["valid"] is True

  time.sleep(STALE_INVALIDATE_S + 0.1)  # no further telemetry - L_knee_joint goes quiet

  st = client.get("/tx/status").json()
  assert st["sync"]["valid"] is False
  assert "stale" in st["sync"]["reason"].lower()

  r = client.post("/tx/arm")
  assert r.status_code == 409, r.text
  assert "stale" in r.json()["detail"].lower() or "L_knee_joint" in r.json()["detail"]


# ------------------------------------------------------------------------------------- (g)
def test_sync_in_idle_then_switch_to_manual_stays_valid(core, client):
  """Live bench regression (2026-09-04): an operator synced while the sim was still in
  ``idle`` (never having entered manual yet) and the VERY NEXT control tick invalidated the
  sync with reason "mode changed to 'idle' while synced" - nothing had actually left manual,
  the mode simply was never manual to begin with. ``HwSyncState.note_mode`` must only
  invalidate on a TRANSITION away from manual, never on "current mode happens not to be
  manual" (see its own docstring) - sync is allowed in any mode; only arming still requires
  manual, unchanged."""
  assert client.post("/mode", json={"mode": "idle"}).status_code == 200
  core.step_n(core.decimation)  # apply the queued mode op + run the structural note_mode check
  assert core.mode == "idle"

  r = client.post("/tx/config", json={"host": "127.0.0.1", "port": 9, "enable": ["L_knee_joint"]})
  assert r.status_code == 200, r.text
  assert client.post("/tx/enable", json={"on": True}).status_code == 200
  _ingest(core, {"L_knee_joint": 0.3}, seq=5)
  r = client.post("/sync_from_real")
  assert r.status_code == 200, r.text
  assert client.get("/tx/status").json()["sync"]["valid"] is True

  # switch to manual - a normal, expected order (sync first, then go manual, then arm) - must
  # NOT invalidate.
  assert client.post("/mode", json={"mode": "manual"}).status_code == 200
  core.step_n(core.decimation)
  assert core.mode == "manual"
  assert client.get("/tx/status").json()["sync"]["valid"] is True

  r = client.post("/tx/arm")
  assert r.status_code == 200, r.text
  assert r.json()["armed"] is True

  # NOW actually leave manual while armed/synced - this SHOULD invalidate (unchanged rule).
  core.mode = "policy_sim"  # bypasses POST /mode, same pattern test_tx_wiring.py uses
  core.step_n(core.decimation)
  st = client.get("/tx/status").json()
  assert st["sync"]["valid"] is False
  assert "left manual" in st["sync"]["reason"].lower()
  # back to a clean state for any test appended after this one
  core.mode = "manual"
  core.step_n(core.decimation)


# ---------------------------------------------------------- 2026-09-04 bench bug fix (h)-(k)
# ------------------------------------------------------------------------------------- (h)
def test_clipped_joint_with_stationary_hardware_arms_successfully():
  """The bench-verified bug, exactly as it happened: L_knee_joint's real pose (171.2 deg) sat
  outside the model's safe_clip ceiling (114.0 deg / ~1.99 rad), so ``POST /sync_from_real``
  correctly clipped it. Before the fix, ``POST /tx/arm`` then refused (409) EVERY time,
  because the drift check compared live telemetry against the CLIPPED target (114.0 deg) -
  permanently ~57 deg away from the real 171.2 deg reading, with or without the hardware
  moving at all. The hardware here does not move between sync and arm - arm must succeed."""
  core, client = _fresh_core_client()
  try:
    _configure_tx(client, ["L_knee_joint"])
    lo, hi = core.c.clip("L_knee_joint")
    real_value = hi + math.radians(57.2)  # the bench's own out-of-range reading
    _ingest(core, {"L_knee_joint": real_value})
    r = client.post("/sync_from_real")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["clipped"]["L_knee_joint"]["applied"] == pytest.approx(hi)
    assert body["synced"]["L_knee_joint"] == pytest.approx(hi)
    core.step_n(core.decimation)  # let the queued /target-equivalent command actually apply
    # (test_out_of_rom_real_value_is_clipped_and_target_stays_in_range's own pattern) - the
    # arm gate's "did the operator move the target since sync" check reads the LIVE applied
    # target, not the just-submitted op still sitting in the queue.

    r = client.post("/tx/arm")
    assert r.status_code == 200, r.text  # was a structural 409 before the fix
    assert r.json()["armed"] is True
    warnings = r.json()["sync"]["clip_warnings"]
    assert len(warnings) == 1
    w = warnings[0]
    assert w["joint"] == "L_knee_joint"
    assert w["real"] == pytest.approx(real_value, abs=1e-6)
    assert w["applied"] == pytest.approx(hi, abs=1e-6)
    assert w["travel"] == pytest.approx(real_value - hi, abs=1e-6)
    assert "(deg)" in w["message"]
    assert "outside the model range" in w["message"]
    assert "57.2" in w["message"]  # the actual travel distance, not a placeholder
  finally:
    core.stop()


# ------------------------------------------------------------------------------------- (i)
def test_real_hardware_movement_after_sync_still_blocks_using_raw_values():
  """The other half of the fix must not regress: a joint that ACTUALLY moves on the real
  hardware after sync still blocks arm, and the reported before/after values are the RAW real
  readings (real-at-sync -> now) - this test uses an IN-RANGE joint (no clip involved) so it
  isolates "did the fix accidentally stop detecting real movement" from scenario (h)'s
  clip-vs-drift distinction."""
  core, client = _fresh_core_client()
  try:
    _configure_tx(client, ["L_knee_joint"])
    lo, hi = core.c.clip("L_knee_joint")
    real_at_sync = (lo + hi) / 2.0  # comfortably in range - never clipped
    _ingest(core, {"L_knee_joint": real_at_sync}, seq=1)
    r = client.post("/sync_from_real")
    assert r.status_code == 200, r.text
    assert r.json()["clipped"] == {}
    core.step_n(core.decimation)  # let the queued /target-equivalent command actually apply

    moved = real_at_sync + math.radians(6.0)  # > DEFAULT_ARM_DRIFT_LIMIT_RAD (5 deg)
    _ingest(core, {"L_knee_joint": moved}, seq=2)

    r = client.post("/tx/arm")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "real hardware moved" in detail
    assert f"{math.degrees(real_at_sync):.1f}" in detail
    assert f"{math.degrees(moved):.1f}" in detail
  finally:
    core.stop()


# ------------------------------------------------------------------------------------- (j)
def test_dead_motor_phantom_zero_is_skipped_not_synced():
  """HUPHY fills a physically disconnected motor's position with 0.0 in every frame -
  indistinguishable from a real 0 deg reading by value alone (the second bench-verified bug).
  Only the robot's own diag fields say the motor never actually answered: ``ack=0``,
  ``miss>=HEALTH_DEAD_MISS`` verdicts the SAME ``GET /health`` state "dead" that the motor
  health grid already uses. A joint verdicting dead this way, with fresh (non-stale)
  reception, must be excluded from sync rather than handed a false target."""
  from pygviewer.telemetry import HEALTH_DEAD_MISS

  core, client = _fresh_core_client()
  try:
    _ingest_diag(core, "L_hip_pitch_joint", q=0.0, ack=0.0, miss=HEALTH_DEAD_MISS, motor_age_ms=20.0)
    r = client.post("/sync_from_real")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "L_hip_pitch_joint" not in body["synced"]
    assert body["skipped"]["L_hip_pitch_joint"] == "no real data (motor not responding)"
  finally:
    core.stop()


# ------------------------------------------------------------------------------------- (k)
def test_never_touched_joint_keeps_plain_no_real_data_reason():
  """A joint that has never carried ANY field (not even a diag one, unlike scenario (j)) must
  keep the pre-existing plain "no real data" reason - the dead-motor distinction requires
  ``diag: true`` in ``GET /health`` and must never fire for a joint with no data at all."""
  core, client = _fresh_core_client()
  try:
    _ingest(core, {"L_knee_joint": 0.5})  # something must sync, or this hits the 409 (rx_count)
    r = client.post("/sync_from_real")
    assert r.status_code == 200, r.text
    assert r.json()["skipped"]["L_hip_pitch_joint"] == "no real data"
  finally:
    core.stop()


# --------------------------------------------------------------- dashboard.js wiring (text)
# No JS test runner in this repo (test_policy_ui_contract.py's docstring already established
# this - checked again here: no package.json/jest.config under tools/pygviewer) - these check
# the dashboard SOURCE TEXT for the wiring this feature depends on, the same convention
# test_tab_build_flags.py / test_violation_console.py already use for other pure JS pieces.
def _dashboard_js_text() -> str:
  return (STATIC_DIR / "dashboard.js").read_text()


def test_dashboard_js_wires_the_sync_button():
  src = _dashboard_js_text()
  assert 'el("btn-sync-from-real").onclick = doSyncFromReal' in src
  assert "async function doSyncFromReal" in src
  assert 'id="btn-sync-from-real"' in src
  assert "/sync_from_real" in src


def test_dashboard_js_joints_lock_never_fires_without_real_telemetry():
  """jointsLockState's whole point (task requirement 3): a pure-sim session with no real
  telemetry at all must never be locked. Assert the early-return guard is still there
  textually, since this is exactly the kind of one-line safety condition a later edit could
  silently drop."""
  src = _dashboard_js_text()
  assert "function jointsLockState(status, txStatus)" in src
  assert "if (!realConnected) return { locked: false, reason: null };" in src


def test_dashboard_js_arm_state_considers_sync_valid():
  """An invalid sync has to be one of the things that stops an arm, and has to be SAID.

  This used to assert the literal `!syncOk` in the button's `disabled` expression. The button
  is no longer disabled when merely blocked (2026-09-08: a disabled button fires no click, so
  pressing it produced no response at all and the operator had no way to learn why), so the
  test now checks the property instead of the old spelling: the sync feeds the ready/blocked
  decision, and it appears in the reason list an operator actually reads."""
  src = _dashboard_js_text()
  m = re.search(r"const armReady = .*?;", src)
  assert m, "the ARM readiness expression was not found"
  assert "syncOk" in m.group(0), "an invalid sync must still block arming"
  blockers = re.search(r"function txArmBlockers\(tx, st, sync\)\s*\{.*?\n\}", src, re.S)
  assert blockers and "sync.valid" in blockers.group(0), \
    "and the operator has to be told the sync is what is missing"


def test_dashboard_js_arm_shows_clip_warnings():
  """2026-09-04 bench fix: a successful arm's response carries ``sync.clip_warnings`` (see
  ``hw_sync.HwSyncState.clip_warnings``) - the dashboard must surface each one, not just the
  bare "TX armed" toast, or an operator has no way to see "this is about to move 57 deg"
  before it happens."""
  src = _dashboard_js_text()
  m = re.search(r'el\("btn-tx-arm"\)\.onclick = async \(\) => \{(.*?)\n  \};', src, re.S)
  assert m, "btn-tx-arm onclick handler not found"
  body = m.group(1)
  assert "r.sync" in body and "clip_warnings" in body
  assert "forEach" in body
  assert 'el("btn-tx-arm").disabled' in src
