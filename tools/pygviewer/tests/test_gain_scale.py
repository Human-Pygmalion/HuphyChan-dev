"""Action scale (2026-10-12): one live factor on every joint's Kp AND Kd, applied where the motor
torque is computed.  s = 1.0 must leave the torque bit-identical to the old expression; s = 0.5 must
halve the PD part exactly (checked on a state small enough that no effort / T-N clamp is active)."""

import numpy as np
import pytest

from pygviewer import CACHE_DIR
from pygviewer.contract import load_contract
from pygviewer.sim_core import SimCore

VARIANT = "LegOnly-AB"


@pytest.fixture
def core():
  c = SimCore(load_contract(CACHE_DIR, VARIANT), realtime=False)
  c.reset("knees_bent")
  yield c
  c.stop()


def _old_tau(c):
  """The torque expression exactly as it was before the Action scale existed."""
  q = c.d.qpos[c.a_q]
  qv = c.d.qvel[c.a_d]
  raw = np.clip(c.kp * (c.target - q) - c.kd * qv, -c.eff, c.eff)
  return c._tn_clamp(raw, qv)


def _step_tau(c):
  c._substep()
  return c._tau.copy()


def _state(c, err):
  """A fixed state with a small tracking error and nonzero joint speed."""
  rng = np.random.default_rng(0)
  c.d.qvel[c.a_d] = 0.05 * rng.standard_normal(len(c.a_d))
  c.target = c.d.qpos[c.a_q] + err * rng.standard_normal(len(c.a_q))


def test_scale_one_is_bit_identical(core):
  _state(core, 0.05)
  want = _old_tau(core)
  assert core.gain_scale == 1.0
  got = _step_tau(core)
  assert got.tobytes() == want.tobytes()


def test_scale_half_halves_the_pd_part(core):
  _state(core, 0.005)  # |kp*err| << effort, qv small: no clamp can bite
  q = core.d.qpos[core.a_q]
  qv = core.d.qvel[core.a_d]
  pd_full = core.kp * (core.target - q) - core.kd * qv
  assert np.all(np.abs(pd_full) < core.eff)
  core.set_gain_scale(0.5)
  got = _step_tau(core)
  assert np.allclose(got, 0.5 * pd_full, rtol=0, atol=1e-12)
  assert core.effective_gains()[0] == pytest.approx(0.5 * core.kp)
  assert core.effective_gains()[1] == pytest.approx(0.5 * core.kd)
  # table gains and the policy's joint targets are untouched
  assert core.kp[core.act_names.index("L_knee_joint")] == 220.0


def test_clamps_still_apply_when_scaled(core):
  _state(core, 5.0)  # huge error: effort clamp must cap the scaled torque too
  core.set_gain_scale(0.5)
  got = _step_tau(core)
  assert np.all(np.abs(got) <= core.eff + 1e-12)


def test_scale_via_command_and_range(core):
  core._apply_cmd({"op": "gain_scale", "value": 0.3})
  assert core.gain_scale == 0.3
  for bad in (0.0, 0.04, 1.05, float("nan")):
    with pytest.raises(ValueError):
      core.set_gain_scale(bad)
  assert core.gain_scale == 0.3  # a refused value changes nothing
  assert core.set_gain_scale(1.0) == 1.0 and core.set_gain_scale(0.05) == 0.05


def test_tx_receives_scaled_gains(core):
  seen = {}

  class _Tx:
    enabled = True
    mode_allowed = staticmethod(lambda m: False)
    check_mode_gate = staticmethod(lambda m: None)

    def on_control_tick(self, mode, target, kp, kd):
      seen["kp"], seen["kd"] = kp, kd

  core.tx = _Tx()  # dummy: nothing leaves this process
  core.set_gain_scale(0.5)
  core.mode = "manual"
  core._on_control_tick()
  assert seen["kp"]["L_knee_joint"] == pytest.approx(110.0)
  assert seen["kd"]["L_knee_joint"] == pytest.approx(3.0)
