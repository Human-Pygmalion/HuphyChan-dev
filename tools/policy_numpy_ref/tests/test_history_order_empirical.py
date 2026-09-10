"""Empirically proves the motor_pos_history byte layout by rebuilding obs[6:30]
from raw ingredients recorded in a real 40-step rollout, per obs.py's docstring.
"""
import numpy as np

from policy_numpy_ref.obs import OBS_JOINT_NAMES


def test_joint_order_matches_contract(contract):
  assert list(contract["obs_joint_names"]) == list(OBS_JOINT_NAMES)
  assert list(contract["action_joint_names"]) == list(OBS_JOINT_NAMES)


def test_term_layout_matches_contract(contract):
  terms = contract["obs_terms"]
  assert [t["name"] for t in terms] == [
    "base_ang_vel",
    "projected_gravity",
    "motor_pos_history",
    "actions",
    "command",
  ]
  assert terms[2]["dim"] == 24
  assert terms[2]["history_length"] == 2


def test_flatten_order_is_time_major_not_joint_major(obs_parity):
  env_obs = obs_parity["env_obs"].astype(np.float64)
  motor_q = obs_parity["motor_q"].astype(np.float64)
  default_q_obs = obs_parity["default_q_obs"].astype(np.float64)
  q_rel = motor_q - default_q_obs[None, :]

  # obs[6:18] must be q_rel(t-1); obs[18:30] must be q_rel(t). Step 0 backfills
  # (t-1 == t). Check every step, not just a sample.
  assert np.allclose(env_obs[0, 6:18], q_rel[0])
  for t in range(1, env_obs.shape[0]):
    assert np.allclose(env_obs[t, 6:18], q_rel[t - 1]), f"step {t}: q(t-1) block mismatch"
    assert np.allclose(env_obs[t, 18:30], q_rel[t]), f"step {t}: q(t) block mismatch"


def test_other_term_offsets(obs_parity):
  env_obs = obs_parity["env_obs"].astype(np.float64)
  assert np.allclose(env_obs[:, 0:3], obs_parity["gyro"].astype(np.float64))
  assert np.allclose(env_obs[:, 3:6], -obs_parity["upvector"].astype(np.float64))
  assert np.allclose(env_obs[:, 30:42], obs_parity["last_action"].astype(np.float64))
  assert np.allclose(env_obs[:, 42:45], obs_parity["cmd"].astype(np.float64))
