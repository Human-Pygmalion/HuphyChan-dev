"""Tests obs.py's OWN recursive history bookkeeping (not just re-consuming the
env's pre-built observation, as test_multistep_rollout_parity.py does): drives
build_observation()/ObsHistory across the real 40-step rollout from raw
ingredients and checks the rebuilt observation matches the env's recorded one at
every step, including the backfill on step 0.
"""
import numpy as np

from policy_numpy_ref.obs import ObsHistory, build_observation


def test_rebuilt_observation_matches_env_every_step(obs_parity):
  env_obs = obs_parity["env_obs"].astype(np.float64)
  motor_q = obs_parity["motor_q"].astype(np.float64)
  gyro = obs_parity["gyro"].astype(np.float64)
  upvector = obs_parity["upvector"].astype(np.float64)
  last_action = obs_parity["last_action"].astype(np.float64)
  cmd = obs_parity["cmd"].astype(np.float64)
  default_q_obs = obs_parity["default_q_obs"].astype(np.float64)

  history = ObsHistory(q_default=default_q_obs)
  # float32-storage roundoff floor: obs_parity.npz's arrays were themselves
  # captured from float32 sim tensors, so exact float64 equality isn't
  # expected -- this bound is about 10x float32 eps, not a tolerance we chose
  # to hide a bug.
  tol = 5e-7
  for t in range(env_obs.shape[0]):
    our_obs = build_observation(gyro[t], upvector[t], motor_q[t], history, last_action[t], cmd[t])
    err = np.abs(our_obs - env_obs[t]).max()
    assert err < tol, f"step {t}: obs builder mismatch, max err {err}"


def test_first_step_backfills_history():
  history = ObsHistory(q_default=np.zeros(12))
  q0 = np.arange(12, dtype=np.float64) * 0.1
  obs0 = build_observation(
    np.zeros(3), np.array([0, 0, 1.0]), q0, history, np.zeros(12), np.zeros(3)
  )
  # step 0: q(t-1) block (obs[6:18]) must equal q(t) block (obs[18:30])
  assert np.array_equal(obs0[6:18], obs0[18:30])

  q1 = q0 + 0.01
  obs1 = build_observation(
    np.zeros(3), np.array([0, 0, 1.0]), q1, history, np.zeros(12), np.zeros(3)
  )
  # step 1: q(t-1) block must now be the OLD q_rel, q(t) block the NEW one
  assert np.array_equal(obs1[6:18], q0)
  assert np.array_equal(obs1[18:30], q1)
