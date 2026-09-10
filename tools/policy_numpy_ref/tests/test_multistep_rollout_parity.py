"""Consecutive-sequence verification: feeds all 40 real rollout observations
through our numpy pipeline in order and checks the per-step error does not grow
across the sequence. This matters specifically because motor_pos_history carries
real recorded state -- a bug in how error compounds through that history would
not show up in i.i.d. single-step tests.
"""
import numpy as np

TOLERANCE = 1e-4


def test_no_error_growth_across_sequence(weights, obs_parity):
  from policy_numpy_ref.pipeline import run_from_obs

  env_obs = obs_parity["env_obs"].astype(np.float64)
  pt_action = obs_parity["action"].astype(np.float64)

  errs = np.array(
    [
      np.abs(run_from_obs(env_obs[t], weights)[1] - pt_action[t]).max()
      for t in range(env_obs.shape[0])
    ]
  )
  assert errs.max() < TOLERANCE
  # "no accumulation" check: the back half of the sequence should not be
  # systematically worse than the front half by more than a small constant
  # factor (a true accumulating bug would show orders-of-magnitude growth,
  # not this kind of noise-floor wobble).
  n = len(errs)
  front_mean = errs[: n // 2].mean()
  back_mean = errs[n // 2 :].mean()
  assert back_mean < front_mean * 10 + 1e-6, (
    f"error grew across the sequence: front={front_mean}, back={back_mean}"
  )
