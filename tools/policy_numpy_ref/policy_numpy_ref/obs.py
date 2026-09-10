"""Pure-numpy construction of the 45-D actor observation vector.

Exact term order and byte layout, CONFIRMED EMPIRICALLY (not just from source
reading) against a real 40-step rollout captured by tools/pygviewer's bake tool
-- see tools/policy_numpy_ref/tests/test_history_order_empirical.py, which
reproduces the check below against
/home/syaro/pyg_fea/pygviewer/cache/LegOnly-AB__2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2__model_27999.obs_parity.npz :

    obs[ 0: 3] = base_ang_vel        = gyro                       (robot/imu_ang_vel sensor, rad/s)
    obs[ 3: 6] = projected_gravity   = -upvector                  (robot/imu_upvector sensor, negated framezaxis)
    obs[ 6:18] = motor_pos_history[0] = q(t-1) - default_q, 12 joints, OBS_JOINT_NAMES order
    obs[18:30] = motor_pos_history[1] = q(t)   - default_q, 12 joints, OBS_JOINT_NAMES order
    obs[30:42] = actions             = previous RAW (unscaled) action, 12 joints, same order
    obs[42:45] = command             = (vx, vy, wz) body-frame twist

On the very first control step of an episode the circular history buffer has not
seen a "t-1" yet and backfills it with the current value (mjlab
utils/buffers/circular_buffer.py:210-217): q(t-1) == q(t) on that first call. This
module reproduces that by initializing ObsHistory.q_prev to None and treating
None as "use q_now for q_prev too."

Joint order matches the checkpoint's own baked contract's `obs_joint_names` (which
is identical, element for element, to `action_joint_names` -- confirmed by
diffing the two lists in the .policy_contract.json). It is hardcoded here as a
plain tuple (not read from the contract at runtime) because a *name list* is what
this module needs to be stable code; the corresponding *numbers* (default_q,
safe_clip, action_scale) are NOT hardcoded anywhere in this package -- callers
must load them from the baked policy_contract.json and pass them in. A test
(test_history_order_empirical.py) asserts this tuple still matches a fresh bake,
so a future re-bake with a different joint order fails loudly here rather than
silently producing a wrong observation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBS_JOINT_NAMES: tuple[str, ...] = (
  "L_hip_pitch_joint",
  "L_hip_roll_joint",
  "L_hip_yaw_joint",
  "L_knee_joint",
  "L_crank_A_joint",
  "L_crank_B_joint",
  "R_hip_pitch_joint",
  "R_hip_roll_joint",
  "R_hip_yaw_joint",
  "R_knee_joint",
  "R_crank_A_joint",
  "R_crank_B_joint",
)


@dataclass
class ObsHistory:
  """Carries the one piece of recursive state build_observation() needs: last
  step's relative joint position. Mutated by build_observation() itself so a
  caller can just keep calling it once per control step."""

  q_default: np.ndarray  # (12,) default_joint_pos for OBS_JOINT_NAMES, from the contract
  q_prev_rel: np.ndarray | None = None  # (12,) None until the first build_observation() call

  def reset(self) -> None:
    """Call at the start of a new episode -- next build_observation() call backfills."""
    self.q_prev_rel = None


def build_observation(
  gyro_rad_s: np.ndarray,  # (3,) raw robot/imu_ang_vel sensordata, rad/s
  up_vector: np.ndarray,  # (3,) raw robot/imu_upvector sensordata (framezaxis), BEFORE negation
  q_now: np.ndarray,  # (12,) current joint_pos in OBS_JOINT_NAMES order, rad
  history: ObsHistory,
  last_raw_action: np.ndarray,  # (12,) the RAW (pre scale/offset) action from the PREVIOUS step
  cmd: np.ndarray,  # (3,) (vx, vy, wz)
) -> np.ndarray:
  """Returns the 45-D actor observation as float64, in the exact byte layout above.
  Mutates history.q_prev_rel to q_now_rel for the next call (this is the "backfill
  on the first step, then genuinely delayed by one step" semantics)."""
  q_now_rel = q_now - history.q_default
  q_prev_rel = history.q_prev_rel if history.q_prev_rel is not None else q_now_rel

  obs = np.empty(45, dtype=np.float64)
  obs[0:3] = gyro_rad_s
  obs[3:6] = -up_vector
  obs[6:18] = q_prev_rel
  obs[18:30] = q_now_rel
  obs[30:42] = last_raw_action
  obs[42:45] = cmd

  history.q_prev_rel = q_now_rel
  return obs
