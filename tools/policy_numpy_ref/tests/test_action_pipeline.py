import numpy as np

from policy_numpy_ref.action import ACTION_JOINT_NAMES, action_to_target


def _contract_arrays(contract):
  default_q = np.array([contract["default_q"][j] for j in ACTION_JOINT_NAMES])
  scale = np.array([contract["action_scale"][j] for j in ACTION_JOINT_NAMES])
  clip_lo = np.array([contract["safe_clip"][j][0] for j in ACTION_JOINT_NAMES])
  clip_hi = np.array([contract["safe_clip"][j][1] for j in ACTION_JOINT_NAMES])
  return default_q, scale, clip_lo, clip_hi


def test_zero_action_gives_default_pose_or_its_clip(contract):
  default_q, scale, clip_lo, clip_hi = _contract_arrays(contract)
  target = action_to_target(np.zeros(12), default_q, scale, clip_lo, clip_hi)
  expected = np.clip(default_q, clip_lo, clip_hi)
  assert np.allclose(target, expected)


def test_huge_action_clips_to_bounds(contract):
  default_q, scale, clip_lo, clip_hi = _contract_arrays(contract)
  target_hi = action_to_target(np.full(12, 1e6), default_q, scale, clip_lo, clip_hi)
  target_lo = action_to_target(np.full(12, -1e6), default_q, scale, clip_lo, clip_hi)
  assert np.allclose(target_hi, clip_hi)
  assert np.allclose(target_lo, clip_lo)


def test_no_clip_when_none_given(contract):
  default_q, scale, _, _ = _contract_arrays(contract)
  raw = np.linspace(-2, 2, 12)
  target = action_to_target(raw, default_q, scale, None, None)
  assert np.allclose(target, raw * scale + default_q)


def test_encoder_bias_subtracted_last(contract):
  default_q, scale, clip_lo, clip_hi = _contract_arrays(contract)
  bias = np.full(12, 0.01)
  raw = np.zeros(12)
  target = action_to_target(raw, default_q, scale, clip_lo, clip_hi, encoder_bias=bias)
  expected = np.clip(default_q, clip_lo, clip_hi) - bias
  assert np.allclose(target, expected)


def test_real_rollout_targets_against_live_trace():
  """Cross-checks action_to_target against real, sim-computed processed/clipped
  targets from the live-rollout hook trace (see scripts/hook_live_trace.py). Skips
  if that trace hasn't been generated yet."""
  import pytest

  from pathlib import Path

  trace_path = (
    Path(__file__).resolve().parents[1] / "artifacts" / "legonly_ab_v3_keyfix_live_trace.npz"
  )
  if not trace_path.exists():
    pytest.skip(f"missing {trace_path}, run scripts/hook_live_trace.py first")

  data = np.load(trace_path, allow_pickle=True)
  raw_action = data["raw_action"].astype(np.float64)
  default_q = data["default_q_action"].astype(np.float64)
  scale = data["action_scale"].astype(np.float64)
  clip_lo = data["clip_lo"].astype(np.float64)
  clip_hi = data["clip_hi"].astype(np.float64)
  expected_target = data["final_target"].astype(np.float64)

  for t in range(raw_action.shape[0]):
    target = action_to_target(raw_action[t], default_q, scale, clip_lo, clip_hi)
    assert np.allclose(target, expected_target[t], atol=1e-4), f"step {t} target mismatch"
