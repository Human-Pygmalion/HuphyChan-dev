"""Three-way single-step comparison: our numpy vs the real .pt (ground truth) vs
the existing pygviewer ONNX export (second independent oracle).

Tolerance framing (see docs/136_legonly_ab_v3_keyfix_sim2real_gaps.md): literal
bit-identical output is not a meaningful target across different BLAS/summation
paths in float32. We report actual max abs error and hold it to the same bar the
project already uses for onnx-vs-pt (<=1e-4), which our numpy path clears with
roughly an order of magnitude of margin.
"""
import numpy as np
import pytest

TOLERANCE = 1e-4  # matches tools/pygviewer/tests/test_policy_parity.py's own bar


def test_numpy_vs_pt(weights, parity):
  from policy_numpy_ref.pipeline import run_from_obs

  obs = parity["obs"].astype(np.float64)
  pt_action = parity["action"].astype(np.float64)
  errs = np.array(
    [np.abs(run_from_obs(obs[i], weights)[1] - pt_action[i]).max() for i in range(obs.shape[0])]
  )
  assert errs.max() < TOLERANCE, f"numpy vs pt max abs error {errs.max()} >= {TOLERANCE}"


def test_numpy_vs_onnx(weights, parity, onnx_session):
  from policy_numpy_ref.pipeline import run_from_obs

  obs_f32 = parity["obs"].astype(np.float32)
  input_name = onnx_session.get_inputs()[0].name
  errs = []
  for i in range(obs_f32.shape[0]):
    _, np_action = run_from_obs(obs_f32[i].astype(np.float64), weights)
    onnx_action = onnx_session.run(None, {input_name: obs_f32[i : i + 1]})[0][0].astype(
      np.float64
    )
    errs.append(np.abs(np_action - onnx_action).max())
  errs = np.array(errs)
  assert errs.max() < TOLERANCE, f"numpy vs onnx max abs error {errs.max()} >= {TOLERANCE}"


def test_pt_vs_onnx_reference(parity, onnx_session):
  """Sanity check on the two EXISTING oracles agreeing with each other (not our
  code) -- if this ever fails, the bake itself is broken, not our module."""
  obs_f32 = parity["obs"].astype(np.float32)
  pt_action = parity["action"].astype(np.float64)
  input_name = onnx_session.get_inputs()[0].name
  errs = []
  for i in range(obs_f32.shape[0]):
    onnx_action = onnx_session.run(None, {input_name: obs_f32[i : i + 1]})[0][0].astype(
      np.float64
    )
    errs.append(np.abs(pt_action[i] - onnx_action).max())
  assert np.array(errs).max() < TOLERANCE
