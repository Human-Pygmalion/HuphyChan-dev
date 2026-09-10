"""Glue: obs -> normalize -> mlp -> raw action -> scale/offset/clip -> target.

Exposes both a one-call convenience function (run_action, for tests that already
have a 45-D observation vector, e.g. from obs_parity.npz) and the individual
stages (so a caller -- e.g. hook_live_trace.py -- can grab every intermediate
value, matching the pipeline order used everywhere else in this package).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action import action_to_target
from .mlp import mlp_forward
from .normalize import empirical_normalize
from .obs import ObsHistory, build_observation
from .weights import Weights


@dataclass
class StepTrace:
  """Every intermediate value for one control step, for tests and worked examples."""

  raw_obs: np.ndarray  # (45,) concatenated, pre-normalization
  normalized_obs: np.ndarray  # (45,)
  raw_action: np.ndarray  # (12,) raw MLP output
  target: np.ndarray  # (12,) final joint position target


def run_from_obs(obs: np.ndarray, weights: Weights) -> tuple[np.ndarray, np.ndarray]:
  """obs (45,) -> (normalized_obs (45,), raw_action (12,)). Does not touch action
  post-processing -- callers needing the final target should use run_step() or
  call action_to_target() themselves with the contract's scale/offset/clip."""
  normalized = empirical_normalize(obs, weights.obs_mean, weights.obs_std, weights.obs_eps)
  raw_action = mlp_forward(normalized, weights)
  return normalized, raw_action


def run_step(
  gyro_rad_s: np.ndarray,
  up_vector: np.ndarray,
  q_now: np.ndarray,
  history: ObsHistory,
  last_raw_action: np.ndarray,
  cmd: np.ndarray,
  weights: Weights,
  default_q_action: np.ndarray,
  action_scale: np.ndarray | float,
  clip_lo: np.ndarray | None,
  clip_hi: np.ndarray | None,
  encoder_bias: np.ndarray | None = None,
) -> StepTrace:
  """Full pipeline for one control step, from raw sensor ingredients to the final
  joint position target, keeping every intermediate value."""
  raw_obs = build_observation(gyro_rad_s, up_vector, q_now, history, last_raw_action, cmd)
  normalized_obs, raw_action = run_from_obs(raw_obs, weights)
  target = action_to_target(
    raw_action, default_q_action, action_scale, clip_lo, clip_hi, encoder_bias
  )
  return StepTrace(
    raw_obs=raw_obs,
    normalized_obs=normalized_obs,
    raw_action=raw_action,
    target=target,
  )
