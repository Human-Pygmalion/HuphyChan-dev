"""Pure-numpy EmpiricalNormalization.forward, bit-for-formula identical to
rsl_rl 5.4.0's implementation (mujoco-sim/mjlab/.venv/lib/python3.13/site-packages/
rsl_rl/modules/normalization.py:46-48):

    def forward(self, x):
        return (x - self._mean) / (self._std + self.eps)

Only the frozen forward direction is needed here: the runner calls alg.eval_mode()
before get_inference_policy() returns the model, so update() (the running-mean/var
accumulator, Welford-style) never runs at inference time and is not reproduced.
"""
from __future__ import annotations

import numpy as np


def empirical_normalize(
  x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 0.01
) -> np.ndarray:
  """(x - mean) / (std + eps), elementwise. x may be (dim,) or (batch, dim)."""
  return (x - mean) / (std + eps)
