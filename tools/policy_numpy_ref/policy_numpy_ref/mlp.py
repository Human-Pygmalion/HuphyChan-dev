"""Pure-numpy forward pass for the actor MLP: Linear(45,512)->ELU->Linear(512,256)->
ELU->Linear(256,128)->ELU->Linear(128,12). No activation after the last layer.

Confirmed from the checkpoint's own printed architecture (bake log, LegOnly-AB
model_27999 bake, 2026-09-10) and from rsl_rl/models/mlp_model.py's forward():
the actor's returned action IS the raw final-Linear output -- GaussianDistribution
.deterministic_output() with std_type="scalar" is an identity on the mean
(rsl_rl/modules/distribution.py:197-199), so nothing further is applied to it.

Deliberately no branching on shape/batch-size beyond plain numpy broadcasting, and
no control flow depending on input values, so this would translate to an ONNX
graph (MatMul+Add+Elu chain) with no structural changes if that conversion is
ever wanted later.
"""
from __future__ import annotations

import numpy as np

from .weights import Weights


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
  """ELU(x) = x if x>0 else alpha*(exp(x)-1). np.expm1 for the standard numerically
  stable exp(x)-1 near x=0 (matches torch.nn.ELU's own formula, alpha=1.0 default,
  confirmed as the Pygmalion actor's activation: rl_cfg.py activation="elu")."""
  return np.where(x > 0, x, alpha * np.expm1(x))


def mlp_forward(x: np.ndarray, weights: Weights) -> np.ndarray:
  """x: (..., 45) normalized observation -> (..., 12) raw action (pre scale/offset).

  weights.linear_w[i] is stored torch-Linear-style as (out_features, in_features),
  so the matmul is x @ w.T + b (never transpose w in place -- keep it exactly as
  extracted from the checkpoint, so a shape mismatch here is a bug, not silently
  "corrected").
  """
  h = x
  n_layers = len(weights.linear_w)
  for i, (w, b) in enumerate(zip(weights.linear_w, weights.linear_b)):
    h = h @ w.T + b
    if i < n_layers - 1:
      h = elu(h)
  return h
