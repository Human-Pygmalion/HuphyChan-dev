"""Checkpoint -> plain-numpy weight extraction.

This is the ONLY module in this package that imports torch, and only to read a
.pt file once and copy its tensors into plain numpy arrays. Everything downstream
(normalize.py, mlp.py, obs.py, action.py, pipeline.py) takes numpy arrays only.

Confirmed checkpoint layout (mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/
2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2/model_27999.pt), read directly with
torch.load and printed -- see tools/policy_numpy_ref/artifacts/state_dict_keys.txt:

    top-level keys: actor_state_dict, critic_state_dict, optimizer_state_dict, iter, infos
    actor_state_dict keys (NO "actor." prefix):
        obs_normalizer._mean   (1, 45)
        obs_normalizer._var    (1, 45)   -- unused at inference, kept in meta for reference
        obs_normalizer._std    (1, 45)
        obs_normalizer.count   ()        -- unused at inference (training-only running count)
        distribution.std_param (12,)     -- IGNORED: deployment always takes the deterministic
                                            mean, which for GaussianDistribution(std_type="scalar")
                                            is simply the raw final-Linear output (rsl_rl
                                            modules/distribution.py: deterministic_output returns
                                            mlp_output unchanged). The std is only used to sample
                                            during training.
        mlp.0.weight/bias      (512, 45) / (512,)
        mlp.2.weight/bias      (256, 512) / (256,)
        mlp.4.weight/bias      (128, 256) / (128,)
        mlp.6.weight/bias      (12, 128) / (12,)

Only critic_state_dict is present too, but the critic is privileged/sim-only and is
never loaded here -- it plays no part in a deployed policy.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OBS_DIM = 45
ACTION_DIM = 12
HIDDEN_DIMS = (512, 256, 128)
# rsl_rl EmpiricalNormalization default eps (rsl_rl/modules/normalization.py:19);
# the Pygmalion RslRlModelCfg does not override it.
OBS_NORM_EPS = 0.01


@dataclass(frozen=True)
class Weights:
  """Everything the pure-numpy forward pass needs, as float64 numpy arrays.

  float64 (not float32) is deliberate: the checkpoint itself is float32, and the
  real deployed network runs float32 math. We upcast once at extraction time so
  that our own numpy matmuls accumulate in float64, which isolates "does our
  formula/order of operations match the trainer's" from "does float32 rounding
  differ between PyTorch's BLAS and numpy's BLAS" -- see docs/136 for how the
  verification suite reports both a float64 self-consistency figure and a
  float32-cast comparison against the real deployed precision.
  """

  obs_mean: np.ndarray  # (45,) float64
  obs_std: np.ndarray  # (45,) float64
  obs_eps: float  # 0.01
  linear_w: tuple[np.ndarray, ...]  # (512,45),(256,512),(128,256),(12,128), float64
  linear_b: tuple[np.ndarray, ...]  # (512,),(256,),(128,),(12,), float64
  meta: dict = field(default_factory=dict)

  def __post_init__(self) -> None:
    assert self.obs_mean.shape == (OBS_DIM,), self.obs_mean.shape
    assert self.obs_std.shape == (OBS_DIM,), self.obs_std.shape
    assert len(self.linear_w) == len(self.linear_b) == len(HIDDEN_DIMS) + 1
    dims = (OBS_DIM, *HIDDEN_DIMS, ACTION_DIM)
    for i, (w, b) in enumerate(zip(self.linear_w, self.linear_b)):
      assert w.shape == (dims[i + 1], dims[i]), (i, w.shape, dims)
      assert b.shape == (dims[i + 1],), (i, b.shape)


def _sha256(path: str | Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def load_checkpoint_numpy(pt_path: str | Path) -> Weights:
  """torch-only function: read a .pt checkpoint's actor_state_dict and return a
  Weights of plain float64 numpy arrays. No torch types escape this function."""
  import torch  # local import: keep torch out of every other module's namespace

  pt_path = Path(pt_path)
  state = torch.load(pt_path, map_location="cpu", weights_only=False)
  actor = state["actor_state_dict"]

  def npf64(key: str) -> np.ndarray:
    return actor[key].detach().cpu().numpy().astype(np.float64)

  obs_mean = npf64("obs_normalizer._mean").reshape(-1)
  obs_std = npf64("obs_normalizer._std").reshape(-1)

  linear_w = tuple(npf64(f"mlp.{i}.weight") for i in (0, 2, 4, 6))
  linear_b = tuple(npf64(f"mlp.{i}.bias") for i in (0, 2, 4, 6))

  meta = {
    "source_checkpoint": str(pt_path),
    "source_sha256": _sha256(pt_path),
    "iter": int(state.get("iter", -1)),
    "obs_dim": OBS_DIM,
    "action_dim": ACTION_DIM,
    "note": "distribution.std_param present in checkpoint but intentionally not loaded "
    "(inference is deterministic-mean-only; see module docstring).",
  }

  return Weights(
    obs_mean=obs_mean,
    obs_std=obs_std,
    obs_eps=OBS_NORM_EPS,
    linear_w=linear_w,
    linear_b=linear_b,
    meta=meta,
  )


def save_npz(weights: Weights, out_path: str | Path) -> None:
  """numpy-only: serialize a Weights to a plain .npz (no torch needed to reload it)."""
  import json

  np.savez(
    out_path,
    obs_mean=weights.obs_mean,
    obs_std=weights.obs_std,
    obs_eps=np.array(weights.obs_eps),
    linear_w0=weights.linear_w[0],
    linear_w1=weights.linear_w[1],
    linear_w2=weights.linear_w[2],
    linear_w3=weights.linear_w[3],
    linear_b0=weights.linear_b[0],
    linear_b1=weights.linear_b[1],
    linear_b2=weights.linear_b[2],
    linear_b3=weights.linear_b[3],
    meta_json=json.dumps(weights.meta),
  )


def load_npz(path: str | Path) -> Weights:
  """numpy-only: load a Weights previously written by save_npz."""
  import json

  data = np.load(path, allow_pickle=False)
  return Weights(
    obs_mean=data["obs_mean"],
    obs_std=data["obs_std"],
    obs_eps=float(data["obs_eps"]),
    linear_w=(data["linear_w0"], data["linear_w1"], data["linear_w2"], data["linear_w3"]),
    linear_b=(data["linear_b0"], data["linear_b1"], data["linear_b2"], data["linear_b3"]),
    meta=json.loads(str(data["meta_json"])),
  )
