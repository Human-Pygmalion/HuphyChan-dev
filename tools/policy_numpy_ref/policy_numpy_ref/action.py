"""Pure-numpy action post-processing: raw MLP output -> joint position target.

Exact order of operations, confirmed by reading mjlab/envs/mdp/actions/actions.py
directly (process_actions then apply_actions):

    1. processed = raw_action * scale + offset        (actions.py:157, offset = default_joint_pos)
    2. processed = clip(processed, clip_lo, clip_hi)   (actions.py:157-163, only if a clip is configured --
                                                         this checkpoint has PYG_SAFE_TARGET_CLIP=1, so it does)
    3. target = processed - encoder_bias               (actions.py:221-222, a sim-only DR term;
                                                         zero/None for a real deploy -- see
                                                         docs/133_..._sim2real_gaps.md)

`target` (the output of this function) is what corresponds to a real robot's
wire q_target -- NOT MuJoCo's data.ctrl. This checkpoint's 12 actuated joints are
all TnPdActuatorCfg (confirmed from the model contract's raw-torque actuator
signature), so data.ctrl for them is a further, already-PD-computed TORQUE, one
step downstream of this function's output and out of this module's scope.

action_joint_names is identical, element for element, to obs.OBS_JOINT_NAMES
(confirmed by diffing the baked contract's obs_joint_names/action_joint_names
lists) -- re-exported here under its own name for readability at call sites, not
because the values actually differ.
"""
from __future__ import annotations

import numpy as np

from .obs import OBS_JOINT_NAMES

ACTION_JOINT_NAMES: tuple[str, ...] = OBS_JOINT_NAMES


def action_to_target(
  raw_action: np.ndarray,  # (12,) raw MLP output, ACTION_JOINT_NAMES order
  default_q: np.ndarray,  # (12,) offset, ACTION_JOINT_NAMES order (from the contract's default_q)
  scale: np.ndarray | float,  # 0.25 flat for every joint (PYG_ACTION_SCALE), or a (12,) array
  clip_lo: np.ndarray | None = None,  # (12,) or None if PYG_SAFE_TARGET_CLIP was unset
  clip_hi: np.ndarray | None = None,
  encoder_bias: np.ndarray | None = None,  # (12,) sim-only DR term; None/omit for real deploy
) -> np.ndarray:
  processed = raw_action * scale + default_q
  if clip_lo is not None:
    assert clip_hi is not None
    processed = np.clip(processed, clip_lo, clip_hi)
  if encoder_bias is not None:
    processed = processed - encoder_bias
  return processed
