"""dr_levels: linear domain-randomization ramp (not in stock mjlab). Copied verbatim from the training fork."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
from typing import TYPE_CHECKING

import torch


def dr_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  start_step: int,
  end_step: int,
  push_max: dict[str, tuple[float, float]],
  friction_max: tuple[float, float],
  encoder_max: tuple[float, float],
  com_max: dict[int, tuple[float, float]],
  inertial_max: dict[str, dict[str, tuple[float, float]]] | None = None,
  pd_gain_max: dict[str, tuple[float, float]] | None = None,
  action_delay_max: float | None = None,
  action_noise_max: float | None = None,
) -> dict[str, torch.Tensor]:
  """Progressively ramp domain-randomization strength from 0 to full.

  factor = 0 for step <= start_step, linearly 0->1 to end_step, then 1. Scales the
  live event term params (push_robot / foot_friction / encoder_bias / base_com /
  pseudo-inertia) so DR grows over training. Missing events (e.g. absent in phase 1)
  are skipped.  ``inertial_max`` is kept separately from the mutable event params:
  scaling an already-scaled range every reset would otherwise collapse it toward zero.
  Used for phase-2 DR curriculum on top of a clean-walking phase-1 policy.
  """
  del env_ids
  step = env.common_step_counter
  if step <= start_step:
    factor = 0.0
  elif step >= end_step:
    factor = 1.0
  else:
    factor = (step - start_step) / float(max(1, end_step - start_step))

  em = env.event_manager

  def _cfg(name):
    try:
      return em.get_term_cfg(name)
    except (KeyError, ValueError):
      return None

  c = _cfg("push_robot")
  if c is not None:
    c.params["velocity_range"] = {
      k: (lo * factor, hi * factor) for k, (lo, hi) in push_max.items()
    }
  c = _cfg("foot_friction")
  if c is not None:
    lo, hi = friction_max
    center = 0.5 * (lo + hi)
    c.params["ranges"] = (
      center - (center - lo) * factor,
      center + (hi - center) * factor,
    )
  c = _cfg("encoder_bias")
  if c is not None:
    lo, hi = encoder_max
    c.params["bias_range"] = (lo * factor, hi * factor)
  c = _cfg("base_com")
  if c is not None:
    c.params["ranges"] = {
      k: (lo * factor, hi * factor) for k, (lo, hi) in com_max.items()
    }
  # 2026-09-17 (열린 일 #11): PD 게인과 명령 지연/잡음도 같은 창에서 0 -> 전체로 켠다.
  # 게인 배율은 **1 을 중심으로** 넓혀야 한다. 마찰과 달리 0 이 "없음" 이 아니라
  # 1 이 "없음" 이기 때문이다 — 0 을 곱하면 모터가 아예 죽는다.
  if pd_gain_max is not None:
    c = _cfg("pd_gains")
    if c is not None:
      for key, pname in (("kp", "kp_range"), ("kd", "kd_range")):
        lo, hi = pd_gain_max[key]
        c.params[pname] = (1.0 - (1.0 - lo) * factor, 1.0 + (hi - 1.0) * factor)
  # 지연과 잡음은 이벤트가 아니라 **행동 항목 자체**에 들어 있으므로 거기를 고친다.
  if action_delay_max is not None or action_noise_max is not None:
    try:
      term = env.action_manager.get_term("joint_pos")
    except (KeyError, ValueError):
      term = None
    if term is not None:
      if action_delay_max is not None and hasattr(term, "max_delay"):
        term.max_delay = float(action_delay_max) * factor
      if action_noise_max is not None and hasattr(term, "noise"):
        term.noise = float(action_noise_max) * factor

  for name, ranges in (inertial_max or {}).items():
    c = _cfg(name)
    if c is None:
      continue
    for key, (lo, hi) in ranges.items():
      c.params[key] = (lo * factor, hi * factor)

  return {"dr_factor": torch.tensor(float(factor))}

