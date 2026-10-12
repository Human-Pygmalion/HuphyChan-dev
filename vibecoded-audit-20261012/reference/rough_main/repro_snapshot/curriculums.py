from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


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


# ---------------------------------------------------------------------------
# V3 (docs/103 §3a): promotion by REPORT CARD instead of by the clock.
#
# commands_vel above advances on fixed step counts - 4,000 iters per stage in the
# pygmalion cfg - while the measured adaptation to a widened command box takes
# 4-20 iters. Roughly 3,980 iters per stage are spent re-confirming something the
# policy already knows, four times over.
#
# The replacement holds a stage until the policy's own numbers say it is ready:
#   (1) trailing-window mean of error_vel_xy_steady <= (previous stage's settled
#       value) x err_ratio        -- "you got back to where you were before I
#                                     widened the box"
#   (2) trailing-window fell_over rate <= fell_max
#   (3) at least min_dwell_iters in the stage
# and - the safety net docs/103 §6 asks for - forced promotion once the stage has
# lasted max_dwell_iters, whatever the metrics say, with a warning. A gate that
# never fires must not be able to stall the run.
#
# The window is measured over FINISHED EPISODES (this function runs inside
# _reset_idx, before the command manager clears its per-episode metrics, so the
# values read here belong to the episode that just ended), bucketed by iteration
# so "trailing 100 iters" means what it says.
# ---------------------------------------------------------------------------


# Which per-environment sample counter belongs to each gate metric. The counter says
# how many steps actually went into that environment's mean, so an episode that ended
# before the metric had a single valid sample can be dropped instead of counted as a
# zero error.
_COUNTER_OF = {
  "error_vel_xy_steady": "_err_n_st",
  "error_vel_xy_steady_moving": "_err_n_st_mv",
}


def _gate_state(env: ManagerBasedRlEnv, start_iter: int) -> dict:
  st = getattr(env, "_pyg_gate_state", None)
  if st is None:
    st = {
      "stage": 0,
      "stage_start_iter": start_iter,
      "baseline": None,  # settled error of the last stage that PASSED its gate
      "buckets": deque(),  # [iter, err_sum, err_n, fell_sum, ep_n] per iteration
      "forced": 0,
      "history": [],  # (stage, dwell, err, fell, reason)
    }
    env._pyg_gate_state = st  # type: ignore[attr-defined]
  return st


def commands_vel_gated(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
  min_dwell_iters: int = 800,
  max_dwell_iters: int = 3000,
  window_iters: int = 100,
  err_ratio: float = 1.10,
  fell_max: float = 0.005,
  min_episodes: int = 64,
  steps_per_iter: int = 24,
  fell_term: str = "fell_over",
  metric: str = "error_vel_xy_steady",
) -> dict[str, torch.Tensor]:
  """Performance-gated version of :func:`commands_vel`. See the note above.

  ``velocity_stages`` is the same list commands_vel takes; the per-stage
  ``"step"`` field is IGNORED (that is the whole point) but left in place so the
  two functions are interchangeable in a cfg.

  ``metric`` selects which tracking error the gate reads. The default counts every
  environment, including the ones told to stand still, whose error falls to ~0 as
  soon as the robot holds position - so a run with many standing environments sets
  itself an easier early baseline and a harder late bar. Pass
  ``"error_vel_xy_steady_moving"`` to score only the environments that were given a
  non-zero command.
  """
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  it = env.common_step_counter // max(1, steps_per_iter)
  st = _gate_state(env, it)
  n_stages = len(velocity_stages)

  # --- accumulate this reset batch into the trailing window -----------------
  if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
    # episode_length_buf is still the FINISHED episode's length here; 0 means this
    # is the startup reset, which carries no performance information.
    live = env.episode_length_buf[env_ids] > 0
    ids = env_ids[live]
    if ids.numel() > 0:
      err = command_term.metrics.get(metric)
      n_st = getattr(command_term, _COUNTER_OF.get(metric, "_err_n_st"), None)
      err_sum, err_n = 0.0, 0.0
      if err is not None:
        e = err[ids]
        if n_st is not None:
          valid = n_st[ids] > 0  # episodes too short to have a steady phase
          e = e[valid]
        err_sum, err_n = float(e.sum().item()), float(e.numel())
      fell_sum = 0.0
      try:
        fell_sum = float(env.termination_manager.get_term(fell_term)[ids].sum().item())
      except (KeyError, AttributeError):
        fell_sum = float("nan")
      b = st["buckets"]
      if not b or b[-1][0] != it:
        b.append([it, 0.0, 0.0, 0.0, 0.0])
      b[-1][1] += err_sum
      b[-1][2] += err_n
      b[-1][3] += 0.0 if fell_sum != fell_sum else fell_sum
      b[-1][4] += float(ids.numel())
      while b and b[0][0] <= it - window_iters:
        b.popleft()

  b = st["buckets"]
  err_n = sum(r[2] for r in b)
  ep_n = sum(r[4] for r in b)
  err_w = sum(r[1] for r in b) / err_n if err_n > 0 else float("nan")
  fell_w = sum(r[3] for r in b) / ep_n if ep_n > 0 else float("nan")
  dwell = it - st["stage_start_iter"]

  # --- promotion decision ---------------------------------------------------
  if st["stage"] < n_stages - 1:
    promote, reason = False, ""
    if dwell >= max_dwell_iters:
      promote, reason = True, "MAX-DWELL forced"
    elif dwell >= min_dwell_iters and err_n >= min_episodes and ep_n >= min_episodes:
      base = st["baseline"]
      ok_err = base is None or err_w <= base * err_ratio
      ok_fell = fell_w <= fell_max
      if ok_err and ok_fell:
        promote, reason = True, "gate passed"
    if promote:
      forced = reason.startswith("MAX-DWELL")
      old_stage = st["stage"]
      st["stage"] += 1
      st["stage_start_iter"] = it
      if forced:
        st["forced"] += 1
        # A forced stage never settled, so its window mean is NOT a settled value
        # and must not become the bar for the next stage. Keep the old baseline.
        print(
          f"[gated-curriculum] WARNING stage {old_stage} -> {st['stage']} at iter {it}: "
          f"{reason} after dwell {dwell} iters "
          f"(err_steady {err_w:.4f} vs baseline {st['baseline']}, fell {fell_w:.4f}); "
          f"baseline kept at {st['baseline']}",
          flush=True,
        )
      else:
        st["baseline"] = err_w
        print(
          f"[gated-curriculum] stage {old_stage} -> {st['stage']} at iter {it}: "
          f"{reason} after dwell {dwell} iters "
          f"(err_steady {err_w:.4f}, fell {fell_w:.4f}, episodes {int(ep_n)}); "
          f"new baseline {err_w:.4f}",
          flush=True,
        )
      st["history"].append((old_stage, dwell, err_w, fell_w, reason))
      st["buckets"].clear()  # the new stage is a different task; start its window clean
      err_n = ep_n = 0.0
      err_w = fell_w = float("nan")
      dwell = 0

  # --- apply every stage up to the current one, in order (same as commands_vel)
  for stage in velocity_stages[: st["stage"] + 1]:
    if stage.get("lin_vel_x") is not None:
      cfg.ranges.lin_vel_x = stage["lin_vel_x"]
    if stage.get("lin_vel_y") is not None:
      cfg.ranges.lin_vel_y = stage["lin_vel_y"]
    if stage.get("ang_vel_z") is not None:
      cfg.ranges.ang_vel_z = stage["ang_vel_z"]

  def _t(x: float) -> torch.Tensor:
    return torch.tensor(float(x))

  return {
    "lin_vel_x_min": _t(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": _t(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": _t(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": _t(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": _t(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": _t(cfg.ranges.ang_vel_z[1]),
    "stage": _t(st["stage"]),
    "stage_dwell_iters": _t(dwell),
    "gate_err_steady": _t(err_w),
    "gate_fell_over": _t(fell_w),
    "gate_baseline": _t(st["baseline"] if st["baseline"] is not None else float("nan")),
    "gate_window_episodes": _t(ep_n),
    "forced_promotions": _t(st["forced"]),
  }
