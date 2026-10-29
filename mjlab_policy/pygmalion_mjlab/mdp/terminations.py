"""Termination terms added for Pygmalion (not in stock mjlab). Copied verbatim from the training fork."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensor
from mjlab.utils.lab_api.math import wrap_to_pi

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def course_heading_error(
  env: ManagerBasedRlEnv,
  max_error: float = math.pi / 4.0,
  target_heading: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the base yaw leaves the fixed world-frame course heading."""
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.abs(wrap_to_pi(asset.data.heading_w - target_heading))
  # Keep an angle represented as float32 at exactly the configured boundary
  # from being classified as an exceedance due only to rounding.
  return error > (max_error + 1.0e-6)


def course_stagnation(
  env: ManagerBasedRlEnv,
  command_name: str,
  window_s: float = 1.0,
  failure_hold_s: float = 0.5,
  min_cmd_speed: float = 0.2,
  min_progress_m: float = 0.10,
  target_heading: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate sustained failure to advance along a fixed world-frame course.

  A sliding window integrates root velocity projected onto the course direction.
  Once the window is full, a moving command with too little progress must remain
  bad for ``failure_hold_s`` before termination.  Standing commands reset the
  failure counter.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  window_steps = max(1, round(window_s / env.step_dt))
  hold_steps = max(1, round(failure_hold_s / env.step_dt))
  buf = getattr(env, "_pyg_course_stagnation_buf", None)
  if buf is None or buf.shape[1] != window_steps:
    buf = torch.zeros(env.num_envs, window_steps, 2, device=env.device)
    env._pyg_course_stagnation_buf = buf  # type: ignore[attr-defined]
  bad_count = getattr(env, "_pyg_course_stagnation_bad_count", None)
  if bad_count is None:
    bad_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._pyg_course_stagnation_bad_count = bad_count  # type: ignore[attr-defined]

  fresh = env.episode_length_buf <= 1
  buf[fresh] = 0.0
  bad_count[fresh] = 0

  direction = torch.tensor(
    [math.cos(target_heading), math.sin(target_heading)],
    device=env.device,
    dtype=asset.data.root_link_lin_vel_w.dtype,
  )
  projected_speed = torch.sum(
    asset.data.root_link_lin_vel_w[:, :2] * direction, dim=1
  )
  command_speed = command[:, 0].abs()
  slot = env.episode_length_buf % window_steps
  rows = torch.arange(env.num_envs, device=env.device)
  buf[rows, slot, 0] = projected_speed * env.step_dt
  buf[rows, slot, 1] = command_speed * env.step_dt

  progress = buf[:, :, 0].sum(dim=1)
  mean_command_speed = buf[:, :, 1].sum(dim=1) / window_s
  window_full = env.episode_length_buf >= window_steps
  bad_window = (
    window_full
    & (mean_command_speed > min_cmd_speed)
    & (progress < min_progress_m)
  )
  bad_count[:] = torch.where(bad_window, bad_count + 1, torch.zeros_like(bad_count))
  return bad_count >= hold_steps


def stagnation(
  env: ManagerBasedRlEnv,
  command_name: str,
  window_s: float = 3.0,
  min_cmd_speed: float = 0.2,
  min_progress_m: float = 0.10,
  min_progress_frac: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the robot makes no progress along its velocity command.

  Over a sliding window of ``window_s`` seconds, sum the base velocity projected
  onto the commanded xy direction (body frame, as track_lin_vel_progress) and the
  commanded distance. Terminate when the mean commanded speed over the window is at
  least ``min_cmd_speed`` and the progress is below both ``min_progress_m`` and
  ``min_progress_frac`` of the commanded distance. Standing commands and the first
  window of every episode never trigger. The window state lives on the env and is
  cleared when an episode restarts (episode_length_buf <= 1).
  Research: docs/reward_research/2026-10-09_stagnation_termination_literature.md
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  n = max(1, round(window_s / env.step_dt))
  buf = getattr(env, "_pyg_stagnation_buf", None)
  if buf is None or buf.shape[1] != n:
    buf = torch.zeros(env.num_envs, n, 2, device=env.device)
    env._pyg_stagnation_buf = buf  # type: ignore[attr-defined]
  fresh = env.episode_length_buf <= 1
  buf[fresh] = 0.0
  cmd_xy = command[:, :2]
  cmd_mag = torch.norm(cmd_xy, dim=1)
  vel_xy = asset.data.root_link_lin_vel_b[:, :2]
  proj = torch.sum(vel_xy * cmd_xy, dim=1) / cmd_mag.clamp(min=1e-6)
  proj = torch.where(cmd_mag > 1e-6, proj, torch.zeros_like(proj))
  slot = env.episode_length_buf % n
  rows = torch.arange(env.num_envs, device=env.device)
  buf[rows, slot, 0] = proj * env.step_dt
  buf[rows, slot, 1] = cmd_mag * env.step_dt
  progress = buf[:, :, 0].sum(dim=1)
  commanded = buf[:, :, 1].sum(dim=1)
  window_full = env.episode_length_buf >= n
  moving_cmd = commanded >= min_cmd_speed * n * env.step_dt
  stuck = (progress < min_progress_m) & (progress < min_progress_frac * commanded)
  return window_full & moving_cmd & stuck


def base_height_above_terrain_below(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  sensor_name: str = "terrain_scan",
  center_radius: float = 0.25,
) -> torch.Tensor:
  """Terminate when the base is closer than ``minimum_height`` to the ground under
  it. The ground is the median hit height of the ``sensor_name`` rays within
  ``center_radius`` (xy) of the sensor frame; on a plane this equals the absolute
  root height test, on stairs and slopes it does not misfire or miss a kneel.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  data = sensor.data
  frame = data.frame_pos_w[:, 0, :]  # [B, 3]
  hit = data.hit_pos_w  # [B, N, 3]
  d_xy = torch.norm(hit[..., :2] - frame[:, None, :2], dim=-1)
  near = (d_xy <= center_radius) & (data.distances >= 0)
  h = frame[:, None, 2] - hit[..., 2]
  h = torch.where(near, h, torch.full_like(h, float("nan")))
  ground_h = torch.nanmedian(h, dim=1).values
  return torch.nan_to_num(ground_h, nan=1e3) < minimum_height


def physics_blowup(
  env: ManagerBasedRlEnv,
  joint_speed_limit: float = 100.0,
  base_speed_limit: float = 10.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate envs whose physics left the physical range before it turns into NaN.

  On rough terrain the passive ankle-linkage joints occasionally spin up to hundreds of
  rad/s (about 2 per million env-steps); the velocity-squared penalties then reach
  1e24-1e31 and wreck PPO's value targets. Any joint faster than ``joint_speed_limit``
  rad/s (the motors top out near 20-33), a base faster than ``base_speed_limit`` m/s,
  or a non-finite value ends the episode as a failure.
  """
  asset: Entity = env.scene[asset_cfg.name]
  qv = asset.data.joint_vel
  bv = asset.data.root_link_lin_vel_w
  fast = qv.abs().amax(dim=1) > joint_speed_limit
  fast |= bv.norm(dim=1) > base_speed_limit
  fast |= ~torch.isfinite(qv).all(dim=1) | ~torch.isfinite(bv).all(dim=1)
  return fast

