from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  # Use the generated grid shape (curriculum mode overrides cfg.num_cols with
  # len(sub_terrains)), and include the flat border around the patch grid.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge


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
