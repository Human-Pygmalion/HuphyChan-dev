from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """Track selected body heights without diluting their Z error over all bodies/axes."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if len(body_indexes) != len(body_names):
    missing = set(body_names) - set(command.cfg.body_names)
    raise ValueError(f"Height reward bodies are not tracked by the motion command: {missing}")
  error = torch.square(
    command.body_pos_w[:, body_indexes, 2]
    - command.robot_body_pos_w[:, body_indexes, 2]
  ).mean(dim=-1)
  return torch.exp(-error / std**2)


def _body_points_w(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  local_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """Transform one fixed local point per body into world coordinates."""
  num_envs, num_bodies = body_pos_w.shape[:2]
  offsets = torch.as_tensor(
    local_offsets, device=body_pos_w.device, dtype=body_pos_w.dtype
  )
  if offsets.shape != (num_bodies, 3):
    raise ValueError(
      f"Expected {num_bodies} body-point offsets with shape ({num_bodies}, 3), "
      f"got {tuple(offsets.shape)}"
    )
  offsets = offsets.unsqueeze(0).expand(num_envs, -1, -1)
  rotated = quat_apply(
    body_quat_w.reshape(-1, 4), offsets.reshape(-1, 3)
  ).reshape(num_envs, num_bodies, 3)
  return body_pos_w + rotated


def _body_point_sets_w(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  local_point_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """Transform the same set of fixed local points on every selected body."""
  num_bodies = body_pos_w.shape[1]
  points = [
    _body_points_w(
      body_pos_w,
      body_quat_w,
      tuple(offset for _ in range(num_bodies)),
    )
    for offset in local_point_offsets
  ]
  return torch.stack(points, dim=2)


def motion_global_body_point_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
  local_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """Track world Z of fixed points attached to selected bodies."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if len(body_indexes) != len(body_names):
    missing = set(body_names) - set(command.cfg.body_names)
    raise ValueError(f"Point-height reward bodies are not tracked: {missing}")
  reference_points = _body_points_w(
    command.body_pos_w[:, body_indexes],
    command.body_quat_w[:, body_indexes],
    local_offsets,
  )
  robot_points = _body_points_w(
    command.robot_body_pos_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
    local_offsets,
  )
  error = torch.square(reference_points[..., 2] - robot_points[..., 2]).mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_global_body_point_set_min_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
  local_point_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """Track the lowest attached point on each selected body in world Z."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if len(body_indexes) != len(body_names):
    missing = set(body_names) - set(command.cfg.body_names)
    raise ValueError(f"Point-set height reward bodies are not tracked: {missing}")
  reference_points = _body_point_sets_w(
    command.body_pos_w[:, body_indexes],
    command.body_quat_w[:, body_indexes],
    local_point_offsets,
  )
  robot_points = _body_point_sets_w(
    command.robot_body_pos_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
    local_point_offsets,
  )
  reference_min_height = reference_points[..., 2].amin(dim=2)
  robot_min_height = robot_points[..., 2].amin(dim=2)
  error = torch.square(reference_min_height - robot_min_height).mean(dim=-1)
  return torch.exp(-error / std**2)


def _motion_time_window_gate(
  command: MotionCommand, start_time_s: float, end_time_s: float
) -> torch.Tensor:
  """Return a hard per-environment gate for a reference-motion time window."""
  motion_time_s = command.time_steps.to(dtype=torch.float32) / command.motion.fps
  return ((motion_time_s >= start_time_s) & (motion_time_s <= end_time_s)).to(
    dtype=torch.float32
  )


def motion_window_global_body_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
  start_time_s: float,
  end_time_s: float,
) -> torch.Tensor:
  """Track selected body heights densely only inside a motion-time window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  reward = motion_global_body_height_error_exp(
    env=env,
    command_name=command_name,
    std=std,
    body_names=body_names,
  )
  return reward * _motion_time_window_gate(command, start_time_s, end_time_s)


def motion_window_global_body_point_set_min_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
  local_point_offsets: tuple[tuple[float, float, float], ...],
  start_time_s: float,
  end_time_s: float,
) -> torch.Tensor:
  """Track each sole's lowest point densely only inside a motion-time window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  reward = motion_global_body_point_set_min_height_error_exp(
    env=env,
    command_name=command_name,
    std=std,
    body_names=body_names,
    local_point_offsets=local_point_offsets,
  )
  return reward * _motion_time_window_gate(command, start_time_s, end_time_s)


def motion_window_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
  start_time_s: float,
  end_time_s: float,
) -> torch.Tensor:
  """Track selected reference joint angles only inside a motion-time window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_ids = asset_cfg.joint_ids
  error = torch.square(
    command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids]
  ).mean(dim=-1)
  reward = torch.exp(-error / std**2)
  return reward * _motion_time_window_gate(command, start_time_s, end_time_s)


def motion_bilateral_airborne_match_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, str],
  height_threshold: float,
  temperature: float,
  std: float,
) -> torch.Tensor:
  """Match whether both tracked feet are airborne using a differentiable height gate."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if len(body_indexes) != 2:
    missing = set(body_names) - set(command.cfg.body_names)
    raise ValueError(f"Airborne reward requires two tracked feet; missing={missing}")
  reference_min_height = command.body_pos_w[:, body_indexes, 2].amin(dim=-1)
  robot_min_height = command.robot_body_pos_w[:, body_indexes, 2].amin(dim=-1)
  reference_airborne = torch.sigmoid(
    (reference_min_height - height_threshold) / temperature
  )
  robot_airborne = torch.sigmoid((robot_min_height - height_threshold) / temperature)
  error = torch.square(reference_airborne - robot_airborne)
  return torch.exp(-error / std**2)


def motion_bilateral_body_point_airborne_match_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, str],
  local_offsets: tuple[tuple[float, float, float], tuple[float, float, float]],
  height_threshold: float,
  temperature: float,
  std: float,
) -> torch.Tensor:
  """Match bilateral airborne state using fixed points such as toe tips."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if len(body_indexes) != 2:
    missing = set(body_names) - set(command.cfg.body_names)
    raise ValueError(f"Point-airborne reward requires two tracked bodies; missing={missing}")
  reference_points = _body_points_w(
    command.body_pos_w[:, body_indexes],
    command.body_quat_w[:, body_indexes],
    local_offsets,
  )
  robot_points = _body_points_w(
    command.robot_body_pos_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
    local_offsets,
  )
  reference_min_height = reference_points[..., 2].amin(dim=-1)
  robot_min_height = robot_points[..., 2].amin(dim=-1)
  reference_airborne = torch.sigmoid(
    (reference_min_height - height_threshold) / temperature
  )
  robot_airborne = torch.sigmoid((robot_min_height - height_threshold) / temperature)
  error = torch.square(reference_airborne - robot_airborne)
  return torch.exp(-error / std**2)


def motion_bilateral_body_point_set_airborne_match_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, str],
  local_point_offsets: tuple[tuple[float, float, float], ...],
  height_threshold: float,
  temperature: float,
  std: float,
) -> torch.Tensor:
  """Match airborne state using the lowest point across both complete soles."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if len(body_indexes) != 2:
    missing = set(body_names) - set(command.cfg.body_names)
    raise ValueError(f"Sole-airborne reward requires two tracked feet; missing={missing}")
  reference_points = _body_point_sets_w(
    command.body_pos_w[:, body_indexes],
    command.body_quat_w[:, body_indexes],
    local_point_offsets,
  )
  robot_points = _body_point_sets_w(
    command.robot_body_pos_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
    local_point_offsets,
  )
  reference_min_height = reference_points[..., 2].amin(dim=(1, 2))
  robot_min_height = robot_points[..., 2].amin(dim=(1, 2))
  reference_airborne = torch.sigmoid(
    (reference_min_height - height_threshold) / temperature
  )
  robot_airborne = torch.sigmoid((robot_min_height - height_threshold) / temperature)
  error = torch.square(reference_airborne - robot_airborne)
  return torch.exp(-error / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
