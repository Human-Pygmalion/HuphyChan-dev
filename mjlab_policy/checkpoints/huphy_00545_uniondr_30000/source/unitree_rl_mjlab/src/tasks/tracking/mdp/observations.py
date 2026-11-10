from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class SparseHistoryObservation:
  """Return selected past samples of another observation term.

  The requested lags are emitted oldest-to-newest so the temporal ordering
  matches mjlab's normal flattened history representation.  For example,
  ``history_lags=(1, 2, 4, 8)`` returns ``[x[t-8], x[t-4], x[t-2], x[t-1]]``.
  Newly reset environments are backfilled with their first valid sample.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    lags = tuple(int(lag) for lag in cfg.params["history_lags"])
    if not lags or any(lag <= 0 for lag in lags):
      raise ValueError("history_lags must contain positive integers")
    if len(set(lags)) != len(lags):
      raise ValueError("history_lags must not contain duplicates")
    self.history_lags = tuple(sorted(lags))
    self._output_lags = tuple(reversed(self.history_lags))
    self._buffer = CircularBuffer(
      max_len=max(self.history_lags) + 1,
      batch_size=env.num_envs,
      device=env.device,
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self._buffer.reset(batch_ids=env_ids)  # type: ignore[arg-type]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    source_func,
    source_params: dict,
    history_lags: tuple[int, ...] = (1, 2, 4, 8),
    source_noise=None,
  ) -> torch.Tensor:
    configured = tuple(sorted(int(lag) for lag in history_lags))
    if configured != self.history_lags:
      raise ValueError(
        f"history_lags changed from {self.history_lags} to {configured}"
      )
    current = source_func(env, **source_params)
    if source_noise is not None:
      current = source_noise.apply(current)
    if current.ndim != 2:
      raise ValueError(
        "SparseHistoryObservation requires a flat (num_envs, dim) source, "
        f"got {tuple(current.shape)}"
      )
    self._buffer.append(current)
    return torch.cat([self._buffer[lag] for lag in self._output_lags], dim=-1)


def joint_torque(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sim robot actuator torques (PD effort). Shape (B, nu).

  Raw Nm values; RL ``obs_normalization`` (vec-z) handles scaling.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.actuator_force[:, asset_cfg.actuator_ids]


def _vel_in_anchor_b(
  command: MotionCommand, vel_w: torch.Tensor
) -> torch.Tensor:
  """Rotate world velocities into robot-anchor frame. ``vel_w``: (B, N, 3)."""
  B, N, _ = vel_w.shape
  q = command.robot_anchor_quat_w[:, None, :].expand(B, N, 4).reshape(B * N, 4)
  return quat_apply_inverse(q, vel_w.reshape(B * N, 3)).reshape(B, N, 3)


# Future reference joint command frames (policy dt steps ahead of current).
MOTION_COMMAND_FUTURE_STEPS: tuple[int, ...] = (1, 2, 3, 4, 5)


def motion_command_future(
  env: ManagerBasedRlEnv,
  command_name: str,
  future_steps: tuple[int, ...] = MOTION_COMMAND_FUTURE_STEPS,
) -> torch.Tensor:
  """Future ref joint pos+vel stacked. (B, 58 * len(future_steps)).

  For each offset in ``future_steps``, appends ``[q*, qdot*]`` at
  ``clamp(t + offset, T-1)``. Default offsets: +1..+5 frames.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  parts: list[torch.Tensor] = []
  for step in future_steps:
    frames = command.clamp_reference_steps(command.time_steps + int(step))
    parts.append(
      torch.cat(
        (
          command.motion.joint_pos[frames],
          command.motion.joint_vel[frames],
        ),
        dim=-1,
      )
    )
  return torch.cat(parts, dim=-1)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim tracked-body linear velocities in robot-anchor frame. (B, 3*N)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return _vel_in_anchor_b(command, command.robot_body_lin_vel_w).reshape(
    env.num_envs, -1
  )


def robot_body_lin_vel_ref_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Ref tracked-body linear velocities in robot-anchor frame. (B, 3*N)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return _vel_in_anchor_b(command, command.body_lin_vel_w).reshape(env.num_envs, -1)


_ANKLE_BODIES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)


def _ankle_indices(command: MotionCommand) -> list[int]:
  names = list(command.cfg.body_names)
  return [names.index(n) for n in _ANKLE_BODIES]


def ankle_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim L/R ankle linear velocities in robot-anchor frame. (B, 6)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idx = _ankle_indices(command)
  return _vel_in_anchor_b(command, command.robot_body_lin_vel_w[:, idx]).reshape(
    env.num_envs, -1
  )


def ankle_lin_vel_ref_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref L/R ankle linear velocities in robot-anchor frame. (B, 6)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idx = _ankle_indices(command)
  return _vel_in_anchor_b(command, command.body_lin_vel_w[:, idx]).reshape(
    env.num_envs, -1
  )


def base_lin_vel_ref(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref floating-base (pelvis) linear velocity in pelvis frame. (B, 3).

  Counterpart to mjlab ``base_lin_vel`` (sim root lin vel in root body frame).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  names = list(command.cfg.body_names)
  idx = names.index("pelvis")
  return quat_apply_inverse(
    command.body_quat_w[:, idx], command.body_lin_vel_w[:, idx]
  )


def robot_body_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim tracked-body angular velocities in robot-anchor frame. (B, 3*N)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return _vel_in_anchor_b(command, command.robot_body_ang_vel_w).reshape(
    env.num_envs, -1
  )


def robot_body_ang_vel_ref_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Ref tracked-body angular velocities in robot-anchor frame. (B, 3*N)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return _vel_in_anchor_b(command, command.body_ang_vel_w).reshape(env.num_envs, -1)


_KEY_HEIGHT_BODIES: tuple[str, ...] = (
  "pelvis",
  "torso_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

# Pre-wrist Hist9-Rich (4D).
_KEY_HEIGHT_BODIES_LEGACY4: tuple[str, ...] = (
  "pelvis",
  "torso_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)


def _key_height_indices(
  command: MotionCommand,
  bodies: tuple[str, ...] = _KEY_HEIGHT_BODIES,
) -> list[int]:
  names = list(command.cfg.body_names)
  return [names.index(n) for n in bodies]


def key_body_height_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """World-frame Z of pelvis / torso / L-R feet / L-R hands. (B, 6)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idx = _key_height_indices(command)
  return command.robot_body_pos_w[:, idx, 2]


def key_body_height_ref_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference world-frame Z of pelvis / torso / L-R feet / L-R hands. (B, 6)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idx = _key_height_indices(command)
  return command.body_pos_w[:, idx, 2]


def key_body_height_err(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Legacy: sim − ref height. Prefer sim+ref only (err is redundant)."""
  return key_body_height_w(env, command_name) - key_body_height_ref_w(
    env, command_name
  )


def key_body_height_w_legacy4(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Legacy Hist9-Rich heights (no wrists). (B, 4)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idx = _key_height_indices(command, _KEY_HEIGHT_BODIES_LEGACY4)
  return command.robot_body_pos_w[:, idx, 2]


def key_body_height_ref_w_legacy4(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Legacy Hist9-Rich ref heights (no wrists). (B, 4)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  idx = _key_height_indices(command, _KEY_HEIGHT_BODIES_LEGACY4)
  return command.body_pos_w[:, idx, 2]


def key_body_height_err_legacy4(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Legacy Hist9-Rich height err (no wrists). (B, 4)."""
  return key_body_height_w_legacy4(env, command_name) - key_body_height_ref_w_legacy4(
    env, command_name
  )
