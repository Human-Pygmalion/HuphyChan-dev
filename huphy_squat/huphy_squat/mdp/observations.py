"""Observation terms.

Actor (deployable): IMU gyro, IMU gravity direction, 12 motor encoder angles (with the
encoder bias), last action, h*. The three sensor terms go through one shared lag per env
and episode (`env.huphy_obs_lag`, 0 or 1 policy step, drawn by the `sample_obs_lag` reset
event), so on a delayed step IMU and encoders are equally old, like one sensor packet.

Critic (privileged): true state + the DR values that actually change the dynamics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import base_height
from .events import get_motor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


class _SharedLagTerm:
  """Ring buffer of the raw term value; returns the frame `env.huphy_obs_lag` steps old.

  Right after a reset the buffer is refilled with the first new frame (no stale data from
  the previous episode). Called once per env step by the observation manager.
  """

  max_lag = 1

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._hist: torch.Tensor | None = None
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def raw(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    raise NotImplementedError

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    x = self.raw(env, **params)
    if self._hist is None:
      self._hist = x[:, None].repeat(1, self.max_lag + 1, 1)
    self._hist = torch.cat((x[:, None], self._hist[:, :-1]), dim=1)
    if self._fresh.any():
      self._hist[self._fresh] = x[self._fresh, None].expand(-1, self.max_lag + 1, -1)
      self._fresh[:] = False
    lag = getattr(env, "huphy_obs_lag", None)
    if lag is None:
      return x
    return self._hist[torch.arange(x.shape[0], device=x.device), lag.clamp(0, self.max_lag)]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True


class imu_gyro(_SharedLagTerm):
  """IMU angular velocity in the IMU frame [rad/s] (gyro sensor at site imu_in_base)."""

  def raw(self, env, sensor_name: str = "robot/imu_ang_vel") -> torch.Tensor:
    return env.scene[sensor_name].data


class imu_gravity(_SharedLagTerm):
  """Unit gravity direction in the IMU frame (negated framezaxis 'up' sensor)."""

  def raw(self, env, sensor_name: str = "robot/imu_upvector") -> torch.Tensor:
    return -env.scene[sensor_name].data


class motor_encoder(_SharedLagTerm):
  """Motor angles as the encoders read them (true + encoder bias) minus the default [rad]."""

  def raw(self, env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return asset.data.joint_pos_biased[:, ids] - asset.data.default_joint_pos[:, ids]


# --- critic: true state -----------------------------------------------------------------

def height(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  return base_height(env, entity_name)[:, None]


def height_error(env: ManagerBasedRlEnv, command_name: str = "height", entity_name: str = "robot") -> torch.Tensor:
  return base_height(env, entity_name)[:, None] - env.command_manager.get_command(command_name)


def root_lin_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def root_ang_vel_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_ang_vel_b


def root_gravity_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.projected_gravity_b


def joint_pos_true(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  a = env.scene[asset_cfg.name]
  return a.data.joint_pos[:, asset_cfg.joint_ids] - a.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel_true(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return (env.scene[sensor_name].data.found > 0).float()


def foot_force(env: ManagerBasedRlEnv, sensor_name: str, scale: float = 1.0 / 250.0) -> torch.Tensor:
  """Net foot-ground force per foot in the world frame, scaled (~1 at one robot weight)."""
  return env.scene[sensor_name].data.force.flatten(1) * scale


def applied_torque(env: ManagerBasedRlEnv, scale: float = 1.0 / 60.0) -> torch.Tensor:
  return get_motor(env).applied_effort * scale


def loop_distances(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  """|rod end - foot ball| of the 4 ankle loops (R A, R B, L A, L B) [m]. Diagnostic, shape [B, 4]."""
  r = env.scene[entity_name]
  names = list(r.site_names)
  a = [names.index(f"{s}_rod_{x}_end") for s in ("R", "L") for x in "AB"]
  b = [names.index(f"{s}_ball_{x}") for s in ("R", "L") for x in "AB"]
  p = r.data.site_pos_w
  return (p[:, a] - p[:, b]).norm(dim=-1)


# --- critic: live DR values --------------------------------------------------------------

def _live(env: ManagerBasedRlEnv, field: str, idx: torch.Tensor) -> torch.Tensor:
  t = getattr(env.sim.model, field)
  if t.shape[0] != env.num_envs:
    t = t.expand((env.num_envs,) + tuple(t.shape[1:]))
  return t[:, idx]


def _default(env: ManagerBasedRlEnv, field: str, idx: torch.Tensor) -> torch.Tensor:
  d = env.sim.get_default_field(field)
  if d.shape[0] == env.num_envs and field in env.sim.per_world_default_fields:
    return d[:, idx]
  return d[idx].unsqueeze(0).expand((env.num_envs,) + tuple(d[idx].shape))


def dof_param(env: ManagerBasedRlEnv, field: str, asset_cfg: SceneEntityCfg, relative: bool = True,
              scale: float = 1.0) -> torch.Tensor:
  """Live dof field (dof_frictionloss / dof_damping / dof_armature) of the selected joints:
  ratio to nominal - 1 (`relative`) or the absolute value x `scale`."""
  a = env.scene[asset_cfg.name]
  idx = _get_entity_indices(a.indexing, asset_cfg, "dof", False)
  live = _live(env, field, idx)
  if relative:
    return live / _default(env, field, idx).clamp(min=1e-9) - 1.0
  return live * scale


def pd_gain_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
  m = get_motor(env)
  return torch.cat((m.stiffness / m.default_stiffness - 1.0, m.damping / m.default_damping - 1.0), dim=1)


def encoder_bias(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT, scale: float = 1.0 / 0.015) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.encoder_bias[:, asset_cfg.joint_ids] * scale


def lags(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[action lag (physics steps) / 2, obs lag (policy steps)]."""
  m = get_motor(env)
  act = m._delay_buffer.current_lags.float() / 2.0 if m._delay_buffer is not None else torch.zeros(env.num_envs, device=env.device)  # noqa: SLF001
  obs = getattr(env, "huphy_obs_lag", torch.zeros(env.num_envs, dtype=torch.long, device=env.device)).float()
  return torch.stack((act, obs), dim=1)


def foot_friction(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  a = env.scene[asset_cfg.name]
  idx = _get_entity_indices(a.indexing, asset_cfg, "geom", False)
  return _live(env, "geom_friction", idx)[:, :, 0].mean(dim=1, keepdim=True)


def mass_offset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """[total mass / nominal - 1, payload kg / 5]."""
  a = env.scene[asset_cfg.name]
  idx = _get_entity_indices(a.indexing, asset_cfg, "body", False)
  live = _live(env, "body_mass", idx).sum(1)
  nom = _default(env, "body_mass", idx).sum(1)
  pl = getattr(env, "huphy_payload", torch.zeros(env.num_envs, device=env.device))
  return torch.stack((live / nom - 1.0, pl / 5.0), dim=1)


def com_offset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, scale: float = 1.0 / 0.02) -> torch.Tensor:
  a = env.scene[asset_cfg.name]
  idx = _get_entity_indices(a.indexing, asset_cfg, "body", False)
  return ((_live(env, "body_ipos", idx) - _default(env, "body_ipos", idx)) * scale).flatten(1)


def imu_mount(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """IMU site orientation error as the vector part of q_default^-1 * q_live, x 1/sin(2 deg / 2)."""
  a = env.scene[asset_cfg.name]
  idx = _get_entity_indices(a.indexing, asset_cfg, "site", False)
  q = _live(env, "site_quat", idx)[:, 0]
  q0 = _default(env, "site_quat", idx)[:, 0]
  w0, v0 = q0[:, :1], -q0[:, 1:]
  w1, v1 = q[:, :1], q[:, 1:]
  vec = w0 * v1 + w1 * v0 + torch.cross(v0, v1, dim=1)
  return vec / 0.01745


def push_state(env: ManagerBasedRlEnv, scale: float = 1.0 / 35.0) -> torch.Tensor:
  """[pelvis push force x, y (scaled), seconds left of the push]."""
  p = getattr(env, "huphy_push", None)
  if p is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return torch.cat((p.force[:, :2] * scale, p.remaining.clamp(min=0)[:, None]), dim=1)
