"""Velocity command with the steady-state tracking-error metrics the gated command
curriculum reads (not in stock mjlab). Copied from the training fork
(src/mjlab/tasks/velocity/mdp/velocity_command.py); sampling is unchanged, so commands
are drawn exactly as stock UniformVelocityCommand draws them.

Used by the flat stages (P1, P2) whose command curriculum is `commands_vel_gated`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class MetricVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # error_vel_xy (stock) sums the error and divides by the LONGEST command window, so
    # it scales with episode length. The metrics below are per-step means; the "steady"
    # ones skip the first 2 s after a command change (the acceleration transient).
    self.metrics["error_vel_xy_mean"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_xy_steady"] = torch.zeros(self.num_envs, device=self.device)
    # Same, counting only the environments given a non-zero command.
    self.metrics["error_vel_xy_steady_moving"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self._err_sum = torch.zeros(self.num_envs, device=self.device)
    self._err_n = torch.zeros(self.num_envs, device=self.device)
    self._err_sum_st = torch.zeros(self.num_envs, device=self.device)
    self._err_n_st = torch.zeros(self.num_envs, device=self.device)
    self._err_sum_st_mv = torch.zeros(self.num_envs, device=self.device)
    self._err_n_st_mv = torch.zeros(self.num_envs, device=self.device)
    self._cmd_age = torch.zeros(self.num_envs, device=self.device)  # s since resample

  def _update_metrics(self) -> None:
    super()._update_metrics()
    e = torch.norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
    )
    self._err_sum += e
    self._err_n += 1.0
    steady = self._cmd_age >= 2.0
    self._err_sum_st += e * steady
    self._err_n_st += steady.float()
    moving = steady & ~self.is_standing_env
    self._err_sum_st_mv += e * moving
    self._err_n_st_mv += moving.float()
    self._cmd_age += self._env.step_dt
    self.metrics["error_vel_xy_mean"] = self._err_sum / self._err_n.clamp(min=1.0)
    self.metrics["error_vel_xy_steady"] = self._err_sum_st / self._err_n_st.clamp(
      min=1.0
    )
    self.metrics["error_vel_xy_steady_moving"] = (
      self._err_sum_st_mv / self._err_n_st_mv.clamp(min=1.0)
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._cmd_age[env_ids] = 0.0
    if len(env_ids) and torch.any(self._env.episode_length_buf[env_ids] == 0):
      new = env_ids[self._env.episode_length_buf[env_ids] == 0]  # episode reset
      self._err_sum[new] = 0.0
      self._err_n[new] = 0.0
      self._err_sum_st[new] = 0.0
      self._err_n_st[new] = 0.0
      self._err_sum_st_mv[new] = 0.0
      self._err_n_st_mv[new] = 0.0
    super()._resample_command(env_ids)


@dataclass(kw_only=True)
class MetricVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> MetricVelocityCommand:
    return MetricVelocityCommand(self, env)
