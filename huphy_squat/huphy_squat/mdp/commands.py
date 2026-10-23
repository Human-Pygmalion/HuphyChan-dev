"""Target pelvis height command h* (pelvis z - ground z).

Every `resampling_time_range` seconds (default 3-6 s, per env) h* jumps to U(h_min, h_max),
except that with probability `keep_prob` the env keeps its previous target for another
interval. The first target of an episode is always freshly drawn. How fast the robot moves to
a new target is left to the policy (the reward penalizes vertical speed above 0.1 m/s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def base_height(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  """Pelvis (base_link origin) height above the env's ground (flat terrain). Shape [B]."""
  asset = env.scene[entity_name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


@dataclass(kw_only=True)
class HeightCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  height_range: tuple[float, float] = (0.65, 0.895)
  keep_prob: float = 0.2
  """Probability that a resample keeps the previous target (never at the episode start)."""
  resampling_time_range: tuple[float, float] = (3.0, 6.0)

  def build(self, env: ManagerBasedRlEnv) -> HeightCommand:
    return HeightCommand(self, env)


class HeightCommand(CommandTerm):
  cfg: HeightCommandCfg

  def __init__(self, cfg: HeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    lo, hi = cfg.height_range
    self.target = torch.full((self.num_envs, 1), 0.5 * (lo + hi), device=self.device)
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_height_settled"] = torch.zeros(self.num_envs, device=self.device)
    self._steps = torch.zeros(self.num_envs, device=self.device)
    self._settled_steps = torch.zeros(self.num_envs, device=self.device)
    self._since_resample = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    lo, hi = self.cfg.height_range
    new = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)
    keep = torch.rand(len(env_ids), device=self.device) < self.cfg.keep_prob
    keep &= self.command_counter[env_ids] > 0  # counter is 0 only for the reset draw
    self.target[env_ids, 0] = torch.where(keep, self.target[env_ids, 0], new)
    self._since_resample[env_ids] = 0.0

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    """Running episode means of |h - h*| (all steps, and steps >= 2 s after a resample)."""
    err = (base_height(self._env, self.cfg.entity_name) - self.target[:, 0]).abs()
    self._steps += 1
    self.metrics["error_height"] += (err - self.metrics["error_height"]) / self._steps
    settled = self._since_resample >= 2.0
    self._settled_steps += settled.float()
    upd = (err - self.metrics["error_height_settled"]) / self._settled_steps.clamp(min=1)
    self.metrics["error_height_settled"] += torch.where(settled, upd, torch.zeros_like(upd))
    self._since_resample += self._env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    self._steps[env_ids] = 0.0
    self._settled_steps[env_ids] = 0.0
    return extras

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    asset = self._env.scene[self.cfg.entity_name]
    pos = asset.data.root_link_pos_w.cpu().numpy()
    origin = self._env.scene.env_origins.cpu().numpy()
    tgt = self.target[:, 0].cpu().numpy()
    for i in visualizer.get_env_indices(self.num_envs):
      p = pos[i].copy()
      p[2] = origin[i, 2] + tgt[i]
      visualizer.add_sphere(center=p, radius=0.03, color=(0.1, 0.8, 0.2, 0.8))
