#!/usr/bin/env python3
"""Run a checkpoint with stock mjlab's player (`mjlab.scripts.play`) on the Pygmalion task.

  python scripts/play.py Pygmalion-Velocity-Rough-Box --checkpoint-file checkpoints/rough_main_model_11999.pt
"""

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygmalion_mjlab  # noqa: E402,F401  (registers the task)
from pygmalion_mjlab.tn_actuator import TnPdActuator, tn_limit  # noqa: E402
import mjlab.scripts.play as mjlab_play  # noqa: E402
from mjlab.tasks.velocity.mdp import (  # noqa: E402
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class StagedForwardVelocityCommand(UniformVelocityCommand):
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    initial_stage = self.command_counter[env_ids] == 0
    self.vel_command_b[env_ids] = 0.0
    self.vel_command_w[env_ids] = 0.0
    self.vel_command_b[env_ids, 0] = torch.where(initial_stage, 0.5, 1.0)

  def _debug_vis_impl(self, visualizer) -> None:
    super()._debug_vis_impl(visualizer)
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    joint_anchors = self._env.sim.data.xanchor
    for actuator in self.robot.actuators:
      if not isinstance(actuator, TnPdActuator):
        continue
      if actuator.applied_effort is None or actuator._vel is None:
        continue

      global_joint_ids = self.robot.indexing.joint_ids[actuator.target_ids]
      peak = actuator.t_tab[0]
      positive_limit = torch.where(
        actuator._vel >= 0,
        tn_limit(actuator.w_tab, actuator.t_tab, actuator._vel),
        peak.expand_as(actuator._vel),
      )
      negative_limit = torch.where(
        actuator._vel < 0,
        tn_limit(actuator.w_tab, actuator.t_tab, -actuator._vel),
        peak.expand_as(actuator._vel),
      )
      directional_limit = torch.where(
        actuator.applied_effort >= 0, positive_limit, negative_limit
      ) * actuator.strength_scale
      utilization = actuator.applied_effort.abs() / directional_limit.clamp_min(1e-6)

      for env_idx in env_indices:
        active_targets = torch.nonzero(utilization[env_idx] >= 0.9).flatten()
        for target_idx in active_targets.tolist():
          joint_id = int(global_joint_ids[target_idx].item())
          center = joint_anchors[env_idx, joint_id].detach().cpu().numpy()
          visualizer.add_sphere(
            np.asarray(center),
            radius=0.045,
            color=(1.0, 0.0, 0.0, 0.9),
            label=(
              f"{actuator.target_names[target_idx]} "
              f"{utilization[env_idx, target_idx].item():.0%}"
            ),
          )


class StagedForwardVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env):
    return StagedForwardVelocityCommand(self, env)


def _pop_terrain_level() -> int | None:
  if "--terrain-level" not in sys.argv:
    return None
  index = sys.argv.index("--terrain-level")
  if index + 1 >= len(sys.argv):
    raise ValueError("--terrain-level requires an integer in [1, 10]")
  level = int(sys.argv[index + 1])
  del sys.argv[index : index + 2]
  if not 1 <= level <= 10:
    raise ValueError(f"--terrain-level must be in [1, 10], got {level}")
  return level


def _pop_flag(flag: str) -> bool:
  if flag not in sys.argv:
    return False
  sys.argv.remove(flag)
  return True


def _pop_float(flag: str, default: float) -> float:
  if flag not in sys.argv:
    return default
  index = sys.argv.index(flag)
  if index + 1 >= len(sys.argv):
    raise ValueError(f"{flag} requires a number")
  value = float(sys.argv[index + 1])
  del sys.argv[index : index + 2]
  return value


def _install_play_overrides(
  level: int,
  terrain_scale: float,
  staged_forward: bool,
  training_dr: bool,
) -> None:
  original = mjlab_play.load_env_cfg

  def load_env_cfg(task_id, *args, **kwargs):
    kwargs["play"] = not training_dr
    cfg = original(task_id, *args, **kwargs)
    generator = cfg.scene.terrain.terrain_generator
    if generator is None:
      raise ValueError(f"Task {task_id} has no generated terrain")
    name, terrain = tuple(generator.sub_terrains.items())[level - 1]
    generator.sub_terrains = {name: terrain}
    generator.num_rows = 1
    generator.num_cols = 1
    generator.curriculum = False
    generator.size = tuple(value * terrain_scale for value in generator.size)
    terrain.size = generator.size
    cfg.curriculum = {}
    cfg.episode_length_s = int(1e9)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    twist = cfg.commands.get("twist")
    if staged_forward:
      assert isinstance(twist, UniformVelocityCommandCfg)
      staged_cfg = StagedForwardVelocityCommandCfg(
        **{field.name: getattr(twist, field.name) for field in fields(twist)}
      )
      staged_cfg.resampling_time_range = (5.0, 5.0)
      staged_cfg.heading_command = False
      staged_cfg.ranges.heading = None
      staged_cfg.rel_standing_envs = 0.0
      staged_cfg.rel_heading_envs = 0.0
      staged_cfg.rel_world_envs = 0.0
      staged_cfg.rel_forward_envs = 0.0
      staged_cfg.init_velocity_prob = 0.0
      cfg.commands["twist"] = staged_cfg
    elif twist is not None and twist.ranges.lin_vel_y == (0.0, 0.0):
      twist.ranges.lin_vel_y = (-0.1, 0.1)
      twist.rel_forward_envs = 1.0
    print(
      f"[pygmalion] fixed terrain level {level}: {name}, size={generator.size}",
      flush=True,
    )
    if staged_forward:
      print(
        "[pygmalion] command profile: vx=0.5 m/s for 5 s, then vx=1.0 m/s; "
        "vy=wz=0",
        flush=True,
      )
    if training_dr:
      print("[pygmalion] play uses the full training DR/event config", flush=True)
    return cfg

  mjlab_play.load_env_cfg = load_env_cfg

if __name__ == "__main__":
  terrain_level = _pop_terrain_level()
  terrain_scale = _pop_float("--terrain-scale", 1.0)
  staged_forward = _pop_flag("--staged-forward")
  training_dr = _pop_flag("--training-dr")
  if terrain_level is not None:
    _install_play_overrides(
      terrain_level,
      terrain_scale,
      staged_forward,
      training_dr,
    )
  mjlab_play.main()
