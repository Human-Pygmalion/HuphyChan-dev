from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .bicycle_data import BICYCLE_CONTACT_PAIRS, BICYCLE_CONTACT_PAIR_KEYS

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]
    self.fps = float(np.asarray(data.get("fps", np.array([50.0]))).reshape(-1)[0])
    self.motion_phase = torch.tensor(
      np.asarray(
        data.get(
          "motion_phase",
          np.linspace(0.0, 1.0, self.time_step_total, dtype=np.float32),
        )
      ),
      dtype=torch.float32,
      device=device,
    )
    if self.motion_phase.shape != (self.time_step_total,):
      raise ValueError(
        f"motion_phase must have shape ({self.time_step_total},), "
        f"got {tuple(self.motion_phase.shape)}"
      )
    self.clip_starts = torch.tensor(
      np.asarray(data.get("clip_starts", np.array([0])), dtype=np.int64),
      dtype=torch.long,
      device=device,
    )
    self.clip_lengths = torch.tensor(
      np.asarray(
        data.get("clip_lengths", np.array([self.time_step_total])), dtype=np.int64
      ),
      dtype=torch.long,
      device=device,
    )
    if self.clip_starts.shape != self.clip_lengths.shape or self.clip_starts.ndim != 1:
      raise ValueError("clip_starts and clip_lengths must be matching 1-D arrays")
    if torch.any(self.clip_lengths <= 0):
      raise ValueError("clip_lengths must be positive")
    if torch.any(self.clip_starts < 0) or torch.any(
      self.clip_starts + self.clip_lengths > self.time_step_total
    ):
      raise ValueError("clip ranges must lie inside the concatenated motion")
    self.clip_names = tuple(
      str(name)
      for name in data.get("clip_names", np.array(["motion"], dtype=str))
    )
    if len(self.clip_names) != len(self.clip_starts):
      raise ValueError("clip_names must match clip_starts")

    self.bike_qpos: torch.Tensor | None = None
    self.bike_qvel: torch.Tensor | None = None
    self.bike_body_names: tuple[str, ...] = ()
    self.bike_body_pos_w: torch.Tensor | None = None
    self.bike_body_quat_w: torch.Tensor | None = None
    self.bike_body_lin_vel_w: torch.Tensor | None = None
    self.bike_body_ang_vel_w: torch.Tensor | None = None
    if "bike_qpos" in data.files:
      self.bike_qpos = torch.tensor(
        data["bike_qpos"], dtype=torch.float32, device=device
      )
      self.bike_qvel = torch.tensor(
        data["bike_qvel"], dtype=torch.float32, device=device
      )
      if self.bike_qpos.shape[0] != self.time_step_total:
        raise ValueError(
          "bike_qpos frame count must match robot motion length "
          f"({self.bike_qpos.shape[0]} vs {self.time_step_total})"
        )
    if "bike_body_pos_w" in data.files:
      self.bike_body_names = tuple(str(n) for n in data["bike_body_names"])
      self.bike_body_pos_w = torch.tensor(
        data["bike_body_pos_w"], dtype=torch.float32, device=device
      )
      self.bike_body_quat_w = torch.tensor(
        data["bike_body_quat_w"], dtype=torch.float32, device=device
      )
      if self.bike_body_pos_w.shape[0] != self.time_step_total:
        raise ValueError("bike_body_pos_w frame count must match robot motion length")
      if "bike_body_lin_vel_w" in data.files:
        self.bike_body_lin_vel_w = torch.tensor(
          data["bike_body_lin_vel_w"], dtype=torch.float32, device=device
        )
      if "bike_body_ang_vel_w" in data.files:
        self.bike_body_ang_vel_w = torch.tensor(
          data["bike_body_ang_vel_w"], dtype=torch.float32, device=device
        )

    self.bike_contact_flags: torch.Tensor | None = None
    if "bike_contact_flags" in data.files:
      flags = np.asarray(data["bike_contact_flags"], dtype=np.float32)
      n_pairs = len(BICYCLE_CONTACT_PAIR_KEYS)
      if flags.ndim != 2 or flags.shape[0] != self.time_step_total:
        raise ValueError(
          "bike_contact_flags shape must be "
          f"({self.time_step_total}, <= {n_pairs}), got {flags.shape}"
        )
      if flags.shape[1] < n_pairs:
        # Older motions (bike-only pairs) → zero-pad new channels (e.g. foot–ground).
        pad = np.zeros(
          (flags.shape[0], n_pairs - flags.shape[1]), dtype=np.float32
        )
        flags = np.concatenate([flags, pad], axis=1)
      elif flags.shape[1] > n_pairs:
        raise ValueError(
          "bike_contact_flags shape must be "
          f"({self.time_step_total}, <= {n_pairs}), got {flags.shape}"
        )
      self.bike_contact_flags = torch.tensor(flags, dtype=torch.float32, device=device)


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.bicycle: Entity | None = None
    if self.cfg.bicycle_entity_name:
      if self.motion.bike_qpos is None:
        raise ValueError(
          f"bicycle_entity_name={self.cfg.bicycle_entity_name!r} but "
          f"{self.cfg.motion_file} has no bike_qpos"
        )
      self.bicycle = env.scene[self.cfg.bicycle_entity_name]
      self._init_bicycle_tracking()
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  def _init_bicycle_tracking(self) -> None:
    from .bicycle_data import BICYCLE_TRACK_BODY

    assert self.bicycle is not None
    if self.motion.bike_body_pos_w is None:
      raise ValueError("bicycle tracking requires bike_body_pos_w in motion file")

    self._bike_track_body = BICYCLE_TRACK_BODY
    self._ee_body_indices = {
      name: self.cfg.body_names.index(name) for name in self.cfg.body_names
    }
    self._ref_bike_body_indices = {
      name: self.motion.bike_body_names.index(name)
      for name in self.motion.bike_body_names
    }
    bike_ids, _ = self.bicycle.find_bodies(
      list(self.motion.bike_body_names), preserve_order=True
    )
    self._sim_bike_body_ids = torch.tensor(
      bike_ids, dtype=torch.long, device=self.device
    )
    self._sim_bike_body_index = {
      name: i for i, name in enumerate(self.motion.bike_body_names)
    }

    masks: list[torch.Tensor] = []
    self._ref_contact_thresholds: list[float] = []
    if self.motion.bike_contact_flags is not None:
      self._ref_contact_mask = self.motion.bike_contact_flags
      print(
        "[MotionCommand] Using explicit bike_contact_flags from motion file "
        f"(active frac={self._ref_contact_mask.float().mean():.3f})"
      )
    else:
      active_frac = 0.12
      for pair in BICYCLE_CONTACT_PAIRS:
        ee_i = self._ee_body_indices[pair.ee_body]
        if pair.target_mode == "ground":
          height = self.motion.body_pos_w[:, ee_i, 2]
          threshold = pair.dist_threshold
          self._ref_contact_thresholds.append(threshold)
          masks.append(height < threshold)
          continue
        bike_i = self._ref_bike_body_indices[pair.bike_body]
        dist = torch.norm(
          self.motion.body_pos_w[:, ee_i] - self.motion.bike_body_pos_w[:, bike_i],
          dim=-1,
        )
        threshold = float(torch.quantile(dist, active_frac).item())
        threshold = max(threshold, pair.dist_threshold * 0.5)
        self._ref_contact_thresholds.append(threshold)
        masks.append(dist < threshold)
      self._ref_contact_mask = torch.stack(masks, dim=-1)
      print(
        "[MotionCommand] Using adaptive distance contact gate "
        f"(active frac~{active_frac:.2f})"
      )

  def _ref_bike_index(self, body_name: str) -> int:
    return self._ref_bike_body_indices[body_name]

  def _sim_bike_index(self, body_name: str) -> int:
    return self._sim_bike_body_index[body_name]

  @property
  def ref_bike_body_pos_w(self) -> torch.Tensor:
    """Reference bicycle body positions, shape (B, N_bike, 3)."""
    return (
      self.motion.bike_body_pos_w[self.time_steps]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def ref_bike_body_quat_w(self) -> torch.Tensor:
    return self.motion.bike_body_quat_w[self.time_steps]

  @property
  def sim_bike_body_pos_w(self) -> torch.Tensor:
    assert self.bicycle is not None
    return self.bicycle.data.body_link_pos_w[:, self._sim_bike_body_ids]

  @property
  def sim_bike_body_quat_w(self) -> torch.Tensor:
    assert self.bicycle is not None
    return self.bicycle.data.body_link_quat_w[:, self._sim_bike_body_ids]

  @property
  def sim_bike_body_lin_vel_w(self) -> torch.Tensor:
    assert self.bicycle is not None
    return self.bicycle.data.body_link_lin_vel_w[:, self._sim_bike_body_ids]

  @property
  def sim_bike_body_ang_vel_w(self) -> torch.Tensor:
    assert self.bicycle is not None
    return self.bicycle.data.body_link_ang_vel_w[:, self._sim_bike_body_ids]

  @property
  def ref_contact_mask(self) -> torch.Tensor:
    """Per-EE reference contact gate, shape (B, num_pairs)."""
    return self._ref_contact_mask[self.time_steps]

  def ref_bike_pos(self, body_name: str) -> torch.Tensor:
    idx = self._ref_bike_index(body_name)
    return self.ref_bike_body_pos_w[:, idx]

  def sim_bike_pos(self, body_name: str) -> torch.Tensor:
    idx = self._sim_bike_index(body_name)
    return self.sim_bike_body_pos_w[:, idx]

  def robot_body_pos_by_name(self, body_name: str) -> torch.Tensor:
    idx = self._ee_body_indices[body_name]
    return self.robot_body_pos_w[:, idx]

  def robot_body_quat_by_name(self, body_name: str) -> torch.Tensor:
    idx = self._ee_body_indices[body_name]
    return self.robot_body_quat_w[:, idx]

  def robot_body_lin_vel_by_name(self, body_name: str) -> torch.Tensor:
    idx = self._ee_body_indices[body_name]
    return self.robot_body_lin_vel_w[:, idx]

  def robot_body_ang_vel_by_name(self, body_name: str) -> torch.Tensor:
    idx = self._ee_body_indices[body_name]
    return self.robot_body_ang_vel_w[:, idx]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    if len(self.motion.clip_starts) != 1:
      raise ValueError("adaptive sampling is not supported for multi-motion bundles")
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    motion_ids = torch.randint(
      0, len(self.motion.clip_starts), (len(env_ids),), device=self.device
    )
    starts = self.motion.clip_starts[motion_ids]
    lengths = self.motion.clip_lengths[motion_ids]
    local_steps = torch.floor(
      torch.rand(len(env_ids), device=self.device) * lengths.float()
    ).long()
    self.motion_ids[env_ids] = motion_ids
    self.time_steps[env_ids] = starts + local_steps
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      start = int(self.cfg.start_frame)
      motion_ids = torch.randint(
        0, len(self.motion.clip_starts), (len(env_ids),), device=self.device
      )
      starts = self.motion.clip_starts[motion_ids]
      lengths = self.motion.clip_lengths[motion_ids]
      local_steps = torch.clamp(torch.full_like(lengths, start), min=0)
      local_steps = torch.minimum(local_steps, lengths - 1)
      self.motion_ids[env_ids] = motion_ids
      self.time_steps[env_ids] = starts + local_steps
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    # ADR boundary probes must evaluate the complete phase-dependent riding
    # sequence.  The controller assigns this mask before CommandManager.reset;
    # ordinary training environments retain the configured adaptive sampler.
    adr_probe_mask = getattr(self._env, "_bicycle_adr_probe_mask", None)
    if adr_probe_mask is not None:
      probe_ids = env_ids[adr_probe_mask[env_ids]]
      if probe_ids.numel() > 0:
        self.time_steps[probe_ids] = self.motion.clip_starts[self.motion_ids[probe_ids]]

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    joint_vel += sample_uniform(
      lower=self.cfg.joint_velocity_range[0],
      upper=self.cfg.joint_velocity_range[1],
      size=joint_vel.shape,
      device=joint_vel.device,
    )
    for joint_pattern, bounds in self.cfg.joint_velocity_range_overrides.items():
      joint_ids, _ = self.robot.find_joints(joint_pattern)
      if not joint_ids:
        raise ValueError(
          f"joint_velocity_range_overrides pattern {joint_pattern!r} matched no joints"
        )
      joint_vel[:, joint_ids] = self.joint_vel[:, joint_ids] + sample_uniform(
        lower=bounds[0],
        upper=bounds[1],
        size=(joint_vel.shape[0], len(joint_ids)),
        device=joint_vel.device,
      )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)
    self.write_bicycle_reference(env_ids=env_ids, apply_rsi_noise=True)

  def write_bicycle_reference(
    self,
    env_ids: torch.Tensor | slice | None = None,
    *,
    apply_rsi_noise: bool = False,
  ) -> None:
    """Write bicycle reference state, optionally with one-shot reset RSI noise."""
    if self.bicycle is None or self.motion.bike_qpos is None:
      return

    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      time_indices = self.time_steps
      env_origins = self._env.scene.env_origins
    else:
      time_indices = self.time_steps[env_ids]
      env_origins = self._env.scene.env_origins[env_ids]

    bike_qpos = self.motion.bike_qpos[time_indices]
    root_velocity, joint_vel = self.reference_bicycle_velocities(time_indices)

    root_pose = bike_qpos[:, :7].clone()
    root_pose[:, :3] += env_origins
    joint_pos = bike_qpos[:, 7:]

    if apply_rsi_noise:
      pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
      pose_ranges = torch.tensor(
        [self.cfg.bicycle_pose_range.get(key, (0.0, 0.0)) for key in pose_keys],
        device=self.device,
      )
      pose_noise = sample_uniform(
        pose_ranges[:, 0],
        pose_ranges[:, 1],
        (root_pose.shape[0], 6),
        device=self.device,
      )
      root_pose[:, :3] += pose_noise[:, :3]
      orientation_delta = quat_from_euler_xyz(
        pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]
      )
      root_pose[:, 3:7] = quat_mul(orientation_delta, root_pose[:, 3:7])

      velocity_keys = ("x", "y", "z", "roll", "pitch", "yaw")
      velocity_ranges = torch.tensor(
        [
          self.cfg.bicycle_velocity_range.get(key, (0.0, 0.0))
          for key in velocity_keys
        ],
        device=self.device,
      )
      velocity_noise = sample_uniform(
        velocity_ranges[:, 0],
        velocity_ranges[:, 1],
        (root_velocity.shape[0], 6),
        device=self.device,
      )
      root_velocity += velocity_noise

      for joint_pattern, bounds in self.cfg.bicycle_joint_position_ranges.items():
        joint_ids, _ = self.bicycle.find_joints(joint_pattern)
        if not joint_ids:
          raise ValueError(
            f"bicycle_joint_position_ranges pattern {joint_pattern!r} matched no joints"
          )
        joint_pos[:, joint_ids] += sample_uniform(
          lower=bounds[0],
          upper=bounds[1],
          size=(joint_pos.shape[0], len(joint_ids)),
          device=joint_pos.device,
        )

    self.bicycle.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    self.bicycle.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
    self.bicycle.write_joint_state_to_sim(
      joint_pos, joint_vel, env_ids=env_ids
    )

  def reference_bicycle_velocities(
    self, time_indices: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical world root velocity and hinge velocity.

    Historical bicycle NPZs store ``bike_qvel`` as the 13-D numerical
    derivative of ``[xyz, quaternion, joints]``.  That quaternion derivative
    is not a MuJoCo angular velocity.  New physics-aware tasks opt into the
    conversion below; legacy tasks retain their exact historical reset behavior.
    """
    assert self.motion.bike_qpos is not None
    assert self.motion.bike_qvel is not None
    q = self.motion.bike_qpos[time_indices]
    v = self.motion.bike_qvel[time_indices]
    if v.shape[-1] >= 13:
      joint_vel = v[:, 7:13]
      if self.cfg.convert_bike_qvel_from_qpos_derivative:
        quat = q[:, 3:7]
        quat_dot = v[:, 3:7]
        # Remove numerical radial drift, then omega_world = 2*qdot*conj(q).
        quat_dot = quat_dot - (quat_dot * quat).sum(dim=-1, keepdim=True) * quat
        omega_quat = quat_mul(quat_dot, quat_inv(quat))
        ang_vel_w = 2.0 * omega_quat[:, 1:4]
        root_vel = torch.cat((v[:, :3], ang_vel_w), dim=-1)
      else:
        root_vel = v[:, :6]
    else:
      root_vel = v[:, :6]
      joint_vel = v[:, 6:12]
    return root_vel, joint_vel

  def _update_command(self):
    self.time_steps += 1
    clip_ends = (
      self.motion.clip_starts[self.motion_ids]
      + self.motion.clip_lengths[self.motion_ids]
    )
    env_ids = torch.where(self.time_steps >= clip_ends)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def clamp_reference_steps(self, frames: torch.Tensor) -> torch.Tensor:
    """Clamp per-environment frame indices to the selected clip boundary."""
    starts = self.motion.clip_starts[self.motion_ids]
    ends = starts + self.motion.clip_lengths[self.motion_ids] - 1
    return torch.maximum(torch.minimum(frames, ends), starts)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        if self.bicycle is not None and self.motion.bike_qpos is not None:
          bike = self.bicycle
          bike_free_q = bike.indexing.free_joint_q_adr.cpu().numpy()
          bike_joint_q = bike.indexing.joint_q_adr.cpu().numpy()
          bike_q = self.motion.bike_qpos[self.time_steps[batch]].cpu().numpy()
          qpos[bike_free_q[0:3]] = bike_q[0:3] + self._env.scene.env_origins[batch].cpu().numpy()
          qpos[bike_free_q[3:7]] = bike_q[3:7]
          qpos[bike_joint_q] = bike_q[7:]

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  bicycle_entity_name: str | None = None
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  joint_velocity_range: tuple[float, float] = (0.0, 0.0)
  joint_velocity_range_overrides: dict[str, tuple[float, float]] = field(
    default_factory=dict
  )
  bicycle_pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  bicycle_velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  bicycle_joint_position_ranges: dict[str, tuple[float, float]] = field(
    default_factory=dict
  )
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  start_frame: int = 0
  """Frame index used when ``sampling_mode='start'`` (clamped to motion length)."""
  convert_bike_qvel_from_qpos_derivative: bool = False
  """Convert legacy 13-D qpos derivatives to canonical root angular velocity."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
