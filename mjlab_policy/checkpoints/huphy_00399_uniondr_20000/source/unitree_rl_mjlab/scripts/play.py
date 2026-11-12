"""Script to play RL agent with RSL-RL."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.tracking.mdp.bicycle_curriculum import STEPS_PER_ITER
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  viser_port: int | None = None
  """Use this TCP port for the Viser server instead of its default."""
  assistance_scale: float | None = None
  """Freeze bicycle curriculum assistance at this value in [0, 1] during play."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  dr_easier: bool = False
  """Wire train DR events, force severity on, and contract ranges to DR-Easier 50%."""
  no_debug_vis: bool = False
  """Disable reference ghost / debug visualization (motion command debug_vis)."""
  show_collision: bool = False
  """Show collision geoms (group 3) together with visual meshes."""
  collision_only: bool = False
  """Hide visual meshes and show only collision geoms (group 3). Viser also gets Mesh/Collision toggles."""
  contact_force_panel: bool = False
  """Viser only: show live pedal, handle, and saddle world-frame forces in N."""
  contact_log_dir: str | None = None
  """If set, dump per-timestep XYZ forces and per-rollout contact quality here."""
  depth_map_panel: bool = False
  """Viser only: show the live 48x27 normalized policy depth input."""
  start_frame: int = 0
  """Tracking only: spawn robot+bike at this motion frame and run policy inference from there."""
  steer_damping: float | None = None
  """Temporary play-only override of bicycle/steer joint damping. Does not change training."""
  steer_frictionloss: float | None = None
  """Temporary play-only override of bicycle/steer joint frictionloss. Does not change training."""
  crank_damping: float | None = None
  """Temporary play-only override of bicycle/crank_spin damping. Does not change training."""
  crank_frictionloss: float | None = None
  """Temporary play-only override of bicycle/crank_spin frictionloss. Does not change training."""
  crank_armature: float | None = None
  """Temporary play-only override of bicycle/crank_spin armature. Does not change training."""
  pelvis_push_n: float | None = None
  """Play-only horizontal pelvis pulses of this magnitude (N). Does not change training."""
  pelvis_push_count: int = 2
  """How many ride-window pelvis pulses to fire (evenly spaced while both pedals are ON)."""
  steer_axis_torque_nm: float | None = None
  """Play-only fork torque about the real steer axis (N·m). Does not change training."""
  steer_axis_torque_frame: int = 50
  """Motion frame at which to fire the one-shot steer-axis torque pulse."""
  resmimic_base_policy: str | None = None
  """If set, policy actions are residuals added to frozen ResMimic base."""
  resmimic_residual_scale: float = 1.0
  default_pose_from_motion: str | None = None
  """Play-only: set robot default/init joint pose from this motion NPZ frame 0."""
  default_pose_frame: int = 0
  """Frame index used with ``--default-pose-from-motion``."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _override_bicycle_joint(
  env: ManagerBasedRlEnv,
  joint_name: str,
  *,
  damping: float | None = None,
  frictionloss: float | None = None,
  armature: float | None = None,
) -> None:
  """Patch compiled bicycle dof params on both CPU mj_model and Warp."""
  sim = env.unwrapped.sim
  mjm = sim.mj_model
  full_name = f"bicycle/{joint_name}"
  try:
    jnt = mjm.joint(full_name)
  except KeyError as exc:
    raise RuntimeError(f"Play env has no {full_name} joint to override") from exc
  dof = int(jnt.dofadr[0])
  parts = []
  if damping is not None:
    mjm.dof_damping[dof] = float(damping)
    sim.model.dof_damping[..., dof] = float(damping)
    if "dof_damping" in sim.default_model_fields:
      sim.default_model_fields["dof_damping"][..., dof] = float(damping)
    parts.append(f"damping={float(damping):g}")
  if frictionloss is not None:
    mjm.dof_frictionloss[dof] = float(frictionloss)
    sim.model.dof_frictionloss[..., dof] = float(frictionloss)
    if "dof_frictionloss" in sim.default_model_fields:
      sim.default_model_fields["dof_frictionloss"][..., dof] = float(frictionloss)
    parts.append(f"frictionloss={float(frictionloss):g}")
  if armature is not None:
    mjm.dof_armature[dof] = float(armature)
    sim.model.dof_armature[..., dof] = float(armature)
    if "dof_armature" in sim.default_model_fields:
      sim.default_model_fields["dof_armature"][..., dof] = float(armature)
    parts.append(f"armature={float(armature):g}")
  sim.recompute_constants(RecomputeLevel.set_const_0)
  print(
    f"[INFO]: Play-only {full_name} override ({', '.join(parts)}; "
    "training model unchanged)"
  )


class _PelvisPushPlayWrapper:
  """Apply a few play-only horizontal pelvis pulses during the ride window."""

  _RIDE_START = 608
  _RIDE_END = 760
  _PULSE_STEPS = 5  # 0.1 s at 50 Hz
  _ARROW_HOLD_STEPS = 150  # keep the direction arrow visible after the pulse
  _ARROW_M_PER_N = 0.01  # 70 N -> 0.7 m
  _ARROW_COLOR = (1.0, 0.28, 0.05, 1.0)
  _ARROW_WIDTH = 0.03

  def __init__(self, env: ManagerBasedRlEnv, *, magnitude: float, count: int):
    self._env = env
    self._magnitude = float(magnitude)
    self._count = max(1, int(count))
    robot = env.scene["robot"]
    pelvis_ids, _ = robot.find_bodies("pelvis")
    self._robot = robot
    self._pelvis_id = int(pelvis_ids[0])
    self._zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)
    span = self._RIDE_END - self._RIDE_START
    self._frames = [
      self._RIDE_START + (i + 1) * span // (self._count + 1)
      for i in range(self._count)
    ]
    self._reset_state()
    inner = env.unwrapped
    orig_update = inner.update_visualizers

    def _update_visualizers(visualizer):
      orig_update(visualizer)
      self.draw_debug(visualizer)

    inner.update_visualizers = _update_visualizers
    print(
      f"[INFO]: Play-only pelvis push |F|={self._magnitude:g} N, "
      f"{self._count} pulses at frames {self._frames} "
      f"(0.1 s, random yaw; orange arrow {self._ARROW_M_PER_N * 100:.0f} cm/N; "
      "training unchanged)"
    )

  def _reset_state(self) -> None:
    n = self._env.num_envs
    dev = self._env.device
    self._fired = torch.zeros(n, self._count, device=dev, dtype=torch.bool)
    self._pulse_left = torch.zeros(n, device=dev, dtype=torch.long)
    self._force = torch.zeros(n, 3, device=dev)
    self._viz_force = torch.zeros(n, self._count, 3, device=dev)
    self._viz_left = torch.zeros(n, self._count, device=dev, dtype=torch.long)
    self._last_t = torch.full((n,), -1, device=dev, dtype=torch.long)

  def _apply(self) -> None:
    from src.tasks.tracking.mdp import MotionCommand
    from src.tasks.tracking.mdp.bicycle_data import BICYCLE_CONTACT_PAIR_KEYS

    cmd = self._env.command_manager.get_term("motion")
    if not isinstance(cmd, MotionCommand):
      return
    t = cmd.time_steps
    if bool((t < self._last_t).any()):
      self._reset_state()
      t = cmd.time_steps
    self._last_t = t.clone()
    left_i = BICYCLE_CONTACT_PAIR_KEYS.index("pedal_left")
    right_i = BICYCLE_CONTACT_PAIR_KEYS.index("pedal_right")
    ride = cmd.ref_contact_mask[:, left_i].bool() & cmd.ref_contact_mask[:, right_i].bool()
    for i, frame in enumerate(self._frames):
      due = (t == frame) & ride & ~self._fired[:, i]
      if not bool(due.any()):
        continue
      ids = due.nonzero().flatten()
      ang = torch.rand(ids.numel(), device=self._env.device) * (2.0 * math.pi)
      self._force[ids, 0] = self._magnitude * torch.cos(ang)
      self._force[ids, 1] = self._magnitude * torch.sin(ang)
      self._force[ids, 2] = 0.0
      self._pulse_left[ids] = self._PULSE_STEPS
      self._viz_force[ids, i] = self._force[ids]
      self._viz_left[ids, i] = self._ARROW_HOLD_STEPS
      self._fired[ids, i] = True
      for j, env_id in enumerate(ids.tolist()):
        print(
          f"[INFO]: pelvis push {i + 1}/{self._count} env={env_id} "
          f"|F|={self._magnitude:g} N yaw={float(ang[j].item()) * 180.0 / math.pi:.1f} deg "
          f"frame={frame}",
          flush=True,
        )
    if bool((self._viz_left > 0).any()):
      self._viz_left = torch.clamp(self._viz_left - 1, min=0)
      expired_viz = self._viz_left == 0
      if bool(expired_viz.any()):
        self._viz_force[expired_viz] = 0.0
    if bool((self._pulse_left > 0).any()):
      self._robot.write_external_wrench_to_sim(
        self._force.unsqueeze(1), self._zeros, body_ids=[self._pelvis_id]
      )
      self._pulse_left = torch.clamp(self._pulse_left - 1, min=0)
      expired = self._pulse_left == 0
      if bool(expired.any()):
        self._force[expired] = 0.0
    else:
      self._robot.write_external_wrench_to_sim(
        self._zeros, self._zeros, body_ids=[self._pelvis_id]
      )

  def draw_debug(self, visualizer) -> None:
    """Draw an orange world-frame arrow along the current pelvis push."""
    if not bool((self._viz_left > 0).any()):
      return
    pos = self._robot.data.body_link_pos_w[:, self._pelvis_id]
    for env_id in visualizer.get_env_indices(self._env.num_envs):
      start = pos[env_id]
      for i in range(self._count):
        if int(self._viz_left[env_id, i].item()) <= 0:
          continue
        force = self._viz_force[env_id, i]
        if float(force.norm().item()) < 1e-6:
          continue
        visualizer.add_arrow(
          start,
          start + force * self._ARROW_M_PER_N,
          color=self._ARROW_COLOR,
          width=self._ARROW_WIDTH,
          label=f"pelvis_push_{env_id}_{i}",
        )

  def reset(self, *args, **kwargs):
    self._reset_state()
    return self._env.reset(*args, **kwargs)

  def step(self, actions):
    self._apply()
    return self._env.step(actions)

  def __getattr__(self, name: str):
    return getattr(self._env, name)


class _SteerAxisTorquePlayWrapper:
  """Apply one play-only steer-axis torque pulse, matching DR's fork axis."""

  _PULSE_STEPS = 5  # 0.1 s at 50 Hz, same as DR
  _STEER_AXIS_LOCAL = (-0.34202, 0.0, 0.93969)
  _ARROW_HOLD_STEPS = 150
  _ARROW_M = 0.45
  _ARROW_COLOR = (0.15, 0.75, 1.0, 1.0)
  _ARROW_WIDTH = 0.025

  def __init__(
    self, env: ManagerBasedRlEnv, *, magnitude_nm: float, frame: int
  ):
    from mjlab.utils.lab_api.math import quat_apply

    self._env = env
    self._quat_apply = quat_apply
    self._magnitude = float(magnitude_nm)
    self._frame = int(frame)
    bicycle = env.scene["bicycle"]
    fork_ids, _ = bicycle.find_bodies("fork")
    self._bicycle = bicycle
    self._fork_id = int(fork_ids[0])
    self._axis_local = torch.tensor(
      self._STEER_AXIS_LOCAL, device=env.device, dtype=torch.float32
    )
    self._axis_local = self._axis_local / self._axis_local.norm().clamp_min(1.0e-8)
    self._zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)
    self._reset_state()
    inner = env.unwrapped
    orig_update = inner.update_visualizers

    def _update_visualizers(visualizer):
      orig_update(visualizer)
      self.draw_debug(visualizer)

    inner.update_visualizers = _update_visualizers
    print(
      f"[INFO]: Play-only steer-axis torque {self._magnitude:g} N·m "
      f"at frame {self._frame} (0.1 s, DR fork axis; training unchanged)"
    )

  def _reset_state(self) -> None:
    n = self._env.num_envs
    dev = self._env.device
    self._fired = torch.zeros(n, device=dev, dtype=torch.bool)
    self._pulse_left = torch.zeros(n, device=dev, dtype=torch.long)
    self._scalar = torch.zeros(n, device=dev)
    self._viz_left = torch.zeros(n, device=dev, dtype=torch.long)
    self._last_t = torch.full((n,), -1, device=dev, dtype=torch.long)

  def _apply(self) -> None:
    from src.tasks.tracking.mdp import MotionCommand

    cmd = self._env.command_manager.get_term("motion")
    if not isinstance(cmd, MotionCommand):
      return
    t = cmd.time_steps
    if bool((t < self._last_t).any()):
      self._reset_state()
      t = cmd.time_steps
    self._last_t = t.clone()
    due = (t == self._frame) & ~self._fired
    if bool(due.any()):
      ids = due.nonzero().flatten()
      self._scalar[ids] = self._magnitude
      self._pulse_left[ids] = self._PULSE_STEPS
      self._viz_left[ids] = self._ARROW_HOLD_STEPS
      self._fired[ids] = True
      for env_id in ids.tolist():
        print(
          f"[INFO]: steer-axis torque env={env_id} "
          f"tau={self._magnitude:g} N·m frame={self._frame}",
          flush=True,
        )
    if bool((self._viz_left > 0).any()):
      self._viz_left = torch.clamp(self._viz_left - 1, min=0)
    quat = self._bicycle.data.body_link_quat_w[:, self._fork_id]
    axis_w = self._quat_apply(
      quat, self._axis_local.expand(self._env.num_envs, -1)
    )
    if bool((self._pulse_left > 0).any()):
      torque_w = axis_w * self._scalar.unsqueeze(-1)
      self._bicycle.write_external_wrench_to_sim(
        self._zeros, torque_w.unsqueeze(1), body_ids=[self._fork_id]
      )
      self._pulse_left = torch.clamp(self._pulse_left - 1, min=0)
      expired = self._pulse_left == 0
      if bool(expired.any()):
        self._scalar[expired] = 0.0
    else:
      self._bicycle.write_external_wrench_to_sim(
        self._zeros, self._zeros, body_ids=[self._fork_id]
      )

  def draw_debug(self, visualizer) -> None:
    if not bool((self._viz_left > 0).any()):
      return
    pos = self._bicycle.data.body_link_pos_w[:, self._fork_id]
    quat = self._bicycle.data.body_link_quat_w[:, self._fork_id]
    axis_w = self._quat_apply(
      quat, self._axis_local.expand(self._env.num_envs, -1)
    )
    for env_id in visualizer.get_env_indices(self._env.num_envs):
      if int(self._viz_left[env_id].item()) <= 0:
        continue
      start = pos[env_id]
      visualizer.add_arrow(
        start,
        start + axis_w[env_id] * self._ARROW_M,
        color=self._ARROW_COLOR,
        width=self._ARROW_WIDTH,
        label=f"steer_axis_torque_{env_id}",
      )

  def reset(self, *args, **kwargs):
    self._reset_state()
    return self._env.reset(*args, **kwargs)

  def step(self, actions):
    self._apply()
    return self._env.step(actions)

  def __getattr__(self, name: str):
    return getattr(self._env, name)


_G1_29_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def _apply_default_pose_from_motion(env_cfg, motion_file: str, *, frame: int) -> None:
  """Play-only: replace robot HOME default joints with a motion frame."""
  import numpy as np
  from dataclasses import replace

  path = Path(motion_file)
  if not path.exists():
    raise FileNotFoundError(f"--default-pose-from-motion not found: {path}")
  data = np.load(path)
  if "joint_pos" not in data.files:
    raise KeyError(f"{path} has no joint_pos")
  q = np.asarray(data["joint_pos"][int(frame)], dtype=np.float32)
  if q.shape != (len(_G1_29_JOINT_NAMES),):
    raise ValueError(
      f"motion joint_pos frame {frame} has shape {q.shape}, expected "
      f"({len(_G1_29_JOINT_NAMES)},)"
    )
  robot_cfg = env_cfg.scene.entities.get("robot")
  if robot_cfg is None:
    raise KeyError("env has no robot entity for default-pose override")
  pelvis_z = None
  if "body_pos_w" in data.files:
    pelvis_z = float(np.asarray(data["body_pos_w"][int(frame), 0, 2]))
  joint_pos = {name: float(q[i]) for i, name in enumerate(_G1_29_JOINT_NAMES)}
  pos = robot_cfg.init_state.pos
  if pelvis_z is not None:
    pos = (float(pos[0]), float(pos[1]), pelvis_z)
  robot_cfg.init_state = replace(
    robot_cfg.init_state, joint_pos=joint_pos, joint_vel={".*": 0.0}, pos=pos
  )
  print(
    f"[INFO]: Play-only default pose from {path.name} frame {frame} "
    f"(pelvis_z={pos[2]:.3f} m); action offset uses this pose"
  )


def _apply_dr_easier_play_overrides(env_cfg) -> None:
  """Reuse the train DR stack with DR-Easier magnitudes for interactive play.

  ``play=True`` factories omit ``bicycle_v3_dr_reset``/``push``. Loading the
  train cfg and then applying play cosmetics keeps checkpoint obs/network
  contracts while making mass/friction/geometry/push and tire DR visible.
  """
  env_cfg.episode_length_s = int(1e9)
  motion = env_cfg.commands.get("motion") if env_cfg.commands else None
  if motion is not None and hasattr(motion, "sampling_mode"):
    motion.sampling_mode = "start"
  actor_obs = env_cfg.observations.get("actor") if env_cfg.observations else None
  if actor_obs is not None:
    actor_obs.enable_corruption = False

  events = env_cfg.events or {}
  schedule = events.get("bicycle_v3_dr_schedule_init")
  if schedule is None:
    raise KeyError("DR-Easier play needs bicycle_v3_dr_schedule_init; use a V3 DR task")
  if "bicycle_v3_dr_reset" not in events:
    raise KeyError("DR-Easier play needs bicycle_v3_dr_reset; train DR was not wired")
  schedule.params.update(
    {
      "core_start_iter": -1.0,
      "core_end_iter": 0.0,
      "late_start_iter": -1.0,
      "late_end_iter": 0.0,
    }
  )
  # Live bicycle_dr consumes this; older snapshots may not declare the kwarg.
  schedule.params["range_scale"] = 0.5
  tire = events.get("bicycle_v3_tire_width_v2")
  if tire is not None:
    tire.params["width_range"] = (0.01375, 0.01875)
  print(
    "[INFO]: DR-Easier play: train DR on, range_scale=0.5, "
    "severity forced to 1, tire 2.75-3.75 cm"
  )


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=not cfg.dr_easier)
  agent_cfg = load_rl_cfg(task_id)
  if cfg.dr_easier:
    _apply_dr_easier_play_overrides(env_cfg)

  if cfg.assistance_scale is not None:
    scale = float(cfg.assistance_scale)
    if not 0.0 <= scale <= 1.0:
      raise ValueError("--assistance-scale must be in [0, 1]")
    init_cfg = env_cfg.events.get("bicycle_curriculum_init")
    if init_cfg is None:
      raise ValueError(f"Task {task_id!r} has no bicycle curriculum to override")
    mode = init_cfg.params.get("mode")
    init_cfg.params["start_scale"] = scale
    # Effectively freeze the selected scale for an interactive play session.
    init_cfg.params["decay_until_iter"] = 1.0e12
    if mode == "PD":
      # A single play env would otherwise receive the first cohort level (0).
      init_cfg.params["pd_assist_levels"] = (1.0,)
    print(f"[INFO]: Frozen bicycle assistance: mode={mode}, scale={scale:.3f}")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )

    if cfg.no_debug_vis:
      motion_cmd.debug_vis = False
      print("[INFO]: Reference ghost / motion debug_vis disabled")

    if cfg.start_frame > 0:
      # Play cfgs already use sampling_mode='start'; override the reset index
      # so robot + bike teleport to mid-clip and the policy tracks from there.
      motion_cmd.sampling_mode = "start"
      motion_cmd.start_frame = int(cfg.start_frame)
      print(
        f"[INFO]: Motion start_frame={motion_cmd.start_frame} "
        "(policy inference begins at this reference frame on each reset)"
      )
  elif cfg.start_frame > 0:
    print("[WARN]: --start-frame ignored (task has no motion command)")
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.default_pose_from_motion:
    _apply_default_pose_from_motion(
      env_cfg,
      cfg.default_pose_from_motion,
      frame=cfg.default_pose_frame,
    )

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if (
    TRAINED_MODE
    and resume_path is not None
    and "bicycle_v3_tire_width_v2" in (env_cfg.events or {})
  ):
    # Wheel-width curriculum/DR tasks now build their play env with the
    # cylinder-tire XML too (see _apply_reward_dr_v3). The env's implicit
    # first reset (inside ManagerBasedRlEnv.__init__ above) does not
    # reliably apply the reset-mode tire-width DR event -- empirically it
    # can leave geom_size at whatever it happened to be before the
    # curriculum event ran (observed: near the curriculum's *end* value
    # regardless of common_step_counter), unlike an explicit post-
    # construction env.reset() call, which does apply it correctly. So:
    # peek the checkpoint's saved iteration, set common_step_counter to
    # match, and force an explicit reset.
    try:
      ckpt_iter = torch.load(resume_path, map_location="cpu", weights_only=False)["iter"]
      env.common_step_counter = int(ckpt_iter) * STEPS_PER_ITER
      env.reset()
      # The Viser/native-viewer collision mesh is built from the raw CPU
      # mj_model, not the live warp tensor env.reset() just updated (see
      # mjlab/viewer/viser/conversions.py) -- sync both or the physics is
      # right but the displayed tire stays at its stale pre-reset size.
      mjm = env.sim.mj_model
      for name in ("rear_tire_col", "front_tire_col"):
        gid = mjm.geom(f"bicycle/{name}").id
        mjm.geom_size[gid, 1] = env.sim.model.geom_size[0, gid, 1].item()
      print(
        f"[INFO]: Re-seeded common_step_counter for checkpoint iter={ckpt_iter} "
        "(wheel-width curriculum/DR tire geometry now matches)"
      )
    except (KeyError, FileNotFoundError):
      print(
        "[WARN]: Could not read checkpoint iteration for tire-width matching; "
        "tire geometry may not match the loaded checkpoint."
      )

  if cfg.steer_damping is not None or cfg.steer_frictionloss is not None:
    _override_bicycle_joint(
      env,
      "steer",
      damping=cfg.steer_damping,
      frictionloss=cfg.steer_frictionloss,
    )
  if (
    cfg.crank_damping is not None
    or cfg.crank_frictionloss is not None
    or cfg.crank_armature is not None
  ):
    _override_bicycle_joint(
      env,
      "crank_spin",
      damping=cfg.crank_damping,
      frictionloss=cfg.crank_frictionloss,
      armature=cfg.crank_armature,
    )

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  if cfg.pelvis_push_n is not None:
    env = _PelvisPushPlayWrapper(
      env,
      magnitude=cfg.pelvis_push_n,
      count=cfg.pelvis_push_count,
    )
  if cfg.steer_axis_torque_nm is not None:
    env = _SteerAxisTorquePlayWrapper(
      env,
      magnitude_nm=cfg.steer_axis_torque_nm,
      frame=cfg.steer_axis_torque_frame,
    )

  if cfg.resmimic_base_policy:
    from src.tasks.tracking.rl import ResMimicResidualVecEnvWrapper

    env = ResMimicResidualVecEnvWrapper(
      env,
      base_policy_path=cfg.resmimic_base_policy,
      clip_actions=agent_cfg.clip_actions,
      residual_scale=cfg.resmimic_residual_scale,
    )
    print(f"[INFO]: ResMimic residual wrapper enabled ({cfg.resmimic_base_policy})")
  else:
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    alg = getattr(runner, "alg", None)
    if alg is not None and hasattr(alg, "get_play_policy"):
      alg.eval_mode()
      policy = alg.get_play_policy().to(device)
      play_name = type(policy).__name__
      phase = getattr(alg, "phase", None)
      print(
        f"[INFO]: Play policy={play_name}"
        + (f" (haic_phase={phase})" if phase is not None else "")
      )
    else:
      policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    viewer = NativeMujocoViewer(env, policy)
    _patch_play_viz(viewer, cfg, backend="native")
    viewer.run()
  elif resolved_viewer == "viser":
    viser_server = None
    if cfg.viser_port is not None:
      import viser

      viser_server = viser.ViserServer(
        host="0.0.0.0", port=int(cfg.viser_port), label="mjlab play"
      )
      print(f"[INFO]: Viser URL: http://localhost:{cfg.viser_port}")
    viewer = ViserPlayViewer(env, policy, viser_server=viser_server)
    _patch_play_viz(viewer, cfg, backend="viser")
    viewer.run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def _apply_viser_geom_preset(scene, *, collision_only: bool) -> None:
  """Mesh view = G0-G2 (+G4/G5); collision-only = G3+G4 (default collision
  class plus RubberBox hand/foot reward-pad geoms, which live in G4)."""
  if collision_only:
    scene.geom_groups_visible = [False, False, False, True, True, False]
  else:
    scene.geom_groups_visible = [True, True, True, False, True, True]
  scene._sync_visibilities()
  scene._request_update()


def _patch_play_viz(
  viewer, cfg: PlayConfig, backend: Literal["native", "viser"]
) -> None:
  """Hook viewer.setup so collision / ghost overrides apply after scene exists."""
  want_collision_ui = cfg.show_collision or cfg.collision_only
  want_contact_log = bool(cfg.contact_force_panel or cfg.contact_log_dir)
  if not (
    want_collision_ui
    or cfg.no_debug_vis
    or want_contact_log
    or cfg.depth_map_panel
  ):
    return

  from src.tasks.tracking.mdp.bicycle_data import (
    BICYCLE_CONTACT_SENSOR_NAMES_BY_PAIR,
  )

  force_sensor_names = (
    ("handle_left", "Left handle", BICYCLE_CONTACT_SENSOR_NAMES_BY_PAIR["handle_left"]),
    ("handle_right", "Right handle", BICYCLE_CONTACT_SENSOR_NAMES_BY_PAIR["handle_right"]),
    ("saddle", "Saddle", ("contact_saddle",)),
    ("pedal_left", "Left pedal", ("contact_pedal_left",)),
    ("pedal_right", "Right pedal", ("contact_pedal_right",)),
  )
  pair_keys = tuple(key for key, _, _ in force_sensor_names)
  log_state = {
    "rollout": 0,
    "rows": [],
    "sched": {k: 0 for k in pair_keys},
    "hit": {k: 0 for k in pair_keys},
  }

  def _env_idx() -> int:
    scene = getattr(viewer, "_scene", None)
    if scene is not None:
      return int(scene.env_idx)
    return 0

  def _contact_snapshot() -> dict | None:
    env = viewer.env.unwrapped
    i = _env_idx()
    try:
      command = env.command_manager.get_term("motion")
    except (AttributeError, KeyError):
      return None
    mask = command.ref_contact_mask
    from src.tasks.tracking.mdp.bicycle_data import BICYCLE_CONTACT_PAIR_KEYS

    snapshot = {
      "step": int(env.episode_length_buf[i].item()),
      "time_s": float(env.episode_length_buf[i].item()) * float(env.step_dt),
      "frame": int(command.time_steps[i].item()),
    }
    for key, _, sensor_names in force_sensor_names:
      pair_idx = BICYCLE_CONTACT_PAIR_KEYS.index(key)
      scheduled = bool(mask[i, pair_idx].item())
      try:
        in_contact = False
        vec = None
        for sensor_name in sensor_names:
          sensor = env.scene[sensor_name]
          found = sensor.data.found
          force = sensor.data.force
          if found is not None:
            in_contact |= bool(found[i].reshape(-1).any().item())
          if force is not None:
            value = force[i].reshape(-1, 3).sum(dim=0)
            vec = value if vec is None else vec + value
        if vec is None:
          fx = fy = fz = 0.0
        else:
          fx, fy, fz = (float(v) for v in vec.tolist())
      except (KeyError, RuntimeError):
        in_contact = False
        fx = fy = fz = 0.0
      snapshot[f"{key}_sched"] = int(scheduled)
      snapshot[f"{key}_found"] = int(in_contact)
      snapshot[f"{key}_fx"] = fx
      snapshot[f"{key}_fy"] = fy
      snapshot[f"{key}_fz"] = fz
    return snapshot

  def _quality_from_counts() -> tuple[float, int, int]:
    sched_n = sum(log_state["sched"].values())
    hit_n = sum(log_state["hit"].values())
    pct = 100.0 * hit_n / sched_n if sched_n else 0.0
    return pct, hit_n, sched_n

  def _flush_rollout(reason: str) -> None:
    rows = log_state["rows"]
    if not rows:
      return
    pct, hit_n, sched_n = _quality_from_counts()
    per_pair = {}
    for key in pair_keys:
      s = log_state["sched"][key]
      h = log_state["hit"][key]
      per_pair[key] = None if s == 0 else 100.0 * h / s
    n = log_state["rollout"]
    mean_f = {}
    for key in pair_keys:
      mag = [
        (r[f"{key}_fx"] ** 2 + r[f"{key}_fy"] ** 2 + r[f"{key}_fz"] ** 2) ** 0.5
        for r in rows
      ]
      mean_f[key] = sum(mag) / len(mag)
    pair_txt = " ".join(
      f"{k}={per_pair[k]:.1f}%" if per_pair[k] is not None else f"{k}=n/a"
      for k in pair_keys
    )
    print(
      f"[CONTACT] rollout={n:04d} steps={len(rows)} reason={reason} "
      f"quality={pct:.1f}% ({hit_n}/{sched_n} scheduled) | {pair_txt}",
      flush=True,
    )
    print(
      "[CONTACT] mean |F| N: "
      + " ".join(f"{k}={mean_f[k]:.1f}" for k in pair_keys),
      flush=True,
    )
    if cfg.contact_log_dir:
      out_dir = Path(cfg.contact_log_dir)
      out_dir.mkdir(parents=True, exist_ok=True)
      csv_path = out_dir / f"rollout_{n:04d}.csv"
      cols = ["step", "time_s", "frame"]
      for key in pair_keys:
        cols.extend(
          [
            f"{key}_sched",
            f"{key}_found",
            f"{key}_fx",
            f"{key}_fy",
            f"{key}_fz",
          ]
        )
      with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
          f.write(",".join(str(row[c]) for c in cols) + "\n")
      summary = {
        "rollout": n,
        "steps": len(rows),
        "reason": reason,
        "quality_pct": pct,
        "scheduled": sched_n,
        "achieved": hit_n,
        "per_pair_pct": per_pair,
        "mean_force_norm_n": mean_f,
        "csv": str(csv_path),
      }
      with (out_dir / "summaries.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
      print(f"[CONTACT] wrote {csv_path}", flush=True)
    log_state["rollout"] += 1
    log_state["rows"] = []
    log_state["sched"] = {k: 0 for k in pair_keys}
    log_state["hit"] = {k: 0 for k in pair_keys}

  def _record_contact_step() -> None:
    snap = _contact_snapshot()
    if snap is None:
      return
    log_state["rows"].append(snap)
    for key in pair_keys:
      if snap[f"{key}_sched"]:
        log_state["sched"][key] += 1
        if snap[f"{key}_found"]:
          log_state["hit"][key] += 1

  def _contact_force_html() -> str:
    snap = _contact_snapshot()
    if snap is None:
      return "<div style='padding:4px 8px'>contact sensors unavailable</div>"
    rows: list[str] = [
      "<tr style='opacity:0.7'>"
      "<td></td><td>sch</td><td>hit</td>"
      "<td>Fx</td><td>Fy</td><td>Fz</td><td>|F|</td></tr>"
    ]
    for key, label, _ in force_sensor_names:
      fx, fy, fz = snap[f"{key}_fx"], snap[f"{key}_fy"], snap[f"{key}_fz"]
      mag = (fx * fx + fy * fy + fz * fz) ** 0.5
      sch = "ON" if snap[f"{key}_sched"] else "off"
      hit = "Y" if snap[f"{key}_found"] else "n"
      rows.append(
        "<tr>"
        f"<td style='padding:2px 10px 2px 0'>{label}</td>"
        f"<td>{sch}</td><td>{hit}</td>"
        f"<td style='font-family:monospace;text-align:right'>{fx:6.1f}</td>"
        f"<td style='font-family:monospace;text-align:right'>{fy:6.1f}</td>"
        f"<td style='font-family:monospace;text-align:right'>{fz:6.1f}</td>"
        f"<td style='font-family:monospace;text-align:right'><b>{mag:6.1f}</b></td>"
        "</tr>"
      )
    pct, hit_n, sched_n = _quality_from_counts()
    pair_bits = []
    for key, label, _ in force_sensor_names:
      s = log_state["sched"][key]
      h = log_state["hit"][key]
      pair_bits.append(
        f"{label}: n/a" if s == 0 else f"{label}: {100.0 * h / s:.0f}%"
      )
    return (
      "<div style='padding:4px 8px'>"
      "<table style='width:100%;font-size:0.85em;border-collapse:collapse'>"
      + "".join(rows)
      + "</table>"
      "<div style='font-size:0.8em;margin-top:6px'>"
      f"<b>Episode contact quality: {pct:.1f}%</b> "
      f"({hit_n}/{sched_n} scheduled frames)<br>"
      + " · ".join(pair_bits)
      + "</div>"
      "<div style='font-size:0.75em;opacity:0.7;margin-top:4px'>"
      "World-frame net force (N) · schedule hit = sensor found while scheduled"
      "</div></div>"
    )

  def _depth_map_rgb():
    env = viewer.env.unwrapped
    env_idx = int(viewer._scene.env_idx)
    sensor = env.scene["g1_depth_hybrid_camera"]
    distances = sensor.data.distances[env_idx]
    depth = distances.clamp(0.0, 1.0)
    depth = torch.where(distances < 0.0, torch.ones_like(depth), depth)
    image = depth.reshape(27, 48)
    image = image.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)
    grayscale = (image * 255.0).to(torch.uint8).cpu().numpy()
    return grayscale[..., None].repeat(3, axis=-1)

  orig_setup = viewer.setup

  def setup_with_viz() -> None:
    orig_setup()
    if backend == "viser":
      scene = viewer._scene
      if cfg.no_debug_vis:
        scene.debug_visualization_enabled = False
        print("[INFO]: Viser debug viz / reference ghost disabled")

      if want_collision_ui:
        viewer._play_collision_only = bool(cfg.collision_only)
        with viewer._server.gui.add_folder("Display"):
          status = viewer._server.gui.add_markdown(
            "**View:** collision-only" if cfg.collision_only else "**View:** mesh"
          )
          collision_btn = viewer._server.gui.add_button(
            "Collision only (G3+G4)",
            hint=(
              "Hide visual meshes; show collision geoms only "
              "(default collision class + RubberBox hand/foot reward pads)."
            ),
          )
          mesh_btn = viewer._server.gui.add_button(
            "Mesh view",
            hint="Show visual meshes; hide collision geoms.",
          )
          both_btn = viewer._server.gui.add_button(
            "Mesh + collision",
            hint="Show visual meshes and collision geoms together.",
          )

          def _set_status(label: str) -> None:
            status.content = f"**View:** {label}"

          @collision_btn.on_click
          def _(_) -> None:
            viewer._play_collision_only = True
            _apply_viser_geom_preset(scene, collision_only=True)
            _set_status("collision-only")

          @mesh_btn.on_click
          def _(_) -> None:
            viewer._play_collision_only = False
            _apply_viser_geom_preset(scene, collision_only=False)
            _set_status("mesh")

          @both_btn.on_click
          def _(_) -> None:
            viewer._play_collision_only = False
            scene.geom_groups_visible = [True, True, True, True, True, True]
            scene._sync_visibilities()
            scene._request_update()
            _set_status("mesh + collision")

        if cfg.collision_only:
          _apply_viser_geom_preset(scene, collision_only=True)
          print("[INFO]: Viser collision-only mode (Display tab to toggle)")
        elif cfg.show_collision:
          scene.geom_groups_visible = [True, True, True, True, True, True]
          scene._sync_visibilities()
          scene._request_update()
          print("[INFO]: Viser mesh + collision (Display tab to toggle)")
      if cfg.contact_force_panel:
        with viewer._server.gui.add_folder("Live contact forces (N)"):
          viewer._contact_force_html = viewer._server.gui.add_html(
            _contact_force_html()
          )
        print("[INFO]: Viser live 5-point contact-force panel enabled")
      if cfg.depth_map_panel:
        with viewer._server.gui.add_folder("Policy depth input"):
          viewer._depth_map_image = viewer._server.gui.add_image(
            _depth_map_rgb(), label="48x27 normalized depth (4x display)"
          )
          viewer._server.gui.add_markdown(
            "`black = 0 m (near)` · `white = 1 m or ray miss`"
          )
        print("[INFO]: Viser live policy depth-map panel enabled")
      return

    if cfg.no_debug_vis:
      viewer._show_debug_vis = False
      print("[INFO]: Native debug viz / reference ghost disabled")
    v = getattr(viewer, "viewer", None)
    if v is not None and want_collision_ui:
      if cfg.collision_only:
        v.opt.geomgroup[:] = [0, 0, 0, 1, 1, 0]
        print("[INFO]: Native collision-only mode (G3+G4)")
      elif cfg.show_collision:
        v.opt.geomgroup[3] = 1
        v.opt.geomgroup[4] = 1
        print(
          "[INFO]: Collision geoms (group 3+4: default collision + "
          "RubberBox hand/foot reward pads) enabled in native viewer"
        )

  viewer.setup = setup_with_viz  # type: ignore[method-assign]

  if want_contact_log:
    orig_execute = viewer._execute_step

    def execute_with_contact_log() -> bool:
      env = viewer.env.unwrapped
      i = _env_idx()
      ok = orig_execute()
      if not ok:
        return False
      done = bool(
        env.reset_terminated[i].item() or env.reset_time_outs[i].item()
      )
      if done:
        _flush_rollout("done")
      _record_contact_step()
      return True

    viewer._execute_step = execute_with_contact_log  # type: ignore[method-assign]
    orig_reset = viewer.reset_environment

    def reset_with_contact_log() -> None:
      _flush_rollout("manual_reset")
      orig_reset()

    viewer.reset_environment = reset_with_contact_log  # type: ignore[method-assign]
    print(
      "[INFO]: Per-rollout contact quality + XYZ force logging enabled"
      + (f" -> {cfg.contact_log_dir}" if cfg.contact_log_dir else "")
    )

  if backend == "viser" and (cfg.contact_force_panel or cfg.depth_map_panel):
    orig_sync = viewer.sync_env_to_viewer

    def sync_with_live_panels() -> None:
      orig_sync()
      # Base viewer runs at 60 Hz; 10 Hz is responsive without flooding the
      # Viser websocket or perturbing policy rollout timing.
      if viewer._counter % 6 == 0:
        if hasattr(viewer, "_contact_force_html"):
          viewer._contact_force_html.content = _contact_force_html()
        if hasattr(viewer, "_depth_map_image"):
          viewer._depth_map_image.image = _depth_map_rgb()

    viewer.sync_env_to_viewer = sync_with_live_panels  # type: ignore[method-assign]


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
