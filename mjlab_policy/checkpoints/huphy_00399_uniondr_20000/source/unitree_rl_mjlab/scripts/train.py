"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Literal, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  resmimic_base_policy: str | None = None
  """If set, wrap env so policy actions are residuals added to frozen ResMimic base."""
  resmimic_residual_scale: float = 1.0

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _cpu_slice(tensor: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
  """Detach a small environment slice for a crash-safe diagnostic snapshot."""
  return tensor[env_ids].detach().cpu()


def _install_observation_nan_diagnostics(
  env: ManagerBasedRlEnv, output_dir: Path
) -> None:
  """Augment ObservationManager's strict finite check with simulator state.

  This hook runs only with ``--enable-nan-guard``.  The manager already knows
  which observation term failed; the hook adds exact channels, the relevant
  robot/bicycle state, recent actions, and a full torch snapshot before the
  manager raises its normal ValueError.
  """
  manager = env.observation_manager
  original_check = manager._check_and_handle_nans
  output_dir.mkdir(parents=True, exist_ok=True)
  manager._detailed_nan_dump_written = False

  def _diagnostic_check(self, tensor, context, policy):
    flat = tensor.reshape(tensor.shape[0], -1)
    invalid = ~torch.isfinite(flat)
    if invalid.any() and not self._detailed_nan_dump_written:
      self._detailed_nan_dump_written = True
      bad_env_ids = invalid.any(dim=-1).nonzero(as_tuple=True)[0]
      shown_env_ids = bad_env_ids[:10]
      bad_channels = invalid[bad_env_ids].nonzero(as_tuple=False)[:, 1].unique()
      nan_channels = torch.isnan(flat[bad_env_ids]).nonzero(as_tuple=False)[:, 1].unique()
      posinf_channels = torch.isposinf(flat[bad_env_ids]).nonzero(as_tuple=False)[:, 1].unique()
      neginf_channels = torch.isneginf(flat[bad_env_ids]).nonzero(as_tuple=False)[:, 1].unique()

      snapshot = {
        "context": context,
        "common_step_counter": int(env.common_step_counter),
        "bad_env_ids": bad_env_ids.detach().cpu(),
        "bad_channels": bad_channels.detach().cpu(),
        "observation": _cpu_slice(tensor, shown_env_ids),
        "episode_length": _cpu_slice(env.episode_length_buf, shown_env_ids),
        "actions": _cpu_slice(env.action_manager.action, shown_env_ids),
        "previous_actions": _cpu_slice(env.action_manager.prev_action, shown_env_ids),
      }
      for asset_name in ("robot", "bicycle"):
        try:
          asset = env.scene[asset_name]
        except (KeyError, TypeError):
          continue
        for field_name in (
          "joint_pos",
          "joint_vel",
          "root_link_pose_w",
          "root_link_vel_w",
        ):
          value = getattr(asset.data, field_name, None)
          if isinstance(value, torch.Tensor):
            snapshot[f"{asset_name}_{field_name}"] = _cpu_slice(value, shown_env_ids)

      stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
      dump_path = output_dir / f"observation_nan_{stamp}.pt"
      torch.save(snapshot, dump_path)

      print("\n[NaN diagnostics] non-finite observation detected", flush=True)
      print(f"  term: {context}", flush=True)
      print(f"  step: {env.common_step_counter}", flush=True)
      print(f"  env IDs ({bad_env_ids.numel()}): {bad_env_ids.detach().cpu().tolist()}", flush=True)
      print(f"  invalid channels: {bad_channels.detach().cpu().tolist()}", flush=True)
      print(f"  NaN channels: {nan_channels.detach().cpu().tolist()}", flush=True)
      print(f"  +Inf channels: {posinf_channels.detach().cpu().tolist()}", flush=True)
      print(f"  -Inf channels: {neginf_channels.detach().cpu().tolist()}", flush=True)
      offending_values = {
        int(env_id): flat[env_id, invalid[env_id]].detach().cpu().tolist()
        for env_id in shown_env_ids
      }
      print(f"  offending values by env: {offending_values}", flush=True)
      for key, value in snapshot.items():
        if key.startswith(("robot_", "bicycle_")) or key in (
          "actions",
          "previous_actions",
          "episode_length",
        ):
          print(f"  {key}: {value}", flush=True)
      print(f"  full snapshot: {dump_path}\n", flush=True)

    return original_check(tensor, context=context, policy=policy)

  manager._check_and_handle_nans = MethodType(_diagnostic_check, manager)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    for obs_group in cfg.env.observations.values():
      obs_group.nan_policy = "error"
      obs_group.nan_check_per_term = True
    os.environ["MJLAB_DUAL_LATENT_NAN_DIAGNOSTICS"] = "1"
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if cfg.enable_nan_guard:
    _install_observation_nan_diagnostics(env, Path(cfg.env.sim.nan_guard.output_dir))

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  if cfg.resmimic_base_policy:
    from src.tasks.tracking.rl import ResMimicResidualVecEnvWrapper

    env = ResMimicResidualVecEnvWrapper(
      env,
      base_policy_path=cfg.resmimic_base_policy,
      clip_actions=cfg.agent.clip_actions,
      residual_scale=cfg.resmimic_residual_scale,
    )
    print(
      f"[INFO] ResMimic residual wrapper enabled "
      f"(base={cfg.resmimic_base_policy}, residual_scale={cfg.resmimic_residual_scale})"
    )
  else:
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


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

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
