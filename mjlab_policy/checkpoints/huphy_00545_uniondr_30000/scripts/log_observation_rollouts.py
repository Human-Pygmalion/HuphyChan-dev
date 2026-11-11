#!/usr/bin/env python3
"""Record full-cycle HUPHY actor observations with Union DR on and off.

Run from the unitree_rl_mjlab checkout with that checkout on ``PYTHONPATH``.
Each rollout starts at reference frame zero and records the observation presented
to the actor before inference.  Terminations are disabled so every file contains
one complete 659-frame reference cycle; this choice does not change physics, DR,
observation corruption, or actuator delay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

import src.tasks  # noqa: F401 - register local tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from src.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


TASK = "HUPHY-Tracking-ProjectedGravity-UnionDR"
SLICES = {
  "reference_joint_pos": (0, 24),
  "reference_joint_vel": (24, 48),
  "projected_gravity": (48, 51),
  "base_ang_vel": (51, 54),
  "joint_pos_rel": (54, 78),
  "joint_vel_rel": (78, 102),
  "previous_action": (102, 114),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  package = Path(__file__).resolve().parents[1]
  parser.add_argument("--checkpoint", type=Path, default=package / "model_30000.pt")
  parser.add_argument(
    "--motion", type=Path, default=package / "motions/00545_interactionmesh_50hz.npz"
  )
  parser.add_argument("--output-dir", type=Path, default=package / "observation_rollouts")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seeds", type=int, nargs=2, default=(20261111, 20261112))
  return parser.parse_args()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_revision(path: Path) -> str | None:
  try:
    return subprocess.check_output(
      ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def disable_observation_corruption(env_cfg) -> None:
  actor = env_cfg.observations["actor"]
  actor.enable_corruption = False
  for term in actor.terms.values():
    term.noise = None
    term.delay_min_lag = 0
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0


def disable_actuator_delay(env_cfg) -> None:
  robot_cfg = env_cfg.scene.entities["robot"]
  for actuator in robot_cfg.articulation.actuators:
    actuator.delay_min_lag = 0
    actuator.delay_max_lag = 0
    actuator.delay_hold_prob = 0.0


@torch.no_grad()
def record_one(args: argparse.Namespace, *, dr_enabled: bool, seed: int, index: int) -> dict:
  torch.manual_seed(seed)
  np.random.seed(seed)

  # play=False is the exact training configuration: Union DR reset/step events,
  # observation noise/delay, and actuator delay are all retained.
  env_cfg = load_env_cfg(TASK, play=not dr_enabled)
  agent_cfg = load_rl_cfg(TASK)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = 1
  # A training episode times out at 10 s while this motion is 13.16 s.  Disable
  # termination only to capture one uninterrupted full reference cycle.
  env_cfg.terminations = {}

  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = str(args.motion.resolve())
  motion_cfg.sampling_mode = "start"
  motion_cfg.start_frame = 0
  if not dr_enabled:
    env_cfg.events = {}
    disable_observation_corruption(env_cfg)
    disable_actuator_delay(env_cfg)
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.joint_velocity_range = (0.0, 0.0)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint.resolve()),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  obs, _ = env.reset()
  command = raw_env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  command.time_steps.zero_()
  command._resample_command(torch.arange(raw_env.num_envs, device=raw_env.device))
  raw_env.sim.forward()
  obs = env.get_observations()

  source = np.load(args.motion, allow_pickle=True)
  frames = int(len(source["joint_pos"]))
  fps = float(np.asarray(source["fps"]).reshape(-1)[0])
  rows: dict[str, list[np.ndarray | float | int]] = {
    "actor_observation": [], "raw_action": [], "reward": [], "done": [],
    "reference_frame": [],
  }
  try:
    for _ in range(frames):
      actor_obs = obs["actor"][0].detach().cpu().numpy().astype(np.float32, copy=True)
      if actor_obs.shape != (114,):
        raise RuntimeError(f"actor observation shape {actor_obs.shape}, expected (114,)")
      rows["actor_observation"].append(actor_obs)
      rows["reference_frame"].append(int(command.time_steps[0].item()))
      action = policy(obs)
      rows["raw_action"].append(
        action[0].detach().cpu().numpy().astype(np.float32, copy=True)
      )
      obs, reward, done, _ = env.step(action)
      rows["reward"].append(float(reward[0].item()))
      rows["done"].append(int(done[0].item()))
  finally:
    env.close()

  actor = np.stack(rows["actor_observation"])
  if not np.array_equal(np.asarray(rows["reference_frame"]), np.arange(frames)):
    raise RuntimeError("reference frame sequence is not exactly 0..frames-1")
  if not np.allclose(actor[0, 102:114], 0.0, atol=0.0, rtol=0.0):
    raise RuntimeError("first-frame previous_action is not exactly zero")

  mode = "dr_on" if dr_enabled else "dr_off"
  filename = f"{mode}_rollout{index}_seed{seed}.npz"
  output = args.output_dir / filename
  payload = {
    "actor_observation": actor,
    "raw_action": np.stack(rows["raw_action"]),
    "reward": np.asarray(rows["reward"], dtype=np.float32),
    "done": np.asarray(rows["done"], dtype=np.int8),
    "reference_frame": np.asarray(rows["reference_frame"], dtype=np.int32),
    "fps": np.asarray([fps], dtype=np.float32),
    "seed": np.asarray([seed], dtype=np.int64),
    "dr_enabled": np.asarray([dr_enabled], dtype=np.bool_),
  }
  for name, (start, end) in SLICES.items():
    payload[name] = actor[:, start:end]
  np.savez_compressed(output, **payload)
  return {
    "file": filename,
    "sha256": sha256(output),
    "mode": mode,
    "seed": seed,
    "frames": frames,
    "fps": fps,
    "duration_s": (frames - 1) / fps,
    "observation_shape": list(actor.shape),
    "raw_action_shape": list(payload["raw_action"].shape),
    "first_previous_action_exact_zero": True,
    "done_count": int(payload["done"].sum()),
    "observation_abs_max": float(np.abs(actor).max()),
    "raw_action_abs_max": float(np.abs(payload["raw_action"]).max()),
    "reward_mean": float(payload["reward"].mean()),
  }


def main() -> None:
  args = parse_args()
  configure_torch_backends()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  records = []
  for dr_enabled in (True, False):
    for index, seed in enumerate(args.seeds, start=1):
      print(f"Recording dr_enabled={dr_enabled} rollout={index} seed={seed}", flush=True)
      records.append(record_one(args, dr_enabled=dr_enabled, seed=seed, index=index))

  manifest = {
    "schema_version": 1,
    "policy": "huphy_00545_uniondr_model_30000",
    "checkpoint": os.path.relpath(args.checkpoint.resolve(), args.output_dir.resolve()),
    "checkpoint_sha256": sha256(args.checkpoint),
    "motion": os.path.relpath(args.motion.resolve(), args.output_dir.resolve()),
    "motion_sha256": sha256(args.motion),
    "task": TASK,
    "observation_slices": {name: list(bounds) for name, bounds in SLICES.items()},
    "terminations_disabled_reason": (
      "The training timeout is 10 s but the reference is 13.16 s; disabled equally in all "
      "four runs to log one complete uninterrupted reference cycle."
    ),
    "dr_on": (
      "Training Union DR reset/step events, observation corruption/delay, and actuator delay on."
    ),
    "dr_off": (
      "All events, observation noise/delay, actuator delay, and motion reset perturbations off."
    ),
    "unitree_rl_mjlab_git": git_revision(Path.cwd()),
    "rollouts": records,
  }
  manifest_path = args.output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(f"Saved {len(records)} rollouts and {manifest_path}", flush=True)


if __name__ == "__main__":
  main()
