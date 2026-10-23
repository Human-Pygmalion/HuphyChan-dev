#!/usr/bin/env python3
"""Validate the Union-DR task and full-resume checkpoint without training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygmalion_mjlab  # noqa: E402,F401
from mjlab.tasks.registry import load_env_cfg  # noqa: E402


TASK = "Pygmalion-Velocity-Generalized-Terrain10-UnionDR"


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("checkpoints/terrain10_2048_model_37400.pt"),
  )
  args = parser.parse_args()

  cfg = load_env_cfg(TASK)
  events = cfg.events
  assert events["pd_gains"].params["kp_range"] == (0.8, 1.2)
  assert events["pd_gains"].params["kd_range"] == (0.8, 1.2)
  assert events["encoder_bias"].params["bias_range"] == (-0.015, 0.015)
  assert events["motor_strength"].params["strength_range"] == (0.8, 1.2)
  assert events["motor_armature"].params["ranges"] == (0.8, 1.2)
  assert events["passive_friction"].params["ranges"] == (0.0, 0.5)
  assert events["passive_damping"].params["ranges"] == (0.0, 0.05)
  assert events["payload"].params["mass_range"] == (-1.0, 8.0)
  assert events["torso_push"].params["magnitude_range"] == (0.0, 100.0)
  assert events["torso_push"].params["duration_s"] == (0.1, 0.2)
  assert events["torso_push"].params["cooldown_s"] == (2.0, 10.0)
  assert events["imu_mount"].params["roll_range"] == (
    -math.radians(2.0), math.radians(2.0)
  )
  assert len([name for name in events if name.startswith("union_inertia_")]) == 8

  for name in ("base_ang_vel", "projected_gravity", "motor_pos_history"):
    term = cfg.observations["actor"].terms[name]
    assert (term.delay_min_lag, term.delay_max_lag) == (0, 2)
    assert term.delay_update_period == 1_000_000
    assert term.delay_per_env_phase is False
  for actuator in cfg.scene.entities["robot"].articulation.actuators:
    assert (actuator.delay_min_lag, actuator.delay_max_lag) == (0, 4)
    assert actuator.delay_update_period == 1_000_000
    assert actuator.delay_per_env_phase is False

  curriculum = cfg.curriculum["terrain_5k_unlock"].params
  assert curriculum == {
    "iterations_per_level": 2_000,
    "steps_per_iteration": 24,
    "num_levels": 10,
    "start_iteration": 19_700,
    "start_max_level": 3,
  }
  twist = cfg.commands["twist"]
  assert twist.ranges.lin_vel_x == (-1.0, 2.0)
  assert twist.ranges.lin_vel_y == (0.0, 0.0)
  assert twist.ranges.ang_vel_z == (-1.0, 1.0)

  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
  assert not (required - checkpoint.keys()), required - checkpoint.keys()
  assert checkpoint["iter"] == 37_400
  assert checkpoint["infos"]["env_state"]["common_step_counter"] == 897_648

  actor = checkpoint["actor_state_dict"]
  critic = checkpoint["critic_state_dict"]
  assert actor["mlp.0.weight"].shape[1] == 44
  assert actor["mlp.6.weight"].shape[0] == 12
  assert critic["mlp.0.weight"].shape[1] == 268

  print(f"PASS task: {TASK}")
  print("PASS checkpoint: iteration 37400, actor 44->12, critic input 268, optimizer present")
  print("PASS resume state: common_step_counter 897648 (Terrain10 level 0-9 remains unlocked)")
  print("PASS Union DR ranges and episode-fixed 0-20/0-40 ms delays")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
