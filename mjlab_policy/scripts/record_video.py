#!/usr/bin/env python3
"""Record a real-time close-up video of one robot following a fixed command schedule (headless).

  python scripts/record_video.py --checkpoint checkpoints/rough_main_model_11999.pt --out demo.mp4

Uses the task's play config (random terrain tile, no pushes, no observation noise) with one env,
on the play terrain grid (5 x 5 tiles of 8 m) with the flat border widened to 30 m. The env is
reset to a new random tile at the start of each command segment, so one segment
(<= 1.5 m/s x 15 s = 22.5 m from a tile centre at most 16 m off the map centre) stays inside the
50 m map edge; the play config has no out-of-bounds termination. (The training-size grid has
more geoms than the stock offscreen renderer's scene buffer holds.)
Each command is held for --dwell seconds. The video runs at 1/step_dt frames per second
(50 fps), i.e. real time. The overlay shows tag, time and the command.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")

SCHEDULE = (  # (label, vx, vy, wz)
  ("STAND", 0.0, 0.0, 0.0),
  ("FORWARD 0.5", 0.5, 0.0, 0.0),
  ("FORWARD 1.0", 1.0, 0.0, 0.0),
  ("FORWARD 1.5", 1.5, 0.0, 0.0),
  ("BACKWARD 0.5", -0.5, 0.0, 0.0),
  ("LATERAL LEFT 0.5", 0.0, 0.5, 0.0),
  ("TURN LEFT 0.8", 0.0, 0.0, 0.8),
  ("FORWARD 0.8 + TURN 0.5", 0.8, 0.0, 0.5),
)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--tag", default="")
  ap.add_argument("--dwell", type=float, default=15.0)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()

  import mediapy
  import numpy as np
  import torch
  from PIL import Image, ImageDraw

  import pygmalion_mjlab
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(pygmalion_mjlab.TASK_ID, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = a.seed
  cfg.viewer.height, cfg.viewer.width = 480, 720
  cfg.viewer.max_extra_envs = 0
  tg = cfg.scene.terrain.terrain_generator
  tg.border_width = 30.0
  tw = cfg.commands["twist"]
  tw.heading_command = False
  tw.ranges.heading = None
  tw.rel_standing_envs = tw.rel_world_envs = tw.rel_forward_envs = 0.0
  tw.init_velocity_prob = 0.0
  tw.resampling_time_range = (1e6, 1e6)
  agent_cfg = load_rl_cfg(pygmalion_mjlab.TASK_ID)

  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode="rgb_array")
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(pygmalion_mjlab.TASK_ID)(wenv, asdict(agent_cfg), device=a.device)
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
  policy = runner.get_inference_policy(device=a.device)
  term = env.command_manager.get_term("twist")

  fps = round(1.0 / env.step_dt)
  steps = int(round(a.dwell * fps))
  tag = a.tag or Path(a.checkpoint).stem
  frames = []
  obs = wenv.get_observations()
  falls = 0
  resets = []
  for label, vx, vy, wz in SCHEDULE:
    cmd = torch.tensor([vx, vy, wz], device=env.device)
    with torch.inference_mode():
      wenv.reset()  # new random tile for every segment (not counted as a fall)
      obs = wenv.get_observations()
    for k in range(steps):
      term.vel_command_b[:] = cmd
      with torch.inference_mode():
        obs, _, dones, _ = wenv.step(policy(obs))
      if dones.any():
        tm = env.termination_manager
        why = [n for n in tm.active_terms if tm.get_term(n).any()]
        falls += 1
        resets.append((round(len(frames) / fps, 2), label, why))
        print(f"reset at t={len(frames) / fps:.2f}s ({label}): {why}", flush=True)
      img = Image.fromarray(env.render())
      d = ImageDraw.Draw(img)
      t = len(frames) / fps
      d.rectangle((0, 0, img.width, 44), fill=(0, 0, 0))
      d.text((8, 4), f"{tag}   t={t:6.2f}s   cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}", fill=(255, 255, 255))
      d.text((8, 24), f"{label}   (real time, {fps} fps)   terminations so far: {falls}", fill=(255, 220, 0))
      frames.append(np.asarray(img))
  mediapy.write_video(a.out, frames, fps=fps)
  print(f"wrote {a.out}: {len(frames)} frames @ {fps} fps = {len(frames)/fps:.1f} s, terminations {falls}")
  env.close()


if __name__ == "__main__":
  main()
