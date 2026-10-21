#!/usr/bin/env python3
"""Interactive viser viewer for a Huphy-Squat-Flat-v0 checkpoint.

  ~/huphy/mjlab/.venv/bin/python scripts/view.py logs/rsl_rl/huphy_squat/<run>/model_4000.pt
      [--num-envs 1] [--port 8080] [--box-edge 0.2] [--box-mass 3.0]

Open http://<host>:<port> in a browser (from a laptop over SSH: `ssh -L 8080:localhost:8080 <host>`).
Tabs added to the stock mjlab viewer:
  Squat    h* (Manual slider / Train-style random / Sweep), live monitor
  DR       every training DR as one slider; Random (train ranges) / Clean / Apply & Reset
  Disturb  pelvis push button, train-style random pushes, throw a box at the pelvis
The robot spawns with clean DR (nominal model); slider changes apply at the next reset.
Observations keep the training noise. Other checkpoints of the same run are in the
Checkpoints tab.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("checkpoint", type=Path, help="model_<iter>.pt")
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--box-edge", type=float, default=0.2, help="edge length of the thrown box [m]")
  ap.add_argument("--box-mass", type=float, default=3.0, help="initial box mass [kg] (GUI can change it)")
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = ap.parse_args()
  ckpt = args.checkpoint.resolve()
  if not ckpt.is_file():
    raise SystemExit(f"checkpoint not found: {ckpt}")

  import viser
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

  from huphy_squat import TASK_ID
  from huphy_squat.viewer.app import HuphyViewer
  from huphy_squat.viewer.scene import view_env_cfg

  env = ManagerBasedRlEnv(cfg=view_env_cfg(args.num_envs, args.box_edge, args.box_mass), device=args.device)
  rl = load_rl_cfg(TASK_ID)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl), device=args.device)

  def load(path: Path):
    runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=args.device)
    return runner.get_inference_policy(device=args.device)

  policy = load(ckpt)
  run_dir = ckpt.parent

  def available() -> list[tuple[str, str]]:
    now = time.time()
    files = sorted(run_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]) if f.stem.split("_")[1].isdigit() else 0)
    return [(f.name, format_time_ago(int(now - f.stat().st_mtime))) for f in files]

  mgr = CheckpointManager(current_name=ckpt.name, fetch_available=available,
                          load_checkpoint=lambda name: load(run_dir / name), run_name=run_dir.name)
  server = viser.ViserServer(host=args.host, port=args.port, label="huphy squat")
  print(f"[INFO] {ckpt}\n[INFO] viewer: http://localhost:{args.port}")
  HuphyViewer(wrapped, policy, box_edge=args.box_edge, viser_server=server, checkpoint_manager=mgr).run()
  env.close()


if __name__ == "__main__":
  main()
