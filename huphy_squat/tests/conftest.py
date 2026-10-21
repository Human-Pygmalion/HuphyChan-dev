"""Shared fixtures. GPU tests build the real task env (needs CUDA)."""

from __future__ import annotations

import pytest
import torch

NUM_ENVS = 256


def make_env(num_envs: int = NUM_ENVS, play: bool = False, edit=None):
  import huphy_squat  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Huphy-Squat-Flat-v0", play=play)
  cfg.scene.num_envs = num_envs
  if edit is not None:
    edit(cfg)
  return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


@pytest.fixture(scope="session")
def env():
  if not torch.cuda.is_available():
    pytest.skip("needs CUDA")
  e = make_env()
  e.reset()
  return e


def sole_corners_z(env) -> torch.Tensor:
  """Bottom corners of all foot boxes, height above the env floor. Shape [B, 24]."""
  r = env.scene["robot"]
  d, m = env.sim.data, env.sim.model
  names = list(r.geom_names)
  gids = [int(r.indexing.geom_ids[names.index(f"{s}_foot{i}_collision")]) for s in "RL" for i in (1, 2, 3)]
  out = []
  for g in gids:
    c, R, h = d.geom_xpos[:, g], d.geom_xmat[:, g], m.geom_size[:, g] if m.geom_size.shape[0] > 1 else m.geom_size[0, g].expand(env.num_envs, 3)
    for sx in (-1, 1):
      for sy in (-1, 1):
        local = torch.stack((sx * h[:, 0], sy * h[:, 1], -h[:, 2]), dim=1)
        out.append(c[:, 2] + (R @ local[:, :, None])[:, 2, 0])
  return torch.stack(out, dim=1) - env.scene.env_origins[:, 2:3]


def loop_residual(env) -> torch.Tensor:
  """|rod end - foot ball| for the 4 loops [m]. Shape [B, 4]."""
  from huphy_squat.mdp.observations import loop_distances

  return loop_distances(env)
