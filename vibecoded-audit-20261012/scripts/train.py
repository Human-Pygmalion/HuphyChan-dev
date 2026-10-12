#!/usr/bin/env python3
"""Train Pygmalion-Velocity-Rough-Box with stock mjlab (`mjlab.scripts.train`).

Extra options (removed from argv before mjlab's own CLI parses the rest):

  --warmstart-actor PATH   load ONLY the actor (weights, action std, obs normalizer) from a
                           checkpoint before learning. The critic, optimizer and iteration
                           counter start new, and the env step counter is set to 0 so the
                           DR ramp (iterations 2,000 -> 8,000) starts from zero.
  --critic-warmup N        for the first N PPO updates keep the actor fixed (its state is
                           restored after each update) so only the new critic learns.

The released policy was trained as:
  python scripts/train.py Pygmalion-Velocity-Rough-Box \\
      --warmstart-actor checkpoints/parent_flat_actor_model_29998.pt --critic-warmup 300 \\
      --env.scene.num-envs 4096 --agent.max-iterations 12000 --agent.seed 43

Everything after the task id is passed to mjlab (see `python scripts/train.py --help`).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _pop(flag: str):
  if flag not in sys.argv:
    return None
  i = sys.argv.index(flag)
  val = sys.argv[i + 1]
  del sys.argv[i : i + 2]
  return val


def _install_warmstart(actor_ckpt: str | None, critic_warmup: int) -> None:
  from mjlab.rl.runner import MjlabOnPolicyRunner

  orig_learn = MjlabOnPolicyRunner.learn

  def learn(self, *args, **kwargs):
    if actor_ckpt:
      self.load(
        actor_ckpt,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
        strict=True,
      )
      self.env.unwrapped.common_step_counter = 0
      print(f"[pygmalion] actor-only warm start from {actor_ckpt}; env step counter -> 0", flush=True)
    if critic_warmup > 0:
      import torch

      alg = self.alg
      orig_update = alg.update
      count = {"n": 0}

      def update(*a, **kw):
        frozen = count["n"] < critic_warmup
        saved = copy.deepcopy(alg.actor.state_dict()) if frozen else None
        out = orig_update(*a, **kw)
        if frozen:
          cur = dict(alg.actor.named_parameters())
          cur.update(dict(alg.actor.named_buffers()))
          for k, v in saved.items():
            t = cur[k]
            with torch.inference_mode() if t.is_inference() else torch.no_grad():
              t.copy_(v)
        count["n"] += 1
        if count["n"] in (1, critic_warmup + 1):
          state = "held" if frozen else "training"
          print(f"[pygmalion] critic warm-up: update {count['n']}, actor {state}", flush=True)
        return out

      alg.update = update
    return orig_learn(self, *args, **kwargs)

  MjlabOnPolicyRunner.learn = learn


def main() -> None:
  actor_ckpt = _pop("--warmstart-actor")
  critic_warmup = int(_pop("--critic-warmup") or 0)
  import pygmalion_mjlab  # noqa: F401  (registers the task)

  if actor_ckpt or critic_warmup:
    _install_warmstart(actor_ckpt, critic_warmup)
  from mjlab.scripts.train import main as mjlab_train

  mjlab_train()


if __name__ == "__main__":
  main()
