#!/usr/bin/env python3
"""Train Pygmalion-Velocity-Rough-Box with stock mjlab (`mjlab.scripts.train`).

Extra options (removed from argv before mjlab's own CLI parses the rest):

  --warmstart-actor PATH   load ONLY the actor (weights, action std, obs normalizer) from a
                           checkpoint before learning. The critic, optimizer and iteration
                           counter start new, and the env step counter is set to 0 so the
                           DR ramp (iterations 2,000 -> 8,000) starts from zero.
  --critic-warmup N        for the first N PPO updates keep the actor fixed (its state is
                           restored after each update) so only the new critic learns.
  --entropy-anneal S E C0 C1
                           set entropy_coef before every PPO update: C0 up to absolute
                           iteration S, linear to C1 at iteration E, C1 after (flat P2:
                           15999 29998 0.01 0.002). Same schedule as the training fork's runner.

The released policy was trained in three stages (scripts/train_from_scratch.sh runs all three):
  python scripts/train.py Pygmalion-Velocity-Flat-P1 --agent.seed 43              # 16,000 it
  python scripts/train.py Pygmalion-Velocity-Flat-P2 --agent.seed 43 \\
      --agent.resume True --agent.load-run <P1 run dir name> \\
      --agent.load-checkpoint model_15999.pt --entropy-anneal 15999 29998 0.01 0.002
  python scripts/train.py Pygmalion-Velocity-Rough-Box --agent.seed 43 \\
      --warmstart-actor <P2 run dir>/model_29998.pt --critic-warmup 300

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


def _pop_n(flag: str, n: int):
  if flag not in sys.argv:
    return None
  i = sys.argv.index(flag)
  vals = sys.argv[i + 1 : i + 1 + n]
  del sys.argv[i : i + 1 + n]
  return vals


def _install_entropy_anneal(start_it: int, end_it: int, c0: float, c1: float) -> None:
  from mjlab.rl.runner import MjlabOnPolicyRunner

  orig_learn = MjlabOnPolicyRunner.learn

  def learn(self, *args, **kwargs):
    alg = self.alg
    orig_update = alg.update
    cursor = {"it": self.current_learning_iteration}
    print(
      f"[pygmalion] entropy anneal {c0} -> {c1} over iterations {start_it}..{end_it} "
      f"(this run starts at {cursor['it']})",
      flush=True,
    )

    def update(*a, **kw):
      it = cursor["it"]
      if it <= start_it:
        coef = c0
      elif it >= end_it:
        coef = c1
      else:
        coef = c0 + (c1 - c0) * (it - start_it) / float(max(1, end_it - start_it))
      alg.entropy_coef = coef
      cursor["it"] = it + 1
      return orig_update(*a, **kw)

    alg.update = update
    return orig_learn(self, *args, **kwargs)

  MjlabOnPolicyRunner.learn = learn


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
  anneal = _pop_n("--entropy-anneal", 4)
  import pygmalion_mjlab  # noqa: F401  (registers the tasks)

  if actor_ckpt or critic_warmup:
    _install_warmstart(actor_ckpt, critic_warmup)
  if anneal:
    _install_entropy_anneal(int(anneal[0]), int(anneal[1]), float(anneal[2]), float(anneal[3]))
  from mjlab.scripts.train import main as mjlab_train

  mjlab_train()


if __name__ == "__main__":
  main()
