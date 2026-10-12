"""Launch `mjlab.scripts.train` but force checkpoint loading to actor-only.

Why this exists: the Rough task's critic observes `height_scan` (187 rays) on
top of the Flat task's 60-dim critic obs, so a normal `runner.load(...)` (which
loads actor+critic+optimizer+iteration with strict=True) hard-crashes on the
critic's first-layer shape mismatch before training even starts. The actor obs
is byte-identical between Flat and Rough (6 proprioceptive terms, 45-dim), so
the actor transfers cleanly -- only the critic needs to start fresh. rsl_rl's
`MjlabOnPolicyRunner.load()` already supports per-component `load_cfg`, it's
just not exposed as a CLI flag, so this monkeypatches the one `runner.load()`
call inside `run_train()` to force `load_cfg={"actor": True, "critic": False,
"optimizer": False, "iteration": False, "rnd": True}` and otherwise defers to
the normal CLI (task id, --agent.resume/--agent.load-run/--agent.load-checkpoint
still pick which checkpoint to load; --env.* / --agent.* flags all still work).

Usage: identical to `uv run train`, e.g.
  uv run python analysis/train_actor_warmstart.py Mjlab-Velocity-Rough-Pygmalion \
    --env.scene.num-envs 4096 --agent.max-iterations 15000 --agent.run-name rough_v1 \
    --agent.resume True --agent.load-run 2026-07-07_05-16-24 \
    --agent.load-checkpoint model_19998.pt --agent.logger wandb
"""

import mjlab.rl.runner as _runner_mod

_orig_load = _runner_mod.MjlabOnPolicyRunner.load
_ACTOR_ONLY = {
  "actor": True,
  "critic": False,
  "optimizer": False,
  "iteration": False,
  "rnd": True,
}


def _actor_only_load(self, path, load_cfg=None, strict=True, map_location=None):
  print(f"[warmstart] actor-only load (critic/optimizer/iteration stay fresh): {path}")
  ret = _orig_load(
    self, path, load_cfg=_ACTOR_ONLY, strict=strict, map_location=map_location
  )
  # MjlabOnPolicyRunner.load restores common_step_counter UNCONDITIONALLY, which
  # fast-forwards step-keyed curricula (command_vel stages, dr_levels ramp) to max
  # from iter 0. Observed 2026-07-09: warm-start with full commands+DR from step 0
  # caused axis churn (child at 9k: flat vx 26% vs parent 81%). PYG_FRESH_STEPS=1
  # resets the counter so curricula ramp from zero like a fresh phase-1 run.
  import os

  if os.environ.get("PYG_FRESH_STEPS"):
    self.env.unwrapped.common_step_counter = 0
    print(
      "[warmstart] PYG_FRESH_STEPS=1 -> common_step_counter reset to 0 (curricula ramp fresh)"
    )
  # PYG_CRITIC_WARMUP_ITERS=N (2026-10-11, docs/162 step 4): the critic starts from scratch,
  # so for the first N updates its advantages are noise. Hold the loaded actor (weights, action
  # std, observation normalizer) fixed while the critic learns: run the normal PPO update, then
  # put the actor's state back. After N updates the actor trains as usual.
  warm = int(os.environ.get("PYG_CRITIC_WARMUP_ITERS", "0") or 0)
  if warm > 0:
    import copy

    alg = self.alg
    orig_update = alg.update
    count = {"n": 0}

    def _warmup_update(*args, **kwargs):
      frozen = count["n"] < warm
      saved = copy.deepcopy(alg.actor.state_dict()) if frozen else None
      out = orig_update(*args, **kwargs)
      if frozen:
        import torch

        # the observation normalizer's buffers are inference tensors (updated under
        # inference_mode), so copy each tensor in place in the mode it was made in
        cur = dict(alg.actor.named_parameters())
        cur.update(dict(alg.actor.named_buffers()))
        for k, v in saved.items():
          t = cur[k]
          with torch.inference_mode() if t.is_inference() else torch.no_grad():
            t.copy_(v)
      count["n"] += 1
      if count["n"] in (1, warm, warm + 1):
        print(f"[warmstart] critic warm-up: update {count['n']} actor {'held' if frozen else 'training'}",
              flush=True)
      return out

    alg.update = _warmup_update
    print(f"[warmstart] PYG_CRITIC_WARMUP_ITERS={warm}: actor held fixed for the first {warm} updates")
  return ret


_runner_mod.MjlabOnPolicyRunner.load = _actor_only_load

from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
