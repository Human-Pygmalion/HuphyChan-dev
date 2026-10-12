"""Pygmalion velocity tasks for stock mjlab - the three training stages of the released policy.

Importing this package registers:
  Pygmalion-Velocity-Flat-P1     flat stage 1: from scratch, gated command curriculum, no DR
  Pygmalion-Velocity-Flat-P2     flat stage 2: resumed from P1 model_15999, DR + entropy anneal
  Pygmalion-Velocity-Rough-Box   rough stage: actor-only warm start from P2 model_29998
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import pygmalion_flat_env_cfg, pygmalion_rough_env_cfg
from .rl_cfg import pygmalion_ppo_runner_cfg

TASK_ID = "Pygmalion-Velocity-Rough-Box"
TASK_FLAT_P1 = "Pygmalion-Velocity-Flat-P1"
TASK_FLAT_P2 = "Pygmalion-Velocity-Flat-P2"
# task id -> stage name used in reference/ and rl_cfg
STAGES = {TASK_FLAT_P1: "flat_p1", TASK_FLAT_P2: "flat_p2", TASK_ID: "rough_main"}

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=pygmalion_rough_env_cfg(),
  play_env_cfg=pygmalion_rough_env_cfg(play=True),
  rl_cfg=pygmalion_ppo_runner_cfg("rough_main"),
  runner_cls=VelocityOnPolicyRunner,
)
for _task, _stage in ((TASK_FLAT_P1, "p1"), (TASK_FLAT_P2, "p2")):
  register_mjlab_task(
    task_id=_task,
    env_cfg=pygmalion_flat_env_cfg(_stage),
    play_env_cfg=pygmalion_flat_env_cfg(_stage, play=True),
    rl_cfg=pygmalion_ppo_runner_cfg(f"flat_{_stage}"),
    runner_cls=VelocityOnPolicyRunner,
  )
