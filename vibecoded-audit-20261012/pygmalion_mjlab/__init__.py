"""Pygmalion rough-terrain velocity task for stock mjlab.

Importing this package registers:
  Pygmalion-Velocity-Rough-Box   training task (box-only rough terrain, 4096 envs default)
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import pygmalion_rough_env_cfg
from .rl_cfg import pygmalion_ppo_runner_cfg

TASK_ID = "Pygmalion-Velocity-Rough-Box"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=pygmalion_rough_env_cfg(),
  play_env_cfg=pygmalion_rough_env_cfg(play=True),
  rl_cfg=pygmalion_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
