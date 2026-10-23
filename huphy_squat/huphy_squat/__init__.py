"""HUPHY squat task for stock mjlab (self-contained: stock mjlab + rsl_rl only).

Importing this package registers
  Huphy-Squat-Flat-v0   in-place squat with target pelvis height, flat ground, full DR.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import huphy_squat_env_cfg
from .rl_cfg import huphy_squat_ppo_cfg

TASK_ID = "Huphy-Squat-Flat-v0"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=huphy_squat_env_cfg(),
  play_env_cfg=huphy_squat_env_cfg(play=True),
  rl_cfg=huphy_squat_ppo_cfg(),
)
