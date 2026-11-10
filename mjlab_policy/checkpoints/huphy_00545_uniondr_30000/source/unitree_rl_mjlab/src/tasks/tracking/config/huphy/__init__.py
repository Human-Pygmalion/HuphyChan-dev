from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  huphy_flat_tracking_env_cfg,
  huphy_flat_tracking_union_dr_env_cfg,
)
from .rl_cfg import huphy_tracking_ppo_runner_cfg


register_mjlab_task(
  task_id="HUPHY-Tracking-ProjectedGravity",
  env_cfg=huphy_flat_tracking_env_cfg(),
  play_env_cfg=huphy_flat_tracking_env_cfg(play=True),
  rl_cfg=huphy_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="HUPHY-Tracking-ProjectedGravity-UnionDR",
  env_cfg=huphy_flat_tracking_union_dr_env_cfg(),
  play_env_cfg=huphy_flat_tracking_union_dr_env_cfg(play=True),
  rl_cfg=huphy_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
