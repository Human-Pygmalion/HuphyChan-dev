"""Reward terms added for Pygmalion (not in stock mjlab). Copied verbatim from the training fork."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def body_orientation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Unitree-style squared roll/pitch orientation cost."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    projected_gravity = quat_apply_inverse(quat_w, asset.data.gravity_vec_w)
  else:
    projected_gravity = asset.data.projected_gravity_b
  return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Unitree-style default-pose cost, active only for a near-zero command."""
  asset: Entity = env.scene[asset_cfg.name]
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[
    :, asset_cfg.joint_ids
  ]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  stopped = (torch.norm(command[:, :2], dim=1) + command[:, 2].abs()) <= command_threshold
  return torch.sum(torch.square(error), dim=1) * stopped.float()


def course_forward_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_heading: float = 0.0,
  heading_gain: float = 2.0,
  reverse_scale: float = 0.5,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward normalized progress along a fixed world-frame course direction.

  The stairs/slopes course runs along world +X.  Progress is normalized by the
  commanded forward speed and capped to [-1, 1].  Forward progress receives
  its signed normalized value, while reverse progress is scaled by
  ``reverse_scale`` and remains negative.  A smooth heading multiplier reduces
  the magnitude as the base yaws away from the course direction.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  course_direction = torch.tensor(
    [math.cos(target_heading), math.sin(target_heading)],
    device=env.device,
    dtype=asset.data.root_link_lin_vel_w.dtype,
  )
  projected_speed = torch.sum(
    asset.data.root_link_lin_vel_w[:, :2] * course_direction, dim=1
  )
  commanded_speed = command[:, 0].abs()
  progress = torch.clamp(
    projected_speed / commanded_speed.clamp(min=command_threshold), min=-1.0, max=1.0
  )
  progress = torch.where(
    commanded_speed > command_threshold, progress, torch.zeros_like(progress)
  )
  signed_progress = torch.where(progress >= 0.0, progress, reverse_scale * progress)

  heading_error = wrap_to_pi(asset.data.heading_w - target_heading)
  heading_smooth = torch.exp(-heading_gain * torch.square(heading_error))
  return signed_progress * heading_smooth


def actuator_clipping_fraction(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Mean fraction of requested PD effort removed by the torque-speed clamp."""
  asset: Entity = env.scene[asset_cfg.name]
  fractions = []
  for actuator in asset.actuators[asset_cfg.actuator_ids]:
    requested = getattr(actuator, "unclipped_effort", None)
    applied = getattr(actuator, "applied_effort", None)
    if requested is not None and applied is not None:
      fractions.append((requested - applied).abs() / requested.abs().clamp(min=1.0e-6))
  if not fractions:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.cat(fractions, dim=1).mean(dim=1)


def negative_mechanical_power(
  env: ManagerBasedRlEnv,
  free_fraction: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint-agnostic braking-power cost normalized by each motor's T-N envelope."""
  asset: Entity = env.scene[asset_cfg.name]
  costs = []
  for actuator in asset.actuators[asset_cfg.actuator_ids]:
    effort = getattr(actuator, "applied_effort", None)
    velocity = getattr(actuator, "_vel", None)
    if effort is None or velocity is None or not hasattr(actuator, "w_tab"):
      continue
    max_power = torch.max(actuator.w_tab * actuator.t_tab).clamp(min=1.0)
    braking_ratio = (-effort * velocity / max_power).clamp(min=0.0)
    costs.append(torch.square((braking_ratio - free_fraction).clamp(min=0.0)))
  if not costs:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.cat(costs, dim=1).mean(dim=1)


def track_lin_vel_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Linear PROGRESS reward: base velocity projected onto the commanded direction,
  saturated at the command magnitude (m/s).

  Unlike the exp tracking reward (whose gradient vanishes once the speed error
  exceeds ~1 m/s), this gives a non-vanishing gradient toward "move faster in the
  commanded direction" for any reachable speed below the command, pulling the
  policy to accelerate to its physical limit under high-speed commands instead of
  freezing (2026-07-10: exp-only std=0.71 froze the robot at cmd>=2.0 despite
  50%/30% torque/speed headroom). Capped at the command magnitude (no overshoot
  reward); clamped >=0 (no wrong-way reward); zeroed for near-zero commands.
  Research: docs/reward_research/2026-07-10_highspeed_freeze_progress_reward.md
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_xy = command[:, :2]
  actual_xy = asset.data.root_link_lin_vel_b[:, :2]
  cmd_mag = torch.norm(cmd_xy, dim=1)
  eps = 0.1
  proj = torch.sum(actual_xy * cmd_xy, dim=1) / (cmd_mag + eps)
  reward = torch.clamp(torch.minimum(proj, cmd_mag), min=0.0)
  return torch.where(cmd_mag > eps, reward, torch.zeros_like(reward))


def stand_still_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  cmd_deadband: float = 0.3,
  rel_floor: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Command-gated standing penalty: 1 while a nonzero command is being ignored.

  Fixes the mid-speed STALL (2026-07-11: at cmd 0.7 the policy walked then
  STOPPED for the last 6 s of a 15 s block; stop-and-go at 1.6; persisted
  through DR/push training so it is a reward-structure problem). Standing under
  a nonzero command earned residual exp-tracking reward while avoiding
  effort/impact penalties, making a stop a low-cost local optimum. This term
  makes "standing while commanded to move" pay a flat cost, without touching
  the exp std (part of the freeze fix) or rewarding overshoot.
  Research: docs/reward_research/2026-07-11_midspeed_stall_overshoot.md +
  2026-07-12_gen2_bundle.md.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_mag = torch.norm(command[:, :2], dim=1)
  wz_mag = torch.abs(command[:, 2])
  moving_cmd = (cmd_mag > cmd_deadband) | (wz_mag > cmd_deadband)
  # Gen-2.1 (2026-07-13): RELATIVE tracking floor. The absolute vel_thresh
  # (0.15) was gamed by creeping at ~0.6 m/s under cmd 1.5-2.5 (gen2 P2 final
  # gate: 56-57% achievement, DR did not fix it). Penalize while the achieved
  # progress ALONG the command stays below rel_floor of the commanded
  # magnitude; yaw analog uses sign-matched rate. Flat cost & weight unchanged
  # (single-variable ablation). Research:
  # docs/reward_research/2026-07-13_stall_relative_threshold.md
  proj = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * command[:, :2], dim=1) / (
    cmd_mag + 1e-6
  )
  lin_stalled = proj < rel_floor * cmd_mag
  yaw_prog = torch.sign(command[:, 2]) * asset.data.root_link_ang_vel_b[:, 2]
  yaw_stalled = yaw_prog < rel_floor * wz_mag
  lin_active = cmd_mag > cmd_deadband
  wz_active = wz_mag > cmd_deadband
  stalled = (lin_active & lin_stalled) | (wz_active & yaw_stalled)
  return (moving_cmd & stalled).float()


def knee_overspeed(
  env: ManagerBasedRlEnv,
  speed_limit: float = 19.9,
  joint_name_expr: str = ".*_knee_joint",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic penalty on knee speed beyond the REAL motor no-load limit.

  Sim enforces no velocity limit (the declared spec value is not applied by the
  position actuator), so learned swing retraction reaches 26.7 rad/s while the
  real RS04 no-load speed is 19.9 rad/s (measured T-N curve) — an unrealizable
  gait. The demand is swing-inherent (persists under a 2.0 m/s command cap), so
  we shape the policy instead: penalize relu(|qdot| - limit)^2 on the knees.
  Actuator-model envelope enforcement (DC actuator) deferred to Gen-2.1.
  Research: docs/reward_research/2026-07-12_gen2_bundle.md +
  reference-robstride-motor-specs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ids, _ = asset.find_joints(joint_name_expr)
  qd = torch.abs(asset.data.joint_vel[:, ids])
  over = torch.clamp(qd - speed_limit, min=0.0)
  return torch.sum(over**2, dim=1)


def torque_limit(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  soft_factor: float = 0.7,
) -> torch.Tensor:
  """Penalize commanded actuator torque past ``soft_factor × effort_limit``.

  Mirror of ``joint_pos_limits``: continuous, zero inside the safe band, grows
  linearly with excess outside. Per-actuator excess summed over actuators;
  returns ``[num_envs]``. The penalty is on **commanded** torque
  ``F_cmd = kp · (ctrl − q) − kd · q̇``, computed before MuJoCo's force-clip,
  so the gradient stays alive at every demand level (post-clip
  ``actuator_force`` would have flat gradient past the limit).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # First-call cache of per-actuator constants (kp, kd, effort_max,
  # ctrl_ids, actuator->joint index). Reading from the simulator's default
  # model fields keeps reward and eval on the same numbers.
  cache = getattr(asset, "_torque_limit_cache", None)
  if cache is None:
    ctrl_ids = asset.indexing.ctrl_ids
    forcerange = env.sim.get_default_field("actuator_forcerange")[ctrl_ids]
    e_max = torch.maximum(forcerange[:, 0].abs(), forcerange[:, 1].abs())
    # Forcerange = 0 means "no limit declared"; map to large finite so
    # divisions don't produce inf and the actuator effectively contributes 0.
    e_max = torch.where(e_max > 0, e_max, torch.full_like(e_max, 1e9))

    gainprm = env.sim.get_default_field("actuator_gainprm")[ctrl_ids]
    biasprm = env.sim.get_default_field("actuator_biasprm")[ctrl_ids]
    # mjlab's <position> actuator uses gain = kp·ctrl, bias = -kp·q - kd·q̇.
    # F_applied = gain - (-bias) = kp·(ctrl − q) − kd·q̇.
    kp = gainprm[:, 0]
    kd = -biasprm[:, 2]

    name_lookup = {jn: i for i, jn in enumerate(asset.joint_names)}
    act_to_joint_idx = torch.tensor(
      [name_lookup.get(an, 0) for an in asset.actuator_names],
      dtype=torch.long,
      device=env.device,
    )
    cache = (ctrl_ids, e_max, kp, kd, act_to_joint_idx)
    asset._torque_limit_cache = cache

  ctrl_ids, e_max, kp, kd, act_to_joint_idx = cache
  ctrl = env.sim.data.ctrl[:, ctrl_ids]  # [B, A]
  q_act = asset.data.joint_pos[:, act_to_joint_idx]  # [B, A]
  qd_act = asset.data.joint_vel[:, act_to_joint_idx]  # [B, A]
  f_cmd = kp * (ctrl - q_act) - kd * qd_act  # [B, A]
  excess = (f_cmd.abs() / e_max - soft_factor).clamp(min=0.0)  # [B, A]
  return excess.sum(dim=-1)


def thermal_effort(
  env: ManagerBasedRlEnv,
  rated: dict[str, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize sum of (torque / rated)^2 over actuated joints (B2).

  Motor heating ~ tau^2, so this is the thermal duty normalized per joint by its
  CONTINUOUS rating. Squaring makes equalizing ratios optimal -> load migrates
  from over-rated joints (knee ~106% raw RMS) toward joints with headroom (hips)
  = the torque redistribution objective. rated: joint-name-regex -> N*m.
  docs/reward_research/2026-07-02_gait_research_q123.md (ALMI C8 direction).
  """
  import re

  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.qfrc_actuator  # [B, n_joints], joint-DOF order
  names = asset.joint_names
  key = "_thermal_rated_vec"
  vec = getattr(env, key, None)
  if vec is None or vec.shape[0] != tau.shape[1]:
    r = torch.ones(tau.shape[1], device=tau.device)
    for i, n in enumerate(names):
      for pat, v in rated.items():
        if re.fullmatch(pat, n):
          r[i] = v
          break
    vec = r
    setattr(env, key, vec)
  ratio2 = (tau / vec) ** 2
  cost = torch.sum(ratio2, dim=1)
  env.extras["log"]["Metrics/thermal_effort_mean"] = cost.mean()
  return cost


def foot_impact_velocity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg,
  height_threshold: float = 0.08,
  contact_threshold: float = 7.0,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  power: float = 2.0,
) -> torch.Tensor:
  """Penalize DOWNWARD foot speed while the foot is close to the ground and still in the
  air (soft landing, 2026-08-24): sum_feet relu(-v_z) * [height < h0 and F < F0].

  Dense, every-step gradient on the descent before touchdown - unlike soft_landing (one
  50 Hz sample at first contact) or contact_force_cap (only the force after impact).
  Measured on the iter-2500 policies the feet hit the floor at 1.3-1.5 m/s (humans 0.1-0.4)
  with loading rates ~150 BW/s (humans 10-20). docs/reward_research/2026-08-24_soft_landing_impact.md
  """
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  foot_height = height_sensor.data.heights  # [B, F] sole height above terrain
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  in_air = torch.norm(contact_sensor.data.force, dim=-1) < contact_threshold  # [B, F]
  v_down = torch.clamp(
    -asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2], min=0.0
  )  # [B, F]
  near = (foot_height < height_threshold) & in_air
  # SQUARED (2026-08-24 test): a linear relu(-v) summed over the steps spent in the band is
  # sum(v dt) = the band height - speed-invariant - and the policy answered by crossing the
  # band FASTER (touchdown 1.24 -> 2.42 m/s in 800 iters). v^2 makes sum(v^2 dt) = v_mean * dz.
  cost_per_foot = (v_down**power) * near.float()
  cost = torch.sum(cost_per_foot, dim=1)
  env.extras["log"]["Metrics/foot_impact_vel_mean"] = (
    v_down[near].mean() if near.any() else torch.zeros((), device=cost.device)
  )
  env.extras["log"]["Metrics/foot_impact_vel_max"] = (v_down * near.float()).max()
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      active = (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold
      ).float()
      cost = cost * active
  return cost


def stance_knee_extension(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  target_deg: float = 25.0,
  contact_threshold: float = 7.0,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalise knee flexion beyond ``target_deg`` WHILE THE FOOT IS LOADED.

  Why the knee and not the base height: the c3 policies walk at 0.87-0.88 m - 93 % of the
  straight-leg height - while holding the knee at -30 to -48 deg. Hip pitch compensates, so
  base height barely moves with knee angle and a height anchor is inert (measured: the
  deadband term contributed -0.0019 with the robot at 0.8824 inside a [0.84, 0.90] band).
  The knee has to be addressed directly. Precedent: arXiv:2505.20619 adds a "straight knee"
  term (weight 0.1) for exactly this failure - "motions often appear overly crouched or
  energetically suboptimal".

  Stance only (contact-gated), so swing flexion - which is already human-like at -58 to
  -65 deg against a human -60 - is untouched. Research:
  docs/reward_research/2026-08-26_human_landing_bundle.md
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  in_contact = (
    torch.norm(contact_sensor.data.force, dim=-1) > contact_threshold
  )  # [B, F]
  q = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids])  # [B, 2] knees
  excess = torch.clamp(q - math.radians(target_deg), min=0.0)
  cost = torch.sum((excess**2) * in_contact.float(), dim=1)
  env.extras["log"]["Metrics/stance_knee_deg"] = torch.rad2deg(
    (q * in_contact.float()).sum() / in_contact.float().sum().clamp(min=1.0)
  )
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      active = (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold
      ).float()
      cost = cost * active
  return cost


def contact_force_cap(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 600.0,
  clip: float = 400.0,
) -> torch.Tensor:
  """Penalize per-foot contact force above a threshold [N], clipped.

  Impact-cap term (B1): unlike soft_landing (first-contact-only), this fires on
  EVERY step a foot presses harder than `threshold`, bounding GRF peaks without
  touching normal support forces (standing ~0.5 BW/foot stays below threshold).
  Real-robot-validated shape: Humanoid-Gym -0.01*max(F-400,0) (clipped) and
  REEM-C -0.01*Fz above 1500 N; threshold here defaults to ~1.2x body weight.
  docs/reward_research/2026-07-02_gait_research_q123.md (F4/C11).
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  force_magnitude = torch.norm(sensor_data.force, dim=-1)  # [B, N]
  excess = torch.clamp(force_magnitude - threshold, min=0.0, max=clip)  # [B, N]
  cost = torch.sum(excess, dim=1)  # [B]
  env.extras["log"]["Metrics/contact_force_excess_mean"] = excess.mean()
  return cost
