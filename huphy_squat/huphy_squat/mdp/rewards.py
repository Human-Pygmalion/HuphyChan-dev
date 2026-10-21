"""Reward terms of the squat task (docs/task_design/reward_design.md, decisions.md).

Sign convention for weights: positive terms return values in [0, 1], penalties return
non-negative costs (the env cfg gives them negative weights). Every term is wrapped by
`clamped` in the env cfg (|w * f| <= 50 per step, NaN -> 0).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .. import robot
from .commands import base_height
from .events import get_motor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def clamped(func, weight: float, limit: float = 50.0):
  """Wrap a term so |weight * value| <= limit and non-finite values become 0."""
  bound = limit / abs(weight)

  def _c(v: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-bound, bound)

  if inspect.isclass(func):

    class Clamped:
      def __init__(self, cfg, env):
        self.inner = func(cfg=cfg, env=env)

      def __call__(self, env, **params):
        return _c(self.inner(env, **params))

      def reset(self, env_ids=None):
        if hasattr(self.inner, "reset"):
          self.inner.reset(env_ids)

    Clamped.__name__ = Clamped.__qualname__ = f"clamped_{func.__name__}"
    return Clamped

  def term(env, **params):
    return _c(func(env, **params))

  term.__name__ = term.__qualname__ = f"clamped_{func.__name__}"
  return term


def _log(env: ManagerBasedRlEnv, name: str, value: torch.Tensor) -> None:
  env.extras.setdefault("log", {})[f"Metrics/{name}"] = value.detach().float().mean()


class SquatReference:
  """q*(h) of the R leg (robot.SQUAT_REF_*), linear interpolation, h clipped to the table."""

  def __init__(self, device: str):
    self.h = torch.tensor(robot.SQUAT_REF_H, device=device)
    self.knee = torch.tensor(robot.SQUAT_REF_KNEE, device=device)
    self.hip = torch.tensor(robot.SQUAT_REF_HIP_PITCH, device=device)
    self.ankle = torch.tensor(robot.SQUAT_REF_ANKLE_PITCH, device=device)
    self.hip_yaw = torch.tensor(robot.SQUAT_REF_HIP_YAW, device=device)
    self.ankle_roll = torch.tensor(robot.SQUAT_REF_ANKLE_ROLL, device=device)
    self.width = torch.tensor(robot.SQUAT_REF_FOOT_WIDTH, device=device)

  def _lerp(self, h: torch.Tensor):
    hc = h.clamp(float(self.h[0]), float(self.h[-1])).contiguous()
    i = torch.searchsorted(self.h, hc).clamp(1, len(self.h) - 1)
    f = (hc - self.h[i - 1]) / (self.h[i] - self.h[i - 1])
    return lambda t: t[i - 1] + f * (t[i] - t[i - 1])

  def __call__(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(hip pitch, knee, ankle pitch) for the R leg at heights h [B]."""
    lerp = self._lerp(h)
    return lerp(self.hip), lerp(self.knee), lerp(self.ankle)

  def full(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
    """All six posture joints of the R leg + the foot width, at heights h [B]."""
    lerp = self._lerp(h)
    return {"hip_pitch": lerp(self.hip), "hip_roll": torch.full_like(h, robot.HIP_SPREAD),
            "hip_yaw": lerp(self.hip_yaw), "knee": lerp(self.knee), "ankle_pitch": lerp(self.ankle),
            "ankle_roll": lerp(self.ankle_roll), "foot_width": lerp(self.width)}


# L-leg sign of each posture joint relative to the R-leg reference (robot.py mirror convention).
MIRROR = {"hip_pitch": -1.0, "hip_roll": -1.0, "hip_yaw": -1.0, "knee": -1.0, "ankle_pitch": 1.0, "ankle_roll": 1.0}


# --- task ---------------------------------------------------------------------------------

def track_height(env: ManagerBasedRlEnv, std: float, command_name: str = "height") -> torch.Tensor:
  """exp(-(h - h*)^2 / std^2)."""
  err = base_height(env) - env.command_manager.get_command(command_name)[:, 0]
  _log(env, "height_abs_error", err.abs())
  return torch.exp(-err.square() / std**2)


def vz_speed(env: ManagerBasedRlEnv, threshold: float, grace_s: float = 0.2,
             asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """relu(|v_z| - threshold)^2, v_z = pelvis world vertical speed; 0 during the first
  `grace_s` of an episode. With the +-0.05 rad reset noise the soles start slightly tilted and
  the robot rocks onto them in ~0.1 s (|v_z| up to ~0.5 m/s even when the action holds the
  pose), which the policy cannot avoid (reward_design.md section 5, item 18)."""
  vz = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2]
  _log(env, "pelvis_abs_vz", vz.abs())
  active = (env.episode_length_buf.float() * env.step_dt) > grace_s
  return (vz.abs() - threshold).clamp(min=0).square() * active.float()


class knee_guidance:
  """sum_knees |s * q_knee - q*_knee(h*)| * min(|h - h*| / gate, 1)  (s = +1 R, -1 L).

  HOMIE-style knee term, but the knee reference is the squat-table knee angle at the target
  height (not the middle of the joint range), so it points the right way going up and down.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    a: Entity = env.scene["robot"]
    self.ids = torch.tensor([a.joint_names.index(f"{s}_knee_joint") for s in robot.SIDES], device=env.device)
    self.sign = torch.tensor([1.0 if s == "R" else -1.0 for s in robot.SIDES], device=env.device)
    self.ref = SquatReference(env.device)

  def __call__(self, env: ManagerBasedRlEnv, gate: float, command_name: str = "height") -> torch.Tensor:
    h_star = env.command_manager.get_command(command_name)[:, 0]
    _, knee_ref, _ = self.ref(h_star)
    q = env.scene["robot"].data.joint_pos[:, self.ids] * self.sign
    g = ((base_height(env) - h_star).abs() / gate).clamp(max=1.0)
    return (q - knee_ref[:, None]).abs().sum(dim=1) * g


class squat_posture:
  """exp(-mean_j (q_j - q*_j(h*))^2 / std_j^2) over hip pitch/roll/yaw, knee, ankle pitch/roll.

  Reference = the squat table at the commanded height (hips spread by robot.HIP_SPREAD, feet
  level and pointing ahead).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    a: Entity = env.scene["robot"]
    names = list(robot.POSTURE_JOINT_NAMES)
    self.ids = torch.tensor([a.joint_names.index(n) for n in names], device=env.device)
    _, _, std = resolve_matching_names_values(cfg.params["std"], names)
    self.std = torch.tensor(std, device=env.device)
    self.ref = SquatReference(env.device)
    self.keys = [n.split("_", 1)[1].removesuffix("_joint") for n in names]
    self.sign = torch.tensor([1.0 if n.startswith("R") else MIRROR[k] for n, k in zip(names, self.keys, strict=True)],
                             device=env.device)

  def reference(self, h_star: torch.Tensor) -> torch.Tensor:
    full = self.ref.full(h_star)
    return torch.stack([full[k] for k in self.keys], dim=1) * self.sign

  def __call__(self, env: ManagerBasedRlEnv, std, command_name: str = "height") -> torch.Tensor:
    del std
    q = env.scene["robot"].data.joint_pos[:, self.ids]
    ref = self.reference(env.command_manager.get_command(command_name)[:, 0])
    return torch.exp(-((q - ref) / self.std).square().mean(dim=1))


class stance_width:
  """Distance [m] of the foot width outside the band [w*(h*) - narrow, w*(h*) + wide].

  w = lateral distance of the two foot-link origins, measured along the pelvis heading's
  left axis (yaw only); w* = the reference width at the commanded height. Lets the feet sit
  anywhere inside the band, and keeps pushing them back out once they get too close (e.g.
  after a push slid them together), which needs a step.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    a: Entity = env.scene["robot"]
    self.feet = [a.body_names.index(b) for b in robot.FOOT_BODIES]  # R, L
    self.ref = SquatReference(env.device)

  def __call__(self, env: ManagerBasedRlEnv, narrow: float, wide: float, command_name: str = "height") -> torch.Tensor:
    a = env.scene["robot"]
    w = foot_width(a, self.feet)
    w_ref = self.ref.full(env.command_manager.get_command(command_name)[:, 0])["foot_width"]
    _log(env, "foot_width", w)
    return (w_ref - narrow - w).clamp(min=0) + (w - w_ref - wide).clamp(min=0)


def foot_width(asset, feet: list[int]) -> torch.Tensor:
  """Lateral distance of feet[1] (L) minus feet[0] (R) along the pelvis heading's left axis."""
  q = asset.data.root_link_quat_w
  w, x, y, z = q.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  left = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=1)
  p = asset.data.body_link_pos_w[:, feet, :2]
  return ((p[:, 1] - p[:, 0]) * left).sum(dim=1)


class pitch_symmetry:
  """exp(-(1/3) sum_j (q_L,j - s_j q_R,j)^2 / std^2) over hip pitch (s -1), knee (-1), ankle pitch (+1)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    a: Entity = env.scene["robot"]
    pairs = (("hip_pitch_joint", -1.0), ("knee_joint", -1.0), ("ankle_pitch_joint", 1.0))
    self.l = torch.tensor([a.joint_names.index(f"L_{j}") for j, _ in pairs], device=env.device)
    self.r = torch.tensor([a.joint_names.index(f"R_{j}") for j, _ in pairs], device=env.device)
    self.s = torch.tensor([s for _, s in pairs], device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos
    d = q[:, self.l] - self.s * q[:, self.r]
    return torch.exp(-d.square().mean(dim=1) / std**2)


# --- body ---------------------------------------------------------------------------------

def upright(env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """exp(-(g_x^2 + g_y^2) / std^2), g = gravity direction in the pelvis frame."""
  g = env.scene[asset_cfg.name].data.projected_gravity_b
  return torch.exp(-g[:, :2].square().sum(dim=1) / std**2)


def lin_vel_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2].square().sum(dim=1)


def ang_vel_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2].square()


def ang_vel_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, :2].square().sum(dim=1)


def angular_momentum(env: ManagerBasedRlEnv, sensor_name: str = "robot/root_angmom") -> torch.Tensor:
  return env.scene[sensor_name].data.square().sum(dim=-1)


# --- limits and linkage -------------------------------------------------------------------

def joint_soft_limits(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """sum_j distance outside the 90 % soft joint range [rad]."""
  a = env.scene[asset_cfg.name]
  q = a.data.joint_pos[:, asset_cfg.joint_ids]
  lim = a.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  return ((lim[..., 0] - q).clamp(min=0) + (q - lim[..., 1]).clamp(min=0)).sum(dim=1)


class loop_residual:
  """sum over the 4 ankle loops of relu(|rod end - foot ball| - r0) / r0."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    a: Entity = env.scene["robot"]
    names = list(a.site_names)
    self.a = torch.tensor([names.index(f"{s}_rod_{x}_end") for s in robot.SIDES for x in "AB"], device=env.device)
    self.b = torch.tensor([names.index(f"{s}_ball_{x}") for s in robot.SIDES for x in "AB"], device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, r0: float) -> torch.Tensor:
    p = env.scene["robot"].data.site_pos_w
    r = (p[:, self.a] - p[:, self.b]).norm(dim=-1)
    env.extras.setdefault("log", {})["Metrics/loop_residual_max_mm"] = r.max() * 1e3
    return ((r - r0).clamp(min=0) / r0).sum(dim=1)


def contact_count(env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 1.0) -> torch.Tensor:
  """Number of substeps (of the last policy step) in which any primary of the sensor had a
  contact force above the threshold."""
  data = env.scene[sensor_name].data
  f = data.force_history.norm(dim=-1)  # [B, N, H]
  return (f > force_threshold).any(dim=1).sum(dim=-1).float()


# --- actuator -----------------------------------------------------------------------------

def torque_margin(env: ManagerBasedRlEnv, soft_ratio: float) -> torch.Tensor:
  """sum_motors relu(max over the step's substeps of |tau_req| / N(qdot, tau_req) - soft_ratio)."""
  r = get_motor(env).step_max_ratio
  _log(env, "torque_ratio_max", r.max(dim=1).values)
  return (r - soft_ratio).clamp(min=0).sum(dim=1)


class thermal_effort:
  """sum_motors (tau / rated)^2 with the applied (clipped) torque of the last substep."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    m = get_motor(env)
    _, _, rated = resolve_matching_names_values(cfg.params["rated"], m.target_names)
    self.rated = torch.tensor(rated, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, rated) -> torch.Tensor:
    del rated
    return (get_motor(env).applied_effort / self.rated).square().sum(dim=1)


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  am = env.action_manager
  return (am.action - am.prev_action).square().sum(dim=1)


def joint_acc(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.joint_acc[:, asset_cfg.joint_ids].square().sum(dim=1)


# --- feet ---------------------------------------------------------------------------------

def foot_slip(env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """sum_feet |v_xy(foot sole site)|^2 while the foot touches the ground (no command gating)."""
  contact = (env.scene[sensor_name].data.found > 0).float()
  v = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return (v.square().sum(dim=-1) * contact).sum(dim=1)


def foot_impact(env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg,
                height_threshold: float = 0.10, contact_threshold: float = 7.0) -> torch.Tensor:
  """sum_feet relu(-v_z)^2 while the sole is within `height_threshold` of the floor and carries
  less than `contact_threshold` N (i.e. descending just before touchdown)."""
  a = env.scene[asset_cfg.name]
  z = a.data.site_pos_w[:, asset_cfg.site_ids, 2] - env.scene.env_origins[:, 2:3]
  v_down = (-a.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]).clamp(min=0)
  light = env.scene[sensor_name].data.force.norm(dim=-1) < contact_threshold
  near = (z < height_threshold) & light
  return (v_down.square() * near.float()).sum(dim=1)

