"""Reset, disturbance and bookkeeping events of the squat task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .. import linkage, robot

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device)
  return env_ids.to(env.device, dtype=torch.long)


def get_motor(env: ManagerBasedRlEnv, entity_name: str = "robot"):
  """The single HuphyMotor actuator group of the robot."""
  acts = env.scene[entity_name].actuators
  assert len(acts) == 1, "huphy_squat expects one actuator group"
  return acts[0]


class reset_squat_pose:
  """Default stance + uniform noise on the 12 motors, ankle loops kept closed, sole on the floor.

  * motors: default + U(-noise, noise) (clamped to the soft joint limits);
  * stance: with probability `spread_prob` each hip roll gets an extra U(spread_range) outward
    (negative = inward, narrower stance); the feet are not re-levelled, so the policy starts
    from narrow / wide and slightly tilted stances it has to recover from;
  * passive rod / ankle joints: bilinear lookup in the crank-angle table
    (assets/linkage_table.npz), so the `connect` equalities start satisfied;
  * base: upright, yaw U(yaw_range), at the env origin, height such that the lowest sole
    corner (2nd-order model in the table) sits `clearance` above the floor;
  * all velocities zero.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _ROBOT).name]
    dev = env.device
    tab = linkage.load_table()
    names = list(asset.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    motors = [str(n) for n in tab["motor_names"]]
    self.motor_ids = torch.tensor([idx[n] for n in motors], device=dev)
    self.legs = []
    for side in robot.SIDES:
      self.legs.append(dict(
        crank_a=idx[f"{side}_crank_A_joint"], crank_b=idx[f"{side}_crank_B_joint"],
        passive=torch.tensor([idx[n] for n in linkage.passive_names(side)], device=dev),
        grid_a=torch.tensor(tab[f"{side}_grid_a"], dtype=torch.float, device=dev),
        grid_b=torch.tensor(tab[f"{side}_grid_b"], dtype=torch.float, device=dev),
        vals=torch.tensor(tab[f"{side}_passive"], dtype=torch.float, device=dev),
      ))
    default = dict(zip(tab["default_names"], tab["default_values"], strict=True))
    self.default_q = torch.tensor([default[n] for n in names], dtype=torch.float, device=dev)
    self.z0 = torch.tensor(tab["corner_z0"], dtype=torch.float, device=dev)
    self.J = torch.tensor(tab["corner_dz_dq"], dtype=torch.float, device=dev)
    self.H = torch.tensor(tab["corner_d2z_dq2"], dtype=torch.float, device=dev)
    lim = asset.data.soft_joint_pos_limits[0, self.motor_ids]
    self.lo, self.hi = lim[:, 0], lim[:, 1]
    # columns of the hip-roll motors in motor order, and their outward sign (R +, L -)
    self.roll_cols = torch.tensor([motors.index(f"{s}_hip_roll_joint") for s in robot.SIDES], device=dev)
    self.roll_sign = torch.tensor([1.0 if s == "R" else -1.0 for s in robot.SIDES], device=dev)

  @staticmethod
  def _bilinear(grid_a, grid_b, vals, a, b):
    n = len(grid_a)
    fa = ((a - grid_a[0]) / (grid_a[1] - grid_a[0])).clamp(0, n - 1 - 1e-6)
    fb = ((b - grid_b[0]) / (grid_b[1] - grid_b[0])).clamp(0, n - 1 - 1e-6)
    i, j = fa.long(), fb.long()
    u, v = (fa - i)[:, None], (fb - j)[:, None]
    return ((1 - u) * (1 - v) * vals[i, j] + u * (1 - v) * vals[i + 1, j]
            + (1 - u) * v * vals[i, j + 1] + u * v * vals[i + 1, j + 1])

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, motor_noise: float = 0.05,
               spread_range: tuple[float, float] = (0.0, 0.0), spread_prob: float = 0.0,
               yaw_range: tuple[float, float] = (-math.pi, math.pi), clearance: float = robot.RESET_CLEARANCE,
               asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    ids = _ids(env, env_ids)
    n = len(ids)
    asset: Entity = env.scene[asset_cfg.name]
    q = self.default_q.expand(n, -1).clone()
    dq = torch.empty(n, len(self.motor_ids), device=env.device).uniform_(-motor_noise, motor_noise)
    if spread_prob > 0:
      extra = torch.empty(n, 2, device=env.device).uniform_(*spread_range)
      extra *= (torch.rand(n, 1, device=env.device) < spread_prob).float()
      dq[:, self.roll_cols] += extra * self.roll_sign
    qm = (self.default_q[self.motor_ids] + dq).clamp(self.lo, self.hi)
    dq = qm - self.default_q[self.motor_ids]
    q[:, self.motor_ids] = qm
    for leg in self.legs:
      q[:, leg["passive"]] = self._bilinear(leg["grid_a"], leg["grid_b"], leg["vals"],
                                            q[:, leg["crank_a"]], q[:, leg["crank_b"]])
    corners = self.z0 + dq @ self.J.T + 0.5 * torch.einsum("cij,ni,nj->nc", self.H, dq, dq)
    root = torch.zeros(n, 13, device=env.device)
    root[:, :3] = env.scene.env_origins[ids]
    root[:, 2] += -corners.min(dim=1).values + clearance
    yaw = torch.empty(n, device=env.device).uniform_(*yaw_range)
    zero = torch.zeros_like(yaw)
    root[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
    asset.write_root_state_to_sim(root, env_ids=ids)
    asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)


def sample_obs_lag(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, max_lag: int = 1) -> None:
  """One observation lag per env and episode, in policy steps, shared by IMU and joint terms."""
  ids = _ids(env, env_ids)
  lag = getattr(env, "huphy_obs_lag", None)
  if lag is None:
    lag = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.huphy_obs_lag = lag  # type: ignore[attr-defined]
  lag[ids] = torch.randint(0, max_lag + 1, (len(ids),), device=env.device)


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def payload_mass(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, mass_range: tuple[float, float],
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("base_link",))) -> None:
  """Add a point mass at the pelvis CoM (zero added inertia): body_mass += U(mass_range).

  Must run after the pelvis pseudo-inertia event of the same reset: that event rewrites
  body_mass from the nominal value every reset, so the payload never accumulates.
  """
  ids = _ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  body = asset.indexing.body_ids[asset_cfg.body_ids].long()
  dm = torch.empty(len(ids), len(body), device=env.device).uniform_(*mass_range)
  env.sim.model.body_mass[ids[:, None], body] += dm
  pl = getattr(env, "huphy_payload", None)
  if pl is None:
    pl = torch.zeros(env.num_envs, device=env.device)
    env.huphy_payload = pl  # type: ignore[attr-defined]
  pl[ids] = dm.sum(dim=1)


class push_pelvis:
  """Horizontal force on the pelvis: magnitude U(force_range) N, random direction, for
  U(duration_s) s, then a pause of U(interval_s) s. Timers restart at every reset (the first
  push of an episode comes after one full pause). Register with mode="step"."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot", body_names=("base_link",)))
    self.asset: Entity = env.scene[asset_cfg.name]
    self.body_ids = asset_cfg.body_ids
    self.dt = env.step_dt
    self.interval = cfg.params["interval_s"]
    self.dev = env.device
    n = env.num_envs
    self.force = torch.zeros(n, 3, device=self.dev)
    self.remaining = torch.zeros(n, device=self.dev)  # >0 while pushing
    self.pause = self._u(self.interval, n)
    self.count = torch.zeros(n, device=self.dev)
    env.huphy_push = self  # critic observation reads force / remaining

  def _u(self, r: tuple[float, float], n: int) -> torch.Tensor:
    return torch.rand(n, device=self.dev) * (r[1] - r[0]) + r[0]

  def _write(self) -> None:
    f = self.force[:, None, :]
    self.asset.write_external_wrench_to_sim(f, torch.zeros_like(f), body_ids=self.body_ids)

  def __call__(self, env: ManagerBasedRlEnv, env_ids, force_range: tuple[float, float],
               duration_s: tuple[float, float], interval_s: tuple[float, float],
               asset_cfg: SceneEntityCfg | None = None) -> None:
    del env, env_ids, asset_cfg
    active = self.remaining > 0
    self.remaining = torch.where(active, self.remaining - self.dt, self.remaining)
    ended = active & (self.remaining <= 0)
    self.force[ended] = 0.0
    self.pause = torch.where(ended, self._u(self.interval, len(ended)), self.pause)
    idle = ~active
    self.pause = torch.where(idle, self.pause - self.dt, self.pause)
    start = idle & (self.pause <= 0)
    if start.any():
      k = int(start.sum())
      mag = self._u(force_range, k)
      ang = self._u((0.0, 2 * math.pi), k)
      self.force[start] = torch.stack((mag * torch.cos(ang), mag * torch.sin(ang), torch.zeros_like(mag)), dim=1)
      self.remaining[start] = self._u(duration_s, k)
      self.pause[start] = 0.0
      self.count[start] += 1
    self._write()

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self.force[idx] = 0.0
    self.remaining[idx] = 0.0
    n = self.num(idx)
    self.pause[idx] = self._u(self.interval, n)
    self.count[idx] = 0.0
    self._write()

  def num(self, idx) -> int:
    return self.force.shape[0] if isinstance(idx, slice) else len(idx)


def motor_step_end(env: ManagerBasedRlEnv, env_ids, entity_name: str = "robot") -> None:
  """Restart the per-env-step torque-ratio maximum of the actuator (mode="step", after rewards)."""
  del env_ids
  get_motor(env, entity_name).begin_env_step()
