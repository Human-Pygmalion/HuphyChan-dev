"""Explicit PD actuator with the measured torque-speed (T-N) envelope, one group for all motors.

Per physics step (5 ms), for every motor:
  tau_req = Kp * (q_target - q) - Kd * qdot                      (requested torque)
  tau     = clamp(tau_req, lower(qdot), upper(qdot))             (the only clip)
  upper   = T_N(|qdot|) if qdot >= 0 else peak,   lower = -T_N(|qdot|) if qdot < 0 else -peak
i.e. in the motoring quadrant (tau * qdot > 0) the measured curve limits the torque; when
braking the motor may apply its peak. The MuJoCo `<motor>` ctrl/force range is set to the
curve's peak, so it never clips a second time (applied effort == MuJoCo actuator force).

All 12 motors form one actuator group so the command delay (bus latency) is one lag per env
shared by every motor. Gains are per joint (regex dicts). The class subclasses mjlab's
IdealPdActuator so `mjlab.envs.mdp.dr.pd_gains` randomizes it through `set_gains`.

For the torque-margin penalty the actuator also keeps, per env step, the maximum over the
physics substeps of |tau_req| / N(qdot, tau_req) (N = the clamp bound on the side tau_req
points to). `begin_env_step()` (called by a step event after the rewards) restarts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCfg, ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.spec import create_motor_actuator


def tn_limit(w_tab: torch.Tensor, t_tab: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
  """T_N(speed) by linear interpolation for speed >= 0: flat peak below the first point,
  0 at and beyond the last point (no-load speed)."""
  s = speed.clamp(min=float(w_tab[0]))
  idx = torch.searchsorted(w_tab, s.contiguous()).clamp(1, len(w_tab) - 1)
  w0, w1, t0, t1 = w_tab[idx - 1], w_tab[idx], t_tab[idx - 1], t_tab[idx]
  out = t0 + ((s - w0) / (w1 - w0)).clamp(0, 1) * (t1 - t0)
  return torch.where(s >= w_tab[-1], torch.zeros_like(out), out)


@dataclass(kw_only=True)
class HuphyMotorCfg(ActuatorCfg):
  joint_stiffness: dict[str, float]
  """Kp [N*m/rad], regex -> value."""
  joint_damping: dict[str, float]
  """Kd [N*m*s/rad], regex -> value."""
  joint_family: dict[str, str]
  """regex -> motor family (key of `tn_curves` / `motor_dynamics`)."""
  tn_curves: dict[str, tuple[tuple[float, float], ...]]
  """family -> ((speed rad/s, max torque N*m), ...), increasing speed, first torque = peak."""
  motor_dynamics: dict[str, dict[str, float]] = field(default_factory=dict)
  """family -> {armature, frictionloss, viscous_damping} written into the MJCF joint."""

  def build(self, entity, target_ids, target_names):  # noqa: ANN001
    return HuphyMotor(self, entity, target_ids, target_names)


class HuphyMotor(IdealPdActuator):
  cfg: HuphyMotorCfg

  def _per_joint(self, data: dict, names: list[str]) -> list:
    _, _, values = resolve_matching_names_values(data, names, preserve_order=False)
    return values

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    families = self._per_joint(self.cfg.joint_family, target_names)
    for name, fam in zip(target_names, families, strict=True):
      dyn = self.cfg.motor_dynamics.get(fam, {})
      act = create_motor_actuator(
        spec,
        name,
        effort_limit=float(self.cfg.tn_curves[fam][0][1]),
        armature=dyn.get("armature"),
        frictionloss=dyn.get("frictionloss"),
        viscous_damping=dyn.get("viscous_damping"),
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(act)

  def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
    super(IdealPdActuator, self).initialize(mj_model, model, data, device)  # base Actuator only
    names = self.target_names
    n_env, n = data.nworld, len(names)
    kp = torch.tensor(self._per_joint(self.cfg.joint_stiffness, names), dtype=torch.float, device=device)
    kd = torch.tensor(self._per_joint(self.cfg.joint_damping, names), dtype=torch.float, device=device)
    self.families = self._per_joint(self.cfg.joint_family, names)
    self.stiffness = kp.expand(n_env, n).clone()
    self.damping = kd.expand(n_env, n).clone()
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    peak = torch.tensor([self.cfg.tn_curves[f][0][1] for f in self.families], dtype=torch.float, device=device)
    self.peak = peak
    self.force_limit = peak.expand(n_env, n).clone()
    self.default_force_limit = self.force_limit.clone()
    # One (speed, torque) table per family + the joint columns that use it.
    self._tables = []
    for fam, curve in self.cfg.tn_curves.items():
      cols = [i for i, f in enumerate(self.families) if f == fam]
      if not cols:
        continue
      w = torch.tensor([c[0] for c in curve], dtype=torch.float, device=device)
      t = torch.tensor([c[1] for c in curve], dtype=torch.float, device=device)
      assert bool((w[1:] > w[:-1]).all()), f"T-N speeds must increase ({fam})"
      self._tables.append((torch.tensor(cols, device=device), w, t))
    self.requested_effort = torch.zeros(n_env, n, device=device)
    self.applied_effort = torch.zeros(n_env, n, device=device)
    self.step_max_ratio = torch.zeros(n_env, n, device=device)
    self._new_env_step = True

  def speed_limit(self, speed: torch.Tensor) -> torch.Tensor:
    """T_N(|speed|) for every column. Shape like `speed`."""
    out = torch.empty_like(speed)
    a = speed.abs()
    for cols, w, t in self._tables:
      out[:, cols] = tn_limit(w, t, a[:, cols])
    return out

  def bounds(self, vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tn = self.speed_limit(vel)
    peak = self.peak.expand_as(vel)
    upper = torch.where(vel >= 0, tn, peak)
    lower = torch.where(vel < 0, -tn, -peak)
    return lower, upper

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    tau_req = self.stiffness * (cmd.position_target - cmd.pos) + self.damping * (cmd.velocity_target - cmd.vel)
    tau_req = tau_req + cmd.effort_target
    lower, upper = self.bounds(cmd.vel)
    tau = torch.minimum(torch.maximum(tau_req, lower), upper)
    bound = torch.where(tau_req >= 0, upper, -lower).clamp(min=1e-3)
    ratio = tau_req.abs() / bound
    if self._new_env_step:
      self.step_max_ratio.copy_(ratio)
      self._new_env_step = False
    else:
      torch.maximum(self.step_max_ratio, ratio, out=self.step_max_ratio)
    self.requested_effort.copy_(tau_req)
    self.applied_effort.copy_(tau)
    return tau

  def begin_env_step(self) -> None:
    """Restart the per-env-step maximum of the torque ratio (call once per env step)."""
    self._new_env_step = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    idx = slice(None) if env_ids is None else env_ids
    self.step_max_ratio[idx] = 0.0
    self.requested_effort[idx] = 0.0
    self.applied_effort[idx] = 0.0
