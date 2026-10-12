"""Position-PD actuator clamped to the motor's measured torque-speed (T-N) curve.

In the motoring quadrant (torque and speed of the same sign) |tau| <= tau_tn(|omega|);
when braking the motor may apply its full peak. Copied from the training fork (AB-mode subset).
"""

from __future__ import annotations
from dataclasses import dataclass
import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

def tn_curve_tensor(curve: tuple[tuple[float, float], ...], device: str) -> tuple[torch.Tensor, torch.Tensor]:
  w = torch.tensor([c[0] for c in curve], dtype=torch.float32, device=device)
  t = torch.tensor([c[1] for c in curve], dtype=torch.float32, device=device)
  assert (w[1:] > w[:-1]).all(), "T-N curve speeds must increase"
  return w, t


def tn_limit(w_tab: torch.Tensor, t_tab: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
  """tau_tn(speed) by linear interpolation; flat (= first torque) below the first speed,
  0 beyond the last. `speed` may be negative (then the flat peak applies)."""
  s = speed.clamp(min=w_tab[0].item())
  idx = torch.searchsorted(w_tab, s.reshape(-1).contiguous()).reshape(s.shape).clamp(1, len(w_tab) - 1)
  w0, w1 = w_tab[idx - 1], w_tab[idx]
  t0, t1 = t_tab[idx - 1], t_tab[idx]
  f = ((s - w0) / (w1 - w0)).clamp(0, 1)
  out = t0 + f * (t1 - t0)
  return torch.where(s >= w_tab[-1], torch.zeros_like(out), out)


def tn_clamp(tau: torch.Tensor, omega: torch.Tensor, w_tab: torch.Tensor, t_tab: torch.Tensor) -> torch.Tensor:
  peak = t_tab[0]
  upper = torch.where(omega >= 0, tn_limit(w_tab, t_tab, omega), peak.expand_as(omega))
  lower = torch.where(omega < 0, -tn_limit(w_tab, t_tab, -omega), (-peak).expand_as(omega))
  return torch.minimum(torch.maximum(tau, lower), upper)


@dataclass(kw_only=True)
class TnPdActuatorCfg(IdealPdActuatorCfg):
  tn_curve: tuple[tuple[float, float], ...]
  """((speed rad/s, max torque N*m), ...) increasing in speed; first torque = peak."""

  def build(self, entity, target_ids, target_names):  # noqa: ANN001
    return TnPdActuator(self, entity, target_ids, target_names)


class TnPdActuator(IdealPdActuator[TnPdActuatorCfg]):
  def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    self.w_tab, self.t_tab = tn_curve_tensor(self.cfg.tn_curve, device)
    self._vel: torch.Tensor | None = None

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    self._vel = cmd.vel
    return super().compute(cmd)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self._vel is not None
    return tn_clamp(effort, self._vel, self.w_tab, self.t_tab)

