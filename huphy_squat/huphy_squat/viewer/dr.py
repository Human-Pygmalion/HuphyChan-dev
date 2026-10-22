"""DR knobs of the viewer: one slider per randomized element, applied at the next reset.

The training cfg draws every reset-mode DR element independently (per motor, per passive
joint, per body). The viewer keeps that granularity: each element gets its own knob, and at
reset the DR events are replaced by calls that write exactly the knob values (the same mjlab
DR functions, with lo = hi, once per element). "Clean" = the MJCF nominal model; "Random"
draws every element independently from its training range (`train`), like a training reset.
Values outside the training range are allowed (the sliders go further) for OOD tests.

Shared like in training (one value per env): payload, foot friction (both feet), IMU mount,
command delay, observation delay.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .. import env_cfg as task
from .. import robot
from ..mdp.events import get_motor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class Knob:
  key: str
  label: str
  folder: str  # "/"-separated GUI folder path
  lo: float
  hi: float
  step: float
  train: tuple[float, float]
  clean: float
  hint: str = ""
  integer: bool = False

  def sample(self) -> float:
    if self.integer:
      return float(random.randint(int(self.train[0]), int(self.train[1])))
    v = random.uniform(*self.train)
    return round(v / self.step) * self.step


def _short(joint: str) -> str:
  side, rest = joint.split("_", 1)
  return f"{side} {rest.removesuffix('_joint')}"


_INERTIA_PARAMS = (("alpha", "mass log-scale α", 0.15), ("t1", "CoM x [m]", 0.05), ("t2", "CoM y [m]", 0.05),
                   ("t3", "CoM z [m]", 0.05), ("d", "shape d", 0.05))
MOTOR_SCALES = (("kp", "Kp scale", (0.8, 1.2)), ("kd", "Kd scale", (0.8, 1.2)),
                ("motor_friction", "frictionloss scale", (0.8, 1.2)), ("motor_damping", "damping scale", (0.8, 1.2)),
                ("motor_armature", "armature scale", (0.8, 1.2)))


class _PerElement:
  """Event func that calls one mjlab DR function once per element with its own fixed value."""

  def __init__(self, base):
    self.base = base
    self.recompute = getattr(base, "recompute", RecomputeLevel.none)
    self.model_fields = getattr(base, "model_fields", ())
    self.__name__ = f"viewer_{getattr(base, '__name__', 'dr')}"

  def __call__(self, env: ManagerBasedRlEnv, env_ids, calls: list[dict]) -> None:
    for kw in calls:
      self.base(env, env_ids, **kw)


def _set_pd(env: ManagerBasedRlEnv, env_ids, kp: torch.Tensor, kd: torch.Tensor) -> None:
  m = get_motor(env)
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device, torch.long)
  m.set_gains(ids, kp=m.default_stiffness[ids] * kp, kd=m.default_damping[ids] * kd)


def _set_encoder_bias(env: ManagerBasedRlEnv, env_ids, joint_ids: torch.Tensor, bias: torch.Tensor) -> None:
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device, torch.long)
  env.scene["robot"].data.encoder_bias[ids[:, None], joint_ids] = bias


class DrControl:
  """Holds the knob values that are applied at the next reset."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    asset = env.scene["robot"]
    em = env.event_manager
    self.motor_names = list(get_motor(env).target_names)  # actuator column order
    self.passive_names = list(robot.ROD_JOINT_NAMES + robot.ANKLE_JOINT_NAMES)
    ff = em.get_term_cfg("foot_friction")
    gid = int(asset.indexing.geom_ids[ff.params["asset_cfg"].geom_ids][0])
    foot_mu = float(env.sim.get_default_field("geom_friction")[gid, 0])

    def dof_default(field: str, joint: str) -> float:
      dof = int(asset.indexing.joint_v_adr[asset.joint_names.index(joint)])
      return float(env.sim.get_default_field(field)[dof])

    deg2 = 2.0
    k = [
      Knob("payload", "payload [kg]", "Mass", -2.0, 20.0, 0.1, task.PAYLOAD_KG, 0.0, "골반 CoM에 더하는 점질량"),
      Knob("foot_friction", "foot friction", "Contact", 0.05, 2.0, 0.01, task.FOOT_FRICTION, foot_mu,
           f"양발 공통 (학습도 공통). 모델 기본 {foot_mu:.2f}"),
      Knob("cmd_delay", "command delay [x5 ms]", "Latency", 0, 4, 1, (0, 4), 0, "버스 지연, 최대 20 ms", integer=True),
      Knob("obs_lag", "obs delay [x20 ms]", "Latency", 0, 2, 1, (0, task.OBS_MAX_LAG), 0,
           "IMU+엔코더 같이 지연, 최대 40 ms", integer=True),
      Knob("imu_roll", "roll [deg]", "Sensor/IMU mount", -5.0, 5.0, 0.1, (-deg2, deg2), 0.0),
      Knob("imu_pitch", "pitch [deg]", "Sensor/IMU mount", -5.0, 5.0, 0.1, (-deg2, deg2), 0.0),
      Knob("imu_yaw", "yaw [deg]", "Sensor/IMU mount", -5.0, 5.0, 0.1, (-deg2, deg2), 0.0),
    ]
    for j in self.motor_names:
      k.append(Knob(f"encoder_bias/{j}", _short(j), "Sensor/encoder bias [rad]", -0.05, 0.05, 0.001, (-0.015, 0.015),
                    0.0, "obs와 명령에 같이 들어감"))
    for key, label, train in MOTOR_SCALES:
      for j in self.motor_names:
        k.append(Knob(f"{key}/{j}", _short(j), f"Motor/{label}", 0.0 if "motor_" in key else 0.5,
                      2.0 if "motor_" in key else 1.5, 0.01, train, 1.0))
    for j in self.passive_names:
      k.append(Knob(f"passive_friction/{j}", _short(j), "Passive joints/frictionloss [N·m]", 0.0, 1.0, 0.01,
                    (0.0, 0.5), dof_default("dof_frictionloss", j)))
    for j in self.passive_names:
      d0 = dof_default("dof_damping", j)
      k.append(Knob(f"passive_damping/{j}", _short(j), "Passive joints/damping [N·m·s/rad]", 0.0, 0.1, 0.001,
                    (0.0, 0.05), d0, f"모델 기본 {d0:g}"))
    self.inertia_bodies: dict[str, tuple[str, ...]] = {}
    for link, (bodies, ranges) in task.INERTIAL_DR.items():
      self.inertia_bodies[link] = bodies
      for b in bodies:
        for p, label, span in _INERTIA_PARAMS:
          k.append(Knob(f"inertia/{b}/{p}", label, f"Inertia/{b}", -span, span, 0.001, ranges[f"{p}_range"], 0.0))
    self.knobs = tuple(k)
    self.by_key = {x.key: x for x in self.knobs}
    self.values = {x.key: float(x.clean) for x in self.knobs}
    self.applied = dict(self.values)
    self._install(env)

  def _install(self, env: ManagerBasedRlEnv) -> None:
    """Swap the per-element DR events for per-element fixed-value versions."""
    em, scene = env.event_manager, env.scene

    def cfg(**kw) -> SceneEntityCfg:
      c = SceneEntityCfg("robot", **kw)
      c.resolve(scene)
      return c

    self._joint_cfg = {j: cfg(joint_names=(j,)) for j in self.motor_names + self.passive_names}
    self._body_cfg = {b: cfg(body_names=(b,)) for bodies in self.inertia_bodies.values() for b in bodies}
    for term, base in (("motor_friction", dr.joint_friction), ("motor_damping", dr.joint_damping),
                       ("motor_armature", dr.joint_armature), ("passive_friction", dr.joint_friction),
                       ("passive_damping", dr.joint_damping)):
      em.get_term_cfg(term).func = _PerElement(base)
    for link in self.inertia_bodies:
      em.get_term_cfg(f"inertia_{link}").func = _PerElement(dr.pseudo_inertia)
    em.get_term_cfg("pd_gains").func = _set_pd
    em.get_term_cfg("encoder_bias").func = _set_encoder_bias
    asset = scene["robot"]
    self._bias_ids = torch.tensor([asset.joint_names.index(j) for j in self.motor_names], device=env.device)
    self.write_params()

  def clean(self) -> dict[str, float]:
    return {x.key: float(x.clean) for x in self.knobs}

  def random(self) -> dict[str, float]:
    return {x.key: x.sample() for x in self.knobs}

  def pending(self) -> list[str]:
    return [key for key, v in self.values.items() if abs(v - self.applied[key]) > 1e-9]

  def write_params(self) -> None:
    """Write the current values into the reset events (takes effect at the next reset)."""
    env, v = self.env, dict(self.values)
    em, dev = env.event_manager, env.device

    def fix(term: str, **ranges: float) -> None:
      em.get_term_cfg(term).params.update({name: (val, val) for name, val in ranges.items()})

    fix("payload", mass_range=v["payload"])
    fix("foot_friction", ranges=v["foot_friction"])
    fix("imu_mount", roll_range=math.radians(v["imu_roll"]), pitch_range=math.radians(v["imu_pitch"]),
        yaw_range=math.radians(v["imu_yaw"]))
    vec = lambda key: torch.tensor([v[f"{key}/{j}"] for j in self.motor_names], device=dev)  # noqa: E731
    em.get_term_cfg("pd_gains").params = {"kp": vec("kp"), "kd": vec("kd")}
    em.get_term_cfg("encoder_bias").params = {"joint_ids": self._bias_ids, "bias": vec("encoder_bias")}
    for term, joints, op in (("motor_friction", self.motor_names, "scale"), ("motor_damping", self.motor_names, "scale"),
                             ("motor_armature", self.motor_names, "scale"),
                             ("passive_friction", self.passive_names, "abs"),
                             ("passive_damping", self.passive_names, "abs")):
      em.get_term_cfg(term).params = {"calls": [
        {"asset_cfg": self._joint_cfg[j], "operation": op, "ranges": (v[f"{term}/{j}"],) * 2} for j in joints]}
    for link, bodies in self.inertia_bodies.items():
      em.get_term_cfg(f"inertia_{link}").params = {"calls": [
        {"asset_cfg": self._body_cfg[b], **{f"{p}_range": (v[f"inertia/{b}/{p}"],) * 2 for p, *_ in _INERTIA_PARAMS}}
        for b in bodies]}
    em.get_term_cfg("obs_lag").params["lag"] = int(v["obs_lag"])  # viewer cfg: fixed_obs_lag
    buf = get_motor(env)._delay_buffer  # noqa: SLF001
    if buf is not None:
      buf.min_lag = buf.max_lag = int(v["cmd_delay"])
    self.applied = v
