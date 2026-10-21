"""DR knobs of the viewer: one scalar per item, written into the event-term params at reset.

Every reset-mode DR event of the training cfg samples U(lo, hi). The viewer instead sets
lo = hi = the GUI value, so a reset spawns the robot with exactly that value (per-motor
items get the same value on every motor). "Clean" = the MJCF nominal model; "Random" draws
one value per item from the training range (`train`). Values outside the training range
are allowed (the sliders go further) for out-of-distribution tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import env_cfg as task
from ..mdp.events import get_motor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class Knob:
  key: str
  label: str
  folder: str
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


_INERTIA_PARAMS = (("alpha", "mass log-scale α", 0.15, 0.001), ("t1", "CoM x [m]", 0.05, 0.001),
                   ("t2", "CoM y [m]", 0.05, 0.001), ("t3", "CoM z [m]", 0.05, 0.001),
                   ("d", "shape d", 0.05, 0.001))


def _knobs(foot_friction_nominal: float) -> tuple[Knob, ...]:
  deg2 = 2.0
  k = [
    Knob("payload", "payload [kg]", "Mass", -2.0, 20.0, 0.1, task.PAYLOAD_KG, 0.0, "골반 CoM에 더하는 점질량"),
    Knob("foot_friction", "foot friction", "Contact", 0.05, 2.0, 0.01, task.FOOT_FRICTION, foot_friction_nominal,
         f"발 마찰계수 (모델 기본 {foot_friction_nominal:.2f})"),
    Knob("kp", "Kp scale", "Motor", 0.5, 1.5, 0.01, (0.8, 1.2), 1.0, "12개 모터 공통 배율"),
    Knob("kd", "Kd scale", "Motor", 0.5, 1.5, 0.01, (0.8, 1.2), 1.0, "12개 모터 공통 배율"),
    Knob("motor_friction", "frictionloss scale", "Motor", 0.0, 2.0, 0.01, (0.8, 1.2), 1.0),
    Knob("motor_damping", "damping scale", "Motor", 0.0, 2.0, 0.01, (0.8, 1.2), 1.0),
    Knob("motor_armature", "armature scale", "Motor", 0.0, 2.0, 0.01, (0.8, 1.2), 1.0),
    Knob("encoder_bias", "encoder bias [rad]", "Sensor", -0.05, 0.05, 0.001, (-0.015, 0.015), 0.0,
         "12개 모터 공통 (obs와 명령에 같이 들어감)"),
    Knob("imu_roll", "IMU mount roll [deg]", "Sensor", -5.0, 5.0, 0.1, (-deg2, deg2), 0.0),
    Knob("imu_pitch", "IMU mount pitch [deg]", "Sensor", -5.0, 5.0, 0.1, (-deg2, deg2), 0.0),
    Knob("imu_yaw", "IMU mount yaw [deg]", "Sensor", -5.0, 5.0, 0.1, (-deg2, deg2), 0.0),
    Knob("cmd_delay", "command delay [x5 ms]", "Latency", 0, 4, 1, (0, 4), 0, "버스 지연, 최대 20 ms", integer=True),
    Knob("obs_lag", "obs delay [x20 ms]", "Latency", 0, 2, 1, (0, task.OBS_MAX_LAG), 0,
         "IMU+엔코더 같이 지연, 최대 40 ms", integer=True),
    Knob("passive_friction", "frictionloss [N·m]", "Passive joints", 0.0, 1.0, 0.01, (0.0, 0.5), 0.0,
         "rod U-joint + 발목 수동 관절, 절대값"),
    Knob("passive_damping", "damping [N·m·s/rad]", "Passive joints", 0.0, 0.1, 0.001, (0.0, 0.05), 0.0,
         "절대값. 0 = 모델 기본값 (rod 0.02, 발목 0)"),
  ]
  for link, (_, ranges) in task.INERTIAL_DR.items():
    for p, label, span, step in _INERTIA_PARAMS:
      k.append(Knob(f"inertia_{link}_{p}", label, f"Inertia/{link}", -span, span, step,
                    ranges[f"{p}_range"], 0.0))
  return tuple(k)


class DrControl:
  """Holds the knob values that are applied at the next reset."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    em = env.event_manager
    ff = em.get_term_cfg("foot_friction")
    asset = env.scene["robot"]
    gid = int(asset.indexing.geom_ids[ff.params["asset_cfg"].geom_ids][0])
    nominal = float(env.sim.get_default_field("geom_friction")[gid, 0])
    self.knobs = _knobs(nominal)
    self.by_key = {k.key: k for k in self.knobs}
    self.values = {k.key: float(k.clean) for k in self.knobs}
    self.applied = dict(self.values)

  def clean(self) -> dict[str, float]:
    return {k.key: float(k.clean) for k in self.knobs}

  def random(self) -> dict[str, float]:
    return {k.key: k.sample() for k in self.knobs}

  def pending(self) -> list[str]:
    return [k for k, v in self.values.items() if abs(v - self.applied[k]) > 1e-9]

  def write_params(self) -> None:
    """Write the current values into the reset events (takes effect at the next reset)."""
    env, v = self.env, dict(self.values)
    em = env.event_manager

    def fix(term: str, **ranges: float) -> None:
      em.get_term_cfg(term).params.update({name: (val, val) for name, val in ranges.items()})

    fix("payload", mass_range=v["payload"])
    fix("foot_friction", ranges=v["foot_friction"])
    fix("pd_gains", kp_range=v["kp"], kd_range=v["kd"])
    fix("motor_friction", ranges=v["motor_friction"])
    fix("motor_damping", ranges=v["motor_damping"])
    fix("motor_armature", ranges=v["motor_armature"])
    fix("encoder_bias", bias_range=v["encoder_bias"])
    fix("imu_mount", roll_range=math.radians(v["imu_roll"]), pitch_range=math.radians(v["imu_pitch"]),
        yaw_range=math.radians(v["imu_yaw"]))
    fix("passive_friction", ranges=v["passive_friction"])
    pd = em.get_term_cfg("passive_damping").params
    if v["passive_damping"] == 0.0:  # model nominal (rods 0.02, ankles 0): scale 1
      pd.update(operation="scale", ranges=(1.0, 1.0))
    else:
      pd.update(operation="abs", ranges=(v["passive_damping"],) * 2)
    for link in task.INERTIAL_DR:
      fix(f"inertia_{link}", **{f"{p}_range": v[f"inertia_{link}_{p}"] for p, *_ in _INERTIA_PARAMS})
    em.get_term_cfg("obs_lag").params["lag"] = int(v["obs_lag"])  # viewer cfg: fixed_obs_lag
    buf = get_motor(env)._delay_buffer  # noqa: SLF001
    if buf is not None:
      buf.min_lag = buf.max_lag = int(v["cmd_delay"])
    self.applied = v
