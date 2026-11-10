from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .bicycle_curriculum import STEPS_PER_ITER
from .bicycle_data import (
  BICYCLE_CONTACT_SENSOR_NAMES_BY_PAIR,
  BICYCLE_CONTACT_PAIRS,
  BICYCLE_CONTACT_PAIR_KEYS,
  FOOT_SOLE_CONTACT_LOCAL,
  HAND_CREASE_L_LOCAL,
  HAND_CREASE_R_LOCAL,
  PELVIS_BOTTOM_POLE_LOCAL,
  SOLE_BOTTOM_CENTROID_L_LOCAL,
  SOLE_BOTTOM_CENTROID_R_LOCAL,
)
from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


class BicyclePedalContactMissingTermination:
  """Terminate after either scheduled foot-pedal contact is absent N steps."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._missing_steps = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._missing_steps[env_ids] = 0

  @staticmethod
  def _has_contact(sensor: ContactSensor, force_threshold: float) -> torch.Tensor:
    found = sensor.data.found
    if found is None:
      raise RuntimeError(f"Sensor {sensor.cfg.name!r} must provide found")
    current = found.reshape(found.shape[0], -1).any(dim=-1)
    history = sensor.data.force_history
    if history is None:
      return current
    # Count a contact seen at any physics substep in the current 50 Hz policy
    # interval so a substep-phase artifact cannot increment the miss streak.
    recent = (torch.norm(history, dim=-1) > float(force_threshold)).any(dim=(1, 2))
    return current | recent

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    left_sensor_name: str,
    right_sensor_name: str,
    max_missing_steps: int = 10,
    force_threshold: float = 0.1,
    require_bilateral_schedule: bool = False,
  ) -> torch.Tensor:
    if max_missing_steps <= 0:
      raise ValueError("max_missing_steps must be positive")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    pair_indices = (
      BICYCLE_CONTACT_PAIR_KEYS.index("pedal_left"),
      BICYCLE_CONTACT_PAIR_KEYS.index("pedal_right"),
    )
    scheduled = command.ref_contact_mask[:, pair_indices].bool()
    if require_bilateral_schedule:
      # Treat only the bilateral pedal-contact window as the riding phase.
      # Outside it, reset both streaks so mount/dismount and single-foot
      # schedule windows cannot trigger this termination.
      scheduled = scheduled & scheduled.all(dim=-1, keepdim=True)
    contacts = torch.stack(
      (
        self._has_contact(env.scene[left_sensor_name], force_threshold),
        self._has_contact(env.scene[right_sensor_name], force_threshold),
      ),
      dim=-1,
    )
    missing = scheduled & ~contacts
    self._missing_steps = torch.where(
      missing, self._missing_steps + 1, torch.zeros_like(self._missing_steps)
    )
    return (self._missing_steps >= int(max_missing_steps)).any(dim=-1)


class BicycleSaddleContactMissingTermination:
  """Terminate after scheduled saddle contact is absent N consecutive steps."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._missing_steps = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._missing_steps[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str = "contact_saddle",
    max_missing_steps: int = 10,
    force_threshold: float = 0.1,
  ) -> torch.Tensor:
    if max_missing_steps <= 0:
      raise ValueError("max_missing_steps must be positive")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    pair_idx = BICYCLE_CONTACT_PAIR_KEYS.index("saddle")
    scheduled = command.ref_contact_mask[:, pair_idx].bool()
    contact = BicyclePedalContactMissingTermination._has_contact(
      env.scene[sensor_name], force_threshold
    )
    missing = scheduled & ~contact
    self._missing_steps = torch.where(
      missing, self._missing_steps + 1, torch.zeros_like(self._missing_steps)
    )
    return self._missing_steps >= int(max_missing_steps)


class BicycleHandleContactMissingTermination:
  """Terminate when either scheduled palm--handle contact is missing N steps."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._missing_steps = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._missing_steps[env_ids] = 0

  @staticmethod
  def _has_any_contact(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    force_threshold: float,
  ) -> torch.Tensor:
    if not sensor_names:
      raise ValueError("At least one handle contact sensor is required")
    contacts = [
      BicyclePedalContactMissingTermination._has_contact(
        env.scene[sensor_name], force_threshold
      )
      for sensor_name in sensor_names
    ]
    return torch.stack(contacts, dim=-1).any(dim=-1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    left_sensor_names: tuple[str, ...],
    right_sensor_names: tuple[str, ...],
    max_missing_steps: int = 30,
    force_threshold: float = 0.1,
  ) -> torch.Tensor:
    if max_missing_steps <= 0:
      raise ValueError("max_missing_steps must be positive")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    pair_indices = (
      BICYCLE_CONTACT_PAIR_KEYS.index("handle_left"),
      BICYCLE_CONTACT_PAIR_KEYS.index("handle_right"),
    )
    scheduled = command.ref_contact_mask[:, pair_indices].bool()
    contacts = torch.stack(
      (
        self._has_any_contact(env, left_sensor_names, force_threshold),
        self._has_any_contact(env, right_sensor_names, force_threshold),
      ),
      dim=-1,
    )
    missing = scheduled & ~contacts
    self._missing_steps = torch.where(
      missing, self._missing_steps + 1, torch.zeros_like(self._missing_steps)
    )
    return (self._missing_steps >= int(max_missing_steps)).any(dim=-1)


class BicycleScheduledForceContactMissingTermination:
  """Terminate when scheduled contacts lack downward world-Z force."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._missing_steps = torch.zeros(
      env.num_envs, 5, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._missing_steps[env_ids] = 0

  @staticmethod
  def _downward_force_and_pad_gate(
    sensors: tuple[ContactSensor, ...],
    force_threshold: float,
    min_contact_pads: int,
  ) -> torch.Tensor:
    vectors_parts: list[torch.Tensor] = []
    found_parts: list[torch.Tensor] = []
    history_parts: list[torch.Tensor] = []
    for sensor in sensors:
      force = sensor.data.force
      found = sensor.data.found
      if force is None or found is None:
        raise RuntimeError(
          f"Sensor {sensor.cfg.name!r} must provide found and force"
        )
      vectors_parts.append(force.reshape(force.shape[0], -1, 3))
      found_parts.append(found.reshape(found.shape[0], -1))
      history = sensor.data.force_history
      if history is not None:
        if history.ndim == 3:
          history = history.unsqueeze(2)
        history_parts.append(
          history.reshape(history.shape[0], history.shape[1], -1, 3)
        )
    vectors = torch.cat(vectors_parts, dim=1)
    found = torch.cat(found_parts, dim=1)
    downward_force = (-vectors.sum(dim=1)[..., 2]).clamp_min(0.0)
    found_count = found.reshape(found.shape[0], -1).sum(dim=-1)
    current = (downward_force >= float(force_threshold)) & (
      found_count >= int(min_contact_pads)
    )

    if not history_parts:
      return current
    history_vectors = torch.cat(history_parts, dim=1)
    history_downward_force = (
      -history_vectors.sum(dim=2)[..., 2]
    ).clamp_min(0.0)
    history_pad_count = (
      torch.linalg.vector_norm(history_vectors, dim=-1) > 1.0e-6
    ).sum(dim=-1)
    recent = (
      (history_downward_force >= float(force_threshold))
      & (history_pad_count >= int(min_contact_pads))
    ).any(dim=1)
    return current | recent

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_missing_steps: int = 10,
    force_threshold: float = 30.0,
    handle_min_contact_pads: int = 3,
  ) -> torch.Tensor:
    if max_missing_steps <= 0:
      raise ValueError("max_missing_steps must be positive")
    if force_threshold < 0.0:
      raise ValueError("force_threshold must be non-negative")
    if handle_min_contact_pads <= 0:
      raise ValueError("handle_min_contact_pads must be positive")

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    pair_keys = (
      "handle_left",
      "handle_right",
      "saddle",
      "pedal_left",
      "pedal_right",
    )
    scheduled = command.ref_contact_mask[
      :, [BICYCLE_CONTACT_PAIR_KEYS.index(key) for key in pair_keys]
    ].bool()
    contacts = torch.stack(
      [
        self._downward_force_and_pad_gate(
          tuple(
            env.scene[name]
            for name in BICYCLE_CONTACT_SENSOR_NAMES_BY_PAIR.get(
              key,
              (
                next(
                  pair.sensor_name
                  for pair in BICYCLE_CONTACT_PAIRS
                  if pair.key == key
                ),
              ),
            )
          ),
          force_threshold,
          handle_min_contact_pads if key.startswith("handle") else 1,
        )
        for key in pair_keys
      ],
      dim=-1,
    )
    missing = scheduled & ~contacts
    self._missing_steps = torch.where(
      missing, self._missing_steps + 1, torch.zeros_like(self._missing_steps)
    )
    return (self._missing_steps >= int(max_missing_steps)).any(dim=-1)


class BicyclePedalDistanceMissingTermination:
  """Terminate when either scheduled sole point stays far from its pedal.

  This deliberately ignores contact sensors.  Distance is measured between
  the same mid-sole point used by the pedal contact reward and the designated
  interaction point on the *simulated*, rotating left/right pedal.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._far_steps = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._far_steps[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_distance: float = 0.10,
    max_far_steps: int = 10,
  ) -> torch.Tensor:
    if max_distance <= 0.0:
      raise ValueError("max_distance must be positive")
    if max_far_steps <= 0:
      raise ValueError("max_far_steps must be positive")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    if command.bicycle is None:
      raise RuntimeError(f"{command_name!r} has no bicycle entity")

    distances: list[torch.Tensor] = []
    pair_indices: list[int] = []
    sole_local = torch.tensor(
      FOOT_SOLE_CONTACT_LOCAL,
      device=env.device,
      dtype=command.robot_body_pos_w.dtype,
    )
    for key in ("pedal_left", "pedal_right"):
      pair_idx = BICYCLE_CONTACT_PAIR_KEYS.index(key)
      pair = BICYCLE_CONTACT_PAIRS[pair_idx]
      pair_indices.append(pair_idx)

      foot_pos = command.robot_body_pos_by_name(pair.ee_body)
      foot_quat = command.robot_body_quat_by_name(pair.ee_body)
      sole_point = foot_pos + quat_apply(
        foot_quat, sole_local.expand(foot_pos.shape[0], -1)
      )

      bike_idx = command._sim_bike_index(pair.bike_body)
      pedal_pos = command.sim_bike_body_pos_w[:, bike_idx]
      pedal_quat = command.sim_bike_body_quat_w[:, bike_idx]
      if pair.target_local_pos is not None:
        pedal_local = torch.tensor(
          pair.target_local_pos,
          device=env.device,
          dtype=pedal_pos.dtype,
        )
        pedal_pos = pedal_pos + quat_apply(
          pedal_quat, pedal_local.expand(pedal_pos.shape[0], -1)
        )
      distances.append(torch.norm(sole_point - pedal_pos, dim=-1))

    scheduled = command.ref_contact_mask[:, pair_indices].bool()
    far = scheduled & (torch.stack(distances, dim=-1) > float(max_distance))
    self._far_steps = torch.where(
      far, self._far_steps + 1, torch.zeros_like(self._far_steps)
    )
    return (self._far_steps >= int(max_far_steps)).any(dim=-1)


# (robot_ee_body, robot_local_point, bike_body, bike_local_point,
#  dr_offset_attr). dr_offset_attr names the per-env (B, 3) offset already
# sampled elsewhere (geometry DR) to add to the bike-side point; None for
# pairs with no such offset (pedals).
_MARKER_PAIR_KEYS: tuple[str, ...] = (
  "handle_left",
  "handle_right",
  "saddle",
  "pedal_left",
  "pedal_right",
)
_MARKER_ROBOT_LOCAL: dict[str, tuple[float, float, float]] = {
  "handle_left": HAND_CREASE_L_LOCAL,
  "handle_right": HAND_CREASE_R_LOCAL,
  "saddle": PELVIS_BOTTOM_POLE_LOCAL,
  "pedal_left": SOLE_BOTTOM_CENTROID_L_LOCAL,
  "pedal_right": SOLE_BOTTOM_CENTROID_R_LOCAL,
}
_MARKER_BIKE_DR_OFFSET_ATTR: dict[str, str | None] = {
  "handle_left": "_bicycle_handle_pos_offset",
  "handle_right": "_bicycle_handle_pos_offset",
  "saddle": "_bicycle_saddle_pos_offset",
  "pedal_left": None,
  "pedal_right": None,
}
# geometry-optimized profiles (init_optimized_bicycle_geometry_profile)
# override the bike-side target at runtime via these env attributes instead
# of BicycleContactPair.target_local_pos (which stays the pre-optimized
# default). Mirrors bicycle_rewards._pair_target_local's override lookup for
# the three pairs it applies to; pedals have no such override.
_MARKER_BIKE_OVERRIDE_ATTR: dict[str, str | None] = {
  "handle_left": "_bicycle_handle_left_contact_local",
  "handle_right": "_bicycle_handle_right_contact_local",
  "saddle": "_bicycle_saddle_contact_local",
  "pedal_left": None,
  "pedal_right": None,
}


def _contact_frames_to_exit(command: MotionCommand) -> torch.Tensor:
  """Frames remaining until each pair's reference contact flag flips.

  Shares the ``_contact_transition_frames`` cache with
  ``bicycle_observations.bicycle_contact_time_to_transition`` (same attribute
  name on the command, built lazily by whichever runs first).
  """
  cache = getattr(command, "_contact_transition_frames", None)
  if cache is None:
    flags = command._ref_contact_mask
    T, N = flags.shape
    cache = torch.zeros(T, N, device=flags.device, dtype=torch.float32)
    for pair_idx in range(N):
      next_change = T - 1
      cache[T - 1, pair_idx] = 0.0
      for frame in range(T - 2, -1, -1):
        if flags[frame, pair_idx] != flags[frame + 1, pair_idx]:
          next_change = frame + 1
        cache[frame, pair_idx] = float(next_change - frame)
    command._contact_transition_frames = cache
  return cache[command.time_steps]


class BicycleMarkerDistanceMissingTermination:
  """Terminate when a fixed robot/bike marker pair drifts apart during its
  scheduled contact window, with a threshold that tightens over training.

  Deliberately bypasses the contact-sensor/collision-surface reward geometry
  entirely: distance is ``||robot_marker_world - bike_marker_world||`` between
  one fixed point per robot body (``HAND_CREASE_*``, ``PELVIS_BOTTOM_POLE``,
  ``SOLE_BOTTOM_CENTROID_*``) and one fixed point per bike body (the existing
  ``OPTIMIZED_*`` geometry targets, offset by the same handle/saddle DR shift
  the reward uses). No RubberBox inner-pad count, no per-foot-strip distance,
  no force sensor.

  Threshold schedule: ``start_threshold`` at iteration ``iteration_offset``,
  tightening by ``step_m`` every ``iters_per_step`` iterations, floored at
  ``final_threshold`` and held there after. ``iteration_offset`` lets a
  resumed/forked run treat its own start as iteration 0 instead of inheriting
  the checkpoint's absolute ``common_step_counter``.

  ``edge_grace_frames`` excludes frames within that many reference frames of
  a schedule transition (checked via ``_contact_frames_to_exit``): the
  retargeted motion's own ``bike_contact_flags`` run a release/dismount ramp
  a bit past the point where the literal marker distance has already grown
  (verified empirically for the current teacher motion file — pedal pairs
  exceed even a 0.30 m threshold for ~15-20 consecutive frames right before
  their contact window closes, and handle pairs exceed 0.10 m for ~20-30
  frames at the same kind of boundary). Without this grace window, a policy
  perfectly tracking the reference would still get terminated.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._far_steps = torch.zeros(
      env.num_envs, len(_MARKER_PAIR_KEYS), device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._far_steps[env_ids] = 0

  @staticmethod
  def _current_threshold(
    env: ManagerBasedRlEnv,
    *,
    start_threshold: float,
    final_threshold: float,
    step_m: float,
    iters_per_step: float,
    iteration_offset: float = 0.0,
  ) -> float:
    iteration = float(env.common_step_counter) / float(STEPS_PER_ITER) - float(
      iteration_offset
    )
    iteration = max(0.0, iteration)
    steps_taken = iteration // iters_per_step
    threshold = start_threshold - step_m * steps_taken
    return max(final_threshold, threshold)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    start_threshold: float = 0.20,
    final_threshold: float = 0.10,
    step_m: float = 0.01,
    iters_per_step: float = 2000.0,
    max_far_steps: int = 10,
    edge_grace_frames: float = 40.0,
    iteration_offset: float = 0.0,
  ) -> torch.Tensor:
    if final_threshold <= 0.0 or start_threshold < final_threshold:
      raise ValueError("start_threshold must be >= final_threshold > 0")
    if max_far_steps <= 0:
      raise ValueError("max_far_steps must be positive")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    if command.bicycle is None:
      raise RuntimeError(f"{command_name!r} has no bicycle entity")

    threshold = self._current_threshold(
      env,
      start_threshold=start_threshold,
      final_threshold=final_threshold,
      step_m=step_m,
      iters_per_step=iters_per_step,
      iteration_offset=iteration_offset,
    )

    distances: list[torch.Tensor] = []
    pair_indices: list[int] = []
    for key in _MARKER_PAIR_KEYS:
      pair_idx = BICYCLE_CONTACT_PAIR_KEYS.index(key)
      pair = BICYCLE_CONTACT_PAIRS[pair_idx]
      pair_indices.append(pair_idx)

      robot_pos = command.robot_body_pos_by_name(pair.ee_body)
      robot_quat = command.robot_body_quat_by_name(pair.ee_body)
      robot_local = torch.tensor(
        _MARKER_ROBOT_LOCAL[key], device=env.device, dtype=robot_pos.dtype
      )
      robot_point = robot_pos + quat_apply(
        robot_quat, robot_local.expand(robot_pos.shape[0], -1)
      )

      bike_idx = command._sim_bike_index(pair.bike_body)
      bike_pos = command.sim_bike_body_pos_w[:, bike_idx]
      bike_quat = command.sim_bike_body_quat_w[:, bike_idx]
      override_attr = _MARKER_BIKE_OVERRIDE_ATTR[key]
      override = getattr(env, override_attr, None) if override_attr is not None else None
      base_local = override if override is not None else pair.target_local_pos
      assert base_local is not None
      bike_local = torch.tensor(
        base_local, device=env.device, dtype=bike_pos.dtype
      ).expand(bike_pos.shape[0], -1)
      dr_attr = _MARKER_BIKE_DR_OFFSET_ATTR[key]
      dr_offset = getattr(env, dr_attr, None) if dr_attr is not None else None
      bike_point = bike_pos + quat_apply(bike_quat, bike_local)
      if dr_offset is not None:
        bike_point = bike_point + dr_offset

      distances.append(torch.norm(robot_point - bike_point, dim=-1))

    scheduled = command.ref_contact_mask[:, pair_indices].bool()
    frames_to_exit = _contact_frames_to_exit(command)[:, pair_indices]
    in_core_window = frames_to_exit >= float(edge_grace_frames)
    far = scheduled & in_core_window & (torch.stack(distances, dim=-1) > threshold)
    self._far_steps = torch.where(
      far, self._far_steps + 1, torch.zeros_like(self._far_steps)
    )
    return (self._far_steps >= int(max_far_steps)).any(dim=-1)


class BicycleFootOffPedalTermination:
  """Terminate for sustained foot distance after a riding-entry grace period."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._far_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._riding_steps = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._far_steps[env_ids] = 0
    self._riding_steps[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_distance: float = 0.08,
    attach_grace_steps: int = 50,
    grace_steps: int = 10,
    require_bilateral_schedule: bool = True,
  ) -> torch.Tensor:
    if max_distance <= 0.0 or attach_grace_steps < 0 or grace_steps <= 0:
      raise ValueError(
        "max_distance and grace_steps must be positive and "
        "attach_grace_steps must be non-negative"
      )
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    if command.bicycle is None:
      raise RuntimeError(f"{command_name!r} has no bicycle entity")

    sole_local = torch.tensor(
      FOOT_SOLE_CONTACT_LOCAL,
      device=env.device,
      dtype=command.robot_body_pos_w.dtype,
    )
    distances: list[torch.Tensor] = []
    for key in ("pedal_left", "pedal_right"):
      pair = BICYCLE_CONTACT_PAIRS[BICYCLE_CONTACT_PAIR_KEYS.index(key)]
      foot_pos = command.robot_body_pos_by_name(pair.ee_body)
      foot_quat = command.robot_body_quat_by_name(pair.ee_body)
      sole_pos = foot_pos + quat_apply(
        foot_quat, sole_local.expand(foot_pos.shape[0], -1)
      )

      pedal_idx = command._sim_bike_index(pair.bike_body)
      pedal_pos = command.sim_bike_body_pos_w[:, pedal_idx]
      pedal_quat = command.sim_bike_body_quat_w[:, pedal_idx]
      center_local = torch.tensor(
        pair.target_local_pos or (0.0, 0.0, 0.0),
        device=env.device,
        dtype=pedal_pos.dtype,
      )
      pedal_center = pedal_pos + quat_apply(
        pedal_quat, center_local.expand(pedal_pos.shape[0], -1)
      )
      distances.append(torch.norm(sole_pos - pedal_center, dim=-1))

    pair_indices = (
      BICYCLE_CONTACT_PAIR_KEYS.index("pedal_left"),
      BICYCLE_CONTACT_PAIR_KEYS.index("pedal_right"),
    )
    scheduled = command.ref_contact_mask[:, pair_indices].bool()
    active_window = (
      scheduled.all(dim=-1)
      if require_bilateral_schedule
      else scheduled.any(dim=-1)
    )
    # Count from this environment's entry into the applicable pedal schedule,
    # including RSI resets directly into the middle of that window.  The first
    # attach_grace_steps control steps are preparation time, not violations.
    self._riding_steps = torch.where(
      active_window,
      self._riding_steps + 1,
      torch.zeros_like(self._riding_steps),
    )
    eligible = active_window & (self._riding_steps > int(attach_grace_steps))
    distance_stack = torch.stack(distances, dim=-1)
    if require_bilateral_schedule:
      relevant_distance = distance_stack.max(dim=-1).values
    else:
      # Ignore an unscheduled foot.  This supports one-foot pedal-touch clips
      # without weakening the original two-foot riding behavior.
      relevant_distance = torch.where(
        scheduled, distance_stack, torch.zeros_like(distance_stack)
      ).max(dim=-1).values
    off = eligible & (relevant_distance > float(max_distance))
    self._far_steps = torch.where(
      off, self._far_steps + 1, torch.zeros_like(self._far_steps)
    )
    return self._far_steps >= int(grace_steps)


def bicycle_abs_roll_exceeded(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_abs_roll: float = math.pi / 4,
) -> torch.Tensor:
  """Terminate when the simulated bicycle frame lean exceeds ``max_abs_roll``.

  ``max_abs_roll`` is a true angle in radians (0 = upright, pi/2 = flat on
  its side). The body-frame gravity Y component (also used by
  ``bicycle_roll_magnitude_penalty_v2`` / ``bicycle_roll_dynamics_tracking_exp``)
  is ``sin(roll)``, not ``roll`` itself -- comparing it directly against a
  radian threshold under-triggers (e.g. comparing against ``pi/4`` would
  actually fire around 51.7 degrees of real tilt, not 45: ``sin(45deg)``
  =0.707 < 0.785 rad). This converts back to the true angle via ``arcsin``
  before comparing.
  """
  if max_abs_roll <= 0.0:
    raise ValueError("max_abs_roll must be positive")
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.bicycle is None:
    raise RuntimeError(f"{command_name!r} has no bicycle entity")
  sim_i = command._sim_bike_index("frame")
  sim_q = command.sim_bike_body_quat_w[:, sim_i]
  gravity = torch.zeros(env.num_envs, 3, device=env.device, dtype=sim_q.dtype)
  gravity[:, 2] = -1.0
  roll_sin = quat_apply_inverse(sim_q, gravity)[:, 1]
  roll_angle = torch.asin(roll_sin.clamp(-1.0, 1.0))
  return roll_angle.abs() > float(max_abs_roll)


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)
