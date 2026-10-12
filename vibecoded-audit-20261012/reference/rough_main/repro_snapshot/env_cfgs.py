"""Pygmalion velocity environment configurations."""

import dataclasses
import math
import os
from pathlib import Path

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots import (
  PYG_ACTION_SCALE,
  PYG_MOTOR_OBS_JOINT_NAMES,
  PYG_OBS_JOINT_NAMES,
  PYG_POSE_JOINT_NAMES,
  get_pygmalion_robot_cfg,
)
from mjlab.asset_zoo.robots.pygmalion.pygmalion_constants import (
  ANKLE_MODE,
  ANKLE_RP_ENVELOPE,
  safe_target_clip,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def pygmalion_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Pygmalion rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # MEMORY (2026-07-03): the EPA/CCD buffer scales as naccdmax(=nconmax*nworld)
  # x ccd_iterations; ccd_iterations=500 + nconmax=150 demanded 37 GB at 8192
  # envs (OOM even at 2048 on 16 GB). Flat trains fine at ccd_iterations=50;
  # use 100 on rough for heightfield robustness headroom.
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 500
  # Pygmalion's full self-collision set plus 10 foot capsules generate ~89
  # contacts per world at init (more than G1's 70), so allocate extra headroom.
  cfg.sim.nconmax = 150

  cfg.scene.entities = {"robot": get_pygmalion_robot_cfg()}

  # Set raycast sensor frame to Pygmalion trunk (base_link).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")
  # Foot collision geoms: pygmalion_v2 has five capsules per foot (L_foot2..6_collision),
  # the v3_printed / loop builds a single box sole (L_foot1_collision) - match by regex so
  # the friction DR never silently targets a geom set that does not exist (2026-08-23).
  geom_names = tuple(rf"{side}_foot\d+_collision" for side in ("L", "R"))

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(L_foot_link|R_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    # 50 Hz policy steps otherwise see only one instantaneous contact sample.
    # Keep four samples (80 ms) for loading-rate/impact diagnostics and rewards.
    history_length=4,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PYG_ACTION_SCALE
  if os.environ.get("PYG_SAFE_TARGET_CLIP"):
    # Clip the processed position target (not the raw policy output) to the same
    # centered 90% mechanical envelope used by Entity.soft_joint_pos_limits.
    # This leaves a stop margin and gives the deployment runtime an exact clamp
    # contract. Previous-action observation intentionally remains the raw action.
    # 2026-09-03: derived PER JOINT from the model instead of a hand-written regex table.
    # The old table had one knee tuple, (-114, -6) deg, correct for a [-120, 0] knee and
    # EMPTY against v30's mirrored [0, +120] left knee - the policy could not command a
    # single reachable left-knee angle for 5,700 iterations. Values are unchanged on every
    # model whose legs share a convention; the mode branch is gone because the derivation
    # already covers exactly the joints this run actuates (cranks in AB, ankles in RP).
    # docs/reward_research/2026-09-03_stiff_knee_root_cause.md
    joint_pos_action.clip = safe_target_clip()

  # PYG_ANKLE_LOOP (2026-08-23): the closed-loop ankle adds passive joints to the tree -
  # per leg the two ankle hinges plus four rod universal hinges. The policy only sees the
  # joints the hardware can report (motors + ankle via the mechanism's FK), and the pose
  # reward shapes the same set minus the cranks (the foot pose is what matters, not the
  # motor angle). In serial mode both tuples are (".*",) and nothing changes.
  for grp in ("actor", "critic"):
    for term in ("joint_pos", "joint_vel"):
      cfg.observations[grp].terms[term].params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=PYG_OBS_JOINT_NAMES)
      }
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=PYG_POSE_JOINT_NAMES
  )

  # Sim-to-real: base linear velocity has no IMU equivalent on hardware (the
  # MuJoCo velocimeter is a sim-only sensor). Drop it from the actor so the
  # deployed policy never depends on it; the critic keeps it as training-only
  # privileged information (asymmetric actor-critic).
  del cfg.observations["actor"].terms["base_lin_vel"]

  # prototype-tempmass Student/Teacher contract (opt-in): the actor receives only
  # deployable command/proprioception.  Joint-position history is chronological
  # [q(t-1), q(t)] and contains exactly the 12 motor encoders; passive loop coordinates,
  # simulated joint velocity, contacts and true base velocity remain critic-only.
  # "Teacher" here is PPO's privileged critic, not a second action-producing policy.
  if os.environ.get("PYG_STUDENT_TEACHER"):
    motor_cfg = SceneEntityCfg("robot", joint_names=PYG_MOTOR_OBS_JOINT_NAMES)
    cfg.observations["actor"].terms = {
      "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=cfg.observations["critic"].terms["base_ang_vel"].noise,
      ),
      "projected_gravity": ObservationTermCfg(
        # Deployable path: EBIMU attitude -> gravity vector. The MJCF framezaxis
        # sensor expresses world +Z in the IMU site frame; negate it to gravity.
        # The privileged critic may keep the exact root-orientation value.
        func=envs_mdp.projected_gravity_from_sensor,
        params={"sensor_name": "robot/imu_upvector"},
        noise=cfg.observations["critic"].terms["projected_gravity"].noise,
      ),
      "motor_pos_history": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": motor_cfg},
        noise=cfg.observations["critic"].terms["joint_pos"].noise,
        history_length=2,
        flatten_history_dim=True,
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
      "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
      ),
    }

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0

  # PYG_REL_HEADING (2026-09-13): what fraction of the environments are told WHERE TO FACE
  # rather than HOW FAST TO TURN.  The default 0.3 leaves 70% of training with a turn-rate
  # command only, and a turn rate is the derivative of heading - so a small steady bias
  # costs almost nothing per step and integrates freely.  Measured on the finished v4
  # policy: commanded dead straight, it turns 12 to 150 degrees over 15 s, and its hip yaw
  # spends nearly as much motion twisting the pelvis (5.32 deg) as stepping (6.02 deg).
  # The two reward terms that could penalise that collect 0.8% of the score between them,
  # and raising a penalty is the reward-hacking road we already rejected.  Raising this
  # fraction instead closes the loop the policy is missing: drift, and the heading
  # controller hands back a turn command that says so (velocity_command.py _update_command).
  # Evidence: docs/reward_research/2026-09-12_limp_root_cause_mirror_test.md §3d.
  # PYG_REL_STANDING (2026-09-14): what fraction of the environments are commanded to
  # stand still.  The default 0.1 is the only place the tight std_standing pose tolerance
  # is ever exercised, and it is not enough: measured on v5, the standing-pose term sits at
  # exp(-8.7) = 1.7e-4 of a possible 1.0 - a weight-1.0 term earning nothing, with a
  # gradient of the same size, so the policy drops it and stands with its toes splayed
  # 12-15 degrees out.  The target pose is not the problem: it is statically stable with a
  # 100 mm margin, the same as the pose the policy actually adopts.  What is missing is
  # practice.  Raising this trains the zero-command case three times as often, and the terms
  # that DO have a healthy gradient there - linear and angular velocity tracking, both near
  # 0.95 while the robot drifts and marches - get the extra samples.
  # Deliberately NOT paired with raising walking_threshold: applying std_standing to
  # commands below 0.5 would drop the pose reward in that band from 0.70 to 0.0034, i.e.
  # spread the dead zone instead of curing it (measured 2026-09-14).
  # Evidence: docs/experiments/2026-09-13_legonly_ab_v5_heading.md section 13-1.
  _rel_standing = os.environ.get("PYG_REL_STANDING", "")
  if _rel_standing:
    v = float(_rel_standing)
    assert 0.0 <= v <= 1.0, f"PYG_REL_STANDING must be in [0, 1], got {v}"
    twist_cmd.rel_standing_envs = v

  _rel_heading = os.environ.get("PYG_REL_HEADING", "")
  if _rel_heading:
    v = float(_rel_heading)
    assert 0.0 <= v <= 1.0, f"PYG_REL_HEADING must be in [0, 1], got {v}"
    twist_cmd.rel_heading_envs = v

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  # Upper-body values reuse the established G1 baseline. PYG_UPPER_DOF expands the pose
  # target from 12 to 17 joints, so omitting these entries leaves variable_posture with a
  # shorter std vector than its joint-error vector. Research: 2026-09-01_upper_dof_preflight.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    # 2026-07-12 Gen-2: 0.15 -> 0.4. Chronic lateral (vy) tracking 31-42% was
    # caused by pose suppressing hip abduction (used 13% of range, no torque
    # saturation); progress reward alone did not fix it (measured at P1 final).
    # Same mechanism as the 2026-07-06 knee stiff-gait fix. Research:
    # 2026-07-11_lateral_hiproll_pose_suppression.md + 2026-07-12_gen2_bundle.md.
    r".*hip_roll.*": 0.4,
    r".*hip_yaw.*": 0.15,
    # 2026-07-06 (user): default is now straight knee (HOME). Loosen the WALKING
    # knee tolerance 0.35 -> 1.2 rad so the pose reward doesn't penalize the ~60deg
    # swing flexion needed for a real gait cycle (0.35 forced a stiff-legged walk,
    # knee only -12deg). std_standing (0.05) still pins the knee straight at rest.
    # Research: 2026-07-06_straight_knee_stiff_gait.md.
    r".*knee.*": 1.2,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 1.5,  # was 0.6; straight-knee default needs loose walking/running flex
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
  }
  if os.environ.get("PYG_UPPER_DOF"):
    cfg.rewards["pose"].params["std_walking"].update(
      {
        r".*waist_yaw.*": 0.2,
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
      }
    )
    cfg.rewards["pose"].params["std_running"].update(
      {
        r".*waist_yaw.*": 0.3,
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
      }
    )

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  # feet_air_time (2026-07-06, user): enable swing-air-time reward to structure a
  # proper gait cycle. The command-gated g1 stack alone left a high-double-support
  # (~49%) shuffle with no clear swing -> reads as foot vibration / no gait cycle.
  # air_time rewards each foot's swing time in [0.05, 0.5]s, command-gated
  # (threshold 0.5 -> standing preserved) and cadence-free (tracking unimpeded).
  # Research: 2026-07-06_gait_cycle_air_time.md.
  cfg.rewards["air_time"].weight = 1.0

  # High-speed FREEZE fix (2026-07-10, user): the exp track_linear_velocity reward
  # (std ~0.71) has a vanishing gradient once the speed error exceeds ~1 m/s, so
  # under the 2.5-doctrine curriculum the policy froze at cmd>=2.0 (achieved
  # 0.03-0.10 m/s) despite 50%/30% torque/speed HW headroom — standing minimizes
  # effort/fall-risk when the tracking reward is hopeless. Add a LINEAR progress
  # reward (velocity projected on the command dir, capped at |cmd|) that always
  # pulls toward faster, so the policy accelerates to its physical limit; and widen
  # the exp std 0.5->0.75 to soften the mid-speed dead zone. Research:
  # docs/reward_research/2026-07-10_highspeed_freeze_progress_reward.md
  cfg.rewards["track_lin_vel_progress"] = RewardTermCfg(
    func=mdp.track_lin_vel_progress,
    weight=1.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.75)

  # Gen-2 (2026-07-12): command-gated standing penalty — mid-speed STALL
  # (walk-then-stop at cmd 0.7, stop-and-go at 1.6) persisted through DR/push
  # training, so it is a reward-structure defect, not robustness. Standing
  # while commanded to move now pays a flat cost.
  # Research: docs/reward_research/2026-07-12_gen2_bundle.md (+ midspeed_stall).
  cfg.rewards["stand_still_penalty"] = RewardTermCfg(
    func=mdp.stand_still_penalty,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  # P1 (2026-08-27, opt-in): PYG_LOWSPEED_DEADBAND=<value> overrides
  # stand_still_penalty.cmd_deadband (function default 0.3). The fc 121-command
  # grid measured 0 % achievement at cmd 0.25 m/s on BOTH ankleAB and ankleRP:
  # at 0.25 the term is structurally OFF (0.25 < 0.3) and air_time is OFF too
  # (0.25 < 0.5), so 0.05-0.3 m/s is a blind spot that pays every one-way cost
  # of walking (clearance/swing/slip/soft_landing, all gated at 0.05) with only
  # +0.34/step of tracking margin over standing — action_rate_l2 alone reverses
  # it. 0.3 was calibrated for the 0.7 m/s stall (2026-07-12) and 1.5-2.5 creep
  # (2026-07-13), never for <0.3. Setting 0.05 adopts the codebase convention
  # (7 other command gates already use it) and widens the walking margin to
  # >= +1.34/step. rel_floor stays 0.3 (a separate, anti-creep axis).
  # Unset = byte-identical to the previous params dict.
  # Research: docs/reward_research/2026-08-27_low_speed_command_research.md §5 P1
  _lowspeed_db = os.environ.get("PYG_LOWSPEED_DEADBAND")
  if _lowspeed_db:
    cfg.rewards["stand_still_penalty"].params["cmd_deadband"] = float(_lowspeed_db)
  # Gen-2 (2026-07-12): knee realizable-speed shaping — real RS04 no-load is
  # 19.9 rad/s (measured), sim enforces none; swing retraction hit 26.7.
  cfg.rewards["knee_overspeed"] = RewardTermCfg(
    func=mdp.knee_overspeed,
    weight=-0.5,
    params={"speed_limit": 19.9},
  )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Torque-limit penalty: penalizes commanded torque past 0.7 x effort_limit
  # per actuator, keeping the policy inside the motors' rated envelope.
  cfg.rewards["torque_limit"] = RewardTermCfg(
    func=mdp.torque_limit,
    # weight=-0.5,
    weight=-0.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "soft_factor": 0.7,
    },
  )

  # B1 impact cap (2026-07-03): bound GRF spikes. Fires only above ~1.2x BW per
  # foot (normal support untouched); real-robot-validated shape (Humanoid-Gym /
  # REEM-C). Research: docs/reward_research/2026-07-02_gait_research_q123.md.
  # Gate: GRF P99 down >=20% vs A1b at matched iters AND air_time >= 0.2 s.
  # Escalation 1 (2026-07-03): -0.005/clip400 plateaued at GRF P99 -19% (gate -20%,
  # 2 consecutive borderline misses) and PEAK worsened 4.4->5.2xBW because clip=400
  # zeroed the gradient above ~1000 N. Double the weight and raise the clip so the
  # largest spikes (2.6 kN ~ HW break limit) stay penalized. air_time 0.19 (no shuffle).
  cfg.rewards["contact_force_cap"] = RewardTermCfg(
    func=mdp.contact_force_cap,
    weight=-0.01,
    params={"sensor_name": "feet_ground_contact", "threshold": 600.0, "clip": 800.0},
  )

  # PYG_SOFT_LANDING=1 (2026-08-24, user "발을 너무 팍팍 내딛는다"): the iter-2500 AB/RP
  # policies hit the floor at 1.3-1.5 m/s with loading rates ~150 BW/s (human walking
  # 0.1-0.4 m/s, 10-20 BW/s). Nothing in the stack penalised the DESCENT: soft_landing is
  # -1e-5 and contact_force_cap's 600 N threshold was sized for the 505 N robot (1.2 BW)
  # - on the 347 N printed robot it is 1.73 BW and barely fires. Two changes, applied to
  # both A/B arms: a dense downward-foot-speed penalty near the ground, and the cap
  # rescaled to 1.2 BW. docs/reward_research/2026-08-24_soft_landing_impact.md
  if os.environ.get("PYG_SOFT_LANDING"):
    cfg.rewards["foot_impact_velocity"] = RewardTermCfg(
      func=mdp.foot_impact_velocity,
      weight=float(os.environ.get("PYG_SOFT_LANDING_W", "-1.0")),
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        "height_threshold": float(os.environ.get("PYG_SOFT_LANDING_H", "0.10")),
        "power": float(os.environ.get("PYG_SOFT_LANDING_P", "2.0")),
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    )
    cfg.rewards["contact_force_cap"].params["threshold"] = 420.0  # 1.2 x 347 N
    cfg.rewards["contact_force_cap"].params["clip"] = 560.0

  # B2 thermal redistribution (2026-07-03): sum (tau/rated)^2 over joints. Squared
  # normalized duty makes ratio-equalizing optimal -> load migrates from the
  # thermally-over-rated knee (~106% raw RMS) to the hips' headroom. Weight sized
  # from b1w2_3000 replay (mean cost ~5.7 -> ~ -0.11/step vs +2 tracking).
  # Gate: knee RMS <100% rated AND |vx err| within 10% of B1w2 at matched iters.
  cfg.rewards["thermal_effort"] = RewardTermCfg(
    func=mdp.thermal_effort,
    weight=-0.02,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "rated": {
        ".*_hip_pitch_joint": 40.0,
        ".*_hip_roll_joint": 40.0,
        ".*_hip_yaw_joint": 20.0,
        ".*_knee_joint": 40.0,
        ".*_ankle_pitch_joint": 20.0,
        ".*_ankle_roll_joint": 5.0,
        # Upper body (PYG_UPPER_DOF): waist is RS04, shoulders are RS03. Without
        # explicit matches thermal_effort assumes 1 N*m, over-penalising these motors by
        # 20x/40x in torque and 400x/1600x after squaring.
        ".*_waist_yaw_joint": 40.0,
        ".*_shoulder_(pitch|roll)_joint": 20.0,
        # loop ankle: the RS03 rated torque at the crank it sits on. Unmatched joints
        # default to rated 1.0, which would turn a 30 N*m crank into a cost of 900.
        ".*_crank_[AB]_joint": 20.0,
      },
    },
  )
  if ANKLE_MODE == "RP":
    # both ankle motors are RS03 through the loop: rated at the joint = the centre-pose
    # torque extent scaled by rated/peak (ankle_rp_envelope.json), 32.7 pitch / 27.9 roll,
    # instead of the old RS03+RS00 serial guess (20 / 5).
    import json as _j

    _env = _j.load(open(ANKLE_RP_ENVELOPE))
    _g = _env["grid"]
    _i0 = _g["pitch_deg"].index(0.0)
    _j0 = _g["roll_deg"].index(0.0)
    _ext = _env["legs"]["L"]["tau_extent"][_i0][_j0]
    _k = _env["motor"]["rated"] / _env["motor"]["peak"]
    cfg.rewards["thermal_effort"].params["rated"][".*_ankle_pitch_joint"] = round(
      _ext[0] * _k, 2
    )
    cfg.rewards["thermal_effort"].params["rated"][".*_ankle_roll_joint"] = round(
      _ext[1] * _k, 2
    )

  # Command DR curriculum (2026-07-06, user): ramp vx firmly to 2.5 by iter 10000.
  # Capped at 2.5 (was 3.0) — 3.0 is beyond what a 51.5 kg biped reliably reaches,
  # so the tracking reward could never converge chasing it; 2.5 is an achievable
  # target -> tracking converges AND 2.5 is fully in-distribution for load measure.
  # G1-like modest command curriculum (2026-07-06, user): omnidirectional ~1.5 m/s
  # + rotation only. Cap vx at 1.5 (not 2.5) and yaw at 1.0 -> learn a clean natural
  # walk at G1-class speeds. Gentle ramp, reaching full range by ~iter 6000, held
  # thereafter. vy stays at the base +-1.0 (omnidirectional). Command range is FIXED
  # across both phases; only DR ramps in phase 2.
  # 2026-07-10 (user): return to the 2.5 m/s sizing doctrine with WIDER stage
  # gaps (4000 iters) and a longer phase-1 (20k) so each stage converges before
  # the next ramp. Backward capped at 2.0 (matches worst-case corners).
  # Research: docs/reward_research/2026-07-10_vx25_curriculum.md
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.8, 0.8), "ang_vel_z": (-0.5, 0.5)},
      {"step": 4000 * 24, "lin_vel_x": (-1.2, 1.2), "ang_vel_z": (-0.8, 0.8)},
      {"step": 8000 * 24, "lin_vel_x": (-1.6, 1.6), "ang_vel_z": (-1.0, 1.0)},
      {"step": 12000 * 24, "lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.0, 1.0)},
      {"step": 16000 * 24, "lin_vel_x": (-2.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
    ]
    # 2026-08-24 (opt-in, default OFF): ramp lin_vel_y with the same schedule.
    # Survey (docs/research_raw/2026-08-24_command_box_lateral_survey.md): EVERY
    # biped-tuned config uses vy 0 / +-0.3 / +-0.5 / +-0.6 (IsaacLab H1 / humanoid-gym /
    # IsaacLab G1 flat / Cassie-Digit walking); +-1.0 is the untuned quadruped-lineage
    # default, and no framework ramps vy at all. We KEEP the final +-1.0 because the
    # frontal-plane load envelope is the deliverable (gate "vy +-1.0 >= 60%",
    # docs/reward_research/2026-07-12_gen2_bundle.md) - only the early stages narrow,
    # so the corner |(vx,vy)| is reachable while the policy is still learning.
    # Research: docs/reward_research/2026-08-24_command_box_and_transient.md
    if os.environ.get("PYG_CMD_VY_STAGES", "0") != "0":
      for stage, vy in zip(
        cfg.curriculum["command_vel"].params["velocity_stages"],
        [0.5, 0.6, 0.7, 0.85, 1.0],
      ):
        stage["lin_vel_y"] = (-vy, vy)

  # V3 (docs/103 §3a): PYG_GATED_CURRICULUM=1 (opt-in, default OFF) swaps the FIXED
  # 4,000-iter stage spacing above for promotion by report card. Measured adaptation to a
  # widened command box is 4-20 iters, so the fixed schedule burns ~3,980 iters per stage,
  # four times over. The gate holds a stage until the trailing-window error_vel_xy_steady
  # is back within err_ratio of the previous stage's SETTLED value and fell_over is under
  # fell_max, with a minimum dwell so it cannot skip on a lucky window - and a MAX-DWELL
  # forced promotion so a gate that never fires cannot stall the run (docs/103 §6 risk).
  # The stage TABLE is unchanged (same velocity_stages, same PYG_CMD_VY_STAGES rungs);
  # only the trigger changes, and the term keeps the name "command_vel" so
  # Curriculum/command_vel/lin_vel_x_max stays where every dashboard expects it.
  if "command_vel" in cfg.curriculum and os.environ.get("PYG_GATED_CURRICULUM"):
    _cv = cfg.curriculum["command_vel"]
    _cv.func = mdp.commands_vel_gated
    _cv.params.update(
      {
        "min_dwell_iters": int(os.environ.get("PYG_GATE_MIN_DWELL", "800")),
        "max_dwell_iters": int(os.environ.get("PYG_GATE_MAX_DWELL", "3000")),
        "window_iters": int(os.environ.get("PYG_GATE_WINDOW", "100")),
        "err_ratio": float(os.environ.get("PYG_GATE_ERR_RATIO", "1.10")),
        "fell_max": float(os.environ.get("PYG_GATE_FELL_MAX", "0.005")),
        "min_episodes": int(os.environ.get("PYG_GATE_MIN_EPISODES", "64")),
        "steps_per_iter": int(os.environ.get("PYG_GATE_STEPS_PER_ITER", "24")),
      }
    )
    # PYG_GATE_EXCLUDE_STANDING=1 (2026-09-16, opt-in, default OFF): score the gate on
    # the environments that were given a non-zero command only. An environment told to
    # stand still reports ~0 tracking error once the robot holds position, so with
    # PYG_REL_STANDING raised the early stages look better than they are: legonly_ab_v6b
    # (standing 0.3) settled stage 3 at 0.1679 where the same family at standing 0.1
    # settled at 0.2131, and its top stage then never cleared baseline x 1.10 - it ran
    # 10,600 iterations on a bar its own easier stages had set. Default stays OFF so the
    # gate numbers of every run up to v6b stay comparable.
    # docs/experiments/2026-09-14_legonly_ab_v6b_standing_flatfoot.md §2c-0
    if os.environ.get("PYG_GATE_EXCLUDE_STANDING", "0") not in ("", "0"):
      _cv.params["metric"] = "error_vel_xy_steady_moving"

  # PYG_SOFT_LANDING_MODE=rate (2026-08-26, opt-in): swap the touchdown-SPEED penalty for a
  # LOADING-RATE one. Penalising speed has one cheapest answer - land on an already-bent leg -
  # which is what c3 learned (docs/104). Humans land near-straight and keep dF/dt low instead.
  # "replace" removes foot_impact_velocity; "both" keeps it at half weight.
  # WATCH air_time / flight_frac / stride: dF/dt is zero if the foot never leaves the ground.
  # Research: docs/reward_research/2026-08-26_human_landing_bundle.md
  # "half" = HALVE foot_impact_velocity, no loading-rate term. This is what bundleB2 / bundleD1
  # actually ran: the loading-rate term was dead on a sign bug (s12 of the research note), so the
  # only live variable in those runs was the impact-velocity weight. It gets its own name so the
  # validated recipe can be reproduced without dragging in a term that has never been exercised.
  _slm = os.environ.get("PYG_SOFT_LANDING_MODE", "")
  if _slm in ("rate", "both", "half"):
    if _slm != "half":
      cfg.rewards["foot_loading_rate"] = RewardTermCfg(
        func=mdp.foot_loading_rate,
        # WEIGHT: 0.002 was chosen while the term was dead, so it was never calibrated. Measured
        # live on bundleB2 model_32798 it contributes -0.664 per step - eight times the whole
        # episode reward budget (~0.086/step). 1e-4 puts it at ~-0.033, the same share as
        # stance_knee_extension. Still UNTESTED in training: watch it in the first 100 iters.
        weight=-float(os.environ.get("PYG_LOADING_RATE_W", "0.0001")),
        params={
          "sensor_name": "feet_ground_contact",
          "body_weight": 346.8,
          "window_s": 0.06,
          "command_name": "twist",
          "command_threshold": 0.05,
        },
      )
    if _slm == "rate":
      cfg.rewards.pop("foot_impact_velocity", None)
    else:
      cfg.rewards["foot_impact_velocity"].weight *= 0.5

  # PYG_KNEE_EXT=1 (2026-08-26, opt-in): penalise stance-knee flexion past a target.
  # The height anchor turned out to be inert - the robot already walks at 93 % of straight-leg
  # height while crouching, because hip pitch compensates - so the knee needs a direct term.
  # Precedent: arXiv:2505.20619 "straight knee" reward, weight 0.1.
  # Research: docs/reward_research/2026-08-26_human_landing_bundle.md
  if os.environ.get("PYG_KNEE_EXT"):
    cfg.rewards["stance_knee_extension"] = RewardTermCfg(
      func=mdp.stance_knee_extension,
      weight=-float(os.environ.get("PYG_KNEE_EXT_W", "2.0")),
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
        "target_deg": float(os.environ.get("PYG_KNEE_EXT_DEG", "25.0")),
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    )

  # PYG_STAND_L1=<weight> (2026-09-25, opt-in, default OFF): an L1 pull back to the default
  # pose while the command is ~zero. The exp pose term's 0.05 rad standing tolerance goes
  # dead far from the default (v5: gradient 1.7e-4); L1 keeps a constant pull at any
  # distance. The point is to hold standing quality WITHOUT raising the standing share,
  # which is what caused the promotion stall and v6b's walking regression.
  # Asimov-1 uses weight 1.0 (as -1) over 23 joints at a 0.1 command gate.
  # Research: docs/reward_research/2026-09-25_isaac_asimov_teardown.md
  _sl1 = os.environ.get("PYG_STAND_L1", "")
  if _sl1:
    cfg.rewards["standing_joint_deviation"] = RewardTermCfg(
      func=mdp.standing_joint_deviation_l1,
      weight=-abs(float(_sl1)),
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=PYG_POSE_JOINT_NAMES),
        "command_name": "twist",
        "command_threshold": float(os.environ.get("PYG_STAND_L1_CMD", "0.1")),
      },
    )

  # PYG_BASE_HEIGHT_ANCHOR=1 (2026-08-26, opt-in, default OFF): the height anchor mjlab
  # never inherited. Quadratic only outside +-deadband so a vault is not suppressed.
  # Target 0.87 m = the height the c3 policies actually walk at (0.861-0.874), which is a
  # circular choice - the test sweeps 0.87 vs 0.90. Weight aims at a 1-3 % contribution.
  # Research: docs/reward_research/2026-08-26_human_landing_bundle.md
  if os.environ.get("PYG_BASE_HEIGHT_ANCHOR"):
    cfg.rewards["base_height"] = RewardTermCfg(
      func=mdp.base_height_deadband,
      weight=-float(os.environ.get("PYG_BASE_HEIGHT_W", "5.0")),
      params={
        "target": float(os.environ.get("PYG_BASE_HEIGHT_REF", "0.87")),
        "deadband": float(os.environ.get("PYG_BASE_HEIGHT_DB", "0.03")),
      },
    )

  # External-force DR (2026-07-06, user "DR에 외력도 넣어"): push_robot is already
  # active in training (interval velocity kick) but the base ±0.5 m/s is mild.
  # Strengthen to ±0.7 fore-aft/lateral so the policy learns real push recovery.
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.7, 0.7),
      "y": (-0.7, 0.7),
      "z": (-0.4, 0.4),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    }

  # Reset-pose randomisation (2026-09-16, user: "초기자세 틀려도 잘 버티도록").
  #
  # Until now the reset put every joint at EXACTLY the keyframe (`position_range` is
  # (0.0, 0.0) in velocity_env_cfg) and the base upright, so the policy only ever saw one
  # starting pose. A robot lifted, set down by hand, or recovering from a nudge never
  # starts there, and a policy that has never been anywhere else has no idea what to do.
  #
  # PYG_RESET_JOINT_NOISE [rad]  every joint starts uniformly +-this off the keyframe.
  # PYG_RESET_TILT       [rad]  the base starts rolled/pitched uniformly +-this.
  #
  # Both are OFF unless set, so every existing run reproduces byte-for-byte.
  _rj = os.environ.get("PYG_RESET_JOINT_NOISE", "")
  if _rj and "reset_robot_joints" in cfg.events:
    v = float(_rj)
    assert 0.0 <= v <= 0.6, f"PYG_RESET_JOINT_NOISE out of range: {v}"
    cfg.events["reset_robot_joints"].params["position_range"] = (-v, v)

  _rt = os.environ.get("PYG_RESET_TILT", "")
  if _rt and "reset_base" in cfg.events:
    v = float(_rt)
    assert 0.0 <= v <= 0.6, f"PYG_RESET_TILT out of range: {v}"
    pr = cfg.events["reset_base"].params["pose_range"]
    pr["roll"] = (-v, v)
    pr["pitch"] = (-v, v)

  # Two-phase curriculum (2026-07-06, user): PHASE 1 learns clean walking with the
  # velocity curriculum only and NO physics DR (push/friction/encoder/CoM), so the
  # gait isn't fought by disturbances while it forms. PHASE 2 re-enables DR to add
  # robustness on top of the learned gait. Toggle via PYG_NO_DR=1 (phase 1).
  # reset_* + command_vel curriculum are kept in both phases.
  if os.environ.get("PYG_NO_DR"):
    for _ev in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
      cfg.events.pop(_ev, None)
  else:
    # DR as a curriculum (2026-08-23, user): friction / encoder bias / CoM were STARTUP
    # events, sampled once before dr_levels ever ran, so the ramp below only ever reached
    # the push. Re-sample them at every episode reset so the ramped ranges apply (the DR
    # functions take env_ids; friction and the encoder bias need no model recomputation).
    # PYG_DR_STARTUP=1 restores the old once-at-startup sampling.
    if not os.environ.get("PYG_DR_STARTUP"):
      for _ev in ("foot_friction", "encoder_bias", "base_com"):
        if _ev in cfg.events:
          cfg.events[_ev].mode = "reset"
    # PHASE 2: ramp DR strength 0 -> full linearly over GLOBAL iters 10000->20000.
    # common_step_counter is GLOBAL (restored on resume), so a phase-2 resume from
    # the iter-10000 checkpoint starts at step 240000 and ramps to full by 480000.
    # Scales push/friction/encoder/CoM. Logged as dr_factor.
    cfg.curriculum["dr_levels"] = CurriculumTermCfg(
      func=mdp.dr_levels,
      params={
        # DR ramp window in GLOBAL common_step_counter (×24 = num_steps_per_env).
        # Default 20k→32k iters matches the flat pipeline (phase1=20k). Override
        # via PYG_DR_START_ITER/PYG_DR_END_ITER when phase1 length differs, else
        # a shorter phase1 leaves the counter below start_step and DR under-ramps
        # (2026-07-15: uneven2 P1=12k → dr stuck 0.33; fix = 12000/24000).
        "start_step": int(os.environ.get("PYG_DR_START_ITER", 20000)) * 24,
        "end_step": int(os.environ.get("PYG_DR_END_ITER", 32000)) * 24,
        "push_max": {
          "x": (-0.7, 0.7),
          "y": (-0.7, 0.7),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
        "friction_max": (0.3, 1.2),
        "encoder_max": (-0.015, 0.015),
        "com_max": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
      },
    )

    # PYG_DR_STRONG (2026-09-16, user: "DR 커리큘럼으로 더 세게 넣어. 실제 사례조사해봐").
    #
    # 조사한 것과 비교해 우리 범위가 좁습니다. 근거를 하나씩 답니다.
    #
    # * 밀기 세기 — Unitree 가 자기 G1 설정에 `max_push_vel_xy = 1.5` 로 둡니다
    #   (unitree_rl_gym/legged_gym/envs/g1/g1_config.py). 우리는 0.7 로 **절반 이하**.
    #   대신 우리 간격(1~3초)은 이미 센 쪽입니다 — "Learning Sim-to-Real Humanoid
    #   Locomotion in 15 Minutes"(arXiv 2512.01996) 가 1~3초를 Push-Strong,
    #   5~10초를 기본으로 부릅니다. 그래서 **간격은 두고 세기만** 올립니다.
    # * 마찰 — 같은 G1 설정이 `friction_range = [0.1, 1.25]`. 우리 하한 0.3 은
    #   미끄러운 바닥을 아예 안 겪습니다.
    # * 엔코더 치우침 — ★ 이것이 가장 큰 구멍입니다. 우리는 ±0.015 rad(±0.86도)만
    #   흔드는데, **실물에서 잰 영점 어긋남이 최대 9.4도(0.164 rad)** 입니다
    #   (2026-09-15 손 측정, 왼무릎). 현실의 11분의 1 만 학습시키고 있었습니다.
    #   "초기자세가 틀려도 버티게" 하려면 여기가 현실만큼은 넓어야 합니다.
    # * 무게중심 — 근거가 약해 보수적으로만 넓힙니다.
    #
    # 아직 없는 것(조사에서 다들 쓰는데 우리에겐 아예 없음): **행동 지연**과
    # **PD 게인 랜덤**. 위 15분 논문이 둘 다 목록에 올립니다. 이벤트를 새로 만들어야
    # 해서 이번 변경에는 넣지 않았고, 열린 일로 남깁니다.
    if os.environ.get("PYG_DR_STRONG"):
      _p = cfg.curriculum["dr_levels"].params
      _p["push_max"] = {
        "x": (-1.5, 1.5),
        "y": (-1.5, 1.5),
        "z": (-0.6, 0.6),
        "roll": (-0.8, 0.8),
        "pitch": (-0.8, 0.8),
        "yaw": (-1.2, 1.2),
      }
      _p["friction_max"] = (0.1, 1.25)
      _p["encoder_max"] = (-0.10, 0.10)
      _p["com_max"] = {0: (-0.04, 0.04), 1: (-0.04, 0.04), 2: (-0.05, 0.05)}
      if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = dict(_p["push_max"])

    # ------------------------------------------------------------------ 열린 일 #11
    # PYG_ACTION_DELAY / PYG_ACTION_NOISE / PYG_PD_GAIN_DR (2026-09-17).
    #
    # 조사한 실물 구현들이 전부 쓰는데 우리에겐 **아예 없던** 두 가지입니다.
    # 왜 필요한지와 구현 근거는 delayed_action.py 의 설명을 볼 것.
    #
    # * 명령 지연 — 시뮬은 명령이 그 순간 닿지만 실물은 늦습니다. 늦은 명령으로
    #   되먹임을 돌리면 위상이 밀려 진동합니다. 발이 뜨면 실효 이득이 올라가 더
    #   그렇습니다 — 사용자가 본 "공중에 들면 다리가 흔들린다" 와 같은 이야기입니다.
    # * PD 게인 — 실물 게인은 모형값과 다릅니다. 게인이 다르면 같은 정책이 다른
    #   감쇠로 돕니다. 게인을 흔들어 두면 정책이 게인에 덜 매달립니다.
    #
    # 값의 출처 (docs/148):
    #   - humanoid-gym `humanoid_config.py`: action_delay 0.5 스텝, action_noise 0.02
    #   - 문헌 요약: 제어지연 0~2 스텝 / 0~20 ms, Kp 배율 [0.9, 1.1], Kd 배율 [0.5, 1.5]
    #   - 다른 구현: 강성·감쇠 배율 [0.75, 1.25]
    # 우리 제어는 50 Hz 라 1 스텝 = 20 ms 입니다. 실물 경로(파이 -> CAN -> 드라이버)가
    # humanoid-gym 쪽보다 길 이유가 있어 기본값을 1.0 스텝(=20 ms)까지 둡니다.
    _p = cfg.curriculum["dr_levels"].params  # PYG_DR_STRONG 이 꺼져 있을 수도 있다
    _ad = float(os.environ.get("PYG_ACTION_DELAY", 0.0) or 0.0)
    _an = float(os.environ.get("PYG_ACTION_NOISE", 0.0) or 0.0)
    if _ad > 0.0 or _an > 0.0:
      from mjlab.tasks.velocity.config.pygmalion.delayed_action import (
        DelayedJointPositionActionCfg,
      )

      _old = cfg.actions["joint_pos"]
      # 기존 설정(배율·오프셋·클립·대상관절)을 **하나도 잃지 않고** 그대로 옮긴다.
      _fields = {f.name: getattr(_old, f.name) for f in dataclasses.fields(_old)}
      cfg.actions["joint_pos"] = DelayedJointPositionActionCfg(
        **_fields, max_delay=_ad, noise=_an
      )

    # PD 게인 무작위화. mjlab 에 이미 `dr.pd_gains` 가 있으므로 새로 만들지 않습니다
    # (프로젝트 규칙: 새로 짜기 전에 있는 것부터 찾는다).
    _pg = os.environ.get("PYG_PD_GAIN_DR", "")
    if _pg:
      _kp_lo, _kp_hi, _kd_lo, _kd_hi = (
        [float(x) for x in _pg.split(",")] if "," in _pg else [0.8, 1.2, 0.7, 1.3]
      )
      cfg.events["pd_gains"] = EventTermCfg(
        mode="reset",
        func=dr.pd_gains,
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "kp_range": (_kp_lo, _kp_hi),
          "kd_range": (_kd_lo, _kd_hi),
          "operation": "scale",
        },
      )
      # 커리큘럼에도 올려서 0 -> 전체로 서서히 켠다 (다른 DR 과 같은 창).
      _p["pd_gain_max"] = {"kp": (_kp_lo, _kp_hi), "kd": (_kd_lo, _kd_hi)}
    if _ad > 0.0 or _an > 0.0:
      _p["action_delay_max"] = _ad
      _p["action_noise_max"] = _an

  # Inertial DR from the measured mass uncertainty (2026-08-23, docs/90). The ranges are
  # not typed here: tools/robot_model/mass_dr.py propagates every body's mass uncertainty
  # through the same aggregation that builds the robot, widens the result to a documented
  # structural floor (harness, print batch, the serial-ankle stand-in) and writes
  # tools/robot_model/mass_dr.json. One physics-consistent pseudo_inertia term per link
  # type, both sides sharing the ranges. It REPLACES the generic base_com term so the base
  # is not randomised twice. Toggle: PYG_INERTIAL_DR=1 (off by default so running
  # experiments keep their DR exactly).
  if os.environ.get("PYG_INERTIAL_DR") and os.environ.get("PYG_NO_DR"):
    import warnings

    warnings.warn(
      "PYG_INERTIAL_DR is set but PYG_NO_DR=1 (phase 1) removes all physics DR - "
      "the inertial terms are NOT applied in this run",
      stacklevel=2,
    )
  if os.environ.get("PYG_INERTIAL_DR") and not os.environ.get("PYG_NO_DR"):
    # PYG_MASS_DR_JSON overrides the file (e.g. mass_dr_loop.json for PYG_ANKLE_LOOP, whose
    # shin no longer contains the cranks and rods).
    _mass_dr = Path(
      os.environ.get("PYG_MASS_DR_JSON")
      or (MJLAB_SRC_PATH.parents[3] / "tools" / "robot_model" / "mass_dr.json")
    )
    assert _mass_dr.exists(), (
      f"PYG_INERTIAL_DR needs {_mass_dr} (run tools/robot_model/mass_dr.py)"
    )
    import json as _json

    _dr = _json.load(open(_mass_dr))
    cfg.events.pop("base_com", None)
    _inertial_max = {}
    for _link, _rec in _dr["links"].items():
      _p = _rec["mjlab_pseudo_inertia_recommended"]
      _mj = _dr["mjlab_body_names"][_link]
      _names = (_mj,) if _link in ("pelvis", "torso") else (f"L_{_mj}", f"R_{_mj}")
      _term_name = f"inertial_dr_{_link}"
      _ranges = {
        "alpha_range": tuple(_p["alpha_range"]),
        "t1_range": tuple(_p["t1_range"]),
        "t2_range": tuple(_p["t2_range"]),
        "t3_range": tuple(_p["t3_range"]),
        "d_range": tuple(_p["d_range"]),
      }
      _inertial_max[_term_name] = _ranges
      cfg.events[_term_name] = EventTermCfg(
        # CurriculumManager resets before EventManager, so dr_levels first scales
        # these ranges for the current global iteration and this reset then samples
        # exactly that strength. pseudo_inertia reconstructs from default fields,
        # making repeated resets non-accumulating.
        mode="reset",
        func=dr.pseudo_inertia,
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=_names),
          **_ranges,
        },
      )
    # dr_levels was created above. Preserve the immutable full-strength ranges there;
    # event params themselves are mutated every reset and therefore cannot be the source.
    cfg.curriculum["dr_levels"].params["inertial_max"] = _inertial_max

  # V9 (docs/103 §2): PYG_CRITIC_DR_OBS=1 (opt-in, default OFF) shows the CRITIC the
  # domain-randomisation draw its env is living under. Two envs with identical
  # proprioception but 0.3 vs 1.2 foot friction have genuinely different returns, and
  # today the critic has to treat that as noise. These terms go into the critic group
  # ONLY - the actor never sees them, so the deployed policy is byte-identical and the
  # deployment cost is zero. Everything read here is a value some event actually wrote:
  # friction/mass/CoM come straight back off the live per-world model fields, and the
  # push - which mjlab does not store anywhere - is recorded by swapping push_robot for
  # the logging variant of the same function. A channel no active event randomises
  # reports a constant, which is the truth for that run.
  if os.environ.get("PYG_CRITIC_DR_OBS"):
    _critic = cfg.observations["critic"].terms
    _critic["dr_friction"] = ObservationTermCfg(
      func=mdp.dr_foot_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=geom_names)},
    )
    _critic["dr_mass"] = ObservationTermCfg(
      func=mdp.dr_mass_offset,
      params={"asset_cfg": SceneEntityCfg("robot")},
    )
    _critic["dr_com"] = ObservationTermCfg(
      func=mdp.dr_com_offset,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    )
    _critic["dr_push"] = ObservationTermCfg(func=mdp.dr_push_state)
    _critic["dr_factor"] = ObservationTermCfg(func=mdp.dr_factor)
    if "push_robot" in cfg.events:
      cfg.events["push_robot"].func = envs_mdp.push_by_setting_velocity_logged
      cfg.events["pyg_push_tick"] = EventTermCfg(
        mode="step",
        func=envs_mdp.push_age_tracker,
        params={"action": "tick"},
      )
      cfg.events["pyg_push_clear"] = EventTermCfg(
        mode="reset",
        func=envs_mdp.push_age_tracker,
        params={"action": "clear"},
      )

  # Gait shaping (2026-07-05): REVERT from the fixed phase-clock (periodic_contact
  # + gait_clock obs) to the g1-vanilla command-gated stack. The fixed clock ran
  # regardless of command -> forced stepping at v=0 (couldn't stand) AND pinned a
  # fixed cadence that fought variable-speed tracking (track_linear_velocity stuck
  # ~0.32). g1 walks with NO forced timing: gait emerges from command-gated
  # foot_clearance / foot_swing_height / foot_slip / soft_landing (all off at
  # zero command -> standing works). Research: 2026-07-05_periodic_contact_removal.
  # foot_swing_height (-0.25) and foot_clearance (-2.0) keep their base weights
  # (site_names set above); contact_force_cap + thermal_effort stay for GRF/thermal.

  # Anti-kneel: terminate the episode if the torso drops below 0.7 m. Without any
  # base-height constraint the policy converged to a kneeling/crouched gait
  # (normal base ~0.81 m). 2026-07-02 (user request).
  cfg.terminations["low_base"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("robot")},
  )
  # PYG_LOW_BASE_REL (2026-10-09): the 0.7 m above is an ABSOLUTE height. On rough
  # terrain the ground itself moves up and down, so it fires on low tiles and misses a
  # kneel on high ones. Measure from the ground under the base instead (median of the
  # terrain_scan rays within 0.25 m). The flat task removes terrain_scan and falls
  # back to the absolute test (identical on a plane).
  if os.environ.get("PYG_LOW_BASE_REL"):
    cfg.terminations["low_base"] = TerminationTermCfg(
      func=mdp.base_height_above_terrain_below,
      params={"minimum_height": 0.7, "sensor_name": "terrain_scan"},
    )
  # PYG_NAN_TERM (2026-10-10): rough terrain occasionally diverges the physics (NaN in
  # qpos/qvel); rsl_rl then aborts the whole run on the NaN observation (the first rough
  # A/B arm died at iteration 232). End and reset just that env instead - terminations
  # are applied before the observation is computed. Failure, not time-out: never
  # bootstrap from a NaN state.
  if os.environ.get("PYG_NAN_TERM"):
    cfg.terminations["nan_detection"] = TerminationTermCfg(func=envs_mdp.nan_detection)
  # PYG_BLOWUP_TERM (2026-10-10): end envs whose physics left the physical range (a joint
  # > 100 rad/s, base > 10 m/s, or non-finite) - see mdp.physics_blowup. Pair with
  # PYG_REWARD_TERM_CLAMP: the reward of the blow-up step itself is still computed.
  if os.environ.get("PYG_BLOWUP_TERM"):
    cfg.terminations["physics_blowup"] = TerminationTermCfg(func=mdp.physics_blowup)
  # PYG_OBS_NAN_GUARD (2026-10-10): the relaunched arm then died at iteration 658 on a NaN in
  # the CRITIC observation while the physics state was finite (a critic-only sensor term).
  # mjlab's own per-group policy "warn": replace NaN/Inf by 0 for that step and print the
  # term name (so the source shows up in the log) instead of aborting the run.
  if os.environ.get("PYG_OBS_NAN_GUARD"):
    for group in ("actor", "critic"):
      cfg.observations[group].nan_policy = "warn"
      cfg.observations[group].nan_check_per_term = True
  # PYG_STAGNATION (2026-10-09, supervisor feedback): end the episode as a FAILURE
  # (no bootstrap) when the robot makes no progress along a moving command for a few
  # seconds - before, only falls ended an episode, so standing still under a command
  # kept harvesting upright/pose/yaw-tracking reward until the 20 s time-out.
  # Research: docs/reward_research/2026-10-09_stagnation_termination_literature.md
  if os.environ.get("PYG_STAGNATION"):
    cfg.terminations["stagnation"] = TerminationTermCfg(
      func=mdp.stagnation,
      params={
        "command_name": "twist",
        "window_s": float(os.environ.get("PYG_STAGNATION_WINDOW_S", "3.0")),
        "min_cmd_speed": float(os.environ.get("PYG_STAGNATION_MIN_CMD", "0.2")),
        "min_progress_m": float(os.environ.get("PYG_STAGNATION_MIN_M", "0.10")),
        "min_progress_frac": float(os.environ.get("PYG_STAGNATION_FRAC", "0.2")),
      },
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def pygmalion_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Pygmalion flat terrain velocity configuration."""
  cfg = pygmalion_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop(
    "height_scan", None
  )  # base already excludes it (asymmetric)
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  # No terrain_scan on a plane: the relative low-base test falls back to absolute.
  if cfg.terminations["low_base"].func is mdp.base_height_above_terrain_below:
    cfg.terminations["low_base"] = TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
