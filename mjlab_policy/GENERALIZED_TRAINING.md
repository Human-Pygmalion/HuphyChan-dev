# Generalized HUPHY walking: training and deployment handoff

## Included checkpoints

Only the latest checkpoint captured from each active training run is included.

| File | Training setup | SHA-256 |
|---|---|---|
| `checkpoints/unitree_rough_4096_model_36300.pt` | Unitree rough terrain, 4,096 envs | `0c74ad40c5763d1db7768adc025f24967ad1dde56d7b2e5eb4a02c9b2225928d` |
| `checkpoints/terrain10_2048_model_37400.pt` | custom Terrain10 curriculum, 2,048 envs | `7bbd5cbc1431fb8085b6533f457b4cbb7eb20b960f4c033a6907c3e7fe0fbe00` |

Both actors have 44 inputs and 12 outputs. These are rsl_rl training checkpoints, not
ONNX exports.

## Complete policy and training contract

### Timing, command, and action

| Item | Value |
|---|---|
| physics step | 0.005 s, 200 Hz |
| policy/environment step | 0.02 s, 50 Hz, decimation 4 |
| training episode | 20 s maximum |
| command resampling | uniformly every 3–8 s |
| command range | `vx ∈ [-1, 2] m/s`, `vy=0`, `wz ∈ [-1, 1] rad/s` |
| standing command probability | 5% |
| heading/forward/world command modes | disabled |
| command gate | removed |
| action | 12 raw motor-position actions |
| action transform | `target = default_pose + 0.25 × action` rad, then joint target clip |
| actuator delay | independently sampled 0–1 physics step, 0–5 ms |

The policy observes `[vx, wz]`. It does not have a `vy` observation slot. The joint and
action order is right leg first, then left leg; within each leg:

```text
hip_pitch, hip_roll, hip_yaw, knee, ankle_a, ankle_b
```

### Actor observations: 44 dimensions

Actor corruption is enabled during training. Noise is per-step additive uniform noise.
The motor-position term also samples a per-environment lag of 0–1 policy step.

| Index | Term | Size | Source/value | Training corruption |
|---:|---|---:|---|---|
| 0–2 | `base_ang_vel` | 3 | IMU angular velocity, rad/s | `U(-0.2, 0.2)` rad/s |
| 3–5 | `projected_gravity` | 3 | IMU up-vector projected gravity | `U(-0.05, 0.05)` |
| 6–17 | `motor_pos_history` old frame | 12 | motor position minus default, rad | `U(-0.01, 0.01)` rad; lag 0–1 |
| 18–29 | `motor_pos_history` new frame | 12 | motor position minus default, rad | `U(-0.01, 0.01)` rad; lag 0–1 |
| 30–41 | `actions` | 12 | previous raw policy action | none |
| 42 | `command/vx` | 1 | forward command, m/s | none |
| 43 | `command/wz` | 1 | yaw-rate command, rad/s | none |

The actor does **not** receive base linear velocity, joint velocity, height scan, foot
contact, contact force, or DR state. The first history sample after reset is initialized
from the current joint position rather than zero.

### Critic observations: 268 dimensions

Critic corruption is disabled. Noise objects inherited on some terms are therefore not
applied to the critic. Critic-only observations are not part of the deployment input.

| Term | Size | Value |
|---|---:|---|
| `base_lin_vel` | 3 | privileged base linear velocity |
| `base_ang_vel` | 3 | base angular velocity |
| `projected_gravity` | 3 | projected gravity |
| `joint_pos` | 16 | 12 motor joints plus 4 passive ankle pitch/roll joints, relative position |
| `joint_vel` | 16 | the same 16 joint velocities |
| `actions` | 12 | previous raw policy action |
| `command` | 2 | `[vx, wz]` |
| `height_scan` | 187 | terrain scan, scaled by 0.2 |
| `foot_height` | 2 | terrain-relative foot height |
| `foot_air_time` | 2 | current air time |
| `foot_contact` | 2 | binary/contact state |
| `foot_contact_forces` | 6 | two 3-D foot contact forces |
| `dr_friction` | 1 | active shared foot friction coefficient |
| `dr_mass` | 1 | total robot mass fractional offset from nominal |
| `dr_com` | 3 | base-link CoM offset from nominal, metres |
| `dr_push` | 8 | intended push vector 6 + age + next-push time; **currently all zero** |
| `dr_factor` | 1 | **currently constant 1.0** because generalized training has no DR ramp |
| **Total** | **268** | |

`dr_push` remains zero because the new `torso_push` force event does not write the old
velocity-push logging buffers and is not named `push_robot`. This is a known critic-side
instrumentation mismatch; it does not mean torso pushes are disabled.

### Rewards: all 18 active terms

The listed weight is the config weight. Each step contribution is
`term_value × weight × dt`, with `dt=0.02 s`.

| Reward term | Weight | Exact role and active parameters |
|---|---:|---|
| `track_linear_velocity` | +1.0 | `exp(-(‖vxy-cxy‖² + vz²)/0.5²)` |
| `track_angular_velocity` | +1.0 | `exp(-((wz-cwz)² + wx² + wy²)/sqrt(0.5)²)` |
| `body_orientation_l2` | -1.0 | squared projected-gravity x/y components of `base_link` |
| `pose` | +1.0 | `exp(-mean(q_error²/std²))`; command-speed-dependent joint tolerances below |
| `body_ang_vel` | -0.05 | squared base roll/pitch angular velocity; yaw is excluded |
| `angular_momentum` | -0.025 | squared whole-body angular-momentum norm |
| `is_terminated` | -200.0 | indicator for non-timeout failure termination |
| `joint_acc_l2` | -2.5e-7 | sum of squared joint accelerations |
| `joint_pos_limits` | -10.0 | summed position excursion beyond soft joint limits |
| `action_rate_l2` | -0.05 | `sum((action_t-action_(t-1))²)` on raw policy actions |
| `air_time` | +0.5 | count feet whose air time is 0.05–0.5 s; active when command magnitude >0.1 |
| `foot_clearance` | -1.0 | `sum(|foot_height-0.1 m| × horizontal_foot_speed)`; command >0.1 |
| `foot_slip` | -0.25 | summed squared horizontal foot speed while in contact; command >0.1 |
| `soft_landing` | -0.001 | summed contact-force magnitude on first-contact frames; command >0.1 |
| `stand_still` | -1.0 | squared motor-joint default-pose error when total command ≤0.1 |
| `self_collisions` | -1.0 | self-collision cost for contacts over 10 N |
| `actuator_clipping_fraction` | -0.1 | mean `|requested-applied|/max(|requested|,1e-6)` after T-N clamp |
| `negative_mechanical_power` | -0.1 | normalized braking-power cost above a free fraction of 0.3 |

Pose tolerances (`std`, rad) use total command speed `‖vxy_cmd‖+|wz_cmd|`:

| Regime | Command speed | hip pitch | hip roll | hip yaw | knee | ankle pitch | ankle roll |
|---|---:|---:|---:|---:|---:|---:|---:|
| standing | `<0.1` | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| walking | `0.1–1.5` | 0.3 | 0.4 | 0.15 | 1.2 | 0.25 | 0.1 |
| running | `≥1.5` | 0.5 | 0.2 | 0.2 | 1.5 | 0.35 | 0.15 |

There is no separate `track_lin_vel_progress`, knee-overspeed, generic torque-L2, or
fixed torque-limit penalty in this generalized reward set. Torque feasibility is handled
by the measured torque-speed clamp plus `actuator_clipping_fraction`.

### Terminations: all 6 active terms

| Termination | Timeout? | Trigger |
|---|---|---|
| `time_out` | yes | episode reaches 20 s |
| `fell_over` | no | base tilt exceeds 70° (`1.22173 rad`) |
| `out_of_terrain_bounds` | yes | root leaves generated terrain footprint, 0.3 m margin |
| `low_base` | no | base height above local terrain median falls below 0.7 m |
| `nan_detection` | no | NaN/Inf in MuJoCo physics state |
| `physics_blowup` | no | any joint speed >100 rad/s, base speed >10 m/s, or non-finite state |

The old `stagnation` termination is explicitly removed. Timeout-class terms truncate an
episode but are not charged the `is_terminated=-200` failure penalty.

### Reset randomization and domain randomization

All reset DR is active from the beginning; there is no DR-strength ramp.

| Event | Mode | Distribution/behavior |
|---|---|---|
| base pose reset | episode reset | x/y offset ±0.5 m, z offset 0.01–0.05 m, yaw `U(-pi,pi)` |
| base velocity reset | episode reset | x/y linear and yaw angular velocity each `U(-0.2,0.2)` |
| motor joint reset | episode reset | position offset `U(-0.05,0.05)` rad; velocity exactly zero |
| link pseudo-inertia | episode reset | per-link `alpha~U(0.5ln0.9,0.5ln1.1)`; mass and inertia scale by `exp(2alpha)`, range 0.9–1.1 |
| base CoM | episode reset | additive x/y/z offset independently `U(-0.03,0.03)` m |
| Kp | episode reset | per actuator scale `U(0.9,1.1)` |
| Kd | episode reset | per actuator scale `U(0.9,1.1)`, sampled independently of Kp |
| motor strength | episode reset | per-motor T-N envelope scale `U(0.8,1.2)` |
| motor joint friction | episode reset | nominal friction-loss scale `U(0.8,1.2)` |
| motor joint damping | episode reset | nominal passive-damping scale `U(0.8,1.2)` |
| foot friction | episode reset | absolute coefficient `U(0.3,1.2)`, one shared draw for all foot geoms |
| encoder bias | episode reset | per-joint persistent bias `U(-0.01,0.01)` rad |
| actuator command delay | every physics step | random lag 0–1 physics step, 0–5 ms |
| actor gyro noise | every policy step | additive `U(-0.2,0.2)` rad/s |
| actor gravity noise | every policy step | additive `U(-0.05,0.05)` |
| actor motor-position noise | every policy step | additive `U(-0.01,0.01)` rad |
| actor motor-position lag | every policy step | independently selected lag 0–1 policy step, 0–20 ms |
| torso push | step event | random horizontal direction, magnitude `U(50,100)` N, 0.1 s duration, cooldown `U(5,10)` s |

Nominal gains before gain DR:

| Joint group | Kp | Kd |
|---|---:|---:|
| hip pitch / roll | 135 | 6 |
| hip yaw | 75 | 4 |
| knee | 130 | 6 |
| ankle A/B crank | 60 | 2.7 |

### Terrain sampling and curriculum

Terrain is privileged critic input through `height_scan`; it is not an actor/deployment
input. Both generators use seed `89500`, and newly reset environments sample uniformly
from every currently unlocked level.

| Checkpoint/task | Patch layout | Terrain distribution | Curriculum |
|---|---|---|---|
| `unitree_rough_4096_model_36300.pt` / `Pygmalion-Velocity-Generalized-UnitreeRough` | 10 difficulty rows; each patch 8×8 m; 20 m outer border | seven types sampled uniformly: flat, pyramid stairs up/down, pyramid slope up/down, random rough, wave | starts at level 0 at iteration 19,500; unlocks one level every 2,000 iterations; levels `0..max` sampled uniformly; level 9 unlocks at 37,500 |
| `terrain10_2048_model_37400.pt` / `Pygmalion-Velocity-Generalized-Terrain10-2kResume19700` | one row of ten 10×10 m patches; no outer border | ten fixed types below; unlocked types sampled uniformly | resumes at iteration 19,700 with levels 0–3 unlocked; unlocks one level every 2,000 iterations; all levels are unlocked from iteration 31,700 |

Custom Terrain10 level definitions:

| Level/index | Geometry |
|---:|---|
| 0 | flat |
| 1 | transverse ridges, 0.5 cm radius |
| 2 | transverse ridges, 1.0 cm radius |
| 3 | +3° slope |
| 4 | -3° slope |
| 5 | random transverse ridges, 0.75–1.5 cm radius |
| 6 | random transverse ridges, 1–2 cm radius |
| 7 | +6° slope |
| 8 | -6° slope |
| 9 | +6° slope with random 1–2 cm ridges |

For stock Unitree rough terrain, difficulty row 0 is the easiest generated parameter
setting, but only the `flat` type is geometrically flat. The other six types remain their
respective stairs, slopes, roughness, or waves at row-0 difficulty. At checkpoint 36,300,
the Unitree run had levels 0–8 unlocked; at checkpoint 37,400, Terrain10 had levels 0–9
unlocked.

No camera, depth, heightmap, gravity-bias, joint-velocity-noise, restitution, action
jitter, or motor-offset DR is configured. Terrain sampling/curriculum is separate from
the physical/sensor DR table above.

## Registered tasks

```text
Pygmalion-Velocity-Generalized-Terrain10
Pygmalion-Velocity-Generalized-UnitreeRough
Pygmalion-Velocity-Generalized-Terrain10-2kResume19700
```

The implementation is in:

```text
pygmalion_mjlab/generalized_cfg.py
pygmalion_mjlab/terrain10.py
pygmalion_mjlab/unitree_rough.py
pygmalion_mjlab/robot.py
pygmalion_mjlab/tn_actuator.py
pygmalion_mjlab/mdp/events.py
pygmalion_mjlab/mdp/rewards.py
scripts/train.py
scripts/play.py
```

`scripts/train.py --gain-dr LO HI` replaces the configured gain randomization range;
it does not stack a second gain randomizer.

## Hardware deployment request

Target repository: `Human-Pygmalion/HUPHY`, `biped` branch. The reviewed baseline was
commit `a1b67b1bbcfe0ab367862c1e4722a11776113e03`.

Run the following policy-start-relative profile exactly once:

| Policy time | Command/state |
|---:|---|
| 0–5 s | `vx=0.5 m/s`, `wz=0`; `vy=0` implicitly |
| 5–10 s | `vx=1.0 m/s`, `wz=0`; `vy=0` implicitly |
| 10–15 s | automatic Damp: `kp=0`, `kd=10` |
| after 15 s | motor disable / torque off |

Important compatibility requirements:

1. Do not replace the existing `stand_bi.pt` file and run it through the old `stand.py`
   contract. That path expects 45 observations and `[vx, vy, wz]`; this policy expects
   44 observations and `[vx, wz]`.
2. Add a separate `generalized_walk` policy spec and observation adapter.
3. Keep `vy=0`; do not append a `vy` observation slot.
4. Use the nominal gain table above on hardware. Do not randomize gains on hardware.
5. Do not implement this with `huphy-run --duration 10`: the existing duration starts
   before approach and Enter waiting. Time from the instant the policy starts.
6. At policy time 10 s, call `ControlLoop.stop()`. Its existing `finally → _exit() →
   _settle()` path sends bilateral Damp for 5 s and then disables all motors.
7. Returning `None` without stopping the loop is unsafe because the last PD target may
   remain active.

Recommended state machine:

```text
APPROACH → WAIT_FOR_ENTER → WALK_0P5 → WALK_1P0 → DAMP → DISABLED
                           5 seconds   5 seconds   5 seconds
```

## Hardware safety checklist

- Use an overhead harness and a dedicated physical E-stop operator.
- Verify encoder signs/offsets, IMU axes, gravity sign, and all 12 joint indices before
  enabling torque.
- Validate model input 44 and output 12 before enabling motors.
- Start from the policy default crouched pose.
- Log observation, action, command, joint state, motor command, faults, and cycle timing.
- Stop on CAN link loss, NaN/Inf, motor fault, cycle overrun, or fall detection.
- First test `vx=0.5` only; run the full profile only after Damp and disable are verified.

The simulator's red sphere at 90% torque-speed utilization is visualization only. The
reviewed HUPHY `biped` code does not yet have an equivalent dynamic torque-utilization
guard or fall detector; those must not be assumed to exist during the hardware test.
