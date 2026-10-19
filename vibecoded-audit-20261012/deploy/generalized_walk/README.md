# Generalized HUPHY walking: training and deployment handoff

## Included checkpoints

Only the latest checkpoint captured from each active training run is included.

| File | Training setup | SHA-256 |
|---|---|---|
| `checkpoints/unitree_rough_4096_model_36300.pt` | Unitree rough terrain, 4,096 envs | `0c74ad40c5763d1db7768adc025f24967ad1dde56d7b2e5eb4a02c9b2225928d` |
| `checkpoints/terrain10_2048_model_37400.pt` | custom Terrain10 curriculum, 2,048 envs | `7bbd5cbc1431fb8085b6533f457b4cbb7eb20b960f4c033a6907c3e7fe0fbe00` |

Both actors have 44 inputs and 12 outputs. These are rsl_rl training checkpoints, not
ONNX exports.

## Policy contract

Actor observation layout:

| Index | Size | Value |
|---:|---:|---|
| 0–2 | 3 | base angular velocity, rad/s |
| 3–5 | 3 | projected gravity in body frame |
| 6–17 | 12 | previous motor position relative to default pose, rad |
| 18–29 | 12 | current motor position relative to default pose, rad |
| 30–41 | 12 | previous raw policy action |
| 42 | 1 | commanded forward velocity `vx`, m/s |
| 43 | 1 | commanded yaw velocity `wz`, rad/s |

The actor does not receive base linear velocity, joint velocity, height scan, contact
state, or DR state. Height scan and privileged state are critic-only during training.

The action is a 12-dimensional motor-position command:

```text
target = default_pose + 0.25 * action  [rad]
```

The joint/action order is right leg first, then left leg; within each leg:

```text
hip_pitch, hip_roll, hip_yaw, knee, ankle_a, ankle_b
```

## Training configuration

- Commands: actor receives `[vx, wz]`; lateral command `vy` is fixed to zero.
- Training ranges: `vx ∈ [-1, 2] m/s`, `wz ∈ [-1, 1] rad/s`.
- The old command gate, heading-command mixture, and lateral-motion command were removed.
- Reward uses one standard linear-velocity tracking term and one yaw tracking term.
- Torque-speed clipping utilization is penalized by the clipped fraction.
- Nominal gains are:

| Joint group | Kp | Kd |
|---|---:|---:|
| hip pitch / roll | 135 | 6 |
| hip yaw | 75 | 4 |
| knee | 130 | 6 |
| ankle A/B crank | 60 | 2.7 |

Training DR:

| Item | Range |
|---|---|
| link mass/inertia | nominal × `U(0.9, 1.1)` |
| base CoM offset | x/y/z `U(-0.03, 0.03)` m |
| Kp and Kd | nominal × `U(0.9, 1.1)` independently |
| motor torque-speed strength | `U(0.8, 1.2)` |
| joint friction/damping | nominal × `U(0.8, 1.2)` |
| foot friction | `U(0.3, 1.2)` |
| encoder episode bias | `U(-0.01, 0.01)` rad |
| action delay | 0–1 physics step, 0–5 ms |
| motor-position observation lag | 0–1 policy step, 0–20 ms |
| observation noise | gyro ±0.2 rad/s, gravity ±0.05, joint position ±0.01 rad |
| torso push | random horizontal direction, 50–100 N for 0.1 s every 5–10 s |

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
