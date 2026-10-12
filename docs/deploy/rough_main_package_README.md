# Pygmalion Rough-Terrain Walking Policy - Deployment Handout (2026-10-12)

This package runs the policy `rough_box_main_v11a` (checkpoint 11,999) on the real robot.
Read this handout before you power on any motor.

The robot is Pygmalion: a lower body with 12 motor joints.
The policy is one small neural network.
It takes 45 numbers as input and gives 12 numbers as output.
You call one function, `inference()`, 50 times each second.

You do not need the simulator. You need only `numpy` and `onnxruntime`.

```bash
pip install -r requirements.txt
python3 policy_runner.py --contract deploy_contract.json --onnx policy.onnx --self-test
python3 example_loop.py --seconds 30
```

Language: Simplified Technical English (ASD-STE100 style).
A full audit of the training of this policy is in `docs/audit/2026-10-12_rough_box_main_v11a/` of the repository.

---

## 1. Package contents

| File | Content | SHA-256 (first 16) |
|---|---|---|
| `policy.onnx` | The network that runs on the robot. Input 45, output 12. It contains the observation normalizer. | `3784a7d1fb65c6f0` |
| `policy.pt` | The same policy in PyTorch format (checkpoint 11,999). Use it to resume training or export again. Do not use it on the robot. | `8a067c9ba6e3d4da` |
| `deploy_contract.json` | The specification of the input and output: the meaning of each of the 45 inputs, the formula for the 12 outputs, and 16 reference input/output pairs. | `78eadd3cdfa1b16c` |
| `obs_adapter.py` | Makes the 45 inputs from the sensor values. It uses only `numpy`. | |
| `policy_runner.py` | Contains `StandingPolicy` and `inference()`. This is the only file that your code calls. | |
| `example_loop.py` | A control loop with a fake robot. Use it to check the wiring, the units and the cycle time without a robot. | |
| `motor_params.json` | The motor values that were used in training (Kp, Kd, torque limit, armature, friction, torque-speed curve). | |
| `train_env_manifest.json` | The full training settings of this policy (all `PYG_*` flags and the launch command). | |
| `requirements.txt` | `numpy` and `onnxruntime`. | |

The name `StandingPolicy` is historical. The class runs any policy of this contract format. It runs this walking policy.

The `checkpoint` path in `deploy_contract.json` is a record of the training machine. In this folder the same file is `policy.pt`.

---

## 2. The policy

| Item | Value |
|---|---|
| Name | `2026-10-11_23-35-20_rough_box_main_v11a`, checkpoint `model_11999.pt` |
| Training | 12,000 iterations on rough terrain made of boxes (stairs up and down, uneven blocks up to 0.12 m, tilted plates up to 15°, flat). Random seed 43. |
| Start point | The actor network of a flat-ground walking policy (30,000 iterations). |
| Robot model | `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix` (MuJoCo XML), 23.63 kg |
| Ankle | Two crank motors (A and B) move the ankle pitch and roll through a 2-RSU linkage. |
| Network | 45 - 512 - 256 - 128 - 12, ELU activation |
| Control rate | 50 Hz (20 ms). Physics step in training: 5 ms, 4 steps for each control cycle. |
| Randomization in training | Pushes, foot friction, joint zero offset (±0.86°), link mass and inertia. It reached full strength at iteration 8,000. |

This is a **walking policy**. You tell it what to do with the `command` input.
The command (0, 0, 0) means "stand".

Command range in training (the full range was used for all 12,000 iterations):

| Axis | Range |
|---|---|
| Forward | -2.0 to +2.5 m/s |
| Sideways | -1.0 to +1.0 m/s |
| Turn | -1.0 to +1.0 rad/s |

Keep each command inside this range. For the first tests, use small commands (Section 3).

---

## 3. Measured behavior (simulation only)

These numbers come from simulation. They are not real-robot results.
Tools: `measure_turn.py` (4 seeds), `measure_stand.py` (4 seeds, 27 s), `gait_kinematics_probe.py` (3 seeds), `terrain_eval.py` (256 robots, 6 s), `trail_walk_check.py`. Measured 2026-10-12.

| Test | Result | Project bar | Pass |
|---|---|---|---|
| Stand at (0, 0, 0): time with a foot off the ground | 1.3 % | ≤ 2 % | yes |
| Stand at (0, 0, 0): drift in 27 s | 0.38 m (one of 4 seeds is large) | ≤ 0.3 m | no |
| Walk at 1.2 m/s command: real speed | 1.32 m/s | 1.14 to 1.26 m/s | no (10 % fast) |
| Walk: left/right difference of hip twist | +18 % | ≤ 25 % | yes |
| Walk straight: heading drift | 8.6° | ≤ 15° | yes |
| Turn in place: achieved / commanded (median of 4 seeds) | 45 % (seeds 26 to 64 %) | ≥ 50 % | no |
| Turn in place: commands with a wrong-way turn | 0 to 2 of 8 | ≤ 1 of 8 | no (one seed) |
| Box terrain, level 6 of 9, walk 0.6 m/s: falls | 0.4 % | ≤ 10 % | yes |
| Box terrain, level 9 of 9: falls | 11.3 % | ≤ 20 % | yes |
| Scanned mountain trail, 182 m, down and up, 0.6 and 1.0 m/s | 4 of 4 complete, no fall, body ≥ 0.79 m | complete, body ≥ 0.7 m | yes |

**First tests on the real robot:** stand (0, 0, 0), then forward 0.3 to 0.6 m/s.
**Do not give large turn commands in the first tests.** Small turn commands (±0.25 rad/s) can turn the wrong way.

---

## 4. The `inference()` function

File: `policy_runner.py`. Class: `StandingPolicy`.

```python
from policy_runner import StandingPolicy

pol = StandingPolicy.from_files("deploy_contract.json", "policy.onnx")
targets = pol.inference(
    gyro=...,                # 3 numbers, rad/s, body frame
    projected_gravity=...,   # 3 numbers, unit vector, body frame, (0, 0, -1) when upright
    joint_pos={...},         # 12 joints, rad, dictionary with joint names
    command=(0.0, 0.0, 0.0), # forward m/s, sideways m/s, turn rad/s
)
```

### 4.1 Inputs

| Name | Content | Unit | Note |
|---|---|---|---|
| `gyro` | Angular velocity of the body | rad/s | Body frame. |
| `projected_gravity` | Direction of gravity in the body frame | none (unit vector) | (0, 0, -1) when the robot is upright. |
| `joint_pos` | Joint angles of the 12 motor joints | rad | Dictionary. The key is the joint name. |
| `command` | Wanted speed: forward, sideways, turn | m/s, m/s, rad/s | (0, 0, 0) = stand. |

### 4.2 Output

A dictionary. The key is the joint name. The value is the **target angle** in rad.
The output is not a torque. The PD controller in each motor makes the torque from the target angle.

### 4.3 What happens inside one call

1. The adapter subtracts the default pose from the joint angles.
2. The adapter adds the new joint angles to a 2-frame history (old frame first).
3. The adapter joins the 5 input parts in a fixed order. The result is 45 numbers.
4. The ONNX network gives 12 raw numbers.
5. The runner keeps the 12 raw numbers. They are the `last_action` input of the next call.
6. The runner makes the targets: `target = offset + 0.25 × raw`. Then it clips each target to the joint limit.

### 4.4 Other methods

| Method | Use |
|---|---|
| `self_test()` | Compare the network with the 16 stored input/output pairs. Call it one time before you power on. |
| `reset()` | Clear the joint history and the last action. Call it each time you stand the robot. |
| `default_pose` | Dictionary of the default joint angles. Move the robot to this pose before you start. |
| `joint_names` | List of the 12 joint names. |
| `last_raw_action` | The last raw output. Use it for logs only. |

---

## 5. The 45 observation numbers

| Index | Name | Content | Unit | Noise added in training (uniform, ±) |
|---|---|---|---|---|
| 0-2 | `base_ang_vel` | Angular velocity of the body | rad/s | 0.2 |
| 3-5 | `projected_gravity` | Gravity direction in the body frame | none | 0.05 |
| 6-29 | `motor_pos_history` | 12 joints × 2 frames, minus the default pose, `[old frame][new frame]` | rad | 0.01 |
| 30-41 | `last_action` | Raw output of the previous cycle | none | none |
| 42-44 | `command` | Wanted speed | m/s, m/s, rad/s | none |

The noise was added on purpose in training.
If your real sensor noise is larger, the policy sees data that it did not see in training.

---

## 6. The 12 action numbers

```
target [rad] = offset + 0.25 × action, then clip to the joint limit
```

An action of 1 moves the joint 0.25 rad (14.3°) from the offset. The offset is the default pose.

Joint order (the same everywhere):

```
 0 R_hip_pitch   1 R_hip_roll   2 R_hip_yaw   3 R_knee   4 R_crank_A   5 R_crank_B
 6 L_hip_pitch   7 L_hip_roll   8 L_hip_yaw   9 L_knee  10 L_crank_A  11 L_crank_B
```

| Joint | Offset [rad] | Offset [deg] | Clip [rad] |
|---|---|---|---|
| R_hip_pitch | -0.17500 | -10.03 | -1.9679 to +0.3098 |
| R_hip_roll | 0.00000 | 0.00 | -0.3403 to +1.3875 |
| R_hip_yaw | 0.00000 | 0.00 | -0.7069 to +0.7069 |
| R_knee | +0.35000 | +20.05 | +0.1047 to +1.9897 |
| R_crank_A | +0.14395 | +8.25 | -1.0800 to +1.0800 |
| R_crank_B | -0.14706 | -8.43 | -1.0800 to +1.0800 |
| L_hip_pitch | +0.17500 | +10.03 | -0.3098 to +1.9679 |
| L_hip_roll | 0.00000 | 0.00 | -1.3875 to +0.3403 |
| L_hip_yaw | 0.00000 | 0.00 | -0.7069 to +0.7069 |
| L_knee | -0.35000 | -20.05 | -1.9897 to -0.1047 |
| L_crank_A | -0.14389 | -8.24 | -1.0800 to +1.0800 |
| L_crank_B | +0.14687 | +8.42 | -1.0800 to +1.0800 |

The default pose is a slight crouch. The left and right signs are opposite. Do not change them.
The input and output layout is the same as the earlier package `pyg_v11a_stand01` (checked field by field). Robot code for that package works with this package. Change only the files.

---

## 7. Motor values used in training

The policy expects motors that behave as shown here.

| Joints | Motor | Kp [N·m/rad] | Kd [N·m·s/rad] | Torque limit [N·m] |
|---|---|---|---|---|
| `*_hip_pitch`, `*_hip_roll` | RobStride RS04 | 150.0 | 6.0 | 120 |
| `*_knee` | RobStride RS04 | 220.0 | 6.0 | 120 |
| `*_hip_yaw` | RobStride RS03 | 150.0 | 6.0 | 60 |
| `*_crank_A`, `*_crank_B` | RobStride RS03 | 22.266 | 1.414 | 60 |

> **WARNING**
> The repository documents a real-robot setting of kp 10 / kd 1 for all joints. That is 15 to 22 times lower than these training gains.
> With much lower gains, the same target angle gives a much weaker and slower movement. The policy did not see this in training.
> Decide the real gains before you run the policy on the ground.

The crank gains are low on purpose. The cranks reach the ankle through a linkage. If you use the hip gains on the cranks, the foot chatters.
All values are in `motor_params.json`. They are identical to the values in `pyg_v11a_stand01`.

---

## 8. Run it

### 8.1 Control loop

```python
pol.self_test()                       # one time, before the motors are on

for name, rad in pol.default_pose.items():
    motor[name].move_to(rad)          # move slowly to the default pose

pol.reset()                           # each time you stand the robot

while True:                           # exactly 50 Hz (every 20 ms)
    targets = pol.inference(
        gyro=imu.gyro_xyz,
        projected_gravity=imu.gravity_xyz,       # (0, 0, -1) when upright
        joint_pos={n: motor[n].position for n in pol.joint_names},
        command=(0.0, 0.0, 0.0),
    )
    for name, rad in targets.items():
        motor[name].set_position(rad)
```

The policy assumes a 20 ms cycle. A slower cycle changes the behavior of the robot.
The runner shows one warning if the median cycle time is more than 20 % wrong. It does not stop.

### 8.2 Viewers on the training machine (simulation)

| Viewer | Command | What it shows |
|---|---|---|
| Deploy-code viewer (same `obs_adapter` and `policy.onnx` as the robot) | `cd mujoco-sim/mjlab && .venv/bin/python3 ../../tools/robot_model/viser_stand_viewer.py --contract ../../deploy_asset/pyg_rough_main_v11a/deploy_contract.json --onnx ../../deploy_asset/pyg_rough_main_v11a/policy.onnx --port 8102` | Flat ground, push and lift buttons, noise switch |
| Mountain-trail viewer (drive it yourself) | `bash tools/terrain_nav/start_trail_viser.sh roughmain` (host 10.8.0.14, port 8098) | Scanned trail terrain with collision |

---

## 9. Checks before the motors are on

| Check | Command | Result for this package |
|---|---|---|
| Self-test (network against 16 stored pairs) | `python3 policy_runner.py --contract deploy_contract.json --onnx policy.onnx --self-test` | Maximum difference 2.522e-06. Tolerance 1e-04. **Pass.** |
| Contract (PyTorch and ONNX against the stored pairs) | `analysis/deploy_contract.py --run-dir <run> --checkpoint <run>/model_11999.pt --verify` | PyTorch 2.098e-07. ONNX 2.522e-06. **Pass.** |
| **Full input/output against mjlab** (closed loop, 1,050 steps, 7 commands) | `tools/deploy/verify_deploy_io.py --contract deploy_contract.json --onnx policy.onnx --checkpoint <run>/model_11999.pt` | See 9.1. **Pass.** |
| Dry run (no robot) | `python3 policy_runner.py ... --dry-run 20` and `python3 example_loop.py --seconds 30` | Runs with no error. |

If the self-test fails, do not power on the motors. The contract and the network are then not a pair.

### 9.1 Full input/output check against the training code

The tool runs the training environment (mjlab) and the deploy code side by side.
At each 20 ms step it gives the deploy code only the values that a robot can read: gyro, IMU direction, 12 joint angles by name, and the command.
It compares three things with mjlab:

1. The 45 observation numbers (deploy `ObsAdapter` against mjlab `obs["actor"]`).
2. The 12 raw network outputs (deploy ONNX against mjlab PyTorch).
3. The 12 joint target angles (deploy `inference()` against mjlab `JointPositionAction`: scale, offset and clip).

| Command (forward, sideways, turn) | Observation max difference | Network max difference | Target max difference [rad] |
|---|---|---|---|
| (0, 0, 0) | 2.0e-06 | 2.0e-06 | 5.2e-07 |
| (0.6, 0, 0) | 6.0e-06 | 6.0e-06 | 1.5e-06 |
| (1.2, 0, 0) | 5.5e-06 | 5.5e-06 | 1.4e-06 |
| (0, 0.5, 0) | 3.8e-06 | 3.8e-06 | 9.5e-07 |
| (0, 0, 0.8) | 4.3e-06 | 4.3e-06 | 1.1e-06 |
| (-0.8, 0, -0.5) | 1.9e-05 | 1.9e-05 | 4.8e-06 |
| (2.0, 0, 0) | 6.7e-06 | 6.7e-06 | 1.7e-06 |

Tolerance 1e-04. 52 target values hit a clip limit, so the clip path is also tested.
The small differences come from float32 (ONNX) against PyTorch arithmetic.
Conditions: flat ground, no observation noise, no randomization (the deploy code adds no noise).

---

## 10. Four errors that people make

> **WARNING**
> A wrong gravity sign makes the robot think it hangs upside down. The length check does not find this error.

1. **Gravity sign.** `projected_gravity` points in the direction of gravity: (0, 0, -1) when upright.
   Many IMUs give the "up" direction (0, 0, +1). If your IMU does this, change the sign before you call `inference()`.
   The adapter checks the first 5 cycles and stops the program if the sign is wrong.
2. **Last action.** The `last_action` part is the raw network output, not the target angle. `StandingPolicy` keeps the correct value for you.
3. **Joint history.** The 24 numbers are `[old frame, 12 joints][new frame, 12 joints]`, relative to the default pose.
   Call `inference()` exactly one time in each cycle. Each call moves the history one frame.
4. **Joint order and names.** Give the joint angles as a dictionary with names. A list makes a wrong order possible.

---

## 11. What the code does not do, and what is not solved

### 11.1 Not in the code (on purpose)

- It has **no safety stop and no timeout.** Add them in your own program.
- It does **not set the PD gains.** Section 7 gives the training values.
- It does **not normalize the observation.** The ONNX file does this inside.
- It does **not detect that the robot is in the air.**

### 11.2 Open items on 2026-10-12

| # | Item | Why it matters |
|---|---|---|
| 1 | Real-robot gains (documented kp 10 / kd 1) are 15 to 22 times lower than the training gains. | The policy did not train with weak motors. See Section 7. |
| 2 | Joint zero offsets: the measured zero error is up to 9.4° (left knee). Training randomized only ±0.86°. | Apply the zero calibration before you run the policy. |
| 3 | Training had no randomization of action delay, IMU bias, gains or motor friction. | Real delay and IMU bias can change the behavior. |
| 4 | The order of the IMU quaternion is not confirmed. | A wrong order gives a wrong `projected_gravity` (`docs/151_imu_quaternion_order_check.md`). |
| 5 | The real motor type on each joint is not confirmed. A hardware map lists RS03 for all 12 joints. The model uses RS04 for hip pitch, hip roll and knee. | RS03 has half the torque of RS04. |
| 6 | The crank range of ±68.75° is not confirmed on the real robot. The hand-tested range was ±46°. | A large crank motion can hit a hard stop. |
| 7 | Turning in place, walking speed and standing drift do not meet the project bars (Section 3). | Use small commands first. |
| 8 | Behavior when a tether lifts the robot is not measured for this policy. | An older policy swung its legs hard in the air (`docs/148_air_shake_and_action_delay.md`). |
| 9 | The model foot mass uses an estimate (0.62 kg aluminium part) that nobody weighed. | The real foot may be lighter. |

---

## 12. Notes on the writing standard

- The text follows the ASD-STE100 writing rules: short sentences, one instruction in each sentence, active voice, the imperative for steps.
- The text was not checked against the official ASD-STE100 dictionary or a checker program.
- Technical names (ONNX, IMU, PD controller, crank) are used as technical names.
