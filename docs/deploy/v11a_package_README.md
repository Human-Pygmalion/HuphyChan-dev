# Pygmalion Policy v11a - Deployment Handout (2026-10-06)

This package runs the policy `legonly_ab_v11a_stand01_s43` on the real robot.
Read this handout before you power on any motor.

The robot is Pygmalion (lower body, 12 joints).
The policy is one small neural network.
It takes 45 numbers as input and gives 12 numbers as output.
You call one function, `inference()`, 50 times each second.

You do not need the simulator. You need only `numpy` and `onnxruntime`.

```bash
pip install -r requirements.txt
python3 policy_runner.py --contract deploy_contract.json --onnx policy.onnx --self-test
python3 example_loop.py --seconds 30
```

Language: Simplified Technical English (ASD-STE100 style). See Section 12 for the limits of this check.

---

## 1. Package contents

| File | Content |
|---|---|
| `policy.onnx` | The network that runs on the robot. Input 45, output 12. It contains the observation normalizer. |
| `policy.pt` | The same policy in PyTorch format. Use it to resume training or export again. Do not use it on the robot. |
| `deploy_contract.json` | The specification of the input and output. It has the meaning of each of the 45 inputs, the formula for the 12 outputs, and 16 reference pairs. |
| `obs_adapter.py` | Makes the 45 inputs from the sensor values. It uses only `numpy`. |
| `policy_runner.py` | Contains `StandingPolicy` and `inference()`. This is the only file that your code calls. |
| `example_loop.py` | A control loop with a fake robot. Use it to check the wiring, the units, and the cycle time without a robot. |
| `motor_params.json` | The motor values that were used in training. |
| `train_env_manifest.json` | The full training settings of this policy. |
| `requirements.txt` | `numpy` and `onnxruntime`. |

The `checkpoint` path in `deploy_contract.json` is a record of the training machine.
In this folder the same file is `policy.pt`.

---

## 2. The policy

| Item | Value |
|---|---|
| Name | `legonly_ab_v11a_stand01_s43` |
| Training | 29,998 iterations, random seed 43 |
| Share of training robots that stand still | 10 % (the others walk or turn) |
| Robot model | `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix` |
| Ankle mode | AB. Two crank motors move the ankle. |
| Network | 45 - 512 - 256 - 128 - 12, ELU activation |
| Control rate | 50 Hz (20 ms). Physics step in training: 5 ms, 4 steps for each control cycle. |

This is a **walking policy**. It is not a stand-only policy.
You tell it what to do with the `command` input.
The command (0, 0, 0) means "stand".

Command range in training:

| Axis | Range |
|---|---|
| Forward | -1.2 to +1.2 m/s |
| Sideways | -0.6 to +0.6 m/s |
| Turn | -0.8 to +0.8 rad/s |

Keep each command inside this range.

---

## 3. Measured behavior (simulation only)

These numbers come from simulation. They are not real-robot results.

### 3.1 At command (0, 0, 0)

Tool: `measure_stand.py` and `standing_verdict.py`. 8 random seeds. 27 s for each run. Measured 2026-10-06.

| Measure | Result |
|---|---|
| Time with a foot off the ground | 1.5 % ± 2.6 % |
| Number of foot lifts in 27 s | 11.4 ± 16.3 |
| Distance moved in 27 s | 0.409 ± 0.377 m |
| Body height | 0.893 m |
| Hip yaw, left / right | +0.1° / -0.2° |

The "±" value is the spread between seeds.

**The robot does not stand perfectly still.** It lifts a foot from time to time and it drifts a little.
The spread between seeds is large. Some seeds move more than others.

The browser viewer (Section 8) shows one run with the sensor noise on:
tilt 0.76°, pose error 0.43°, joint-speed sum 0.30 rad/s, torque RMS 3.73 N·m.

### 3.2 Walking and turning

Tool: `gait_kinematics_probe.py` (6 seeds, command 1.2 m/s) and `measure_turn.py` (8 turn speeds). Measured 2026-10-06.

| Test | Result |
|---|---|
| Forward speed at the command of 1.2 m/s | 1.214 ± 0.002 m/s |
| Heading drift | 8.5° ± 4.3° |
| Turn in place, achieved speed / commanded speed (median of 8) | 17 % |
| Turn in place, number of commands where the robot turned the wrong way | 2 of 8 |

The project bar for turning is a median of 50 % or more and no wrong-way turn.
This policy does **not** meet the bar. **Do not give a turn command in the first tests.**

---

## 4. The `inference()` function

File: `policy_runner.py`. Class: `StandingPolicy`.

```python
from policy_runner import StandingPolicy

pol = StandingPolicy.from_files("deploy_contract.json", "policy.onnx")
targets = pol.inference(
    gyro=...,                # 3 numbers, rad/s
    projected_gravity=...,   # 3 numbers, unit vector
    joint_pos={...},         # 12 joints, rad, dictionary with joint names
    command=(0.0, 0.0, 0.0), # forward, sideways, turn
)
```

### 4.1 Inputs

| Name | Content | Unit | Note |
|---|---|---|---|
| `gyro` | Angular velocity of the body | rad/s | Use the body frame. |
| `projected_gravity` | Direction of gravity in the body frame | none (unit vector) | The value is (0, 0, -1) when the robot is upright. |
| `joint_pos` | Joint angles | rad | Use a dictionary. The key is the joint name. |
| `command` | Wanted speed: forward, sideways, turn | m/s, m/s, rad/s | Use (0, 0, 0) to stand. |

### 4.2 Output

`inference()` gives a dictionary. The key is the joint name. The value is the **target angle** in rad.
The output is not a torque.
The PD controller in each motor makes the torque from the target angle.

### 4.3 What happens inside one call

1. The adapter subtracts the default pose from the joint angles.
2. The adapter joins the 5 input parts in a fixed order. The result is 45 numbers.
3. The ONNX network gives 12 raw numbers.
4. The runner keeps the 12 raw numbers for the next call.
5. The runner makes the targets: `target = offset + 0.25 × raw`. Then it clips each target to the joint limit.

### 4.4 Other methods

| Method | Use |
|---|---|
| `self_test()` | Compare the network with 16 stored input and output pairs. Call it one time before you power on. |
| `reset()` | Clear the joint history and the last action. Call it each time you stand the robot. |
| `default_pose` | Dictionary of the default joint angles. Move the robot to this pose before you start. |
| `joint_names` | List of the 12 joint names. |
| `last_raw_action` | The last raw output. Use it for logs only. |

---

## 5. The 45 observation numbers

| Index | Name | Content | Unit | Sensor noise added in training |
|---|---|---|---|---|
| 0-2 | `base_ang_vel` | Angular velocity of the body | rad/s | ±0.2 |
| 3-5 | `projected_gravity` | Gravity direction in the body frame | none | ±0.05 |
| 6-29 | `motor_pos_history` | 12 joints × 2 frames, minus the default pose | rad | ±0.01 |
| 30-41 | `last_action` | Raw output of the previous cycle | none | none |
| 42-44 | `command` | Wanted speed | m/s, m/s, rad/s | none |

The noise was added on purpose in training.
If your real sensor noise is larger, the policy sees data that it did not see in training.

---

## 6. The 12 action numbers

```
target [rad] = offset + 0.25 × action
```

Then clip the target to the joint limit.
The scale is 0.25 for all 12 joints. An action of 1 moves the joint 0.25 rad (14.3°) from the offset.
The offset is the default pose.

Joint order (the order is the same everywhere):

```
 0 R_hip_pitch   1 R_hip_roll   2 R_hip_yaw   3 R_knee   4 R_crank_A   5 R_crank_B
 6 L_hip_pitch   7 L_hip_roll   8 L_hip_yaw   9 L_knee  10 L_crank_A  11 L_crank_B
```

The ankle has no direct motor. Two cranks (A and B) move the ankle pitch and roll through a 2-RSU linkage.

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

The default pose is a slight crouch. It is not a straight leg.
The left and right signs are opposite. Do not change them.

---

## 7. Motor values used in training

The policy expects motors that behave as shown here.
If your real gains are very different, the same target angle gives a different movement.

| Joints | Motor | Kp [N·m/rad] | Kd [N·m·s/rad] | Torque limit [N·m] |
|---|---|---|---|---|
| `*_hip_pitch`, `*_hip_roll` | RobStride RS04 | 150.0 | 6.0 | 120 |
| `*_knee` | RobStride RS04 | 220.0 | 6.0 | 120 |
| `*_hip_yaw` | RobStride RS03 | 150.0 | 6.0 | 60 |
| `*_crank_A`, `*_crank_B` (ankle) | RobStride RS03 | 22.266 | 1.414 | 60 |

The crank gains are low on purpose. The cranks reach the ankle through a linkage.
If you use the hip gains on the cranks, the foot chatters.

All values are in `motor_params.json`. They are the same as in the v9_caps package.

---

## 8. Run it and watch it

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

The policy assumes a 20 ms cycle.
A slower cycle changes the behavior of the robot.
The runner shows one warning if the median cycle time is more than 20 % wrong.
The runner does not stop. A stop can be more dangerous for a moving robot.

### 8.2 Browser viewer (on the training machine)

The viewer runs the same deploy code as the robot: `obs_adapter` and `policy.onnx`.
It has buttons to push the robot, to lift it, and to switch the sensor noise on or off.
Keep the noise on. The tremble does not show when the noise is off.

```bash
cd mujoco-sim/mjlab
.venv/bin/python3 ../../tools/robot_model/viser_stand_viewer.py \
  --contract ../../deploy_asset/pyg_v11a_stand01/deploy_contract.json \
  --onnx ../../deploy_asset/pyg_v11a_stand01/policy.onnx --port 8102
```

---

## 9. Checks before the motors are on

| Check | Command | Result for this package |
|---|---|---|
| Self-test (network against 16 stored pairs) | `python3 policy_runner.py --contract deploy_contract.json --onnx policy.onnx --self-test` | Maximum difference 2.109e-06. Tolerance 1e-04. Pass. |
| Contract (PyTorch file and ONNX file against the stored pairs) | `analysis/deploy_contract.py --verify` | PyTorch 1.347e-07. ONNX 2.109e-06. Pass. |
| Adapter against the simulator (all 45 numbers, 200 steps) | `tools/deploy/verify_obs_adapter.py --contract <contract> --checkpoint <model.pt>` | Maximum difference 0.000e+00. Pass. |
| Dry run (no robot) | `python3 policy_runner.py ... --dry-run 20` and `python3 example_loop.py --seconds 30` | Runs with no error. |

If the self-test fails, do not power on the motors.
The contract and the network are then not a pair.

The adapter check needs the simulator. Run it on the training machine.

---

## 10. Four errors that people make

> **WARNING**
> A wrong gravity sign makes the robot think it hangs upside down.
> The length check does not find this error.

1. **Gravity sign.** `projected_gravity` points in the direction of gravity. The value is (0, 0, -1) when the robot is upright.
   Many IMUs give the opposite direction (0, 0, +1). If your IMU does this, change the sign before you call `inference()`.
   The adapter checks the first 5 cycles and stops the program if the sign is wrong.
2. **Last action.** Do not put the target angle in the `last_action` part. Use the raw output of the network.
   Both have 12 numbers, so the program shows no error. `StandingPolicy` keeps the correct value for you.
3. **Joint history.** The 24 numbers are `[old frame, 12 joints][new frame, 12 joints]`. They are not grouped by joint.
   The values are relative angles (minus the default pose).
4. **Joint order and names.** Give the joint angles as a dictionary with names. A list makes a wrong order possible, and nobody sees it.

---

## 11. What the code does not do, and what is not solved

### 11.1 Not in the code (on purpose)

- It has **no safety stop and no timeout.** Add them in your own program.
- It does **not set the PD gains.** The values in Section 7 are the training values. The real motor settings are your robot settings.
- It does **not normalize the observation.** The ONNX file does this inside.
- It does **not detect that the robot is in the air.**

### 11.2 Open items on 2026-10-06

| # | Item | Why it matters | Where to read |
|---|---|---|---|
| 1 | The order of the IMU quaternion is not confirmed. It can be (w, x, y, z) or the reverse. | The gravity vector comes from the quaternion. A wrong order gives a wrong `projected_gravity`. Find this error before you run any policy. | `docs/151_imu_quaternion_order_check.md` |
| 2 | The behavior of v11a when a tether lifts the robot is not measured. | An older policy (v7b) swung its legs hard in the air. | `docs/148_air_shake_and_action_delay.md` |
| 3 | The push test of v11a is not done. | The push limit of the policy is not known. | `tools/robot_model/stand_robustness.py` |
| 4 | The policy is not still at command (0, 0, 0). See Section 3.1. | A foot can lift. The robot can drift. | Section 3.1 |
| 5 | The turn command does not meet the project bar. See Section 3.2. | The robot can turn the wrong way. | Section 3.2 |
| 6 | The crank range of ±68.75° is not confirmed on the real robot. The tested range was ±46°. | A large crank motion can hit a hard stop. | `docs/deploy/v9_caps_package_README.md`, Section 5 |
| 7 | The formal experiment note of v11a is not complete. | The final judgment of the training run is not written. | `docs/experiments/2026-10-01_legonly_ab_v11a_stand01_s43.md` |
| 8 | The state of the 12 real motors is not known to the author. | The last record in the briefing is from 2026-09-12. | `docs/000.Real-time Brefing.md` |

---

## 12. Notes on the writing standard (delete before the final version)

- I used the ASD-STE100 writing rules: short sentences, one instruction in each sentence, active voice, the imperative for steps, simple tenses, no contractions.
- I did **not** check the text against the official ASD-STE100 dictionary or a checker program. Some words may not be approved words.
- Technical names (for example ONNX, IMU, PD controller, crank) are used as technical names.
- If you need a formal STE claim, run the text through a checker such as HyperSTE.
