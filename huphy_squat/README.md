# huphy_squat — HUPHY in-place squat task for mjlab

Task id **`Huphy-Squat-Flat-v0`**: the HUPHY LegOnly biped (2-RSU "AB" ankle) holds and changes its
pelvis height on flat ground while being pushed. This package is independent of `mjlab_policy/`
(`pygmalion_mjlab`). It copied the robot assets and the few pieces it needs, and it is built on stock mjlab
(upstream commit 20f10e96) without patches.

Design record (why every number is what it is): `~/huphy/docs/task_design/decisions.md`,
`reward_design.md`, `height_range.md`, `review_plan.md`.

## Install / train / evaluate

```bash
# once (mjlab venv at ~/huphy/mjlab/.venv, stock mjlab 20f10e96 + cu128 torch)
cd ~/huphy/mjlab && uv pip install --python .venv/bin/python -e ../HuphyChan-dev/huphy_squat

# train: 5,001 iterations x 4096 envs (~3.2 h on an RTX 5060 Ti), wandb project "huphy_squat"
cd ~/huphy/HuphyChan-dev/huphy_squat && scripts/train.sh main

# evaluate a checkpoint (tracking, push resistance, margins, reward liveness)
~/huphy/mjlab/.venv/bin/python scripts/eval_squat.py all --ckpt logs/rsl_rl/huphy_squat/<run>/model_<it>.pt --out eval.json

# watch it (stock viewer)
~/huphy/mjlab/.venv/bin/play Huphy-Squat-Flat-v0 --checkpoint-file logs/rsl_rl/huphy_squat/<run>/model_<it>.pt

# interactive viewer: h* control, DR sliders (Random / Clean / Apply & Reset), pushes, thrown box
~/huphy/mjlab/.venv/bin/python scripts/view.py logs/rsl_rl/huphy_squat/<run>/model_<it>.pt [--port 8080]

# tests (CPU linkage tests + GPU env tests)
~/huphy/mjlab/.venv/bin/python -m pytest -q tests
```

## What the task is

| | |
|---|---|
| Command | target pelvis height h* (pelvis z − ground z) ∈ [0.65, 0.895] m; jumps to a new U(0.65, 0.895) every 3–6 s; 20 % of resamples keep the old target |
| Actor obs (43) | IMU gyro (3), IMU gravity (3), 12 encoder angles × 2 frames (24), last action (12), h* (1) |
| Critic obs (186) | true height / error / base velocity / gravity, motor + passive-ankle angles and speeds, torques, foot contacts and forces, live DR values |
| Action (12) | motor target = default + 0.25·a, clipped (hips/knee: 90 % of range, cranks ±0.95 rad) |
| Actuator | explicit PD (real driver gains: hip pitch/roll 135/6, yaw 75/4, knee 130/6, crank 60/2.7) clamped once by the measured T-N curve (peak when braking) |
| Episode | 20 s; ends on tilt > 70°, any non-foot contact with the ground, NaN, blow-up |
| Policy | MLP 512-256-128 ELU, PPO (rsl_rl), entropy 0.01, 50 Hz, physics 5 ms |

Rewards (w · f per step, then × dt):

| term | w | f |
|---|---|---|
| track_height | +2.0 | exp(−(h−h*)²/0.05²) |
| vz_speed | −600 | relu(\|v_z\|−0.1)², off for the first 0.2 s of an episode |
| knee_guidance | −0.2 | Σ\|knee − q*_knee(h*)\| · min(\|h−h*\|/0.05, 1) |
| squat_posture | +1.0 | exp(−mean((q−q*(h*))²/σ²)), σ pitch chain 0.25/0.25/0.15, hip roll 0.12, hip yaw / ankle roll 0.08. q*(h): hips spread 0.05 rad, feet level and straight; below 0.71 m ankle pitch held at 0.43 and the robot sits back (CoM up to 2 cm behind the sole centre at 0.65 m) |
| pitch_symmetry | +0.5 | L/R mirror of hip pitch, knee, ankle pitch, σ 0.1 |
| upright | +1.5 | exp(−(g_x²+g_y²)/0.05) |
| body_ang_vel / angular_momentum | −0.05 / −0.02 | ω_xy² / ‖L‖² |
| lin_vel_xy / ang_vel_z | −0.5 / −0.2 | pelvis v_xy² / ω_z² |
| motor_limits / ankle_limits | −5 / −10 | distance outside the 90 % range (12 motors / 4 passive ankle joints) |
| loop_residual | −1 | Σ relu(loop gap − 0.5 mm)/0.5 mm |
| rod_collision | −2 | substeps with a rod-capsule contact |
| torque_margin | −2 | Σ relu(max-substep \|τ_req\|/T-N bound − 0.9) |
| thermal_effort | −0.02 | Σ(τ/rated)² |
| action_rate / joint_acc | −0.1 / −2.5e-7 | ‖Δa‖² / Σq̈² (motors + ankles) |
| self_collisions / foot_slip / foot_impact | −1 / −0.5 / −0.5 | contact substeps / sole speed² in contact / descent speed² just before touchdown |
| stance_width | −10 | metres of foot width (along the pelvis heading's left axis) outside [w*(h*) − 4 cm, w*(h*) + 12 cm]; w* = 34–41 cm |

Every term is clamped to |w·f| ≤ 50.

Domain randomization (all drawn at every reset, full strength from iteration 0):

| what | range |
|---|---|
| link inertia (pseudo-inertia, CAD uncertainty), 7 link groups | as Rough-Box |
| payload (point mass at pelvis CoM) | −1 … +8 kg |
| PD gains Kp, Kd | × U(0.8, 1.2) per motor |
| motor frictionloss / damping / armature | × U(0.8, 1.2) per motor |
| rod + ankle joint frictionloss / damping | abs U(0, 0.5) N·m / U(0, 0.05) N·m·s/rad per joint |
| foot friction | U(0.3, 1.2) |
| encoder bias (12 motors; same value in obs and in the command) | ±0.015 rad |
| IMU mounting (roll, pitch, yaw) | ±2° each |
| command delay (all 12 motors, one lag per env, held for the episode) | 0 / 5 / 10 / 15 / 20 ms |
| observation delay (IMU + encoders together, one lag per env, held for the episode) | 0 / 20 / 40 ms |
| IMU noise | gyro ±0.2 rad/s, gravity ±0.05 (uniform, every step); encoders ±0.01 rad |
| pelvis push | horizontal force U(0, 60) N for 0.1–0.2 s, every 2–5 s |

Reset: default stance (hips spread 0.05 rad, feet 33 cm apart) + U(±0.05 rad) on the 12 motors; in half the
resets each hip roll also gets U(−0.10, +0.06) rad outward (narrow / wide stances to step out of, feet not re-levelled); the passive rod/ankle joints set from a
precomputed closed-loop table (loops start closed, < 0.1 mm), base upright with random yaw, lowest sole
corner 1 mm above the floor.

## Files

| path | content |
|---|---|
| `huphy_squat/assets/huphy_legonly_ab.xml` | MJCF copy: foot sites and shin motor geoms renamed correctly, rod collision capsules added (header comment lists the changes) |
| `huphy_squat/assets/linkage_table.npz` | crank → passive-joint table + sole-corner height model (`scripts/build_linkage_table.py`) |
| `huphy_squat/robot.py` | joints, gains, default pose, action clip, squat reference table q*(h) |
| `huphy_squat/actuator.py` | one 12-motor PD + T-N actuator group (single torque clip, shared delay, torque-ratio tracking) |
| `huphy_squat/linkage.py` | CPU closed-loop solver used to build the table |
| `huphy_squat/env_cfg.py`, `rl_cfg.py` | the task |
| `huphy_squat/mdp/` | command, observations, events (reset, push, payload, lags), rewards, terminations |
| `huphy_squat/viewer/` | viser viewer (`scripts/view.py`): viewer env cfg with a free box, DR knobs, GUI tabs; not used in training |
| `scripts/` | `train.sh`, `eval_squat.py`, `eval_ood.py`, `view.py`, `build_linkage_table.py` |
| `tests/` | `test_linkage.py` (CPU), `test_env.py` (GPU) |
