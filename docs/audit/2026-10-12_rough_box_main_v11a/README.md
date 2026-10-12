# Audit: training configuration of policy `rough_box_main_v11a`

| Item | Value |
|---|---|
| Document type | Configuration audit (description and findings) |
| Writing standard | ASD-STE100 Simplified Technical English, Issue 9 rules (see 0.3) |
| Date | 2026-10-12 |
| Audited policy | `2026-10-11_23-35-20_rough_box_main_v11a`, checkpoint `model_11999.pt` |
| Audit request | User, 2026-10-12. Scope: environment, stock-code changes, curriculum, domain randomization (DR), robot model, Kp and Kd. |
| Method | Five read-only evidence surveys (A to E). Each fact has a source: `file:line`, commit, document section, or a command output. |

---

## 0. General

### 0.1 Purpose

This document records how the policy was trained.
It gives the logic of each setting and the source of each value.
It also gives the audit findings (section 6).

### 0.2 Path names

| Short name | Full path |
|---|---|
| `MJ` | `mujoco-sim/mjlab` (git submodule, fork of `https://github.com/mujocolab/mjlab`) |
| `ENVCFG` | `MJ/src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py` |
| `CONST` | `MJ/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py` |
| `XML` | `MJ/src/mjlab/asset_zoo/robots/pygmalion/xmls/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml` |
| `RUN` | `MJ/logs/rsl_rl/pygmalion_velocity/2026-10-11_23-35-20_rough_box_main_v11a/` on host 10.8.0.14 |
| `ENV.yaml` | `RUN/params/env.yaml` (the resolved configuration of the run) |
| `MANIFEST` | `RUN/repro/launch_manifest.json` (flags, command, parent, source hashes) |

All other paths start at the repository root `/home/syaro/MikuchanRemote/Human-Pygmalion`.

### 0.3 Writing rules

This document obeys the ASD-STE100 writing rules:

- One topic in each sentence. Descriptive sentences have 25 words or fewer.
- Active voice. Present tense for facts. Past tense for events.
- Technical names (for example "PPO", "MJCF", "armature") are permitted.
- Tables and lists replace long sentences.
- "UNVERIFIED" identifies a statement that the audit did not prove.
- "NO SOURCE" identifies a value that has no record of its origin.

### 0.4 Identification of the policy

| Item | Value | Source |
|---|---|---|
| Task | `Mjlab-Velocity-Rough-Pygmalion` | `MANIFEST` command |
| Launcher | `MJ/analysis/launch_rough_main.py` (commit `baccb01b`) | `MANIFEST` |
| Training script | `MJ/analysis/train_actor_warmstart.py` | `MANIFEST` command |
| Start point | Actor only, from `2026-10-02_23-52-14_legonly_ab_v11a_stand01_s43_p2/model_29998.pt` | `MANIFEST` parent |
| Length | 12,000 iterations, 4,096 environments, 24 steps for each environment in each iteration | `RUN/params/agent.yaml`; console log |
| Seed | 43 | `MANIFEST` command |
| Hardware | One RTX 3090, host 10.8.0.14 | console log |
| Software | mjlab 1.5.0 (editable fork), rsl_rl 5.4.0, mujoco 3.10.0, mujoco_warp 3.10.0.1, warp 1.14.0, torch 2.9.0 | `.venv` package records |
| Third-party packages | rsl_rl and mujoco_warp have no local changes (all RECORD hashes match) | survey A |

### 0.5 Flags of the run (`PYG_*`)

The run uses these environment flags. `MANIFEST` is the master record.

| Group | Flags |
|---|---|
| Robot model | `V2=1`, `ANKLE_MODE=AB`, `MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix`, `MOTOR_MEAS=1`, `TN=1`, `ARM_ABD_DEG=15` (no effect on LegOnly), `SAFE_TARGET_CLIP=1` |
| Start pose | `INIT_BENT=1`, `INIT_MID=1` |
| Observations | `STUDENT_TEACHER=1`, `CRITIC_DR_OBS=1`, `OBS_NAN_GUARD=1` |
| DR | `INERTIAL_DR=1`, `MASS_DR_JSON=tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json`, `DR_START_ITER=2000`, `DR_END_ITER=8000` |
| Curriculum | `GATED_CURRICULUM=1`, `GATE_ERR_RATIO=1.1`, `GATE_FELL_MAX=0.005`, `GATE_MIN_DWELL=0`, `GATE_MAX_DWELL=0`, `GATE_MIN_EPISODES=64`, `GATE_WINDOW=100`, `CMD_VY_STAGES=1`, `FRESH_STEPS=1` |
| Commands | `REL_STANDING=0.1`, `REL_HEADING=0.3` |
| Terrain | `BOX_TERRAIN=1` |
| Rewards | `SOFT_LANDING=1`, `SOFT_LANDING_MODE=half`, `KNEE_EXT=1`, `KNEE_EXT_DEG=25`, `KNEE_EXT_W=2.0`, `REWARD_TERM_CLAMP=50` |
| Terminations | `LOW_BASE_REL=1`, `NAN_TERM=1`, `BLOWUP_TERM=1`, `STAGNATION=1` |
| Optimizer | `ENTROPY_ANNEAL=1` with start and end coefficient 0.002 and start and end iteration 0; `CRITIC_WARMUP_ITERS=300` |

These flags are NOT set: `NO_DR`, `DR_STRONG`, `DR_STARTUP`, `PD_GAIN_DR`, `ACTION_DELAY`, `ACTION_NOISE`, `RESET_JOINT_NOISE`, `RESET_TILT`, `BEYONDMIMIC`, `REL_YAW_ONLY`, `CMD_NORM_CAP`, `UNEVEN`.

---

## 1. Training environment and changes to the stock code

### 1.1 Simulation

| Item | Value | Source |
|---|---|---|
| Physics time step | 0.005 s (200 Hz) | `ENV.yaml:2357`; `MJ/src/mjlab/tasks/velocity/velocity_env_cfg.py:455` |
| Control period | Decimation 4, 0.02 s (50 Hz) | `ENV.yaml:1`; `velocity_env_cfg.py:460` |
| Episode length | 20 s (1,000 policy steps) | `ENV.yaml:2398` |
| Solver | Newton, implicitfast integrator, pyramidal cone, impratio 1.0 | `ENV.yaml:2358-2362` |
| Solver limits | iterations 10, line-search iterations 20, tolerance 1e-8 | `ENV.yaml:2363-2366` |
| Contact buffers | ccd_iterations 100, nconmax 150, njmax 1500, contact_sensor_maxmatch 500 | `ENV.yaml:2352-2367`; `ENVCFG:52-56` |
| Stock defaults (replaced) | Time step 0.002, iterations 100, line-search 50, ccd 50 | `MJ/src/mjlab/sim/sim.py:90-104` |

### 1.2 Terrain

The flag `PYG_BOX_TERRAIN=1` selects `BOX_TERRAINS_CFG` (`MJ/src/mjlab/terrains/config.py:336-352`, commit `9af9bae3`).
All sub-terrains are made of boxes. The set has no heightfield.

Reason: MuJoCo-Warp heightfield contact made the walking foot slip, also on a flat heightfield.
On a flat heightfield, 28 to 43 of 64 robots continued to walk. On flat boxes, 64 of 64 continued.
Source: `docs/experiments/2026-10-11_flat_vs_rough_policy_compare.md` §2b; `docs/162_next_plan_rough_policy.md` step 2.

| Sub-terrain | Share | Geometry at difficulty t (t = row / 9) |
|---|---|---|
| flat | 0.2 | No change |
| pyramid_stairs | 0.2 | Step height 0.1·t m, step width 0.3 m, total rise 0.6·t m |
| pyramid_stairs_inv | 0.2 | Same steps, pit depth 0.6·t m |
| box_random_grid | 0.2 | 0.4 m cells, cell height uniform in ±0.12·t m |
| tilted_grid | 0.2 | 1.0 m plates, tilt uniform in ±15·t degrees, height noise ±0.1·t m |

Grid: 10 rows, one column for each sub-terrain, 8 × 8 m tiles, border 20 m.
Source: `config.py:338-352`; `MJ/src/mjlab/terrains/primitive_terrains.py` (lines 79-81, 449, 507, 1488-1489).

### 1.3 Observations

The actor and the critic have different observations (asymmetric actor-critic).

**Actor (45 values, noise ON):**

| Term | Size | Noise (uniform, ±) | Source |
|---|---|---|---|
| `base_ang_vel` (IMU gyro) | 3 | 0.2 rad/s | `ENV.yaml:797-815` |
| `projected_gravity` (IMU up vector) | 3 | 0.05 | `ENV.yaml:816-834` |
| `motor_pos_history` (12 motor joints, 2 frames) | 24 | 0.01 rad | `ENV.yaml:835-908`; `ENVCFG:157-185` |
| `actions` (last raw action) | 12 | none | `ENV.yaml:909` |
| `command` (vx, vy, wz) | 3 | none | `ENV.yaml:923` |

The actor has no base linear velocity, no joint velocity and no height scan.
`PYG_STUDENT_TEACHER` removes them (`ENVCFG:150-185`). The reason is in `docs/110_prototype_tempmass_student_teacher_report.md` §2.

**Critic (269 values, noise OFF):**
- Body state: base linear and angular velocity, gravity, 16 joint positions and velocities, actions, command.
- Terrain and feet: height scan (187 rays, scale 0.2), foot height, air time, contact, contact forces.
- DR state: friction, mass, CoM, push, dr_factor.
Source: console observation table; `ENVCFG:857-884`; `MJ/src/mjlab/tasks/velocity/mdp/observations.py:55-215`.

### 1.4 Action

| Item | Value | Source |
|---|---|---|
| Type | Joint position target, 12 values | `ENV.yaml:1497-1548` |
| Formula | target = default pose + 0.25 × action | `CONST:1121-1132`; `MJ/src/mjlab/envs/mdp/actions/actions.py:157,218` |
| Clip | Each target is clipped to centre ± 0.9/2 × joint range (`PYG_SAFE_TARGET_CLIP`) | `ENVCFG:118-130`; `CONST:1074-1099` |
| Clip values (rad) | crank ±1.08; hip_yaw ±0.707; R hip_pitch −1.968…0.310; R hip_roll −0.340…1.388; R knee 0.105…1.990 (L is the mirror) | `ENV.yaml:1499-1535` |
| Delay | 0 (no action latency) | `ENV.yaml` actuator `delay_*` = 0 |

### 1.5 Commands

| Item | Value | Source |
|---|---|---|
| Range in use | vx −2.0…2.5 m/s, vy ±1.0 m/s, wz ±1.0 rad/s (see 2.1) | TensorBoard `Curriculum/command_vel/*` |
| Resample time | 3 to 8 s | `ENV.yaml:3089-3117` |
| Standing share | 0.1 (all commands 0) | `ENV.yaml`; `ENVCFG:218-228` |
| Heading share | 0.3, wz = clip(0.5 × heading error) | same |
| Forward-only share | 0.2, vx ≥ 0.3, vy = wz = 0 | same; `velocity_command.py` |

### 1.6 Rewards

There are 22 terms. 21 terms have a weight that is not zero.
The reward manager clamps each weighted term to ±50·dt (±1.0) for each step (`PYG_REWARD_TERM_CLAMP=50`).
Source: `MJ/src/mjlab/managers/reward_manager.py:65-67,134-135`; `ENV.yaml:3347`.

| Term | Weight | Origin | Key parameter |
|---|---|---|---|
| track_linear_velocity | 2.0 | stock | std 0.866 (stock 0.5; `ENVCFG:314`) |
| track_angular_velocity | 2.0 | stock | std 0.707 |
| upright | 1.0 | stock | std 0.447 |
| pose (variable_posture) | 1.0 | stock | standing std 0.05; walking std per joint |
| body_ang_vel | −0.05 | stock | |
| angular_momentum | −0.02 | stock | |
| dof_pos_limits | −1.0 | stock | |
| action_rate_l2 | −0.1 | stock | |
| air_time | 1.0 | stock | 0.05…0.5 s |
| foot_clearance | −2.0 | stock | target 0.1 m |
| foot_swing_height | −0.25 | stock | target 0.1 m |
| foot_slip | −0.1 | stock | |
| soft_landing | −1e-5 | stock | |
| track_lin_vel_progress | 1.0 | fork | `rewards.py:48` |
| stand_still_penalty | −1.0 | fork | deadband 0.3 (`rewards.py:77`) |
| knee_overspeed | −0.5 | fork | limit 19.9 rad/s (`rewards.py:121`) |
| self_collisions | −1.0 | stock | threshold 10 N |
| torque_limit | 0 | fork | skipped |
| contact_force_cap | −0.01 | fork | threshold 420 N, clip 560 N (`ENVCFG:403-404`) |
| foot_impact_velocity | −0.5 | fork | half mode (`ENVCFG:389-424,557`) |
| thermal_effort | −0.02 | fork | rated torque table (`ENVCFG:411-433`) |
| stance_knee_extension | −2.0 | fork | target 25° (`ENVCFG:564-575`) |

### 1.7 Terminations

| Term | Rule | Time-out (bootstrap)? | Source |
|---|---|---|---|
| time_out | 20 s | yes | stock |
| fell_over | tilt > 70° | no | `ENV.yaml:3056-3060` |
| out_of_terrain_bounds | outside the grid (margin 0.3 m) | yes | stock |
| low_base | base < 0.7 m above the terrain under it (median of rays within 0.25 m) | no | fork, `terminations.py:152` |
| nan_detection | not-a-number in the state | no | `ENVCFG:918-919` |
| physics_blowup | joint speed > 100 rad/s, base speed > 10 m/s, or non-finite | no | fork, `terminations.py:175` |
| stagnation | 3 s window, command ≥ 0.2 m/s, progress < 0.10 m and < 0.2 × commanded | no | fork, `terminations.py:105` |

### 1.8 PPO and networks

| Item | Value | Source |
|---|---|---|
| Actor and critic | MLP 512-256-128, ELU, observation normalization ON | `RUN/params/agent.yaml`; `rl_cfg.py:12-70` |
| Algorithm | stock rsl_rl PPO (the fork's SmoothPPO is not used) | `agent.yaml` |
| Epochs, mini-batches | 5, 4 | same |
| Learning rate | 1e-3 start, adaptive (desired KL 0.01, bounds 1e-5…1e-2) | same; `ppo.py:246-249` |
| gamma, lambda, clip | 0.99, 0.95, 0.2 | same |
| Entropy coefficient | 0.002 for all iterations (see 2.4). The value 0.01 in `agent.yaml` is not used. | `MJ/src/mjlab/rl/runner.py:167-225`; console line 275 |

### 1.9 Warm start

1. The launcher loads only the actor from the parent checkpoint. The critic, the optimizer and the iteration counter start new.
   Reason: the rough critic has 269 inputs. The flat parent critic has fewer inputs.
   Source: `MJ/analysis/train_actor_warmstart.py:23-42`.
2. `PYG_FRESH_STEPS=1` sets the environment step counter to 0 after the load. This restarts all step-keyed schedules.
   Source: `train_actor_warmstart.py:47-51`; console line 273.
3. `PYG_CRITIC_WARMUP_ITERS=300` holds the actor weights and the action standard deviation for the first 300 updates. Only the critic learns in this period.
   Source: `train_actor_warmstart.py:56-86`.
   Proof: in checkpoints 0, 100 and 200 the actor weights are equal to the parent weights (difference 0.0).
   Exception: the observation normalizer continues to update in this period. The maximum drift is 8e-4. The code comment says that the normalizer is held. This is not correct (finding F13).

### 1.10 Changes to the stock code

Base: upstream mjlab commit `20f10e96` (2026-06-28). The fork has 137 commits after it.
The fork changes 27 existing files under `src/` (+1,379 / −80 lines). All other changes are new files.

**Changes that are ACTIVE in this run:**

| File | Change | Flag | Commit |
|---|---|---|---|
| `src/mjlab/rl/runner.py` | Entropy coefficient schedule | `ENTROPY_ANNEAL` | `ba200351` |
| `src/mjlab/managers/reward_manager.py:65-67,134-135` | Clamp of each reward term | `REWARD_TERM_CLAMP` | `19b63f6e` |
| `src/mjlab/envs/mdp/events.py:340-398` | Push with a log of the push (for the critic) | `CRITIC_DR_OBS` | `ba200351` |
| `src/mjlab/tasks/velocity/velocity_env_cfg.py` | Terrain-set selector; removal of the actor height scan | `BOX_TERRAIN`; the removal has no flag | `e4b1f6ff`, `9af9bae3` |
| `src/mjlab/terrains/config.py:301-352` | Box-only terrain set | `BOX_TERRAIN` | `9af9bae3` |
| `src/mjlab/tasks/velocity/mdp/terminations.py:105-195` | stagnation, terrain-relative low base, physics blow-up | `STAGNATION`, `LOW_BASE_REL`, `BLOWUP_TERM` | `ea1b6373`, `19b63f6e` |
| `src/mjlab/tasks/velocity/mdp/curriculums.py` | DR ramp; gated command curriculum | `DR_*`, `GATED_CURRICULUM` | `e4b1f6ff` and others |
| `src/mjlab/tasks/velocity/mdp/rewards.py` | 8 new reward terms (table 1.6, "fork") | various | several |
| `src/mjlab/tasks/velocity/mdp/observations.py` | Critic DR-state terms | `CRITIC_DR_OBS` | `e4b1f6ff`, `ba200351` |
| `src/mjlab/tasks/velocity/mdp/velocity_command.py` | Extra error metrics for the gate | none | `0f330399` |
| `analysis/train_actor_warmstart.py` (outside `src/`) | Actor-only load, step reset, critic warm-up | `FRESH_STEPS`, `CRITIC_WARMUP_ITERS` | `7b2bc390` and earlier |

**Changes that are NOT active in this run:**
- `rl/smooth_ppo.py`, `rl/connectome_policy.py`, `config/pygmalion/delayed_action.py`.
- The `evaluation/` package.
- Flag blocks with flags that are not set (list in 0.5).

**Accidental reverts of upstream code (finding F6):** commit `d0c85448` (2026-09-05) replaced four stock files with older content.
This commit is a snapshot with the note "Not reviewed file by file".
It removed upstream `MeshCfg` (#1076), the `ls_parallel` removal (#1063) and #1069.
This run does not use these features. `tests/test_spec_config.py:13` still imports `MeshCfg`. A failure of this test is UNVERIFIED.

**New files:** the Pygmalion task configuration (`config/pygmalion/`), the robot package (`asset_zoo/robots/pygmalion/`, 31 XML files, 15 STL files), and the evaluation tools.

### 1.11 Reproducibility check

| Check | Result |
|---|---|
| 10 source hashes in `MANIFEST` against the local and remote files | All equal |
| All `src/**/*.py` and `*.xml`, local against remote | Equal (4 unused, untracked `*_shell*.xml` files differ) |
| Mass DR file, keyframe file, `validation.json` | Equal (local and remote) |
| Source files changed after the launch | None |
| Git record of the run | None. The remote tree is not a git repository and `RUN/git/` is empty (finding F7). |
| Submodule pointer in the parent repository | `27d16f82` (2026-10-11 13:25). This is older than the code of the run (finding F7). |

---

## 2. Curriculum

The run has three schedules in the environment and two in the optimizer.

### 2.1 Command velocity curriculum

**Stage table** (`ENVCFG:466-488`):

| Stage | vx (m/s) | vy (m/s) | wz (rad/s) |
|---|---|---|---|
| 0 | ±0.8 | ±0.5 | ±0.5 |
| 1 | ±1.2 | ±0.6 | ±0.8 |
| 2 | ±1.6 | ±0.7 | ±1.0 |
| 3 | ±2.0 | ±0.85 | ±1.0 |
| 4 | −2.0…+2.5 | ±1.0 | ±1.0 |

Sources of the values:
- vx and wz: the "2.5 m/s doctrine" for load measurement. `docs/reward_research/2026-07-10_vx25_curriculum.md`.
- vy steps: `docs/reward_research/2026-08-24_command_box_and_transient.md` §5. The final ±1.0 is kept for load coverage (same document, §3).

**Gate logic** (`MJ/src/mjlab/tasks/velocity/mdp/curriculums.py:255-397`, function `commands_vel_gated`):
1. The gate collects the steady velocity error and the fall rate of finished episodes over the last 100 iterations.
2. If the stage time ≥ `max_dwell_iters`, the gate moves up one stage ("forced").
3. Else, the gate tests the performance. It needs a stage time ≥ `min_dwell_iters` and ≥ 64 episodes.
   The gate moves up when the error ≤ 1.1 × the previous error and the fall rate ≤ 0.005.
4. The gate state is not in the checkpoint. Each new process starts at stage 0.

**What occurred in this run:**
- `GATE_MAX_DWELL=0`. The condition in step 2 is always true. The gate moved up one stage on each call, with no test of performance.
- The console shows four forced moves at iteration 0 (log lines 237 and 278-280).
- From iteration 1, the range is stage 4 for all 12,000 iterations.
- Reason: this is the "snap-back" method. It restores the range of the parent, because the gate state is not saved. Source: `docs/103_v2_training_plan.md:124,129`. The launcher copies these flags from the parent (`launch_rough_main.py:58-60`).
- Caution: `ENV.yaml:3102-3112` shows the stage-1 range (vx ±1.2). The file was written after the first forced move. Do not use this file to find the range of training (finding F5).

**Parent (v11a phase 1):** the gate was real (min dwell 800, max dwell 3000). It moved up at iterations 800, 1,600 and 2,400. It forced stage 4 at iteration 5,400.

### 2.2 Terrain curriculum

**Rule** (`curriculums.py:26-81`, function `terrain_levels_vel`):
1. At each reset, the function measures the distance from the start point.
2. Distance > 4.0 m (half the tile size): the robot goes up one level.
3. Distance < |command| × 20 s × 0.5: the robot goes down one level.
4. A robot above level 9 goes to a random level from 0 to 9 (`terrain_entity.py:225-230`).
5. Start levels are random from 0 to 5.

The rule does not use the yaw command. The rule comes from legged_gym and Isaac Lab (`docs/reward_research/2026-10-11_turn_in_place_recovery_literature.md` §2-1).

**What occurred** (TensorBoard `Curriculum/terrain_levels/*`):

| Iteration | 0 | 100 | 300 | 1,000 | 4,000 | 11,999 |
|---|---|---|---|---|---|---|
| Mean level | 3.43 | 4.12 | 5.01 | 5.76 | 5.97 | 5.97 |

- The mean stays at 5.8 to 6.1 after iteration 1,000. The maximum level 9 occurs from iteration 50.
- The flat type has the same mean level as the rough types. Thus the level shows walked distance more than terrain skill (finding F14).
- The start mean (3.43) is higher than the start distribution (2.5). The cause is UNVERIFIED (finding F14).

### 2.3 DR ramp

**Logic** (`curriculums.py:112-202`; `ENVCFG:671-693`):
- dr_factor = 0 before step 48,000 (iteration 2,000).
- dr_factor increases linearly to 1 at step 192,000 (iteration 8,000).
- dr_factor = 1 after this point.

**Terms that the ramp scales:** push velocity, foot friction range, encoder bias, and all seven inertia DR terms. Section 3 gives the ranges.
**Terms that the ramp does NOT scale:** observation noise, push interval, reset pose, terrain.

**What occurred** (TensorBoard `Curriculum/dr_levels/dr_factor`): 0 until iteration 1,999; 0.333 at 4,000; 0.667 at 6,000; 1.0 from 7,999.

**Reason for 2,000 to 8,000:** earlier rough runs used the parent window (from iteration 15,999). Their counter started at 0, so DR stayed off for all 3,000 iterations. Source: `launch_rough_main.py:13-17`; `docs/162_next_plan_rough_policy.md:36`.

### 2.4 Entropy coefficient

The coefficient is 0.002 for all iterations (`runner.py:197-206`).
Reason: with 0.01, the action standard deviation increased from 0.13 to 0.60 and turning in place stopped (51 % to 1 %).
A test series showed a dose response: 0.01 gave no turning, 0.005 gave some, 0.002 gave turning back.
Source: `docs/reward_research/2026-10-11_turn_in_place_recovery_literature.md` §7, §7-4.

Result: the action standard deviation went from 0.130 (iteration 0) to 0.190 (iteration 11,999).

### 2.5 Critic warm-up

See 1.9 item 3. Reason: with a new critic and a low entropy coefficient, standing became bad (feet off the ground 29 %). With 300 warm-up updates, it became 2.2 %.
Source: the same research note, §7-3 and §7-4.

---

## 3. Domain randomization and noise

### 3.1 Summary table

"Ramped" means multiplied by dr_factor (section 2.3). Full strength = dr_factor 1.

| Quantity | Range at full strength | When sampled | Ramped | Implementation | Source of the value |
|---|---|---|---|---|---|
| Gyro noise (actor) | ±0.2 rad/s, uniform, each step | each step | no | `ENV.yaml:797-815` | Upstream default (`velocity_env_cfg.py:90`, commit `a61e0fff`). NO SOURCE for Pygmalion IMU. |
| Gravity-vector noise (actor) | ±0.05 | each step | no | `ENV.yaml:816-834` | Upstream default (`velocity_env_cfg.py:94`). NO SOURCE. |
| Joint position noise (actor) | ±0.01 rad for each frame | each step | no | `ENVCFG:173-179` | Upstream default (`velocity_env_cfg.py:98`). NO SOURCE for encoder resolution. |
| Critic noise | none (corruption OFF) | — | — | `observation_manager.py:436-437` | Design: clean critic (`docs/101` §1b) |
| Encoder bias | ±0.015 rad (±0.86°) for each joint | each reset | yes | `MJ/src/mjlab/envs/mdp/dr/joint.py:209-241` | Upstream default (`velocity_env_cfg.py:251-258`). Measured real zero error is up to 9.4° (finding F2). |
| Foot friction | 0.3…1.2, one value for both feet | each reset | yes (centre 0.75) | `MJ/src/mjlab/envs/mdp/dr/geom.py:117-147` | Upstream default. `docs/16_dr_expansion.md` cites legged_gym. NO SOURCE for a measured floor. |
| Push (base velocity change) | x, y ±0.7 m/s; z ±0.4 m/s; roll, pitch ±0.52 rad/s; yaw ±0.78 rad/s | every 1 to 3 s | yes | `events.py:353-377`; `ENVCFG:614-622` | User instruction 2026-07-06 (from ±0.5 to ±0.7; `ENVCFG:611-613`). NO SOURCE for a hardware impulse. |
| Link mass, CoM, inertia | Table 3.3 | each reset | yes | `MJ/src/mjlab/envs/mdp/dr/body.py:423-541`; `ENVCFG:803-845` | `tools/robot_model/mass_dr.py` (section 3.3) |
| Reset base pose | x, y ±0.5 m; z +0.01…0.05 m; yaw ±180°; no tilt; no velocity | each reset | no | `events.py:111-190` | Upstream default |
| Reset joint pose | none (keyframe) | each reset | no | `events.py:279-313` | `RESET_JOINT_NOISE` not set |
| Terrain | box set (section 1.2) | each reset | no (own curriculum) | `config.py:336-352` | `docs/162` step 2 |

### 3.2 Items that the run does NOT randomize

| Item | Status | Recommendation that exists |
|---|---|---|
| Action latency, motor delay | 0 | 0 to 20 ms (`docs/70_sim2real_pd_gains.md` §5.3; `docs/148_air_shake_and_action_delay.md` §3a) |
| IMU bias, drift, mount angle | none (ideal sensors in `XML:84,270-274`) | Measurement is a TODO (`docs/113:181`; `docs/114` §6 step 5) |
| Kp, Kd | fixed | Kp ×0.9…1.1, Kd ×0.7…1.3 (`docs/70` §5.3) |
| Effort limit, motor strength | fixed | none |
| Joint friction, damping, armature | fixed (measured values, section 4.4) | armature ×0.9…1.1, friction +0…1.5 (`docs/70` §5.3) |
| Contact solref, solimp, restitution | MuJoCo defaults | NO SOURCE |
| Crank and rod masses | not randomized | none |

### 3.3 Mass DR file

File: `tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json`. Script: `tools/robot_model/mass_dr.py`.

**Method:**
1. The script gives each CAD body a 1σ mass uncertainty for its class:

   | Class | σ | Basis |
   |---|---|---|
   | printed, weighed | 0.03 | weighing error |
   | printed, mean | 0.10 | spread of the part survey (`docs/89` §1) |
   | printed, common | 0.03 | batch bias |
   | motor | 0.02 | catalogue tolerance |
   | bearing | 0.02 | catalogue tolerance |
   | fastener | 0.02 | length and count |
   | aluminium | 0.02 | machining tolerance |

2. The script adds a fastener term ("fastener50"). The CAD may contain only 50 to 100 % of the real screws. The script scales CAD fastener mass by 1/c, with c from U(0.5, 1.0). Source: `docs/114_huphy_proxyfix_rotation_dr_audit.md` §5.
3. The script samples 4,000 robots and takes the 95 % interval for each link.
4. The script widens each interval to fixed floors.
   The floors are: link mass ±5 %, CoM ±5 mm, trunk mass +10 %, trunk CoM ±20 mm, shape cap 0.02. The code calls these floors "judgment" (`mass_dr.py:56-69`). They have NO SOURCE in measurement.

**Ranges in use** (mass scale = e^(2α)):

| Link | Mass scale | CoM shift | Shape d |
|---|---|---|---|
| base_link (pelvis) | 0.950…1.150 | ±0.02 m | ±0.0154 |
| hip_pitch_link | 0.950…1.076 | ±0.005 m | ±0.02 |
| hip_roll_link | 0.950…1.078 | ±0.005 m | ±0.02 |
| thigh_link | 0.950…1.095 | ±0.005 m | ±0.0146 |
| shin_link | 0.950…1.107 | ±0.005 m | ±0.02 |
| ankle_pitch_link | 0.889…1.117 | ±0.005 m | ±0.02 |
| foot_link | 0.950…1.138 | ±0.005 m | ±0.02 |

The upper bounds are larger than the lower bounds. Thus the DR makes the robot heavier on average.
Whole robot: about 22.5 kg (all minimum) to 26.2 kg (all maximum), against 23.63 kg nominal (calculation by survey C).

### 3.4 Encoder bias: how it acts

1. The event adds a bias to each joint at reset.
2. The action term subtracts the bias from the target (`actions.py:222-224`).
3. The observation does NOT contain the bias (`joint_pos_rel` with `biased=False`, `MJ/src/mjlab/envs/mdp/observations.py:50-61`).

Thus the bias is an offset of the actuator target. It is not a sensor offset (finding F2).

---

## 4. Robot model

### 4.1 Format

The training uses MJCF (MuJoCo XML). It does not use URDF.
`CONST:428` loads the XML with `MjSpec.from_file`. The code has no URDF loader.
The build script also writes a URDF from the same data (`pygmalion_locomotion/assets/pygmalion_v2/LegOnly_..._proxyfix_loop.urdf`). The training does not read it.
Source: `tools/robot_model/build_robot.py`; `docs/146_where_each_value_lives.md` §0; `docs/145_model_parameter_provenance.md` §2 (URDF and MJCF agree).

File selection: `ANKLE_MODE=AB` and `MODEL_TAG` give the file name `MODEL_TAG + "_loop.xml"` (`CONST:85-89, 211-245`).
The hash of this XML (`5a02f8f9…`) is equal on both hosts and in the baked contract.

### 4.2 Build steps (CAD to MJCF)

| Step | Action | Tool | Source |
|---|---|---|---|
| 0 | Use CAD master "v30" (Fusion 360, 2026-09-01). Do not change the master. | Fusion 360 through MCP | `docs/111_fusion_v30_urdf_rebuild.md` §1 |
| 1 | Dump each solid body: mass, CoM, inertia, material. | `tools/fusion/dump_bodies.py` | docstring |
| 2a | Correct the density of printed parts (mean ratio 0.3287 to CAD, from weighed parts). | `printed_density_snapshot.py` | `docs/89_printed_parts_density_ratio.md` §1 |
| 2b | Set 9 aluminium bodies on each leg to a CAD-density estimate ("tempmass"). These are not weighed values. | `prototype_tempmass.py` | `docs/aluminium_parts_3dprint_masses.xlsx` column D |
| 2c | Scale motor solids to the weighed motor mass: RS04 1.514 kg, RS03 0.9195 kg. | `massprops_fusion.py:60-70` | commit `c394c34` |
| 3 | Add the bodies into rigid links. | `massprops_fusion.py` | commit `bc2526d` |
| 4 | Read motor cylinder positions from the live CAD ("proxyfix"). | `motor_proxies_fusion.py` | `docs/114` §4 |
| 5 | Make visual and hull meshes. | `meshes_fusion_full.py` | `docs/156` B4 |
| 6 | Measure the joint range with a mesh sweep. Apply design caps. | `rom_check.py`; `build_robot.py:121-191` | `docs/107_joint_rom_table.md` |
| 7 | Write MJCF and URDF (capsules, 3-box soles, 4 loop constraints). | `build_robot.py` | lines 352, 418, 541, 1010 |
| 8 | Swap L and R names. Write measured motor values into the XML. | `swap_leg_names.py`; `write_motor_params_xml.py` | `docs/136_left_right_naming_history.md` §5; commit `691ea902` |
| 9 | Run the checks (validation, URDF cross-check, loop check, keyframe, motor values). | `tools/robot_model/*` | `docs/145` |
| 10 | Make the mass DR file. | `mass_dr.py` | `docs/117` §2 |

### 4.3 Kinematics

| Item | Value | Source |
|---|---|---|
| Frame | +X forward, +Y left, +Z up | `build_robot.py:11-13` |
| Hip | Pitch, roll and yaw axes meet at one point, (0, ±0.1237, 0) m. Hip-to-hip distance 247.4 mm. | `XML:87-101`; commit `42afc086` |
| Thigh | 0.3727 m (knee at −0.045, 0, −0.37) | `XML:106` |
| Shank | 0.4909 m (ankle at −0.03, 0, −0.49) | `XML:138-143` |
| Hip to sole | 0.903 m | computed; `validation.json` |
| Foot collision | 3 boxes; sole length 0.260 m (x −0.080…0.180), width 0.100 m | `XML:146-148`; `docs/119_legonly_collision_review.md` §3 |
| Other collision | 15 capsules and boxes; rods, cranks and ankle link have no collision | `XML`; `docs/119` §1 |
| Toe joint | none (rigid foot) | `XML` |
| Ankle linkage | 2-RSU: two cranks on the shin (65.0 and 61.9 mm radius), rods 289.0 and 195.0 mm, ball sites on the foot | `XML:115-151`; `/home/syaro/pyg_fea/fusion/ankle_loop_points_v3_printed.json` |
| Loop closure | 4 `connect` constraints, solref 0.002 (MuJoCo raises it to 10 ms), solimp 0.999/0.9999/0.0001 | `XML:264-267`; `CONST:506-509` |
| Lever ratio | pitch −1.21, roll −1.42 (crank degrees per ankle degree) | `docs/91_closed_loop_ankle_rl.md` §4.1 |
| Degrees of freedom | 2 per ankle; whole model nq 31, nv 30, 12 actuators | computed |
| Reset pose | hip 0.175 rad, knee 0.35 rad, ankle 0.175 rad, base z 0.8975 m (sole 5 mm above the floor) | `CONST:809-858`; keyframe JSON |

### 4.4 Mass and joint properties

**Mass:** total 23.630 kg (per leg 9.136 kg, base 5.358 kg).
The weighed lower body is 23.25 ± 0.5 kg. The model is +1.6 %. Source: `docs/120_design_vs_measured_mass_audit.md` §0.
Mass by class: motors 16.115, printed 2.839, fasteners 1.820, aluminium 1.851, bearings 0.865, other 0.204 kg (`docs/120` §1).

**Per-body values:** `XML:76-231` (table in survey D). The foot is 0.845 kg. This includes a 0.618 kg "tempmass" estimate for the aluminium foot plate. The weighed printed plate is 0.182 kg (`docs/89` §1) (finding F10).

**Joint ranges** (`docs/107_joint_rom_table.md`; design caps):

| Joint | Range (R side; L is the mirror) |
|---|---|
| hip_pitch | −120…+25° |
| hip_roll | −25…+85° |
| hip_yaw | ±45° |
| knee | 0…120° |
| crank A, B | ±68.75° |
| ankle_pitch (passive) | −50…+30° |
| ankle_roll (passive) | ±20° |

**Joint dynamics:**

| Joint group | Motor | Armature (kg·m²) | Viscous (N·m·s/rad) | Friction (N·m) | Source |
|---|---|---|---|---|---|
| hip_pitch, hip_roll, knee | RS04 | 0.0773 | 0.0275 | 0.519 | bench, 2026-09-09/10 |
| hip_yaw, crank A, B | RS03 | 0.0638 | 0.0252 | 0.251 | bench, 2026-09-10 |
| rod hinges (8) | — | 0.0005 | 0.02 | 0 | placeholder, `build_robot.py:464-465`, NO SOURCE |
| ankle pitch, roll (passive) | — | 0 | 0 | 0 | not measured (finding F8) |

Bench method for the motor values: unloaded output shaft, direct CAN at about 4 kHz.
Inertia came from three methods (coast-down, torque-to-acceleration slope, torque-limited acceleration).
The RS04 value is 77.3 ± 0.6 g·m². A resonance check gave 2.48 Hz measured against 2.56 Hz predicted.
Source: `docs/experiments/2026-09-09_motor_friction_retest.md`; `2026-09-09_motor_rotor_inertia.md`; `2026-09-10_motor_value_validity.md`; `2026-09-10_rs03_friction_correction.md`; `docs/131_motor_params_handoff.md` §2-3.

### 4.5 Actuators

| Joint | Motor | Gear | Effort limit | Torque-speed limit |
|---|---|---|---|---|
| hip_pitch, hip_roll, knee | RS04 | 9:1 (not modelled; output-shaft model) | ±120 N·m | RS04 48 V curve: 120.09 N·m to 95 rpm, 0 at 200 rpm |
| hip_yaw, crank A, B | RS03 | 9:1 | ±60 N·m | RS03 48 V curve: 59.68 N·m to 120 rpm, 0 at 200 rpm |

Torque path: `TnPdActuator` computes the PD torque at each 5 ms physics step. It then clamps the torque with the torque-speed curve in the motor quadrant.
Source: `MJ/src/mjlab/asset_zoo/robots/pygmalion/tn_actuator.py:55-86`; curves in `Motor_Spec/RS03_TN_curve_48V.csv` and `RS04_TN_curve_48V.csv` (commit `9fb807f`).

### 4.6 Contact options

| Item | Value | Source |
|---|---|---|
| Foot geoms | condim 3, friction 0.6, priority 1 | `CONST:1024-1036` |
| Other collision geoms | condim 1 (self-collision only) | same |
| Contact solref, solimp | MuJoCo defaults (0.02, 1; 0.9, 0.95, 0.001) | compiled model; NO SOURCE for a calibration |
| Joint limits | soft limit factor 0.9 | `ENV.yaml:630` |

---

## 5. Kp and Kd for each joint

### 5.1 Values in use

The values are the same for the left and the right leg. Gains are fixed (no gain DR).

| Joint | Motor | Kp (N·m/rad) | Kd (N·m·s/rad) | Effort (N·m) | Action scale (rad) | Source |
|---|---|---|---|---|---|---|
| hip_pitch | RS04 | 150 | 6 | 120 | 0.25 | `ENV.yaml:465-481`; `CONST:731-744` |
| hip_roll | RS04 | 150 | 6 | 120 | 0.25 | same |
| hip_yaw | RS03 | 150 | 6 | 60 | 0.25 | `ENV.yaml:338-353`; `CONST:670-681` |
| knee | RS04 | 220 | 6 | 120 | 0.25 | `ENV.yaml:548-562`; `CONST:750-757` |
| crank_A | RS03 | 22.2656 | 1.4141 | 60 | 0.25 | `ENV.yaml:210-225`; `CONST:720-729` |
| crank_B | RS03 | 22.2656 | 1.4141 | 60 | 0.25 | same |
| ankle_pitch, ankle_roll | none (passive) | — | — | — | — | `CONST:1059-1065` |

Control law: τ = Kp (q_target − q) − Kd q̇. Then clamp with the torque-speed curve and the effort limit.

### 5.2 How the values were set

**Hip and knee (150 / 6, 220 / 6): reference gains scaled by mass.**
1. Take the gains of Unitree G1 and K-Bot.
2. Multiply by the mass ratio 51.5 / 35 = 1.47.
3. Hip: G1 Kp 100 and K-Bot Kp 100 give 150. Knee: G1 Kp 150 gives 220.
4. Kd 6 is between the G1 value (3) and the K-Bot value (15).

Source: `CONST:672-676, 736-739, 746-749`; `docs/mujoco/2026-07-06_kp_kd_history.md` §1a.
These values are NOT calculated from inertia.

**Crank motors (22.27 / 1.41): ankle gain through the linkage.**
1. Target ankle gain: 28.5 N·m/rad, Kd 1.81 (Unitree G1 ankle specification, `CONST:627-628`).
2. Two cranks move one ankle axis with lever ratio r = 1.25 (CAD value, `docs/87`).
3. Virtual work gives K_ankle = 2 K_crank / r². Thus K_crank = K_ankle × r² / 2.
4. Kp = 28.5 × 1.5625 / 2 = 22.2656. Kd = 1.81 × 1.5625 / 2 = 1.4141.

Source: `CONST:713-721`.

**Effective ankle stiffness:** survey E calculated it from the loop Jacobian.
Pitch is 30.5 N·m/rad (+7 % to the target). Roll is 22.2 N·m/rad (−22 % to the target).
The Jacobian file is tagged for the v3 model. Its use for v30 is UNVERIFIED (finding F16).

**Code that is not used:** `NATURAL_FREQ` (10 Hz), `DAMPING_RATIO` (2.0) and `STIFFNESS_0x` / `DAMPING_0x` (`CONST:593-604`) are dead code. No file reads them.

### 5.3 History of the gains

| Generation | Date | Gains | Basis | Status |
|---|---|---|---|---|
| 1 | 2026-07-02 | hip and knee 28 / 1.76 | Kp = rotor armature × ω², rotor only. Hip bandwidth was 0.56 Hz (too soft). | replaced |
| 2 | 2026-07-02 | hip and knee 400 / 28 | chirp test | replaced |
| 3 | 2026-07-06/07 | hip 150 / 6, knee 220 / 6 | reference gains × mass ratio (user: "hip gain too big") | IN USE |
| Ankle | 2026-07-04 | 28.5 / 1.81 per ankle axis | G1 specification | IN USE (through cranks) |
| Crank | 2026-08-23 | 22.27 / 1.41 | formula in 5.2 | IN USE |

Source: `docs/mujoco/2026-07-06_kp_kd_history.md`; `CONST:610-632`; commits `42afc086`, `895cb157`.

**Tests that kept Kd = 6:**
- Critical-damping Kd (35 / 14 / 16, `PYG_BEYONDMIMIC`) was tested on 2026-07-08 in a controlled A/B. Joint torque increased 2.2 to 3.5 times. Foot force peak increased from 2.1 to 6.1 body weights. The test rejected it. Source: `docs/53_bc_kd_controlled_ab.md`.
- After the new motor armature (2026-09-11), a Kd increase was proposed and then withdrawn. Instruction: "do not touch Kp / Kd". Source: `docs/137_training_handoff_motor_params.md` §1, §4-§5.

**Resulting dynamics (calculations from `docs/137` §5 inertia):**

| Joint | Natural frequency | Damping ratio |
|---|---|---|
| hip_pitch | about 1.5 Hz | about 0.19 |
| knee | 3.9 Hz | 0.33 |
| hip_roll | 6.4 Hz | 0.80 |

### 5.4 Validation

| Check | Result | Source |
|---|---|---|
| PD fit on this run | NOT DONE | — (finding F20) |
| PD fit on earlier runs (same gains) | R² = 1.00 on all joints, fitted Kp and Kd equal to the configuration | `docs/53` lines 55-62; `docs/55_init_pose_straight_vs_bent.md` |
| Sim against bench motor (kp 20, Kd 1, no load) | resonance 2.5 Hz in both; peak 1.41 real, 1.32 sim | `docs/129_mujoco_motor_parameters.md` |
| Bench knee step at kp 220, Kd 6 (no load) | overshoot 1.3 %, 39 ms to 90 % | `docs/137` §4 |

**Gap between sim and real gains (finding F1):**
- Sim: 150 to 220. Real robot configuration: kp 10, kd 1 for all 12 joints. This is a 15 to 22 times difference.
- Source: `docs/136_legonly_ab_v3_keyfix_sim2real_gaps.md:37-38`; `docs/150_real_vs_sim_joint_response_plan.md:20`.
- The audit did not find the file `config/robot_v1.0.yaml`. The value kp 10 is UNVERIFIED against that file.
- `docs/150` §5-1 gives three options. No document records a decision.

---

## 6. Audit findings

Severity: HIGH = can cause failure on hardware. MEDIUM = reduces trust in a result or in reproducibility. LOW = documentation or minor error.

| ID | Severity | Finding | Evidence | Recommended action |
|---|---|---|---|---|
| F1 | HIGH | Real-robot gains (kp 10, kd 1) are 15 to 22 times lower than the training gains (150 to 220). No decision is recorded. | §5.4; `docs/150:20` | Decide the hardware gains before any deployment. Find and check `config/robot_v1.0.yaml`. |
| F2 | HIGH | Encoder bias DR is ±0.86°. The measured zero error is up to 9.4° (left knee). The bias acts on the target only, not on the observation. | §3.1, §3.4; `ENVCFG:706-709`; `docs/109:56` | Apply the measured zero offsets (tasks #2, #14). Widen the DR to the measured residual. |
| F3 | HIGH | The run has no DR for latency, IMU bias, Kp/Kd, motor strength or joint dynamics. | §3.2 | Measure latency and IMU bias. Then add these DR terms (code exists for most). |
| F4 | MEDIUM | Observation noise values are upstream defaults. No Pygmalion sensor measurement supports them. | §3.1 | Measure EBIMU gyro noise and encoder noise. Compare with ±0.2 rad/s and ±0.01 rad. |
| F5 | MEDIUM | The command gate is bypassed (`MAX_DWELL=0`). `ENV.yaml` records the stage-1 range, not the range of training. | §2.1 | Write the effective range in the run note. Save the gate state in the checkpoint, or log the final range. |
| F6 | MEDIUM | Commit `d0c85448` reverted three upstream changes by accident. One test may fail at import. | §1.10 | Restore the upstream code. Run `tests/test_spec_config.py`. |
| F7 | MEDIUM | The run has no git record. The submodule pointer in the parent repository is old (`27d16f82`). | §1.11 | Commit the submodule pointer. Write the mjlab commit hash into `MANIFEST` at launch. |
| F8 | MEDIUM | Passive ankle joints have zero friction, damping and armature. Rod hinges use placeholders. | §4.4; open task #20 | Measure ankle bearing friction. Replace the placeholders. |
| F9 | MEDIUM | The ankle loop geometry comes from a v3 file. Nobody re-extracted it from the v30 CAD. | survey D §7; `build_robot.py:70,412` | Extract the loop points from v30 and compare. |
| F10 | MEDIUM | The foot mass uses a 0.618 kg aluminium estimate ("tempmass"). The weighed printed plate is 0.182 kg. | §4.4; `docs/112` §5 | Weigh the real foot parts. Update the model. Do not use "tempmass" checkpoints on hardware (`docs/114` §5). |
| F11 | MEDIUM | The hardware motor map lists all 12 joints as RS03. The model uses RS04 for hip pitch, hip roll and knee. | `docs/136` §3 | Confirm the motor type on each joint of the real robot. |
| F12 | LOW | The site `left_foot` is on `R_foot_link` and `right_foot` is on `L_foot_link`. The effect on left/right rewards is UNVERIFIED. | `XML:149,222` | Check each reward that uses these sites. |
| F13 | LOW | The critic warm-up comment says that the observation normalizer is held. It is not held (drift ≤ 8e-4). | §1.9 | Correct the comment, or hold the normalizer. |
| F14 | LOW | The terrain level shows walked distance more than terrain skill. The start level is higher than the start distribution (cause UNVERIFIED). | §2.2 | Use the same-terrain evaluation (`terrain_eval.py`) for terrain skill, not the curriculum level. |
| F15 | LOW | The mass DR file has nominal masses that differ from the XML (pelvis 5.42 against 5.36 kg). The mass-property file of the final build is not in the repository. | §3.3; survey D §7 | Add the build input to the repository. Make the DR file from the same snapshot. |
| F16 | LOW | Effective ankle roll stiffness is about 22 % below the design target (one scalar lever for two axes). | §5.2 | Use the loop Jacobian of v30 to set the crank gains, if the ankle roll response is important. |
| F17 | LOW | Contact solref and solimp are MuJoCo defaults with no calibration. | §4.6 | Record this as a known model limit. |
| F18 | INFO | Dead code: `NATURAL_FREQ`, `DAMPING_RATIO`, `STIFFNESS_0x`, `DAMPING_0x`, `ElectricActuator.velocity_limit`. | §5.2; survey E | Remove or mark as not used. |
| F19 | LOW | Some documents have old values: 21.086 kg (`docs/111`, `docs/156`); RS03 friction 0.250 (code 0.251). | survey D §7 | Correct the documents. |
| F20 | LOW | This run has no PD regression and no q / q_target analysis. | §5.4 | Run `analysis/analyze_qtarget.py` on the load measurement data. Commit this tool (it is not tracked). |

---

## 7. Evidence surveys

Five read-only surveys made the facts in this document on 2026-10-12:

| Survey | Subject | Main sources |
|---|---|---|
| A | Environment and stock-code changes | `ENV.yaml`, `agent.yaml`, `git diff 20f10e96 HEAD -- src/`, source hashes |
| B | Curriculum | `curriculums.py`, `ENVCFG:466-693`, TensorBoard tags, console log |
| C | DR and noise | `ENV.yaml` events and observations, `MJ/src/mjlab/envs/mdp/dr/*`, mass DR file |
| D | Robot model | `XML`, `tools/robot_model/*`, `docs/107`, `docs/111`-`docs/120`, `docs/145`, `docs/146` |
| E | Kp and Kd | `CONST`, `ENV.yaml` actuators, `docs/53`, `docs/137`, `docs/150`, `docs/mujoco/2026-07-06_kp_kd_history.md` |

Values marked "computed" come from the compiled model (mujoco 3.10.0) or from arithmetic by the survey. They are not from a project document.
