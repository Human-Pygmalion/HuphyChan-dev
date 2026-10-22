# Walking control stack audit against the training contract (2026-10-22)

**Question.** Which flaws in the real-robot control stack (not the policy, not the hardware) could explain
(1) veering to one side while walking straight, (2) a gait that differs from the simulator, (3) jitter or tremor?

**What was read.**
- Robot side: HUPHY `origin/biped` at `32dd6e6` (2026-10-21), read-only. Extracted with `git archive` into the session scratchpad. The HUPHY working tree was not touched. Paths below are relative to the repository root, for example `src/huphy/control/generalized_walk.py:167`.
- Training side: `origin/vibecoded-audit-20261012` (`ebeec10`), folder `vibecoded-audit-20261012/`. Paths are written `T:deploy/generalized_walk/README.md:30`, `T:pygmalion_mjlab/robot.py:40`.
- Checkpoint on the robot: `config/policies/generalized_walk.pt`, sha256 `7bbd5cbc…fe0fbe00`. It is the same file as `terrain10_2048_model_37400.pt` (checked).

**How strong each statement is.** Everything is from reading code and from small offline probes run on this PC (a torch comparison, a static sensitivity probe, a timing micro-benchmark). Nothing here was measured on the robot. Statements about the robot's physical state (where the sensor sits, how the joint zeros were set) are marked "unverified". These probes are not evaluator statistics, so none of them is a judgement about gait quality.

**Important limit.** I audited the committed `origin/biped`. The board that ran the tests may carry local edits (calibration files, `robot.yaml`, command-line options). Section 6 lists what to fetch from the robot to close that gap.

---

## 1. Summary of findings

Severity means "how likely this is a real cause given the evidence, times how large the effect would be". Symptom numbers: 1 veering, 2 gait differs, 3 tremor.

| ID | Severity | Symptoms | Finding | Evidence |
|---|---|---|---|---|
| F1 | **High, only if the ROS path was used** | 2, 3, maybe 1 | In the two-machine ROS path the four ankle crank angles never reach the policy. `control_node` publishes `ankle_pitch/ankle_roll`; the policy reads `ankle_a/ankle_b` and silently gets 0.0. Reproduced offline. Unit tests miss it because they build a fake observation keyed by the policy's own joint list. | §2.1 |
| F2 | **High (unverified)** | 1, 2 | Joint zero offsets are placeholders. Every joint, both legs: `sign=1`, `offset_deg=0`. The file's own note says nothing was measured. The only guard against this (`zero_reference` non-empty) was satisfied by the string `"0"` in commit `394f5d3` ("json config"). The limit values are the training model's ranges, not measured hard stops. Training randomizes only ±0.57° of encoder bias. | §4.1 |
| F3 | **High (unverified)** | 1, 2 | The inertial sensor (IMU) has no mounting-rotation or bias handling. `robot.yaml` labels it `right_leg` (probably a stale note). Axes are assumed to equal the robot body axes (x forward, y left, z up). The policy's gravity input is very sensitive: its trained spread is only 0.03–0.05, so a 2° tilt is about 0.7 of one spread. | §2.1, §4.2 |
| F4 | **High with `--hz 100`, medium with `--hz 50`** | 2, 3 | Policy timing differs from training. At `--hz 100` the policy is called every second loop cycle, counted by cycles not by time (`control/loop.py:192-196`); the 2026-10-21 timing note measured 60.6 Hz loop and 33±10 ms between policy calls (training: exactly 20 ms). Even at 50 Hz the joint angles the policy sees are about one period old. Part of the slow loop is avoidable work: ankle kinematics solved 4 times per cycle although the policy never uses them (§3.3). | §3 |
| F5 | Medium | 2, 3 | The inertial sensor data is late and not time-aligned with the joints: about 9.6 ms of serial transmission per packet at the default 115200 baud with 15 fields (95% of the link), no age or validity check anywhere in the control path, and the attitude comes from the sensor's own filter, not exact projected gravity. | §3.4 |
| F6 | Medium | 2 | The command guard narrows the training action clip by 3° on every side of every joint (knee full extension 9° instead of 6°, hip back-swing 14.75° instead of 17.75°). The training clip has no such margin. | §2.2 |
| F7 | Medium | 2, 3 | ROS path timing tolerances are far looser than training: 60 ms stale joint state accepted, gravity input never age-checked, frozen sensor values look fresh, timer jitter on a busy single-threaded executor. | §3.5 |
| F8 | Medium / low | 3 | A missing motor reply leaves that joint's last angle in the observation with no flag; the command that went with it may also have been lost. Measured loss in earlier runs: 1.4–3.1%. | §3.6 |
| F9 | Low | – | No rate limiter, smoothing or filter on the targets (answers checklist item 2). The 50° "jump guard" compares target with measured position, so it is a tracking-error cap, not a rate limiter, and is almost never active. | §2.2 |
| F10 | Low | – | Gains match the contract exactly. Twelve-bit message quantization changes them by −0.28% to −0.69%. Position step 0.022°. Zero-command offsets (velocity −0.004 rad/s, torque −0.03 Nm) are negligible. | §2.3 |
| F11 | Low | – | The loop's deadline is taken from each cycle's own start, so the real rate is slightly under the target (about 0.1–0.3 ms per cycle). | §3.2 |

What was checked and found **correct** is in §5.

---

## 2. Observation and action path (checklist items 1 and 2)

### 2.1 Observation: 44 entries

Contract: `T:deploy/generalized_walk/README.md:40-57` (layout), `T:pygmalion_mjlab/env_cfg.py:259-277` (terms).
Implementation: `src/huphy/control/generalized_walk.py:153-188` (`observation_vector`), `:206-231` (`motion`).

| Check | Contract | Implementation | Result |
|---|---|---|---|
| Order of 44 entries | gyro 3, gravity 3, old joints 12, new joints 12, previous action 12, vx, wz (README:45-53) | `generalized_walk.py:167-183`, names at `:138-143` | Match |
| Joint order, right leg first | `hip_pitch, hip_roll, hip_yaw, knee, ankle_a, ankle_b`, right then left (README:36-38) | `ORDER`, `:51-64` | Match |
| Units | radians, rad/s | `math.radians` at `:167` and `:177-180` (robot reports degrees) | Match |
| Default-pose subtraction | motor position minus default pose (README:49-50), default = `INIT_STATE` (`T:robot.py:110-135`) | `:177-180` subtracts `DEFAULT_POSE_DEG` | Match (numbers in §2.2) |
| History frames | old, then new; first sample initialized from current position (README:55-57) | `:170-176` order old, new; `:223-224` fills the first old frame with the current angles; `:228` updates after inference | Match |
| Previous action | raw network output | `:226` stores the raw output (`joint_targets` clips only the target list, not the stored action) | Match (unclipped) |
| Command slots | `[vx, wz]`, no `vy`, unscaled (README:33; `T:generalized_cfg.py:29-32` takes columns 0 and 2) | `:183`, `run.py:432-434`; fixed for the whole run | Match |
| Normalization | `(x−mean)/(std+0.01)`, no clipping | `control/rsl_rl.py:53`, `:188-193` | Match. Numpy loader against torch network on 200 random inputs: max difference 1.9e-6 (offline probe) |
| Gyro units and frame | simulator gyro at site `imu_in_base`, no rotation (`T:assets/pygmalion_legonly_ab.xml:84,271`) | `sensors/ebimu/protocol.py:156` passes the sensor's gyro (degrees per second) unrotated | Match **only if** the sensor's gyro axes equal the robot body axes. Unverified |
| Gravity sign and quaternion order | simulator gives `−(world z axis in site frame)`, so `(0,0,−1)` upright (mjlab `projected_gravity_from_sensor`) | `protocol.py:112-130` reorders `(z,y,x,w)` to `(w,x,−y,−z)`; `sensors/base.py:155-168` computes `Rᵀ(0,0,−1)` | Correct. I re-derived it: the flip is a 180° turn about x, so the code returns `D·R_Eᵀ·(0,0,1)` with `D = diag(1,−1,−1)`, which is the gravity direction in the sensor's gyro axes. The earlier x-sign error was fixed in `bb558c6` on 2026-10-15, so runs from 10-20 onward include the fix |

Two entries are not code bugs but unverified physical assumptions, and they are the highest-value checks for symptom 1:

- **Sensor mounting (F3).** `config/robot.yaml:375` says `mount: right_leg`. No code reads that label, and it is probably stale: it dates from August (`6d5fe34`, `3dde252`), and `docs/full_robot.md:217-228` says the sensor moved from a leg to the torso and that `mount` is now only a note. The policy needs the sensor on the pelvis (simulator site: `base_link`, offset (0, 0.007, −0.0695) m). If it sits on a thigh, gyro and gravity follow the leg swing and the policy cannot work as trained. If it sits on the pelvis but tilted, there is a constant offset. Nothing compensates a constant offset (no zeroing step, no mount rotation setting). `docs/sim2real/2026-10-14_dr_reference_latency_imu.md` already noted initial gravity x of 0.06–0.12 (3.5–7°) and could not tell tilt from mounting.
- **How sensitive the policy is.** Normalizer spread (`std`) from the checkpoint: gravity x 0.053, y 0.047, z 0.029; gyro 0.45, 0.49, 0.70 rad/s. A mount tilt of 2° gives a gravity offset of 0.035, which is 0.66–0.74 spread. Training noise is zero-mean per step, and no gravity bias was randomized (README:207-209).

  Static probe (default pose, previous action 0, `vx` = 0.5, targets in degrees, change from the unbiased case; this is a local response of the network, not a gait result):

  | Input offset | Right leg [hip pitch, hip roll, hip yaw, knee, crank A, crank B] | Left leg |
  |---|---|---|
  | gravity y +0.035 (2° roll) | 0.11, −0.48, −0.79, 1.01, 0.23, −0.51 | −0.12, −0.80, −1.04, 1.23, 0.13, 0.36 |
  | gravity x +0.035 (2° pitch) | −0.25, −0.05, −0.33, 0.27, −1.85, 2.08 | 0.06, −0.04, 0.55, −0.73, 1.55, −1.92 |
  | gyro z +0.05 rad/s | −0.27, −0.09, 0.08, 0.13, −0.02, −0.06 | −0.22, −0.14, −0.03, 0.06, −0.07, −0.03 |
  | right hip roll reads +2° | −0.41, −0.66, −0.90, 0.40, −0.42, −0.08 | −0.04, 0.53, 0.15, 0.19, −0.02, 0.21 |

  The gravity-y row moves both legs' hip yaw in the same direction (about −0.8 and −1.0°). Both legs yawing the same way is a turning input. This is a plausible route from a small mount roll error to veering. Previously measured static gyro bias (`docs/sim2real/2026-10-14_dr_reference_latency_imu.md` §2) was at most 0.0073 rad/s (z: 0.42°/s), so the gyro bias route is smaller than the tilt route unless the new run differs.

- **The ROS path loses the ankle observation (F1).** `ros/huphy_ros/huphy_ros/control_node.py:141` takes joint names from `robot.joint_names`. `Biped.joint_names` (`robots/biped.py:154-157`) is built from `Leg.JOINT_NAMES` (`robots/leg.py:92`), which lists `ankle_pitch` and `ankle_roll`, not `ankle_a`, `ankle_b`. `/huphy/joint_states` therefore has no crank angles. `policy_node.py:227` reads `ankle_a.pos` and `ankle_b.pos` with a default of 0.0 (also `generalized_walk.py:170`), and `_on_start` (`policy_node.py:160`) starts the approach from 0 for them.
  Offline reproduction (published names, robot exactly at the default pose, both legs): crank entries of the new-frame block come out as −0.144, +0.147, +0.144, −0.147 rad instead of 0. They are constants, so the network never sees any crank motion. The unit tests (`tests/test_ros_convert.py:25`) build the fake observation from `gw.ORDER`, which is why they pass. The direct `huphy-run` path is not affected, because `Biped.get_observation` (`robots/biped.py:276-282`) includes `ankle_a.pos`. Check with `ros2 topic echo /huphy/joint_states --once` and look at the `name` list.

### 2.2 Action path

Contract: `T:deploy/generalized_walk/README.md:30`, `T:pygmalion_mjlab/robot.py:37-53,110-135`.
Implementation: `generalized_walk.py:71-84,96-109,117,191-203`; `robots/leg.py:596-648`; `safety/guards.py:107-150`; `safety/limits.py:66-82`.

| Item | Contract | Implementation | Result |
|---|---|---|---|
| Scale | 0.25 rad per action unit, all 12 joints | `ACTION_SCALE = 0.25` (`:117`), used at `:202` | Match |
| Target formula | `default + 0.25·a`, then clip | `:202-203` | Match |
| Default pose, degrees | R hip pitch −10.0268, knee 20.0535, crank A 8.2480, B −8.4261; L mirrored (0.175 rad = 10.0268°, 0.35 rad = 20.0535°, ±0.14395/0.14706 rad) | `:71-84` | Match to 0.011°. The right/left crank values differ by 0.004° and 0.011° because the simulator solved its own closed loop (`T:robot.py:117,125-128`); negligible |
| Policy clip | `ACTION_CLIP` = centre ± 0.45 × range, `T:robot.py:40-53` | `LIMITS_DEG` `:96-109` | Identical (hip pitch −112.75..17.75, hip roll −19.5..79.5, yaw ±40.5, knee 6..114, crank ±61.88; left mirrored) |
| Extra clip on the robot | none | `command_margin_deg: 3.0` (`robot.yaml:349`) applied inside `guards.apply` → `limits.clamp` (`leg.py:619-626`), against the calibration `limits_deg`, which equal the policy clip (commit `b1ac519` filled them from the "training model range") | **Narrower than training by 3° on both sides** |
| Rate limiter, smoothing, filter | none | none found. The only target change besides the clips is the jump guard: `clamp_jump(target, measured, 50°)` (`guards.py:85-104`), a cap on target minus measured position, not on target change per cycle | Match (nothing extra) |
| Ankle mapping | crank angles A and B are the actuated joints; the policy's output is the crank angle itself | `ankle_a`/`ankle_b` sent in motor space, no kinematics on this path (`leg.py:568-578`) | Match. Only valid if the crank zero and sign equal the simulator's; see F2 |

Effective target range in action units after the 3° margin (default pose subtracted, 14.32° per unit):

| Joint (right leg) | Training clip | With 3° margin |
|---|---|---|
| hip pitch | −7.17 … +1.94 | −6.96 … +1.73 |
| hip roll | −1.36 … +5.55 | −1.15 … +5.34 |
| hip yaw | ±2.83 | ±2.62 |
| knee | −0.98 … +6.56 | −0.77 … +6.35 |
| crank A / B | −4.9 … +3.7 / −3.7 … +4.9 | −4.7 … +3.5 / −3.5 … +4.7 |

The knee at full extension (swing leg before touchdown) and the hip back-swing are the two places where walking could plausibly reach the margin. `policy_log` records `tgt` (before the margin), so the fraction of rows within 3° of the limits tells how often the margin acts (§6).

### 2.3 Gains and message quantization

Contract: hip pitch/roll 135/6, yaw 75/4, knee 130/6, crank 60/2.7 (`T:deploy/generalized_walk/README.md:166-173`, `T:robot.py:161-166`).
Implementation: `config/robot.yaml:405-411,422-427`. All 12 motors match. `huphy-run` does not override gains (`run.py:370-378`); `--gain-scale` multiplies all (`run.py:360-377`), default 1.0. **The policy log does not record gains**, so the command line must be checked (§6).

Message encoding (`motors/robstride/codec/mit.py:83-138`, ranges `motors/robstride/tables.py:101-120`), computed offline:

| Quantity | Effect |
|---|---|
| Stiffness `kp` (12 bit, truncated) | 135 → 134.31, 75 → 74.48, 130 → 129.43, 60 → 59.83 (−0.28% … −0.69%) |
| Damping `kd` (12 bit, truncated) | 6 → 5.983, 4 → 3.981, 2.7 → 2.686 (−0.28% … −0.51%) |
| Position target | 16 bit over ±12.57 rad: step 0.022°; zero encodes as −0.011° |
| Zero velocity / zero torque | encode as −0.0037 rad/s (RS04), −0.0049 (RS03) and −0.029 Nm (RS04), −0.015 Nm (RS03); equivalent to 0.01–0.02° of position |

All negligible. (The earlier note's 22.266 → 22.222 effect on the old ankle gain no longer applies.)

Not visible from the repository: motor-side current limit, velocity filter and any protective settings inside the motor firmware. The training model assumes ideal spring-damper torque clipped by the measured torque–speed curve. Open item.

---

## 3. Timing (checklist item 3)

### 3.1 What runs, in order (direct `huphy-run`)

`ControlLoop.step` (`control/loop.py:314-335`), one policy period at `--hz 50`:

1. `robot.get_observation()` (`:320`) returns the **cached** motor state from the previous cycle's replies.
2. The policy motion runs (`:327`): reads the **latest** sensor packet, builds 44 inputs, runs the numpy network, clips targets.
3. `robot.send(build_commands(...))` (`:329`). In the default `seq` mode (`run.py:188-195`, `robots/biped.py:416-436`) each of the 6 commands per leg is sent and its reply awaited before the next, the two legs in parallel (separate CAN channels `canr`, `canl`).
4. `robot.collect()` (`:330`).
5. Sleep until the next deadline.

### 3.2 Rates and sleeping

- Policy rate: `POLICY_HZ = 50` (`run.py:73`), `--hz` default 50 and must be a whole multiple (`run.py:173-180,323-326`). With `--hz 100` the policy runs through `held()` (`loop.py:171-204`): it is called on every second **cycle**, not every 20 ms. If the loop cannot keep up, the policy interval grows. `q(t-1)` is the angle at the previous policy call (`generalized_walk.py:228`), so the history stays consistent, but the time between the two frames is irregular.
- Sleeping (`loop.py:126-148,506-521`): sleep until 3 ms before the deadline, then spin. The deadline is `cycle_start + period` where `cycle_start` is taken at the top of each cycle (`:390,409`), not accumulated from the first cycle. Each cycle therefore adds its own overhead to the period (F11, low). The docstring says the deadline is absolute; it is absolute only within one cycle.
- Reply timing measured on the robot earlier (`docs/can_communication.md`, seq mode, torque on): per-motor reply p50 0.70 ms, p95 1.19 ms, p99 1.73 ms, 0% lost. The 2026-10-21 run measured all 12 replies in a cycle spanning an average of 6.6 ms.

### 3.3 Read-then-act ordering and latency

Joint angles in the observation were taken when each motor answered the **previous** cycle's command, then used one period later:
- age of the joint angles at the time of the inference: about 15–20 ms at 50 Hz (period 20 ms minus the 0–6.6 ms position of that motor in the previous send sequence);
- commands then leave staggered over 0–6.6 ms (hips first, ankles last), since the order is hip pitch, hip roll, hip yaw, knee, crank A, crank B (`leg.py:566-577`, dict order);
- so sensing-to-actuation is roughly 16–27 ms every cycle.

Training (`T:generalized_cfg.py:255-258`, `T:robot.py:177-178`): joint-angle lag drawn from {0, 1} policy steps (0 or 20 ms) plus actuator delay 0–5 ms, so a mean near 12 ms with 25 ms the worst case. The robot always sits at the worst corner of the trained range, never at the easy corner. This is inside the training range, so it is not an error, but it is a systematic offset (F4, medium). The earlier simulation note (`2026-10-21_generalized_walk_jitter_analysis.md` §3, H8) found 10–40 ms of command delay harmless for standing; walking was not tested there.

With `--hz 100`, the 2026-10-21 timing note (file `1020generalizedwalk00.csv`) measured 60.6 Hz loop, 16.5 ms mean cycle, and 33.0 ± 10.4 ms between policy calls, up to 111 ms. That is the largest timing departure from training found, and it only needs a command-line option to avoid (`--hz 50`; earlier logs show 20.0 ms median with p99 about 21 ms).

**Avoidable compute inside each cycle** (offline micro-benchmark on this PC, 12th-gen i5; the motion board is slower by an unknown factor, the earlier note implies a large one since the policy call alone added 1.4 ms there against 0.03 ms here):

| Work per cycle (both legs) | Time here | Needed by the policy? |
|---|---|---|
| Ankle forward-kinematics solve inside `build_commands` → `_as_joint_space` (`leg.py:645-647,727-761`), done only to fill `last_sent` for logs | about 0.3–0.4 ms, about 80% of `build_commands` | No |
| Ankle forward-kinematics solve in `collect` → `_update_ankle_pose` (`leg.py:494-516`) | 0.06 ms | No (`ankle_pitch.pos` is unused by this policy) |
| `get_observation` including ankle Jacobian solve (`leg.py:480`) | 0.085 ms | Partly |
| Whole `build_commands` (guards, packing) | 0.53 ms | Yes, but mostly the kinematics |
| Observation vector + network | 0.006 + 0.029 ms | Yes |

On this PC the avoidable part is about 0.5 ms of about 0.7 ms Python time per cycle. If the board is 10–40 times slower, that is 5–20 ms, the same size as the "8–9 ms of unexplained cycle time" the 2026-10-21 note could not account for. This is a lead, not a measurement on the board: profile one cycle there (`cProfile` around `loop.step`). Other candidates named in that note remain (wake-up error, logging, network, thread switching with the sensor and CAN reader threads).

### 3.4 Inertial sensor data timing (F5)

- Output: 15 numbers per packet at 100 Hz (`config/robot.yaml:381,388`). `baudrate` is commented out (`:391`), so the driver default 115200 applies (`sensors/ebimu/imu.py:46-52`). The driver's own comment puts this at about 11 KB/s of an 11.5 KB/s link. One packet takes about 9.6 ms to transmit, so every sample is at least that old when parsed, before the sensor's internal filter delay and before the 0–10 ms between packets. Training has no sensor delay (README:47-48 lists noise only).
- The policy reads the latest packet at call time (`imu.read()`, `generalized_walk.py:225`), newer than the joint angles of the same call, so the inputs are not time-aligned (joints old by about 20 ms, gyro by about 10–20 ms, but with different offsets).
- `EbimuImu.read()` (`imu.py:244-251`) returns the last value forever if the reader thread dies (`:223-230`), and returns "level, still" if no packet ever arrived (`sensors/base.py:105-120`). The control path never checks `is_valid` or `age_ms()` (`generalized_walk.py:167-168`; confirmed in the 10-07 audit §2.2, §9.2, still true).
- The attitude is the sensor's filtered estimate, not the exact projected gravity. During steps and swings, linear acceleration disturbs such estimates; the simulator gives exact gravity and trains with only ±0.05 uniform noise.

### 3.5 ROS two-machine path (F7)

Design: `ros/README.md` §0, §6. Control loop at 100 Hz on the motion board publishes state at 100 Hz; policy node on the work machine runs on a 20 ms timer and publishes targets at 50 Hz; control loop re-sends the newest target every 10 ms.

- State freshness is "time since last arrival", threshold 60 ms (`ros/huphy_ros/huphy_ros/freshness.py:33-37`, `policy_node.py:211-218`): the policy may run on joint angles up to 60 ms old without noticing, three policy periods against a trained maximum of one.
- Rough latency chain: state age at publish (0–10 ms) + publish timer (0–10 ms) + network + timer phase (0–20 ms) + inference + target wait at the control loop (0–10 ms) + staggered sends (0–6.6 ms). My estimate: 10–55 ms, mean about 33 ms (not measured). Training worst case is 25 ms.
- The policy node's timer shares one Python executor thread with about 300 subscription callbacks per second (joint state, sensor, gravity at 100 Hz each), so timer jitter is likely; the policy never sees the real elapsed time between its two history frames.
- Gravity is a `Latest` value with no age check (`policy_node.py:98,215`); the gyro goes through the 60 ms check. `control_node._publish` publishes the sensor state at every timer tick even if the sensor thread died (`control_node.py:262-278`), so a frozen sensor looks fresh on the network.
- Missing the ankle crank angles (F1) comes on top of all of this.

### 3.6 Missing or late replies (F8)

- `RobStrideBus._absorb` (`motors/robstride/bus.py:316-352`) keeps the previous state when a reply is missing; a reply that arrives after the wait still updates the cache but counts as "missing" for that cycle. The observation carries no flag: `q(t)` and `q(t-1)` become equal for that joint for one step.
- The command for that motor may also have been lost, in which case the motor kept its previous target for one more period (an extra 20 ms delay on that joint; not in training).
- Stop rule: 5 consecutive missed cycles (`robot.yaml:361`, `safety/link.py`). At 50 Hz that is 100 ms (the comment in `robot.yaml` assumes 100 Hz).
- Earlier measured loss: 1.4–3.1% (`2026-10-14_dr_reference_latency_imu.md` §1 table). The `latency_*.csv` log (§7) gives the per-joint count for the walking runs.

---

## 4. Calibration and left–right symmetry (checklist item 4)

### 4.1 Zero offsets and signs (F2)

- `config/calibration/left_leg_v1.0.json` and `right_leg_v1.0.json`: every joint `sign: 1.0`, `offset_deg: 0.0`, `zero_reference: "0"`. Line 4 of each file still says "nothing has been measured yet" for sign, offset and limits.
- `is_calibrated` is decided by `calibration/store.py:246-255` (`unmeasured`): a joint counts as measured when `zero_reference` is non-empty. Commit `394f5d3` (2026-10-12, message "json config") changed it from `""` to `"0"` for all 12 entries (6 per leg). A human may have set the motor zeros at a reference pose and typed a placeholder; the repository cannot show which. The README procedure (`README.md:420-460`) holds the leg by hand in a reference pose (note example: "leg straight, sole flat") and sets all 6 zeros there. The accuracy of that jig is unknown.
- What an offset error does: the controller works in `cal = raw` space, so an offset `δ` on one joint means the policy believes the joint is `δ` away from where it is, and the PD controller drives the real joint to `target − δ`. Training randomizes encoder bias only ±0.01 rad (±0.57°, `T:generalized_cfg.py:199-202`) and reset position by ±0.05 rad. A hand-set zero error of 2° is 3.5 times the trained bias range. A left–right difference in hip roll, hip yaw or the crank pair gives a one-sided pose and one-sided stance, which is a direct candidate for veering. The 2026-10-21 standing simulation (H6, random ±0.5–4° per joint) saw no tremor, but it was a standing test and did not test straight-line walking.
- Limits are the training model's ranges (commit `b1ac519`, "filled from training model range"), not measured stops. If a real hard stop is closer than the simulator's range, the controller does not know.
- The sign convention in the simulator is not a simple mirror: both hip-roll joints use axis (−1,0,0) and both yaw joints (0,0,1), while pitch and crank axes are mirrored (`T:assets/pygmalion_legonly_ab.xml:87-216`). With `sign=1` everywhere, the real motors must already turn the same way as the simulator joints. A reversed joint would fail loudly, so a subtle error here is unlikely; still, `huphy-commission nudge` results for the 1.0 hardware are not in the repository.
- Which physical leg is "right": the policy's right leg is the robot's −y leg (x forward). CAN adapters were swapped by a rewiring on 2026-10-19 and the udev names were swapped to compensate (`docs/candlelight.md`); motor ids 7–12 on `canr` and 1–6 on `canl` are checked by who answers. That the ids 7–12 leg is physically the robot's right leg is assumed.

### 4.2 Other asymmetry sources checked

| Source | Result |
|---|---|
| Gains, motor models | Identical left and right (`robot.yaml:405-427`) |
| Limits | Mirrored exactly |
| Default pose | Mirror to 0.011° |
| Ankle kinematics mirrored for left only (`scripts/bringup.py:137-138`) | Not on the policy path |
| Command order within a cycle | Right leg parts first in code, but both legs' i-th commands go out together (`biped.py:416-436`); only a missing reply delays both legs equally |
| Separate adapters, separate latencies | Possible; the per-joint latency file shows it (right vs left p50, p95, lost) |
| Inertial sensor mount tilt, bias | F3: no handling; a constant tilt becomes a constant one-sided pose bias (static probe in §2.1) |
| Policy has no heading input | Trained with heading mode off (`T:generalized_cfg.py:243-250`), so any yaw-rate error integrates. Some drift exists in simulation too; the simulated baseline for a 10 s straight walk should be the reference before blaming the robot |

---

## 5. Checked and found consistent

- Observation layout, units, joint order, history handling, previous-action content, command slots (§2.1).
- Action scale, default pose numbers, policy clip, gains (§2.2, §2.3).
- Network loader equals torch (max difference 1.9e-6, 200 random inputs); checkpoint hash equals the contract file.
- Gravity direction maths and quaternion flip (§2.1).
- Damp stop uses `kp=0`, `kd=10` (`robots/leg.py:79,950-975`) as the contract asks; not a walking-gait factor.
- No temperature derating and no torque limit beyond the message ranges exist in code (searched `src/` and `ros/`); the motor temperature is only read, not used.
- Approach to the default pose: linear 3 s interpolation from the pre-loop pose (`scripts/selftest.py:189-206`, `run.py:424-451`); a jump only if the robot moved between reading the start pose and enabling torque. Low.

---

## 6. What to fetch from the robot (closes the "board may differ" gap)

1. The exact `huphy-run` command line (`--hz`, `--drain-ms`, `--gain-scale`, `--vx/--wz`, `--log`, `--live`), or the ROS launch and parameters.
2. Whether the run used `huphy-run` or the ROS nodes (decides whether F1 applies).
3. `git -C <repo> rev-parse HEAD`, `git status`, `git diff`, and the actual `config/robot.yaml` and `config/calibration/*_v1.0.json` on the board.
4. The whole `logs/` folder (§7) and the terminal text printed at the end of each run (the end-of-run report is not saved anywhere).
5. Physical facts: where the inertial sensor is bolted (pelvis or leg, orientation of its printed axes), how and where the motor zeros were set, and a hand-tilt check (the earlier note proposes it: lean forward and read gravity x; roll right and read gravity y).

---

## 7. What the walking logs contain

`huphy-run` records nothing unless `--log [DIR]` is given. `DIR` defaults to `logs/`, relative to the directory `huphy-run` was started in (`run.py:219-225`, `:428-431`). It does **not** attach the telemetry sink (`ControlLoop(leg, hz=..., mode=CONTROL)`, `run.py:396`), so `robot.yaml`'s UDP target `192.168.0.15:9870` (`:364-367`) is unused. Files are held in memory and written when the run ends (`policy_log.py:23-30`); a hard kill loses them, Ctrl-Q/Ctrl-C does not.

| File | Written by | Rate | Columns |
|---|---|---|---|
| `logs/policy_generalized_walk_YYYYMMDD_HHMMSS.csv` | `telemetry/policy_log.py` (`default_path` `:138-141`) | One row per **policy call** (50 Hz target; every second cycle at `--hz 100`). Row starts at the first call, so the approach/wait phase is absent | First line `# limits name=lo:hi …` (calibration limits; read with `comment="#"`). Then 81 columns: `t` (seconds since first call, monotonic, stamped after the network call); 44 inputs `obs/gyro/x,y,z` (rad/s), `obs/grav/x,y,z`, `obs/q_prev/<joint>` (rad, minus default), `obs/q/<joint>` (rad, minus default), `obs/last_act/<joint>` (raw output), `obs/cmd/vx`, `obs/cmd/wz`; 12 `act/<joint>` (raw network output); 12 `tgt/<joint>` (degrees, after default + 0.25·a and the policy clip, **before** the 3° margin and jump guard); 12 `pos/<joint>` (degrees, measured, cal space, same sample as `obs/q`). Joint names: `right_leg/hip_pitch` … `left_leg/ankle_b` (`ankle_a`, `ankle_b` are crank angles) |
| `logs/latency_generalized_walk_YYYYMMDD_HHMMSS.csv` | `telemetry/latency_log.py` (`:138-156`) | One row per motor reply while the policy is active (not the approach, wait or damping phases) | First line `# drain_ms=… hz=…`; then `t, joint, latency_ms, status` with status `ontime`, `late` or `lost` (lost rows have empty time and latency) |
| `logs/track_live_generalized_walk_*.csv` | `telemetry/live.py` + `track_log.py`, only with `--live` and after pressing "record" in the browser | One row per loop cycle (at `--hz`) | `t`; 12 `tgt/<joint>` = targets **actually sent** (after the 3° margin and jump guard, `robot.last_sent`); 12 `pos/<joint>`; `imu/main/{roll,pitch,yaw,gx,gy,gz,grav_x,grav_y,grav_z,ax,ay,az,age}` (degrees, deg/s, gravity unit vector, m/s², ms since parse) |

Not logged anywhere by `huphy-run`: motor velocity, motor-reported torque, temperatures, per-cycle loop time (but see `diff(t)`), the guard's clip counters per cycle, gains, sensor `sensor_ms` (packet loss), CAN counters. The end-of-run report (`scripts/failures.py`) prints mean rate, overruns, worst cycle, missed replies per motor, CAN counters and guard clip counts to the terminal only. The full telemetry sink (`telemetry/snapshot.py` / `csv_log.py`) exists but is only used by `huphy-bringup`.

The ROS path has no file output; use `ros2 bag record -a` (`ros/README.md` §3-3). Topics: `/huphy/joint_states` (radians; position, velocity, effort; **names are ankle_pitch/ankle_roll, not the cranks**), `/huphy/imu`, `/huphy/gravity`, `/huphy/joint_targets`, `/huphy/loop_status` (5 Hz: mean rate, overruns, missing cycles), `/huphy/policy_status`.

### Analysis recipes for the logs

- **Integrity.** `obs/q_prev[k] == obs/q[k−1]`; `tgt = default + 14.32°·act`, clipped; `obs/q = radians(pos − default)`. If any fails, the log or the adapter has a bug.
- **Timing.** `diff(t)`: mean, spread, p99 (training: 20.0 ms exactly). Fraction above 30 ms.
- **Start state.** First 0.3 s (robot nearly still): mean of `obs/gyro/z` (bias), `obs/grav/x,y` (tilt or mount offset), compared with the sensor's own `imu/main/roll,pitch` in the `track_live` file.
- **Veering.** Integral of `obs/gyro/z` over the straight segment against commanded 0. Cross-check with `imu/main/yaw` change. Compare with the simulated drift for the same command.
- **Asymmetry.** Mirror rule from the simulator defaults: left joint = −right joint for all six joint types. Over a straight segment compute the mean of `pos_R + pos_L`, `tgt_R + tgt_L` and `act_R + act_L` per joint type. A non-zero mean in `act` is the policy's own response (likely to inputs); a non-zero mean in `pos` but not in `tgt` is offset or load; compare right and left rows of the latency file.
- **Margin clipping.** Fraction of rows where `tgt` is within 3° of the `# limits` line, per joint (F6).
- **Tremor.** Spectrum of `pos − tgt` and of `act` differences per joint; split by leg; compare 1–2 Hz (load or hanging, per the 10-21 note) against above 5 Hz (noise or latency).

---

## 8. Open items (not resolved here)

1. Physical sensor mount and axis orientation (F3), and whether a constant tilt or gyro bias exists in the walking runs.
2. Real zero accuracy per joint and left–right difference (F2). The cleanest check is a static one: command the default pose, read the sensor's roll/pitch and compare the two legs' stance with the simulator pose.
3. Which path ran (`huphy-run` or ROS), and the `--hz` used (F1, F4, F7).
4. Per-cycle time budget on the motion board: profile one cycle; test whether skipping the unused ankle kinematics restores the loop rate (F4). I changed nothing in HUPHY.
5. Motor-side limits and filters inside the firmware (not readable from the repository).
6. Effect size of any of the above on a real walk is unknown. The offline probes show direction and scale of the network's response only; none of this is a gait result.
