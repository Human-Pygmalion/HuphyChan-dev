# Flat Union-DR knee LR2x checkpoint 129,400

Portable release of
`2026-11-15_12-04-59_flat_uniondr_fullplanar_clearance10_kneeflex_lr2x_resume100898_plus30k_4096_gpu1`.
The latest available checkpoint is iteration 129,400. The run resumed the included
iteration-100,898 checkpoint and was stopped after 28,502 additional updates.

## Included files

| File | Purpose |
|---|---|
| `model_129400.pt` | Full actor, critic, optimizer, iteration, and environment-step state |
| `model_100898.pt` | Exact parent checkpoint used to start this continuation |
| `policy.onnx` | Deterministic 45-input/12-output deployment actor; normalizer embedded |
| `deploy.yaml` | Observation/action indices, commands, gains, DR summary, hashes, and safety contract |
| `env.yaml` | Exact serialized 4,096-environment training configuration |
| `agent.yaml` | Exact PPO and resume configuration captured by the run |
| `scripts/play.sh` | Play the released checkpoint in simulation |
| `scripts/reproduce_from_100898.sh` | Repeat the continuation from the exact parent checkpoint |
| `scripts/resume_from_129400.sh` | Continue training from the released checkpoint |
| `scripts/verify_release.py` | Verify hashes, checkpoint structure, ONNX structure, and ONNX/PyTorch parity |

## Deployment contract

The actor is `45 -> 512 -> 256 -> 128 -> 12` with ELU activations. Its input is:

| Slice | Signal |
|---|---|
| `[0:3]` | body angular velocity |
| `[3:6]` | projected gravity |
| `[6:30]` | two frames of 12 motor-position errors |
| `[30:42]` | previous raw policy action; use zeros on the first control frame |
| `[42:45]` | body-frame command `[vx, vy, wz]` |

The output is ordered
`R hip pitch, roll, yaw, knee, crank A, crank B, L hip pitch, roll, yaw, knee, crank A, crank B`.
Apply `q_target = q_default + 0.25 * action`, followed by the per-joint clips in
`deploy.yaml`. The control loop is 50 Hz and the simulation physics step was 5 ms with
decimation 4.

This policy used the current deployment gains, not the old 150/220/22.265625 values:
hip pitch/roll `135/6`, hip yaw `75/4`, knee `130/6`, ankle crank A/B `60/2.7`
(`Kp/Kd`, bilaterally). The complete nominal vector is in `deploy.yaml`.

The robot asset is the `capsule7plus5` foot model:

```text
pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml
```

## Training setup

- Flat plane, 4,096 environments, seed 43.
- Commands: `vx [-1.5, 1.5] m/s`, `vy [-1, 1] m/s`, `wz [-1, 1] rad/s`.
- Foot-clearance target 10 cm and self-collision weight -10.
- Separate left/right swing-knee target 0.55 rad, weight -0.2 per side.
- Separate left/right landing-peak target 0.70 rad, sigma 0.20, weight +0.6 per side.
- Union DR includes foot friction 0.3–1.7, payload -1–15 kg, PD and motor-strength
  scales 0.8–1.2, encoder/IMU noise, actuator/observation delay, inertial and passive-joint
  randomization, and torso pushes. `env.yaml` is authoritative for every range.

## Verify

From `mjlab_policy/`:

```bash
MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  checkpoints/flat_uniondr_kneeflex_lr2x_129400/scripts/verify_release.py
```

The verifier checks all hashes, both checkpoint iterations, actor/critic dimensions,
ONNX validity, and deterministic ONNX parity against `model_129400.pt`.

## Play

```bash
cd mjlab_policy
MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  checkpoints/flat_uniondr_kneeflex_lr2x_129400/scripts/play.sh
```

Extra arguments are forwarded to `scripts/play.py`.

## Reproduce the continuation

```bash
cd mjlab_policy
CUDA_VISIBLE_DEVICES=1 MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  checkpoints/flat_uniondr_kneeflex_lr2x_129400/scripts/reproduce_from_100898.sh
```

The default is the original 4,096 environments and 28,502 additional updates, ending at
iteration 129,400. Pass `[updates] [num_envs] [run_name]` to override them. This is a
configuration-level reproduction: rsl_rl checkpoints do not store the complete process RNG
state, so a rerun is not expected to be bit-for-bit identical.

To continue from iteration 129,400 instead:

```bash
CUDA_VISIBLE_DEVICES=1 MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  checkpoints/flat_uniondr_kneeflex_lr2x_129400/scripts/resume_from_129400.sh 30000 4096
```

## Real-robot warning

This release is simulation-validated only. Before enabling motors, independently verify the
45 observation values, motor order/signs, crank A/B ankle conversion, command frame, gain
units, target clipping, emergency stop, and physical fall protection.
