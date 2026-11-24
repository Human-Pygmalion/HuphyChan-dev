# Flat Union-DR knee LR2x vx-max 1.8 checkpoint 185,100

Portable, resumable snapshot of
`2026-11-23_14-15-48_flat_uniondr_fullplanar_clearance10_kneeflex_lr2x_vxmax18_resume172200_plus30k_4096_gpu0`.
The snapshot was taken at iteration 185,100 while the original job remained running.

## What is included

| File | Purpose |
|---|---|
| `model_185100.pt` | Full actor, critic, optimizer, iteration, and environment-step state; resume from this file |
| `model_172200.pt` | Exact parent checkpoint used by the original continuation |
| `policy.onnx` | Deterministic deployment actor exported with the 185,100 checkpoint |
| `env.yaml` | Exact serialized 4,096-environment configuration used by the run |
| `agent.yaml` | Exact PPO and original resume configuration used by the run |
| `deploy.yaml` | Observation/action contract, gains, command and DR summary, versions, and hashes |
| `scripts/resume_from_185100.sh` | Continue from the portable checkpoint on another server |
| `scripts/reproduce_from_172200.sh` | Re-run the original 12,900-update continuation |
| `scripts/play.sh` | Play the packaged checkpoint |
| `scripts/verify_release.py` | Verify hashes, checkpoint state, ONNX validity, and ONNX/PyTorch parity |

## Exact training configuration

- Task: `Pygmalion-Velocity-Flat-UnionDR-FullPlanar-Clearance10-KneeFlex-LR2x`
- Flat plane; 4,096 environments; seed 43; 50 Hz policy, 5 ms physics step, decimation 4.
- Commands: body-frame `vx [-1.5, 1.8] m/s`, `vy [-1, 1] m/s`, `wz [-1, 1] rad/s`.
- 10 cm foot-clearance target; self-collision reward weight `-10`.
- Per-side swing-knee target 0.55 rad, weight `-0.2`; per-side landing-peak target
  0.70 rad, sigma 0.20, weight `+0.6`.
- Union DR includes foot friction 0.3-1.7, base payload -1 to +15 kg, actuator,
  encoder/IMU, delay, inertial/passive-joint, and torso-push randomization.
- Updated `capsule7plus5` foot collision and current deployment PD gains are used.

`env.yaml` and `agent.yaml` are the authoritative snapshots for every parameter. The
command-range override is explicitly repeated by both training scripts because the task's
registered default is narrower than this continuation.

## New-server setup

Requirements: Linux, NVIDIA driver/CUDA support, Git, and `uv`.

```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
git checkout 20f10e96b84814ae11a70a4fdd3cdcf046ed0e57
uv sync
uv pip install -e /path/to/HuphyChan-dev/mjlab_policy
```

The source repository must be checked out at the release commit that contains this folder.
The original run used HuphyChan source commit
`e56eff121fc14d059cad7a2458d41f0a28bcdaa3` and stock mjlab commit
`20f10e96b84814ae11a70a4fdd3cdcf046ed0e57`.

Verify the transfer before training:

```bash
cd /path/to/HuphyChan-dev/mjlab_policy
MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  checkpoints/flat_uniondr_kneeflex_lr2x_vxmax18_185100/scripts/verify_release.py
```

## Resume from 185,100

```bash
cd /path/to/HuphyChan-dev/mjlab_policy
CUDA_VISIBLE_DEVICES=0 MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  checkpoints/flat_uniondr_kneeflex_lr2x_vxmax18_185100/scripts/resume_from_185100.sh \
  30000 4096 my_resume_run
```

Arguments are `[additional_updates] [num_envs] [run_name]`; additional CLI overrides can
follow them. To disable W&B, append `--agent.logger tensorboard`. The script creates a local
source run, copies the packaged checkpoint into it, pins `vx` to `[-1.5, 1.8]`, and restores
the complete optimizer/checkpoint state.

## Reproduce the original continuation

```bash
CUDA_VISIBLE_DEVICES=0 MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  checkpoints/flat_uniondr_kneeflex_lr2x_vxmax18_185100/scripts/reproduce_from_172200.sh
```

This runs 12,900 additional updates from the included iteration-172,200 parent. It is a
configuration-level reproduction, not a bit-for-bit replay: the rsl_rl checkpoint does not
store the complete Python/Torch/CUDA process RNG state.

## Deployment contract

The actor is `45 -> 512 -> 256 -> 128 -> 12` with ELU activations and an embedded
observation normalizer in ONNX. Actor input:

| Slice | Signal |
|---|---|
| `[0:3]` | body angular velocity |
| `[3:6]` | projected gravity |
| `[6:30]` | two frames of 12 motor-position errors, oldest to newest |
| `[30:42]` | previous raw policy action; initialize to zero |
| `[42:45]` | body-frame command `[vx, vy, wz]` |

Output order and joint signs are fully specified in `deploy.yaml`. Apply
`q_target = q_default + 0.25 * action`, then the listed joint clips. The gains are hip
pitch/roll `135/6`, hip yaw `75/4`, knee `130/6`, ankle crank A/B `60/2.7` (`Kp/Kd`, both
sides).

This policy is simulation-validated only. Verify observation order/signs, A/B ankle
conversion, motor signs, limits, gains, emergency stop, and physical fall protection before
enabling a real robot.
