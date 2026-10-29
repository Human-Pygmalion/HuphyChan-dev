# Stairs/Slopes-12 Union-DR checkpoint 99,998

Portable release snapshot of training run
`2026-10-25_11-36-56_generalized_stairs_slopes12_courseprogress_stagnation_resume49999_50k_v1`
at iteration 99,998.

## Files

| File | Purpose |
|---|---|
| `model_99998.pt` | Full checkpoint: actor, critic, optimizer, iteration, and environment step |
| `policy.onnx` | Deterministic actor; the empirical observation normalizer is embedded |
| `deploy.yaml` | Deployment contract, training provenance, terrain curriculum, hashes, and source versions |
| `env.yaml` | Exact serialized environment configuration from the run |
| `agent.yaml` | Exact serialized PPO configuration from the run |

The actor is `44 -> 512 -> 256 -> 128 -> 12`, the critic input is 268, and
`common_step_counter=2,400,048`. The checkpoint was produced by a full resume from
`../terrain10_union_dr_49999/model_49999.pt`; it was not an actor-only warm start.

The parent policy trained with `vx` and `wz`, but this 50,000-update fine-tune sampled
forward-only commands: `vx in [0.1, 2.0]`, `vy=0`, and `wz=0`. The actor input still has
two command entries `[vx, wz]`; deployment must supply zero for `wz`.

## Training configuration

Training retained the full Union-DR envelope and actor observation corruption. This includes
link pseudo-inertia/COM variation, payload mass, PD gains, motor strength and armature,
active/passive joint friction and damping, foot friction, encoder bias, IMU mounting error,
pushes, observation lag/noise, and actuator delay. Exact values are in `env.yaml`.

The 12 cumulative-uniform curriculum levels were:

```text
slope 5 deg, stairs 3 cm, slope 10 deg, stairs 6 cm,
slope 15 deg, stairs 9 cm, slope 20 deg, stairs 12 cm,
slope 25 deg, stairs 15 cm, slope 30 deg, stairs 18 cm
```

One level unlocked every 2,000 updates. Each generated course was 54 m long by 45 m wide
and supported travel in both directions from a center spawn.

## Reproduce training

Use mjlab commit `20f10e96b` and this repository's source commit
`981a245d3ca124467d83d83aceffb0462b7953cd`. Install this package into the matching
mjlab environment, then run from `mjlab_policy/`:

```bash
export MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python
$MJLAB_PYTHON -m pip install -e .
$MJLAB_PYTHON scripts/verify_stairs_slopes_resume.py
$MJLAB_PYTHON scripts/verify_stairs_slopes_release.py
MJLAB_PYTHON="$MJLAB_PYTHON" bash scripts/train_stairs_slopes_resume.sh
```

The script defaults to the released parent checkpoint, seed 43, 2,048 environments, and
50,000 requested updates. Override `NUM_ENVS`, `MORE_ITERS`, `SEED`, `RUN_NAME`, or `CKPT`
only when intentionally changing the experiment. Exact source configuration snapshots are
included here because GPU training is not promised to be bitwise deterministic.

## Playback

Play a fixed curriculum level (level 10 is the 15 cm stair course):

```bash
$MJLAB_PYTHON scripts/play.py \
  Pygmalion-Velocity-Generalized-StairsSlopes12-UnionDR \
  --terrain-level 10 --staged-forward \
  --checkpoint-file checkpoints/stairs_slopes12_union_dr_99998/model_99998.pt
```

Play the 2 m-wide custom test course with a 20 degree ascent, 50 cm landing, and four
15 cm descending steps at fixed foot friction 0.6:

```bash
$MJLAB_PYTHON scripts/play.py \
  Pygmalion-Velocity-Generalized-StairsSlopes12-UnionDR \
  --four-step-upslope-course --staged-forward --foot-friction 0.6 \
  --checkpoint-file checkpoints/stairs_slopes12_union_dr_99998/model_99998.pt
```

## Integrity and ONNX verification

```bash
cd checkpoints/stairs_slopes12_union_dr_99998
sha256sum model_99998.pt policy.onnx env.yaml agent.yaml
```

Expected hashes are in `deploy.yaml`. The ONNX output was compared with the checkpoint's
deterministic PyTorch actor for 32 seeded observations and passed `rtol=1e-5, atol=1e-6`;
the maximum absolute difference was `4.8828125e-4`.

## Deployment warning

This is a simulation artifact, not real-robot safety validation. Before motor enable,
independently verify observation and joint order, signs, units, PD gains, action scaling and
clipping, torque/speed limits, latency, emergency stop, fall protection, and the zero-yaw
command requirement.
