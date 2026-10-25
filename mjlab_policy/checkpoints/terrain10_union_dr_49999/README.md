# Terrain10 Union-DR checkpoint 49,999

This folder is the portable release snapshot of training run
`2026-10-24_02-05-41_generalized_terrain10_uniondr_resume37400` at iteration 49,999.

## Files

| File | Purpose |
|---|---|
| `model_49999.pt` | Full training checkpoint: actor, critic, optimizer, iteration, and environment step |
| `policy.onnx` | Deterministic actor for inference; observation normalizer is embedded |
| `deploy.yaml` | Compact 44-input/12-output deployment contract, command layout, action mapping, gains, and hashes |
| `env.yaml` | Exact serialized training environment configuration |
| `agent.yaml` | Exact serialized PPO configuration |

The checkpoint has actor `44 -> 512 -> 256 -> 128 -> 12`, critic input 268,
iteration 49,999, and `common_step_counter=1,200,048`. It is a full-resume checkpoint,
not an actor-only warm start.

The policy input command has **two** entries: body-frame forward speed `vx` and yaw rate
`wz`. The environment sampled lateral velocity as zero, and `vy` is not part of the actor
observation. Do not feed a three-entry command to this ONNX model.

## Verify hashes

From this folder:

```bash
sha256sum model_49999.pt policy.onnx env.yaml agent.yaml
```

Expected hashes are recorded in `deploy.yaml`.

## Simulation playback

From `mjlab_policy/`, with the repository task installed into the matching mjlab Python:

```bash
MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python
$MJLAB_PYTHON scripts/play.py \
  Pygmalion-Velocity-Generalized-Terrain10-UnionDR \
  --checkpoint-file checkpoints/terrain10_union_dr_49999/model_49999.pt
```

## Resume training

Use `model_49999.pt` with `--agent.resume True`. The checkpoint contains the optimizer
and curriculum step state. The serialized `env.yaml` is a record of the original run;
the executable task implementation is the `generalized-union-dr` branch at the commit
listed in `deploy.yaml`.

## Deployment warning

The ONNX output was checked against the PyTorch checkpoint. This is not real-robot safety
validation. Before motor enable, independently verify joint signs/order, observation signs,
PD gains, target clipping, emergency stop, and fall protection.
