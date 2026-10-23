# Terrain10 Union-DR full resume

This task keeps the policy, reward, command, Terrain10 curriculum, actor observation, critic
observation, and action contracts of
`2026-10-19_21-27-42_generalized_terrain10_resume19700_curriculum2k_2048`. It changes only
domain randomization and reset disturbances.

## Registered task

```text
Pygmalion-Velocity-Generalized-Terrain10-UnionDR
```

The task resumes the included full checkpoint:

```text
checkpoints/terrain10_2048_model_37400.pt
```

The checkpoint contains actor, critic, optimizer, iteration `37400`, and environment step
`897648`. Restoring the environment step keeps Terrain10 levels 0–9 unlocked. Actor input remains
44, critic input remains 268, and action remains 12, so full strict resume is supported.

## Union rule

If only walking or squat used an item, include it. If both used the item, use the envelope of
their ranges. DR is active at full strength from the first resumed update and is redrawn at reset
unless stated otherwise.

| Item | Union range or behavior |
|---|---|
| link pseudo-inertia | per-link walking/squat envelope; mass/inertia scales listed below |
| pelvis CoM | x/y/z `U(-0.03, 0.03)` m |
| other link CoM | squat CAD ranges, normally `±0.005 m` |
| pelvis payload | `U(-1, 8)` kg point mass at base-link CoM |
| Kp and Kd | independent per actuator `U(0.8, 1.2)` scales |
| motor T-N strength | per motor `U(0.8, 1.2)` |
| motor friction, damping, armature | independent `U(0.8, 1.2)` scales |
| passive rod/ankle friction | absolute `U(0, 0.5)` N m |
| passive rod/ankle damping | absolute `U(0, 0.05)` N m s/rad |
| foot friction | shared absolute coefficient `U(0.3, 1.2)` |
| encoder episode bias | per motor `U(-0.015, 0.015)` rad |
| IMU mounting R/P/Y | independently `U(-2, 2)` degrees |
| action/command delay | episode-fixed 0–4 physics steps, 0–20 ms |
| gyro, gravity, motor observation lag | episode-fixed 0–2 policy steps, 0–40 ms |
| observation noise | gyro `±0.2 rad/s`, gravity `±0.05`, motor position `±0.01 rad` |
| reset motor position | `±0.05 rad` plus 50% hip-roll spread perturbation `U(-0.10, 0.06)` |
| reset base | x/y `±0.5 m`, z `0.01–0.05 m`, yaw `±pi`, x/y/yaw velocity `±0.2` |
| torso push | horizontal random direction, `U(0,100)` N, `U(0.1,0.2)` s, `U(2,10)` s cooldown |
| terrain | original Terrain10 2k curriculum; restored checkpoint state has levels 0–9 unlocked |

The three sensor observation terms use the same 0–40 ms envelope and hold their sampled lag for
the episode. Stock mjlab gives each term its own delay buffer, so gyro, gravity, and encoder lag
draws are independent; the observation vector and checkpoint dimensions remain unchanged.

Pseudo-inertia mass/inertia scale is `exp(2 alpha)`. The resulting envelopes are:

| Link group | Scale envelope |
|---|---:|
| pelvis | `0.90–1.15` |
| hip pitch / hip roll / thigh | `0.90–1.10` |
| shin | `0.90–1.107` |
| ankle pitch link | `0.889–1.117` |
| foot | `0.90–1.138` |
| crank and rod | `0.90–1.10` |

## Server use

From this directory, install the task into the server's mjlab Python and validate it without
starting training:

```bash
MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python
$MJLAB_PYTHON -m pip install -e .
$MJLAB_PYTHON scripts/verify_union_dr.py
```

After W&B credentials are available, start the default 2,048-environment full resume from 37,400
to 50,000:

```bash
MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  bash scripts/train_generalized_union_resume.sh
```

Useful overrides:

```bash
NUM_ENVS=4096 TARGET_ITER=60000 RUN_NAME=my_union_run \
MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python \
  bash scripts/train_generalized_union_resume.sh --agent.save-interval 100
```

The runner is the generalized runner, so logging uses W&B project `pygmalion_velocity`. The script
does not warm-start: it restores actor, critic, optimizer, iteration, and curriculum step state.
