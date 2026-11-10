# HUPHY 00545 Union-DR checkpoint 30,000

배포 및 재현용 스냅샷입니다. 원본 학습 런은
`2026-11-09_09-31-46_huphy_00545_uniondr_footfric03to17_fromscratch_4096_gpu1`이며,
베이스 체크포인트 없이 4,096개 환경에서 처음부터 학습했습니다.

최종 체크포인트의 `iter=30000`, `common_step_counter=720024`를 확인했습니다.
체크포인트에는 actor, critic, optimizer와 환경 step 상태가 모두 들어 있습니다.

## 핵심 파일

| 파일 | 용도 |
|---|---|
| `model_30000.pt` | 학습 재개가 가능한 전체 체크포인트 |
| `policy.onnx` | observation normalizer가 내장된 deterministic actor |
| `motion_policy.onnx` | actor와 00545 reference frame 출력을 함께 제공하는 ONNX |
| `deploy.yaml` | observation/action 계약, DR, 제어 주기, gain, termination, hash |
| `env.yaml`, `agent.yaml` | 실제 런에서 저장된 환경/PPO 설정 원본 |
| `motions/00545_interactionmesh_50hz.npz` | 학습에 직접 사용한 659-frame reference |
| `motions/00545.npz` | 원본 00545 SMPL-X, 396 frames, 30 Hz |
| `motions/00545_interactionmesh_lower9_30hz.npz` | 50 Hz 변환 전 retarget 결과 |
| `source/` | 학습 당시 HUPHY 관련 실행 코드와 retarget/변환 코드 스냅샷 |
| `scripts/` | 재학습, 재생, release 검증 스크립트 |

모든 artifact hash와 구조화된 전체 계약은 `deploy.yaml`에 있습니다.

## Actor observation: 114 dimensions

입력은 아래 순서를 반드시 지켜야 합니다.

| Slice | 크기 | 내용 |
|---|---:|---|
| `[0:48]` | 48 | 현재 reference frame의 24개 `joint_pos`, 이어서 24개 `joint_vel` |
| `[48:51]` | 3 | body-frame projected gravity |
| `[51:54]` | 3 | body-frame IMU angular velocity, rad/s |
| `[54:78]` | 24 | default pose 기준 relative joint position, rad |
| `[78:102]` | 24 | relative joint velocity, rad/s |
| `[102:114]` | 12 | 직전 raw policy action |

24개 joint 순서는 다음과 같습니다.

```text
R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee,
R_crank_A, R_rod_A_u1, R_rod_A_u2, R_crank_B,
R_rod_B_u1, R_rod_B_u2, R_ankle_pitch, R_ankle_roll,
L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee,
L_crank_A, L_rod_A_u1, L_rod_A_u2, L_crank_B,
L_rod_B_u1, L_rod_B_u2, L_ankle_pitch, L_ankle_roll
```

각 이름에는 실제 파일에서 `_joint`가 붙습니다. 학습 중 gravity에는 uniform
`[-0.05, 0.05]`, angular velocity에는 `[-0.2, 0.2]`, joint position에는
`[-0.01, 0.01]`, joint velocity에는 `[-0.5, 0.5]` noise가 적용됐습니다.
Gravity와 angular velocity에는 환경별 0--2 policy-step delay가 추가됐고,
episode별 encoder bias는 `[-0.015, 0.015] rad`였습니다. Play config에서는 actor
corruption을 끕니다.

Critic은 204차원입니다. 정확한 term 순서와 slice는 `deploy.yaml`의
`observations.critic`에 기록했습니다.

## Action 및 제어

Actor 출력 12개는 아래 motor 순서입니다.

```text
R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_crank_A, R_crank_B,
L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_crank_A, L_crank_B
```

목표 위치는 `q_target = q_default + 0.25 * action`입니다. 물리 timestep은
5 ms, decimation은 4이므로 policy/control rate는 50 Hz입니다. 학습 wrapper의
action clipping은 꺼져 있었습니다(`clip_actions: null`). 실제 로봇에서는 별도의
검증된 joint/velocity/torque safety limit를 반드시 적용해야 합니다.

`motion_policy.onnx`의 두 입력은 `obs: float32[1,114]`와
`time_step: float32[1,1]`입니다. `time_step`은 초가 아니라 0부터 658까지의
reference **frame index**입니다. 출력은 action 외에 해당 frame의 reference
joint/body state를 제공합니다. 순수 policy 배포에는 `policy.onnx`를 쓰면 됩니다.

## Union DR

이 런은 Union DR이 맞으며 발 마찰은 episode별 shared uniform
`[0.3, 1.7]`입니다. Nominal 발 마찰은 0.6, foot geom priority는 1입니다.
그 외 주요 범위는 다음과 같습니다.

- base payload 추가 질량: `[-1, 8] kg`
- PD gain, motor strength/armature/friction/damping scale: 각각 `[0.8, 1.2]`
- passive joint friction: `[0, 0.5]`, damping: `[0, 0.05]`
- IMU mount roll/pitch/yaw: 각각 `[-2, 2] deg`
- motor command delay: 0--4 physics steps
- base horizontal push: 0--100 N, 0.1--0.2 s, 2--10 s cooldown
- link별 pseudo-inertia/CoM DR와 hip reset spread도 활성화

link별 pseudo-inertia 수치는 `env.yaml` 및
`source/unitree_rl_mjlab/src/tasks/tracking/config/huphy/env_cfgs.py`가 기준입니다.

## Robot 및 foot collision

학습에 사용한 MJCF는 저장소의 다음 파일입니다.

```text
pygmalion_locomotion/assets/pygmalion_v2/
LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml
```

한 발당 sole capsule 7개와 visual 안쪽 upper capsule 5개, 총 12개입니다. 이
release에는 해당 MJCF가 참조하는 `meshes_footv5`, `meshes_reinforce`,
`meshes_d455`와 portable `meshes_v30_armfix` symlink도 함께 커밋되어 있습니다.
`tools/robot_model/build_m19_capsule7plus5.py`로 원본 `_loop.xml`에서 다시 만들 수
있습니다.

## 코드 설치와 학습 재현

학습 당시 `unitree_rl_mjlab` worktree의 HEAD는
`05c7538e95a300edf7dfdd5fca5907f0bf00cd43`이었고, HUPHY 코드는 아직 그
저장소에 커밋되지 않은 상태였습니다. 따라서 이 release의 `source/unitree_rl_mjlab`
스냅샷을 기준 코드 위에 overlay해야 합니다.

```bash
export HUPHYCHAN_ROOT=/path/to/HuphyChan-dev
export UNITREE_RL_MJLAB=/path/to/unitree_rl_mjlab
export MJLAB_PYTHON=/path/to/miniconda/envs/mjlab/bin/python

cd "$HUPHYCHAN_ROOT/mjlab_policy/checkpoints/huphy_00545_uniondr_30000"
cp -a source/unitree_rl_mjlab/. "$UNITREE_RL_MJLAB/"
```

복사된 `src/assets/robots/huphy/huphy_constants.py`의 `HUPHY_XML`은 학습 당시의
절대 경로를 그대로 보존합니다. 저장소 위치가 다르면 그 상수의 경로만 현재
`$HUPHYCHAN_ROOT`로 바꾸십시오. 동역학 값은 바꾸지 마십시오.

원본과 같은 from-scratch 조건으로 실행합니다.

```bash
UNITREE_RL_MJLAB="$UNITREE_RL_MJLAB" \
MJLAB_PYTHON="$MJLAB_PYTHON" GPU=0 \
bash scripts/train_from_scratch.sh
```

기본값은 seed 42, 4,096 env, `max_iterations=30001`, 24 steps/env,
save interval 500입니다. GPU/JAX/MuJoCo 병렬 학습은 bitwise deterministic를
보장하지 않으므로 동일 설정에서도 weight가 byte 단위로 같지는 않을 수 있습니다.

## Retarget부터 재현

포함된 코드는 원본 SMPL-X에서 lower-body 9-keypoint InteractionMesh retarget을
수행한 뒤 30 Hz 결과를 50 Hz MJLab motion으로 변환합니다.

```bash
# HoloSoma/SMPL-X/GMR 의존성이 설치된 환경에서
python source/holosoma/retarget_huphy.py \
  --source motions/00545.npz \
  --huphy-xml "$HUPHYCHAN_ROOT/pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml" \
  --output-dir /tmp/00545_huphy_retarget \
  --method interactionmesh --skip-render

"$MJLAB_PYTHON" source/unitree_rl_mjlab/scripts/convert_huphy_interactionmesh_motion.py \
  --input /tmp/00545_huphy_retarget/00545_huphy_interactionmesh_lower9.npz \
  --xml "$HUPHYCHAN_ROOT/pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml" \
  --output /tmp/00545_interactionmesh_50hz.npz --fps 50
```

주의: 30 Hz retarget NPZ 내부 `source` 문자열은 retarget 스크립트의 오래된 기본값
때문에 잘못 `00399.npz`라고 기록되어 있습니다. 실제 입력이 00545였다는 근거는
동일 실행에서 생성된 `motions/retarget_validation.json`과 포함된 원본
`00545.npz`입니다. 이 release는 provenance 오류를 숨기지 않기 위해
원본 NPZ를 수정하지 않았습니다.

## 재생 및 검증

```bash
UNITREE_RL_MJLAB="$UNITREE_RL_MJLAB" \
MJLAB_PYTHON="$MJLAB_PYTHON" GPU=0 VISER_PORT=8081 \
bash scripts/play.sh

"$MJLAB_PYTHON" scripts/verify_release.py
```

검증 스크립트는 artifact SHA-256, checkpoint iteration/dimension, 두 ONNX와
PyTorch actor의 32개 seeded input parity, motion shape/fps/closed-loop residual,
MJCF 로딩과 양발 12개 capsule 이름을 확인합니다. 두 ONNX의 최대 actor 오차는
`1.9073486328125e-6`으로 `rtol=1e-5`, `atol=1e-6` 검증을 통과했습니다.

## Safety

이 결과는 시뮬레이션 검증본이며 실제 로봇 안전 검증본이 아닙니다. Motor enable
전에 sensor axis/sign, 24-joint observation 순서, 12-motor action 순서, default
offset, PD gain, joint/torque/speed limit, delay, E-stop, fall protection을 독립적으로
확인하십시오.
