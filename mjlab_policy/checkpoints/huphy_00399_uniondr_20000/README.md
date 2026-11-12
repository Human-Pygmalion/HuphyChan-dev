# HUPHY 00399 Union-DR checkpoint 20,000

00399 HUPHY motion-tracking 정책의 배포 및 재현 패키지입니다. 이 정책은 한 번에
from-scratch Union DR로 학습된 것이 아니라 다음 두 단계로 만들어졌습니다.

1. `HUPHY-Tracking-ProjectedGravity`를 4,096 env에서 처음부터 `model_13000`까지 학습
2. 그 체크포인트를 actor/critic/optimizer까지 그대로 resume하고
   `HUPHY-Tracking-ProjectedGravity-UnionDR`로 `model_20000`까지 추가 학습

원본 2단계 launch는 17,001 additional iterations로 시작했지만 사용자가 iteration
20,000에서 종료했습니다. 따라서 포함된 stage-2 재현 스크립트 기본값은 공개된 최종
체크포인트와 같은 지점까지 가도록 7,001 iterations입니다.

## 핵심 파일

| 파일 | 용도 |
|---|---|
| `model_13000.pt` | 1단계 부모 체크포인트, actor/critic/optimizer 포함 |
| `model_20000.pt` | 최종 Union-DR 체크포인트, 학습 재개 가능 |
| `policy.onnx` | observation normalizer가 내장된 deterministic actor |
| `motion_policy.onnx` | actor와 00399 reference frame 출력을 함께 제공 |
| `deploy.yaml` | observation/action/DR/reward/termination/hash 전체 계약 |
| `stage{1,2}_{env,agent}.yaml` | 실제 두 런이 저장한 원본 설정 |
| `motions/00399_interactionmesh_50hz.npz` | 학습에 사용한 1,511-frame reference |
| `motions/00399.npz` | 원본 907-frame, 30 Hz SMPL-X |
| `motions/00399_interactionmesh_lower9_30hz.npz` | 50 Hz 변환 전 retarget 결과 |
| `source/` | 당시 HUPHY 학습·retarget·motion 변환 코드 스냅샷 |
| `observation_rollouts/` | DR on/off 각 2회 전체-cycle observation/action 로그 |

## Actor observation: 114 dimensions

배포 입력은 아래 순서를 반드시 지켜야 합니다.

| Slice | 크기 | 내용 |
|---|---:|---|
| `[0:24]` | 24 | 현재 reference joint position |
| `[24:48]` | 24 | 현재 reference joint velocity |
| `[48:51]` | 3 | body-frame projected gravity |
| `[51:54]` | 3 | body-frame IMU angular velocity, rad/s |
| `[54:78]` | 24 | default pose 기준 relative joint position |
| `[78:102]` | 24 | relative joint velocity |
| `[102:114]` | 12 | 직전 raw policy action; 첫 프레임은 정확히 0 |

24개 joint 순서는 다음과 같습니다.

```text
R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee,
R_crank_A, R_rod_A_u1, R_rod_A_u2, R_crank_B,
R_rod_B_u1, R_rod_B_u2, R_ankle_pitch, R_ankle_roll,
L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee,
L_crank_A, L_rod_A_u1, L_rod_A_u2, L_crank_B,
L_rod_B_u1, L_rod_B_u2, L_ankle_pitch, L_ankle_roll
```

정확한 실제 joint 이름과 critic 204차원 slice는 `deploy.yaml`에 있습니다. 학습 중
gravity noise `[-0.05,0.05]`, angular velocity `[-0.2,0.2]`, joint position
`[-0.01,0.01]`, joint velocity `[-0.5,0.5]`와 gravity/gyro 0--2 policy-step delay가
적용됐습니다. Play config에서는 actor corruption을 끕니다.

## Action 및 제어

출력 12개 motor 순서는 아래와 같습니다.

```text
R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_crank_A, R_crank_B,
L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_crank_A, L_crank_B
```

`q_target = q_default + 0.25 * action`이며 physics timestep 5 ms, decimation 4,
policy/control rate 50 Hz입니다. 학습 wrapper의 action clipping은 꺼져 있었습니다.
`motion_policy.onnx`의 `time_step`은 초가 아니라 0--1510 reference frame index입니다.

## Union DR와 reward

Union DR은 2단계에서만 활성화됐습니다. 주요 범위는 다음과 같습니다.

- 발 마찰: episode별 양발 shared uniform `[0.3, 1.7]`; nominal 0.6
- base 추가 payload: `[-1, 8] kg`
- PD gain, motor strength/armature/friction/damping scale: `[0.8, 1.2]`
- passive joint friction `[0,0.5]`, damping `[0,0.05]`
- encoder bias `[-0.015,0.015] rad`, IMU mount RPY 각 `[-2,2] deg`
- motor command delay 0--4 physics steps
- torso horizontal push 0--100 N, 0.1--0.2초, cooldown 2--10초
- link별 pseudo-inertia/CoM 및 hip reset spread 활성화

Reward는 root position/orientation 각 `0.5`, body position/orientation/linear velocity/
angular velocity 각 `1.0`, action-rate `-0.1`, joint-limit `-10`, self-collision
`-10`입니다. 원본 함수와 std는 `deploy.yaml`, `stage2_env.yaml`, `source/`에
고정되어 있습니다.

## Observation rollout 검증

`observation_rollouts/`는 `model_20000`을 00399 전체 1,511 frame 동안 기록한 네
롤아웃입니다. DR on 두 회는 학습 Union DR, observation corruption/delay와 actuator
delay를 유지합니다. DR off 두 회는 event, observation noise/delay, actuator delay와
motion reset perturbation을 끕니다.

각 NPZ에는 `actor_observation[1511,114]`, `raw_action[1511,12]`, reward, done,
reference frame과 term별 slice가 들어 있습니다. 모든 파일에서 첫 previous action은
정확히 0이며 이후 `previous_action[t] == raw_action[t-1]`을 검증합니다. 학습 timeout은
10초이고 모션은 첫 샘플부터 마지막까지 30.20초이므로 전체 cycle 기록에서만
termination을 공통으로 껐습니다. seed와 hash는 `observation_rollouts/manifest.json`에
있습니다.

재생성:

```bash
cd "$UNITREE_RL_MJLAB"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "$MJLAB_PYTHON" \
  /path/to/huphy_00399_uniondr_20000/scripts/log_observation_rollouts.py --device cuda:0
```

## 코드 설치와 2단계 학습 재현

학습 당시 `unitree_rl_mjlab` HEAD는
`05c7538e95a300edf7dfdd5fca5907f0bf00cd43`이었고 HUPHY 관련 코드는 아직 그
저장소에서 uncommitted 상태였습니다. 따라서 포함된 executable snapshot을 해당
checkout 위에 overlay합니다.

```bash
export HUPHYCHAN_ROOT=/path/to/HuphyChan-dev
export UNITREE_RL_MJLAB=/path/to/unitree_rl_mjlab
export MJLAB_PYTHON=/path/to/miniconda/envs/mjlab/bin/python

cd "$HUPHYCHAN_ROOT/mjlab_policy/checkpoints/huphy_00399_uniondr_20000"
cp -a source/unitree_rl_mjlab/. "$UNITREE_RL_MJLAB/"
```

`source/.../huphy_constants.py`의 `HUPHY_XML`은 학습 당시 절대경로를 보존합니다.
저장소 위치가 다르면 그 경로만 현재 `HUPHYCHAN_ROOT`로 바꾸고 동역학 값은
변경하지 마십시오.

1단계 from scratch:

```bash
UNITREE_RL_MJLAB="$UNITREE_RL_MJLAB" MJLAB_PYTHON="$MJLAB_PYTHON" GPU=0 \
  bash scripts/train_stage1_from_scratch.sh
```

포함된 원본 부모 체크포인트에서 2단계 Union DR 재현:

```bash
UNITREE_RL_MJLAB="$UNITREE_RL_MJLAB" MJLAB_PYTHON="$MJLAB_PYTHON" GPU=0 \
  bash scripts/train_stage2_uniondr_resume.sh
```

GPU/MuJoCo 병렬 학습은 bitwise determinism을 보장하지 않으므로 같은 설정이어도
weight가 byte 단위로 같지는 않을 수 있습니다.

## Retarget부터 재현

```bash
python source/holosoma/retarget_huphy.py \
  --source motions/00399.npz \
  --huphy-xml "$HUPHYCHAN_ROOT/pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml" \
  --output-dir /tmp/00399_huphy_retarget \
  --method interactionmesh --skip-render

"$MJLAB_PYTHON" source/unitree_rl_mjlab/scripts/convert_huphy_interactionmesh_motion.py \
  --input /tmp/00399_huphy_retarget/00399_huphy_interactionmesh_lower9.npz \
  --xml "$HUPHYCHAN_ROOT/pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml" \
  --output /tmp/00399_interactionmesh_50hz.npz --fps 50
```

## 재생 및 release 검증

```bash
UNITREE_RL_MJLAB="$UNITREE_RL_MJLAB" MJLAB_PYTHON="$MJLAB_PYTHON" GPU=0 \
  VISER_PORT=8081 bash scripts/play.sh

"$MJLAB_PYTHON" scripts/verify_release.py
```

학습 로봇은 M19 `capsule7plus5_loop.xml`이며 한 발당 sole 7개와 upper 5개, 총
12개 capsule collision을 사용합니다.

## Safety

시뮬레이션 검증 결과이지 실제 로봇 안전 검증본은 아닙니다. Motor enable 전 sensor
axis/sign, 24-joint observation 순서, 12-motor action 순서, default offset, gain,
joint/torque/speed limit, delay, E-stop과 fall protection을 독립 검증하십시오.
