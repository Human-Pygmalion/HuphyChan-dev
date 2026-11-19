# HUPHY 00399 Union-DR DeployGains checkpoint 39,000

00399 모션 트래커의 실제 배포 gain 파인튜닝 스냅샷입니다. 기존
`huphy_00399_uniondr_20000`의 전체 체크포인트에서 이어서 학습했으며, 이번 단계부터
아래 nominal gain을 사용했습니다.

```text
hip pitch/roll: Kp 135, Kd 6
hip yaw:        Kp 75,  Kd 4
knee:           Kp 130, Kd 6
ankle A/B:      Kp 60,  Kd 2.7
```

기존 `model_20000`의 옛 gain은 150/150/220/22.265625 계열이므로 배포할 때 서로
혼동하면 안 됩니다. 이 폴더의 `model_39000.pt`, ONNX와 `deploy.yaml`은 위 새 gain
task에 해당합니다.

## 포함 파일

| 파일 | 용도 |
|---|---|
| `model_20000.pt` | 파인튜닝 시작점 전체 체크포인트 |
| `model_39000.pt` | 최신 actor/critic/optimizer 전체 체크포인트 |
| `policy.onnx` | 114 입력, 12 출력 deterministic actor |
| `motion_policy.onnx` | actor와 00399 reference 출력을 함께 포함 |
| `motions/00399_interactionmesh_50hz.npz` | 1,511 frame, 30.22초 학습 reference |
| `env.yaml`, `agent.yaml` | 실제 런이 저장한 원본 설정 |
| `deploy.yaml` | observation/action/gain/DR/hash 배포 계약 |
| `source_overlay/` | 기존 00399 릴리스 위에 적용할 DeployGains 소스 변경분 |

Actor observation 순서는 reference joint position/velocity 48, projected gravity 3,
base angular velocity 3, joint position 24, joint velocity 24, previous action 12입니다.
첫 policy frame의 previous action은 0이어야 합니다. 출력은 12개 A/B motor position
target이며 `q_target = q_default + 0.25 * action`입니다. 정확한 slice와 joint order는
`deploy.yaml`을 따르십시오.

Union DR은 발 마찰 0.3–1.7, payload -1–8 kg, gain 및 motor strength 0.8–1.2,
encoder/IMU/지연/관성 randomization과 0–100 N torso push를 포함합니다. 전체 값은
`env.yaml`이 authoritative source입니다.

## 소스 설치와 재현

```bash
export HUPHYCHAN_ROOT=/path/to/HuphyChan-dev
export UNITREE_RL_MJLAB=/path/to/unitree_rl_mjlab
export MJLAB_PYTHON=/path/to/miniconda/envs/mjlab/bin/python
cd "$HUPHYCHAN_ROOT/mjlab_policy/checkpoints/huphy_00399_uniondr_deploygains_39000"
cp -a ../huphy_00399_uniondr_20000/source/unitree_rl_mjlab/. "$UNITREE_RL_MJLAB/"
cp -a source_overlay/unitree_rl_mjlab/. "$UNITREE_RL_MJLAB/"

GPU=0 NUM_ENVS=2048 bash scripts/reproduce_finetune.sh
```

재현 스크립트는 포함된 `model_20000.pt`에서 19,000 iteration을 실행합니다. GPU 병렬
학습 RNG 상태 전체는 체크포인트에 저장되지 않으므로 bitwise 동일성은 보장되지 않습니다.
최신 체크포인트에서 더 이어가려면 `bash scripts/resume_latest.sh`를 사용합니다.

재생 및 검증:

```bash
GPU=0 VISER_PORT=8081 bash scripts/play.sh
"$MJLAB_PYTHON" scripts/verify_release.py
```

실제 로봇 배포 전 observation 순서·부호, A/B motor 방향, PD gain 단위, default offset,
joint limit, emergency stop과 낙상 방지를 독립적으로 확인해야 합니다.
