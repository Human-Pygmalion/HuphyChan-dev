# HUPHY 00545 Union-DR DeployGains checkpoint 49,000

00545 모션 트래커의 실제 배포 gain 파인튜닝 스냅샷입니다. 기존
`huphy_00545_uniondr_30000`의 전체 체크포인트에서 이어서 학습했으며, 이번 단계부터
hip pitch/roll `135/6`, hip yaw `75/4`, knee `130/6`, ankle A/B `60/2.7`
(`Kp/Kd`)을 사용했습니다. 기존 체크포인트의 옛 gain과 혼용하지 마십시오.

## 포함 파일

| 파일 | 용도 |
|---|---|
| `model_30000.pt` | 파인튜닝 시작점 전체 체크포인트 |
| `model_49000.pt` | 최신 actor/critic/optimizer 전체 체크포인트 |
| `policy.onnx` | 114 입력, 12 출력 deterministic actor |
| `motion_policy.onnx` | actor와 00545 reference 출력을 함께 포함 |
| `motions/00545_interactionmesh_50hz.npz` | 659 frame, 13.18초 학습 reference |
| `env.yaml`, `agent.yaml` | 실제 런이 저장한 원본 설정 |
| `deploy.yaml` | observation/action/gain/DR/hash 배포 계약 |
| `source_overlay/` | 기존 00545 릴리스 위에 적용할 DeployGains 소스 변경분 |

Actor observation은 reference joint position/velocity 48, projected gravity 3, base
angular velocity 3, joint position 24, joint velocity 24, previous action 12 순서입니다.
첫 frame의 previous action은 0이며 출력은 12개 A/B motor target입니다.
`q_target = q_default + 0.25 * action`을 적용합니다.

Union DR은 발 마찰 0.3–1.7, payload -1–8 kg, gain 및 motor strength 0.8–1.2,
encoder/IMU/지연/관성 randomization과 0–100 N torso push를 포함합니다. 전체 값은
`env.yaml`, 실제 배포 계약은 `deploy.yaml`을 따르십시오.

## 소스 설치와 재현

```bash
export HUPHYCHAN_ROOT=/path/to/HuphyChan-dev
export UNITREE_RL_MJLAB=/path/to/unitree_rl_mjlab
export MJLAB_PYTHON=/path/to/miniconda/envs/mjlab/bin/python
cd "$HUPHYCHAN_ROOT/mjlab_policy/checkpoints/huphy_00545_uniondr_deploygains_49000"
cp -a ../huphy_00545_uniondr_30000/source/unitree_rl_mjlab/. "$UNITREE_RL_MJLAB/"
cp -a source_overlay/unitree_rl_mjlab/. "$UNITREE_RL_MJLAB/"

GPU=0 NUM_ENVS=2048 bash scripts/reproduce_finetune.sh
```

포함된 `model_30000.pt`에서 19,000 iteration을 실행합니다. 최신 체크포인트에서 더
이어가려면 `bash scripts/resume_latest.sh`를 사용합니다. GPU 병렬 학습은 bitwise
동일성을 보장하지 않습니다.

재생 및 검증:

```bash
GPU=0 VISER_PORT=8082 bash scripts/play.sh
"$MJLAB_PYTHON" scripts/verify_release.py
```

실제 로봇 배포 전 observation 순서·부호, A/B motor 방향, PD gain 단위, default offset,
joint limit, emergency stop과 낙상 방지를 독립적으로 확인해야 합니다.
