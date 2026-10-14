# 실기 배포 안내 (2026-10-14)

배포 정책 `rough_box_main_v11a` (checkpoint 11999) 를 실물 HUPHY 로봇에서 돌리는 데 필요한 것을 이 폴더에 모았다. 원래는 학습 저장소·HUPHY 저장소·문서 여러 곳에 흩어져 있었다.

| 경로 | 내용 |
|---|---|
| `asset/` | 배포 꾸러미 `pyg_rough_main_20261012` 사본 (영문). 관측 45 칸 정의·행동 공식·참조 입출력 16 쌍 (`deploy_contract.json`), numpy+onnxruntime 실행기, 모터 표. `policy.pt` 는 `../checkpoints/rough_main_model_11999.pt` 와 같은 파일이라 뺐다 |
| `sim2real/2026-10-13_huphy_archivedlogs_conditions.md` | 2026-10-13 실기 시험 로그별 조건, 실행 코드 한 주기 추적 (HUPHY 파일·줄 번호) |
| `sim2real/2026-10-13_sim2real_gap_analysis.md` | 실기 실패 원인 분석, IMU 부호 문제 (§4) |
| `sim2real/2026-10-14_dr_reference_latency_imu.md` | 실측 지연·IMU 값 → DR 설정 항목·권장 범위 |
| `sim2real/2026-10-13_imu_frames.html` | 시뮬레이션·실기 IMU 좌표계 설명 (브라우저로 열기, 3D) |

`sim2real/` 문서 안의 `tools/sim2real/...`, `docs/...` 경로는 원 학습 저장소 (main) 기준이다. 이 브랜치에는 없다.

## 1. 배포 코드는 HUPHY `biped` 브랜치

로봇 쪽 실행 코드는 [Human-Pygmalion/HUPHY](https://github.com/Human-Pygmalion/HUPHY) 의 **`biped` 브랜치**다 (2026-10-13 기준 commit `a1b67b1`).

**`main` 브랜치를 보면 안 된다.** `main` 의 `config/robot.yaml` 은 옛 단일다리 시제품 설정 `robot_v0.5.yaml` 을 가리킨다. 그래서 다음이 보인다:
- 왼다리 Kp/Kd 가 비어 있음 ("아직 연결 전")
- 관절 한계 null
- 모터 RS02/RS00

`biped` 의 `robot.yaml` 은 `robot_v1.0.yaml` 을 가리키고, 아래 표의 값이 들어 있다.

실행 (모터 전원 전 `asset/README.md` 를 먼저 읽을 것):
```bash
huphy-run --robot --policy stand --ankle-space ab --weights config/policies/stand_bi.pt \
          --gain-scale 1.0 1.0 --drain-ms 4 --hz 50 --log
```
`config/policies/stand_bi.pt` 는 이 브랜치의 `checkpoints/rough_main_model_11999.pt` 와 같은 파일이다 (sha256 `8a067c9b…`).

## 2. 모터와 게인 (학습 = 실기, 2026-10-14 확정)

| 관절 | 모터 | Kp | Kd | 토크 한계 (Nm) |
|---|---|---|---|---|
| hip_pitch, hip_roll | RobStride RS04 | 150 | 6 | 120 |
| knee | RS04 | 220 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| crank_A, crank_B (발목, 2-RSU) | RS03 | 22.266 | 1.414 | 60 |

- 학습 모델 (`pygmalion_mjlab/robot.py`), HUPHY `biped` 의 `robot_v1.0.yaml`, 배포 꾸러미 모두 같다.
- `asset/README.md` 의 미해결 항목 5 ("하드웨어 맵에는 12 개 전부 RS03") 는 2026-10-14 사용자 확인으로 위 표가 맞다고 정리됐다.
- HUPHY 는 Kp·Kd 를 MIT 명령 12 비트로 보낸다 (Kp 0~5000, Kd 0~100, 버림). 배율 1.0 에서는 발목 Kp 가 22.266 → 22.222 로 거의 같다. 배율 0.05 에서는 발목 Kp 가 0 이 된다.

관측·행동 규약 요약 (정본은 `asset/deploy_contract.json`):
- 45 칸 = 자이로 (rad/s) 3 + 투영 중력 3 + 관절각 (기본 자세를 뺀 값) 12 × 2 프레임 (한 주기 전, 지금) + 직전 행동 (모델 출력 그대로) 12 + 명령 (vx, vy, wz) 3
- 관절 순서: 오른다리 6 개, 그다음 왼다리 6 개
- 목표각 = 기본 자세 + 0.25 × 행동 (rad)

## 3. 2026-10-13 실기 시험에서 나온 것 — 다음 시험 전에 볼 것

| 항목 | 상태 | 할 일 |
|---|---|---|
| **IMU 투영 중력 x (앞뒤 기울기) 부호가 시뮬레이션과 반대로 정책에 들어감** | 실기 로그 6 개 모두에서 중력과 자이로가 x 축 거울상 (HUPHY 자체 검사 함수로도 같은 판정). 원인: EBIMU 자세 좌표계 (몸체 위 = −Z) 를 자이로 좌표계로 바꾸지 않고 `R^T(0,0,−1)` 계산. 시뮬레이션에서 이 부호만 뒤집어도 게인 1.0 에서 2.2 초 안에 8/8 넘어짐. 센서 부착 방향은 미확인 | 모터 끄고 앞으로 20° 숙였을 때 정책 입력 중력 x 가 +0.34, 숙이는 동안 자이로 y 가 + 인지 확인 (`sim2real/2026-10-13_sim2real_gap_analysis.md` §4-1e 판별표). **DR 로 학습시키지 말고 실기 쪽에서 고칠 것** |
| 게인 배율 | 배율 0.8 미만은 이 정책의 학습 범위 밖. 시뮬레이션에서도 0.5 이하는 바닥에서 넘어짐. 배율을 0.05 부터 올리며 보는 방식으로는 정책을 평가할 수 없음 | 배율 1.0 (또는 0.8 이상) 으로, 크레인이 몸무게를 받친 채 시작해 천천히 내림 (시뮬레이션 8/8 서기) |
| `--hz 100` | 실제 송신 78 Hz 라 정책이 약 39 Hz 로 돎 | `--hz 50` (20.0 ms 로 지켜짐) |
| 관절 관측 한 주기 지연, 명령 지연 약 1~1.5 ms, 응답 유실 1.4~3 % | 시뮬레이션에서 단독으로는 무해. 학습에는 무작위화가 없었음 | 학습 옵션 `--obs-delay`, `--act-delay`, `--gain-dr` (§4) |

## 4. 학습에 넣을 수 있는 무작위화 옵션 (기본은 꺼짐)

배포 정책은 이 무작위화 없이 학습됐다. `scripts/train.py` 와 모든 학습 스크립트에 옵션으로 넣었다.

| 옵션 | 내용 | 실측 근거 |
|---|---|---|
| `--obs-delay N` | 정책이 보는 관절각 (두 프레임 모두) 을 0~N 정책 주기 늦춤 (환경별 추첨) | 실기는 N = 1 (직전 주기 수신값, 50 Hz 에서 20 ms) |
| `--act-delay N` | 목표각이 액추에이터에 0~N 물리 스텝 (5 ms) 늦게 도달 | 실측 약 1~1.5 ms → N = 1 |
| `--gain-dr LO HI` | reset 마다 관절별 Kp·Kd 를 U(LO, HI) 배 (`mjlab.envs.mdp.dr.pd_gains`) | 예: 0.8 1.2 |

예:
```bash
# 배포 정책에서 actor 만 가져와 무작위화를 켜고 다시 학습 (critic 300 update 먼저 적응)
MJLAB_PYTHON=... bash scripts/train_continue.sh warmstart 12000 4096 dr_try --obs-delay 1 --act-delay 1 --gain-dr 0.8 1.2
# 처음부터 세 단계 모두 무작위화 켜고
TRAIN_EXTRA="--obs-delay 1 --act-delay 1 --gain-dr 0.8 1.2" MJLAB_PYTHON=... bash scripts/train_from_scratch.sh
```
응답 유실 (관절별로 직전 관측 유지) 과 IMU 부착각 치우침 (`mjlab.envs.mdp.dr.site_quat`) 은 아직 옵션이 없다. 범위는 `sim2real/2026-10-14_dr_reference_latency_imu.md` 에 있다.
