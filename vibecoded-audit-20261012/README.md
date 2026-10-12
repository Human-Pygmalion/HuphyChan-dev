# vibecoded-audit-20261012

Pygmalion 하체 이족 로봇의 최신 보행 policy (`rough_main_model_11999`, 학습 run
`2026-10-11_23-35-20_rough_box_main_v11a`) 에 대한 **audit 문서**와, 같은 학습을 **순정 mjlab**
에서 재현하는 **task 꾸러미**다. mjlab 소스는 수정하지 않는다.

| 폴더 | 내용 |
|---|---|
| `audit/` | Audit 문서: [README_ko.md](audit/README_ko.md) (한국어, 최신), [README.md](audit/README.md) (English, 이전판 — reward 계산식 §1.6.2 와 curriculum §2 개정은 한국어판에만 있음) |
| `pygmalion_mjlab/` | Task 패키지. 설치하면 mjlab 에 `Pygmalion-Velocity-Rough-Box` 가 등록된다 |
| `checkpoints/` | Warm start 용 평지 parent actor, 배포 policy (.pt / .onnx) |
| `reference/` | 원 학습 run 의 `env.yaml`, `agent.yaml`, launch 기록 (비교 기준) |
| `scripts/` | 학습·재생·영상·검증 스크립트 |

- 기준 mjlab: upstream `mujocolab/mjlab` commit **`20f10e96`** (mjlab 1.5.0, 2026-06-28). 원 학습 코드는 이 commit 에서 갈라진 fork 였다.
- 환경 변수 (`PYG_*`) 없음. 모든 값이 `pygmalion_mjlab/*.py` 에 직접 적혀 있다.
- 로봇 모델: `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop` (최신 하체 모델, 12 motor, 2-RSU ankle). 공통 관절 24 개의 축·위치·범위가 최신 전신 모델 (`FullDoF_..._v30_proxyfix_shell_loop`) 과 같다 (2026-10-12 확인).

## 1. 설치

요구: Linux, NVIDIA GPU (CUDA), [uv](https://docs.astral.sh/uv/).

```bash
# 1) mjlab 을 기준 commit 으로 받는다
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
git checkout 20f10e96
uv sync                     # torch, mujoco 3.10, mujoco-warp 3.10, rsl-rl-lib 5.4.0 등

# 2) 이 폴더를 mjlab 환경에 설치한다 (editable)
uv pip install -e /path/to/vibecoded-audit-20261012

# 3) 설치 확인: task 가 보이면 된다
uv run list-envs | grep Pygmalion          # -> Pygmalion-Velocity-Rough-Box
```

`pyproject.toml` 이 `mjlab.tasks` entry point 를 등록한다. mjlab 이 시작할 때 이 패키지를 import 하므로
stock 명령 (`uv run train`, `uv run play`) 에서 task 를 바로 쓸 수 있다.

이 폴더의 `scripts/*` 는 mjlab 환경의 python 으로 이 폴더에서 실행한다.

```bash
cd /path/to/vibecoded-audit-20261012
export MJLAB_PYTHON=/path/to/mjlab/.venv/bin/python
$MJLAB_PYTHON scripts/verify_config.py     # -> RESULT: MATCH
```

`scripts/rollout_trace.py` 만 `onnxruntime` 이 추가로 필요하다 (`uv pip install onnxruntime`).

## 2. 사용법

| 목적 | 명령 |
|---|---|
| 설정 검증 (원 run 의 `env.yaml`·`agent.yaml` 과 값 비교) | `$MJLAB_PYTHON scripts/verify_config.py` |
| 배포 policy 재학습 (원 run 과 같은 절차, 학습 영상 포함) | `bash scripts/train_released.sh 12000 4096 released` |
| 짧은 확인 학습 | `bash scripts/train_released.sh 1000 4096 check` |
| 처음부터 학습 (warm start 없이) | `uv run train Pygmalion-Velocity-Rough-Box --env.scene.num-envs 4096` (mjlab 폴더에서) |
| Policy 재생 (interactive viewer) | `$MJLAB_PYTHON scripts/play.py Pygmalion-Velocity-Rough-Box --checkpoint-file checkpoints/rough_main_model_11999.pt` |
| 시연 영상 (headless, 실시간 50 fps, 명령당 15 s) | `$MJLAB_PYTHON scripts/record_video.py --checkpoint checkpoints/rough_main_model_11999.pt --out demo.mp4` |
| 학습 곡선 비교 | `$MJLAB_PYTHON scripts/compare_curves.py RUN_A RUN_B --upto 1000 --png curves.png` |
| 같은 seed 롤아웃 기록·비교 | `scripts/rollout_trace.py` (docstring 참고) |

`train_released.sh` 는 `MJLAB_PYTHON` 을 쓴다. 로그는 `logs/rsl_rl/pygmalion_velocity/<날짜>_<run_name>/`
(checkpoint, `params/`, `videos/train/`). `scripts/train.py` 의 나머지 인자는 mjlab `train` CLI 로 그대로 넘어간다.

### 배포 policy 의 학습 절차

1. **Actor-only warm start**: `checkpoints/parent_flat_actor_model_29998.pt` (평지 30,000 iter) 에서 actor weight·action std·obs normalizer 만 불러온다. Critic·optimizer·iteration 은 새로 시작하고 env step counter 를 0 으로 되돌린다. Rough task 의 critic 입력 (height scan 포함) 이 parent 와 달라 critic 은 불러올 수 없다.
2. **Critic warm-up 300 update**: 처음 300 PPO update 동안 actor 를 update 직후 되돌려 critic 만 학습한다.
3. 이후 12,000 iter PPO (4,096 env, seed 43, entropy_coef 0.002).

1·2 는 stock `train` 에 옵션이 없어 `scripts/train.py` 가 `MjlabOnPolicyRunner.learn` / `alg.update` 를 감싸 구현한다.

Reward·curriculum·DR·Kp/Kd 의 계산 방법과 근거는 [audit/README_ko.md](audit/README_ko.md) §1.6, §2, §3, §5 에 있다.

## 3. 원 학습 코드 (fork) 와 다른 점

동작은 같고 표현만 다르다.

| 항목 | fork | 이 꾸러미 | 같은 이유 |
|---|---|---|---|
| Command 범위 | gated command curriculum. 첫 iteration 에 최종 stage 로 승급 (audit §2.3) | 최종 범위 (vx −2.0~2.5, vy ±1.0, wz ±1.0) 를 처음부터 사용 | 원 run 의 `Curriculum/command_vel/stage` 가 iter 1 부터 최종값 |
| Reward 항별 clamp | `RewardManager` 를 고쳐 항별 ±50·dt | 각 reward 함수를 `clamped()` 로 감쌈 | 함수 출력 × weight × dt 순서와 nan→0 처리가 같다 |

Fork 가 mjlab core 에 넣었던 나머지 변경 (box-only terrain, actor height scan 제거 등) 은 `env_cfg.py` 의 override 로 옮겼다.

## 4. 검증 결과 (RTX 3090, 2026-10-12)

순정 mjlab `20f10e96` 에서 실행했다. `mjlab.__file__` 이 순정 경로인 것을 확인했다.

1. **설치**: 별도 venv 에 순정 mjlab + 이 패키지를 설치 → stock `list-envs` 에 task 표시, `verify_config` MATCH.
2. **설정 동일성** (`verify_config.py`): env 0 건, agent 0 건 불일치. 예상된 차이만 있다 (command 범위, command curriculum, mesh API 필드, `ls_parallel`, run bookkeeping). Weight ≠ 0 인 reward 항은 전부 clamp 적용.
3. **동작 동일성** (`rollout_trace.py`): 배포 policy (ONNX), 8 env × 200 step, 같은 seed·명령 (vx 0.6). 이 꾸러미 + 순정 mjlab 과 원 fork 코드의 actor obs·action·base 위치 **최대 차이 0.0** (비트 단위 일치).
4. **시연 영상** (`record_video.py`): 120 s, 8 개 명령 × 15 s. 넘어짐 2 회 — 둘 다 terrain tile 경계의 약 0.2 m 단차 (tilted_grid tile 가장자리) 로 걸어 들어갈 때.
5. **실제 학습**: §5.

## 5. 순정 mjlab 학습 확인

<!-- TRAINING-RESULT -->
진행 중.
