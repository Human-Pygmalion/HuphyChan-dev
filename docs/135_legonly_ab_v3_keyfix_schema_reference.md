# 135. `legonly_ab_v3_keyfix` 정책 입출력 스키마 — 개발자 레퍼런스

> 표 중심, 개발자가 배포 코드를 쓸 때 바로 찾아보는 문서. 왜 이렇게 됐는지의 설명은
> [[134_legonly_ab_v3_keyfix_policy_eli5]], 실기와의 대응/격차는
> [[136_legonly_ab_v3_keyfix_sim2real_gaps]] 참고.

## 0. 체크포인트 정체성

| 항목 | 값 |
|---|---|
| 체크포인트 | `mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2/model_27999.pt` |
| SHA-256 | `9eec72518a5475bf49fcdd03aae065778b0f3de2ede7edb290cf0039772488ee` |
| iter | 27999 |
| 로봇 모델 | `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml` (12 액션 관절, 발목 2-RSU 폐루프) |
| 태스크 | `Mjlab-Velocity-Flat-Pygmalion` |
| 학습 환경변수(`PYG_*`) | `PYG_V2=1 PYG_INIT_BENT=1 PYG_INIT_MID=1 PYG_MOTOR_MEAS=1 PYG_TN=1 PYG_SAFE_TARGET_CLIP=1 PYG_STUDENT_TEACHER=1 PYG_ARM_ABD_DEG=15 PYG_ANKLE_MODE=AB PYG_MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix` |
| 정책 계약(baked) | `/home/syaro/pyg_fea/pygviewer/cache/LegOnly-AB__2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2__model_27999.policy_contract.json` |
| ONNX 사본(오라클용) | `...model_27999.onnx`, SHA-256 `47a49fbcb886f938d0490bdc2a4706edafa81463c011e10f425a8a0fbc50c396` |

## 1. 체크포인트 파라미터 키 이름 (actor만, `torch.load(...)['actor_state_dict']`)

프리픽스 `actor.` 없음 — 다음 키가 그대로 최상위:

| 키 | shape | 배포에서 사용? |
|---|---|---|
| `obs_normalizer._mean` | `(1, 45)` | 예 (squeeze → `(45,)`) |
| `obs_normalizer._std` | `(1, 45)` | 예 (squeeze → `(45,)`) |
| `obs_normalizer._var` | `(1, 45)` | 아니오(학습 중 running-var 갱신용, 추론엔 `_std`만 씀) |
| `obs_normalizer.count` | `()` | 아니오 |
| `distribution.std_param` | `(12,)` | **아니오** — 배포는 항상 결정론적 평균(=raw MLP 출력 그대로)을 씀. std는 학습 중 샘플링에만 쓰임 |
| `mlp.0.weight` / `mlp.0.bias` | `(512,45)` / `(512,)` | 예 |
| `mlp.2.weight` / `mlp.2.bias` | `(256,512)` / `(256,)` | 예 |
| `mlp.4.weight` / `mlp.4.bias` | `(128,256)` / `(128,)` | 예 |
| `mlp.6.weight` / `mlp.6.bias` | `(12,128)` / `(12,)` | 예 |

(`critic_state_dict`도 파일에 들어있지만 통째로 무관 — critic은 privileged/sim-only이며 배포되지 않음.)

## 2. 관측(actor), 45차원, 정확한 순서와 바이트 오프셋

**실측으로 확인됨**(정적 코드 추적이 아니라, 실제 40스텝 롤아웃 `obs_parity.npz`에서
각 블록을 역산해 대조 — `tools/policy_numpy_ref/tests/test_history_order_empirical.py`).

| 오프셋 | 항목(`obs_terms[].name`) | 차원 | 원천 |
|---|---|---:|---|
| `[0:3]` | `base_ang_vel` | 3 | `robot/imu_ang_vel` 센서(rad/s), 노이즈는 학습중만·추론시 `enable_corruption=False` |
| `[3:6]` | `projected_gravity` | 3 | `-robot/imu_upvector` 센서(framezaxis 부호반전) |
| `[6:18]` | `motor_pos_history[0]` = q(t−1) | 12 | 아래 §4 관절순서, `q(t−1) − default_q` |
| `[18:30]` | `motor_pos_history[1]` = q(t) | 12 | 같은 순서, `q(t) − default_q` |
| `[30:42]` | `actions` | 12 | 직전 스텝의 **raw**(스케일 전) action, 같은 관절순서 |
| `[42:45]` | `command` | 3 | `(vx, vy, ωz)`, 바디 프레임 |

`motor_pos_history`는 `history_length=2, flatten_history_dim=True`로 만든 `(2,12)`를
row-major reshape한 것 — **시간블록이 먼저, 관절축이 나중**(`[q(t−1)_전관절, q(t)_전관절]`,
관절이 뒤섞여 시간별로 나뉜 게 **아니라** 시간이 크게 나뉘고 그 안에서 관절순.
`t-1`은 "직전 컨트롤 스텝"이며, 에피소드의 첫 스텝에서는 `t-1 == t`로 백필됨
(순환버퍼가 처음엔 과거가 없어 현재값을 채움).

## 3. 관측·행동 관절 순서 (둘 다 동일, 12개)

```
L_hip_pitch_joint, L_hip_roll_joint, L_hip_yaw_joint, L_knee_joint,
L_crank_A_joint,   L_crank_B_joint,
R_hip_pitch_joint, R_hip_roll_joint, R_hip_yaw_joint, R_knee_joint,
R_crank_A_joint,   R_crank_B_joint
```

(`obs_joint_names`와 `action_joint_names`가 원소 단위로 완전히 동일함을 계약 JSON에서
직접 대조 확인.) 발목 pitch/roll은 이 12개에 없음 — 2-RSU 폐루프의 **패시브** 관절이라
크랭크 A/B(액추에이터)에 종속적으로 따라오며, 관측/행동 어디에도 직접 나타나지 않음.

## 4. 관측 정규화

$$\text{normalized} = \frac{\text{obs} - \text{mean}}{\text{std} + \epsilon}, \quad \epsilon = 0.01$$

`mean`/`std`는 §1의 체크포인트 버퍼에서 그대로 읽음(학습 중 누적된 값이 추론시엔 동결).
`rsl_rl.modules.normalization.EmpiricalNormalization`의 `eps` 기본값(오버라이드 없음).

## 5. 신경망 (actor MLP)

| 층 | 형태 | 활성함수 | 체크포인트 키 |
|---|---|---|---|
| 0 | Linear(45→512) | ELU(α=1.0) | `mlp.0.*` |
| 2 | Linear(512→256) | ELU(α=1.0) | `mlp.2.*` |
| 4 | Linear(256→128) | ELU(α=1.0) | `mlp.4.*` |
| 6 | Linear(128→12) | **없음**(선형 그대로) | `mlp.6.*` |

배포 시 행동 = `mlp.6`의 출력 그대로(결정론적 평균, `distribution.std_param` 무관).

## 6. 행동 후처리 → 최종 목표각

**순서 고정**(코드로 확인: `mjlab/envs/mdp/actions/actions.py` `process_actions`→`apply_actions`):

$$\text{processed} = \text{raw\_action} \times \text{scale} + \text{default\_q}$$
$$\text{clipped} = \mathrm{clip}(\text{processed},\ \text{safe\_clip\_lo},\ \text{safe\_clip\_hi})$$
$$\text{target} = \text{clipped} - \text{encoder\_bias}$$

- `scale`: 전 관절 동일값 **0.25** (`PYG_ACTION_SCALE`).
- `default_q`(액션 오프셋, §3 순서):

  | 관절 | rad | deg |
  |---|--:|--:|
  | L_hip_pitch | −0.175 | −10.03 |
  | L_hip_roll | 0.0 | 0.0 |
  | L_hip_yaw | 0.0 | 0.0 |
  | L_knee | 0.350 | 20.05 |
  | L_crank_A | 0.29715 | 17.03 |
  | L_crank_B | −0.30121 | −17.26 |
  | R_hip_pitch | 0.175 | 10.03 |
  | R_hip_roll | 0.0 | 0.0 |
  | R_hip_yaw | 0.0 | 0.0 |
  | R_knee | −0.350 | −20.05 |
  | R_crank_A | −0.29715 | −17.03 |
  | R_crank_B | 0.30121 | 17.26 |

- `safe_clip`(이 체크포인트는 `PYG_SAFE_TARGET_CLIP=1` — 각 관절 자체 기계범위의 **중앙
  90%**, `pygmalion_constants.py`의 `safe_target_clip()` 공식 그대로):

  | 관절 | 하한(rad) | 상한(rad) |
  |---|--:|--:|
  | L_hip_pitch | −1.9679 | 0.3098 |
  | L_hip_roll | −0.3403 | 1.3875 |
  | L_hip_yaw | −0.7069 | 0.7069 |
  | L_knee | 0.1047 | 1.9897 |
  | L_crank_A | −1.08 | 1.08 |
  | L_crank_B | −1.08 | 1.08 |
  | R_hip_pitch | −0.3098 | 1.9679 |
  | R_hip_roll | −1.3875 | 0.3403 |
  | R_hip_yaw | −0.7069 | 0.7069 |
  | R_knee | −1.9897 | −0.1047 |
  | R_crank_A | −1.08 | 1.08 |
  | R_crank_B | −1.08 | 1.08 |

  (L/R가 부호만 미러인 것에 주목 — v30 모델의 좌우 미러축을 이 값들이 이미 반영함;
  `legonly_ab_v1`이 이걸 놓쳐 폐기된 배경은 [[136_legonly_ab_v3_keyfix_sim2real_gaps]] 참고.)

- `encoder_bias`: sim 전용 DR 항, **실기엔 대응 없음 — 항상 0으로 취급**할 것
  ([[136_legonly_ab_v3_keyfix_sim2real_gaps]] §4).

**이 `target`이 실기 wire의 `q_target`에 대응하는 지점이다.** MuJoCo `data.ctrl`이 아님
— 이 체크포인트의 12관절은 전부 `TnPdActuatorCfg`(토크-속도 곡선으로 clamp된 파이썬
PD)라 `ctrl`은 한 단계 더 지나간 **토크**값이며, 이 스키마의 범위 밖.

## 7. 제어 주기

| 항목 | 값 |
|---|--:|
| physics timestep | 0.005 s |
| decimation | 4 |
| 정책/제어 스텝 | 0.02 s = **50 Hz** |
| 실기 대응 | HUPHY 브릿지 UDP 송신도 50 Hz(`tx_client.py DEFAULT_HZ=50.0`) — 정확히 일치 |

## 8. 아티팩트 경로 (재현·검증용)

| 아티팩트 | 경로 |
|---|---|
| numpy 전용 재구현 | `tools/policy_numpy_ref/policy_numpy_ref/` |
| 추출된 numpy 가중치 | `tools/policy_numpy_ref/artifacts/legonly_ab_v3_keyfix_weights.npz` |
| 실제 롤아웃 훅 트레이스 | `tools/policy_numpy_ref/artifacts/legonly_ab_v3_keyfix_live_trace.npz` |
| 검증 결과 요약 | `tools/policy_numpy_ref/artifacts/tolerance_report.json` |
| 검증 테스트 | `tools/policy_numpy_ref/tests/` (`pytest`) |
| 정책 계약(원본, pygviewer가 구움) | `/home/syaro/pyg_fea/pygviewer/cache/LegOnly-AB__...__model_27999.policy_contract.json` |
