# bundleCTL_RP — RP arm 대조군 (2026-08-26)

> *한 줄*: AB에서 확정한 착지 레시피를 RP(직렬 발목)로 옮길 때 **RP 자신의 대조군**. 변경 없음.

| | |
|---|---|
| 런 | `logs/rsl_rl/pygmalion_velocity/2026-08-26_15-44-45_bundleCTL_RP` |
| 계보 | `ankleRP_c3` `model_31999` → +800 iter → **`model_32798`** (16384 env) |
| 변인 | **없음**(c3와 동일 보상). `PYG_ANKLE_MODE=RP` |
| 짝 | [[2026-08-26_bundleD1_RP]] — 같은 체크포인트·같은 배치·같은 iter, 세 변경만 다름 |

## §1 재현성
```
PYG_V2=1 PYG_INIT_BENT=1 PYG_ARM_ABD_DEG=15 PYG_INERTIAL_DR=1 PYG_ANKLE_MODE=RP
PYG_DR_START_ITER=0 PYG_DR_END_ITER=1 PYG_TN=1 PYG_MOTOR_MEAS=1 PYG_SOFT_LANDING=1
--env.scene.num-envs 16384 --agent.max-iterations 800
```
모델 `pygmalion_v3_printed.xml`(35.347 kg, 직렬 발목 + 루프 자코비안 크랭크공간 클램프).
정본 env: `analysis/out/watchdog_runs.json`.

<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->

## 1b. 이 run의 Reward & Gains (config에서 파싱 — 재현용)

**Reward 항목** (weight·왜·어떻게 — '어떻게' 칸의 수치는 이 런의 `params/env.yaml` `rewards.<항>.params` 값):

| reward | weight | 왜 | 어떻게 |
|---|--:|---|---|
| foot_clearance | **-2** | 스윙발 지면 이격(발끌림 방지) | Σ_발 \|h−0.1 m\|·‖v_발,xy‖; 명령>0.05 |
| track_angular_velocity | **+2** | 명령 회전속도 추종 | exp(-((ω_cmd,z−ω_z)²+‖ω_xy‖²)/σ²), σ=0.707 rad/s |
| track_linear_velocity | **+2** | 명령 전진/측방 속도 추종 | exp(-(‖v_cmd,xy−v_xy‖²+v_z²)/σ²), σ=0.866 m/s |
| air_time | **+1** | 체공시간 보상(질질끌기 억제) | 0.05<체공시간<0.5 s 인 발 수; 명령>0.5 |
| dof_pos_limits | **-1** | 관절범위 한계 접근 벌점 | Σ 소프트 한계 밖 초과량 [rad] |
| foot_impact_velocity | **-1** | 착지 직전 발 하강속도 감소 | Σ_발 relu(−v_z)^2 (높이<0.1 m, 힘<7† N); 명령>0.05 |
| pose | **+1** | 기본 관절자세 정규화(기괴자세 억제) | exp(-mean((q−q₀)²/σ²)) [rad]; σ 서기(명령<0.05): 전 관절 0.05 / 걷기: hip_pitch 0.3·hip_roll 0.4·hip_yaw 0.15·knee 1.2·ankle_pitch 0.25·ankle_roll 0.1 / 달리기(명령≥1.5): hip_pitch 0.5·hip_roll 0.2·hip_yaw 0.2·knee 1.5·ankle_pitch 0.35·ankle_roll 0.15 |
| self_collisions | **-1** | 자기충돌 벌점 | 자기충돌 힘>10 N 인 부분스텝 수 |
| stand_still_penalty | **-1** | 이동 명령을 무시하고 서 버리는 정지 방지 | \|v_cmd\|>0.3† 또는 \|ω_cmd\|>0.3† 인데 명령방향 진행<0.3†×명령이면 1 |
| track_lin_vel_progress | **+1** | 고속 명령에서 정지하는 국소 최적해 방지 | clamp(min(v·ĉ, \|v_cmd\|), 0) [m/s] (명령방향 진행 속도, 명령 크기에서 상한) |
| upright | **+1** | 몸통 직립 유지(넘어짐 방지) | exp(-‖g_xy‖²/σ²) (몸통 좌표계로 투영한 중력), σ=0.447 |
| knee_overspeed | **-0.5** | 실측 RS04 무부하 속도를 넘는 보행 억제 | Σ_무릎 relu(\|q̇\|−19.9 rad/s)² |
| foot_swing_height | -0.25 | 스윙발 높이 성형 | 착지 순간 Σ(최고높이/0.1 m−1)²; 명령>0.05 |
| action_rate_l2 | -0.1 | 액션 급변 벌점(진동/저크 억제) | ‖a_t−a_(t−1)‖² |
| foot_slip | -0.1 | 접지발 미끄러짐 벌점 | Σ_접지발 ‖v_발,xy‖²; 명령>0.05 |
| body_ang_vel | -0.05 | 몸통 각속도 벌점(흔들림 억제) | ‖ω_xy‖² (몸통 롤·피치 각속도) |
| angular_momentum | -0.02 | 전신 각운동량 벌점(회전 낭비 억제) | ‖L‖² (전신 각운동량) |
| thermal_effort | -0.02 | ★열분배: Σ(τ/정격)² 정규화(관절 균등화) | Σ(τ/τ_정격)², τ_정격 [N·m]: hip_pitch 40·hip_roll 40·hip_yaw 20·knee 40·ankle_pitch 32.7·ankle_roll 27.9·crank_[AB] 20 |
| contact_force_cap | -0.01 | ★충격 상한: 발 지면반력 역치초과분 벌점(사뿐착지) | Σ_발 min(max(‖F‖−420 N, 0), 560 N) |
| soft_landing | -1e-05 | 착지 첫접촉 충격 벌점(약) | Σ 첫 접촉 순간 ‖F‖ [N]; 명령>0.05 |
| torque_limit | -0 | 명령 토크의 한계초과 벌점 | Σ_구동기 relu(\|τ_명령\|/τ_max−0.7) |

† env.yaml에 값이 없어 함수 기본값으로 실행된 것 — 이 런이 실행한 코드(mjlab 커밋 3286e112)의 함수 정의에서 읽음.

**관절별 Kp/Kd** (position-PD, effort=관절측 peak):

| 관절 | 모터 | Kp(stiffness) | Kd(damping) | effort [N·m] |
|---|---|--:|--:|--:|
| hip_pitch | RS04 | 150 | 6 | 120 |
| hip_roll | RS04 | 150 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| knee | RS04 | 220 | 6 | 120 |
| ankle_pitch | RS03 | 28.5 | 1.81 | 110 |
| ankle_roll | RS00 | 28.5 | 1.81 | 110 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-08-26_15-44-45_bundleCTL_RP`)

| 관절 그룹 | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] | effort 한계 [N·m] | 무부하 속도 [rad/s] | 로터 관성 armature [kg·m²] | 쿨롱 마찰 [N·m] | 점성 [N·m·s/rad] | T-N 곡선 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| ankle_pitch | RS03 | 28.5 | 1.81 | 110 (stall 59.7) | 20.94 | — | — | — | 실측 37점 (`PYG_TN=1`) |
| hip_yaw | RS03 | 150 | 6 | 60 (stall 59.7) | 20.94 | 0.01527 | 0.285 | 0.0223 | 실측 37점 (`PYG_TN=1`) |
| hip_pitch | RS04 | 150 | 6 | 120 (stall 120.1) | 20.94 | 0.01633 | 0.269 | 0.0095 | 실측 22점 (`PYG_TN=1`) |
| knee | RS04 | 220 | 6 | 120 (stall 120.1) | 20.94 | 0.01633 | 0.269 | 0.0095 | 실측 22점 (`PYG_TN=1`) |

토크는 `effort_limit`과 (있으면) 실측 T-N 곡선의 속도의존 상한 중 **작은 값**으로 클램프된다. armature/쿨롱/점성은 모터 실측값(`PYG_MOTOR_MEAS=1`)이면 실측, 아니면 카탈로그 추정치다.

**§1b-3. ROM 한계·액션 창** (모델 XML range · soft 한계 = 중심±0.5·range×0.9 (mjlab `Entity` 규약) · 액션 clip = env.yaml `actions.joint_pos.clip` · 창 = clip 폭 · default = 액션 0 자세)

**soft 한계와 액션 clip은 같은 공식이다** — `Entity.soft_joint_pos_limits`와 `pygmalion_constants.safe_target_clip()`이 둘 다 *중심 ± 0.5·range·factor*를 쓴다(각 경계에 factor를 곱하는 것이 아니다: 비대칭 관절에서 두 식이 갈린다 — knee `[0,120]`은 `[6,114]`이지 `[0,108]`이 아니다). 그래서 `PYG_SAFE_TARGET_CLIP=1`인 런에서는 두 열이 정확히 일치하고, 정책이 통과하는 클램프와 시뮬레이터가 강제하는 클램프가 하나의 계약이 된다.

모델 출처: v3 printed 계열 — 노트 [[2026-08-23_ankleAB_c2]]/[[2026-08-23_ankleRP_c2]]가 선언한 모델이며, v4 XML은 2026-08-26_21-22에야 생성되어 이 런이 로드할 수 없었다(객관 상한) — `pygmalion_v3_printed.xml`

| 관절 | XML range [°] | soft 한계 [°] | 액션 clip [°] | 사용가능 창 [°] | default [°] | 구동 |
|---|---|---|---|--:|--:|---|
| L/R_hip_pitch_joint | [-120, 25] | [-112.8, 17.7] | n/a (구 설정: clip 없음) | 130.5 | -18.33 | 액션 |
| L/R_hip_roll_joint | [-85, 25] | [-79.5, 19.5] | n/a (구 설정: clip 없음) | 99 | 0 | 액션 |
| L/R_hip_yaw_joint | [-45, 45] | [-40.5, 40.5] | n/a (구 설정: clip 없음) | 81 | 0 | 액션 |
| L/R_knee_joint | [-120, 0] | [-114, -6] | n/a (구 설정: clip 없음) | 108 | -38.39 | 액션 |
| L/R_ankle_pitch_joint | [-50, 30] | [-46, 26] | n/a (구 설정: clip 없음) | 72 | 20.63 | 액션 |
| L/R_ankle_roll_joint | [-20, 20] | [-18, 18] | n/a (구 설정: clip 없음) | 36 | 0 | 액션 |
| waist_yaw_joint | [-60, 60] | [-54, 54] | — (수동) | — | 0 | 수동 |
| L/R_shoulder_pitch_joint | [-180, 60] | [-168, 48] | — (수동) | — | 0 | 수동 |
| L/R_shoulder_roll_joint | [-32, 30] | [-28.9, 26.9] | — (수동) | — | 0 | 수동 |

액션 스케일 0.25 rad/단위, 오프셋 = default (`use_default_offset`). clip이 없는 구 설정에서는 정책 목표각을 시뮬레이터의 soft 한계가 사후에 잡는다 — 창은 soft 한계 폭으로 읽는다.

**§1b-4. 이 런의 스택 플래그 (`PYG_*`)**

출처: 이 노트의 실행 명령/본문에서 추출 (런처 매니페스트 미기록 — 값 없는 항목은 노트가 이름만 언급)

| 플래그 | 값 |
|---|---|
| `PYG_ANKLE_MODE` | RP |
| `PYG_ARM_ABD_DEG` | 15 |
| `PYG_DR_END_ITER` | 1 |
| `PYG_DR_START_ITER` | 0 |
| `PYG_INERTIAL_DR` | 1 |
| `PYG_INIT_BENT` | 1 |
| `PYG_MOTOR_MEAS` | 1 |
| `PYG_SOFT_LANDING` | 1 |
| `PYG_TN` | 1 |
| `PYG_V2` | 1 |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

<!-- SPEC-TABLES:END -->

## §2 최종 지표 (평가기 32 ep × 3시나리오, 전진 1.6 m/s, DR off)
성공률 **96/96**. `eval_raw_stats.py` 중앙값(p10–p90):

| duty | stride/s | peak GRF | rate | 입각 무릎 | 전진 오차 |
|---|---|---|---|---|---|
| 0.532 | 2.667 | 1.214 BW | 15.93 BW/s | **45.9°** | 0.147 |

## §3 200 Hz 다중 env 충격 (24 env, DR off / DR on)
| | peak GRF | 하중률 | 스트라이크/s | 낙상 |
|---|---|---|---|---|
| DR off | 1.248 BW | **81.5 BW/s** | 2.27 | 0 |
| DR on | 1.253 BW | 79.9 BW/s | 2.28 | 0 |

⇒ 50 Hz 평가기(15.9)와 200 Hz(81.5)가 5배 차이나는 것은 [[2026-08-26_human_landing_bundle]] §11c의
에일리어싱 그대로다. **판정은 200 Hz 쪽을 쓴다.**

★기록해둘 관찰: AB 대조군(`bundleCTL_AB`)은 같은 조건에서 피크 **2.353 BW / 하중률 277 BW/s**에
DR 하 낙상 12회였는데, **RP 대조군은 1.248 BW / 81.5 BW/s / 낙상 0**이다. 초기자세가 같은데도
RP 쪽이 훨씬 부드럽게 딛는다 — 발목 기구가 다르면 같은 초기자세라도 착지 충격이 다르다는 뜻이고,
AB의 "굽힌 초기자세 = 하드랜딩"이 **AB 고유 현상**일 가능성을 남긴다. 단일 체크포인트 비교라 확정은 아니다.

## §4 판정
대조군으로 유효. 절대 성능도 정상(낙상 0, 성공률 100 %).

## §R 참조
[[2026-08-26_bundleD1_RP]] · [[2026-08-26_human_landing_bundle]] · [[103_v2_training_plan]] §4a
