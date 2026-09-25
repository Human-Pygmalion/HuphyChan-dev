# bundleTRT_AB — 인간형 착지 번들 시험 (처치군, 2026-08-26)

## §1 목적·설정
c3 완주 정책(AB, `model_31999`)에서 **+800 iter · 1024 env** warm-start로 세 변경을 한 번에 시험한다.
근거: [[2026-08-26_human_landing_bundle]] · 원인 분석 [[104_init_pose_gait_style]] · 상위 계획 [[103_v2_training_plan]].

| | 대조군 `bundleCTL_AB` | 처치군 `bundleTRT_AB` |
|---|---|---|
| 초기자세 | bent (knee −38.4°, hip −18.3°) | **PYG_INIT_MID** (knee −20.05°, hip −10.03°) |
| base_height 앵커 | 없음 | **PYG_BASE_HEIGHT_ANCHOR** (h_ref 0.87, deadband ±0.03, w −5.0) |
| soft-landing | 접지속도 제곱 (현행) | **PYG_SOFT_LANDING_MODE=rate** (접지 후 60 ms dF_z/dt 최대 제곱, w −0.002) |
| 그 외 | 전부 동일 | 전부 동일 |

공통: PYG_V2 · PYG_INIT_BENT · PYG_ARM_ABD_DEG=15 · PYG_INERTIAL_DR · PYG_TN · PYG_MOTOR_MEAS · PYG_ANKLE_MODE=AB,
DR은 이미 만배인 체크포인트를 잇는 것이라 `PYG_DR_START_ITER=0 / END=1`로 즉시 full.
런처: `tools/robot_model/loop_tests/run_landing_bundle_test.sh AB` · 시작 2026-08-26 11:02.

<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->

## 1b. 이 run의 Reward & Gains (config에서 파싱 — 재현용)

**Reward 항목** (weight·왜·어떻게 — '어떻게' 칸의 수치는 이 런의 `params/env.yaml` `rewards.<항>.params` 값):

| reward | weight | 왜 | 어떻게 |
|---|--:|---|---|
| base_height | **-5** | ★몸통 높이 앵커(까치발/무릎꿇기 방지) | relu(\|h−0.87 m\|−0.03 m)² |
| foot_clearance | **-2** | 스윙발 지면 이격(발끌림 방지) | Σ_발 \|h−0.1 m\|·‖v_발,xy‖; 명령>0.05 |
| track_angular_velocity | **+2** | 명령 회전속도 추종 | exp(-((ω_cmd,z−ω_z)²+‖ω_xy‖²)/σ²), σ=0.707 rad/s |
| track_linear_velocity | **+2** | 명령 전진/측방 속도 추종 | exp(-(‖v_cmd,xy−v_xy‖²+v_z²)/σ²), σ=0.866 m/s |
| air_time | **+1** | 체공시간 보상(질질끌기 억제) | 0.05<체공시간<0.5 s 인 발 수; 명령>0.5 |
| dof_pos_limits | **-1** | 관절범위 한계 접근 벌점 | Σ 소프트 한계 밖 초과량 [rad] |
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
| thermal_effort | -0.02 | ★열분배: Σ(τ/정격)² 정규화(관절 균등화) | Σ(τ/τ_정격)², τ_정격 [N·m]: hip_pitch 40·hip_roll 40·hip_yaw 20·knee 40·ankle_pitch 20·ankle_roll 5·crank_[AB] 20 |
| contact_force_cap | -0.01 | ★충격 상한: 발 지면반력 역치초과분 벌점(사뿐착지) | Σ_발 min(max(‖F‖−420 N, 0), 560 N) |
| foot_loading_rate | -0.002 | 착지 직후 하중 증가율 감소 | (착지 후 0.06 s 안 최대 relu(dF_z/dt)/체중)², 체중=347 N; 명령>0.05 |
| soft_landing | -1e-05 | 착지 첫접촉 충격 벌점(약) | Σ 첫 접촉 순간 ‖F‖ [N]; 명령>0.05 |
| torque_limit | -0 | 명령 토크의 한계초과 벌점 | Σ_구동기 relu(\|τ_명령\|/τ_max−0.7) |

† env.yaml에 값이 없어 함수 기본값으로 실행된 것 — 이 런이 실행한 코드(mjlab 커밋 a7f825f3)의 함수 정의에서 읽음.

**관절별 Kp/Kd** (position-PD, effort=관절측 peak):

| 관절 | 모터 | Kp(stiffness) | Kd(damping) | effort [N·m] |
|---|---|--:|--:|--:|
| hip_pitch | RS04 | 150 | 6 | 120 |
| hip_roll | RS04 | 150 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| knee | RS04 | 220 | 6 | 120 |
| crank_A | RS03 | 22.3 | 1.41 | 60 |
| crank_B | RS03 | 22.3 | 1.41 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-08-26_11-03-17_bundleTRT_AB`)

| 관절 그룹 | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] | effort 한계 [N·m] | 무부하 속도 [rad/s] | 로터 관성 armature [kg·m²] | 쿨롱 마찰 [N·m] | 점성 [N·m·s/rad] | T-N 곡선 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| crank_A, crank_B | RS03 | 22.266 | 1.414 | 60 (stall 59.7) | 20.94 | 0.01527 | 0.285 | 0.0223 | 실측 37점 (`PYG_TN=1`) |
| hip_yaw | RS03 | 150 | 6 | 60 (stall 59.7) | 20.94 | 0.01527 | 0.285 | 0.0223 | 실측 37점 (`PYG_TN=1`) |
| hip_pitch | RS04 | 150 | 6 | 120 (stall 120.1) | 20.94 | 0.01633 | 0.269 | 0.0095 | 실측 22점 (`PYG_TN=1`) |
| knee | RS04 | 220 | 6 | 120 (stall 120.1) | 20.94 | 0.01633 | 0.269 | 0.0095 | 실측 22점 (`PYG_TN=1`) |
| ankle_pitch / ankle_roll | (수동, 크랭크가 구동) | — | — | — | — | — | — | — | 폐루프 `equality/connect` 등식구속 |

토크는 `effort_limit`과 (있으면) 실측 T-N 곡선의 속도의존 상한 중 **작은 값**으로 클램프된다. armature/쿨롱/점성은 모터 실측값(`PYG_MOTOR_MEAS=1`)이면 실측, 아니면 카탈로그 추정치다.

**§1b-3. ROM 한계·액션 창** (모델 XML range · soft 한계 = 중심±0.5·range×0.9 (mjlab `Entity` 규약) · 액션 clip = env.yaml `actions.joint_pos.clip` · 창 = clip 폭 · default = 액션 0 자세)

**soft 한계와 액션 clip은 같은 공식이다** — `Entity.soft_joint_pos_limits`와 `pygmalion_constants.safe_target_clip()`이 둘 다 *중심 ± 0.5·range·factor*를 쓴다(각 경계에 factor를 곱하는 것이 아니다: 비대칭 관절에서 두 식이 갈린다 — knee `[0,120]`은 `[6,114]`이지 `[0,108]`이 아니다). 그래서 `PYG_SAFE_TARGET_CLIP=1`인 런에서는 두 열이 정확히 일치하고, 정책이 통과하는 클램프와 시뮬레이터가 강제하는 클램프가 하나의 계약이 된다.

모델 출처: v3 printed 계열 — 노트 [[2026-08-23_ankleAB_c2]]/[[2026-08-23_ankleRP_c2]]가 선언한 모델이며, v4 XML은 2026-08-26_21-22에야 생성되어 이 런이 로드할 수 없었다(객관 상한) — `pygmalion_v3_printed_loop.xml`

> ⚠ **파일 최신성 주의** — 이 XML은 런 시작(2026-08-26_11-03-17)보다 뒤인 2026-08-26 20:36에 수정되었다. 런 디렉토리에 모델 스냅샷이 없으므로 아래 range는 **현재 파일 기준**이며 런 당시와 다를 수 있다.

| 관절 | XML range [°] | soft 한계 [°] | 액션 clip [°] | 사용가능 창 [°] | default [°] | 구동 |
|---|---|---|---|--:|--:|---|
| L/R_hip_pitch_joint | [-120, 25] | [-112.8, 17.7] | n/a (구 설정: clip 없음) | 130.5 | -10.03 | 액션 |
| L/R_hip_roll_joint | [-85, 25] | [-79.5, 19.5] | n/a (구 설정: clip 없음) | 99 | 0 | 액션 |
| L/R_hip_yaw_joint | [-45, 45] | [-40.5, 40.5] | n/a (구 설정: clip 없음) | 81 | 0 | 액션 |
| L/R_knee_joint | [-120, 0] | [-114, -6] | n/a (구 설정: clip 없음) | 108 | -20.05 | 액션 |
| L_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | n/a (구 설정: clip 없음) | 123.8 | -17.12 | 액션 |
| L_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | n/a (구 설정: clip 없음) | 123.8 | -17.12 | 액션 |
| L_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.6 | 수동 |
| L_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0.15 | 수동 |
| R_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | n/a (구 설정: clip 없음) | 123.8 | -17.14 | 액션 |
| R_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | n/a (구 설정: clip 없음) | 123.8 | -17.14 | 액션 |
| R_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.63 | 수동 |
| R_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0.15 | 수동 |
| waist_yaw_joint | [-60, 60] | [-54, 54] | — (수동) | — | 0 | 수동 |
| L/R_shoulder_pitch_joint | [-180, 60] | [-168, 48] | — (수동) | — | 0 | 수동 |
| L/R_shoulder_roll_joint | [-32, 30] | [-28.9, 26.9] | — (수동) | — | 0 | 수동 |

액션 스케일 0.25 rad/단위, 오프셋 = default (`use_default_offset`). clip이 없는 구 설정에서는 정책 목표각을 시뮬레이터의 soft 한계가 사후에 잡는다 — 창은 soft 한계 폭으로 읽는다.

**§1b-4. 이 런의 스택 플래그 (`PYG_*`)**

출처: 이 노트의 실행 명령/본문에서 추출 (런처 매니페스트 미기록 — 값 없는 항목은 노트가 이름만 언급)

| 플래그 | 값 |
|---|---|
| `PYG_ANKLE_MODE` | AB |
| `PYG_ARM_ABD_DEG` | 15 |
| `PYG_BASE_HEIGHT_ANCHOR` | (값 미기재) |
| `PYG_DR_START_ITER` | 0 |
| `PYG_INERTIAL_DR` | (값 미기재) |
| `PYG_INIT_BENT` | (값 미기재) |
| `PYG_INIT_MID` | (값 미기재) |
| `PYG_MOTOR_MEAS` | (값 미기재) |
| `PYG_SOFT_LANDING_MODE` | rate |
| `PYG_TN` | (값 미기재) |
| `PYG_V2` | (값 미기재) |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

<!-- SPEC-TABLES:END -->

## §2 학습 중 리뷰
| 시각 | iter | reward | ep_len | noise σ | fell | low_base | err_vel | 판정 |
|---|---|---|---|---|---|---|---|---|
| (게이트마다 추가) | | | | | | | | |

## §3 판정 결과 (2026-08-26 11:45, model_32798)
측정: `impact_probe`(200 Hz) + 내장 평가기(1.6 m/s 3시나리오 × 32 ep) + `apex_recheck`(1.2 m/s 보행 롤아웃).

| 지표 | c3 (변경 전) | **CTL** | **TRT** | 목표 | TRT 판정 |
|---|---|---|---|---|---|
| 접지 무릎 | −30.5° | −47.9° | **−42.9°** | ≤ −15° | ✗ (단 CTL 대비 +5°) |
| 중간입각 무릎 | −29.2° | −40.3° | −36.6° | −20° | ✗ |
| 스윙 최대굴곡 | −58.3° | −63.9° | −69.6° | −55~−65 유지 | △ 초과 |
| **접지속도 중앙** | 0.97 | **0.873** | **1.589** | ↓ | **✗ 82 % 악화** |
| **하중률 중앙 [BW/s]** | 88 | **46.2** | **84.5** | ↓ | **✗ 83 % 악화** |
| GRF 피크 중앙 [BW] | 1.16 | 0.909 | 1.180 | ↓ | ✗ |
| **stride/s** | ~1.05 | 1.40 | **0.93** | 유지 | **⚠ −34 % = 해킹 징후** |
| duty | 0.55 | 0.57 | 0.57 | 유지 | ✓ |
| 추종 전진 1.6 | 0.143(c3) | 0.161 | **0.143** | 열화 <10 % | ✓ |
| 성공률(96 ep) | 100 % | **100 %** | **100 %** | 낙상 0 | ✓ |

## §4 결과 — **기각**
1. **하중률 벌점이 하중률을 낮추지 못했다**(84.5 vs 대조군 46.2). 목표 지표가 오히려 83 % 나빠졌다.
2. **예측했던 해킹 경로가 실제로 나타났다**: duty는 같은데 **stride/s가 1.40 → 0.93 (−34 %)**. 걸음 수를 줄이면
   접지 횟수가 줄어 하중률 노출이 줄어든다 — 연구 노트 §3c에서 "발을 안 떼면 dF/dt = 0"으로 경고한 그 경로다.
   (`air_time`은 duty가 같아 겉보기엔 정상이라, **stride/s가 없었으면 놓쳤을 것**이다.)
3. **접지속도 항을 제거한 대가**가 컸다: 0.873 → 1.589 m/s. `MODE=rate`가 이 항을 완전히 대체하도록 한 것이 실수다.
4. **초기자세 변경은 유효했다**: 접지 무릎이 대조군 대비 **+5°**(−47.9 → −42.9) 개선. 다만 목표(−15°)에는 크게 못 미친다.
5. 추종·낙상은 두 arm 모두 정상(성공률 100 %, 전진 오차 0.143/0.161).

### ⚠ 교란 — 1024 env·+800 iter 자체가 보행을 흔든다
**대조군도 c3보다 크게 나빠졌다**(접지 무릎 −30.5 → −47.9°, stride 1.05 → 1.40). 1024 env는 학습 배치가 본런(16384)의
**1/16**이라 짧은 warm-start만으로도 정책이 드리프트한다. ⇒ **이 시험의 판정은 TRT vs CTL 비교로만 유효**하고,
c3와의 절대 비교는 쓸 수 없다. 다음 세대 시험은 **배치를 4096 이상**으로 올리거나 iter를 늘려야 한다.

### 후속 (즉시 착수, 2026-08-26 11:45)
| 런 | 변경 | 묻는 것 |
|---|---|---|
| `bundleB2_AB` | INIT_MID + 앵커 + **MODE=both**(접지속도 절반 가중 유지 + 하중률 병행) | 해로운 부분(접지속도 제거)만 되돌리면 회복되나 |
| `bundleB3_AB` | **INIT_MID 단독** | 초기자세 변경의 순수 효과 |

## §정정 (2026-08-26 오후)
여기서 내린 "착지 번들 1차 = 기각"은 **단일 롤아웃 근거**라 무효다. 평가기 32 ep 재측정에서
**`MODE=both`(병행형)는 전 지표 최고로 채택**되었고, 기각은 **`MODE=rate`(대체형)에만** 남는다.
TRT가 나빴던 원인은 하중률 항 자체가 아니라 **검증된 접지속도 항을 빼앗은 것**이다(TRT 하중률 15.78 BW/s = 전 arm 최악).
상세 [[2026-08-26_bundleB2_AB]] §5 · 확정 레시피 [[103_v2_training_plan]] §4a
