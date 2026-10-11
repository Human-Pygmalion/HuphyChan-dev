# rough_turn_ent002_yaw15_v11a_c3k — 시험 2 를 3,000 회까지 이어 학습 (2026-10-11)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> 이 노트는 한 런의 정식 기록이다. **1,000~3,000 회짜리 짧은 진단 시험**이라 하중 측정·영상·모터 활용(§5·§7·§8·§10)은 하지 않았다. 시험 전체의 서사·해석은 조사 노트 [[../reward_research/2026-10-11_turn_in_place_recovery_literature]] §7·§7-1~§7-4, 계획은 [[../162_next_plan_rough_policy]] 단계 4.
> 부모: [[2026-10-11_rough_turn_ent002_yaw15_v11a]] (시험 2, 저장본 `model_999.pt` 에서 **전체 이어받기**). 시험 1 과 비교: [[2026-10-11_rough_turn_ent002_v11a]].
> 계보: 평지 v11a → 험지 B → 시험 1 → 시험 2 → **이 런**.

## §0 한눈에

| 항목 | 값 |
|---|---|
| 한 줄 | 시험 2(엔트로피 0.002 + 순수 회전 15 %)를 1,000 회 저장본에서 **전체 체크포인트 이어받기(full resume)로** 1,000 → 3,000 회(2,000 회 더) 학습 |
| 결과 | 회전은 계속 돌아온 채(중앙값 4 씨앗 64~93 %) 유지. 그러나 **서기(발 뗀 28.3 %·흘러감 3.3 m)와 직진(방향 틀어짐 52°, 걷기 속도 0.90)은 풀리지 않았다**. 학습 중 물리 폭주 종료는 0.063 → 0.005 로 12 분의 1 |
| 결론 | 더 학습해도 "회전은 되나 가만히·똑바로가 무너지는" 맞바꿈은 남는다 — 1,000 회가 덜 수렴해서만이 아니다 |
| 학습 시간 | 2,000 회, 경과 약 110 분 (6,627 초) |

## §1 재현성

| 항목 | 값 |
|---|---|
| 런 폴더 | `mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/2026-10-11_15-55-47_rough_turn_ent002_yaw15_v11a_c3k` (원격 3090 의 `/home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/` 아래). 이 PC 에는 `params/`·`repro/` 만 복사했다 — 체크포인트와 텐서보드 기록은 3090 에만 있다 |
| 과제 | `Mjlab-Velocity-Rough-Pygmalion` (험지: 8 m × 8 m 타일 10 행 × 20 열, 7 종 — 부모 B 노트 §1 과 같음) |
| 시작점 | 시험 2 의 `model_999.pt` 에서 **전체 이어받기**(`repro/launch_manifest.json` 의 `parent.load` = `full resume`; 시험 1~4 처럼 배우만 가져온 것이 아님). 반복 번호는 999 에서 이어짐 — 텐서보드 기록의 단계가 999~2998 이다 |
| 신경망 | 배우 입력 45 → 512 → 256 → 128 → 출력 12, 비평가 입력 269 → 512 → 256 → 128 → 1 (환경 설정 `params/env.yaml` 이 부모 B 와 한 줄도 다르지 않음) |
| 규모 | 환경 4,096개, 2,000 회 더 (반복 1,000 → 3,000, 합계 3,000 회), 씨앗 43, 기록 방식 tensorboard, 제어 50 Hz, 에피소드 20 초 |
| 지형 단계·명령 상자 | 첫 회에 체류 한도 0 회 규칙(`PYG_GATE_MIN_DWELL=0`·`PYG_GATE_MAX_DWELL=0`)으로 단계 4 로 강제 승급되어 끝까지 단계 4 (선속도 x −2.0 ~ +2.5, y ±1.0, 각속도 ±1.0). 부모 B 와 같다 — 이 런에서 명령 상자는 학습 변수가 아니다 |
| 무작위화 | 계수 `Curriculum/dr_levels/dr_factor` 가 내내 0.0 — 꺼진 명목 조건 (부모 B 와 같음) |
| 발사 | `mjlab.scripts.train` 직접 실행 (시험 2 의 발사 스크립트가 아니라 일반 학습 진입점으로 이어받음) (재현 기록에 복사됨) |
| 재현 기록 | 런 폴더 `repro/launch_manifest.json`(명령·`PYG_*` 전부·부모·입력 파일 해시), `params/env.yaml`, `params/agent.yaml` |
| 저장된 설정 파일의 한계 | `params/agent.yaml` 의 `entropy_coef` 는 **0.01 로 찍혀 있다**(시험 1·2·3·c3k 는 실제로 0.002, 시험 4 는 0.005). 엔트로피 계수는 `PYG_ENTROPY_*` 환경 변수를 학습 실행기(`mujoco-sim/mjlab/src/mjlab/rl/runner.py`)가 읽어 덮어쓰므로 설정 파일에 반영되지 않는다. **정본은 `repro/launch_manifest.json` 의 `PYG_ENTROPY_START_COEF`·`PYG_ENTROPY_END_COEF`·`PYG_ENTROPY_ANNEAL_START/END`** (아래 §1b-4 표) |
| 발사 명령 | `.venv/bin/python3 -m mjlab.scripts.train Mjlab-Velocity-Rough-Pygmalion --env.scene.num-envs 4096 --agent.max-iterations 2000 ... --agent.resume True --agent.load-run 2026-10-11_14-51-46_rough_turn_ent002_yaw15_v11a --agent.load-checkpoint model_999.pt` (`repro/launch_manifest.json`) |

<!-- 아래 마커 사이의 §1b ~ §1b-4 는 analysis/run_spec_tables.py 가 생성한다 -->

<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->

## 1b. 이 run의 Reward & Gains (config에서 파싱 — 재현용)

**Reward 항목** (weight·왜·어떻게 — '어떻게' 칸의 수치는 이 런의 `params/env.yaml` `rewards.<항>.params` 값):

| reward | weight | 왜 | 어떻게 |
|---|--:|---|---|
| foot_clearance | **-2** | 스윙발 지면 이격(발끌림 방지) | Σ_발 \|h−0.1 m\|·‖v_발,xy‖; 명령>0.05 |
| stance_knee_extension | **-2** | 입각 중 과도한 웅크림 억제 | Σ_접지발(힘>? N) relu(\|q_무릎\|−25°)² [rad²]; 명령>0.05 |
| track_angular_velocity | **+2** | 명령 회전속도 추종 | exp(-((ω_cmd,z−ω_z)²+‖ω_xy‖²)/σ²), σ=0.707 rad/s |
| track_linear_velocity | **+2** | 명령 전진/측방 속도 추종 | exp(-(‖v_cmd,xy−v_xy‖²+v_z²)/σ²), σ=0.866 m/s |
| air_time | **+1** | 체공시간 보상(질질끌기 억제) | 0.05<체공시간<0.5 s 인 발 수; 명령>0.5 |
| dof_pos_limits | **-1** | 관절범위 한계 접근 벌점 | Σ 소프트 한계 밖 초과량 [rad] |
| pose | **+1** | 기본 관절자세 정규화(기괴자세 억제) | exp(-mean((q−q₀)²/σ²)) [rad]; σ 서기(명령<0.05): 전 관절 0.05 / 걷기: hip_pitch 0.3·hip_roll 0.4·hip_yaw 0.15·knee 1.2·ankle_pitch 0.25·ankle_roll 0.1 / 달리기(명령≥1.5): hip_pitch 0.5·hip_roll 0.2·hip_yaw 0.2·knee 1.5·ankle_pitch 0.35·ankle_roll 0.15 |
| self_collisions | **-1** | 자기충돌 벌점 | 자기충돌 힘>10 N 인 부분스텝 수 |
| stand_still_penalty | **-1** | 이동 명령을 무시하고 서 버리는 정지 방지 | \|v_cmd\|>? 또는 \|ω_cmd\|>? 인데 명령방향 진행<?×명령이면 1 |
| track_lin_vel_progress | **+1** | 고속 명령에서 정지하는 국소 최적해 방지 | clamp(min(v·ĉ, \|v_cmd\|), 0) [m/s] (명령방향 진행 속도, 명령 크기에서 상한) |
| upright | **+1** | 몸통 직립 유지(넘어짐 방지) | exp(-‖g_xy‖²/σ²) (몸통 좌표계로 투영한 중력), σ=0.447 |
| foot_impact_velocity | **-0.5** | 착지 직전 발 하강속도 감소 | Σ_발 relu(−v_z)^2 (높이<0.1 m, 힘<? N); 명령>0.05 |
| knee_overspeed | **-0.5** | 실측 RS04 무부하 속도를 넘는 보행 억제 | Σ_무릎 relu(\|q̇\|−19.9 rad/s)² |
| foot_swing_height | -0.25 | 스윙발 높이 성형 | 착지 순간 Σ(최고높이/0.1 m−1)²; 명령>0.05 |
| action_rate_l2 | -0.1 | 액션 급변 벌점(진동/저크 억제) | ‖a_t−a_(t−1)‖² |
| foot_slip | -0.1 | 접지발 미끄러짐 벌점 | Σ_접지발 ‖v_발,xy‖²; 명령>0.05 |
| body_ang_vel | -0.05 | 몸통 각속도 벌점(흔들림 억제) | ‖ω_xy‖² (몸통 롤·피치 각속도) |
| angular_momentum | -0.02 | 전신 각운동량 벌점(회전 낭비 억제) | ‖L‖² (전신 각운동량) |
| thermal_effort | -0.02 | ★열분배: Σ(τ/정격)² 정규화(관절 균등화) | Σ(τ/τ_정격)², τ_정격 [N·m]: hip_pitch 40·hip_roll 40·hip_yaw 20·knee 40·ankle_pitch 20·ankle_roll 5·waist_yaw 40·shoulder_(pitch\|roll) 20·crank_[AB] 20 |
| contact_force_cap | -0.01 | ★충격 상한: 발 지면반력 역치초과분 벌점(사뿐착지) | Σ_발 min(max(‖F‖−420 N, 0), 560 N) |
| soft_landing | -1e-05 | 착지 첫접촉 충격 벌점(약) | Σ 첫 접촉 순간 ‖F‖ [N]; 명령>0.05 |
| torque_limit | -0 | 명령 토크의 한계초과 벌점 | Σ_구동기 relu(\|τ_명령\|/τ_max−0.7) |

? env.yaml에도 기록된 소스에도 값이 없어 확인 불가.

**관절별 Kp/Kd** (position-PD, effort=관절측 peak):

| 관절 | 모터 | Kp(stiffness) | Kd(damping) | effort [N·m] |
|---|---|--:|--:|--:|
| hip_pitch | RS04 | 150 | 6 | 120 |
| hip_roll | RS04 | 150 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| knee | RS04 | 220 | 6 | 120 |
| crank_A | RS03 | 22.3 | 1.41 | 60 |
| crank_B | RS03 | 22.3 | 1.41 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-10-11_15-55-47_rough_turn_ent002_yaw15_v11a_c3k`)

| 관절 그룹 | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] | effort 한계 [N·m] | 무부하 속도 [rad/s] | 로터 관성 armature [kg·m²] | 쿨롱 마찰 [N·m] | 점성 [N·m·s/rad] | T-N 곡선 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| crank_A, crank_B | RS03 | 22.266 | 1.414 | 60 (stall 59.7) | 20.94 | 0.0638 | 0.251 | 0.0252 | 실측 37점 (`PYG_TN=1`) |
| hip_yaw | RS03 | 150 | 6 | 60 (stall 59.7) | 20.94 | 0.0638 | 0.251 | 0.0252 | 실측 37점 (`PYG_TN=1`) |
| hip_pitch | RS04 | 150 | 6 | 120 (stall 120.1) | 20.94 | 0.0773 | 0.519 | 0.0275 | 실측 22점 (`PYG_TN=1`) |
| knee | RS04 | 220 | 6 | 120 (stall 120.1) | 20.94 | 0.0773 | 0.519 | 0.0275 | 실측 22점 (`PYG_TN=1`) |
| ankle_pitch / ankle_roll | (수동, 크랭크가 구동) | — | — | — | — | — | — | — | 폐루프 `equality/connect` 등식구속 |

토크는 `effort_limit`과 (있으면) 실측 T-N 곡선의 속도의존 상한 중 **작은 값**으로 클램프된다. armature/쿨롱/점성은 모터 실측값(`PYG_MOTOR_MEAS=1`)이면 실측, 아니면 카탈로그 추정치다.

**§1b-3. ROM 한계·액션 창** (모델 XML range · soft 한계 = 중심±0.5·range×0.9 (mjlab `Entity` 규약) · 액션 clip = env.yaml `actions.joint_pos.clip` · 창 = clip 폭 · default = 액션 0 자세)

**soft 한계와 액션 clip은 같은 공식이다** — `Entity.soft_joint_pos_limits`와 `pygmalion_constants.safe_target_clip()`이 둘 다 *중심 ± 0.5·range·factor*를 쓴다(각 경계에 factor를 곱하는 것이 아니다: 비대칭 관절에서 두 식이 갈린다 — knee `[0,120]`은 `[6,114]`이지 `[0,108]`이 아니다). 그래서 `PYG_SAFE_TARGET_CLIP=1`인 런에서는 두 열이 정확히 일치하고, 정책이 통과하는 클램프와 시뮬레이터가 강제하는 클램프가 하나의 계약이 된다.

모델 출처: launch_manifest `PYG_MODEL_TAG` — `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml`

| 관절 | XML range [°] | soft 한계 [°] | 액션 clip [°] | 사용가능 창 [°] | default [°] | 구동 |
|---|---|---|---|--:|--:|---|
| R_hip_pitch_joint | [-120, 25] | [-112.8, 17.7] | [-112.8, 17.7] | 130.5 | -10.03 | 액션 |
| R_hip_roll_joint | [-25, 85] | [-19.5, 79.5] | [-19.5, 79.5] | 99 | 0 | 액션 |
| R_hip_yaw_joint | [-45, 45] | [-40.5, 40.5] | [-40.5, 40.5] | 81 | 0 | 액션 |
| R_knee_joint | [0, 120] | [6, 114] | [6, 114] | 108 | 20.05 | 액션 |
| R_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | 8.25 | 액션 |
| R_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -8.43 | 액션 |
| R_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 10.04 | 수동 |
| R_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0 | 수동 |
| L_hip_pitch_joint | [-25, 120] | [-17.7, 112.8] | [-17.7, 112.8] | 130.5 | 10.03 | 액션 |
| L_hip_roll_joint | [-85, 25] | [-79.5, 19.5] | [-79.5, 19.5] | 99 | 0 | 액션 |
| L/R_hip_yaw_joint | [-45, 45] | [-40.5, 40.5] | [-40.5, 40.5] | 81 | 0 | 액션 |
| L_knee_joint | [-120, 0] | [-114, -6] | [-114, -6] | 108 | -20.05 | 액션 |
| L_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -8.24 | 액션 |
| L_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | 8.42 | 액션 |
| L_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 10.03 | 수동 |
| L_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0 | 수동 |

액션 스케일 0.25 rad/단위, 오프셋 = default (`use_default_offset`). clip이 없는 구 설정에서는 정책 목표각을 시뮬레이터의 soft 한계가 사후에 잡는다 — 창은 soft 한계 폭으로 읽는다.

**§1b-4. 이 런의 스택 플래그 (`PYG_*`)**

출처: 런 디렉토리 `repro/launch_manifest.json` (권위)

| 플래그 | 값 |
|---|---|
| `PYG_ANKLE_MODE` | AB |
| `PYG_ARM_ABD_DEG` | 15 |
| `PYG_BLOWUP_TERM` | 1 |
| `PYG_CMD_VY_STAGES` | 1 |
| `PYG_CRITIC_DR_OBS` | 1 |
| `PYG_DR_END_ITER` | 25999 |
| `PYG_DR_START_ITER` | 15999 |
| `PYG_ENTROPY_ANNEAL` | 1 |
| `PYG_ENTROPY_ANNEAL_END` | 0 |
| `PYG_ENTROPY_ANNEAL_START` | 0 |
| `PYG_ENTROPY_END_COEF` | 0.002 |
| `PYG_ENTROPY_START_COEF` | 0.002 |
| `PYG_FRESH_STEPS` | 1 |
| `PYG_GATED_CURRICULUM` | 1 |
| `PYG_GATE_ERR_RATIO` | 1.1 |
| `PYG_GATE_FELL_MAX` | 0.005 |
| `PYG_GATE_MAX_DWELL` | 0 |
| `PYG_GATE_MIN_DWELL` | 0 |
| `PYG_GATE_MIN_EPISODES` | 64 |
| `PYG_GATE_WINDOW` | 100 |
| `PYG_INERTIAL_DR` | 1 |
| `PYG_INIT_BENT` | 1 |
| `PYG_INIT_MID` | 1 |
| `PYG_KNEE_EXT` | 1 |
| `PYG_KNEE_EXT_DEG` | 25 |
| `PYG_KNEE_EXT_W` | 2.0 |
| `PYG_LOW_BASE_REL` | 1 |
| `PYG_MASS_DR_JSON` | `mass_dr_legonly_fastener50_prototype-tempmass.json` |
| `PYG_MODEL_TAG` | LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix |
| `PYG_MOTOR_MEAS` | 1 |
| `PYG_NAN_TERM` | 1 |
| `PYG_OBS_NAN_GUARD` | 1 |
| `PYG_REL_HEADING` | 0.3 |
| `PYG_REL_STANDING` | 0.1 |
| `PYG_REL_YAW_ONLY` | 0.15 |
| `PYG_REWARD_TERM_CLAMP` | 50 |
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_SOFT_LANDING` | 1 |
| `PYG_SOFT_LANDING_MODE` | half |
| `PYG_STAGNATION` | 1 |
| `PYG_STUDENT_TEACHER` | 1 |
| `PYG_TN` | 1 |
| `PYG_V2` | 1 |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

<!-- SPEC-TABLES:END -->

## §2 학습 곡선 숫자

값은 텐서보드 `Policy/mean_std`·`Episode_Reward/...`·`Train/mean_episode_length`·`Curriculum/terrain_levels/mean`·`Episode_Termination/...` 의 **마지막 50 회 단순 평균**이다(보상 기록값은 부모 노트 [[2026-10-10_rough_stag_ab_b_v11a]] §2b 와 같은 기록 항목). 종료 항목의 "회당"은 한 회(환경 4,096개 × 24 걸음) 동안 그 이유로 끝난 에피소드 수에 해당하는 기록값이다. **조사 노트 §7-1 의 숫자(예: 시험 1 회전 몫 0.95, 부모 B 지형 단계 5.03·에피소드 606)는 마지막 한 시점·다른 창으로 읽은 값이라 이 표와 조금 다르다** — 창이 다를 뿐 같은 런이다. 이 런은 반복 999~2998 이므로 "마지막 50 회" 는 반복 2949~2998 이다. 이어받은 첫 기록(반복 999)은 에피소드 길이 20, 평균 보상 −0.5 로 비정상적으로 낮다(환경을 새로 시작한 영향으로 보이나 확인하지 않음) — 마지막 50 회 평균이라 아래 표에는 영향이 없다.

| 구간 (마지막 50 회 평균) | 행동 잡음 표준편차 (`Policy/mean_std`) | 각속도 추종 보상 | 선속도 추종 보상 | 회전 몫 (각속도 / 선속도) | 에피소드 길이 (최대 1,000) | 험지 단계 평균 | 물리 폭주 종료 (회당) | 정체 종료 (회당) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **시험 2 이어 학습 (이 런), 반복 2949~2998** | 0.174 | 0.619 | 0.740 | 0.84 | 584 | 4.60 | 0.0050 | 1.093 |
| 부모 시험 2, 950~999 회 (이어받은 저장본 시점) | 0.152 | 0.338 | 0.391 | 0.87 | 326 | 3.15 | 0.0000 | 1.438 |
| 시험 1, 950~999 회 | 0.163 | 0.437 | 0.478 | 0.91 | 413 | 4.04 | 0.0017 | 1.204 |
| 부모 B, 3,000 회 끝 (마지막 50 회) | 0.614 | 0.052 | 0.825 | 0.06 | 623 | 5.39 | 0.0634 | 0.670 |

학습 건강도 (이 런, 마지막 50 회 평균):

| 평균 에피소드 보상 | 넘어짐 종료 (회당) | 몸통 낮음 종료 (회당) | 시간 만료 (회당) | 가치 손실 | 학습률 (적응형) | 무작위화 계수 |
|--:|--:|--:|--:|--:|--:|--:|
| 45.1 | 0.0585 | 3.10 | 2.63 | 0.432 | 0.00012 | 0.0 |

- 행동 잡음 표준편차는 0.152(이어받은 시점, 950~999 회 평균) → 0.174 로 약간만 늘었다. 엔트로피 계수 0.002 고정이 3,000 회까지 잡음을 묶었다 (부모 B 는 0.61).
- 각속도·선속도 추종 보상이 함께 올랐다(0.338 → 0.619, 0.391 → 0.740): 회전 몫 0.84 로 회전을 버리지 않은 채 선속도 추종도 부모 B 수준(0.825)에 근접(0.740).
- 지형 단계 평균 4.60 (부모 B 3,000 회 끝 5.39), 에피소드 길이 584 (623). 평균 보상 45.1.
- **물리 폭주 종료 0.005 (부모 B 3,000 회 끝 0.063)** — 잡음을 묶으면 12 분의 1 이다 (조사 노트 §7-3; 계획 단계 5 의 근거).

## §3b 영상

해당 없음 — 짧은 진단 시험(하중 측정은 본 학습에서)

## §3c 측정 결과 (조사 노트 [[../reward_research/2026-10-11_turn_in_place_recovery_literature]] §7-3 에서 옮김)

측정은 조사 노트 §7-2·§7-4 의 같은 도구(`analysis/matched_set.sh`)·같은 씨앗: 회전 4 씨앗, 서기 4 씨앗(27 초), 걷기 3 씨앗(앞으로 1.2 m/s 명령). **숫자는 새로 재지 않고 조사 노트에서 옮겼다.** 이 노트는 하중 측정(`measure_full` 깨끗·밀기)을 하지 않으므로 §3c 는 이 진단 측정이 전부다. 아래 부모 B 열은 **3,000 회** 저장본이다 (조사 노트 §7-3 이 3,000 회끼리 견줌).

사전 기준(조사 노트 §7-4 표 마지막 열): 회전 중앙값 ≥ 50 %, 회전 반대 방향 0, 서기 발 뗀 ≤ 2 %, 서기 27 초 흘러감 ≤ 0.3 m, 걷기 속도 1.14~1.26 m/s, 엉덩이 비틀림 좌우 차 ≤ 25 %.

| 지표 | 험지 B (3,000 회) | **시험 2 이어 학습 (3,000 회, 이 런)** | 사전 기준 |
|---|---|---|---|
| 제자리 회전 중앙값 (4 씨앗) | 1.0 % (1 회 측정) | **92.7 / 64.2 / 92.7 / 83.7 %** | ≥ 50 % |
| 회전 반대 방향 (씨앗별 횟수) | — | 0·2·1·1 | 0 |
| 서기 발 뗀 시간 | 0.1 % | **28.3 %** | ≤ 2 % |
| 서기 27 초 흘러감 | 0.14 m | **3.3 m** | ≤ 0.3 m |
| 걷기 실제 속도 (1.2 m/s 명령) | 1.43 | **0.90** | 1.14~1.26 |
| 엉덩이 비틀림 좌우 차 | −37 % | −6.3 % | ≤ 25 % |
| 걷기 방향 틀어짐 | 5.1° | **52°** | — |
| 학습 중 물리 폭주 종료 (마지막 50 회 평균) | 0.063 / 회 | **0.005 / 회** | — |

## §4 부모와 비교

부모 = 시험 2 [[2026-10-11_rough_turn_ent002_yaw15_v11a]] (`2026-10-11_14-51-46_rough_turn_ent002_yaw15_v11a`). **단일 변인 = 학습 길이 (1,000 → 3,000 회, 전체 이어받기).** 설정은 같다.

| 비교 | 결과 |
|---|---|
| `params/env.yaml` | 시험 2 와 **차이 없음** |
| `params/agent.yaml` | 차이: `max_iterations` 1000 → 2000, `run_name`, `load_run`(평지 v11a → 시험 2)·`load_checkpoint`(`model_29998.pt` → `model_999.pt`). `resume: true` |
| `repro/launch_manifest.json` | 차이: `parent.run` = 시험 2, `parent.checkpoint` = `model_999.pt`, `parent.load` = `actor only` → `full resume`, 진입점 `analysis/train_actor_warmstart.py` → `mjlab.scripts.train`, 최대 반복 2000. `PYG_*` 환경 변수는 **시험 2 와 완전히 같다**(엔트로피 0.002 고정·`PYG_REL_YAW_ONLY=0.15`) |

정량 비교(1,000 회 → 3,000 회): 회전 중앙값 4 씨앗 48~80 % → 64~93 %, 서기 발 뗀 15.9 → 28.3 %, 흘러감 8.0 → 3.3 m, 걷기 방향 틀어짐 145° → 52°, 걷기 속도 1.03 → 0.90 (조사 노트 §7-2·§7-3 의 두 측정을 나란히 놓음).

## §5 부하 분석 · §7 모터 활용 · §8 설계·부하 선도 · §8b · §8c · §10 링크 wrench · §10b · §11

해당 없음 — 짧은 진단 시험(하중 측정은 본 학습에서)

## §9 무작위화

꺼짐(계수 0.0, 3,000 회 내내). 밀기 시험 없음.

## §12 판정

- 3,000 회까지 학습해도 회전(64~93 %)은 유지되지만 **서기 발 뗀 28.3 %·흘러감 3.3 m, 방향 틀어짐 52°, 걷기 속도 0.90 은 사전 기준에 못 미친다.** 맞바꿈은 학습이 짧아서가 아니라 남는 문제다.
- 서기 흘러감의 방향이 시험 1 은 뒤로(−0.17 m/s), 시험 2 는 옆으로여서, 회전만이 아니라 **영 명령을 지키는 능력 전체**가 약해졌다는 해석이다 (조사 노트 §7-3).
- 잡음을 묶으면 물리 폭주 종료가 12 분의 1 — 엔트로피 고정의 부수 이득으로 본 학습 묶음(계획 단계 5·6)에 반영되었다.
- 계보 안의 위치: 시험 2 의 연장. 이 계열(순수 회전 15 %)은 본 학습 권고 묶음에서 빠졌다.
- 정리하지 못한 점: 부모 B 3,000 회의 회전 중앙값은 1 회 측정(씨앗 반복 아님)이라 이 열의 비교폭은 좁다; 하중 미측정; 이어받은 가치 함수가 서기를 회복시키지 못한 이유는 확인하지 않았다(가치 함수 예열은 시험 3 이 별도로 시험).
