# rough_box_trial_v11a — 배포 정책 설정에 권장 묶음을 한꺼번에 적용한 시험 (2026-10-14)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> **이것은 묶음 시험이다. 여러 변경을 한꺼번에 넣었으므로 단일 변인 실험이 아니다.** 어느 변경이 효과를 냈는지는 이 런만으로 가를 수 없다. 이 런이 묻는 것은 "권장 묶음을 모두 넣어도 학습이 무너지지 않고, 실물에서 무너졌던 조건(낮은 게인 배율, 한 주기 늦은 관절값, 큰 영점 어긋남)에 더 강해지는가" 이다.
> 기준선: `rough_box_main_v11a` (2026-10-11 발사, 10-13 실물 배포, 서지 못함). 설정 사본: `.claude/worktrees/vibecoded-audit-20261012/vibecoded-audit-20261012/reference/rough_main/` (`env.yaml`, `launch_manifest.json`).
> 계보: 평지 v11a → 험지 `rough_box_main_v11a` → **이 런**.

## §0 왜 하는가

- 개선안 전체는 [[../168_v11a_improvement_plan]]. 그중 지연·게인 무작위화(1절), 밀기 완화(2절), 발목 크랭크 게인(3절)을 한 번에 시험한다. 계획 168 의 7절은 원래 이 셋을 따로 비교하는 단일 변인 묶음(A/B/C)을 제안했다. 이번 시험은 그 앞에 두는 **빠른 확인**이다: 합쳐서 넣었을 때 학습 자체가 되는지, 되면 방향이 맞는지를 먼저 본다. 방향이 맞으면 그다음에 갈라서 원인을 찾는다.
- 크랭크 게인의 근거는 [[2026-10-14_v7b_pd_gain_check]]: 걷기 중 크랭크 토크 P99 23 N·m 를 내려면 목표와 실제의 위치 차이가 60° 필요하다. 행동 창이 ±62° 라서 목표가 창 끝에 붙어 있고, 그래서 발목 토크가 약 24 N·m 에서 막힌다. 서 있을 때도 목표-실제 차이가 12~16° 였다.
- 실물이 무너진 쪽(10-13 실기 실패)은 낮은 게인 배율, 한 주기 늦은 관절값, 영점 어긋남이었고, 학습에는 이것들이 전부 없었다 (계획 168 1절).

## §1 기준선에서 바뀐 것

기준선과 **같은 것**: 시작 체크포인트(평지 v11a `model_29998.pt` 의 배우만), 씨앗 43, 12,000 회, 환경 4,096개, 기록 방식 tensorboard, 지형(상자 지형), 보상 전부(항·가중치 변경 없음), 명령 상자, 관절 범위, 로봇 모델, 기준선의 `PYG_*` 환경 변수 전부(엔트로피 0.002 고정, 가치 함수 예열 300 회, 무작위화 램프 2,000→8,000 회 등).

| 변경 | 환경 변수 | 기준선 | 이번 시험 | 이유 (잰 숫자) |
|---|---|---|---|---|
| 명령 지연 | `PYG_ACTION_DELAY` | 없음 | 1.0 (정책 주기 0~1 개, 즉 0~20 ms 무작위, 무작위화 램프를 따라 켜짐) | 실측 관측 지연이 한 주기(20 ms). 정확히는 "관절값이 늦게 보이는 것" 이고 이 변수는 "명령이 늦게 닿는 것" 을 흉내 낸다 — 되먹임 고리의 총 지연을 0~20 ms 로 만든다는 점에서 대용이다 (관측 지연 자체를 넣는 것은 계획 168 의 1-1, 이번에는 안 함) |
| 게인 무작위화 | `PYG_PD_GAIN_DR` | 없음 | 0.8,1.2,0.7,1.3 (위치 게인 ×0.8~1.2, 감쇠 게인 ×0.7~1.3, 리셋마다, 램프를 따라 켜짐) | 실기 조건 시뮬에서 게인 배율 0.7 은 8 번 중 7 번, 0.5 는 8 번 모두 실패 (기준선) |
| 밀기 완화 | `PYG_PUSH_SOFT` | 간격 1~3 초, 수평 ±0.7 m/s, 수직 ±0.4 m/s, 회전 롤·피치 ±0.52 · 요 ±0.78 rad/s | 간격 **8~15 초**, 수평 **±0.5 m/s**, 수직·회전 0 (램프의 최대값 `push_max` 도 같이 바꿈) | 1~3 초 간격이면 한 걸음마다 밀린다. 밀기가 과하면 자세를 포기하고 넓게 버티는 쪽으로 간다 (계획 168 2절) |
| 엔코더 영점 치우침 | `PYG_ENC_BIAS_MAX` | ±0.015 rad (±0.86°) | **±0.05 rad (±2.9°)** (램프의 최대값 `encoder_max` 도 같이 바꿈) | 실물 영점 어긋남은 최대 9.4° (0.164 rad, 왼무릎, 2026-09-15 손 측정). 재보정 뒤 1~5° 로 예상 (사용자 10-14). 9.4° 전부는 아니고 재보정 뒤 범위 위쪽까지만 |
| 크랭크 위치 게인 | `PYG_CRANK_KP` | 22.3 N·m/rad | **55** | 걷기 중 크랭크 목표 오차 P99 60° 가 행동 창 ±62° 끝에 붙음. 55 이면 같은 토크에 약 24° |
| 크랭크 감쇠 게인 | `PYG_CRANK_KD` | 1.41 N·m·s/rad | **2.5** | 위치 게인을 올리면 감쇠비가 떨어지므로 함께 올림 (감쇠비 0.58 → 약 0.65, 관성 0.067 kg·m² 기준) |

구현 위치: `mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py` (밀기·엔코더), `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py` (크랭크 게인). 새 변수 넷은 꺼져 있으면 아무것도 바꾸지 않는다. 크랭크 게인은 발목 크랭크 서보(`.*_crank_[AB]_joint`)에만 적용되고, 행동 크기(관절당 1 단위 = 0.25 rad 고정)와 토크 상한은 게인에서 유도되지 않으므로 그대로다.

**발사 전 실증**: 기준선 `PYG_*` 로 만든 환경 설정과 이번 시험 `PYG_*` 로 만든 환경 설정을 덤프해 비교했다. 달라진 곳은 위 표의 항목뿐이다 — 행동 항목의 지연·잡음 인자, 게인 무작위화 이벤트와 커리큘럼 키, 밀기 간격과 속도 범위와 `push_max`, 엔코더 범위와 `encoder_max`, 크랭크 강성·감쇠. 그 밖의 줄은 한 줄도 다르지 않다.

**일부러 넣지 않은 것**
- 응답 유실 무작위화 (계획 168 의 1-5, 새 함수가 필요함).
- 모터 세기 무작위화 (1-4, 새 이벤트가 필요함).
- 엉덩이·무릎 게인 하향: 실물 모터가 RS04 인지 RS03 인지 답이 오기 전에는 표를 다시 계산해야 할 수 있어 보류 (계획 168 0-1).
- 보상 정리(중복 항 제거·규제 보강): 근거 조사 노트가 먼저 필요해서 보류. 보상은 이번에 한 줄도 바꾸지 않았다.
- 관절 관측 지연 그 자체 (1-1), IMU 부착각 (1-6), 게이트 수정.


> 주의 (발사 뒤 확인, 2026-10-14): 기준선과 같이 이 런도 평지 v11a 2단계 최종 정책에서 **행위자만 이어 받아** 시작한다. 그 정책은 크랭크 게인 22.3/1.41 에서 배웠으므로, 시작 직후 55/2.5 에 맞춰 다시 적응하는 구간이 생긴다. 초반 수백 회의 보상·넘어짐이 기준선보다 나쁘더라도 그것만으로 실패로 보지 않는다 — 판정은 §1c 의 완주 뒤 측정으로만 한다.

## §1c 사전 등록한 판정 기준

완주 뒤 아래를 순서대로 확인한다 (발사 전에 적어 둔다).

1. 학습이 완주하고, 낙상이 기준선의 최종 낙상률보다 올라가지 않는다. (기준선 마지막 200회 평균: `Episode_Termination/fell_over` 0.0021, `low_base` 1.67 — 같은 지표로 비교)
2. 제자리 회전 중앙값이 기준선(45 %) 이상이고, 반대 방향 회전 명령이 더 늘지 않는다.
3. 서기 흘러감이 0.5 m 이하.
4. 실기 조건 시뮬(`tools/sim2real/real_condition_sim.py`): 게인 배율 0.7 과 0.5 에서 기준선보다 잘 버틴다 (기준선 0.7: 8 번 중 7 번, 0.5: 8 번 중 0 번). 실제 타이밍 조건에서는 8/8 을 유지한다.
5. 걷는 동안 크랭크 목표 오차 P99 가 행동 창 62° 보다 한참 아래 (기준선 약 60°).

## §1d 실행

| 항목 | 값 |
|---|---|
| 발사 스크립트 | `mujoco-sim/mjlab/analysis/launch_rough_trial_v11a.py` (기준선 발사기 `launch_rough_main.py` 에 시험 팔 한 줄과 재현 사본 대상 두 파일을 더한 것) |
| 명령 | 기준선 `launch_manifest.json` 의 명령과 같음: `analysis/train_actor_warmstart.py Mjlab-Velocity-Rough-Pygmalion --env.scene.num-envs 4096 --agent.max-iterations 12000 --agent.seed 43 --agent.logger tensorboard --agent.resume True --agent.load-run 2026-10-02_23-52-14_legonly_ab_v11a_stand01_s43_p2 --agent.load-checkpoint model_29998.pt` |
| 발사 위치 | 원격 3090 (`10.8.0.14`), 사본 폴더 (git 아님) |
| 발사 전 시험 | 같은 환경 변수로 환경 4,096개의 설정을 덤프해 비교, 환경 64개로 4회 짧은 학습을 돌려 오류 없음 확인 |
| 발사 시각·런 폴더·진행 | (발사 후 기입) |

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
| crank_A | RS03 | 55 | 2.5 | 60 |
| crank_B | RS03 | 55 | 2.5 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-10-14_18-24-35_rough_box_trial_v11a`)

| 관절 그룹 | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] | effort 한계 [N·m] | 무부하 속도 [rad/s] | 로터 관성 armature [kg·m²] | 쿨롱 마찰 [N·m] | 점성 [N·m·s/rad] | T-N 곡선 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| crank_A, crank_B | RS03 | 55 | 2.5 | 60 (stall 59.7) | 20.94 | 0.0638 | 0.251 | 0.0252 | 실측 37점 (`PYG_TN=1`) |
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
| `PYG_ACTION_DELAY` | 1.0 |
| `PYG_ANKLE_MODE` | AB |
| `PYG_ARM_ABD_DEG` | 15 |
| `PYG_BLOWUP_TERM` | 1 |
| `PYG_BOX_TERRAIN` | 1 |
| `PYG_CMD_VY_STAGES` | 1 |
| `PYG_CRANK_KD` | 2.5 |
| `PYG_CRANK_KP` | 55 |
| `PYG_CRITIC_DR_OBS` | 1 |
| `PYG_CRITIC_WARMUP_ITERS` | 300 |
| `PYG_DR_END_ITER` | 8000 |
| `PYG_DR_START_ITER` | 2000 |
| `PYG_ENC_BIAS_MAX` | 0.05 |
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
| `PYG_PD_GAIN_DR` | 0.8,1.2,0.7,1.3 |
| `PYG_PUSH_SOFT` | 1 |
| `PYG_REL_HEADING` | 0.3 |
| `PYG_REL_STANDING` | 0.1 |
| `PYG_REWARD_TERM_CLAMP` | 50 |
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_SOFT_LANDING` | 1 |
| `PYG_SOFT_LANDING_MODE` | half |
| `PYG_STAGNATION` | 1 |
| `PYG_STUDENT_TEACHER` | 1 |
| `PYG_TN` | 1 |
| `PYG_V2` | 1 |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

**P2 (`2026-10-02_23-52-14_legonly_ab_v11a_stand01_s43_p2`) — P1과 다름, 별도 표:**

**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-10-02_23-52-14_legonly_ab_v11a_stand01_s43_p2`)

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

모델 출처: 런 디렉토리 `repro/` 스냅샷 (권위) — `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml`

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

<!-- SPEC-TABLES:END -->

## §2 이후

(완주 후)

## 중간 점검 — 4,000 회 (2026-10-14 22:19, 학습 기록 직전 100 회 평균, 기준선 같은 반복 대비)

| 지표 | 기준선 | 시험 런 |
|---|--:|--:|
| 평균 보상 | 71.9 | 73.1 |
| 평균 에피소드 길이 | 783 | 780 |
| 넘어짐 종료 (fell_over) | 0.003 | 0.006 |
| 낮은 몸통 종료 (low_base) | 1.63 | 1.64 |
| 직선 속도 추종 보상 | 1.068 | 1.070 |
| 회전 속도 추종 보상 | 0.930 | 0.898 (−3 %) |
| 움직일 때 속도 오차 [m/s] | 0.476 | 0.458 |
| 무작위화 램프 | 0.325 | 0.325 |

판단: 기준선과 거의 같은 궤적 — 크랭크 게인 변경 적응 구간에서도 무너짐 없음. 무작위화(지연·게인 배율 포함)는 아직 33 % 램프라 차이는 8,000 회 이후에 드러날 것. 중단 조건 없음.

## 중간 점검 — 8,000 회 (2026-10-15 02:11, 직전 100 회 평균, 기준선 같은 반복 대비)

| 지표 | 기준선 6k | 시험 6k | 기준선 8k | 시험 8k |
|---|--:|--:|--:|--:|
| 무작위화 램프 | 0.66 | 0.66 | 0.99 | 0.99 |
| 평균 보상 | 69.3 | 70.1 | 62.9 | 59.8 |
| 넘어짐 종료 (fell_over) | 0.006 | 0.007 | 0.018 | **0.005** |
| 낮은 몸통 종료 (low_base) | 1.73 | 1.71 | 1.89 | 2.05 |
| 직선 속도 추종 보상 | 1.027 | 1.051 | 0.924 | 0.935 |
| **회전 속도 추종 보상** | 0.905 | 0.838 (−7 %) | 0.831 | **0.628 (−24 %)** |
| 움직일 때 속도 오차 [m/s] | 0.478 | 0.455 | 0.491 | 0.462 |

판단: 무작위화가 최대(지연 0~20 ms·게인 배율 포함)에 가까워지면서 **회전 추종이 기준선보다 24 % 낮아졌다** — 이 시험의 첫 경고 신호. 넘어짐 종료는 오히려 1/3 이고 직선 추종은 같거나 낫다.
학습 중 보상은 무작위화가 켜진 상태에서 잰 값이라, 무작위화를 끈 회전 측정(§1c 2번)이 실제로 떨어졌는지는 완주 뒤 측정으로 판정한다. 남은 4,000 회는 최대 무작위화 유지 구간이라 회복 여부도 그때 본다. 중단 조건(회전 측정 < 20 % 두 번 연속)은 측정 전이라 해당 없음 — 계속.
