# 학습 리포트 — `ABORTED_2026-09-14_19-37-39_legonly_ab_v6_standing_p1` (반복 2,434에서 **중단**)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> **이 런은 끝까지 가지 않았다.** 결과를 쓰는 노트가 아니라 **왜 멈췄는지와 그래서 무엇을
> 고쳤는지**를 남기는 노트다. 이 런을 대체한 것은
> [[2026-09-14_legonly_ab_v6b_standing_flatfoot]] 이고, 계보는
> [[2026-09-13_legonly_ab_v5_heading]] → 이 런 → v6b 이다.
>
> **소급 작성(2026-09-14 밤).** 중단 사유는 대체 런 노트 §0a 에서, 숫자는 이 런의 학습
> 기록에서 가져왔다.

## §1 재현성

| 항목 | 값 |
|---|---|
| 런 디렉터리 | `mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/ABORTED_2026-09-14_19-37-39_legonly_ab_v6_standing_p1` |
| 발사 | 2026-09-14 19:37:39 |
| 중단 | 같은 날 22:5x · 마지막 저장본 `model_2400.pt` (기록은 반복 2,459점) |
| 의도한 변인 | `PYG_REL_STANDING` 0.1 → 0.3 (명령 0 구간을 3배로) |
| 로봇 모델 | `LegOnly_..._v30_proxyfix` · 발목 AB · 평지 · 시드 42 · 환경 16384 |
| 발사 전 검사 | 통과 — 구동관절 12개, 좌우 이름, 폐루프 0.0000 mm |

발사 전 검사는 **통과했는데도** 아래 결함이 걸러지지 않았다. 그 자체가 이 런의 교훈이다.

## §2 중단 시점의 지표 (학습 기록에서 그대로)

| 항목 | 값 |
|---|--:|
| 평균 보상 | 94.00 (학습 중 최고 105.51) |
| 평균 한 판 길이 | 967.63 / 1000 |
| 넘어져서 끝남 | 0.0000 |
| 커리큘럼 단계 | 3 / 4 (아직 올라가는 중) |
| 흔들기 세기 | 0.0000 (1단계라 정상) |
| 방향(회전) 오차 | 0.6364 |
| 앞뒤·좌우 속도 오차(정상 구간) | 0.1765 |
| **자세 추종 항** | **0.5297** |

**학습 자체는 건강했다.** 넘어짐 0, 한 판 길이 968/1000. 자세 추종 항도 v5 의
1.7e−04 에서 **0.53 으로 살아났다** — 의도한 변인은 효과가 있었다. 그런데도 멈춘 이유는
아래다.

## §3 ★ 왜 멈췄나 — **목표 자세부터 틀렸다**

사용자가 렌더를 보고 지적했다: *"target pose 부터가 이상한데? 발이 들려있는거 아닌가?"*
**맞았다.**

리셋과 목표로 쓰는 웅크림 자세의 **발목 값이 깊은 웅크림 것**이었다. 기하로는 엉덩이
굽힘 `h`, 무릎 굽힘 `k` 에 대해 발바닥이 수평이려면 발목이 `k − h` 여야 한다. 이 런들이
쓰는 얕은 웅크림(`PYG_INIT_MID=1`)은 **0.175 rad** 이 맞는데, 키프레임에는 깊은 웅크림
값 **0.36** 이 상수로 박혀 있었다.

| | 전 (이 런) | 후 (v6b) |
|---|--:|--:|
| 발목 숙임 목표 | +20.63° | **+10.03°** |
| 발바닥 앞뒤 기울기 | +10.600° | **+0.000°** |
| 앞뒤 단차 | 47.8 mm | **0.1 mm** |
| 접지 귀퉁이 | **4 / 24** | **24 / 24** |
| 접지 지지면 앞뒤 폭 | **0 mm** (선 위에 섬) | **260 mm** |
| 폐루프 벌어짐 | 0.00000 mm | 0.00000 mm |

즉 **발끝만 바닥에 닿은 자세를 "목표" 로 놓고 학습시키고 있었다.** 지지면 앞뒤 폭이
0 mm — 선 위에 서 있는 셈이라, 정지 자세를 잡으라고 시켜도 잡을 수 있는 자세가 아니었다.

## §4 이 런이 남긴 것

1. **의도한 변인은 효과가 있었다** (§2: 자세 항 1.7e−04 → 0.53). 그래서 v6b 는 같은
   변인을 유지한 채 발목 키프레임만 고쳐 다시 발사했다.
2. **발사 전 검사에 구멍이 있었다.** 폐루프 벌어짐과 좌우 이름은 봤지만 **발바닥이
   바닥에 평평하게 닿는지**는 안 봤다. 접지 귀퉁이 4/24 로 통과했다.
3. 그래서 **"목표 자세를 눈으로 확인한다"** 가 숫자 검사만큼 중요하다는 실례가 하나 더
   생겼다 — 이번에도 사람이 렌더를 보고 잡았다.

## §5 안 쓰는 것

* 저장본(`model_*.pt` 25개)은 **쓰지 않는다.** 목표 자세가 틀린 채 배운 정책이다.
* 부하·모터활용·설계 숫자를 이 런에서 뽑지 않았고, 뽑아서도 안 된다.
* 후속은 전부 [[2026-09-14_legonly_ab_v6b_standing_flatfoot]] 에서 본다.

<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->

## 1b. 이 run의 Reward & Gains (config에서 파싱 — 재현용)

**Reward 항목** (weight·왜·어떻게):

| reward | weight | 왜 | 어떻게 |
|---|--:|---|---|
| foot_clearance | **-2** | 스윙발 지면 이격(발끌림 방지) | 목표 높이 오차×발 수평속도 |
| stance_knee_extension | **-2** | 입각 중 과도한 crouch 억제 | 접촉 중 \|knee\|>25 deg 초과량^2 |
| track_angular_velocity | **+2** | 명령 회전속도 추종 | exp(-err²/std²) |
| track_linear_velocity | **+2** | 명령 전진/측방 속도 추종 | exp(-err²/std²) |
| air_time | **+1** | 체공시간 보상(질질끌기 억제) | 0.05~0.5 s 체공 발 수; \|command\|>0.5에서 활성 |
| dof_pos_limits | **-1** | 관절범위 한계 벌점 | 한계초과 L1 |
| pose | **+1** | 기본 관절자세 정규화(기괴자세 억제) | default-pose L2 |
| self_collisions | **-1** | 자기충돌 벌점 | -접촉수 |
| stand_still_penalty | **-1** | 이동 명령을 무시하고 서는 stall 방지 | 명령 대비 진행률<30%이면 flat cost |
| track_lin_vel_progress | **+1** | 고속 명령에서 정지하는 local optimum 방지 | 명령방향 실제속도 투영값, 명령크기에서 cap |
| upright | **+1** | 몸통 직립 유지(넘어짐 방지) | exp 자세 |
| foot_impact_velocity | **-0.5** | 착지 직전 발 하강속도 감소 | 지면 근처 공중 발의 downward velocity^2 |
| knee_overspeed | **-0.5** | 실측 RS04 무부하 속도를 넘는 보행 억제 | relu(\|knee velocity\|-19.9)^2 |
| foot_swing_height | -0.25 | 스윙발 높이 성형 | 스윙 중 목표 높이 오차 |
| action_rate_l2 | -0.1 | 액션 급변 벌점 | -\|Δa\|² |
| foot_slip | -0.1 | 접지발 미끄러짐 벌점 | -\|v_contact\| |
| body_ang_vel | -0.05 | 몸통 각속도 벌점(흔들림 억제) | -\|ω\|² |
| angular_momentum | -0.02 | 전신 각운동량 벌점(회전 낭비 억제) | -\|L\|² |
| thermal_effort | -0.02 | ★열분배: Σ(τ/rated)² 정규화(관절 균등화) | -Σ(τ/rated)² |
| contact_force_cap | -0.01 | ★충격 cap: 발 GRF 역치초과분 벌점(사뿐착지) | -min(max(F-600,0),800) |
| soft_landing | -1e-05 | 착지 첫접촉 충격 벌점(약) | -첫접촉 GRF |
| torque_limit | -0 | commanded 토크 한계초과 벌점 | off(0) |

**관절별 Kp/Kd** (position-PD, effort=관절측 peak):

| 관절 | 모터 | Kp(stiffness) | Kd(damping) | effort [N·m] |
|---|---|--:|--:|--:|
| hip_pitch | RS04 | 150 | 6 | 120 |
| hip_roll | RS04 | 150 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| knee | RS04 | 220 | 6 | 120 |
| crank_A | RS03 | 22.3 | 1.41 | 60 |
| crank_B | RS03 | 22.3 | 1.41 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `ABORTED_2026-09-14_19-37-39_legonly_ab_v6_standing_p1`)

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
| R_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | 17.03 | 액션 |
| R_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.26 | 액션 |
| R_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.63 | 수동 |
| R_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0 | 수동 |
| L_hip_pitch_joint | [-25, 120] | [-17.7, 112.8] | [-17.7, 112.8] | 130.5 | 10.03 | 액션 |
| L_hip_roll_joint | [-85, 25] | [-79.5, 19.5] | [-79.5, 19.5] | 99 | 0 | 액션 |
| L/R_hip_yaw_joint | [-45, 45] | [-40.5, 40.5] | [-40.5, 40.5] | 81 | 0 | 액션 |
| L_knee_joint | [-120, 0] | [-114, -6] | [-114, -6] | 108 | -20.05 | 액션 |
| L_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.03 | 액션 |
| L_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | 17.26 | 액션 |
| L_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.63 | 수동 |
| L_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0 | 수동 |

액션 스케일 0.25 rad/단위, 오프셋 = default (`use_default_offset`). clip이 없는 구 설정에서는 정책 목표각을 시뮬레이터의 soft 한계가 사후에 잡는다 — 창은 soft 한계 폭으로 읽는다.

**§1b-4. 이 런의 스택 플래그 (`PYG_*`)**

출처: 런 디렉토리 `repro/launch_manifest.json` (권위)

| 플래그 | 값 |
|---|---|
| `PYG_ANKLE_MODE` | AB |
| `PYG_ARM_ABD_DEG` | 15 |
| `PYG_CMD_VY_STAGES` | 1 |
| `PYG_CRITIC_DR_OBS` | 1 |
| `PYG_DR_END_ITER` | 100000001 |
| `PYG_DR_START_ITER` | 100000000 |
| `PYG_GATED_CURRICULUM` | 1 |
| `PYG_GATE_ERR_RATIO` | 1.1 |
| `PYG_GATE_FELL_MAX` | 0.005 |
| `PYG_GATE_MAX_DWELL` | 3000 |
| `PYG_GATE_MIN_DWELL` | 800 |
| `PYG_GATE_MIN_EPISODES` | 64 |
| `PYG_GATE_WINDOW` | 100 |
| `PYG_INIT_BENT` | 1 |
| `PYG_INIT_MID` | 1 |
| `PYG_KNEE_EXT` | 1 |
| `PYG_KNEE_EXT_DEG` | 25 |
| `PYG_KNEE_EXT_W` | 2.0 |
| `PYG_MASS_DR_JSON` | `mass_dr_legonly_fastener50_prototype-tempmass.json` |
| `PYG_MODEL_TAG` | LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix |
| `PYG_MOTOR_MEAS` | 1 |
| `PYG_REL_HEADING` | 0.8 |
| `PYG_REL_STANDING` | 0.3 |
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_SOFT_LANDING` | 1 |
| `PYG_SOFT_LANDING_MODE` | half |
| `PYG_STUDENT_TEACHER` | 1 |
| `PYG_TN` | 1 |
| `PYG_V2` | 1 |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

<!-- SPEC-TABLES:END -->
