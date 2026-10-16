# rough_gain_mountain_v11a — 배포 정책 설정에 등반 권장 게인을 적용한 팔 B (2026-10-15)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> **이 런은 게인만 바꾼 시험이다.** 바뀐 것은 다리 관절 네 그룹의 위치·감쇠 게인뿐이다. 보상, 지형, 무작위화는 기준선과 같다.
> 짝: 팔 A [[2026-10-15_rough_gain_crank60_v11a]]. 비교 노트: [[2026-10-15_ab_rough_gain_compare]].
> 기준선: `rough_box_main_v11a` (2026-10-11 발사, 10-13 실물 배포).
> 계보: 평지 v11a → 험지 `rough_box_main_v11a` → **이 런**.

## 용어 정의

| 용어 | 뜻 |
|---|---|
| 기준선 | 배포 중인 험지 정책 `rough_box_main_v11a` 의 결과를 말한다. |
| 위치 게인 (Kp) | 목표 각도와 실제 각도가 1 rad 어긋날 때 모터가 내는 토크이다. 단위는 N·m/rad 이다. |
| 감쇠 게인 (Kd) | 관절 속도 1 rad/s 마다 모터가 내는 제동 토크이다. 단위는 N·m·s/rad 이다. |
| 엉덩이 앞뒤·좌우 | 엉덩이 피치 관절과 롤 관절이다. 모터는 RS04 이다. |
| 엉덩이 비틀기 | 엉덩이 요 관절이다. 모터는 RS03 이고 앞뒤·좌우와 다른 액추에이터 설정을 쓴다. |
| 크랭크 | 발목을 구동하는 모터 축이다. 모터는 RS03 이다. |
| 등반 권장 | [[../169_pd_gain_selection_criteria]] §8-7 이 관악산 등반 기준으로 정한 게인 한 벌이다. |
| 사전 등록 기준 | 결과를 보기 전에 적어 둔 합격 조건이다. 완주 뒤 숫자만 보고 기계적으로 적용한다. |

## §0 왜 하는가

- 사용자 결정 (2026-10-15): 실물 모터는 URDF 와 같다. 엉덩이 앞뒤·좌우와 무릎은 RS04, 엉덩이 비틀기와 크랭크는 RS03 이다.
- 질량은 하체만이며 모델 23.6 kg 이다. 목표는 관악산 등반이다.
- [[../169_pd_gain_selection_criteria]] §8-7 은 이 조건에서 엉덩이 앞뒤·좌우 135/6, 엉덩이 비틀기 75/4, 무릎 130/6, 크랭크 60/2.7 을 권장한다.
- 배포 게인은 엉덩이 150/6, 무릎 220/6, 크랭크 22.3/1.41 이다. 권장은 무릎과 엉덩이 비틀기를 크게 낮춘다.
- 직전 묶음 시험 [[2026-10-14_rough_box_trial_v11a]] §12 의 판정은 "채택하지 않는다"였다. 회전·방향·서기 정밀도가 나빠졌고, 원인을 가를 수 없었다.
- 이 팔은 "등반 권장 게인을 정책이 소화하는가" 를 묻는다. 무작위화는 넣지 않아서, 회전 손실이 생기면 원인은 게인 쪽이다.
- 팔 A 와의 차이(엉덩이·비틀기·무릎)가 곧 이 팔이 더하는 효과이다.

## §1 기준선에서 바뀐 것

기준선과 **같은 것**: 시작 체크포인트(평지 v11a `model_29998.pt` 의 배우만), 씨앗 43, 12,000 회, 환경 4,096개, 기록 방식 tensorboard, 지형(상자 지형), 보상 전부, 명령 상자, 관절 범위, 로봇 모델, 기준선의 `PYG_*` 환경 변수 전부.

| 변경 | 환경 변수 | 기준선 | 이번 팔 | 이유 (잰 숫자) |
|---|---|---|---|---|
| 엉덩이 앞뒤·좌우 위치 게인 | `PYG_HIP_KP` | 150 N·m/rad | **135** | 오르막 필요 Kp 117(앞뒤)·136(좌우). 규칙 범위 82~134 |
| 엉덩이 앞뒤·좌우 감쇠 게인 | `PYG_HIP_KD` | 6 N·m·s/rad | **6** (같은 값, 명시) | 감쇠비 0.93 |
| 엉덩이 비틀기 위치 게인 | `PYG_HIP_YAW_KP` | 150 N·m/rad | **75** | 오르막 필요 Kp 67. 규칙 범위 41~67 |
| 엉덩이 비틀기 감쇠 게인 | `PYG_HIP_YAW_KD` | 6 N·m·s/rad | **4** | 감쇠비 0.91 |
| 무릎 위치 게인 | `PYG_KNEE_KP` | 220 N·m/rad | **130** | 오르막 필요 Kp 116. 25 cm 단 착지 2 배 하중 78 N·m 에서 오차 34°, 창 54° 안 |
| 무릎 감쇠 게인 | `PYG_KNEE_KD` | 6 N·m·s/rad | **6** (같은 값, 명시) | 감쇠비 0.95 |
| 크랭크 위치 게인 | `PYG_CRANK_KP` | 22.3 N·m/rad | **60** | 팔 A 와 같다. 오르막 필요 Kp 66, 열 상한 64 |
| 크랭크 감쇠 게인 | `PYG_CRANK_KD` | 1.41 N·m·s/rad | **2.7** | 팔 A 와 같다. 감쇠비 0.69 |

**무작위화 플래그는 기준선의 것 그대로이다.** 명령 지연, 게인 무작위화, 밀기 완화, 엔코더 치우침 확대를 켜지 않는다. 기준선의 무작위화(밀기·마찰·질량·엔코더·무게중심, 2,000→8,000 회 램프)는 그대로 켜져 있다.

구현 위치: `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py` (mjlab 저장소 커밋 `8fa9ebe0`). 새 변수 여섯은 꺼져 있으면 아무것도 바꾸지 않는다.
- 엉덩이 앞뒤·좌우는 `PYG_ACTUATOR_04` (`.*_hip_pitch_joint`, `.*_hip_roll_joint`) 이다.
- 엉덩이 비틀기는 별도 액추에이터 `PYG_ACTUATOR_03_HIP_YAW` (`.*_hip_yaw_joint`) 이다. 앞뒤·좌우와 설정을 공유하지 않는다.
- 무릎은 `PYG_ACTUATOR_04_KNEES` (`.*_knee_joint`) 이다.
- 허리와 어깨(상체 관절을 켜는 경우)는 건드리지 않는다.
- 행동 크기(관절당 1 단위 = 0.25 rad 고정), 토크 상한, 행동 한계(clip)는 게인에서 유도되지 않는다. 코드에서 확인했다.

**발사 전 실증** (`mujoco-sim/mjlab/analysis/verify_gain_flags.py`, 환경 2개, CPU):
- 기준선 `PYG_*` 로 만든 환경 설정은 배포 런의 `env.yaml` 과 환경 개수 두 줄(4,096 대 2)을 빼고 한 줄도 다르지 않다.
- 이 팔의 환경 설정은 기준선과 게인 필드 열두 줄만 다르다.
- 만들어진 액추에이터의 값:

| 그룹 | 기준선 Kp / Kd | 이번 팔 Kp / Kd (만들어진 값) | 토크 상한 [N·m] |
|---|---|---|---|
| 크랭크 네 개 | 22.27 / 1.41 | 60 / 2.7 | 60 |
| 엉덩이 비틀기 둘 | 150 / 6 | 75 / 4 | 60 |
| 엉덩이 앞뒤·좌우 넷 | 150 / 6 | 135 / 6 | 120 |
| 무릎 둘 | 220 / 6 | 130 / 6 | 120 |

> 주의: 이 런도 평지 v11a 2단계 최종 정책에서 **배우만 이어 받아** 시작한다. 그 정책은 배포 게인에서 배웠다. 특히 무릎 220→130, 비틀기 150→75 는 변화가 크다. 초반 수백 회의 보상·넘어짐이 기준선보다 나빠도 그것만으로 실패로 보지 않는다. 판정은 §1c 의 완주 뒤 측정으로만 한다.

## §1c 사전 등록한 판정 기준

완주 뒤 아래를 순서대로 확인한다. 기준선 숫자는 직전 시험 노트에서 가져온다.

1. 마지막 200 회 평균 `fell_over` 와 `low_base` 가 기준선보다 높지 않다 (기준선 0.0021 / 1.672).
2. 제자리 회전: 씨앗 8 개의 중앙값이 기준선 이상이고, 반대 방향 회전이 더 늘지 않는다 (기준선 45.3 % 는 씨앗 4 개 값이다. **기준선도 씨앗 8 개로 다시 잰다**).
3. 서 있는 동안 흘러간 거리가 0.5 m 이하이다.
4. 실기 조건 시뮬레이터(`tools/sim2real/real_condition_sim.py`): 게인 배율 0.7 과 0.5 에서 기준선보다 잘 버틴다 (기준선 7/8 과 0/8). 실제 타이밍 조건은 8/8 이다. 이 시뮬레이터는 팔의 게인을 받아야 한다. 크랭크는 `--crank-kp 60 --crank-kd 2.7` 로 이미 줄 수 있다. 엉덩이·비틀기·무릎용 `--hip-kp` 류 플래그는 아직 없다. **평가할 때 추가한다. 지금은 추가하지 않는다.**
5. 걷는 동안 크랭크 목표 오차 99 백분위수가 62° 미만이고, 크랭크 토크 제곱평균제곱근이 20 N·m (정격) 이하이다.
6. 오르막 20°와 계단 오름 15 cm: `tools/robot_gains/rough_load_probe.py` 로 넘어짐 횟수와 토크 99 백분위수를 잰다.

## §1d 실행

| 항목 | 값 |
|---|---|
| 발사 스크립트 | `mujoco-sim/mjlab/analysis/launch_rough_gain_ab_v11a.py --arm rough_gain_mountain_v11a` |
| 명령 | 기준선과 같음 (팔 A 노트 §1d 참조) |
| 발사 위치 | 원격 3090 (`10.8.0.14`), 사본 폴더 (git 아님) |
| 발사 전 시험 | 환경 설정 비교(위), 환경 64개로 4회 짧은 학습 |
| 발사 시각 | 팔 A 가 끝나면 자동 시작 (대기열 `analysis/gain_ab_queue_b.sh`). 예상 2026-10-16 10:00 경. 실제 시각은 시작 뒤 기입 |
| 대기열을 쓰는 이유 | 팔 A 혼자 그래픽 메모리 16.9 GB 를 쓴다. 남는 7.7 GB 는 팔 A 사용량에 1.5 GB 를 더한 값(18.4 GB)보다 작다 |
| 속도·예상 완주 | 팔 A 와 같다면 12,000 회에 약 11.5 시간. 예상 완주 2026-10-16 22:00 경 |
| 발사 전 시험 결과 | 환경 설정 비교 통과. 환경 64개 4회 학습 통과: 런 `smoke_gain_mountain`, 반복 시간 0.80 초, `params/env.yaml` 의 게인이 위 표와 같음 |

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
| hip_pitch | RS04 | 135 | 6 | 120 |
| hip_roll | RS04 | 135 | 6 | 120 |
| hip_yaw | RS03 | 75 | 4 | 60 |
| knee | RS04 | 130 | 6 | 120 |
| crank_A | RS03 | 60 | 2.7 | 60 |
| crank_B | RS03 | 60 | 2.7 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-10-16_10-11-25_rough_gain_mountain_v11a`)

| 관절 그룹 | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] | effort 한계 [N·m] | 무부하 속도 [rad/s] | 로터 관성 armature [kg·m²] | 쿨롱 마찰 [N·m] | 점성 [N·m·s/rad] | T-N 곡선 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| crank_A, crank_B | RS03 | 60 | 2.7 | 60 (stall 59.7) | 20.94 | 0.0638 | 0.251 | 0.0252 | 실측 37점 (`PYG_TN=1`) |
| hip_yaw | RS03 | 75 | 4 | 60 (stall 59.7) | 20.94 | 0.0638 | 0.251 | 0.0252 | 실측 37점 (`PYG_TN=1`) |
| hip_pitch | RS04 | 135 | 6 | 120 (stall 120.1) | 20.94 | 0.0773 | 0.519 | 0.0275 | 실측 22점 (`PYG_TN=1`) |
| knee | RS04 | 130 | 6 | 120 (stall 120.1) | 20.94 | 0.0773 | 0.519 | 0.0275 | 실측 22점 (`PYG_TN=1`) |
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
| `PYG_BOX_TERRAIN` | 1 |
| `PYG_CMD_VY_STAGES` | 1 |
| `PYG_CRANK_KD` | 2.7 |
| `PYG_CRANK_KP` | 60 |
| `PYG_CRITIC_DR_OBS` | 1 |
| `PYG_CRITIC_WARMUP_ITERS` | 300 |
| `PYG_DR_END_ITER` | 8000 |
| `PYG_DR_START_ITER` | 2000 |
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
| `PYG_HIP_KD` | 6 |
| `PYG_HIP_KP` | 135 |
| `PYG_HIP_YAW_KD` | 4 |
| `PYG_HIP_YAW_KP` | 75 |
| `PYG_INERTIAL_DR` | 1 |
| `PYG_INIT_BENT` | 1 |
| `PYG_INIT_MID` | 1 |
| `PYG_KNEE_EXT` | 1 |
| `PYG_KNEE_EXT_DEG` | 25 |
| `PYG_KNEE_EXT_W` | 2.0 |
| `PYG_KNEE_KD` | 6 |
| `PYG_KNEE_KP` | 130 |
| `PYG_LOW_BASE_REL` | 1 |
| `PYG_MASS_DR_JSON` | `mass_dr_legonly_fastener50_prototype-tempmass.json` |
| `PYG_MODEL_TAG` | LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix |
| `PYG_MOTOR_MEAS` | 1 |
| `PYG_NAN_TERM` | 1 |
| `PYG_OBS_NAN_GUARD` | 1 |
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

<!-- SPEC-TABLES:END -->

## §2 이후

(완주 후)
