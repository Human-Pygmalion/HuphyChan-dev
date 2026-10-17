# rough_gain_crank60_gaindr_v11a — 팔 A 에 게인 무작위화만 더한 팔 C (2026-10-17)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> **이 런은 단일 변인 시험이다.** 팔 A 와 다른 것은 위치·감쇠 게인 무작위화 하나뿐이다.
> 짝: 팔 D [[2026-10-17_rough_gain_crank60_actdelay_v11a]] (지연만). 기준 팔: [[2026-10-15_rough_gain_crank60_v11a]]. 비교 노트: [[2026-10-15_ab_rough_gain_compare]].
> 묶음 시험: [[2026-10-14_rough_box_trial_v11a]].
> 계보: 평지 v11a → 험지 `rough_box_main_v11a` → 팔 A (크랭크 60/2.7) → **이 런**.

## 용어 정의

| 용어 | 뜻 |
|---|---|
| 기준선 | 배포 중인 험지 정책 `rough_box_main_v11a` 의 결과를 말한다. |
| 팔 A | 크랭크 게인만 60/2.7 로 바꾼 런이다. 무작위화를 더하지 않았다. |
| 팔 C | 팔 A 에 게인 무작위화만 더한 이 런이다. |
| 팔 D | 팔 A 에 명령 지연 무작위화만 더한 짝 런이다. |
| 묶음 시험 | 크랭크 55/2.5 와 무작위화 넷(지연, 게인, 밀기 완화, 엔코더 확대)을 한꺼번에 넣은 런이다. |
| 위치 게인 (Kp) | 목표 각도와 1 rad 어긋날 때 모터가 내는 토크이다. 단위는 N·m/rad 이다. |
| 감쇠 게인 (Kd) | 관절 속도 1 rad/s 마다 모터가 내는 제동 토크이다. 단위는 N·m·s/rad 이다. |
| 게인 무작위화 | 에피소드가 시작될 때마다 관절마다 Kp 와 Kd 에 배율을 곱하는 일이다. 배율은 범위 안에서 임의로 뽑는다. |
| 무작위화 곡선 | 무작위화 강도를 0 에서 최대로 서서히 올리는 학습 단계 계획이다. 기준선에서 2,000 회부터 8,000 회까지 오른다. |
| 사전 등록 기준 | 결과를 보기 전에 적어 둔 합격 조건이다. 완주 뒤 숫자만 보고 기계적으로 적용한다. |

## §0 왜 하는가

- 묶음 시험은 게인 배율 0.5 에서 8/8 을 버텼지만 제자리 회전이 45.3 % 에서 23.4 % 로 내려갔다 ([[2026-10-14_rough_box_trial_v11a]] §12).
- 팔 A 는 크랭크 게인만 바꿨다. 제자리 회전은 53.5 % 로 유지했지만 게인 배율 0.5 에서는 1/8 만 버텼다 ([[2026-10-15_rough_gain_crank60_v11a]] §12).
- 따라서 게인 배율 0.5 의 내성은 크랭크 게인이 아니라 무작위화에서 왔고, 회전 손실의 원인도 크랭크가 아니다. 남은 후보는 지연과 게인 무작위화이다.
- 이 팔은 게인 무작위화 하나만 팔 A 에 더한다. 가설은 "게인 무작위화가 게인 배율 0.5 의 내성을 주고, 회전을 크게 해치지 않는다" 이다.
- 이 가설은 아직 검증하지 않았다. 짝 팔 D 는 지연만 더해 두 원인을 가른다.

## §1 팔 A 에서 바뀐 것

팔 A 와 **같은 것**: 시작 체크포인트(평지 v11a `model_29998.pt` 의 배우만), 씨앗 43, 12,000 회, 환경 4,096개, 기록 방식 tensorboard, 지형, 보상 전부, 명령 상자, 관절 범위, 로봇 모델, 크랭크 게인 60/2.7, 팔 A 의 `PYG_*` 환경 변수 전부, 같은 발사기.

| 변경 | 환경 변수 | 팔 A | 팔 C | 이유 |
|---|---|---|---|---|
| 게인 무작위화 | `PYG_PD_GAIN_DR` | 없음 | **`0.8,1.2,0.7,1.3`** | 위치 게인 배율 0.8~1.2, 감쇠 게인 배율 0.7~1.3. 관절마다, 에피소드가 시작될 때마다 새로 뽑는다. 묶음 시험과 같은 값이다. |

- 무작위화 곡선: 코드가 게인 범위를 기존 무작위화 곡선(`dr_levels`)에 등록한다. 강도가 2,000 회에서 8,000 회 사이에 0 에서 1 로 오른다.
- 구현 위치: `mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py` 의 `PYG_PD_GAIN_DR` 블록. 이벤트는 mjlab 의 `dr.pd_gains` 이다.
- 지연, 밀기 완화, 엔코더 확대는 켜지 않는다 (`PYG_ACTION_DELAY`, `PYG_PUSH_SOFT`, `PYG_ENC_BIAS_MAX` 모두 없음).

**발사 전 실증** (`mujoco-sim/mjlab/analysis/verify_gain_flags.py --arms ...`, 환경 2개, CPU, 원자료 `analysis/out/verify_cd/`):
- 팔 C 의 환경 설정은 팔 A 와 74 줄 다르다. 모두 추가된 줄이다. 삭제된 줄은 없다.
- 추가된 줄은 둘이다. 하나는 `pd_gains` 이벤트(범위 0.8~1.2 와 0.7~1.3, 모드는 에피소드 시작, 연산은 곱하기)이다. 다른 하나는 무작위화 곡선의 `pd_gain_max` 이다.
- 만들어진 액추에이터의 값은 팔 A 와 같다. 크랭크 네 개 60/2.7, 엉덩이 비틀기 150/6, 엉덩이 앞뒤·좌우 150/6, 무릎 220/6. 무작위화는 에피소드가 시작될 때 이 값에 배율을 곱한다.
- 환경 64개로 4회 짧은 학습을 통과했다. 런 이름은 `smoke_cd_gaindr_v11a` 이다. 폴더의 `params/env.yaml` 에 `pd_gains` 가 한 번 있다.

> 주의: 이 런도 평지 v11a 2단계 최종 정책에서 **배우만 이어 받아** 시작한다. 초반 수백 회의 보상·넘어짐이 팔 A 보다 나빠도 그것만으로 실패로 보지 않는다. 판정은 §1c 의 완주 뒤 측정으로만 한다.

## §1c 사전 등록한 판정 기준 (팔 C 와 팔 D 에 같은 기준을 적용한다)

완주 뒤 아래를 순서대로 확인한다. 비교 대상은 팔 A 이다.

1. 마지막 200 회 평균 `fell_over` 와 `low_base` 가 팔 A 보다 높지 않다 (팔 A 는 0.0134 / 1.470).
2. 제자리 회전: 씨앗 8 개의 중앙값이 팔 A 의 53.5 % 에서 10 점을 뺀 값 이상이다. 반대 방향 회전은 팔 A 의 3/64 에 3 을 더한 값 이하이다.
3. 서 있는 동안 흘러간 거리가 0.5 m 이하이다.
4. 실기 조건 시뮬레이터(`tools/sim2real/real_condition_sim.py --crank-kp 60 --crank-kd 2.7`, 14 조건, 조건마다 8 번): 게인 배율 0.5 에서 팔 A 의 1/8 보다 잘 버틴다. 팔 C 의 가설은 6/8 이상이다. 팔 D 는 방향을 예측하지 않는다. 실제 타이밍 조건은 8/8 이다.
5. 직진 1.2 m/s 명령에서 걷는 동안 방향각 변화율(`tools/robot_gains/heading_drift_probe.py`)을 팔 A 의 +4.48 도/초와 비교한다. 임계값은 없고 값을 기록한다.

## §1d 실행

| 항목 | 값 |
|---|---|
| 발사 스크립트 | `mujoco-sim/mjlab/analysis/launch_rough_gain_ab_v11a.py --arm rough_gain_crank60_gaindr_v11a` (대기열 `analysis/gain_queue_cd.sh` 가 호출한다) |
| 명령 | 팔 A 와 같음: `analysis/train_actor_warmstart.py Mjlab-Velocity-Rough-Pygmalion --env.scene.num-envs 4096 --agent.max-iterations 12000 --agent.seed 43 --agent.logger tensorboard --agent.resume True --agent.load-run 2026-10-02_23-52-14_legonly_ab_v11a_stand01_s43_p2 --agent.load-checkpoint model_29998.pt` |
| 발사 위치 | 원격 3090 (`10.8.0.14`), 사본 폴더 (git 아님) |
| 발사 전 시험 | 환경 설정 비교(위), 환경 64개로 4회 짧은 학습 |
| 발사 방식 | 팔 B 의 학습이 끝나면 대기열이 자동으로 발사한다. 팔 D 는 이 런이 끝난 뒤 발사한다 |
| 발사 시각 (예상) | 2026-10-17 07:55 경 (팔 B 예상 완주 07:52 뒤 자동). 실제 시각은 대기열 기록 `analysis/out/gain_queue_cd.log` 에 있다 |
| 예상 완주 | 2026-10-17 19:30 경 (반복 3.4~3.5 초 × 12,000 회, 약 11.5 시간) |

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
| crank_A | RS03 | 60 | 2.7 | 60 |
| crank_B | RS03 | 60 | 2.7 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-10-17_07-45-38_rough_gain_crank60_gaindr_v11a`)

| 관절 그룹 | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] | effort 한계 [N·m] | 무부하 속도 [rad/s] | 로터 관성 armature [kg·m²] | 쿨롱 마찰 [N·m] | 점성 [N·m·s/rad] | T-N 곡선 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| crank_A, crank_B | RS03 | 60 | 2.7 | 60 (stall 59.7) | 20.94 | 0.0638 | 0.251 | 0.0252 | 실측 37점 (`PYG_TN=1`) |
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

## §2 이후 (완주 후)

(완주 후)

## 중간 점검 — 4,000 회 (2026-10-17 11:49, 직전 100 회 평균, 무작위화 램프 0.325)

| 지표 | 기준선 | A 런 (크랭크 60) | 이 런 (A + 게인 무작위화) |
|---|--:|--:|--:|
| 평균 보상 | 71.9 | 77.1 | 72.9 |
| 회전 속도 추종 보상 | 0.930 | 0.940 | **0.871 (A 대비 −7 %)** |
| 직선 속도 추종 보상 | 1.068 | 1.114 | 1.083 |
| 낮은 몸통 종료 | 1.63 | 1.42 | 1.69 |
| 넘어짐 종료 | 0.003 | 0.008 | 0.004 |

- 회전 속도 추종 보상이 A 런보다 7 % 낮다. 묶음 시험 런도 같은 시점에 −3 % 였다 (기준선 대비).
- 무작위화가 아직 33 % 램프이다. 판단은 8,000 회 점검과 완주 측정에서 한다. 중단 조건은 없다.
