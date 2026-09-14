# legonly_ab_v6_standing — 정지 구간을 3배로 학습시키면 서 있는 자세가 잡히는가 (2026-09-14)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> 계보: [[2026-09-13_legonly_ab_v5_heading]] 의 **유일 변인 후속**. 근거는 그 노트 §13·§13-1.

## §0 왜 이 학습을 하는가

사용자가 화면을 보고 지적했다: *"가만히 서 있는 자세가 이상해."* 실제로 이상하다.

![[v5_standing_pose_vs_target.png]]

왼쪽이 정지 자세 추종 항이 노리는 목표(다리 평행, 발 정면), 오른쪽이 실제다 — 발끝이 벌어지고
발목이 비틀리고 다리가 꼬여 있다.

### 원인: 정지 자세 추종 항이 죽어 있다

`pose` 항(가중치 +1.0)은 `exp(−평균(오차²/허용²))`로 채점하고, 목표는 `default_joint_pos`
(= 리셋 웅크림 = 액션 0 자세)다. **별도의 정지용 목표 자세는 없다.** 허용 오차는 명령 크기에
따라 세 규모로 갈리는데:

| 규모 | 엉덩이앞뒤 | 엉덩이좌우 | 엉덩이회전 | 무릎 | 발목숙임 | 발목기울임 |
|---|--:|--:|--:|--:|--:|--:|
| **정지** | **2.86°** | **2.86°** | **2.86°** | **2.86°** | **2.86°** | **2.86°** |
| 걷기 | 17.2° | 22.9° | 8.6° | 68.8° | 14.3° | 5.7° |
| 빨리 | 28.6° | 11.5° | 11.5° | 85.9° | 20.1° | 8.6° |

걷기·빨리는 관절별로 맞춰져 있는데 **정지만 `.*: 0.05` 한 줄**이다. 명령 0에서 실제 자세는 허용의
**2~5.6배** 벗어나 있어 항이 `exp(−8.7) = 1.7e−04`, 즉 **가중치 1.0짜리가 0을 낸다**(양수 항 합
+6.33/s의 16 %). 그 근처에서 **기울기도 1.7e−04**라 조금 나아져도 돌아오는 게 없어 정책이 통째로
무시한다. 게다가 명령 0 구간은 학습의 **10 %**(`rel_standing_envs`)뿐이라 버려도 손해가 작다.

### 목표 자세 자체는 멀쩡하다 (2026-09-14 확인)

허용 오차가 잘못된 것이지 목표가 무리한 게 아니다. 정적 안정성을 재니:

| 자세 | 지지면 크기 | 무게중심 여유 | 판정 |
|---|---|--:|---|
| 목표(리셋 웅크림) | 앞뒤 256 × 좌우 347 mm | **+100.3 mm** | 지지면 안 |
| 정책이 실제로 쓰는 자세 | 앞뒤 425 × 좌우 243 mm | +99.4 mm | 지지면 안 |

**여유가 같다.** 정책이 어쩔 수 없이 다리를 벌린 게 아니라, 앞뒤로 어긋선 자세를 고른 것뿐이다.

## §1 설계 — v5와 유일 변인만 다르다

**변인: `PYG_REL_STANDING` 0.1 → 0.3** (설정 숫자 하나)

명령 0 구간을 **3배** 학습시킨다. 노림수는 자세 추종 항을 직접 되살리는 게 아니라, 그 구간에서
**기울기가 살아 있는 항들에 표본을 더 주는 것**이다 — 전후·좌우 속도 추종과 회전 속도 추종은
로봇이 흘러가고 제자리걸음을 하는 동안 0.95 근처라 건강한 기울기를 갖고 있다. 그 항들이
흘러감·제자리걸음을 줄이면 자세가 목표 쪽으로 돌아오고, 그때 자세 추종 항의 기울기도 되살아난다.

### `walking_threshold`는 건드리지 않는다 — 역효과를 실측했다

또 다른 후보는 `walking_threshold`를 코드 기본값 0.5로 되돌려 느린 구간도 정지 규모로 묶는 것이었다.
**재 보니 역효과다.** 명령 합 0.05~0.5 구간(전체의 5.0 %)에 정지 규모를 적용하면:

| 그 구간의 자세 추종 보상 | |
|---|--:|
| 지금(걷기 규모) | **0.7045** |
| 정지 규모로 바꾸면 | **0.0034** |

잘 작동하던 항까지 죽여 **죽은 구간을 넓히는** 셈이다. 채택하지 않는다.

### 보상 파라미터는 건드리지 않는다

정지 규모를 관절별로 다시 주는 것(엉덩이회전은 좁게, 무릎·크랭크는 넓게)이 더 직접적이지만,
**보상 파라미터 변경이라 규칙상 연구노트가 먼저**다. 이번에는 과제 분포만 바꾸는 쪽을 먼저 시험하고,
그것으로 부족하면 그때 정식 연구노트를 쓴다.

**v5와 같게 유지하는 것**: `PYG_REL_HEADING=0.8`을 포함해 로봇 모델·모터 값·세기·보상 항과 가중치
전부·커리큘럼·도메인 무작위화·시드 42·환경 수 16384·반복 수 일정.

## §1a 실행 명령

```
cd mujoco-sim/mjlab
bash analysis/run_v2_scratch.sh --run legonly_ab_v6_standing --ankle AB --seed 42 --num-envs 16384 \
  --p1-max-iters 16000 --ramp-iters 10000 --digest-iters 4000 --settle-hold 10 --n-stages 5 \
  --gate-min-dwell 800 --gate-max-dwell 3000 --gate-window 100 --gate-err-ratio 1.1 --gate-fell-max 0.005 --gate-min-episodes 64 \
  --entropy-start 0.01 --entropy-end 0.002 --video-interval 8000 --video-length 500 --logger wandb --vy-stages \
  --env PYG_MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix \
  --env PYG_MASS_DR_JSON=/home/syaro/MikuchanRemote/Human-Pygmalion/tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json \
  --env PYG_REL_HEADING=0.8 \
  --env PYG_REL_STANDING=0.3
```

v5의 명령과 **마지막 한 줄만** 다르다. 발사 직후 `params/env.yaml`을 v5 것과 대조해 확인한다.

## §1a-1 발사 기록

**2026-09-14 19:37:39 발사**, 1단계 디렉터리 `2026-09-14_19-37-39_legonly_ab_v6_standing_p1`.

발사 전 검사 통과:

```
[preflight] 12 actuated joints (need window >= max(15 deg, 0.3 x range))
[preflight] leg naming: L_hip_pitch_link sits at y=+0.1237 m -> left (+y)
[preflight] reset pose: 12 equality rows, worst loop closure 0.0000 mm (limit 1 mm)
[preflight] PASS (0 warning(s))
[preflight] RAM available=16.1 GiB, GPU free=15.3 GiB, envs=16384
```

**단일 변인 실증**: v5의 `params/env.yaml`과 대조해 런 이름 줄을 빼면 **한 줄만** 다르다.

```
-    rel_standing_envs: 0.1
+    rel_standing_envs: 0.3
```

## §1c 완주 판정 기준

v5 완주본을 같은 방법으로 잰 값이 기준선이다. **정지 항목이 주 판정**이고 보행은 지키기만 하면 된다.

| 기준 | v5 | v6 목표 |
|---|--:|---|
| **정지 자세 추종 보상**(명령 0) | 1.7e−04 | **0.05 이상** (지금의 300배, exp(−3) 수준) |
| **엉덩이회전 좌우 벌어짐**(명령 0) | 왼 +14.9° / 오른 −11.8° | **양쪽 5° 이내** |
| **엉덩이회전 액션 한계 포화**(명령 0) | 왼 41.0 % / 오른 18.7 % | **10 % 이하** |
| **발 뗀 시간**(명령 0, 흐트러진 시작) | 81 % | **40 % 이하**(v4 수준 36 %) |
| 이동 거리 / 방향 변화(명령 0, 12초) | 0.361 m / 105° | 절반 이하 |
| 이미 서 있을 때 몸통 변화 폭 | 2 mm / 20초 | 유지 |
| — 보행은 지키기만 — | | |
| 엉덩이회전 좌우차(1.2 m/s) | +7.5 % | 20 % 이내 유지 |
| 걷는 시간 비율 / 속도 | 93.4 % / 1.21 m/s | 유지 |
| 넘어짐 | 0.000 | 0.000 |
| 보상 | 113.8 | −5 % 이내 |

**주의**: 명령 0 환경이 10 %→30 %로 늘면 **보상의 절대값이 내려갈 수 있다**(서 있는 환경은
속도 추종으로 벌 수 있는 점수가 적다). 그 자체는 실패가 아니다 — 위의 정지 항목으로 판정한다.

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


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-09-14_19-37-39_legonly_ab_v6_standing_p1`)

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

## §2 최종 지표

[완주 후 채움]

## §2c 학습 중 리뷰

[게이트마다 채움]

## §R 참조

- 원인 진단: [[2026-09-13_legonly_ab_v5_heading]] §13(명령 0 거동), §13-1(정지 자세 추종이 죽은 이유)
- 정지 q목표·토크(형제 세션): [[2026-09-14_v5_stand_zero_command]]
- 부모 런: [[2026-09-13_legonly_ab_v5_heading]]
- 코드: `PYG_REL_STANDING`·`PYG_REL_HEADING` (`src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py`),
  정지 자세 추종 (`src/mjlab/tasks/velocity/mdp/rewards.py` `variable_posture`)

## §2c 학습 중 리뷰 (게이트마다 스냅샷, docs/27 체크리스트)

![progress](mujoco/assets/legonly_ab_v6_standing_p1_progress.png)

| 시각 | iter | reward | ep_len | noise σ | value loss | entropy | surrogate / LR | fell / low_base | err_vel xy / yaw | dr_factor / vx_max | thermal | 판정(docs/27) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09-14 20:37 | 733 | 104.6 (50avg 103.9) | 1000 | 0.222 | 0.0281 | -4.31 | 0.0007 / 8.8e-05 | 0.000 / 0.042 | 0.406 / 0.502 | 0.00 / 0.8 | 1.12 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-14 21:37 | 1471 | 103.3 (50avg 103.3) | 1000 | 0.241 | 0.0342 | -3.13 | 0.0005 / 8.8e-05 | 0.000 / 0.083 | 0.491 / 0.568 | 0.00 / 1.2 | 1.42 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-14 22:37 | 2227 | 102.0 (50avg 101.8) | 1000 | 0.252 | 0.0323 | -2.35 | -0.0013 / 2.0e-04 | 0.000 / 0.083 | 0.604 / 0.594 | 0.00 / 1.6 | 1.63 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
