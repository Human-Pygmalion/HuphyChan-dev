# LegOnly AB 미러축 버그픽스 검증 스모크 `legonly_ab_sideaware_smoke` (2026-09-03)

> **한 줄 판정:** v30 모델의 좌우 미러 축과 mjlab 단일 정규식 설정이 충돌해 왼쪽 무릎의
> 사용 가능 창이 0°였던 버그([[../reward_research/2026-09-03_stiff_knee_root_cause]])를
> 설정/코드 레이어에서 고친 뒤, 같은 오케스트레이션이 여전히 완주하는지와 **양 무릎이 실제로
> 명령을 받는지**를 확인하는 인프라 시험이다. **보행 성능을 판정하는 런이 아니다.**

| 항목 | 값 |
|---|---|
| 상태 | ✅ **완주 — 인프라 PASS + 무릎 액션창 열림 확인** (P1 iter 200 settled, P2 iter 399 DONE, 17분 01초, 크래시 0). 보행 성능 판정 대상 아님 |
| 대상 버그 | 모델-설정 좌우 부호 불일치 3건 (default pose / action clip / 폐루프 발목 bent 키프레임) |
| 로봇 | `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml` (**모델 무수정**) |
| 모델 질량 | 23.630 kg (1 BW = 231.8 N) |
| 질량 DR | `mass_dr_legonly_fastener50_prototype-tempmass.json` (본런 `legonly_ab_v1`과 동일 = fastener50 권위본) |
| 직전 스모크 | [[2026-09-03_legonly_ab_smoke_test]] (동일 오케스트레이션, 구 DR JSON, 버그 잔존 상태에서 완주) |
| 중단된 본런 | [[2026-09-03_legonly_ab_v1]] — 이 버그로 iter 5,700에서 보수적 중단 |
| 정량 근거 | [[2026-09-03_legonly_gait_kinematics]] (L_knee qtarget 진폭 0.00°, 스톱 상시 21.8 N·m) |

---

## §1 재현 조건

### §1a 실행 명령

```bash
cd mujoco-sim/mjlab
nohup .venv/bin/python3 analysis/run_v2_scratch.py --smoke \
  --run legonly_ab_sideaware_smoke --ankle AB --logger tensorboard \
  --env PYG_MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix \
  --env PYG_MASS_DR_JSON=<repo>/tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json \
  > analysis/out/legonly_ab_sideaware_smoke.out 2>&1 &
```

- 스모크 규격은 런처 기본값 그대로: `num_envs=1024`, P1 cap 400 iter, P2 = ramp 120 + digest 80
  = 200 iter, 게이트 완화(`min_dwell 20 / max_dwell 60 / window 20 / fell_max 1.0 /
  err_ratio 100`), `settle_hold 3`.
- 직전 스모크와 다른 점은 **두 가지뿐**: (1) 이번 픽스가 들어간 설정 코드, (2) 질량 DR JSON을
  본런과 같은 fastener50 권위본으로 교체(직전 스모크는 구 round4 서브셋).
- 권위 원장: `analysis/out/v2_scratch_legonly_ab_sideaware_smoke.json`.

### §1b 이번 런에서 바뀐 것 (버그픽스, 리워드 변경 0건)

리워드 가중치·항 구성은 `legonly_ab_v1`과 **완전히 동일**하다. 바뀐 것은 부호 유도 방식뿐이다.

| # | 위치 | 이전 | 이후 |
|---|---|---|---|
| 1 | `pygmalion_constants._bent_joint_pos` | `{".*_hip_pitch_joint": -0.175, ".*_knee_joint": -0.35}` 단일 정규식 | `signed_pose({"hip_pitch": 0.175, "knee": 0.35})` — **굽힘 크기**만 적고 부호는 각 관절의 MJCF range 장축 방향에서 유도 |
| 2 | `env_cfgs.py` `PYG_SAFE_TARGET_CLIP` | 손으로 적은 7행 정규식 표(무릎 `(-114°, -6°)` 등) | `safe_target_clip()` — 관절별로 자기 range의 중심 ±90 % 로 계산 |
| 3 | `pygmalion_constants._reexpress_loop_pose` (신규) | v3 기하로 푼 bent 키프레임 크랭크/로드 각을 v30에 그대로 대입 | 기준 모델과 축을 비교해 축이 뒤집힌 힌지의 각도만 부호 반전 |
| 4 | `assert_unmirrored()` (신규) | — | 단일 정규식으로 남겨둔 관절(`ankle_pitch`)의 좌우 range가 갈라지면 **import 시점에 실패** |
| 5 | `analysis/preflight_action_window.py` (신규) + `run_v2_scratch.py` 훅 | — | 발사 전 전 액추에이터 관절의 `default ∈ range` 와 `명령대역 ∩ range ≥ max(15°, range의 30 %)` 검사, 실패 시 발사 거부 |

### §1c 프리플라이트 게이트: 수정 전 FAIL / 수정 후 PASS

동일 명령(`analysis/preflight_action_window.py`), 동일 환경변수, 코드만 교체.

**수정 전 (= `legonly_ab_v1`이 돌던 설정):**

| joint | range [°] | clip [°] | 사용 가능 창 [°] | 폭 | 필요 | default | 판정 |
|---|---|---|---|---|---|---|---|
| L_knee_joint | [0.00, 120.00] | [−114.00, −6.00] | **없음** | **0.00** | 36.00 | −20.05 | **FAIL** default-outside-range + window-too-narrow |
| R_hip_pitch_joint | [−25.00, 120.00] | [−112.75, 17.75] | [−25.00, 17.75] | **42.75** | 43.50 | −10.03 | **FAIL** window-too-narrow |
| L_hip_roll_joint | [−25.00, 85.00] | [−79.50, 19.50] | [−25.00, 19.50] | 44.50 | 33.00 | 0.00 | ok (통과하지만 오른쪽의 45 %) |
| (나머지 9개) | — | — | — | — | — | — | ok |

→ `[preflight] FAIL: 2 joint(s) cannot be commanded` (exit 1), 실행시간 약 1 초.

**수정 후:**

| joint | range [°] | clip [°] | 사용 가능 창 [°] | 폭 | 필요 | default | 판정 |
|---|---|---|---|---|---|---|---|
| L_knee_joint | [0.00, 120.00] | [6.00, 114.00] | [6.00, 114.00] | **108.00** | 36.00 | **+20.05** | ok |
| R_knee_joint | [−120.00, 0.00] | [−114.00, −6.00] | [−114.00, −6.00] | 108.00 | 36.00 | −20.05 | ok |
| L_hip_pitch_joint | [−120.00, 25.00] | [−112.75, 17.75] | [−112.75, 17.75] | 130.50 | 43.50 | −10.03 | ok |
| R_hip_pitch_joint | [−25.00, 120.00] | [−17.75, 112.75] | [−17.75, 112.75] | **130.50** | 43.50 | **+10.03** | ok |
| L_hip_roll_joint | [−25.00, 85.00] | [−19.50, 79.50] | [−19.50, 79.50] | **99.00** | 33.00 | 0.00 | ok |
| R_hip_roll_joint | [−85.00, 25.00] | [−79.50, 19.50] | [−79.50, 19.50] | 99.00 | 33.00 | 0.00 | ok |
| L/R_hip_yaw_joint | [−45.00, 45.00] | [−40.50, 40.50] | [−40.50, 40.50] | 81.00 | 27.00 | 0.00 | ok |
| L/R_crank_A/B_joint | [−68.75, 68.75] | [−61.88, 61.88] | [−61.88, 61.88] | 123.76 | 41.25 | ±17.1 | ok |

→ `[preflight] PASS (0 warning(s))`. **좌우 12관절의 사용 가능 창이 처음으로 완전히 대칭**이다.

### §1d 폐루프 발목 초기자세 (부수 발견, 같은 유형의 세 번째 버그)

bent 키프레임의 크랭크/로드 각은 v3 기하에서 푼 값인데(`pygmalion_v3_printed_loop_bent.json`,
closure 0.001 mm), v30은 `L_crank_A`와 `R_crank_B`의 축이 뒤집혀 있다. 그대로 넣으면 매 리셋이
**로드 끝과 발 볼조인트가 어긋난 상태**에서 시작하고 equality 구속이 그 간격을 튕겨 닫는다.

| 모델 | 수정 전 최대 closure | 수정 후 최대 closure |
|---|---|---|
| `pygmalion_v3_printed_loop` | 0.001 mm | 0.001 mm (불변) |
| `pygmalion_v4_printed_loop` | 0.001 mm | 0.001 mm (불변) |
| `LegOnly_..._v30_proxyfix_loop` | **37.270 mm** (L rod A) / 36.347 mm (R rod B) | **0.001 mm** |
| `FullDoF_..._v30_proxyfix_loop` | 동일 유형 | **0.001 mm** |

수정 후 네 모델 모두 리셋 시 발목 피치 각이 **+20.60° / +20.63°** 로 동일 — 물리적으로 같은
자세임이 확인된다.

### §1e 회귀 안전성 (기존 정책 계통 불변 검증)

컴파일된 모델의 관절 순서로 런타임이 실제로 계산하는 두 벡터(`default_joint_pos` =
`resolve_expr`, action clip = `resolve_matching_names_values`)를 **픽스 전/후 코드로 각각 덤프해
수치 비교**했다.

| 모델 | 관절/액션 집합·스케일 | max Δ default | max Δ clip | 바뀐 default |
|---|---|---|---|---|
| v4 printed loop (AB) | 동일 | **0.0000e+00°** | 2.67e−04° | 없음 |
| v3 printed serial (RP) | 동일 | **0.0000e+00°** | 2.67e−04° | 없음 |
| v3 printed loop (AB) | 동일 | **0.0000e+00°** | 2.67e−04° | 없음 |
| v30 LegOnly loop (AB) | 동일 | 4.01e+01° | 1.20e+02° | L_knee, R_hip_pitch, L_crank_A, R_crank_B, L_rod_A_u1/u2, R_rod_B_u1/u2 (=고쳐야 했던 8개) |

clip의 2.67e−04° (= 4.7 µrad)는 **도→라디안 왕복 오차**다: 이전 표는 도 단위로 손으로 적은
값(`math.radians(-112.75)`)이고 새 유도는 MJCF의 라디안을 직접 읽는다. 인코더 분해능보다 4자리
아래이며, 회귀 검사(`--legacy-equivalence`)가 이 편차의 최대값을 항상 출력한다.

<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->

## 1b. 이 run의 Reward & Gains (config에서 파싱 — 재현용)

**Reward 항목** (weight·왜·어떻게 — '어떻게' 칸의 수치는 이 런의 `params/env.yaml` `rewards.<항>.params` 값):

| reward | weight | 왜 | 어떻게 |
|---|--:|---|---|
| foot_clearance | **-2** | 스윙발 지면 이격(발끌림 방지) | Σ_발 \|h−0.1 m\|·‖v_발,xy‖; 명령>0.05 |
| stance_knee_extension | **-2** | 입각 중 과도한 웅크림 억제 | Σ_접지발(힘>7† N) relu(\|q_무릎\|−25°)² [rad²]; 명령>0.05 |
| track_angular_velocity | **+2** | 명령 회전속도 추종 | exp(-((ω_cmd,z−ω_z)²+‖ω_xy‖²)/σ²), σ=0.707 rad/s |
| track_linear_velocity | **+2** | 명령 전진/측방 속도 추종 | exp(-(‖v_cmd,xy−v_xy‖²+v_z²)/σ²), σ=0.866 m/s |
| air_time | **+1** | 체공시간 보상(질질끌기 억제) | 0.05<체공시간<0.5 s 인 발 수; 명령>0.5 |
| dof_pos_limits | **-1** | 관절범위 한계 접근 벌점 | Σ 소프트 한계 밖 초과량 [rad] |
| pose | **+1** | 기본 관절자세 정규화(기괴자세 억제) | exp(-mean((q−q₀)²/σ²)) [rad]; σ 서기(명령<0.05): 전 관절 0.05 / 걷기: hip_pitch 0.3·hip_roll 0.4·hip_yaw 0.15·knee 1.2·ankle_pitch 0.25·ankle_roll 0.1 / 달리기(명령≥1.5): hip_pitch 0.5·hip_roll 0.2·hip_yaw 0.2·knee 1.5·ankle_pitch 0.35·ankle_roll 0.15 |
| self_collisions | **-1** | 자기충돌 벌점 | 자기충돌 힘>10 N 인 부분스텝 수 |
| stand_still_penalty | **-1** | 이동 명령을 무시하고 서 버리는 정지 방지 | \|v_cmd\|>0.3† 또는 \|ω_cmd\|>0.3† 인데 명령방향 진행<0.3†×명령이면 1 |
| track_lin_vel_progress | **+1** | 고속 명령에서 정지하는 국소 최적해 방지 | clamp(min(v·ĉ, \|v_cmd\|), 0) [m/s] (명령방향 진행 속도, 명령 크기에서 상한) |
| upright | **+1** | 몸통 직립 유지(넘어짐 방지) | exp(-‖g_xy‖²/σ²) (몸통 좌표계로 투영한 중력), σ=0.447 |
| foot_impact_velocity | **-0.5** | 착지 직전 발 하강속도 감소 | Σ_발 relu(−v_z)^2 (높이<0.1 m, 힘<7† N); 명령>0.05 |
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

† env.yaml에 값이 없어 함수 기본값으로 실행된 것 — 이 런이 실행한 코드(mjlab 커밋 36b76aeb)의 함수 정의에서 읽음.

**관절별 Kp/Kd** (position-PD, effort=관절측 peak):

| 관절 | 모터 | Kp(stiffness) | Kd(damping) | effort [N·m] |
|---|---|--:|--:|--:|
| hip_pitch | RS04 | 150 | 6 | 120 |
| hip_roll | RS04 | 150 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| knee | RS04 | 220 | 6 | 120 |
| crank_A | RS03 | 22.3 | 1.41 | 60 |
| crank_B | RS03 | 22.3 | 1.41 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-09-03_12-07-44_legonly_ab_sideaware_smoke_p1`)

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

모델 출처: 런 디렉토리 `repro/` 스냅샷 (권위) — `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml`

| 관절 | XML range [°] | soft 한계 [°] | 액션 clip [°] | 사용가능 창 [°] | default [°] | 구동 |
|---|---|---|---|--:|--:|---|
| L_hip_pitch_joint | [-120, 25] | [-112.8, 17.7] | [-112.8, 17.7] | 130.5 | -10.03 | 액션 |
| L_hip_roll_joint | [-25, 85] | [-19.5, 79.5] | [-19.5, 79.5] | 99 | 0 | 액션 |
| L/R_hip_yaw_joint | [-45, 45] | [-40.5, 40.5] | [-40.5, 40.5] | 81 | 0 | 액션 |
| L_knee_joint | [0, 120] | [6, 114] | [6, 114] | 108 | 20.05 | 액션 |
| L_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | 17.12 | 액션 |
| L_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.12 | 액션 |
| L_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.6 | 수동 |
| L_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0.15 | 수동 |
| R_hip_pitch_joint | [-25, 120] | [-17.7, 112.8] | [-17.7, 112.8] | 130.5 | 10.03 | 액션 |
| R_hip_roll_joint | [-85, 25] | [-79.5, 19.5] | [-79.5, 19.5] | 99 | 0 | 액션 |
| R_knee_joint | [-120, 0] | [-114, -6] | [-114, -6] | 108 | -20.05 | 액션 |
| R_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.14 | 액션 |
| R_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | 17.14 | 액션 |
| R_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.63 | 수동 |
| R_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0.15 | 수동 |

액션 스케일 0.25 rad/단위, 오프셋 = default (`use_default_offset`). clip이 없는 구 설정에서는 정책 목표각을 시뮬레이터의 soft 한계가 사후에 잡는다 — 창은 soft 한계 폭으로 읽는다.

**§1b-4. 이 런의 스택 플래그 (`PYG_*`)**

출처: 런 디렉토리 `repro/launch_manifest.json` (권위)

| 플래그 | 값 |
|---|---|
| `PYG_ANKLE_MODE` | AB |
| `PYG_ARM_ABD_DEG` | 15 |
| `PYG_CRITIC_DR_OBS` | 1 |
| `PYG_DR_END_ITER` | 100000001 |
| `PYG_DR_START_ITER` | 100000000 |
| `PYG_GATED_CURRICULUM` | 1 |
| `PYG_GATE_ERR_RATIO` | 100.0 |
| `PYG_GATE_FELL_MAX` | 1.0 |
| `PYG_GATE_MAX_DWELL` | 60 |
| `PYG_GATE_MIN_DWELL` | 20 |
| `PYG_GATE_MIN_EPISODES` | 32 |
| `PYG_GATE_WINDOW` | 20 |
| `PYG_INIT_BENT` | 1 |
| `PYG_INIT_MID` | 1 |
| `PYG_KNEE_EXT` | 1 |
| `PYG_KNEE_EXT_DEG` | 25 |
| `PYG_KNEE_EXT_W` | 2.0 |
| `PYG_MASS_DR_JSON` | `mass_dr_legonly_fastener50_prototype-tempmass.json` |
| `PYG_MODEL_TAG` | LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix |
| `PYG_MOTOR_MEAS` | 1 |
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_SOFT_LANDING` | 1 |
| `PYG_SOFT_LANDING_MODE` | half |
| `PYG_STUDENT_TEACHER` | 1 |
| `PYG_TN` | 1 |
| `PYG_V2` | 1 |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

**P2 (`2026-09-03_12-14-12_legonly_ab_sideaware_smoke_p2`)**: 액추에이터 게인·한계·액션 clip이 위 P1 표와 **동일** (env.yaml 대조). 달라지는 것은 도메인 랜덤화·push 등 학습 조건뿐이다.

<!-- SPEC-TABLES:END -->

## §2 결과

### §2a 오케스트레이션 (인프라 판정)

**✅ 완주.** 12:07:24 launch → 12:24:25 ALL PHASES DONE, **17분 1초**, 크래시 0건.

| 판정 항목 | 결과 |
|---|---|
| 프리플라이트 게이트 | **PASS**(12관절 전부 ok, 경고 0). 발사 로그에 표가 그대로 기록됨 |
| P1 | 게이트 top stage(4) 도달 → **iter 154 settle**(err_steady 0.6256 vs baseline 1.0760, streak 15/3, fell 0.0000) → iter 200 종료. stage 0→1만 `FORCED`(MAX-DWELL, iter 60) — zero-command 워밍업이라 err가 NaN인 **스모크 정상 경로**(직전 스모크와 동일) |
| P2 | P1 실측 종료 iter 200에서 `PYG_DR_START_ITER=200`/`END=320` 계산 → full-resume, `dr_factor` 0→**1.000**(iter 324) 도달 후 iter 399까지 유지 |
| 질량 DR (fastener50 권위본) | 바디명 해석 실패 없음, pseudo-inertia 이벤트 7종 정상 등록(`inertial_dr_pelvis/hip_pitch_link/hip_roll_link/thigh/shin/ankle_pitch_link/foot`) |
| 낙상 | 전 구간 `fell_over` **0.0000** |

**최종 지표(iter 399)**: reward 21.37(50avg 21.5), ep_len 500.6, `error_vel_xy` 0.797,
`error_vel_xy_steady` 0.659, `error_vel_yaw` 1.375, `thermal_effort_mean` 2.73,
`stance_knee_deg` 8.11°, `foot_impact_vel_max` 1.99, `dr_factor` 1.000.

> ⚠ 이 숫자들은 **성능 판정이 아니다**(1024 env, 399 iter, 단일 시드). 프로젝트 판정 규칙은
> 평가기 32-ep 통계 또는 200 Hz 멀티환경 프로브다.

### §2b 무릎 액션창 열림 확인 (이 스모크의 본론)

`analysis/gait_kinematics_probe.py`를 P2 체크포인트(`model_399.pt`)에 돌려 **명령 대역이 실제로
열렸는지**만 확인했다. 노미널 로봇(DR 이벤트 7종 제거), 17 s 기록 중 앞 2 s 과도구간 제외 =
15 s 분석, 50 Hz, num_envs=1, CPU. 원자료
`analysis/out/legonly_ab_sideaware_smoke_399_vx{0.6,1.2}.npz`.

| 무릎 | 조건 | qtarget 진폭 | qtarget p5..p95 [°] | 클립 경계 고착률 | q 사용 ROM | \|τ\| RMS [N·m] |
|---|---|---|---|---|---|---|
| **L_knee** (전, v1 model_5600) | 0.6 / 1.2 | **0.00° / 0.00°** | [−6.00, −6.00] (도달불가) | 100 % / 100 % | 0.21° / 0.37° | 21.79 / 21.87 |
| **L_knee** (후, smoke model_399) | 0.6 / 1.2 | **27.45° / 32.84°** | [6.00, 31.18] / [6.00, 36.93] | 58.5 % / 54.8 % | **30.14° / 37.21°** | **6.36 / 10.18** |
| **R_knee** (전) | 0.6 / 1.2 | 0.00° / 0.00° | [−6.00, −6.00] | 100 % / 100 % | 8.77° / 11.29° | 12.25 / 13.13 |
| **R_knee** (후) | 0.6 / 1.2 | 3.95° / 5.97° | [−7.17, −6.00] / [−6.53, −6.00] | 92.7 % / 93.5 % | 9.16° / 10.02° | 9.67 / 9.77 |

**판정: 통과.** 기준(양 무릎 qtarget 진폭 > 0 **AND** q 사용 ROM > 5°)을 두 속도 모두 충족한다.
왼무릎의 **상시 하드스톱 밀기(21.8 N·m 상수 토크)가 사라졌고**(6.4~10.2 N·m, 그것도 이제 변동),
목표각이 15초 내내 상수였던 것이 27~33° 진폭으로 바뀌었다.

⚠ **정직한 단서 두 가지**:
1. 오른무릎은 여전히 **명령의 93 %를 클립 상한(−6°, 최대 신전)에 붙여** 둔다. 이건 액션창
   버그가 아니라 근본원인 노트 §1의 **2순위(swing 무릎을 요구하는 리워드 항이 0개)** 그대로다.
   이번 픽스는 그걸 고치지 않았고(고칠 계획도 아니었다), 판단은 v2 본런 완주 후로 미룬다.
2. 좌우가 아직 비대칭이다(L 30~37° vs R 9~10°). 다만 **399 iter 정책의 비대칭을 5600 iter
   정책과 비교하는 것 자체가 like-for-like가 아니며**, 이 표는 "창이 열렸는가"만 답한다.

### §3b 영상

![[accum_legonly_ab_sideaware_smoke_p2.mp4]]

학습경과 accumulate 영상(런처 자동 생성, 2클립). **실시간 검증**: 1,000 스텝 × 50 Hz = 20.0 s
시뮬레이션, 파일 604 frames / 30 fps = **20.13 s** (`ffprobe`) → fps = rate/downsample =
50/1.667 = 30, 배속 없음.

최종정책 loadviz 시연 영상은 **만들지 않았다**: 399 iter 스모크 정책의 부하 시각화는 판정
가치가 없고, 직전 스모크([[2026-09-03_legonly_ab_smoke_test]])도 같은 이유로 생략했다.
본런 `legonly_ab_v2`는 두 영상 모두 필수다.

## §3 다음 단계

- 스모크 PASS + 양 무릎 창 열림 확인 시: 본런 `legonly_ab_v2`(16384 env, 32k, fastener50 DR,
  `--vy-stages`) 발사 판단은 **계획자 몫**. 이 노트는 발사하지 않는다.
- swing 무릎 리워드 항(근본원인 노트 §2의 2순위, Booster T1 knee-height)은 **별도 트랙**이며
  이번 픽스에 번들하지 않는다 — v2에서 자연 해소될 수 있고, 2026-08-24 규칙상 신규 항은
  +800 iter warm-start 단독 A/B로 시험한다.

## §2c 학습 중 리뷰 (게이트마다 스냅샷, docs/27 체크리스트)

![progress](mujoco/assets/legonly_ab_sideaware_smoke_p1_progress.png)

| 시각 | iter | reward | ep_len | noise σ | value loss | entropy | surrogate / LR | fell / low_base | err_vel xy / yaw | dr_factor / vx_max | thermal | 판정(docs/27) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09-03 12:13 | 210 | 12.9 (50avg 9.3) | 306 | 0.417 | 0.0719 | 6.31 | -0.0079 / 1.1e-04 | 0.000 / 2.952 | 0.533 / 0.784 | 0.00 / 2.5 | 1.93 | P1 phase-end: review before P2 |
| 09-03 12:24 | 399 | 21.4 (50avg 21.5) | 501 | 0.382 | 0.126 | 5.14 | -0.0093 / 1.1e-04 | 0.000 / 2.000 | 0.797 / 1.375 | 1.00 / 2.5 | 2.73 | P2 phase-end: final smoke/training health review |
