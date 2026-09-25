# legonly_ab_v1 — LegOnly(12-DOF, 상체 없음) 본학습 (2026-09-03~) 〔진행 중 — 뼈대〕

> *한 줄*: 상체를 완전히 제거한 12-DOF 다리전용 모델의 첫 본학습. v2s1(2026-08-28 완주,
> `flat-2.5max` 확정 landing recipe + v4 스택 + vy stages + gated curriculum + critic DR 82ch)
> 레시피를 그대로 상속하고 로봇 모델만 LegOnly로 교체한다. 오케스트레이션은
> [[2026-09-03_legonly_ab_smoke_test]]에서 사전 검증(인프라 PASS) 완료.

| | |
|---|---|
| 로봇 | `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml` (23.630 kg, nbody=22, njnt=25, AB 폐루프 발목, waist-yaw 액추에이터 질량만 유지·DOF 없음) |
| 질량 DR | `mass_dr_legonly_fastener50_prototype-tempmass.json` (body 귀속 수정 + docs/114 §5 나사 50% 완전성 방법론 합본의 leg-only 서브셋 — 스모크가 쓴 구버전 대비 갱신, docs/117 §2) |
| 스택 | v2s1과 동일: 착지 레시피(INIT_MID·KNEE_EXT 2.0@25°·SOFT_LANDING_MODE=half) + vy 스테이지 + 게이트 커리큘럼 + critic DR 관측(82ch) + P2 entropy 어닐링(0.01→0.002) |
| env | 16384 (v2s1과 동일 풀스케일) |
| 로거 | wandb — launch 전 `wandb.Api()` 연결 확인 + launch 직후 실제 sync 확인:
[nm2hk12i](https://wandb.ai/dongyub39-snu/pygmalion/runs/nm2hk12i) (P1, project `pygmalion`, entity `dongyub39-snu`) |
| 계보 | 레시피 [[103_v2_training_plan]] §4a · 로봇 모델 [[117_model_finalization_and_oneleg_training_plan]] §0/§5 · 오케스트레이션 사전검증 [[2026-09-03_legonly_ab_smoke_test]] |

## §1a 실행 명령

```bash
bash analysis/run_v2_scratch.sh \
  --run legonly_ab_v1 --ankle AB --vy-stages \
  --env PYG_MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix \
  --env PYG_MASS_DR_JSON=/home/syaro/MikuchanRemote/Human-Pygmalion/tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json
```

정확한 인자·P1/P2 환경 변수·승급 기록은 `analysis/out/v2_scratch_legonly_ab_v1.json`이 권위
원장이다. `gate_watch.sh`로 게이트/하트비트 백그라운드 감시(docs 하드룰).

## §1b LegOnly 특유 리스크 (스모크에서 확인됨, 재확인 불필요)

- `PYG_MASS_DR_JSON`을 leg-only 서브셋으로 명시하지 않으면 P2에서 존재하지 않는
  torso/shoulder_pitch_link/arm 바디명 참조로 크래시 예상 — 이번 실행은 위 명령대로 서브셋을
  명시했으므로 안전. (스모크가 쓴 서브셋은 나사50% 미반영 구버전이었고, 이번 본학습은 §2에서
  확정한 fastener50 합본 서브셋을 쓴다 — 유일한 차이.)
- Critic 82D, Actor 45D/12-DOF는 스모크에서 실측 확인됨(v2s1과 동일 폭).

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

† env.yaml에 값이 없어 함수 기본값으로 실행된 것 — 이 런이 실행한 코드(mjlab 커밋 8449237f)의 함수 정의에서 읽음.

**관절별 Kp/Kd** (position-PD, effort=관절측 peak):

| 관절 | 모터 | Kp(stiffness) | Kd(damping) | effort [N·m] |
|---|---|--:|--:|--:|
| hip_pitch | RS04 | 150 | 6 | 120 |
| hip_roll | RS04 | 150 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| knee | RS04 | 220 | 6 | 120 |
| crank_A | RS03 | 22.3 | 1.41 | 60 |
| crank_B | RS03 | 22.3 | 1.41 | 60 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-09-03_02-47-35_legonly_ab_v1_p1`)

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
| L_hip_pitch_joint | [-120, 25] | [-112.8, 17.7] | [-112.8, 17.8] | 130.5 | -10.03 | 액션 |
| L_hip_roll_joint | [-25, 85] | [-19.5, 79.5] | [-79.5, 19.5] | 99 | 0 | 액션 |
| L/R_hip_yaw_joint | [-45, 45] | [-40.5, 40.5] | [-40.5, 40.5] | 81 | 0 | 액션 |
| L_knee_joint | [0, 120] | [6, 114] | [-114, -6] | 108 | -20.05 | 액션 |
| L_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.12 | 액션 |
| L_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.12 | 액션 |
| L_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.6 | 수동 |
| L_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0.15 | 수동 |
| R_hip_pitch_joint | [-25, 120] | [-17.7, 112.8] | [-112.8, 17.8] | 130.5 | -10.03 | 액션 |
| R_hip_roll_joint | [-85, 25] | [-79.5, 19.5] | [-79.5, 19.5] | 99 | 0 | 액션 |
| R_knee_joint | [-120, 0] | [-114, -6] | [-114, -6] | 108 | -20.05 | 액션 |
| R_crank_A_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.14 | 액션 |
| R_crank_B_joint | [-68.8, 68.8] | [-61.9, 61.9] | [-61.9, 61.9] | 123.8 | -17.14 | 액션 |
| R_ankle_pitch_joint | [-50, 30] | [-46, 26] | — (수동) | — | 20.63 | 수동 |
| R_ankle_roll_joint | [-20, 20] | [-18, 18] | — (수동) | — | 0.15 | 수동 |

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
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_SOFT_LANDING` | 1 |
| `PYG_SOFT_LANDING_MODE` | half |
| `PYG_STUDENT_TEACHER` | 1 |
| `PYG_TN` | 1 |
| `PYG_V2` | 1 |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

**P2 (`2026-09-03_08-44-28_legonly_ab_v1_p2`)**: 액추에이터 게인·한계·액션 clip이 위 P1 표와 **동일** (env.yaml 대조). 달라지는 것은 도메인 랜덤화·push 등 학습 조건뿐이다.

<!-- SPEC-TABLES:END -->

## §2 이하 — 완주 후 측정으로 채운다

측정은 v2s1과 동일하게 fc/fcp(15s dwell, 학습박스 전체 커버) + 200Hz 프로브 + §7 모터활용.
LegOnly는 상체 질량이 없어 GRF/토크 절대값이 v2s1과 직접 비교 불가(체중 23.6 vs 31.3 kg) —
비교는 BW/rated 비 등 정규화된 지표로.

## §R 참조
[[103_v2_training_plan]] · [[117_model_finalization_and_oneleg_training_plan]] ·
[[2026-09-03_legonly_ab_smoke_test]] · [[110_prototype_tempmass_student_teacher_report]]

## §2c 학습 중 리뷰 (게이트마다 스냅샷, docs/27 체크리스트)

![progress](mujoco/assets/legonly_ab_v1_p1_progress.png)

| 시각 | iter | reward | ep_len | noise σ | value loss | entropy | surrogate / LR | fell / low_base | err_vel xy / yaw | dr_factor / vx_max | thermal | 판정(docs/27) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09-03 03:47 | 707 | 64.5 (50avg 64.8) | 983 | 0.317 | 0.0437 | -0.77 | 0.0012 / 1.1e-04 | 0.000 / 0.458 | 1.123 / 0.659 | 0.00 / 0.8 | 1.18 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 04:47 | 1413 | 58.1 (50avg 59.6) | 937 | 0.338 | 0.0536 | -0.10 | 0.0003 / 1.7e-04 | 0.000 / 1.750 | 1.496 / 0.575 | 0.00 / 1.2 | 1.33 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 05:47 | 2185 | 90.7 (50avg 90.5) | 991 | 0.286 | 0.0334 | -1.94 | -0.0013 / 7.6e-05 | 0.000 / 0.667 | 0.992 / 0.671 | 0.00 / 1.6 | 2.08 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 06:47 | 2975 | 90.3 (50avg 90.5) | 981 | 0.304 | 0.0487 | -1.30 | -0.0014 / 7.6e-05 | 0.000 / 0.333 | 1.120 / 0.683 | 0.00 / 2.0 | 2.19 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 07:47 | 3765 | 91.1 (50avg 89.8) | 1000 | 0.315 | 0.0438 | -0.89 | -0.0013 / 1.1e-04 | 0.000 / 0.083 | 1.327 / 0.723 | 0.00 / 2.5 | 2.22 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 08:44 | 4508 | 90.3 (50avg 90.6) | 985 | 0.321 | 0.0441 | -0.72 | -0.0011 / 7.6e-05 | 0.000 / 0.250 | 1.194 / 0.710 | 0.00 / 2.5 | 2.19 | P1 phase-end: review before P2 |
| 09-03 08:47 | 4531 | 69.0 (50avg 35.7) | 762 | 0.323 | 0.0453 | -0.60 | 0.0010 / 7.6e-05 | 0.000 / 0.333 | 0.802 / 0.551 | 0.00 / 2.5 | 2.27 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 09:47 | 5058 | 91.6 (50avg 91.5) | 1000 | 0.319 | 0.0434 | -0.86 | -0.0011 / 1.1e-04 | 0.000 / 0.250 | 1.218 / 0.719 | 0.06 / 2.5 | 2.23 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 10:47 | 5579 | 92.3 (50avg 93.2) | 994 | 0.311 | 0.0468 | -1.13 | 0.0001 / 7.6e-05 | 0.000 / 0.167 | 1.203 / 0.720 | 0.11 / 2.5 | 2.29 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 11:47 | 5978 | 93.7 (50avg 93.2) | 993 | 0.309 | 0.0459 | -1.29 | 0.0006 / 1.1e-04 | 0.000 / 0.375 | 1.161 / 0.706 | 0.15 / 2.5 | 2.30 | (자동 스냅샷, 판정은 게이트 리뷰에서) |
| 09-03 12:47 | 5978 | 93.7 (50avg 93.2) | 993 | 0.309 | 0.0459 | -1.29 | 0.0006 / 1.1e-04 | 0.000 / 0.375 | 1.161 / 0.706 | 0.15 / 2.5 | 2.30 | (자동 스냅샷, 판정은 게이트 리뷰에서) |

**게이트 판정 (09-03 11:00, iter ~5.6k, 세션 복귀 후 첫 리뷰)**: **계속(진행)**. P1→P2 전이
dip(4531: reward 69/50avg 35.7, ep_len 762)은 resume 직후 1시간 내 완전 회복(5058: 91.6/1000).
낙상 0.000 유지, ep_len 포화(≈1000), noise σ 0.31 안정, DR 램프 정상 진행(0.00→0.11), vx_max 2.5
도달. 보수적 중단 사유(docs/27) 없음. ⚠질적 플래그: 사용자 라이브 관찰(09-02 23:49) stiff-knee /
AB 미활용 / toe-off 부재 — 별도 운동학 정량화(2026-09-03_legonly_gait_kinematics) + 리워드
연구노트 진행 중. 리워드 개입은 연구노트 확정 후 결정(현 런은 계속 학습).

**★최종 판정 (09-03 12:05, iter ~5.7k): 보수적 중단 — 모델-설정 부호 불일치로 런 무효.**
운동학 정량화([[2026-09-03_legonly_gait_kinematics]], model_5600)가 원인을 확정: v30 MJCF는
L/R 축이 미러인데 설정은 단일 default/clip을 양쪽에 적용 → **L_knee 사용가능 창 0°**(명령대역
−114~−6° vs 기계범위 0~+120°), default −20°가 관절범위 밖, 스톱에 상시 21.8 N·m. R_hip_pitch
43/145°, L_hip_roll 44/110°로 동반 축소. 무릎 qtarget 100% 클립 고정(정책 문제 아님, 구조 문제),
하중 데이터 오염으로 측정 목적 상실 → 잔여 13k iter 중단이 옳음. 근본원인·조치:
[[../reward_research/2026-09-03_stiff_knee_root_cause]]. 후속 = 설정 side-aware 수정 +
프리플라이트 게이트 + `legonly_ab_v2` 재발사. 와치독 엔트리 disabled(자동부활 차단),
launcher(754406)·trainer(1325381) PID 지정 종료 12:04.
