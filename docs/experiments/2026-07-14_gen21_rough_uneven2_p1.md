# gen21_rough_uneven2_p1 — rough 트랙 소생 (계단·급슬로프 제거로 fell 0.3→0.00)

> **한 줄**: rough P1 실패([[2026-07-13_gen21_rough_p1]], fell 0.3 정체)의 진단([[2026-07-14_rough_p1_blind_stairs_diagnosis]])을 2단 처방(계단 제거→슬로프 45°→17°)으로 검증. **fell iter 600서 ~0.00 수렴** = 진단 완전확증. DR-off P1(gait 형성) 완주, 다음은 v2 측정 + P2(DR+push).

## §1 가설·변인
- **단일변인**: 지형 구성 `UNEVEN_TERRAINS_CFG`(계단 0%·**슬로프 rise/run 0.3=~17°**·random_rough·wave) — vs 실패 P1의 `ROUGH_TERRAINS_CFG`(계단 40%·슬로프 45°). reward/gains/init/warm-start는 동일.
- 가설: "장님 액터(height_scan=critic-only)는 걸을 수 없는 지형(계단·급슬로프)에서만 실패한다. walkable uneven만 남기면 fell→0."

## §1b Reward & Gains
- 앵커 [[2026-07-13_gen21_bent_p2]]와 동일(Gen-2.1 번들: 상대임계 stand_still·knee_overspeed·bent init·hip_roll std 0.4). Kp/Kd·effort·speed limit 변경 없음. 지형만 변인.

<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->

**§1b-1. 리워드 가중치·관절 게인** (이 런의 `params/env.yaml` 파싱 — 위 §1b 서술의 정량 원본)

**Reward 항목** (weight·왜·어떻게 — '어떻게' 칸의 수치는 이 런의 `params/env.yaml` `rewards.<항>.params` 값):

| reward | weight | 왜 | 어떻게 |
|---|--:|---|---|
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
| thermal_effort | -0.02 | ★열분배: Σ(τ/정격)² 정규화(관절 균등화) | Σ(τ/τ_정격)², τ_정격 [N·m]: hip_pitch 40·hip_roll 40·hip_yaw 20·knee 40·ankle_pitch 20·ankle_roll 5 |
| contact_force_cap | -0.01 | ★충격 상한: 발 지면반력 역치초과분 벌점(사뿐착지) | Σ_발 min(max(‖F‖−600 N, 0), 800 N) |
| soft_landing | -1e-05 | 착지 첫접촉 충격 벌점(약) | Σ 첫 접촉 순간 ‖F‖ [N]; 명령>0.05 |
| torque_limit | -0 | 명령 토크의 한계초과 벌점 | Σ_구동기 relu(\|τ_명령\|/τ_max−0.7) |

† env.yaml에 값이 없어 함수 기본값으로 실행된 것 — 이 런이 실행한 코드(mjlab 커밋 20f10e96 + 실행 당시 미커밋 수정분)의 함수 정의에서 읽음.

**관절별 Kp/Kd** (position-PD, effort=관절측 peak):

| 관절 | 모터 | Kp(stiffness) | Kd(damping) | effort [N·m] |
|---|---|--:|--:|--:|
| hip_pitch | RS04 | 150 | 6 | 120 |
| hip_roll | RS04 | 150 | 6 | 120 |
| hip_yaw | RS03 | 150 | 6 | 60 |
| knee | RS04 | 220 | 6 | 120 |
| ankle_pitch | RS03 | 28.5 | 1.81 | 90 |
| ankle_roll | RS00 | 28.5 | 1.81 | 50 |


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-07-14_18-21-12_gen21_rough_uneven2_p1`)

| 관절 그룹 | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] | effort 한계 [N·m] | 무부하 속도 [rad/s] | 로터 관성 armature [kg·m²] | 쿨롱 마찰 [N·m] | 점성 [N·m·s/rad] | T-N 곡선 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| ankle_roll | RS00 | 28.5 | 1.81 | 50 | — | 0.0005 | — | — | 미사용 (effort_limit 상수 클램프) |
| hip_yaw | RS03 | 150 | 6 | 60 | — | 0.005 | — | — | 미사용 (effort_limit 상수 클램프) |
| ankle_pitch | RS03 | 28.5 | 1.81 | 90 | — | 0.005 | — | — | 미사용 (effort_limit 상수 클램프) |
| hip_pitch | RS04 | 150 | 6 | 120 | — | 0.007 | — | — | 미사용 (effort_limit 상수 클램프) |
| knee | RS04 | 220 | 6 | 120 | — | 0.007 | — | — | 미사용 (effort_limit 상수 클램프) |

토크는 `effort_limit`과 (있으면) 실측 T-N 곡선의 속도의존 상한 중 **작은 값**으로 클램프된다. armature/쿨롱/점성은 모터 실측값(`PYG_MOTOR_MEAS=1`)이면 실측, 아니면 카탈로그 추정치다.

**§1b-3. ROM 한계·액션 창** (모델 XML range · soft 한계 = 중심±0.5·range×0.9 (mjlab `Entity` 규약) · 액션 clip = env.yaml `actions.joint_pos.clip` · 창 = clip 폭 · default = 액션 0 자세)

**soft 한계와 액션 clip은 같은 공식이다** — `Entity.soft_joint_pos_limits`와 `pygmalion_constants.safe_target_clip()`이 둘 다 *중심 ± 0.5·range·factor*를 쓴다(각 경계에 factor를 곱하는 것이 아니다: 비대칭 관절에서 두 식이 갈린다 — knee `[0,120]`은 `[6,114]`이지 `[0,108]`이 아니다). 그래서 `PYG_SAFE_TARGET_CLIP=1`인 런에서는 두 열이 정확히 일치하고, 정책이 통과하는 클램프와 시뮬레이터가 강제하는 클램프가 하나의 계약이 된다.

모델 출처: 이 시기 `pygmalion_constants._XML_NAME` 기본 분기 — `PYG_V2`/`PYG_HIP_CANT*`/`PYG_ROLLOFF30` 미설정 시 `pygmalion.xml`. 노트의 hip_roll 하드스톱 진술(외전 −45° / 내전 +25°)과 이 파일의 range가 일치 — `pygmalion.xml`

| 관절 | XML range [°] | soft 한계 [°] | 액션 clip [°] | 사용가능 창 [°] | default [°] | 구동 |
|---|---|---|---|--:|--:|---|
| L/R_hip_pitch_joint | [-125, 30] | [-117.2, 22.3] | n/a (구 설정: clip 없음) | 139.5 | -18.33 | 액션 |
| L/R_hip_roll_joint | [-45, 25] | [-41.5, 21.5] | n/a (구 설정: clip 없음) | 63 | 0 | 액션 |
| L/R_hip_yaw_joint | [-50, 50] | [-45, 45] | n/a (구 설정: clip 없음) | 90 | 0 | 액션 |
| L/R_knee_joint | [-140, 0] | [-133, -7] | n/a (구 설정: clip 없음) | 126 | -38.39 | 액션 |
| L/R_ankle_pitch_joint | [-50, 40] | [-45.5, 35.5] | n/a (구 설정: clip 없음) | 81 | 20.63 | 액션 |
| L/R_ankle_roll_joint | [-20, 20] | [-18, 18] | n/a (구 설정: clip 없음) | 36 | 0 | 액션 |
| L/R_toe_joint | [-50, 0] | [-47.5, -2.5] | — (수동) | — | 0 | 수동 |

액션 스케일 0.25 rad/단위, 오프셋 = default (`use_default_offset`). clip이 없는 구 설정에서는 정책 목표각을 시뮬레이터의 soft 한계가 사후에 잡는다 — 창은 soft 한계 폭으로 읽는다.

**§1b-4. 이 런의 스택 플래그 (`PYG_*`)**

출처: 이 노트의 실행 명령/본문에서 추출 (런처 매니페스트 미기록 — 값 없는 항목은 노트가 이름만 언급)

| 플래그 | 값 |
|---|---|
| `PYG_DR_END_ITER` | (값 미기재) |
| `PYG_DR_START_ITER` | (값 미기재) |
| `PYG_FRESH_STEPS` | 1 |
| `PYG_HIP_CANT` | (값 미기재) |
| `PYG_INIT_BENT` | 1 |
| `PYG_MOTOR_MEAS` | 1 |
| `PYG_NO_DR` | 1 |
| `PYG_ROLLOFF30` | (값 미기재) |
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_UNEVEN` | 1 |
| `PYG_V2` | (값 미기재) |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

**P2 (`2026-07-15_00-58-24_gen21_rough_uneven2_p2`)**: 액추에이터 게인·한계·액션 clip이 위 P1 표와 **동일** (env.yaml 대조). 달라지는 것은 도메인 랜덤화·push 등 학습 조건뿐이다.

<!-- SPEC-TABLES:END -->

## §2 설정
- run: `logs/rsl_rl/pygmalion_velocity/2026-07-14_18-21-12_gen21_rough_uneven2_p1` (최종 model_11999)
- launch: `PYG_UNEVEN=1 PYG_NO_DR=1 PYG_INIT_BENT=1 PYG_FRESH_STEPS=1` + actor-only warm-start(flat 앵커 gen21_bent_p2 model_31998, `analysis/train_actor_warmstart.py`), 4096 env, 12k iter, DR-off.
- 코드: `src/mjlab/terrains/config.py` UNEVEN_TERRAINS_CFG(slope_range 0.3), `velocity_env_cfg.py` PYG_UNEVEN 토글. end-to-end 검증(토글 on→계단없음·슬로프0.3, off→계단있음, stale-pyc 아님).

## §2b 변인격리 3단 (진단의 핵심)
| run | 계단 | 슬로프 | fell 종착 | 판정 |
|---|---|---|---|---|
| gen21_rough_p1 | 40% | 45° | **0.30 영구정체** | 장님×불가지형 |
| gen21_rough_uneven_p1 | 0% | 45° | 0.32 (감소하나 둔화) | 슬로프 잔여병목 |
| **gen21_rough_uneven2_p1** | 0% | **17°** | **~0.00 (iter 600)** | ✅ walkable |
→ 각 요인 기여 분해: 계단이 주범(제거로 명확 감소), 급슬로프가 잔여(하향으로 완치).

## §12 결과·판정
- **fell_over: iter 600서 ~0.002 수렴, 종료까지 0.000–0.012 유지** (최종 0.0000). 계단런 0.3정체·uneven1 0.32와 극명 대비.
- 추종: track_linear reward 1.10–1.17·track_angular 0.99–1.04, Mean reward ~57–58. (DR-off P1 기준 건강; 절대 추종%는 v2 측정에서 확정.)
- **판정: ✅ 진단 완전확증 + rough P1 성공.** rough 트랙 소생. 다음: v2 텔레포트 측정(tile 오염 없이) → P2(DR+push).

## §11 이상징후 — 주기적 reward 스파이크 (watch-item)
- **36/12000 iter(0.3%)** 에서 reward가 −26k~−36k로 순간 폭락(그 외 정상 57–70). ~3–4k iter 주기. **fell엔 영향 0**(0.000 유지), reward 즉시 회복 → PPO가 흡수.
- 추정: uneven 지형 엣지의 드문 대형 접촉 → **캡 없는 페널티 항**(토크/가속/action_rate류)이 특정 env서 폭주. DR-off라 희소하나 **P2(DR+push)에선 빈발 가능** → P2 npz에서 어느 항이 튀는지 진단 필요(캡 추가 후보).

## ★P2 DR-ramp 버그 & 수정 (2026-07-15)
- 첫 P2(`00-58-24_p2`)에서 **dr_factor가 iter 17571에도 0.0 고정** 발견 → P2가 robust 앵커 무효(DR 미주입).
- 근본원인: `dr_levels` 윈도우가 `start_step=20000×24=480000`(P1=20k iter 하드코딩). 본 P1은 **12k+PYG_FRESH_STEPS**라 common_step_counter가 288000에서 끝 → P2가 counter 576000까지만 가 **dr=0.33에서 정체**.
- 수정: `env_cfgs.py`에 **`PYG_DR_START_ITER`/`PYG_DR_END_ITER` env override** 추가(기본 20k/32k=flat 파이프라인 보존). P2 재학습(`03-48-03_p2b`)을 `12000/24000`로 정렬 → **dr가 P2b 시작(iter 12068)부터 램프**(0.0057→...→1.0@iter24000) 검증.
- 교훈: **P1 iter 길이를 바꾸면 dr 윈도우도 맞춰야** 함(counter는 resume 시 복원됨). ★rolloff30 등 12k-P1 파생 P2도 동일 override 필수.

## 후속 (파이프라인)
1. **v2 텔레포트 측정**(measure_full_v2, PYG_UNEVEN·INIT_BENT 재지정, block별 중심 텔레포트+tile_dwell 기록) → 깨끗한 rough 부하 + tile>90% 확인.
2. **P2**(DR+push 램프, resume from model_11999) → rough 설계앵커 후보.
3. 스파이크 항 진단 → 필요시 캡(Gen-2.2 후보).

## 계보/링크
[[2026-07-13_gen21_rough_p1]](FAIL) → [[2026-07-14_rough_p1_blind_stairs_diagnosis]] → [[2026-07-14_gen21_rough_uneven_p1]](부분) → **본 런**(성공) → P2. flat 앵커 [[2026-07-13_gen21_bent_p2]] warm-start 모체. 등록: [[66_experiment_registry]] Era-9, [[experiment_map.canvas]].
