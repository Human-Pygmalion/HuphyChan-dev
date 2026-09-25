# gen21_rough_uneven2_p2b — rough 설계앵커 (uneven, DR+push full ramp)

> **한 줄**: uneven2 P1(계단·급슬로프 제거, fell 0 수렴)에 **DR+push 램프**를 정상 주입한 P2. 첫 P2(`2026-07-15_00-58-24_gen21_rough_uneven2_p2`)는 dr_factor 0.33 정체 버그로 폐기, dr윈도우 정렬 override(PYG_DR_START/END_ITER=12000/24000)로 재학습한 것이 본 런. **dr 0→1.0 완전 램프하는 내내 fell ~0** = 유효 robust rough 앵커. 부하 측정(v2)·A/B는 아래 §후속.

## §1 재현성
- run: `logs/rsl_rl/pygmalion_velocity/2026-07-15_03-48-03_gen21_rough_uneven2_p2b` (최종 model_23998)
- launch: `PYG_UNEVEN=1 PYG_INIT_BENT=1 PYG_DR_START_ITER=12000 PYG_DR_END_ITER=24000` + `train_wandb_video.py Mjlab-Velocity-Rough-Pygmalion --resume --load-run <uneven2_p1> --load-checkpoint model_11999`, 4096 env, +12k iter(→abs 23999).
- 지형: `UNEVEN_TERRAINS_CFG`(flat0.2·slope0.15+0.15@rise/run 0.3·rough0.25·wave0.25, **계단 0%**). config: params/env.yaml.
- 측정소스: `p2b_v2_fc` **완료**(2026-07-16 01:07, v2 텔레포트, tile 88.6%) — §3c/§5/§7 채움.

## §1b Reward & Gains
- Gen-2.1 번들 동일([[2026-07-13_gen21_bent_p2]]) — 지형(UNEVEN)·DR override만 변인. Kp/Kd/effort/speed 불변.

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


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-07-15_03-48-03_gen21_rough_uneven2_p2b`)

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
| `PYG_DR_END_ITER` | 24000 |
| `PYG_DR_START` | (값 미기재) |
| `PYG_DR_START_ITER` | 12000 |
| `PYG_HIP_CANT` | (값 미기재) |
| `PYG_INIT_BENT` | 1 |
| `PYG_MOTOR_MEAS` | 1 |
| `PYG_ROLLOFF30` | (값 미기재) |
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_UNEVEN` | 1 |
| `PYG_V2` | (값 미기재) |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

**P2 (`2026-07-15_00-58-24_gen21_rough_uneven2_p2`)**: 액추에이터 게인·한계·액션 clip이 위 P1 표와 **동일** (env.yaml 대조). 달라지는 것은 도메인 랜덤화·push 등 학습 조건뿐이다.

<!-- SPEC-TABLES:END -->

## §2 최종 지표 (full DR+push, dr_factor=1.0)
- **fell_over 0.0000**(최종), DR 램프 구간(iter 12k→24k) 내내 0.00–0.04 유지.
- track_linear reward 0.66·track_angular 0.60·Mean reward 27.0 (rough+full DR라 flat 1.33보다 낮음이 정상; 절대 추종%는 v2 측정에서).
- dr_factor 궤적: 0(iter12k)→0.33(16k)→0.50(18k)→0.67(20k)→0.83(22k)→**1.0(24k)**.

## §4 부모/변인 비교
- vs 첫 P2 `2026-07-15_00-58-24_gen21_rough_uneven2_p2`(폐기): 동일 launch, **dr override만 추가**. 그 런은 dr_factor가 iter 17571에도 0.0 → DR 미주입(robust 무효). 원인·수정은 [[2026-07-14_gen21_rough_uneven2_p1]] §P2-버그.
- vs P1: DR+push 램프 추가(단일변인).

## §9 DR/push 램프
- ★핵심 수정: dr윈도우를 `start_step=288000(iter12k)·end_step=576000(iter24k)`로 정렬(env override). counter는 resume 시 복원(P1 12k+FRESH→288000)되므로 P2b 시작부터 램프 개시. push_max x/y±0.7·z±0.4·rpy±0.52/0.78, friction 0.3–1.2, encoder±0.015, com±0.025~0.03. dr=1.0 완전 도달 확인.

## §3c 측정 커버리지 (v2 텔레포트, p2b_v2_fc — 완료 2026-07-16 01:07)
- **tile_dwell 88.6%** · grid_dwell 100% — 구 p2r_fc(60% 오염) 대비 **대폭 개선**, v2 텔레포트 프로토콜 유효. (목표 90%에 1.4%p 근접 — 앵커로 수용, 잔여는 블록 경계 settle 구간.)

## §5·§7 부하 — rough 앵커 vs flat 앵커 (실측, 적응정책)
![[rough_vs_flat_anchor.png]]

| 관절 (모터) | rough RMS/P99 | flat RMS/P99 | rough %rated/%peak | rough−flat |
|---|---|---|---|---|
| hip_pitch (RS04) | 28.4/94.5 | 27.7/91.7 | 71%/79% | +2/+2 |
| hip_yaw (RS04) | 12.3/38.2 | 12.8/32.9 | 31%/32% | −1/+4 |
| hip_roll (RS04) | 23.2/62.1 | 23.6/58.1 | 58%/52% | −1/+3 |
| knee (RS04) | 38.6/107.4 | 45.5/112.4 | 96%/89% | **−17/−4** |
| ankle_pitch (RS03) | 14.9/49.5 | 13.6/54.7 | 75%/83% | +7/−9 |
| **ankle_roll (RS00)** | **5.2/17.7** | 2.9/10.3 | **104%/126%** | **+45/+53** |
| **GRF (BW)** | **P99 1.74** | P99 1.20 | | **+45%** |

### 해석 (rough가 flat 대비 하중을 어떻게 바꾸나)
- ★**ankle_roll(RS00)이 험지의 지배 병목**: P99 17.7 = **RS00 peak 14의 126% 초과**, RMS 104% rated. 울퉁불퉁·경사에서 **발목 측방(내번/외번) 보정이 급증** → 최약 모터(RS00 14/5)가 flat에선 여유(73% peak)였다가 rough서 초과. **RS00→상위(RS02급) 상향 or ankle_roll 링크레버 재설계 검토 필요**.
- **GRF P99 1.74BW**(flat 1.20, +45%) — 험지 착지 충격↑. 구조·베어링 사이징은 rough 앵커값으로. (raw peak 13.8BW는 클립 아티팩트, P99×1.25로 사이징.)
- **knee는 오히려 −17%p RMS/−4%p P99**(더 신중한 gait) — flat이 knee 열부하 worst 유지.
- hip_pitch/hip_roll P99 소폭↑(+2/+3), 나머지 무해.
- **설계 하중 세트 결론**: flat=knee 열(114% rated)이 worst / rough=**ankle_roll(RS00)·GRF**가 worst. 두 앵커의 **관절별 max**를 설계 상한으로 채택.

## §11 이상징후 — reward 스파이크
- neg-spike 20/12k iter(P1 36→감소), fell 무영향. uneven 엣지 대형접촉의 캡없는 페널티 추정 → Gen-2.2 캡 후보.

## §12 판정
- ✅ **유효 robust rough 설계앵커 확정** — full DR+push fell ~0, v2 tile 88.6%, 부하 실측 완료.
- ★설계 반영: **rough는 ankle_roll(RS00 126% peak)·GRF 1.74BW가 병목** = flat(knee 열)과 다른 관절이 worst. 하중 세트 = flat∪rough 관절별 max. RS00 ankle_roll 상향/레버 재설계가 rough 대응 핵심 과제.
- 계보: flat 앵커 gen21_bent_p2 → uneven2 P1(지형수정) → **본 P2b**(DR정상) = **flat+rough 설계 하중 세트 완성**.

## 후속 (선택 — 앵커 승격 시)
- §8c TN 설계선도·§10 링크 wrench 6관절·§3b loadviz 영상은 앵커 확정에 따라 gen21_bent_p2급으로 확장 가능(현재 §5/§7 부하판정으로 설계 의사결정 충분).
- 등록: [[66_experiment_registry]] Era-9, [[experiment_map.canvas]], INDEX.
