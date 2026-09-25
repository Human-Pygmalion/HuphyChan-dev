# legonly_ab_v10b_stand01_l1 — A/B v10 의 처리 팔 (2026-09-25)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> 짝: [[2026-09-25_ab_v10_stand_l1_compare]] (비교노트 — 가설·사전 등록 예측은 거기)

## §0 왜 이 학습을 하는가

서 있는 환경 비율을 0.1 로 되돌려도 **서기 전용 L1 편차 보상**으로 서기를 지킬 수 있는가.
0.3 으로 올렸을 때 커리큘럼이 5,400 회에 강제 승급하고(v6b·v7·v8 모두) 걷기가 후퇴했습니다(v6b).
근거: [[../reward_research/2026-09-25_isaac_asimov_teardown]] §3-2.

## §1 무엇이 다른가

| 설정 | v7 | **이 런** |
|---|---|---|
| 서 있는 환경 비율 | 0.3 | **0.1** |
| 방향유지 비율 | 0.3 | 0.3 |
| 서기 전용 L1 편차 | 없음 | **1.0** |

나머지는 v7 과 같습니다(모델 `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix`,
질량 불확실성 JSON, 좌우 단계 명령, 게이트 최대 체류 3,000, 1단계 16,000 · 2단계 10,000+4,000,
씨앗 42, 16,384 환경). 짝 팔과의 차이는 `PYG_STAND_L1` **한 줄뿐**입니다.

발사: `run_v2_scratch.sh --run legonly_ab_v10b_stand01_l1 --ankle AB --vy-stages --gate-max-dwell 3000 --allow-concurrent`
+ `PYG_REL_STANDING=0.1 PYG_REL_HEADING=0.3` + `PYG_STAND_L1=1.0`. 짝 팔과 **같은 그래픽카드에서 동시에**.

**실제 발사 (2026-09-25 17:01:13)**: `analysis/launch_v10b_only.sh` — 짝 팔(16:55:46 발사)보다 **5분 27초 늦게**.
첫 시도는 사전점검이 막았습니다(그래픽 메모리 여유 10.8 GiB < 12.0 GiB). 한 판이 실제로 쓰는 것은
6.6 GB 라서, 런처에 그 기준 하나만 바꾸는 `--min-gpu-free` 를 추가하고(mjlab ae286188) 7.5 GiB 로
띄웠습니다. 메모리·디스크 점검은 그대로 적용됐습니다(여유 메모리 13.9 GiB, 디스크 113.8 GiB, 그래픽 8.9 GiB).
학습 폴더 `logs/rsl_rl/pygmalion_velocity/2026-09-25_17-01-32_legonly_ab_v10b_stand01_l1_p1`.

**★ 3090 으로 이전 (2026-09-25 17:50, 사용자 지시 "3090 / 5060 나눠쓰라고")** — 한 카드(5060)를 둘이
나눠 쓰니 두 팔 모두 반복당 약 11초라 완주까지 약 90시간이었습니다. 원격 3090(10.8.0.14, 17:46 재부팅 후
복귀)으로 이 팔을 옮겼습니다.
- 원격 코드는 이 PC 와 동기화 후 차이 0 개를 확인(`src/`, `analysis/`), 모델 XML·질량 JSON 은 원래 같음.
- 원격 첫 시도(17:49)는 원격에 wandb 로그인이 없어 학습 프로그램이 곧바로 죽었습니다. 기록 방식만
  `--logger tensorboard` 로 바꿔 17:50:51 재발사 — 학습 내용에는 영향이 없고, 런처가 게이트 판정에 쓰는
  텐서보드 기록은 그대로 남습니다. 실패 시도는 지우지 않고 `ABORTED_..._17-49-25_...` 로 옮김.
- **학습 폴더(원격)**: `logs/rsl_rl/pygmalion_velocity/2026-09-25_17-50-55_legonly_ab_v10b_stand01_l1_p1`
- 원격 시작 속도: 반복당 약 3.1초(113초에 36회). 5060 공유 때의 약 3.5배.
- 이 PC 의 첫 시도(17:01, 301회까지)는 원격이 반복을 쌓는 것을 확인한 **뒤에** 멈췄고,
  `ABORTED_2026-09-25_17-01-32_legonly_ab_v10b_stand01_l1_p1` 로 보존했습니다(저장본 0~300).
- 이 이전 중 런처 버그 하나를 찾아 고쳤습니다: 런 폴더를 찾는 패턴 `*_<런이름>` 이 `ABORTED_` 로 옮긴
  폴더에도 걸리고, 이름순 정렬에서 `A` 가 날짜 숫자보다 뒤라 **옮겨 둔 실패 폴더가 "가장 최근"으로
  뽑혔습니다** — 원격의 재현용 기록(repro)이 거기에 쓰였습니다(내용은 이번 발사의 것임을 발사 시각
  17:50:51 과 `--agent.logger tensorboard` 로 확인하고 새 학습 폴더로 옮김). 같은 함수가 재개 시 저장본
  찾기에도 쓰이므로 `ABORTED_` 폴더를 제외하도록 고쳤습니다.
- 원격 학습의 게이트 스냅샷은 **원격 노트**에 쌓입니다 — `tools/remote/pull_remote_run.sh legonly_ab_v10b_stand01_l1` 로 가져옴.

**단일 변인 실증 (원격 발사 직후, 이 PC 의 v10a 와 비교)**: `env.yaml` 차이 = `standing_joint_deviation`
블록(62줄) 하나, `agent.yaml` 차이 = 런 이름과 기록 방식(wandb → tensorboard), 발사 기록의 `PYG_*` 차이 =
`PYG_STAND_L1=1.0` 한 줄. 그래픽카드가 다른 것(3090 대 5060)은 학습 내용(반복 수·환경 수·씨앗)을 바꾸지
않습니다 — 다만 GPU 부동소수 연산 순서가 달라 비트 단위 재현은 원래도 되지 않습니다.

(아래는 이 PC 첫 시도 때의 실증 — 같은 결과)
**단일 변인 실증 (발사 직후)**: 두 팔의 `params/env.yaml` 차이는 `standing_joint_deviation` 보상항 블록
(가중치 −1.0, 명령 문턱 0.1, 엉덩이·무릎·발목 관절) **하나뿐**이고, `agent.yaml` 차이는 런 이름뿐,
발사 기록(`repro/launch_manifest.json`)의 `PYG_*` 차이는 `PYG_STAND_L1=1.0` 한 줄뿐입니다.

## §1c 판정 기준

비교노트 §2 의 사전 등록 예측 P1~P5 와 같은 도구·같은 조건(§3).

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
| standing_joint_deviation | **-1** | - | - |
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


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-09-25_17-01-32_legonly_ab_v10b_stand01_l1_p1`)

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
| `PYG_REL_HEADING` | 0.3 |
| `PYG_REL_STANDING` | 0.1 |
| `PYG_SAFE_TARGET_CLIP` | 1 |
| `PYG_SOFT_LANDING` | 1 |
| `PYG_SOFT_LANDING_MODE` | half |
| `PYG_STAND_L1` | 1.0 |
| `PYG_STUDENT_TEACHER` | 1 |
| `PYG_TN` | 1 |
| `PYG_V2` | 1 |

§1b의 리워드 가중치 표가 정본이다 — 플래그는 그 가중치가 어떻게 조립됐는지의 기록이다.

<!-- SPEC-TABLES:END -->

## §2 최종 지표

(완주 후)

## §2c 학습 중 리뷰 (게이트마다 스냅샷, docs/27 체크리스트)

| 시각 | iter | reward | ep_len | noise σ | value loss | entropy | surrogate / LR | fell / low_base | err_vel xy / yaw | dr_factor / vx_max | thermal | 판정(docs/27) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
