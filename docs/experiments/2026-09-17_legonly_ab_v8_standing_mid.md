# legonly_ab_v8_standing_mid — 정지를 지키면서 보행을 되찾는 중간값 찾기 (2026-09-17)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> 계보: [[2026-09-14_legonly_ab_v6b_standing_flatfoot]] 의 **유일 변인 후속**.
> 사용자 지시(2026-09-17): *"10.8.0.14 장비도 써서 병렬로 돌려. Lock 쓰는거 잊지 말고."*
> **이 런은 원격 기계(10.8.0.14, RTX 3090 24 GB)에서 돈다** — 로컬에서 도는
> [[2026-09-15_legonly_ab_v7_heading_revert]] 와 **변인이 직교**하므로 병렬로 돌려도
> 해석이 깨지지 않는다.

## §0 왜 이 학습을 하는가

v6b 가 답과 대가를 같이 가져왔다:

| | v5 | **v6b** | |
|---|--:|--:|---|
| 정지 자세 추종 보상 | 0.0023 | **0.8444** | ✅ **367배 — 목표 달성** |
| 발 뗀 시간 | 16.2 % | **5.0 %** | ✅ |
| 엉덩이회전 벌어짐 | +14.4 / −13.6° | **+0.0 / +0.2°** | ✅ |
| — 그런데 — | | | |
| 보행 엉덩이회전 좌우차 | +18.4 % | **−39.7 %** | ❌ (±20 % 이내여야) |
| 비틀기 / 걸음 비율 | 0.96 | **1.49** | ❌ (0.5 이하여야) |
| 15초 방향 표류 | 14.5° | **40.4°** | ❌ (30° 이하여야) |

원인은 표본 배분이다([[2026-09-14_legonly_ab_v6b_standing_flatfoot]] §2h): **정지 환경을
10 % → 30 % 로 올리자 걷는 환경이 90 % → 70 % 로 줄었고**, 그 대가가 걸음의 좌우 대칭과
직진성에서 나왔다.

**0.1 은 정지가 죽고, 0.3 은 보행이 상한다. 사이를 찾는다.**

## §1 설계 — 변인 하나

| | v6b | **v8 ← 이번 런** |
|---|--:|--:|
| `PYG_REL_STANDING` | 0.3 | **0.175** |
| `PYG_REL_HEADING` | 0.8 | 0.8 (동일) |
| 나머지 전부 | — | 동일 |

0.175 는 0.1 과 0.3 의 중간보다 약간 아래다. **정지 성과는 비선형으로 얻어졌을 가능성이
높기 때문이다** — 중단된 v6(발바닥 수정 전, 정지 0.3)에서 자세 추종 항이 이미 1.7e−04 →
0.53 으로 되살아났으므로(§4), 0.3 만큼 올리지 않아도 항이 살아날 수 있다. 살아나기만 하면
나머지 표본은 보행에 주는 것이 이득이다.

### 읽는 법

* 정지 4항목이 v6b 수준을 지키고 보행 3항목이 목표 안에 들어오면 → **0.175 채택**.
* 정지가 무너지면 → 정지 부활에 **0.3 이 필요**하다는 뜻. 그때는 표본 배분이 아니라
  **보상 쪽**(정지 규모 허용 오차를 관절별로)으로 가야 한다(연구노트 선행).
* 보행이 안 돌아오면 → 원인이 표본 배분이 아니다. **조향 성분 벌점** 설계로 간다.

### ★ 게이트 플래그는 **켠다** (v7 과 반대)

`PYG_GATE_EXCLUDE_STANDING=1`. v7 에서는 일부러 껐는데 여기서는 켜는 이유:

**이 런의 변인이 게이트 지표 자체를 움직이기 때문이다.** 게이트는 속도 추종 오차를 전 환경
평균으로 재는데 서 있는 환경은 오차가 0 에 가깝다([[2026-09-14_legonly_ab_v6b_standing_flatfoot]]
§2c-0). 정지 비율을 0.3 → 0.175 로 바꾸면 **기준선이 저절로 달라져** 커리큘럼 진행 속도가
변인에 끌려다닌다. 플래그를 켜면 게이트가 **0 이 아닌 명령을 받은 환경만** 보므로 커리큘럼
일정이 변인과 무관해지고, **비교가 더 깨끗해진다.**

v7 은 방향유지만 바꾸고 정지 비율은 그대로라 게이트 지표가 흔들리지 않으므로 끈 것이 맞다.

## §1a 실행 명령 (원격 10.8.0.14)

```
# ★ setsid 로 세션에서 떼어내야 한다 (§1a-1) — 안 그러면 ssh 가 끊길 때 같이 죽는다
ssh syaro@10.8.0.14 'cd ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab && setsid nohup \
bash analysis/run_v2_scratch.sh --run legonly_ab_v8_standing_mid --ankle AB --seed 42 --num-envs 16384 \
  --p1-max-iters 16000 --ramp-iters 10000 --digest-iters 4000 --settle-hold 10 --n-stages 5 \
  --gate-min-dwell 800 --gate-max-dwell 3000 --gate-window 100 --gate-err-ratio 1.1 \
  --gate-fell-max 0.005 --gate-min-episodes 64 \
  --entropy-start 0.01 --entropy-end 0.002 --video-interval 8000 --video-length 500 \
  --logger tensorboard --vy-stages \
  --env PYG_MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix \
  --env PYG_MASS_DR_JSON=/home/syaro/MikuchanRemote/Human-Pygmalion/tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json \
  --env PYG_REL_STANDING=0.175 \
  --env PYG_REL_HEADING=0.8 \
  --env PYG_GATE_EXCLUDE_STANDING=1 \
  < /dev/null > /tmp/v8_launch.log 2>&1 & disown; echo ok'
```

기록기가 `tensorboard` 인 이유: 원격에 wandb 자격증명이 없다. 기록되는 지표는 같다.

## §1a-0 원격 기계 준비 (2026-09-17)

| 항목 | 값 |
|---|---|
| 기계 | `syaro-Standard-PC-Q35-ICH9-2009` (가상머신) |
| 그래픽카드 | **RTX 3090 24 GB** (로컬 16 GB보다 큼), 준비 시점 사용률 0 % |
| CPU / 메모리 / 디스크 | 4코어 / 39 GB(37 여유) / 342 GB 여유 |
| 접속 | **키 인증으로 됨** — `~/.secrets/10.8.0.14` 는 **열지 않았다**(사용자 지시) |

★**파일시스템은 로컬과 별개다.** 로컬 `MikuchanRemote` 는 NFS(192.168.20.202:/volume5/…)이고
원격은 자기 ext4 의 복사본이다. 표식 파일을 로컬에 만들고 원격에서 찾아 **없음**을 확인했다.
따라서:

* **잠금은 기계마다 따로 필요하다.** 원격에 `/home/syaro/pyg_fea/locks/`(`git.lock`,
  `gpu.lock/`)를 새로 만들었다. 로컬 잠금은 원격을 막지 못한다 — 그래픽카드가 다른 물건이므로
  막을 이유도 없다.
* **결과는 원격 디스크에 쌓인다.** 주기적으로 당겨와야 한다.

★**발사 전에 잡은 함정**: 원격에는 `mujoco-sim`·`docs`·`tools` 만 있었고
**`pygmalion_locomotion/` 과 `Motor_Spec/` 이 통째로 없었다.** 웅크림 키프레임 자산이 거기
있으므로 그대로 발사했으면 **기준 자세가 없는 채로 돌거나 옛 자세로 학습**했을 것이다
(v6 이 폐기된 것과 같은 함정). 동기화 후 확인:

| | 원격 값 | 기대값 |
|---|--:|--:|
| 웅크림 발목 숙임 | **+10.027°** | +10.03 (평평한 발바닥) |
| 폐루프 오차 | **0.0 mm** | 0 |
| 리셋 높이 | **0.8975 m** | 0.8975 |

원격 발사 전 검사 **통과** — 좌우 이름 왼쪽 +0.1237 m, 등식 12행, 폐루프 0.0000 mm.

`pygmalion_locomotion/logs`(9.2 GB)는 **보내지 않았다** — 학습에 불필요하고 원격 디스크와
네트워크만 먹는다.

## §1a-1 발사 기록 — 원격 발사는 **ssh 세션과 함께 죽는다**

| 시각 | 무슨 일 |
|---|---|
| 14:13:40 | 1차 발사. 발사 전 검사 통과, 학습 프로세스 시작(그래픽카드 81 %) |
| 14:1x | **내 ssh 연결이 끊기자 학습 프로세스가 같이 죽음** — 그래픽카드 0 %, 저장 디렉터리 미생성, 런처만 남아 정지 |
| 14:18 | 정리: 멈춘 런처 종료, 상태파일 삭제, **내가 미리 만든 빈 `gpu.lock/` 제거** |
| 14:19:15 | **`setsid` 로 세션 분리 재발사** |
| 14:19:21 | 재현 스냅샷 기록 → `2026-09-17_14-19-15_legonly_ab_v8_standing_mid_p1/repro` |
| 14:20:26 | **반복 19, 낙상 0.0000, 그래픽카드 92 % / 6,772 MiB** — 발사 확정 |

### 왜 첫 발사가 죽었나

`nohup` 은 **SIGHUP 만** 무시한다. 도구 시간 초과로 ssh 클라이언트가 **SIGTERM** 을 받자 같은
프로세스 그룹에 있던 학습 자식까지 종료됐다. 런처(부모)는 `nohup` 덕에 살아남았지만 자식을
잃고 멈춰 있었다.

**고침**: `setsid` 로 **새 세션·새 프로세스 그룹**에 넣는다. 그러면 ssh 쪽 신호가 닿지 않는다.

```
ssh <host> 'cd <dir> && setsid nohup bash analysis/run_v2_scratch.sh ... \
    < /dev/null > /tmp/<run>_launch.log 2>&1 & disown; echo ok'
```

`< /dev/null` 과 출력 리다이렉트도 함께 필요하다 — 열린 표준입출력이 남아 있으면 ssh 가
종료를 기다리거나 신호를 전달한다.

### 곁다리 — 잠금 디렉터리를 미리 만들면 안 된다

내가 `/home/syaro/pyg_fea/locks/gpu.lock/` 을 **빈 디렉터리로 미리** 만들어 두었더니 런처가
**"주인 미상의 잠금이 잡혀 있다"**(`held by ? pid None`)고 읽고 기다렸다. 죽은 주인으로 판정해
스스로 풀긴 했지만 2분을 버렸다. **잠금은 런처가 직접 만들게 둔다** — 준비할 것은 상위
`locks/` 디렉터리까지다.

### 이 런의 실제 발사 정보

| | |
|---|---|
| 발사 | **2026-09-17 14:19:15** (원격 10.8.0.14) |
| 저장 디렉터리 | `logs/rsl_rl/pygmalion_velocity/2026-09-17_14-19-15_legonly_ab_v8_standing_mid_p1`(**원격 디스크**) |
| 발사 전 검사 | 통과 — 좌우 이름 왼쪽 +0.1237 m, 등식 12행, 폐루프 0.0000 mm |
| 기록기 | `tensorboard`(원격에 wandb 자격증명이 없어 로컬 v7 과 다름 — 지표 내용은 동일) |
| 14:20 상태 | 반복 19, 단계 0/4, **낙상 0.0000**, 그래픽카드 92 % / 6,772 MiB |

★**결과 회수**: 원격 디스크에 쌓이므로 주기적으로 당겨와야 한다(할 일 #23).

## §1b-0 ★ 이 런의 게이트 숫자는 v6b·v7 과 **비교하면 안 된다** (2026-09-17)

`PYG_GATE_EXCLUDE_STANDING=1` 이므로 게이트가 읽는 값이 **다른 모집단**이다:

| | 게이트가 세는 환경 | 첫 단계 오차(관측) |
|---|---|--:|
| v6b · v7 | **전 환경**(서 있는 환경 포함, 그쪽 오차 ≈ 0) | 0.16~0.18 |
| **v8** | **0 이 아닌 명령을 받은 환경만** | **0.21** |

v8 의 숫자가 큰 것은 **더 못해서가 아니라 쉬운 표본을 빼고 재기 때문이다.** 같은 런 안에서
단계 사이를 비교하는 것은 유효하지만, **런끼리 게이트 숫자를 나란히 놓으면 틀린다.**
런 사이 비교는 §1c 의 측정 도구(정지·보행·회전)로만 한다.

## §1c 판정 기준 — v6b·v5 와 같은 도구·같은 시드 수·각자 자기 자세

### 정지 (`analysis/standing_verdict.py`, 시드 8개 × 15초, 앞 3초 버림)

| 기준 | v5 | v6b | **v8 목표** |
|---|--:|--:|---|
| 정지 자세 추종 보상 | 0.0023 | **0.8444** | **0.5 이상**(v6b 수준 유지) |
| 발 뗀 시간 | 16.2 % | **5.0 %** | 10 % 이하 |
| 엉덩이회전 벌어짐 왼/오른 | +14.4 / −13.6° | **+0.0 / +0.2°** | 5° 이내 |
| 액션 한계 포화 | 28.6 / 20.5 % | **0.0 / 0.0 %** | 5 % 이하 |

### 보행 (`analysis/walking_verdict.py`, 흔들림 6개 × 15초 × 1.2 m/s 똑바로)

| 기준 | v5 | v6b | **v8 목표** |
|---|--:|--:|---|
| **엉덩이회전 좌우차** | +18.4 % | **−39.7 %** | **\|·\| 20 % 이내** |
| **비틀기 / 걸음 비율** | 0.96 | **1.49** | **1.0 이하** |
| **15초 방향 표류** | 14.5° | **40.4°** | **20° 이하** |
| 실제 속도 | 1.215 | 1.173 m/s | 1.15 이상 |
| 넘어짐 | 0 | 0 | 0 |

### 회전 (`analysis/measure_turn.py`, 8개 요 속도 × 15초) — 참고만

방향유지 비율을 건드리지 않으므로 **v6b 수준(중앙값 12 %)에서 달라지지 않을 것으로 예상**한다.
달라지면 회전 고장의 원인이 방향유지 비율이 아니라는 뜻이고, 그건 v7 의 판정과 교차 검증된다.

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


**§1b-2. 액추에이터 동역학·한계** (이 런의 `params/env.yaml` 파싱 — `2026-09-17_14-19-15_legonly_ab_v8_standing_mid_p1`)

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
| `PYG_GATE_EXCLUDE_STANDING` | 1 |
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
| `PYG_REL_STANDING` | 0.175 |
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

## §R 참조

- 부모 런(완주·전 측정): [[2026-09-14_legonly_ab_v6b_standing_flatfoot]]
- 보행 후퇴 진단: 같은 노트 §2h · 게이트 지표 함정: 같은 노트 §2c-0
- 병렬로 도는 직교 변인 런: [[2026-09-15_legonly_ab_v7_heading_revert]]
- 측정 도구: `analysis/standing_verdict.py` · `analysis/walking_verdict.py` ·
  `analysis/measure_turn.py`
- 코드: `PYG_REL_STANDING`·`PYG_GATE_EXCLUDE_STANDING`
  (`src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py`)
