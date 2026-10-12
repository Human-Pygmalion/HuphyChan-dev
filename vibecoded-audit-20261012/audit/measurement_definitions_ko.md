## 성능 측정 정의

> 이 문서의 경로는 원 학습 코드(mjlab fork) 기준이다. Fork 코드는 `../fork/` 의 patch 로 복원한다. 측정 도구 자체는 이 폴더에 넣지 않았다(정의만).

대상: 끝 저장본 `rough_box_main_v11a` (12,000 회, `model_11999`). 판정표는 `docs/162_next_plan_rough_policy.md` L56-82. 아래 경로는 따로 적지 않으면 `mujoco-sim/mjlab/` 기준이고, `A:` 는 `analysis/`, `S:` 는 `src/mjlab/` 의 약어다.
조건 3 개(서기·회전·걷기)는 평지 `Mjlab-Velocity-Flat-Pygmalion` 을 `play=True` 로 연 **mjlab 환경 1 개(num_envs=1)** 를 CPU 장치(`--device cpu`, MuJoCo-Warp 의 CPU 경로)에서 굴린 것이다. 상자 험지는 **GPU 256 환경**, 산길은 mjlab 이 아닌 **pygviewer 의 순수 CPU MuJoCo**(ONNX 정책)다. 세 계열의 환경 조건이 서로 다르므로 항목마다 따로 적는다.

### 0. 한눈에 보기

| 항목 | 도구 | 시뮬레이터 / 환경 | 지표 (한 줄) | 통과 기준 | 끝 저장본 결과 |
|---|---|---|---|---|---|
| 제자리 회전 | `A:measure_turn.py` → `A:turn_verdict.py` | mjlab Flat, CPU, 1 env | 명령 8 점 각각의 (yaw 기울기 / 명령) % 의 중앙값 | 4 씨앗 각각의 중앙값의 중앙값 ≥ 50 %, 반대 방향 ≤ 1/8 | 45 %, 씨앗 하나가 2/8 |
| 서기 | `A:measure_stand.py` → `A:standing_verdict.py` | 같음 | 발 뗀 비율 %, 27 초 속력 적분 m | 4 씨앗 평균: 발 뗀 ≤ 2 %, 흘러감 ≤ 0.3 m | 1.3 %, 0.38 m |
| 걷기 속도 | `A:gait_kinematics_probe.py` → `A:walking_verdict.py` | mjlab Flat, CPU, 1 env, DR 제거 | 15 초 창의 몸체 좌표 vx 평균 | 3 씨앗 평균 1.14~1.26 m/s | 1.32 |
| 좌우 차 | 같음 | 같음 | 엉덩이 yaw 진폭 좌우 차 % | ≤ 25 % | +18 % |
| 방향 틀어짐 | 같음 | 같음 | 경로 처음 1/10 과 끝 1/10 의 진행 방향 차이 | ≤ 15° | 8.6° |
| 상자 험지 6 / 9 | `A:terrain_eval.py --set-env PYG_BOX_TERRAIN=1` | mjlab Rough, **GPU 256 env**, 훈련 환경 설정 | 넘어진 환경 비율, 추종 오차 | 단계 6: 넘어짐 ≤ 10 %, 오차 ≤ 0.25 m/s / 단계 9: 넘어짐 ≤ 20 % | 0.4 % / 0.21, 11.3 % |
| 산길 182 m | `tools/terrain_nav/trail_walk_check.py` | pygviewer SimCore, 순수 CPU MuJoCo + 스캔 높이 지도 | 완주 여부, 몸통의 발밑 지면 위 높이 최저 | 4 조건 완주, 최저 ≥ 0.7 m | 4/4, 최저 0.79 m |

---

### 1. 평지 공통 (서기·회전: `measure_loads.py` 를 그대로 쓴다)

`measure_turn.py` 와 `measure_stand.py` 는 둘 다 `measure_loads.main()` 을 부르되 `build_command_blocks` 만 바꿔 끼운다 (`A:measure_turn.py:93`, `A:measure_stand.py:59`). 그래서 환경·정책·기록 방식이 같다.

| 항목 | 값 | 근거 |
|---|---|---|
| 호출 | `CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl nice -n 5 python measure_*.py --run-dir D --checkpoint D/CK --tag … --device cpu --rollout-seed s` | `A:matched_set.sh:10,13,16` |
| PYG_* 플래그 | 그 런의 `repro/launch_manifest.json` 의 `environment` 중 `PYG_` 로 시작하는 것 전부 | `A:matched_set.sh:9` (값은 `deploy_asset/pyg_rough_main_20261012/train_env_manifest.json` 와 동일) |
| 과제 | `Mjlab-Velocity-Flat-Pygmalion`, `load_env_cfg(task, play=True)`, `num_envs=1` | `A:measure_loads.py:156,225-226` (`matched_set.sh` 가 `--task` 를 안 주므로 기본값) |
| 평지 | `terrain_type="plane"`, 높이 지도 관측은 행위자에서 제거 (훈련 때도 행위자는 높이 지도를 안 봄) | `S:tasks/velocity/config/pygmalion/env_cfgs.py:986-1004` |
| 제어 주기 | 물리 0.005 s × decimation 4 = 50 Hz (`step_dt` 0.02 s) | 설정 덤프로 확인 (아래 §1-1), `A:measure_loads.py:469` |
| 정책 | 저장본의 행위자만 로드, `get_inference_policy` = **결정적 평균 행동**. 행동 잡음·관측 잡음 모두 꺼짐 (`PYG_OBS_NOISE`, `PYG_ACTION_NOISE`, `PYG_MEASURE_ACTION_NOISE` 를 안 줌; 매니페스트에도 없음) | `A:measure_loads.py:229,296,300,313`; `S:…/env_cfgs.py:955` (play 는 `enable_corruption=False`) |
| 밀기 | 없음 (`--push` 를 안 줌; play 설정이 `push_robot` 도 제거) | `A:measure_loads.py:199-203`; `S:…/env_cfgs.py:956` |
| 시작 | `env.reset()` 후 **명령 (0.5, 0, 0) 으로 120 걸음(2.4 s)** 걷는 준비 운동. 기록하지 않음 | `A:measure_loads.py:166,413-419` |
| 매 걸음 | 명령을 명령 버퍼에 직접 덮어씀 → 관측 재계산 → 행동 → `env.step` → 기록 | `A:measure_loads.py:408-411,503-507` |
| 기록 | 걸음마다 `time`(=걸음 번호×0.02, 준비 운동 제외하고 0 부터), `qpos_full`(전체 qpos), `cmd_*`, `base_vx/vy`(몸체 좌표), `base_height`, `GRF_*`, 관절별 `qpos/qtarget/tau` | `A:measure_loads.py:449-466,636-686` |
| 몸체 좌표 속도 | `R^T · qvel[:3]` (R = base 쿼터니언의 회전행렬, `qvel[:3]` 는 free joint 의 월드 선속도) | `A:measure_loads.py:659-681` |
| 씨앗 | `--rollout-seed s` → `torch.manual_seed(s)`, `np.random.seed(s)`. 환경 설정의 `seed` 는 None 이라 `env.seed` 가 덮어쓰지 않음. Warp 쪽 난수는 심지 않음 | `A:measure_turn.py:87-92`, `A:measure_stand.py:61-66`; `S:envs/manager_based_rl_env.py:186-187`; 설정 덤프에서 `seed None` |
| 넘어짐 처리 | **기록 중 넘어짐을 검출하는 코드 없음.** `env.step` 이 종료 조건(몸 기울기 > 70°, 몸통 높이 < 0.7 m, NaN, 물리 폭주)에서 환경을 자동 리셋하므로, 넘어지면 리셋된 새 로봇이 이어서 기록됨 | `S:envs/manager_based_rl_env.py:436-447`; 종료 목록은 설정 덤프 |

#### 1-1. 무작위화 (DR) 상태 — 코드를 읽어야 알 수 있는 함정

`play=True` 는 `curriculum = {}` 로 비우고 `push_robot` 만 빼며 (`S:…/env_cfgs.py:950-967`), **다른 무작위화 이벤트는 그대로 둔다.** 그런데 DR 강도를 0 → 최대로 키우는 장치는 곡선 항목 `dr_levels` (`S:tasks/velocity/mdp/curriculums.py:112-172`) 이고 곡선이 비었으므로, 이벤트는 **생성 때의 최대 범위 그대로** 켜진다. 즉 서기·회전 측정은 **리셋마다 최대 강도의 DR** 아래에서 이뤄진다:

- 발 마찰 `abs` 0.3~1.2, 두 발 공유 (`S:tasks/velocity/velocity_env_cfg.py:241-250`; 모드 `reset` 으로 바뀜 `S:…/env_cfgs.py:662-667`)
- 관절 엔코더 편향 ±0.015 rad (`velocity_env_cfg.py:251-258`)
- 링크별 관성 DR `inertial_dr_*` 7 개 (`PYG_INERTIAL_DR=1`, 범위는 `PYG_MASS_DR_JSON`) — 최대 범위로 생성 (`S:…/env_cfgs.py:803-845`)
- 시작 위치 x,y ±0.5 m, 높이 0.01~0.05 m, yaw ±3.14 rad 무작위 (`velocity_env_cfg.py:204-216`). 관절 초기값은 키프레임 그대로 (`position_range (0,0)`, `PYG_RESET_JOINT_NOISE` 없음)

이 결론은 매니페스트 플래그를 환경변수로 준 뒤 `load_env_cfg(..., play=True)` 가 돌려준 이벤트 목록을 찍어서 확인했다 (`…/scratchpad/dump_cfg.py` 의 출력: `foot_friction reset (0.3,1.2)`, `encoder_bias reset ±0.015`, `inertial_dr_* reset` 존재, `push_robot` 없음, `curriculum: []`). **걷기 측정(§4)은 이 DR 을 일부러 제거**하므로 서기·회전과 조건이 다르다.

#### 1-2. 발 접촉력 (서기에서 쓰는 `GRF_*_mag`)

`GRF_<발>_mag = √(fx²+fy²+fz²)`, 여기서 f 는 `cfrc_ext[발 본체][3:6]` (`A:measure_loads.py:649-655`). `cfrc_ext` 는 워프 경로의 값이거나, 워프가 접촉을 빠뜨리면 CPU `mj_rnePostConstraint` 재계산값이다 (`A:measure_loads.py:421-448,521-531`; 어느 쪽이었는지는 실행 로그의 `wrench source =` 줄에만 남고 저장소에서는 **코드에서 확인 못 함**).
코드 주석이 경고하는 점: 발목이 닫힌 고리(`PYG_ANKLE_MODE=AB`)인 모델에서는 발 본체의 `cfrc_ext` 에 **연결봉 구속력이 섞인다** (정지 시 L+R 이 347 N 로봇에 1169 N, `A:measure_loads.py:627-629`). 접촉만 더한 `GRFc_<발>_{x,y,z}` 도 같이 기록하지만 (`:630-648`) 서기 판정은 `GRF_*_mag` 를 쓴다 (`A:standing_verdict.py:161-162`). 구속력이 섞인 값에 5 N 문턱을 건 것이므로, 재구현 때 접촉력만 쓰면 "발 뗀 %" 가 같은 값이 나온다는 보장이 없다 (구속력이 섞였을 때와 접촉만 쓸 때의 차이를 비교한 기록은 **코드에서 확인 못 함**).

---

### 2. 제자리 회전 (`measure_turn.py` + `turn_verdict.py`)

**명령 일정**: (vx, vy) = (0, 0), wz = −1.0, −0.75, −0.5, −0.25, +0.25, +0.5, +0.75, +1.0 rad/s 를 이 순서로, 명령당 **15 s = 750 걸음** (`A:measure_turn.py:56-57,84-86,93`). 블록 사이 리셋 없음 — 이전 블록의 상태에서 이어짐. 첫 블록은 §1 의 준비 운동(0.5 m/s 전진 2.4 s)에서 바로 이어짐. 총 120 s.
**씨앗**: `--rollout-seed 0,1,2,3` 4 회 (`A:matched_set.sh:12-14`). 환경은 §1 (DR 최대, 밀기 없음).

**지표** (`A:turn_verdict.py:145-171`):

1. 요(yaw) = 기록한 `qpos_full[:,3:7]` (w,x,y,z 쿼터니언)에서 `atan2(2(wz+xy), 1−2(y²+z²))` (`:133-136`), 전체 시계열에 `np.unwrap` (`:150`).
2. 명령이 바뀌는 지점으로 블록을 나눔 (`:139-142`). 명령 0 인 블록은 건너뜀 (`:154`).
3. 각 블록에서 **앞 3 s(150 걸음)를 버리고** (`--settle` 기본 3.0, `:156,181-184`) 남은 **12 s(600 걸음)** 에 대해 `np.polyfit(시간, 요, 1)` 의 기울기 = 달성 속도 `rate` (`:159-162`). 블록의 남은 길이가 10 걸음 미만이면 제외 (`:157`).
4. `pct = 100 · rate / 명령` (`:163`).
5. 씨앗 하나의 `median_pct` = 8 개 `pct` 의 중앙값 (`:168`). `wrong_way` = `pct < 0` 인 블록 수 (`:169`) — 즉 달성 속도의 부호가 명령과 반대인 블록 수 (n = 8).

**통과 기준** (`docs/162` L78 "평지 회전"; 끝 저장본 판정표 L58-70):
- 4 씨앗 각각의 `median_pct` 4 개의 중앙값 ≥ 50 %. 끝 저장본: 씨앗별 64/26/40/51 % → 중앙값 45 % (`docs/162` L52 가 이 값을 쓰며 "중앙값 45 %" 로 적음; 코드 안에 4 씨앗을 묶는 계산은 없고 `matched_set.sh:23-25` 가 씨앗별 줄만 출력하므로 4 개를 묶는 중앙값은 사람이 문서에서 계산한 것).
- 반대 방향: **씨앗마다** 8 개 중 ≤ 1 개 (끝 저장본은 씨앗 하나가 2/8 → 미달). 코드가 찍는 막대(`:195-196`)는 "wrong way 0" 이라 사전 기준(≤ 1/8, 완화안)과 다르다. 판정 기준은 문서 쪽이다.

---

### 3. 서기 (`measure_stand.py` + `standing_verdict.py`)

**명령 일정**: 걷기 구간 없이 (0, 0, 0) 을 **30 s = 1500 걸음** (`A:measure_stand.py:46-48,51-55`; `--walk-m` 기본 0, `--stand-s` 기본 30). 앞의 준비 운동(0.5 m/s 로 2.4 s 걷기)에서 곧바로 정지 명령으로 넘어간다 (`A:measure_loads.py:414-419`).
**씨앗**: `--rollout-seed 0,1,2,3` (`A:matched_set.sh:15-17`). 환경 §1 (DR 최대).

**측정 창** (`A:standing_verdict.py:151-159`):
- "서 있는" 걸음 = |cmd_vx|, |cmd_vy|, |cmd_wz| 모두 < 1e-6 인 걸음.
- 첫 서 있는 걸음 시각 `t[i0]` 에서 **3 s 를 버린** (`--settle` 기본 3.0, `:219-221`) 뒤 전부 = 27 s = 1350 걸음. (이것이 "27 초"의 정체.)

**지표**:
- **발 뗀 %** `off_frac`: 창 안에서 `min(GRF_L_foot_link_mag, GRF_R_foot_link_mag) < 5 N` (`CONTACT_N = 5.0`, `:116`)인 걸음의 비율 — 둘 중 **어느 한 발이라도** 5 N 미만이면 "뗀" 걸음 (`:161-163`). 힘의 정의는 §1-2.
- **흘러감** `travel_m` = `Σ hypot(base_vx, base_vy) · dt` (`:172,192`), `dt` = 시간 차의 중앙값(0.02 s, `:148`). `base_vx/vy` 는 몸체 좌표 속도(§1). 즉 **순변위가 아니라 속력을 시간 적분한 경로 길이**이며, 제자리 흔들림도 더해진다.
- 씨앗 간 요약은 평균과 표본 표준편차(ddof=1) (`:198-202`).

**통과 기준** (`docs/162` L80 "평지 서기"): 4 씨앗 **평균** 발 뗀 ≤ 2 %, 27 s 흘러감 ≤ 0.3 m. 끝 저장본: 1.3 %, 0.38 m (±0.44, 씨앗 하나가 큼). `matched_set.sh:26` 가 `foot off ground (%)`, `travelled`, `lift events` 줄만 뽑는다.
(`standing_verdict` 의 `--run-dir` 를 안 주므로 자세 보상 `pose_reward` 는 NaN 이고 판정에 안 쓰인다 — `:172,175`.)

---

### 4. 걷기: 속도·좌우 차·방향 틀어짐 (`gait_kinematics_probe.py` + `walking_verdict.py`)

이 도구만 mjlab 환경에서 **DR 이벤트를 모두 제거**한 "공칭 로봇"을 쓴다. 서기·회전(§1-1)과 다르다.

| 항목 | 값 | 근거 |
|---|---|---|
| 호출 | `--vx 1.2 --seed {11,12,13} --init-jitter 0.05` (`--dwell`·`--settle`·`--warm` 기본값) | `A:matched_set.sh:18-21` |
| 환경 | `Mjlab-Velocity-Flat-Pygmalion`, `play=True`, `num_envs=1`, `device="cpu"` | `A:gait_kinematics_probe.py:142-143,196,957` |
| 제거한 이벤트 | `push_robot`, `foot_friction`, `encoder_bias`, `base_com`, `inertial_dr_*` 전부 (물리 DR 없음) | `:147-155` |
| 명령 | (1.2, 0, 0) 고정. 명령 재추첨 방지(`resampling_time_range` 를 에피소드 길이+100 s 로), `rel_standing/heading/forward/world_envs = 0`, 매 걸음 명령 버퍼에 덮어씀 | `:157-170,306-316` |
| 시작 | `envw.reset()` 직후 준비 운동 없이 **3 s(150 걸음) 기록 안 한 채 같은 명령으로 걸음**(`--settle` 3.0) | `:303-318,971` |
| 기록 | 그 뒤 **17 s = 850 걸음** (`--dwell` 17.0) | `:303,962-967` |
| 씨앗 | `--seed s`: `random/np/torch` 심고 `cfg.seed = s` (환경이 `seed_rng` 호출). 기본 리셋은 결정적이라 `--init-jitter` 0.05 rad 를 줘 **모든 관절의 초기 위치를 ±0.05 rad 균일 무작위**로 연다. 시작 위치·yaw 무작위는 §1-1 과 같음 | `:172-194,977-982`; `S:envs/manager_based_rl_env.py:186-187` |
| 정책·잡음 | §1 과 같음 (평균 행동, 잡음 없음, 밀기 없음) | `:202-204` |

**분석 창** (`A:walking_verdict.py:48`): 기록한 17 s 중 `time ≥ 2.0 s` (`--warm` 기본 2.0, `:86-91`) → **15 s = 750 걸음**. (기록 시간은 `t·dt` 로 0 부터라 3 s 설정 구간은 시간축에 안 들어 있다.)

**지표** (`A:walking_verdict.py:45-73`):
- **걷기 속도** = 창 안 `base_vx`(몸체 좌표 vx) 의 평균 (`:66`). 기록: `rec["base_vx"] = (R^T qvel[:3])[0]` (`A:gait_kinematics_probe.py:410`).
- **엉덩이 yaw 좌우 차 %** (`yaw_gap_pct`): `al, ar` = 창 안 `q_L_hip_yaw_joint`, `q_R_hip_yaw_joint`(도 단위, 관절 각도 그대로, 부호 변환 없음)의 **95 백분위 − 5 백분위** (`_amp`, `:40-42,50-52`). `yaw_gap_pct = (al − ar) / (0.5·(al + ar)) · 100` (`:67`). 부호 있음(왼쪽이 크면 +).
- **방향 틀어짐** (`drift_deg`): 창 안 월드 좌표 `base_x, base_y` (`:56`; 기록은 `qpos[0], qpos[1]`, `A:gait_kinematics_probe.py:412-413`). `n = max(3, len(x)//10)` (750 걸음이면 75 = 1.5 s). `h0 = atan2(y[n]−y[0], x[n]−x[0])` (처음 1/10 의 진행 방향), `h1 = atan2(y[−1]−y[−n], x[−1]−x[−n])` (끝 1/10). `drift = |wrap(h1 − h0)|` 를 도로 환산 (`:59-62`). 몸의 요 각이 아니라 **궤적의 방향**이다.
- 씨앗 3 개의 평균 ± 표본 표준편차 (`:123`). `matched_set.sh:27` 가 `forward speed`, `L-R gap`, `heading drift` 줄만 뽑는다.

**통과 기준**: 속도 1.14~1.26 m/s (명령 1.2 의 ±5 %), 좌우 차 ≤ 25 %, 방향 틀어짐 ≤ 15° (`docs/162` L79,81). 좌우 차를 부호 있는 값 그대로 비교하는지 절댓값으로 비교하는지는 판정 코드가 없어 **코드에서 확인 못 함** (8,000 회 저장본의 −32 % 를 25 % 초과로 읽은 정황이 `docs/162` L51 에 있어 절댓값 비교로 보이나 확정 아님). 끝 저장본: 1.32 m/s, +18 %, 8.6° (±11).

---

### 5. 상자 험지 단계 6 / 9 (`terrain_eval.py --set-env PYG_BOX_TERRAIN=1`)

**환경** (`A:terrain_eval.py:23-70`):

| 항목 | 값 | 근거 |
|---|---|---|
| 호출 | `--env-run <run dir> --policy NAME=<체크포인트 .pt 또는 .onnx> --set-env PYG_BOX_TERRAIN=1` 와 기본값 `--levels 0,3,6,9 --envs 256 --vx 0.6 --seed 7`. 문서·배포 README 는 **6 s** 로 쟀다고 적었으나 코드 기본값은 20 s (`--seconds 20.0`). 실제 명령줄은 저장소에 없음 → `--seconds 6` 은 `docs/experiments/2026-10-11_flat_vs_rough_policy_compare.md` L72 와 `deploy_asset/pyg_rough_main_20261012/README.md` L75 의 서술에 의존. 끝 저장본에 쓴 정확한 명령줄은 **코드에서 확인 못 함** | `A:terrain_eval.py:23-34` |
| 플래그 | 매니페스트의 `PYG_*` 를 환경변수로 주입 후 `--set-env` 덮어쓰기, **`PYG_STAGNATION` 은 제거** (정체 종료 규칙 자체가 판정에 섞이지 않게) | `:35-41` |
| 과제 | `Mjlab-Velocity-Rough-Pygmalion`, **`play=False`**(훈련 설정 그대로), 장치 `cuda`, 256 env, `cfg.seed = 7` | `:51-53,70` |
| 지형 | `BOX_TERRAINS_CFG`: 8×8 m 타일, 10 행(=단계 0~9) × 5 종(평지, 오르는 계단, 내리는 계단, 불규칙 블록, 기울어진 판; 훈련 설정은 `curriculum=True` 라 종류마다 열 하나) | `S:terrains/config.py:336-352`, `S:…/env_cfgs.py:112-113`, `S:tasks/velocity/velocity_env_cfg.py:421-428` |
| 난이도 | 행 r 의 난이도 `d = r/9` (단계 6 → 0.667, 단계 9 → 1.0) | `S:terrains/terrain_generator.py:248-262` |
| 난이도→기하 | 계단 높이 `0 + d·0.1` m, 폭 0.3 m, 중앙 평평 3 m, 가장자리 1 m (`config.py:46-54`; `primitive_terrains.py:79`). 불규칙 블록: 칸 0.4 m, 칸 높이 `[−b, +b]` 균일, `b = d·0.12` m, 중앙 평평 1 m (`config.py:154-163`; `primitive_terrains.py:427-449`). 기울어진 판: 칸 1 m, 기울기 최대 `d·15°`, 높이 변동 `d·0.2` m (`config.py:265-277,336-352`; `primitive_terrains.py:1488-1489`) | 좌측 인용 |
| 단계 고정 | 단계 L 마다 `terrain.terrain_levels[:] = L`, `env_origins` 를 그 단계 타일 원점으로 다시 계산, `torch.manual_seed(7)`, `env.reset()` (곡선 `{}` 이라 단계가 자동으로 안 변함) | `:54,119-124` |
| 종류 배분 | 256 env 를 5 종에 거의 균등 (타일 비율 0.2 씩, `terrain_types` 를 비율대로 `repeat_interleave`). 같은 종류 env 는 같은 타일 원점에서 출발 (서로 다른 sim 이라 간섭 없음) | `S:terrains/terrain_entity.py:343-364` |
| 시작 | `reset_base`: 타일 원점에서 x,y ±0.5 m, yaw ±3.14 rad 무작위. 로봇은 몸체 앞쪽으로만 걸으므로 방향은 매 env 가 다름 | `S:tasks/velocity/velocity_env_cfg.py:204-216` |
| 명령 | 전 env 에 vx = 0.6 m/s 고정, vy = 0, wz = 0, 서 있는 env·방향 지정 env 0 %, 재추첨 간격 1e6 s | `A:terrain_eval.py:56-68` |
| 에피소드 | `episode_length_s = seconds + 5` (시간 초과로 끝나지 않게) | `:55` |
| 밀기 | 제거 | `:69` |
| 무작위화 | 훈련 설정의 DR 이벤트가 **곡선 없이** 남음 → §1-1 과 같은 이유로 마찰 0.3~1.2, 엔코더 편향, 관성 DR 이 **최대 범위** (설정 덤프: `Rough play=False` 에 이벤트 존재, `curriculum` 은 코드가 `{}` 로 비움 `:54`) | 설정 덤프, `S:tasks/velocity/mdp/curriculums.py:112-172` |
| 관측 잡음 | **켜짐** (`play=False` 의 행위자 `enable_corruption=True`), 행동은 평균(결정적) | 설정 덤프 `obs corruption actor True`; `:76-97,131-133` |
| 정책 실행 | `.pt` 이면 `obs_normalizer` 로 `(x−mean)/(std+0.01)` 후 MLP(ELU) 평균 행동 (`TorchActor`), `.onnx` 이면 onnxruntime CPU. env 마다 배치 1 로 따로 추론 | `:76-107,131-133` |

**넘어짐 정의**: `tm.terminated & alive` — 시간 초과가 아닌 **첫 종료**. 훈련 설정에서 시간 초과가 아닌 종료는 (설정 덤프, `PYG_STAGNATION` 제거 후):
- `fell_over`: `acos(−projected_gravity_z)` > 1.2217 rad (70°) (`S:envs/mdp/terminations.py:24-32`)
- `low_base`: `PYG_LOW_BASE_REL=1` 이라 `base_height_above_terrain_below(minimum_height=0.7)` — 높이 광선 센서 `terrain_scan` 의 **중심 반경 0.25 m 안 광선의 땅 높이 중앙값**을 뺀 몸통 높이 < 0.7 m (`S:tasks/velocity/mdp/terminations.py:152-172`)
- `nan_detection`, `physics_blowup`(관절 > 100 rad/s 또는 몸통 > 10 m/s 또는 비유한 값, `terminations.py:175-192`)
`out_of_terrain_bounds` 는 `time_out=True` 라서 넘어짐에 안 센다. 한 env 가 넘어지면 그 걸음 이후 `alive=False` 로 고정 (`:137,143`).

**지표** (`:126-155`):
- `fall` = `1 − 평균(alive)` — 전체 256 env 중 넘어진 비율 (5 종 평균, 종류별 값은 `per_type`).
- `err` = env 별로 `Σ_{살아있는 걸음} ‖v_xy,body − cmd_xy‖₂ / 살아있는 걸음 수` (`root_link_lin_vel_b[:, :2]` 와 명령 (0.6, 0)의 벡터 차 노름; 넘어진 걸음은 제외, 분모 최소 1), 그것을 **env 전체에 평균** (넘어진 env 는 그 전까지의 평균으로 포함). 단위 m/s.
- 정책이 넘어지고 리셋된 뒤의 걸음은 `alive` 로 걸러져 안 들어감.

**통과 기준** (`docs/162` L77): 단계 6 `fall ≤ 10 %` 와 `err ≤ 0.25 m/s`, 단계 9 `fall ≤ 20 %`. 끝 저장본: 단계 6 → 0.4 % / 0.21, 단계 9 → 11.3 % (`docs/162` L52, L61-62). 통계 반복은 없음(단일 씨앗 7, 256 env 가 표본).

---

### 6. 산길 182 m (`tools/terrain_nav/trail_walk_check.py`)

**시뮬레이터**: mjlab 이 아니라 `tools/pygviewer` 의 `SimCore` (`realtime=False`) — 배포용 순수 CPU MuJoCo. 정책은 **ONNX**(`<캐시>/<정책이름>.onnx` + 정책 계약 파일), 관측 잡음·행동 지연·무작위화 코드는 `sim_core.py`/`policy.py` 에 **없다**(`noise|random|delay` 검색 결과 0 건) — 결정적. 같은 입력이면 같은 궤적이 나온다 (`docs/161` §7 "물리·정책이 결정적"). `trail_walk_check.py:51-53`.
**지형**: 산길 스캔에서 만든 높이 지도(충돌 칸 10 cm)와 나무 기둥 268 개를 모델에 굽는다 (`PYG_BAKE_TERRAIN`, `docs/161` §5-2). 높이 지도 파일 `trail_sim_terrain.npz`, 경로 `terrain.npz` 는 **이 PC 에 없음**(`~/pyg_demo/…` 부재)이라 지형 데이터 자체의 내용은 **코드에서 확인 못 함**. 경로는 총 182.4 m, 내리막 방향 −9 m (`docs/161` §5-2 표).
**조건 4 개**: 내리막 0.6, 내리막 1.0, 오르막 0.6, 오르막 1.0 m/s. 내리막 = 기본(경로 출발 → 목적지), 오르막 = `--reverse` (`trail_walk_check.py:29,40-41`); 속도는 `--vx` (`:26`). 끝 저장본에 쓴 정확한 명령줄(`--policy` 이름, `--seconds`)은 저장소에 없음 → **코드에서 확인 못 함** (기본 `--seconds 60` 은 182 m 를 못 걷는 값이라 더 길게 줬을 것이다; 0.5 m/s 오르막 완주가 499 s 였으므로 최소 수백 초).

**제어** (`trail_walk_check.py:55-92`):
- 시작: 키프레임 `knees_bent`, 경로 첫 점에 두고 땅 높이를 맞추며, 요는 `PATH[8]−PATH[0]` 방향.
- 매 정책 걸음(50 Hz, `decimation` 걸음마다)에 경로 추종기가 명령을 만든다: 경로에서 현재 위치에 가장 가까운 점 `k_near` 를 앞쪽 80 점 안에서 찾고, 거기서 경로 길이로 **1.0 m 앞**(`--lookahead`)의 점을 목표로 삼아 `err = atan2(목표−위치) − 요` (±π 로 감김). `wz = clip(1.5·err, ±0.6)`, `vx = vx_cmd · max(0.2, cos(err))` (단 |err| ≥ 1.2 rad 이면 0), `vy = 0`.

**지표와 판정**:
- 몸통의 발밑 높이 `above = 몸통 z − ground_z(x, y)`, `ground_z` 는 높이 지도 칸을 직접 조회한 값(나무 기둥은 안 셈; `:45-49,83`).
- 기울기 `tilt = acos(body z축 · 월드 z)`, `up_z = 1−2(qx²+qy²)` (`:72,84`).
- 스크립트의 **넘어짐 종료**: `above < 0.45 m` 또는 `tilt > 60°` (`:88`). **완주**: `k_near ≥ len(PATH)−3` 이면 루프 종료 (`:91`). 시간 초과(`--seconds`)로 끝난 경우에도 "no fall" 이 찍히므로 **완주 여부는 출력의 "along path X m of 182.4"** 로 확인해야 한다 (`:96-99`).
- 출력되는 "몸통 `above` 최소~최대" 와 "tilt 최대" 가 문서의 "몸통 0.86~0.97 m·기울기 7°" 이다 (`docs/162` L54; 예전 팔 표는 `docs/experiments/2026-10-10_rough_stag_ab_b_v11a.md` 에서 "몸통 최저·기울기 최대"로 같은 값을 옮김).
- 판정 "몸통 ≥ 0.7 m" 은 **스크립트의 넘어짐 문턱(0.45 m)보다 엄격한, 사람이 출력에서 `above` 최소를 읽어 적용한 기준**이다 — 이 문턱을 코드로 검사하는 줄은 없다.

**통과 기준** (`docs/162` L76): 4 조건 모두 완주(182 m 끝까지) + 넘어짐 0 + `above` 최소 ≥ 0.7 m. 끝 저장본: 4/4, 최소 0.79 m.

---

### 7. 학습 건강 (사전 기준: 폭주·NaN 0, 물리 폭주 종료 ≤ 0.1 / 회)

이 항목을 계산하는 측정 도구는 이번 요청의 대상이 아니다. 학습 로그의 종료 항목 `nan_detection`, `physics_blowup` (정의는 §5)를 센 값이라는 점만 코드로 확인했다. 집계 방법(분모 등)은 **코드에서 확인 못 함**.

---

### 8. 재구현 때 걸리는 점 (코드에서 확인된 것만)

1. **서기·회전은 DR 최대 강도, 걷기는 DR 없음.** 같은 정책의 측정 조건이 다르다 (§1-1, §4). 걷기 속도 1.32 가 "공칭 로봇"에서 나온 값이라는 뜻.
2. **서기 "발 뗀" 은 구속력이 섞인 `cfrc_ext` 5 N 문턱** (§1-2). 접촉력만 쓰는 재구현은 다른 값이 나올 수 있다.
3. **"흘러감"은 속력의 시간 적분**(경로 길이)이며 27 s 창이 아니라 "정지 명령 시작 후 3 s 를 뺀 나머지"다. 준비 운동에서 이어지는 0.5 m/s → 0 감속이 앞 3 s 안에 들어 있다.
4. **회전은 12 s 창의 요 선형회귀 기울기 / 명령**, 블록 8 개의 중앙값이며, 사전 기준이 말하는 "4 씨앗 중앙값"을 계산하는 코드는 없다 (씨앗별 값을 사람이 모음).
5. **넘어짐 검출 코드는 상자 험지에만 있다.** 서기·회전·걷기 측정은 넘어져도 리셋돼 계속 기록한다. 걷기 측정은 `play` 설정의 `fell_over`/`low_base` 종료가 켜져 있어도 기록 중 리셋 여부를 남기지 않는다.
6. 실제 실행 로그(`analysis/out/*.npz`, 실행 로그의 `wrench source`, 상자 험지·산길의 정확한 명령줄)는 이 PC 에 없다. 위에서 "코드에서 확인 못 함" 으로 적은 것이 그 부분이다.
