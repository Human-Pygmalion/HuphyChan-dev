# 로봇 값이 세 군데에 나뉘어 있는 이유 — 쉽게 풀어 쓴 설명

2026-09-15 · 사용자 질문:

> *"conventional 한 구조인가? 이해가 잘 안되는데, urdf → mjcf → robots/\*\_constants.py
> 순으로 우선순위를 갖고, 각각 중복값들을 오버라이드한다는건가?"*

> 약어를 쓰지 않습니다. 처음 보는 사람도 읽을 수 있게 씁니다.

---

## §0 먼저 — **URDF → MJCF 순서가 아닙니다**

이 부분이 오해였고, 제 앞선 설명이 그렇게 읽히게 했습니다. 실제 구조는 이렇습니다.

```
   CAD (Fusion 에서 내보낸 형상·질량)
              │
              │  tools/robot_model/build_robot.py 가 한 번에 둘을 찍어낸다
              ├──────────────┐
              ▼              ▼
          MJCF (.xml)     URDF (.urdf)      ← 둘은 **형제**다. 서로 안 읽는다
              │              │
              │              └─► 남에게 주는 설명서. **우리 학습은 안 읽는다**
              ▼
     mjlab 이 읽어 "초안" 을 만든다
              │
              ▼
   pygmalion_constants.py 의 액추에이터 설정이 **일부 칸만 덮어쓴다**
              │
              ▼
          컴파일 → 이때 비로소 물리 모형이 된다 → 학습
```

**확인한 근거**

* `build_robot.py:1007` 이 MJCF 를, `:1119` 가 URDF 를 씁니다 — **같은 프로그램이 같은
  CAD 에서 둘 다** 만듭니다.
* 학습이 읽는 파일은 **MJCF 하나뿐**입니다: `pygmalion_constants.py:428`
  `spec = mujoco.MjSpec.from_file(str(PYG_XML))`.
* mjlab 전체 소스에서 `.urdf` 를 여는 코드는 **없습니다**(다른 로봇 주석에 링크가 하나
  있을 뿐).
* MJCF 파일 안에 `urdf` 라는 글자는 **0번** 나옵니다.

그러니 우선순위는 **두 단계**뿐입니다:

> **MJCF = 기본값 · constants.py = 덮어쓰기**
>
> URDF 는 이 줄에 끼어 있지 않습니다.

## §1 이게 흔한 구조인가 — **예, 이 바닥의 표준입니다**

로봇 학습 도구들은 거의 다 이렇게 합니다.

| 도구 | 형상 파일 | 그 위에 덮어쓰는 코드 |
|---|---|---|
| **Isaac Lab** (NVIDIA) | USD / URDF | `ArticulationCfg` 안의 `ActuatorCfg` — 강성·감쇠·최대토크·회전자관성 |
| **mjlab** (우리가 쓰는 것) | MJCF | `BuiltinPositionActuatorCfg` 등 — 같은 것들 |
| legged_gym / IsaacGymEnvs | URDF | 파이썬 설정의 `stiffness` / `damping` 딕셔너리 |

**왜 이렇게 하나**: 형상(뼈대·무게)은 CAD 에서 나오고 잘 안 바뀌는데, **모터 값과 제어
게인은 자주 바뀌기 때문**입니다. 게인 하나 바꾸려고 모델 파일을 다시 뽑는 것은 번거롭고,
그때마다 형상까지 흔들릴 위험이 있습니다. 그래서 **형상은 파일에, 모터는 코드에** 둡니다.

**Unitree 도 같은 생각을 다르게 풉니다** — 그쪽은 제어 게인을 `g1.yaml` 같은 설정 파일에
두고 로봇 펌웨어에 보냅니다([[142_how_others_clip]]). 형상 파일에는 안 넣습니다.

### 이 구조의 약점 (우리가 실제로 물린 것)

덮어쓰기는 **"설정에 값이 있을 때만"** 일어납니다. 설정을 빠뜨리면 **조용히 파일 값이
쓰입니다.** 오류도 안 납니다.

`mujoco-sim/mjlab/src/mjlab/utils/spec.py:122`

```python
"""Apply joint- or tendon-level overrides. ``None`` preserves the XML value."""
if armature is not None:
    target.armature = armature
if frictionloss is not None:
    target.frictionloss = frictionloss
if viscous_damping is not None:
    target.damping[0] = viscous_damping
```

`None` 이면 파일 값이 남습니다. 실제로 **직렬(RP) 발목 모드**가 이 상태입니다 — 발목 두
관절에 마찰·점성 설정이 없어 파일의 임시값이 그대로 들어갑니다. 그래서
`tools/robot_model/check_motor_params.py` 라는 **검사**를 따로 만들어 막고 있습니다.

## §2 세 파일이 각각 무엇을 담을 수 있나

### §2-1 URDF — 로봇 설명서의 **공통 규격**

위치: `pygmalion_locomotion/assets/pygmalion_v2/<모델이름>.urdf`

| 담을 수 있는 것 | 뜻 | 우리 파일에 |
|---|---|---|
| `<link><inertial>` 질량·무게중심·관성 | 뼈대 하나의 무게와 회전하기 어려운 정도 | ✅ |
| `<joint><origin>` `<axis>` | 관절이 어디 붙고 어느 축으로 도는가 | ✅ |
| `<limit lower upper>` | 관절이 갈 수 있는 각도 | ✅ |
| `<limit effort>` | 최대 토크 | ✅ |
| `<limit velocity>` | 최대 속도 | ⚠ 자리표시자 20 |
| `<dynamics damping>` | **끈적임** — 빨리 돌수록 커지는 저항 | ✅ 2026-09-15 에 넣음 |
| `<dynamics friction>` | **뻑뻑함** — 속도와 무관한 일정한 저항 | ✅ 2026-09-15 에 넣음 |
| 회전자 관성 | 모터 속 원판의 무거움 | ❌ **규격에 칸이 없음** → 주석 + 옆 파일 |
| 제어 게인·폐루프·접촉 설정 | | ❌ 칸 없음 |

### §2-2 MJCF — MuJoCo 전용, 담을 수 있는 게 훨씬 많다

위치: `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/<모델이름>.xml`

URDF 가 담는 것 전부 + 아래.

| 담을 수 있는 것 | 뜻 | 우리 파일에 |
|---|---|---|
| `armature` | 회전자 관성. **관절 관성에 더해짐** | ⚠ 임시값 (덮어써짐) |
| `frictionloss` | 뻑뻑함. **구속으로 풀려서 정지마찰도 됨** | ❌ 아예 없음 (덮어써짐) |
| `damping` | 끈적임 | ⚠ 임시값 (덮어써짐) |
| `equality/connect` | **폐루프** — 로드 끝을 발판에 붙임 | ✅ 4개 |
| `<actuator>` | 모터 정의 | ❌ **0개** (코드가 붙임) |
| 접촉·솔버·메시·키프레임 | | ✅ |

### §2-3 `pygmalion_constants.py` — 코드로 덮어쓰는 층

위치: `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py`

| 담는 것 | 뜻 | 파일을 덮어쓰나 |
|---|---|---|
| `stiffness` (Kp) | 목표 각도에서 벗어난 만큼 당기는 세기 | 파일에 없던 것을 **새로 만듦** |
| `damping` (Kd) | 그 제어기의 제동 | 〃 |
| `effort_limit` | 최대 토크 | 〃 |
| `armature` | 회전자 관성 | **덮어씀** |
| `frictionloss` | 뻑뻑함 | **덮어씀** |
| `viscous_damping` | 끈적임 | **덮어씀** |
| `tn_curve` | 실측 토크-속도 곡선 | 새로 만듦 |
| `soft_joint_pos_limit_factor` | 한계 안쪽 여유 (0.9) | 새로 만듦 |

**중요**: `stiffness`·`damping`(제어기)과 `viscous_damping`(관절 끈적임)은 **이름이 비슷한
다른 것**입니다. 앞은 "목표를 따라가려는 제어", 뒤는 "기계가 원래 갖는 저항" 입니다.

## §3 그래서 실제로 어떤 값이 들어갔나

### §3-1 모터 물성 — 세 파일 비교

| 항목 | URDF | MJCF 파일 | **constants.py (최종)** |
|---|--:|--:|--:|
| 회전자 관성 RS04 | 주석·옆 파일 `0.0773` | `0.0070` (임시) | **0.0773** |
| 회전자 관성 RS03 | 주석·옆 파일 `0.0638` | `0.0050` (임시, 크랭크는 파일에 `0.005`) | **0.0638** |
| 뻑뻑함 RS04 | `friction="0.519"` | **없음** | **0.519** |
| 뻑뻑함 RS03 | `friction="0.251"` | **없음** | **0.251** |
| 끈적임 RS04 | `damping="0.0275"` | `0.02`~`0.2` (임시) | **0.0275** |
| 끈적임 RS03 | `damping="0.0252"` | 〃 | **0.0252** |

출처: 2026-09-10 벤치 실측. 라즈베리파이 CAN 직결(루프 약 4 kHz). 관성은 두 경로가
합치는 것으로 확정(타성 감속 / 토크 대 가속 기울기), 공진 2.48 Hz 로 재확인.
정의 위치 `pygmalion_constants.py:110` `MOTOR_MEAS`.

**지금 돌고 있는 학습이 실제로 쓴 값** (`v6b` 런의 `params/env.yaml`):

```yaml
armature: 0.0638   frictionloss: 0.251   viscous_damping: 0.0252   # RS03 (크랭크·엉덩이회전)
armature: 0.0773   frictionloss: 0.519   viscous_damping: 0.0275   # RS04 (엉덩이앞뒤·좌우·무릎)
```

### §3-2 제어 게인·최대 토크 — **코드에만 있음**

| 관절 | Kp | Kd | 최대 토크 |
|---|--:|--:|--:|
| 엉덩이 앞뒤·좌우·회전 | 150.0 | 6.0 | 120 / 120 / 60 |
| 무릎 | 220.0 | 6.0 | 120 |
| 발목 관절 (2-RSU) | 28.5 | 1.81 | 숙임 90 · 기울임 50 |
| 크랭크 A·B | 22.266 | 1.414 | 60 |

크랭크 값은 손으로 고른 게 아니라 유도값입니다 —
`CRANK_KP = ANKLE_2RSU_KP × 1.25² / 2` (1.25 는 실측 지렛대비, /2 는 모터 두 개).

### §3-3 관절 가동범위 — **세 군데가 다 같음**

| 관절 | 범위 [도] |
|---|---|
| 엉덩이 앞뒤 | −120 ~ +25 (오른) / −25 ~ +120 (왼) |
| 엉덩이 좌우 | −25 ~ +85 (오른) / −85 ~ +25 (왼) |
| 엉덩이 회전 | ±45 |
| 무릎 | 0 ~ +120 (오른) / −120 ~ 0 (왼) |
| 크랭크 A·B | ±68.75 |
| 발목 숙임 | −50 ~ +30 |
| 발목 기울임 | ±20 |

URDF·MJCF 가 같은 것은 **독립 파서로 대조해 확인**했습니다(회전축 0.0000도, 회전중심
0.0000 mm — [[145_model_parameter_provenance]] §2). 액추에이터 설정은 범위를 건드리지
않고, 대신 **안쪽으로 10 % 여유**(`soft_joint_pos_limit_factor = 0.9`)를 둔 값을
정책의 목표 클램프로 씁니다.

### §3-4 질량

* 모터: 저울 실측 — RS04 **1.514 kg**, RS03 **0.9195 kg** (카탈로그 0.880 보다 무거움)
* 링크: Fusion CAD — 골반 5.358 · 허벅지 2.118 · 정강이 2.702 · 엉덩이앞뒤 1.842 ·
  엉덩이좌우 1.373 · 발 0.845 kg
* 전체 **23.63014 kg** (URDF 는 23.63414 — 차이 4 g 은 URDF 규격이 요구하는 더미 링크
  4개 × 1 g, [[145_model_parameter_provenance]] §3)

## §4 한 줄 정리

* **URDF 와 MJCF 는 형제**입니다. 같은 CAD 에서 같이 나오고, 서로 안 읽습니다.
* **학습은 MJCF 만** 읽고, 그 위에 **코드가 모터 값을 덮어씁니다.** 이건 Isaac Lab 등이
  쓰는 **표준 구조**입니다.
* **URDF 는 남에게 주는 것**이라 그 안에 값이 다 있어야 합니다 — 그래서 2026-09-15 에
  끈적임·뻑뻑함을 채웠습니다([[144_urdf_motor_params_portability]]).
* **MJCF 파일에 남은 임시값은 학습에 영향이 없습니다**(덮어써짐). 다만 파일만 보면
  오해하게 되고, 설정을 빠뜨리면 조용히 그 값이 쓰이는 위험이 있습니다.
