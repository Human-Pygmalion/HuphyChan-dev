# 실측한 모터 값을 시뮬레이터에 넣는 법

2026-09-09 · 사용자 지시: **"이 값을 어떻게 mujocolab 에 넣어야 하지? 구체적으로 정리한다."**

측정 근거는 `docs/experiments/2026-09-09_motor_rotor_inertia.md`(회전자 관성),
`2026-09-09_motor_friction_retest.md`(마찰),
`2026-09-09_pd_step_chirp_sim_vs_real.md`(같은 지령으로 응답 대조)에 있습니다.

---

## 0. 한눈에 — 무엇을 어디에 넣는가

**딱 한 곳입니다.**

파일: `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py`
줄: **98~99** (`MOTOR_MEAS` 사전)

```python
MOTOR_MEAS = {
  "RS04": dict(armature=0.016333, damping=0.009492, frictionloss=0.269456),
  "RS03": dict(armature=0.015265, damping=0.022342, frictionloss=0.285370),
}
```

이것을 이렇게 바꿉니다.

```python
MOTOR_MEAS = {
  # 2026-09-09 라즈베리파이 직결 재측정. 감속기 뒤(출력축) 기준, 축에 부착물 없음, 37~44도.
  #   armature      회전자 관성 [kg m^2]  — 타성 감속과 마찰 소거법이 합치 (RS04 77.3±0.6 g m^2)
  #   frictionloss  움직일 때 마찰 [N m]  — 순수 타성 감속에서 J x 감속률
  #   damping       점성 [N m s / rad]    — 같은 맞춤의 기울기
  "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
  "RS03": dict(armature=0.0638, damping=0.0025, frictionloss=0.400),
}
```

바뀌는 배수:

| | RS04 | | RS03 | |
|---|---|---|---|---|
| | 지금 → 실측 | 배수 | 지금 → 실측 | 배수 |
| `armature` | 0.016333 → **0.0773** | **4.7배 ↑** | 0.015265 → **0.0638** | **4.2배 ↑** |
| `frictionloss` | 0.269456 → **0.519** | 1.9배 ↑ | 0.285370 → **0.400** | 1.4배 ↑ |
| `damping` | 0.009492 → **0.0275** | 2.9배 ↑ | 0.022342 → **0.0025** | **8.9배 ↓** |

★ RS03 의 점성만 **내려갑니다.** 세 값이 모두 같은 방향으로 틀린 게 아닙니다.

---

## 1. 세 값이 각각 무엇인가 — 그리고 ★함정 하나

이 세 값은 `_motor()` (같은 파일 147~154줄)를 거쳐 액추에이터 설정으로 들어갑니다.

```python
def _motor(family: str) -> dict:
  m = MOTOR_MEAS[family]
  return dict(
    armature=m["armature"], frictionloss=m["frictionloss"], viscous_damping=m["damping"]
  )
```

| 넣는 이름 | 무엇인가 | 물리 |
|---|---|---|
| `armature` | **회전자 관성** | 관절을 돌리기 시작할 때 느끼는 무게. 관절 공간의 질량 행렬 대각에 더해집니다 |
| `frictionloss` | **움직일 때 마찰** | 방향만 거스르는 일정한 토크. 속도와 무관 |
| `viscous_damping` | **점성** | 속도에 비례하는 저항. `f = −b·v` |

### ★ 함정 — `damping` 이라는 이름이 두 군데 있고 뜻이 다릅니다

액추에이터 설정에는 `damping` 이라는 항목이 **따로** 있습니다. 그건 **제어 이득 Kd**
(위치 제어기의 미분 이득)이고, **점성 마찰이 아닙니다.**

```python
"knee_rs04": TnPdActuatorCfg(
    stiffness=200.0,          # 제어 이득 Kp
    damping=5.0,              # ← 제어 이득 Kd. 여기에 0.0275 를 넣으면 안 됩니다
    **_motor("RS04"),         # ← armature / frictionloss / viscous_damping 이 여기로
)
```

mjlab 원본 설명도 그렇게 적어 두었습니다
(`mjlab/actuator/actuator.py:56-62`): *"Unlike `damping` (the PD derivative gain kv,
which is active control), this is a passive property."*

**우리가 잰 점성 0.0275 는 `viscous_damping` 자리입니다.** `MOTOR_MEAS` 의 키 이름이
`damping` 인 것은 `_motor()` 안에서 `viscous_damping=` 으로 옮겨 담기 때문이며, 헷갈리기
쉬우니 위 주석을 꼭 같이 넣으십시오.

---

## 2. 되돌리는 스위치

같은 파일 100줄에 이미 있습니다.

```python
_MOTOR_MEAS = os.environ.get("PYG_MOTOR_MEAS", "1") != "0"
```

`PYG_MOTOR_MEAS=0` 으로 돌리면 **카탈로그 값**(RS04 0.007 / RS03 0.005, 마찰 없음)으로
돌아갑니다. 옛 설정을 재현할 때 씁니다. 새 값을 넣어도 이 스위치는 그대로 동작합니다.

**권합니다**: 새 값을 바로 덮어쓰지 말고 스위치를 하나 더 두어 A/B 를 하십시오.

```python
_MEAS_GEN = os.environ.get("PYG_MOTOR_MEAS_GEN", "2")   # "1" = 2026-08-23, "2" = 2026-09-09
MOTOR_MEAS = {
  "1": {"RS04": dict(armature=0.016333, damping=0.009492, frictionloss=0.269456),
        "RS03": dict(armature=0.015265, damping=0.022342, frictionloss=0.285370)},
  "2": {"RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
        "RS03": dict(armature=0.0638, damping=0.0025, frictionloss=0.400)},
}[_MEAS_GEN]
```

이유는 아래 4번에 적습니다.

---

## 3. 이 값이 **못 담는** 것 — 정지 마찰

RS04 는 **붙어 있다 떨어지는 힘이 0.901 뉴턴미터**로, 움직일 때 마찰(0.519)보다 **1.7배**
큽니다. 그런데 **MuJoCo 의 관절 `frictionloss` 는 값이 하나뿐**이라 이 구분을 담을 자리가
없습니다.

실측으로 확인한 결과(`2026-09-09_pd_step_chirp_sim_vs_real.md`):

* 정지 마찰 값(0.901)을 `frictionloss` 에 넣으면 → 계단이 과하게 묵직해지고 공진이 과하게
  눌립니다(봉우리 1.05, 실물 1.41).
* 움직일 때 마찰(0.519)을 넣으면 → 실물에 제일 가깝습니다(봉우리 1.32, 계단 잔류 오차 1.07도
  대 실물 1.29도).

**→ `frictionloss` 에는 움직일 때 마찰(0.519)을 넣으십시오.**

정지 마찰을 정말 담고 싶다면 MuJoCo 의 `<dcmotor>` 액추에이터 + `lugre` 마찰 모형을 써야
합니다. `lugre` 는 `coulomb`(동)과 `static`(정지)을 **따로** 받습니다. 다만 지금 우리
액추에이터 경로(`TnPdActuatorCfg`)는 그 길을 쓰지 않으므로 **구조 변경**이 필요하고,
Stribeck 속도와 미끄러지기 전 강성을 **추가로 측정**해야 합니다. 아직 안 쟀습니다.

---

## 4. ★ 바로 넣지 마십시오 — 이유와 순서

`armature` 를 4.7배 올리는 것은 **작은 수정이 아닙니다.** 관절이 그만큼 무거워지므로
지금까지 학습한 **정책 15개 전부의 전제가 바뀝니다.** 같은 이득으로 같은 걸음을 낼 수
없습니다.

실측으로 확인된 변화의 크기:

* 고유진동수 **5.6 Hz → 2.6 Hz** (RS04, kp=20 기준). 관성이 4.7배면 √4.7 = 2.2배 느려집니다.
* 초당 10번 흔들 때 명령 추종량 **0.86 → 0.21**. 즉 **빠른 동작을 지금의 1/4밖에 못 냅니다.**
* 계단에서 못 지우는 오차 **0.22도 → 1.07도**.

권하는 순서:

1. **모터만 세운 대조를 먼저 다시 돌립니다** — 값을 넣은 뒤 `pd_sim_vs_real.py` 로 실물
   기록과 다시 겹쳐 확인. 이건 학습 없이 몇 초면 끝납니다.
2. **짧은 A/B 학습**(같은 시드, 같은 리워드, `PYG_MOTOR_MEAS_GEN` 만 다르게)으로 학습이
   수렴하는지, 걸음이 어떻게 달라지는지 봅니다.
3. **제어 이득을 다시 사이징**합니다. 관성이 4.7배면 같은 감쇠비를 내는 `damping`(Kd)이
   √4.7 = 2.2배 커야 합니다. 지금 값 그대로면 **덜 감쇠**되어 출렁입니다.
   `stiffness`(Kp)도 대역폭 기준으로 다시 잡아야 합니다.
4. 그 뒤에야 본 학습.

★ 3번을 건너뛰면 "관성만 무거워지고 제어는 그대로"가 되어, 정책이 **더 나빠진 것처럼**
보입니다. 값이 틀려서가 아니라 이득이 안 맞아서입니다.

---

## 5. 같이 손봐야 하는 것들

### 5-1. 발목 (링키지로 반사되는 값)

발목 두 관절은 `MOTOR_MEAS` 를 직접 안 쓰고, 링키지를 지난 **반사된 값**을 파일에서 읽습니다
(같은 파일 156~175줄).

```python
ANKLE_RP_ENVELOPE = .../pygmalion_locomotion/assets/pygmalion_v2/ankle_rp_envelope.json
c = json.load(open(ANKLE_RP_ENVELOPE))["legs"]["L"]["reflected_center"]
```

그 json 은 `tools/robot_model/ankle_rp_envelope.py` 가 만듭니다. **모터 값이 바뀌면 그
도구를 새 값으로 다시 돌려 json 을 갱신해야 합니다.** 안 하면 발목만 옛 값으로 남습니다.

### 5-2. 토크-속도 곡선과 무부하 속도

같은 파일 118줄:

```python
NO_LOAD_RPM = {"RS04": 200.0, "RS03": 200.0}
```

벤치 실측은 **227 rpm** 이었습니다(2.0 뉴턴미터로 밀어 자리잡은 속도 1362 도/s).
48 볼트 곡선의 무부하가 191.5 rpm 이므로, 지금 벤치 전원은 **약 57 볼트**로 보입니다(추론).

**지금은 바꾸지 마십시오.** 실제 로봇을 몇 볼트로 돌릴지 정해진 뒤에 맞춰야 합니다. 그리고
57 볼트는 모터의 과전압 고장 문턱(60 볼트)에 너무 가까워 **48 볼트로 낮추는 것이
권고**입니다(`2026-09-09_motor_friction_retest.md`). 48 볼트로 내리면 지금 곡선이 맞습니다.

### 5-3. 아직 안 잰 모터

발목의 RS03·RS00 은 **안 쟀습니다.** 위 두 대와 같은 비율로 틀렸는지 **모릅니다.**
RS03 의 점성이 모형보다 8.9배 **작았다**는 것이, 비율을 짐작으로 옮기면 안 된다는 증거입니다.

---

## 6. 넣은 뒤 확인하는 법

값을 바꾸고 이걸 돌리면 실물 기록과 다시 겹쳐 볼 수 있습니다.

```bash
cd mujoco-sim/mjlab && .venv/bin/python3 ../../System_ID/pd_sim_vs_real.py \
  --step-csv ../../System_ID/data/pd_step_4_kp20_kd1_a8_rs04.csv \
  --chirp-csv ../../System_ID/data/pd_chirp_4_kp20_kd1_a8_rs04.csv \
  --family RS04 --out ../../docs/img/pd_sim_vs_real_rs04.png
```

맞으면 이렇게 나와야 합니다(실물 기준):

| | 실물 | 실측값 넣은 모형 |
|---|---|---|
| 공진 봉우리 자리 | 2.5 Hz | 2.5 Hz |
| 봉우리 높이 | 1.41 | 1.32 |
| 초당 10번에서 남는 양 | 0.22 | 0.21 |
| 계단 잔류 오차 | 1.29도 | 1.07도 |

---

## 7. 요약 — 붙여넣기용

```python
# mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:98
# 2026-09-09 라즈베리파이 직결 재측정 (감속기 뒤 기준, 축에 부착물 없음, 37~44도).
# 근거: docs/experiments/2026-09-09_motor_rotor_inertia.md
#       docs/experiments/2026-09-09_motor_friction_retest.md
#       docs/experiments/2026-09-09_pd_step_chirp_sim_vs_real.md
# 주의: 여기 'damping' 은 _motor() 에서 viscous_damping(수동 점성)으로 들어간다.
#       액추에이터의 damping(=제어 이득 Kd)과 다른 값이다.
# 주의: armature 가 4.7배 커지므로 제어 이득(stiffness/damping)을 다시 사이징해야 한다.
MOTOR_MEAS = {
  "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
  "RS03": dict(armature=0.0638, damping=0.0025, frictionloss=0.400),
}
```

★ 이 문서는 **어떻게 넣는지**를 적은 것이고, **넣을지 말지는 결정하지 않았습니다.**
4번의 순서를 밟은 뒤에 판단하십시오.
