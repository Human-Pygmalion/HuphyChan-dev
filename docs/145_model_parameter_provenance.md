# 모델 파일(MJCF·URDF)의 모든 값 — 어디서 왔고, 서로 맞는가

2026-09-15 · 사용자 지시: *"모든 파라미터들의 출처와 정합성이 맞는지 검토하고, 검증의
근거를 포함해 정리해서 mjcf / urdf 보고해줘."*

> 약어를 쓰지 않습니다. 처음 보는 사람도 읽을 수 있게 씁니다.
> **주장하지 않고 돌려서 확인한 것만 적습니다.** 각 항목마다 무엇을 실행해 확인했는지
> 같이 적었고, 확인 못 한 것은 못 했다고 적었습니다.

대상 파일 (지금 학습이 쓰는 세대):

* MJCF `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml`
* URDF `pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.urdf`

---

## §1 한눈에 — 파라미터별 출처와 판정

| # | 파라미터 | 정본(출처) | URDF | MJCF 파일 | 학습이 실제로 쓰는 값 | 판정 |
|---|---|---|---|---|---|---|
| 1 | 링크 질량·무게중심·관성 | Fusion CAD 내보내기 + 모터 실측 질량 | ✅ | ✅ | 같음 | ✅ **일치**(§2) |
| 2 | 관절 위치·회전축 | 같은 CAD | ✅ | ✅ | 같음 | ✅ **일치**(§2) |
| 3 | 관절 가동범위 | `docs/107` ROM 표 (CAD 자기충돌 스윕 + 설계캡) | ✅ | ✅ | 같음 | ✅ **일치**(§2·§6) |
| 4 | 최대 토크(effort) | RobStride 설명서 (`pygmalion_constants.py:568` 에 출처 링크) | ✅ 120/60 | — | 120/60 | ✅ 일치 |
| 5 | 발목 effort 90/50 | `mech_design_eval.py` 2-RSU 공동구동 계산 | ✅ 90/50 | — | RP 모드만 사용 | ✅ 일치 |
| 6 | **회전자 관성(armature)** | **벤치 실측 2026-09-09/10** | ⚠ 규격에 자리 없음 → 주석+사이드카 | ❌ **자리표시자** | **0.0773 / 0.0638** | ⚠ **§4** |
| 7 | **점성 감쇠** | 같은 실측 | ✅ **넣음(2026-09-15)** | ❌ 자리표시자 | 0.0275 / 0.0252 | ⚠ **§4** |
| 8 | **마른 마찰** | 같은 실측 | ✅ **넣음(2026-09-15)** | ❌ **아예 없음** | 0.519 / 0.251 | ⚠ **§4** |
| 9 | 위치제어 게인(Kp/Kd) | G1·K-bot 기준 질량 환산 (`pygmalion_constants.py:673`) | — 제어기라 자리 없음 | — | 150 / 220, 감쇠 6 | ✅ 설계 의도대로 |
| 10 | 토크-속도 곡선 | `Motor_Spec/RS0*_TN_curve_48V.csv` 실측 | — 자리 없음 | — | RS04 21점 · RS03 36점 | ✅ |
| 11 | 속도 한계 | 층이 셋 (카탈로그 무부하 / 실측 곡선 / `PYG_TN=0` 용 상한) | ❌ 20 (전 관절, 자리표시자) | — | **실측 곡선**(21·36점) | ⚠ **§6-1** |
| 12 | 폐루프 등식구속 | CAD 로드 기하 | 링크로 표현 | `equality/connect` 4개 | 같음 | ✅ (§3) |
| 13 | 전체 질량 | 위 합 | 23.63414 kg | 23.63014 kg | 23.63014 kg | ✅ 차이 설명됨(§3) |
| 14 | 메시 파일 이름 | CAD 내보내기 | `L_` 자산이 `R_` 파일 참조 | 같음 | 같음 | ⚠ 이름만, 뜻 없음(§6-3) |

---


## §1b 파라미터마다 — **파일·줄·변수명·지금 값**

§1 의 표는 요약이라 추상적입니다. 아래는 **프로그램이 파일에서 직접 읽어** 쓴 것이라,
값이 바뀌면 같이 바뀌고 변수가 사라지면 못 찾았다고 적힙니다.

```
python3 tools/robot_model/param_provenance.py --insert docs/145_model_parameter_provenance.md
```

<!-- PARAM-PROVENANCE:BEGIN (tools/robot_model/param_provenance.py) -->

> 이 절은 `tools/robot_model/param_provenance.py` 가 **파일에서 직접 읽어** 씁니다.
> 값이 바뀌면 여기도 바뀌고, 변수가 사라지면 못 찾았다고 적힙니다. 손으로 고치지 마십시오.

### 회전자 관성·점성·마른 마찰 (실측 정본)

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:110`
* **언제·어떻게** 2026-09-10 · 라즈베리파이에 CAN 어댑터 직결(제어 루프 약 4 kHz). 관성은 **두 경로가 합치는 것**으로 확인 — ① 토크를 끊고 타성으로 감속 ② 토크 대 가속의 기울기(마찰이 뺄셈에서 지워짐). 모델 없이 공진으로 재확인: kp=20 에서 2.48 Hz = sqrt(kp/J)/2π. 마찰·점성은 순수 타성 감속, 삼각파 지연(마찰/kp)과 1 % 안에서 일치
* **근거 문서** docs/130(방법) · docs/131(인수인계·감사)

```python
MOTOR_MEAS = {
  "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
  "RS03": dict(armature=0.0638, damping=0.0252, frictionloss=0.251),
}
```

### 같은 값의 두 번째 사본 (URDF 주입용)

* **정의 위치** `tools/robot_model/write_motor_params_xml.py:52`
* **언제·어떻게** 위와 같은 측정. `check_motor_params.py` 가 둘이 갈라지지 않는지 봄
* **근거 문서** docs/144

```python
  "bench-2026-09-10": {
    "RS04": dict(armature=0.0773, damping=0.0275, frictionloss=0.519),
    "RS03": dict(armature=0.0638, damping=0.0252, frictionloss=0.251),
  },
```

### 옛 값 (재현 전용, 학습 금지)

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:120`
* **언제·어떻게** 2026-09-12 · e3c67da2 이전 정책을 **자기 시절 모터**에 태워 재생할 때만. ★그 값 자체가 옮겨 적기 오류였음(벤치 실제값 0.030605/0.026008/0.526084)
* **근거 문서** docs/131

```python
if _MOTOR_MEAS_LEGACY:
```

### 카탈로그 관성 (PYG_MOTOR_MEAS=0 일 때만)

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:565`
* **언제·어떻게** RobStride 설명서. 실측 이전의 값이라 지금 학습에는 안 쓰임
* **근거 문서** —

```python
ARMATURE_04 = 0.0070
```

### 최대 토크·속도 한계 (카탈로그)

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:582`
* **언제·어떻게** RobStride 설명서. 파일 주석에 출처 링크가 적혀 있음. **속도 한계는 `PYG_TN=0` 일 때만 쓰임** — 켜져 있으면 실측 곡선이 이김
* **근거 문서** pygmalion_constants.py 의 주석 링크

```python
ACTUATOR_04 = ElectricActuator(
  reflected_inertia=ARMATURE_04,
  velocity_limit=15.0,
  effort_limit=120.0,
)
```

### 실측 토크-속도 곡선

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:148`
* **언제·어떻게** `Motor_Spec/RS0*_TN_curve_48V.csv` (48 V 실측). 곡선은 **최고 토크가 꺾이기 시작하는 속도**부터 적혀 있고, 끝점으로 공식 무부하 속도를 붙임
* **근거 문서** Motor_Spec/

```python
def tn_curve(family: str) -> tuple[tuple[float, float], ...]:
```

### 공식 무부하 속도

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:145`
* **언제·어떻게** 카탈로그 값(둘 다 200 rpm = 20.94 rad/s). 곡선의 토크 0 끝점으로만 쓰임 — **실측 최고점은 이보다 낮음**(RS04 190 rpm @ 11.2 N·m)
* **근거 문서** Motor_Spec/

```python
NO_LOAD_RPM = {"RS04": 200.0, "RS03": 200.0}
```

### 엉덩이·무릎 위치제어 게인

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:670`
* **언제·어떻게** 2026-07-06 사용자 지시 · G1(100/2)·K-bot(100/10, 같은 RS04)을 질량비 51.5/35=1.47 로 환산 → Kp 150. 무릎만 220. Kd 는 G1 환산 3 ~ K-bot 환산 15 사이에서 6
* **근거 문서** pygmalion_constants.py 주석

```python
PYG_ACTUATOR_03_HIP_YAW = _pd(
  "RS03",
  target_names_expr=(".*_hip_yaw_joint",),
  # 2026-07-06 (user): 400/9 chirp gains were ~4x G1 -> lowered to reference-scaled
  # 150/6. G1 hip 100/2 & K-bot (same RS04) hip 100/10 at ~35 kg -> mass-scaled
  # x(51.5/35=1.47) -> Kp 150; Kd from G1-scaled 3 .. K-bot-scaled 15 -> 6
  # (K-bot over-damped, discounted). Uniform across hip DOFs like G1/K-bot.
  stiffness=150.0,
  damping=_HIP_YAW_KD,
  effort_limit=ACTUATOR_03.effort_limit,
  **_motor("RS03"),
)
```

### 발목 관절 게인 (2-RSU)

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:627`
* **언제·어떻게** G1 급 발목 강성에 맞춤. 최대 토크는 `mech_design_eval.py` 의 야코비 전치 계산 (RS03 두 개 공동구동, 크랭크 약 30 mm)
* **근거 문서** docs/87 §5

```python
ANKLE_2RSU_KP = 28.5  # G1-class ankle stiffness (position gain, per DOF)
```

### 발목 최대 토크 (2-RSU 공동구동)

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:634`
* **언제·어떻게** `mech_design_eval.py` 의 야코비 전치 계산. 숙임은 모터 두 개가 같이 밀어 관절측 약 120 이 되는 것을 연속 정격으로 90 으로 낮춤
* **근거 문서** docs/87 §5

```python
ANKLE_2RSU_EFF_PITCH = 90.0
```

### 크랭크 게인 (발목에서 유도)

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:720`
* **언제·어떻게** 발목 관절 게인을 **실측 지렛대비 1.25(숙임)** 로 반사하고 모터 두 개로 나눔: 28.5 × 1.25² / 2. 손으로 고른 숫자가 아니라 유도값
* **근거 문서** docs/87 §5

```python
CRANK_KP = ANKLE_2RSU_KP * 1.25**2 / 2  # 22.3
```

### 관절 가동범위

* **정의 위치** `tools/robot_model/build_robot.py:82`
* **언제·어떻게** CAD 자기충돌 스윕(`rom_measured.json`) + 설계캡. 2026-09-02 에 사용자가 뷰어에서 관절을 **손으로 다 돌려 보고** 부호 방향이 뒤집힌 것을 찾아 축과 범위를 같이 정정
* **근거 문서** docs/107 (전 관절 ROM 표)

```python
JOINT = {   # child body -> (joint name suffix, axis in sim frame, range rad, inherits old model)
    'hip_pitch_link': ('hip_pitch', (0, 1, 0), (-2.18166, 0.523599)),
    # 2026-09-02: hip_roll, hip_yaw, knee and waist_yaw axes flipped (see below) after the
    # user drove every joint by hand in tools/viewer/mjcf_joint_viewer.py and found the
    # positive-qpos direction backwards against the design intent. Flipping the axis alone
    # was tried first on the theory that a range bound is just a physical stop distance and
    # does not care which sign is positive -- the user found that WRONG by direct viewer
    # check: the (lo, hi) bound must be negated and swapped too (new = (-old_hi, -old_lo)),
    # same as flipping any signed axis value elsewhere in this file. See SIGN_FLIPPED below,
    # which is where that renegotiation actually happens (against DESIGN_CAP and against the
    # measured rom_measured.json values feeding the assert, both of which predate the flip
    # and are still written in the OLD sign convention).
    'hip_roll_link': ('hip_roll', (-1, 0, 0), (-0.785398, 0.436332)),
    'thigh': ('hip_yaw', (0, 0, 1), (-0.872665, 0.872665)),
    # knee: the shin knee plates meet the thigh clevis plates at about -120 deg (mesh
    # interference check, red team 2026-08-20) - the old -140 is not reachable on this CAD
    'shin': ('knee', (0, 1, 0), (-2.094395, 0.0)),
    # ankle pitch: the DESIGN cap is -50/+30 (docs/71 s8g, docs/76 s12), not the old +40
    'ankle_pitch_link': ('ankle_pitch', (0, -1, 0), (-0.872665, 0.523599)),
    'foot': ('ankle_roll', (-1, 0, 0), (-0.349066, 0.349066)),
    # Upper body. Signs follow the legs so the two halves read the same way: pitch +q =
    # extension (limb back), roll +q = adduction (limb toward the centreline, axis flipped
    # on the right), yaw +q = turn left. Ranges are the CAD collision sweep (rom_check.py).
    'torso': ('waist_yaw', (0, 0, -1), (-1.047198, 1.047198)),
    # shoulder_pitch: the L axis below is correct as-is (user confirmed); only R was found
```

### 설계캡 (범위의 상한)

* **정의 위치** `tools/robot_model/build_robot.py:121`
* **언제·어떻게** 관절마다 기구가 허용하는 끝값. hip_roll 내전 +25° 는 2026-07-13 에 gen21 앵커의 밀기 회복 스텝이 +25 에 포화하는 것을 보고 **유지**로 확정
* **근거 문서** docs/experiments/2026-07-13_gen21_bent_p2.md §11

```python
DESIGN_CAP = {
    # the 2-RSU ankle is a CLOSED chain; a serial sweep drives the foot into push rods that
    # would really have followed it, so its range comes from the mechanism studies
    'ankle_pitch': (-50.0, 30.0),      # docs/71 s8g, docs/76 s12
    'ankle_roll': (-20.0, 20.0),       # JS6 clevis swing cone, docs/74 (PYG_ANKLE_ROLL15 -> 15)
    # Geometry allows more than the design wants. Set by the user 2026-08-23; each sits
    # inside the measured stop, which the assert below enforces.
    'hip_pitch': (-120.0, 25.0),       # extension 25 (metal at +26)
    # hip_roll, hip_yaw, knee, waist_yaw: these four bounds are already written in the
    # POST-FLIP convention (2026-09-02) -- e.g. hip_roll used to read (-85.0, 25.0) meaning
    # "abduction 85 (metal at -86) / adduction 25"; with the axis flipped, abduction is now
    # the POSITIVE direction, so the bound became (-25.0, 85.0). SIGN_FLIPPED below applies
    # the same negate-and-swap to the matching rom_measured.json entry before the assert, so
    # both sides of that comparison stay in this same, current convention.
    'hip_roll': (-25.0, 85.0),         # adduction 25 / abduction 85 (metal at -86); the
                                       # inherited -45 had no basis at all - it was the old
                                       # MJCF's number, kept by mistake after a sweep that
                                       # stopped searching at -70 and so never found the
                                       # stop. Adduction 25 is the gen21 verdict (2026-07-13)
    'hip_yaw': (-45.0, 45.0),          # symmetric: unaffected by the negate-and-swap. metal
                                       # at -48 / +52
    'knee': (0.0, 120.0),              # no hyperextension (the sweep finds metal only at +2)
    # Latest walking-model trims (docs/107, user-adjusted 2026-08-28). Geometry permits
    # wider travel, but those wider values are not the policy/model contract.
    'waist_yaw': (-45.0, 45.0),        # symmetric: unaffected by the negate-and-swap
```

### 크랭크 범위

* **정의 위치** `tools/robot_model/build_robot.py:191`
* **언제·어떻게** 발목이 필요로 하는 것보다 넓게 두고 스윕으로 검증
* **근거 문서** docs/107

```python
CRANK_RANGE = (-1.2, 1.2)      # rad, wider than the ankle needs; verified by the sweep
```

### 모터 질량 (저울 실측)

* **정의 위치** `tools/robot_model/massprops_fusion.py:60`
* **언제·어떻게** 사용자가 저울로 잼. RS03 은 카탈로그 0.880 kg 보다 무거운 0.9195 kg
* **근거 문서** URDF 머리말 주석에 출처 문장이 그대로 들어감

```python
RS04_MEASURED_KG = float(os.environ.get('PYG_RS04_MASS_KG', '1.514'))
```

### 링크 질량·무게중심·관성의 원천 파일

* **정의 위치** `tools/robot_model/build_robot.py:4`
* **언제·어떻게** Fusion CAD 내보내기(`massprops_step.py` 가 STEP 에서 계산). 모터는 위의 저울 실측 질량으로 바꿔치기함
* **근거 문서** docs/62 계통

```python
  massprops   ~/pyg_fea/steps/robot_massprops_step.json (massprops_step.py) today; the
```

### 질량 무작위화 범위

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py:658`
* **언제·어떻게** `tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_*.json`. 부품별 공차·체결구 50 % 가정에서 나옴
* **근거 문서** 런의 launch_manifest.json 에 경로가 박힘

```python
    _mass_dr = Path(
      os.environ.get("PYG_MASS_DR_JSON")
      or (MJLAB_SRC_PATH.parents[3] / "tools" / "robot_model" / "mass_dr.json")
    )
```

### 액션 자르기 = soft 한계 공식

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:1058`
* **언제·어떻게** 중심 ± 0.5 × 범위 × 계수(0.9). mjlab `Entity.soft_joint_pos_limits` 와 **같은 식** 이라 정책이 통과하는 클램프와 시뮬레이터가 강제하는 클램프가 하나의 계약이 됨. 경계에 계수를 곱하는 것이 아님(비대칭 관절에서 갈림)
* **근거 문서** 런 노트 §1b-3

```python
def safe_target_clip() -> dict[str, tuple[float, float]]:
```

### 리셋·목표 웅크림 자세

* **정의 위치** `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py:801`
* **언제·어떻게** ★2026-09-14: 여기 발목 값이 **깊은 웅크림 것**이라 발바닥이 10.6도 들려 있었음. 얕은 웅크림은 발목 = 무릎 − 엉덩이 로 유도해야 맞음. v6 을 중단시킨 원인
* **근거 문서** docs/experiments/ABORTED_2026-09-14_19-37-39_legonly_ab_v6_standing_p1.md

```python
# PYG_INIT_MID=1 (2026-08-26): a MIDDLE init pose. The bent default puts a human-like
```

<!-- PARAM-PROVENANCE:END -->

## §2 기하·질량 — **독립 파서로 대조해서 완전 일치**

주장이 아니라 실행 결과입니다. `tools/robot_model/urdf_crosscheck.py` 는 URDF 를
**우리 코드가 아닌 MuJoCo 자체 URDF 파서**로 읽고, MJCF 는 평소 경로로 읽은 뒤 맞댑니다.

```
cd mujoco-sim/mjlab && CUDA_VISIBLE_DEVICES="" .venv/bin/python3 \
  ../../tools/robot_model/urdf_crosscheck.py \
  --tag=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop --fast
```

무엇을 보는가 / 결과:

| 검사 | 결과 |
|---|---|
| 관절마다 이름·세계좌표 회전축·회전중심·가동범위 | **회전축 0.0000도 · 회전중심 0.0000 mm** |
| 관절을 하나씩 가동범위 전체로 쓸어 보며 모든 바디의 세계 위치·자세 | **0.0000 mm / 0.0000도** |
| 모든 가동범위 안에서 **무작위 자세 200개** | **0.0000 mm / 0.0000도** |
| 바디마다 질량·무게중심·관성 | 질량차 **0.0e+00**, 무게중심 **0.0000 mm**, 관성차 최대 **2.2e−08** |

```
VERDICT MATCH: worst axis 0.0000 deg, anchor 0.0000 mm, sweep pos 0.0000 mm / 0.0000 deg
```

![[urdf_crosscheck_LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.png]]

겹쳐 돌린 영상도 함께 나옵니다 —
`docs/video/urdf_crosscheck_LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.mp4`
(MJCF 는 실선 메시, URDF 는 빨간 철사. 관절마다 쓸어 보며 겹칩니다).

**즉 "두 파일이 같은 로봇을 말하는가" 는 기하·질량에 관한 한 검증 끝났습니다.**
2026-09-15 에 URDF 에 `<dynamics>` 를 넣은 뒤에도 이 검사를 다시 돌려 MATCH 를 확인했습니다.

## §3 전체 질량이 4 g 다른 이유 — **URDF 규격 탓이지 로봇 차이가 아니다**

| | 값 |
|---|--:|
| URDF 링크 25개 합 | 23.63414 kg |
| MJCF 바디 21개 합 | 23.63014 kg |
| 차이 | **0.004 kg** |

바디별로 대조하면 **공통인 것은 전부 소수점까지 같고**(§2), 차이는 URDF 에만 있는 링크
넷입니다.

```
URDF 에만: {'L_rod_A_u': 0.001, 'L_rod_B_u': 0.001, 'R_rod_A_u': 0.001, 'R_rod_B_u': 0.001}
MJCF 에만: {}
```

로드 끝의 **유니버설 조인트**는 회전축이 둘입니다. URDF 는 관절마다 자식 링크가 반드시
있어야 해서 그 사이에 **1 g 짜리 더미 링크**를 하나씩 넣어야 하고, MJCF 는 한 바디에 힌지
둘을 달 수 있어 그럴 필요가 없습니다. 1 g 은 파서가 0 을 싫어해서 넣은 최소값입니다.

**로봇이 다른 것이 아니라 파일 형식이 다른 것입니다.** 전체의 0.017 % 이고, 그 링크는
회전 관성도 사실상 0 이라 동역학에 영향이 없습니다.

## §4 ★ 실질적인 불일치는 여기 하나였다 — 런타임 주입

**기하는 완전히 맞는데, 모터 물성은 파일에 없었습니다.** 우리 학습 도구(mjlab)가 파일을
읽은 **뒤에** 액추에이터 설정에서 집어넣기 때문입니다.

| 항목 | MJCF 파일에 적힌 값 | 학습이 실제로 쓰는 값 | 비율 |
|---|--:|--:|--:|
| 회전자 관성 (크랭크) | 0.005 | **0.0638** | **12.8배** |
| 회전자 관성 (로드) | 0.0005 | (수동 링크) | — |
| 점성 감쇠 (크랭크) | 0.2 | **0.0252** | 1/7.9 |
| 마른 마찰 | **없음** | **0.251 / 0.519** | ∞ |
| 구동기 | **0개** | 12개 위치 서보 | — |

그래서 **이 파일을 다른 시뮬레이터에 그대로 물리면 관절이 거의 무저항으로 도는 다른
로봇**이 됩니다. 사용자 지적(2026-09-15)이 가리킨 자리입니다.

**고친 것**(2026-09-15, `docs/144`): 표준 URDF 로 표현할 수 있는 둘 —
`<dynamics damping= friction=/>` — 을 현세대 URDF **9개**에 직접 써 넣었습니다. URDF 의
`friction` 은 MuJoCo 의 `frictionloss` 와 같은 뜻입니다. 회전자 관성은 URDF 규격에 자리가
없어 **관절마다 주석 + 옆 파일 `*.motor_params.json`** 으로 냈습니다.

**아직 안 고친 것**: **MJCF 원본 파일**은 자리표시자 그대로입니다. 바꾸면 학습 쪽 계약이
달라지므로 여쭙고 하기로 했습니다(`docs/144` §4).

## §5 모터 값이 한 출처에서 나오는가 — 검사 통과 (AB), 실패 (RP)

`tools/robot_model/check_motor_params.py` 는 값이 **여러 군데 손으로 적혀 갈라지는 것**을
막으려고 2026-09-11 에 만든 검사입니다(실제로 발목 포락표가 옛 값을 들고 있어 한 모형
안에서 4.2배 어긋난 적이 있습니다).

```
PYG_ANKLE_MODE=AB .venv/bin/python3 tools/robot_model/check_motor_params.py
```

**AB 모드(지금 학습하는 방식) — 통과**

```
측정값 (MOTOR_MEAS, 하나뿐인 출처):
  RS04  회전자 관성 0.0773   마찰 0.519    점성 0.0275
  RS03  회전자 관성 0.0638   마찰 0.251    점성 0.0252
발목 포락표 ankle_rp_envelope.json: RS03 J=0.0638 b=0.0252 tc=0.251  → 맞음
  크랭크 A·B     0.0638  0.251  0.0252
  엉덩이회전      0.0638  0.251  0.0252
  엉덩이앞뒤·좌우  0.0773  0.519  0.0275
  무릎           0.0773  0.519  0.0275
모든 관절이 같은 출처의 값을 씁니다. 학습을 걸어도 됩니다.
```

**직렬(RP) 모드 — 실패, 그대로 학습하면 안 됨**

```
  발목 기울임   0.0005   마찰 없음   점성 없음   ★
  발목 숙임     0.005    마찰 없음   점성 없음   ★
★★ 그대로 학습하면 안 됩니다 — 이 관절만 다른 물리로 학습됩니다
```

지금 계통은 AB 로만 학습하므로 당장 영향은 없지만, **RP 로 돌리려면 먼저 고쳐야 합니다.**

## §6 남은 불일치 세 가지

### §6-1 ⚠ 속도 한계 — 세 숫자가 있고, **뜻이 서로 다르다**

처음엔 "세 값이 어긋난다" 고 적었는데, 정의 위치를 찾아보니 **셋은 서로 다른 것을
가리킵니다.** 정정합니다.

| 어디 | RS04 | RS03 | 무엇인가 |
|---|--:|--:|---|
| `NO_LOAD_RPM` (`pygmalion_constants.py:145`) | 200 rpm = **20.94 rad/s** | 200 rpm | **카탈로그 무부하 속도.** 실측 곡선의 **토크 0 끝점**으로만 붙임 |
| 실측 곡선의 마지막 점 (`Motor_Spec/*.csv`) | 190 rpm = **19.9 rad/s** @ 11.2 N·m | 188 rpm @ 10.11 N·m | **실제로 잰 가장 빠른 점.** 여기서도 토크가 남아 있음 |
| `ElectricActuator.velocity_limit` (`:582`) | **15** | 20 | `PYG_TN=0` **일 때만** 쓰이는 평평한 상한. 지금 학습은 `PYG_TN=1` 이라 **안 쓰임** |
| URDF `<limit velocity>` | 20 | 20 | ❌ **전 관절 같은 값 — 유일한 진짜 자리표시자** |

즉 **모순이 아니라 층이 다른 것**이고, 문제는 URDF 의 평평한 20 하나입니다. 그마저도
**숫자 하나로는 옳게 적을 수 없습니다** — 실제 한계는 토크에 따라 변하는 곡선이고
(RS04 21점 · RS03 36점) 학습은 그 곡선을 씁니다. URDF 에는 곡선을 적을 자리가 없습니다.

**손대지 않았습니다.** 어떤 숫자를 넣어도 틀리기 때문입니다. 다른 플랫폼으로 옮길 때는
**`Motor_Spec/` 의 곡선 파일을 같이 가져가야** 합니다.

### §6-2 ⚠ MJCF 원본의 자리표시자 (§4)

### §6-3 ⚠ 메시 파일 이름이 좌우 반대로 보인다

2026-09-12 좌우 이름 바로잡기 때 **자산 이름만 바꾸고 STL 파일 이름은 그대로** 뒀습니다.
그래서 지금 `<mesh name="L_..." file="R_....stl">` 처럼 보입니다. URDF·MJCF 머리말에
적혀 있고, **자산 이름이 기준이라 물리에는 영향이 없습니다**(§2 의 0.0000 이 그 증거).
다만 **다른 사람이 파일만 보면 오해합니다.** 파일 이름까지 맞추는 것은 별건입니다.

## §7 다시 확인하는 방법

```
# 1) 기하·질량이 두 파일에서 같은가 (독립 파서 대조 + 영상)
cd mujoco-sim/mjlab && CUDA_VISIBLE_DEVICES="" .venv/bin/python3 \
  ../../tools/robot_model/urdf_crosscheck.py --tag=<모델이름> --fast

# 2) 모터 값이 한 출처에서 나오는가
PYG_ANKLE_MODE=AB .venv/bin/python3 tools/robot_model/check_motor_params.py

# 3) URDF 에 모터 값을 다시 써 넣기 (두 번 돌려도 같은 결과)
python3 tools/robot_model/write_motor_params_xml.py --in-place --sidecar \
  --xml pygmalion_locomotion/assets/pygmalion_v2/*proxyfix*.urdf

# 4) §1b 의 파일·줄·값 표를 다시 뽑기 (변수가 옮겨졌으면 "못 찾았습니다" 로 알려 줌)
python3 tools/robot_model/param_provenance.py --insert docs/145_model_parameter_provenance.md
```

## §8 안 한 것

* **상체 모델(FullDoF·SemiFullDoF)** 은 `urdf_crosscheck` 를 이번에 안 돌렸습니다.
  `<dynamics>` 는 같이 넣었지만 기하 대조는 LegOnly 만 확인했습니다.
* **실물과의 대조**는 못 했습니다 — 모터가 통신에 무응답이라
  ([[2026-09-14_motor_bus_silent]]) 관절 방향·영점을 손으로 재는 절차(`docs/143`)가
  아직 시작을 못 했습니다. **여기까지는 전부 "모형 안에서 서로 맞는가" 이지
  "모형이 실물과 맞는가" 가 아닙니다.**
* 링크 질량의 원천인 CAD 내보내기 자체는 이번에 다시 재지 않았습니다(`docs/62` 계통).
