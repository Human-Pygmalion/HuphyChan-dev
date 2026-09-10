# Pygmalion A/B·플랜지 방향·외부 저장소 재감사

작성일: 2026-09-02 KST  
대상: `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop`, AB motor-space 12-action  
판정: **시뮬레이션 축선 PASS / 전 관절 motor sign UNKNOWN / 배포 NO-GO**

> **정정(2026-09-02):** 아래의 과거 sign 후보표는 폐기했다. 로터 플랜지 법선은
> `rotor relative to stator`의 raw +축만 정한다. URDF의 `child relative to parent` 부호를
> 얻으려면 로터/스테이터가 각각 어느 링크에 고정됐는지 나타내는 attachment parity가
> 추가로 필요하다. 이전 계산은 모든 관절에서 `child=rotor`를 암묵 가정했다.

## ELI5

모터 앞면을 정면에서 볼 때, 앞면에서 바깥으로 나오는 화살표는 우리 눈 쪽, 즉 화면 밖을 향한다.
그 상태에서 시계방향 회전의 회전축 화살표는 반대로 화면 안쪽을 향한다. 따라서:

```text
외측 플랜지 법선 n : 화면 밖(관찰자 쪽)
카메라가 보는 방향   : -n (화면 안쪽)
모터 CW(+) 회전축    : -n
```

이전 영상은 `n`을 화면 안쪽에 둬서 모터 뒷면에서 본 셈이었다. 그래서 `q+=CCW`라는 결론을
일괄 적용할 수 없으며, 이전 영상·manifest는 폐기한다.

발목 이름도 바로잡았다. **위 모터가 A**, **아래에서 발목에 가까운 모터가 B**다. 구 XML의
시각화 proxy 이름이 반대로 붙어 있었을 뿐, 그것을 이유로 CAN 매핑까지 교차하면 안 된다.

## 1. 확정한 A/B 계약

| Policy action | 물리 위치 | HUPHY 이름 | Left ID | Right ID | 상태 |
|---|---|---|---:|---:|---|
| crank A | 상단, 무릎 쪽 | `ankle_a` | 5 | 11 | **확정** |
| crank B | 하단, 발목 쪽 | `ankle_b` | 6 | 12 | **확정** |

Live Fusion occurrence도 상단 `Robstride RS03 - Ankle_A (1):1`(z 약 -500 mm), 하단
`Robstride RS03 - Ankle_A:3`(z 약 -600 mm)로 구분된다. 하단 occurrence의 CAD component 이름은
stale하지만, 물리 위치 기준 B 계약이 우선한다.

구 모델의 `L/R_ankle_b_motor`가 상단에, `L/R_ankle_a_motor`가 하단에 있는 것은 **내부 proxy
label 결함**이다. 이는 렌더 geom 탐색에만 임시 호환하고 runtime ID에는 전파하지 않는다.

## 2. 회전방향 — 기존 후보 전부 무효화

부호 정의는 `q_model = sign * q_raw + offset`이다. `q_raw+`는 사용자가 정한 “출력/로터
플랜지를 외부에서 정면으로 볼 때 CW”이다. 정확한 식은 다음과 같다.

```text
n                  = 로터 플랜지 외향 법선
raw rotor/stator + = -n
attachment parity  = +1: URDF child가 rotor측
                     -1: URDF child가 stator측
required q+ axis   = attachment parity * (-n)
sign               = sign(dot(URDF q axis, required q+ axis))
```

현재 Fusion 모터 STEP은 rotor/stator 내부 조인트가 없는 fused body다. MCP로 플랜지와 인접
링크는 찾을 수 있지만, 플랜지 법선만으로 attachment parity를 자동 확정할 수 없다.

Fusion MCP의 선택 로터면–링크 최소거리 검사는 다음을 확인했다.

| 모터 | 로터측 링크/부품 | 거리 | attachment parity |
|---|---|---:|---:|
| Hip pitch 구동 (`Hip_R`) | `HipPitch2Roll` | 0 mm | +1 (child=rotor) |
| Hip roll 구동 (`Hip_P`) | `HipRoll2Yaw` | 0 mm | +1 |
| Hip yaw | `HipYaw2Knee` | 0 mm | +1 |
| Knee | `Knee2Ankle` | 0 mm | +1 |
| Ankle A 상단 | `Crank_A` | 0 mm | +1 |
| Ankle B 하단 | `Crank_B` | 0 mm | +1 |

따라서 하체 한쪽의 attachment parity는 해결됐다. 그러나 이 데이터는 Fusion 경로상 **물리
`Actuators-LegR`**이다. 현재 생성기는 이 원본을 `L_*`로 내보내고(`L_foot y=-0.1237 m`),
mirror본을 `R_*`로 내보낸다(`R_foot y=+0.1237 m`). 교차검증 코드에도
`reference_R -> ours_L`가 명시되어 있다. 반면 배포 계약은 `L_* -> CAN-left`로 가정한다.
즉 CAD physical side, simulator label, CAN side가 서로 충돌하며 이것이 해소되기 전에는 위
한쪽 normal/parity를 어느 action에 붙일지 확정할 수 없다.

| Action | sign 후보 | 근거 | 배포 사용 |
|---|---:|---|---|
| L/R hip pitch, roll, yaw, knee | — | 한쪽 parity 확보; physical-side/CAN mapping 충돌 | **UNKNOWN** |
| L/R crank A upper | — | A=upper와 한쪽 parity 확보; physical-side mapping 충돌 | **UNKNOWN** |
| L/R crank B lower | — | B=lower와 한쪽 parity 확보; physical-side mapping 충돌 | **UNKNOWN** |

한쪽 하체 attachment parity는 해소됐지만 physical-side mapping P0가 남아 있으므로 최종
sign은 계속 UNKNOWN이다.

왼쪽 normal을 오른쪽으로 mirror하거나 그대로 복사하지 않는다. 구매 모터는 거울상 물체가 아니며,
오른쪽 설치 자세는 별도 occurrence/사진/실측으로 증명해야 한다.

## 3. `huphy_mjcf` 교차검증 결과

검토 버전: [`8fa92bea`](https://github.com/Human-Pygmalion/huphy_mjcf/tree/8fa92beae31d787724facd3e9206b1ffb29e53db)

### 3.1 형상

직접 anchor XYZ를 빼면 27–157 mm 차이가 보이지만, revolute anchor는 같은 축선 위 어느 점이어도
동일하다. 그래서 점이 아니라 **무한 직선 간 수직거리**를 계산했다.

| 항목 | 결과 |
|---|---|
| hip pitch/roll/yaw + knee 8축 방향 | abs(dot) = 1.0 |
| 최대 축선 수직거리 | **0.8 mm** |
| 판정 | **PASS** |

즉 현재 builder의 hip 축 동시교차 가정은 독립 모델 및 live Fusion과 양립한다. Hip Pitch actuator
body proxy가 과거 뒤·아래로 밀렸던 결함은 질량/렌더 위치 문제였지만, hinge 축선 자체의 P0 오류는
아니다. 재현 스크립트와 결과는 `tools/robot_model/crosscheck_huphy_mjcf.py`,
`tools/robot_model/huphy_mjcf_crosscheck.json`이다.

### 3.2 모터·동역학

독립 모델은 RS04에 `J=0.031950 kg·m²`, `b=0.027184 N·m/(rad/s)`,
`frictionloss=0.525318 N·m`; RS03에 `J=0.015183`, `b=0.022540`,
`frictionloss=0.287808`을 쓴다. 그러나 저장소에는 이 숫자를 만든 raw CSV/시험조건/fit residual이
없다. 따라서 **참고 nominal 후보**이지 측정 검증값은 아니다.

또 이 모델의 발목은 serial pitch/roll 2축이다. Pygmalion의 2-RSU closed-loop crank A/B topology,
branch, Jacobian, motor ID는 검증하지 못한다. 질량 문서도 알루미늄 8개/총 약 21.95 kg 버전이라 현재
알루미늄 18 placement 및 RS03=919.5 g, RS04=1514 g 실측 계약보다 오래됐다.

## 4. 요청한 두 저장소에서 가져올 것과 버릴 것

### `izione/Huphychan`

검토 버전: [`e486f69e`](https://github.com/izione/Huphychan/tree/e486f69ec58ac87fd17cb4f4a0e2ff4c975e884b)

가져올 패턴:

- torque-off 각도 모니터링/CSV 기록 (`monitor_angles.py`)
- 부팅 시 multi-turn 값이 360° 가까이 튈 때 기준점 근처로 wrap하는 방어
- 마지막으로 보낸 setpoint에서 시작하는 rate-limited ramp
- measured-state hard limit E-stop, command margin, near-limit damping

참고 가능한 기하 정보로 `A1.z=-255 mm`, `A2.z=-313 mm`라서 1/A가 위, 2/B가 아래라는 현재
위치 정의와 독립적으로 일치한다. 다만 파일 스스로 실제 leg와 CAN ID를 TODO로 명시하므로 ID 증거로는
승격하지 않는다.

그대로 쓰면 안 되는 값:

- `CALIBRATED=False`; sign/offset/limit가 모두 placeholder다.
- 단일 다리 6모터용이며 2-CAN 12모터 동기 barrier가 없다.
- ankle side, motor ID, handedness가 TODO다.
- `tmax=17/14 N·m`, `vmax=44/33 rad/s`는 RS02 계열 placeholder라 RS03/RS04에 맞지 않는다.

따라서 **안전 구조 아이디어는 유용, 현재 로봇 데이터 증거는 없음**이다.

### `izione/Robstride_example`

검토 버전: main [`3de717a`](https://github.com/izione/Robstride_example/tree/3de717a1a578f13a48446fa7c0eeb12fc277ac99),
sysid [`7ef0c3a`](https://github.com/izione/Robstride_example/tree/7ef0c3a27f0463bb3b3b86a9fdb441c81ac99bd1)

main의 `step_response.csv`와 actuator-net은 `Robstride_S02` 단진자 실험이므로 RS03/04 파라미터로
복사하지 않는다. 반면 sysid branch의 방법은 바로 재사용할 가치가 있다.

1. `velocity_hold_sweep.py`: 정상속도에서 `tau=b*w+tau_c*sign(w)`로 damping/friction을 분리한다.
2. `sine_freq_sweep.py` + `fit_sine_freq.py`: 여러 주파수의 position/torque phasor로
   `J,b,tau_c`를 공동 추정하고 condition number/residual을 남긴다.
3. 측정 torque replay로 MuJoCo 응답을 검증한다.

하지만 sysid branch에는 실제 CSV/JSON 결과가 하나도 commit돼 있지 않다. GitHub에서 RS03/RS04
raw step/sine/hold log도 추가 검색했지만, 모델·프로토콜 코드 외에 시험조건과 함께 공개된 재사용 가능
로그는 찾지 못했다. 이는 “존재하지 않는다”는 증명이 아니라 이번 검색 범위의 결과다.

안전상 중요한 함정도 있다. `velocity_hold_sweep.py`, `identify.py`, `sine_freq_sweep.py`의 기본
`vmax=44 rad/s`, `tmax=17 N·m`는 RS02 값이다. RS03은 encoding range `20 rad/s, 60 N·m`, RS04는
`15 rad/s, 120 N·m`를 명시적으로 넘겨야 한다. 이는 안전 운전 한계가 아니라 codec 범위이므로 실제
시험 command limit는 이보다 훨씬 낮게 별도 설정한다. `full_motor_test.py --motor-type 03|04`만 이
모델별 범위를 자동 선택한다.

## 5. RobStride 공식 자료가 확정하는 것

공식 [`Product_Information`](https://github.com/RobStride/Product_Information) 2026-07-13 자료:

| 모델 | peak | rated | no-load speed | catalog mass | 본 프로젝트 nominal |
|---|---:|---:|---:|---:|---:|
| RS03 | 60 N·m | 21 N·m @ 100 rpm | 195 rpm ±10% | 900±20 g | **919.5 g 실측** |
| RS04 | 120 N·m | 40/35 N·m @ 50 rpm | 200 rpm ±10% | 1420±20 g | **1514 g 실측** |

설명서는 운전 방향을 “output/exit shaft 쪽에서 본 CW/CCW”로 기술한다. 다만 명료한 `positive=CW`
화살표를 제공하지 않으므로 프로젝트의 CW(+) 계약을 공식 문서만으로 자동 확정하지 않는다. 또한
firmware 버전에 따라 zero 정확도, multi-turn overflow, CAN ID, PDO timing, zero deadband가 바뀌었고
일부 업그레이드는 mechanical zero 재설정을 요구한다. 배포 manifest에 **모터별 firmware version**을
반드시 추가해야 한다.

### 5.1 새 P0 — 현재 HUPHY MIT codec 표가 최신 공식 매뉴얼과 다름

HUPHY commit `00153a37...`의 `src/huphy/motors/robstride/tables.py`는 기본 11-bit MIT 경로에서
RS03/RS04에 공통 `vmax=33 rad/s`, `Kp max=500`, `Kd max=5`를 쓴다. RS04 `pmax`는 15 rad다.
그러나 공식 260713 매뉴얼의 **Command 3: MIT Dynamic Parameters**는 다음이다.

| 모델 | pmax | vmax | torque | Kp max | Kd max |
|---|---:|---:|---:|---:|---:|
| RS03 | 12.57 rad | 20 rad/s | 60 N·m | 5000 | 100 |
| RS04 | 12.57 rad | 15 rad/s | 120 N·m | 5000 | 100 |

설치 펌웨어가 최신 표를 따른다면 HUPHY의 `kp=10`, `kd=1`은 모터에서 대략 `Kp=100`, `Kd=20`으로
해석될 수 있다. velocity command는 RS03에서 약 0.606배, RS04에서 약 0.455배가 되고 feedback은
반대 비율로 부풀 수 있다. RS04 position도 약 0.838배가 된다. 이는 단순 tuning 오차가 아니라 codec
scale 계약 오류다.

따라서 다음을 확인하기 전 HUPHY v1.0 torque enable은 **절대 NO-GO**다.

1. 12개 모터 각각 firmware version과 현재 protocol(MIT/private/CANopen)을 읽어 manifest에 고정한다.
2. 해당 firmware의 공식 표로 `pmax/vmax/tmax/kpmax/kdmax`를 선택하며 조용한 fallback을 금지한다.
3. torque-off feedback decode, 매우 낮은 gain의 +1° nudge, velocity/torque echo를 raw CAN과 함께 비교한다.
4. codec table을 수정한 뒤 모델별 golden-vector test를 추가한다.

`s2015-turtle/robstride_ros2`와 `Robstride_example/full_motor_test.py`도 최신 공식 범위인
RS03 `20/60`, RS04 `15/120`, 위치 ±4π를 사용하므로 독립 구현 교차검증과 일치한다. 단, 제3자
구현은 공식 매뉴얼을 대신하지 않는다.

## 6. 다음 실행 gate

1. **우선 HUPHY MIT codec을 모터별 firmware/공식 매뉴얼과 일치시키고 golden-vector로 검증한다.**
2. 오른쪽 6개 모터의 rotor face occurrence/사진을 직접 취득해 normal을 확정한다.
3. 로봇을 지지하고 링크 간섭을 제거한 뒤 torque/current 제한, 낮은 Kp/Kd로 모터 한 개씩 raw +1–2°만
   움직인다. 전 ROM 자동 sweep은 금지한다.
4. 각 모터에서 `ID, firmware, raw angle before/after, face-on CW 여부, zero reference, ROM margin`을
   한 행의 calibration artifact로 저장한다.
5. A upper=5/11, B lower=6/12를 포함한 전체 12-action permutation을 software loopback으로 검사한다.
6. RS03 한 개와 RS04 한 개를 대표로 velocity-hold + multi-frequency sine sysid하고, 좌우/개체 편차는
   추가 샘플로 envelope화한다.
7. 이 결과가 채워질 때만 `hardware_sign:null`을 숫자로 승격하고 policy torque enable을 허용한다.

최적해는 위의 물리 실측을 완료하는 것이다. 차선책은 현재 sign 후보를 simulation visualization과
offline deploy-code test에만 사용하고, 실제 torque enable은 계속 fail-closed하는 것이다.

## 7. 산출물

- 교정된 flange 계산: `tools/robot_model/motor_sign_check.py`
- live flange manifest: `tools/robot_model/rotor_faces.json`
- 독립 MJCF 비교: `tools/robot_model/huphy_mjcf_crosscheck.json`
- 배포 계약 v2: `tools/robot_model/fusion_snapshots/v30_inspection/runtime_contract/HUPHY_AB_deploy_contract_proxyfix.json`
- 새 grid/video: `docs/video/rom_axis_grid_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop_reaudit_faceon_v2/`
