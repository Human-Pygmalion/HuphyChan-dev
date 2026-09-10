# Pygmalion HUPHY 배포·모터축·proxyfix·질량 DR 감사

작성일: 2026-09-02 KST  
대상: `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop` / AB 12-action  
판정: **학습 진행 확인 GO / 실물 Policy 구동 NO-GO**

> **2026-09-02 정정:** 이 문서의 최초판에 있던 A↔B 교차 매핑과 `outward normal을 화면
> 안쪽으로 둔 CW` 판정은 폐기했다. 물리 기준은 **상단=A/ankle_a/ID 5·11,
> 하단(발목 쪽)=B/ankle_b/ID 6·12**이다. 플랜지를 외부에서 정면으로 보면 외측 법선은
> 화면 밖을 향하며, CW(+)의 각속도 벡터는 `-normal`이다. 기존
> `rom_axis_grid_..._loop/` 영상·manifest는 배포 근거로 사용하지 않는다. 최신 재감사는
> `docs/115_motor_flange_huphy_crosscheck.md`를 따른다.

## ELI5 결론

GPU와 새 로봇 파일은 학습을 시작할 수 있는 상태다. 뒤·아래로 밀렸던 Hip Pitch 모터도 최신 Fusion
위치로 바로잡았다. 그러나 실물 쪽 12개 모터의 방향·영점·끝각이 아직 빈칸이고, 현재 HUPHY는 한쪽
다리 6개만 제어하는 프로그램이다. 따라서 시뮬레이션 학습은 진행하되, 아래 배포 계약의 빈칸이 채워질
때까지 Policy가 실물 모터에 토크를 넣어서는 안 된다.

## 1. 이번에 확정·수정한 것

| 항목 | 결과 | 근거/산출물 |
|---|---|---|
| GPU | **PASS** — RTX 5060 Ti, 16,311 MiB, driver 570.195.03 | host `nvidia-smi`; 이전 실패는 sandbox device 노출 문제 |
| Hip Pitch 모터 위치 | **수정** — 구 snapshot `[-27.325,122.880,6.196]` → live Fusion `[-29.625,69.347,59.563]` mm | 약 75.8 mm 오차라서 영상 착시가 아니었음 |
| Full/Semi/Leg URDF | **PASS** — movable 17/13/12 | `validation_v30_proxyfix.json` |
| AB runtime contract | **PASS** — action 12, actor 45D, critic 68D, loop equality 4 | 기존 runtime contract + 새 HUPHY manifest |
| HUPHY 코드 | **956 tests PASS** | commit `00153a37...`; host에서 전체 test 실행 |
| 실물 캘리브레이션 | **NO-GO** | v1.0 L/R JSON 모두 sign=+1 임시값, offset=0, zero 빈칸, limit=null |

수정 모델:

- Full serial: `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix.{xml,urdf}`
- Full AB loop: `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.{xml,urdf}`
- Semi: `SemiFullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix.{xml,urdf}`
- Leg: `LegOnly_prototype-tempmass-motormeasured_v30_proxyfix.{xml,urdf}`

기존 태그는 덮어쓰지 않았다. Full/AB 총질량은 35.7071 kg, LegOnly는 21.887 kg이다.

## 2. HUPHY에 Policy를 붙일 때의 정확한 계약

현재 [HUPHY](https://github.com/chanhue/HUPHY/tree/00153a37ca2875515ec7c54146509f6af674eda2)는
CAN, RobStride, EBIMU, calibration, ROM/jump guard라는 좋은 기반을 이미 갖췄다. 하지만 현재 runner는
한쪽 limb를 고르고 6-action/24·26D joint-space policy를 실행한다. 목표 Policy는 양쪽 다리의 **AB
motor-space 12-action / actor 45D**이므로 기존 `policy_motion`에 weight만 바꿔 끼우면 안 된다.

### 2.1 12-action → HUPHY 모터 매핑

| idx | Policy action | CAN / ID | HUPHY 이름 | 주의 |
|---:|---|---|---|---|
| 0–3 | L hip pitch/roll/yaw/knee | can0 / 1,2,3,4 | 동일 | RS04,RS04,RS03,RS04 |
| 4 | **L crank A (상단)** | can0 / **5** | **ankle_a** | 사용자 물리 정의 |
| 5 | **L crank B (하단·발목 쪽)** | can0 / **6** | **ankle_b** | 사용자 물리 정의 |
| 6–9 | R hip pitch/roll/yaw/knee | can1 / 7,8,9,10 | 동일 | RS04,RS04,RS03,RS04 |
| 10 | **R crank A (상단)** | can1 / **11** | **ankle_a** | 사용자 물리 정의 |
| 11 | **R crank B (하단·발목 쪽)** | can1 / **12** | **ankle_b** | 사용자 물리 정의 |

구 XML의 proxy geom 이름만 상단을 `ankle_b_motor`, 하단을 `ankle_a_motor`로 잘못 달고 있었다.
이 내부 시각화 이름을 물리 ID 계약으로 오인해 교차 매핑한 것이 오류 원인이다. 기계가 보는 12개 값은 순서대로
`q(t-1), q(t)` 24개이며 joint FK 값이 아니다.

### 2.2 actor와 output

- actor 45D: base angular velocity 3 + projected gravity 3 + motor position history 24 + previous
  raw action 12 + command `(vx,vy,yaw rate)` 3.
- Teacher/critic의 privileged 68D는 실물 actor/export에 포함하지 않는다.
- output: `q_target = q_default + 0.25 × raw_action` rad, 정확히 위 12개 순서.
- 최적 실행: CAN/IMU/guard 100 Hz, policy 50 Hz, 중간 tick은 zero-order hold. 차선은 50 Hz 단일
  deterministic loop이나 deadline/jitter 측정이 먼저 PASS해야 한다.

기계가 읽는 계약은
`tools/robot_model/fusion_snapshots/v30_inspection/runtime_contract/HUPHY_AB_deploy_contract_proxyfix.json`
에 고정했다. `hardware_sign`을 일부러 `null`로 두어 실측 전 토크 enable을 막는다.

### 2.3 반드시 고쳐야 할 HUPHY P0

1. `WholeBodyABRobot` 계층을 두어 can0/can1을 한 tick barrier에서 수집·명령한다.
2. `Leg.get_observation()`은 position에만 sign을 적용하고 velocity/torque에는 적용하지 않는다.
   `sign=-1` 모터의 Student 속도와 안전 telemetry가 반대로 들어가므로 함께 변환해야 한다.
3. joint-space ankle FK/torque path 대신 crank A/B encoder와 position target을 직접 사용한다.
4. IMU invalid/stale/timestamp jump/frame mismatch를 0값으로 대체하지 말고 FAULT로 닫는다. IMU mount도
   `right_leg`가 아니라 URDF의 실제 base/pelvis frame으로 한 번만 변환한다.
5. load 시 model·normalizer hash, 45/12 dimension, term/action order, unit, 50 Hz를 검증한다.
6. `raw action finite+clip → target ROM margin → slew/qdot → torque/current/temp/voltage/fault/age` 순으로
   clamp한다. 현재 raw action clip은 unset이다.
7. `DISABLED→CALIBRATED→HOLD→POLICY→FAULT` 상태기계, E-stop, heartbeat/watchdog, torque ramp-down을
   넣는다. Policy 모드에서 `--allow-uncalibrated`는 금지한다.
8. raw CAN/IMU → 45D → raw 12D → clamped target과 clamp/fault reason을 monotonic timestamp로 기록한다.

## 3. 모터 회전방향: 최초판 폐기, 재감사로 대체

최초판은 외측 법선을 화면 안쪽으로 놓아 플랜지 뒤쪽에서 본 셈이므로 모든 CW/CCW 표기가 반전됐다.
또 오른쪽 모터 normal을 왼쪽에서 자동 생성했는데, 구매 모터는 거울상 부품이 아니므로 이 추론도
증거가 아니다. 교정된 수학·직접 CAD 결과·독립 MJCF 교차검증 및 새 영상은
`docs/115_motor_flange_huphy_crosscheck.md`에 기록한다.

## 4. Hip Pitch 위치 결함의 원인

구 builder가 읽은 `motor_proxies_fusion.json`이 2026-08-22 snapshot이라 최신 Fusion 조립 위치를 반영하지
못했다. 최신 문서 `260819_HumanMesh_wUpper_URDFexport_v30 v1`의 occurrence를 read-only로 다시 추출했고,
그 결과 Hip Pitch 중심의 y/z가 각각 약 -53.5/+53.4 mm 이동했다. `proxyfix` 모델은 이 live 좌표를 사용한다.
향후 빌드는 mesh timestamp만 보지 말고 `document name + occurrence full path + transform digest`를 함께
manifest에 저장하고, 하나라도 바뀌면 proxy를 자동 재추출해야 한다.

## 5. 나사 50%와 Mass/CoM DR

“2 kg”은 대략 기억값이므로 기존 CAD 포함량 2.431547 kg과 모순으로 취급하지 않는다. CAD 집계에는
870개 물리 배치가 포함돼 있지만 SCM에 실제 나사의 약 50%만 들어갔을 가능성을 별도 구조 불확실성으로
계산했다.

- 한 개의 robot-wide completeness `c ~ U(0.5,1.0)`를 뽑고, CAD fastener mass/inertia만 `1/c`로
  늘렸다. 링크마다 독립적으로 나사를 만들어내지 않으므로 좌우·조립 상관관계를 보존한다.
- 최악 `c=0.5`이면 Full 질량 민감도 상한은 `35.7071 + 2.431547 = 38.138647 kg`(+6.81%)이다.
- 하체 95% 질량 scale: hip pitch 0.980–1.074, hip roll 0.983–1.082, thigh 0.983–1.096,
  shin 0.992–1.107, foot 0.989–1.139. 기존 일률 ±5%만으로는 부족하다.
- 나사 위치를 따라 계산된 추가 CoM 95% 편향은 thigh z +6.1 mm, shin z +2.2 mm, foot x -5.3 mm
  수준이다. 무근거로 모든 링크 COM을 ±큰 값으로 흔드는 것보다 이 방식이 물리적으로 일관된다.
- 미래 전장/케이블 변경은 알려진 장착점의 additive payload event로 별도 모델링한다. 임시 sensitivity로
  torso/pelvis COM ±20 mm, distal leg 기본 ±5 mm(측정이 끝날 때까지 최대 약 10 mm)를 쓰되 nominal
  성능 평가를 반드시 분리한다.

DR은 잘못된 nominal을 숨기는 도구가 아니다. 최적해는 실제 장착 BOM/질량/위치를 nominal에 반영하고
잔차만 randomize하는 것이다. 차선책은 지금처럼 `prototype-tempmass`를 유지한 sensitivity envelope로
학습 progress만 확인하며 checkpoint를 실물용으로 승격하지 않는 것이다. 이 원칙은 dynamics
randomization의 원 논문인 [Peng et al.](https://arxiv.org/abs/1710.06537) 및 온라인 적응을 병행한
[RMA](https://arxiv.org/abs/2107.04034)의 취지와도 일치한다.

## 6. 실행 순서와 중단 조건

1. `proxyfix_loop` 명목 smoke/train을 실행한다. NaN, loop residual, contact explosion, action saturation이면 중단.
2. 그리드 승인 후 12개 실물 motor ID·CW sign·zero·ROM을 저전류로 기록한다. 이 단계 전 torque-policy 금지.
3. HUPHY의 signed velocity, whole-body AB, 45D actor adapter, clamp/state machine/watchdog를 구현한다.
4. torque-disabled log replay에서 Python simulator와 HUPHY의 45D와 12 target digest를 비교한다.
5. latency/drop/IMU bias/CAN jitter를 실측하고 P2 DR 범위를 그 측정치로 교체한다.
6. supported low-gain hold → tethered stand → bounded command → 보행 순으로 한 단계씩 승격한다.

가장 큰 병목은 GPU나 reward가 아니라 **실물 calibration + 양다리 동기 배포 계약**이다. 학습 progress는
볼 수 있지만, H/W GO 조건은 `hardware_sign/zero/limits != null`, parity replay PASS, watchdog/E-stop PASS다.

## 7. 80점 보완 체크

이미 확보: 최신 proxy, 3종 URDF, 12-action/45D 계약, 실측 모터 질량, 구조화 fastener DR, 방향 검토 영상.

남은 20점:

- 물리 사진/저전류 로그로 12개 sign·ID·zero·ROM 확정.
- action clip 수치와 RS03/RS04 전류·온도·전압·torque limit 확정.
- EBIMU sensor→base extrinsic, timestamp age, 실측 noise/latency 확정.
- closed-loop ankle residual/singularity와 CAN dual-bus deadline 실물 검증.
- 실제 장착 나사/케이블/전장 payload를 링크별 BOM으로 갱신.

## 산출물

- 회전 영상/그리드: `docs/video/rom_axis_grid_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop/`
- 전체 렌더: `tools/robot_model/fusion_snapshots/v30_inspection/renders_proxyfix/Pygmalion_v30_variants_all_views.png`
- HUPHY 계약: `tools/robot_model/fusion_snapshots/v30_inspection/runtime_contract/HUPHY_AB_deploy_contract_proxyfix.json`
- DR JSON/그림: `tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_fastener50_prototype-tempmass.json`,
  `docs/img/mass_dr_fastener50_prototype-tempmass.png`
- URDF 검증: `tools/robot_model/fusion_snapshots/v30_inspection/validation_v30_proxyfix.json`
