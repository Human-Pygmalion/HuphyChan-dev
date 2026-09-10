# Pygmalion 회전부호·질량·DR·H/W 배포 감사

> **2026-09-02 정정:** 이 문서의 GPU FAIL, H/W runtime UNKNOWN, 구 Hip Pitch proxy, 오른쪽 모터
> normal mirror, “2 kg 모순” 판정은 폐기되었다. 최신 판정과 산출물은
> [`114_huphy_proxyfix_rotation_dr_audit.md`](114_huphy_proxyfix_rotation_dr_audit.md)를 사용한다.

작성일: 2026-09-02 KST  
명목 태그: `prototype-tempmass-motormeasured-armfix_v30`  
판정: **CONDITIONAL GO — 짧은 명목 시뮬레이션 진행 확인만 허용; 실물 구동/Sim2Real은 NO-GO**

## ELI5

로봇의 12개 버튼 순서와 학생이 보는 45개 계기판은 시뮬레이터 안에서 맞았습니다. 파란 부품은
한쪽 CAD 9개를 양쪽에 복제한 18개 알루미늄 배치로 확정했고, 모터 무게도 저울값으로 고쳤습니다.
하지만 나사 무게와 실제 모터 ID·영점·방향표는 아직 실물과 한 장의 표로 잠기지 않았습니다.
DR은 작은 측정 오차를 견디게 하는 우산이지, PLA와 알루미늄을 헷갈린 모델을 가리는 천막이 아닙니다.
따라서 걷기 학습 진행은 볼 수 있지만, 실물에 전류를 넣는 단계는 안전 계약이 완성될 때까지 멈춥니다.

## 1. 결정표

| 기준 | 상태 | 증거 | 결과 / 다음 행동 |
|---|---|---|---|
| AB action/actor 계약 | VERIFIED | 12 actions, actor 45D, critic 68D, 4 loop equalities | 학습 interface 사용 가능 |
| 상체 질량 유지·관절 고정 | VERIFIED | upper mass 12.24226 kg, runtime에서 waist/shoulder joint 삭제 | 12-action 목표와 일치 |
| 모터 실측 질량 | VERIFIED | RS03 0.9195 kg, RS04 1.514 kg | 새 immutable 모델에 반영 |
| 알루미늄 배치 | VERIFIED | 사용자 확정: 파란 9 CAD body × L/R = 18 placements | 이전 “10개 불일치” 폐기 |
| 회전 부호 | DERIVED / 사용자 검토 대기 | Fusion rotor normal + MJCF axis: 10 match, 7 inverted 후보 | 17개 영상 승인 후 sign table 동결 |
| fastener 총량 | CONTRADICTED 가능 | 모델 포함 2.431547 kg; 사용자 진술 2 kg | 2 kg가 실측 exact인지 scope 확인 |
| H/W runtime | UNKNOWN / FAIL | ONNX/JIT exporter만 있고 Pygmalion CAN/IMU safety runtime 없음 | 실물 배포 금지 |
| 현재 GPU | FAIL | `nvidia-smi`가 driver 통신 실패, EGL context 생성 실패 | 이 세션에서 GPU 학습 시작 금지 |

## 2. 회전방향 검토 세트

사용자 규약은 **로터 플랜지에서 외측으로 나가는 법선 `n`을 따라 바라봤을 때 CW가 +**이다.
이 해석에서는 회전벡터 `+axis = +n`이다. 오른쪽 normal은 단일측 Fusion CAD를 simulator sagittal
plane으로 대칭해 유도했으므로, 영상 승인이 최종 증거다.

모든 영상은 `0° → URDF/MJCF +상한 → -하한 → 0°`이며 각 이동 4.00초, 상·하한 hold 0.25초,
총 12.50초다. overview와 close-up에 대상 body를 빨강으로 칠하고 joint anchor를 십자로 표시하며,
caption에 joint, lower/upper, 현재각도, base-frame 위치, 부호 후보를 넣었다. GPU driver가 없어 같은
compiled MJCF를 읽는 CPU orthographic renderer를 사용했다. AB crank는 회전 부호 확인용 kinematic
render이며 loop constraint를 동역학적으로 settle한 보행 영상은 아니다.

| # | 모터 joint | ROM ° | 정적 normal/axis 판정 |
|---:|---|---:|---|
| 1 | waist_yaw | -45 / +45 | match |
| 2–5 | L/R shoulder pitch, roll | pitch -90/+60; roll -32/+30 | L pitch·R roll inverted 후보, 나머지 match |
| 6–9 | L hip pitch/roll/yaw, knee | -120/+25; -85/+25; -45/+45; -120/0 | L hip pitch inverted 후보, 나머지 match |
| 10–11 | L crank A/B | -68.75/+68.75 | A inverted 후보, B match |
| 12–15 | R hip pitch/roll/yaw, knee | 동일 | R hip roll·knee inverted 후보, 나머지 match |
| 16–17 | R crank A/B | -68.75/+68.75 | A match, B inverted 후보 |

artifact:

- `docs/video/rom_sign_prototype-tempmass-motormeasured-armfix_v30_loop_cpu/joint_index.png`
- `docs/video/rom_sign_prototype-tempmass-motormeasured-armfix_v30_loop_cpu/manifest.json`
- `tools/robot_model/motor_sign_convention.json`

영상 승인은 **기구학적 방향**만 확정한다. 실제 motor ID, encoder sign, mechanical zero와 통신 명령
부호는 별도 H/W map이며 영상 승인만으로 자동 확정하지 않는다.

## 3. 알루미늄·출력물 밀도

확정된 알루미늄은 한쪽 CAD 기준 다음 9개이며 모두 좌우 대칭 배치된다.

| CAD body | 한쪽 nominal g | 실물 배치 |
|---|---:|---:|
| FEET | 617.72 | 2 |
| AnkleA2B_A_R / Support / L | 31.84 / 40.82 / 32.03 | 각 2 |
| AnkleA2B_A_R (1) / Support (1) / L (1) | 32.27 / 40.82 / 32.27 | 각 2 |
| Arm_A / Arm_B | 54.76 / 41.55 | 각 2 |
| 합계 | 924.08/side | **1.84816 kg / 18 placements** |

이 값은 `prototype-tempmass`용 worksheet/CAD-derived nominal이며 개별 새 저울값이라는 뜻은 아니다.
전체 표와 solo render는 `tools/robot_model/fusion_snapshots/v30_inspection/part_inventory/`에 있다.

출력물 밀도가 미세하게 다른 것은 오류가 아니다. 기존 실측 11개 confident sample은 Al6061 대비
질량비 평균 0.329, 중앙 0.334, 범위 0.294–0.407, 표준편차 0.033이었다. 즉 평균 유효밀도는
0.887 g/cm³이지만, 같은 filament/settings라도 shell·top/bottom·infill과 형상비가 달라 part-level
solid fraction이 달라진다. 현재 규칙은 다음과 같다.

1. 신뢰 가능한 개별 실측이 있으면 `mass/volume`의 part별 밀도 사용.
2. 실측이 없거나 배정이 불확실하면 confident 평균 약 0.888 g/cm³ 사용.
3. 확정 알루미늄은 출력물 평균을 적용하지 않고 Al nominal 사용.

## 4. 새 질량 모델

| 항목 | 값 / 근거 |
|---|---|
| RS04 | **1.514 kg**, 사용자 실측; Full 모델 7개 |
| RS03 | **0.9195 kg**, 사용자 실측; Full 모델 10개 |
| PSU RSP-2000-48 | 1.950 kg, spec |
| Full/AB/RP total | **35.7071 kg** |
| upper retained mass | **12.24226 kg** |
| LegOnly total | **21.887 kg** |

기존 35.0491 kg 모델에서 RS04만 `7 × (1.514−1.420) = +0.658 kg`이므로 새 Full total은
35.7071 kg이어야 하며 generated readback과 정확히 일치했다. AB runtime contract도 action 12,
actor 45, equalities 4, upper welded/retained, total mass를 모두 PASS했다.

## 5. 나사 포함 여부

**포함되어 있다.** 현재 AB mass aggregation에서 screw-like CAD body는 물리 대칭 수량으로 870개,
합계 **2.431547 kg**이다.

| owner | 포함 kg |
|---|---:|
| pelvis / torso | 0.200874 / 0.150595 |
| hip pitch / hip roll | 0.243040 / 0.198572 |
| thigh / shin / foot | 0.377968 / 0.623006 / 0.251852 |
| shoulder pitch / arm | 0.358712 / 0.026928 |

다만 classifier에 배정되지 않은 raw CAD screw 37개, 0.062114 kg이 있고 그중 현재 visible인 EBIMU
screw 16개, **0.025040 kg**은 모델에서 빠져 있다. 사용자의 “2 kg”가 장착 나사 전체의 정확한 실측이면
모델은 포함분만 +0.431547 kg 과대이므로 CONTRADICTED다. 대략값이면 scope/tolerance만 기록하면 된다.
증거는 `fastener_mass_audit.json`이다. 임의로 2.0 kg에 맞춰 전 링크를 비례 축소하지 않는다.

## 6. Al/PLA를 넓은 DR로 모두 흡수할 수 있는가

기술적으로 넓은 분포는 만들 수 있지만 **production convention은 아니다**. 동일 형상의 출력물
0.89 g/cm³와 Al6061 2.70 g/cm³는 약 3배이고, 발/발목처럼 distal한 위치에서는 질량뿐 아니라 COM,
관성과 접촉 충격을 크게 바꾼다. 이를 각 part별 독립 randomization으로 섞으면 존재하지 않는 로봇들을
학습하고 nominal 성능·sample efficiency를 희생하며, 잘못된 CAD 소유권도 숨길 수 있다.

최적 규칙:

- P1: 새 35.7071 kg nominal, broad inertial DR 없이 gait formation.
- P2: 측정 오차만 randomize — 실측 motor 약 ±1–2%, CAD-derived Al ±2–3%, 개별 실측 print ±3%,
  평균밀도 print ±10% + batch 공통 ±3%.
- fastener/cable은 링크별 additive payload와 COM으로 모델링하고 실물 위치에 맞춰 correlated draw.
- 좌우 같은 print batch 공통오차와 2-RSU A/B 결합부는 shared/correlated draw.
- mass를 바꾸면 COM·inertia도 물리적으로 일관되게 함께 바꾼다.

현재 생성된 DR artifact는 measured propagation을 수행하지만 recommended floor가 일반 link
0.95–1.05, pelvis/torso 0.95–1.15와 COM ±5/±20 mm로 넓다. 더 중요한 문제는 분석에서 사용한
print-batch 공통오차가 runtime `pseudo_inertia`에서는 L/R body별 독립 draw로 풀린다는 점이다.
따라서 현 파일은 sensitivity/P2 초안이고, 최종 P2에는 correlated event가 필요하다.

근거: Tan et al.은 system identification, actuator/latency modeling과 물리 randomization을 함께
사용했고, Peng et al.은 dynamics randomization이 calibration error에 대한 적응성을 줄 수 있음을
보였다. Isaac Lab 공식 지침도 plausible uncertainty만 span하고, mass/inertia를 양수·물리 일관되게,
coupled mechanism을 coherent하게 randomize하라고 명시한다.

- <https://arxiv.org/abs/1804.10332>
- <https://arxiv.org/abs/1710.06537>
- <https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html>

## 7. 실제 H/W deploy 코드를 기준으로 본 누락

현재 저장소에는 policy ONNX/JIT exporter는 있지만 Pygmalion용 motor-bus/EBIMU 실시간 runtime과
safety state machine은 없다. 따라서 아래가 구현되기 전 실물 배포는 NO-GO다.

| 우선순위 | 누락/수정 | 통과 기준 |
|---:|---|---|
| P0 | motor ID→12-name→sign→zero→gear mapping | signed manifest + disabled/저전류 test log |
| P0 | E-stop/watchdog/fault state machine | DISABLED→HOMING→STAND→POLICY→FAULT, comm timeout에서 torque ramp-down |
| P0 | output clamp | raw action, target ROM, qdot, torque/current, temperature, voltage 모두 명시적 clamp |
| P0 | actor parity | 45-float order/units/frame/history/normalization checksum이 simulator replay와 bit/허용오차 일치 |
| P0 | EBIMU pipeline | sensor→base quaternion/gyro transform, rad/s, timestamp alignment, stale/drop rejection |
| P1 | timing | physics/control inner loop과 50 Hz policy deadline, zero-order hold, jitter/latency histogram |
| P1 | AB mechanism | crank encoder→passive ankle FK, closure/calibration residual과 singularity guard |
| P1 | policy package | model hash, obs/action names, scale 0.25 rad, default offsets, normalizer, rate, git/model tag metadata |
| P1 | logging/replay | raw packets→45D obs→12D action→targets를 monotonic timestamp로 기록·재생 |
| P2 | staged safety | torque-disabled replay→restrained low gain→supported stand→bounded step→tethered command box |

추가 핵심 결함: current runner의 `clip_actions`와 action-term `clip`은 unset이다. 정책 출력은
`q_target = q_default + 0.25 × action`이지만 inference mean에 하드 clip이 보장되지 않는다. 새 학습 전
joint별 target-safe raw-action 범위를 계약으로 정하고 simulator와 deploy 양쪽에서 동일하게 적용해야 한다.
Teacher/critic의 68D privileged input은 exporter/runtime actor에 절대 포함하지 않는다.

## 8. 실행 순서와 stop gate

1. **영상 승인** — 17개 joint 각각 `+상한`이 실물 CW+ 의미와 일치하는지 승인. 하나라도 불명확하면
   sign 수정 전 학습 checkpoint를 배포용으로 승격하지 않는다.
2. **H/W map 동결** — motor ID/sign/zero/ROM/limit table. restrained 단일 모터 test에서 인접 motor가
   움직이거나 current/angle sign이 다르면 즉시 E-stop.
3. **fastener 2 kg 해소** — exact 실측인지 대략값인지 결정. exact면 link별 장착 manifest로 nominal 재생성.
4. **action clamp + deploy parity test 작성** — 45D replay와 12D target digest가 일치하지 않으면 중단.
5. **GPU host 복구/자원 확인** — driver, VRAM, RAM, disk, 중복 trainer 확인. 현재 Codex host는 GPU NO-GO.
6. **P1 nominal smoke→학습** — 100/1,000 iter와 60분 gate에서 NaN, closure, saturation, fall, impact 검토.
7. **latency/IMU/motor log 측정** — 측정 분포로 noise/delay/drop DR를 정한다.
8. **P2 structured DR** — correlated inertial event, friction, encoder, latency. nominal eval은 별도로 유지.
9. **HIL/단계적 실물 시험** — 각 단계에서 watchdog/E-stop/limit log가 PASS해야 다음으로 이동.

가장 큰 병목은 reward가 아니라 **실물 interface/safety 계약의 부재**다. 최적해는 이 계약과 replay
harness를 먼저 구현하고 새 nominal로 P1부터 학습하는 것이다. 차선책은 현재 temp tag로 짧은 학습
progress/sensitivity만 확인하되 checkpoint에 `prototype-tempmass`를 유지하고 실물 배포를 금지하는 것이다.
