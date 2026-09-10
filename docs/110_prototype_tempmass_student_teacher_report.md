# `prototype-tempmass` 12-DOF 학습 준비·스모크 보고서

작성일: 2026-09-01 (KST)  
판정: **CONDITIONAL GO — 프로토타입 시뮬레이션/스모크만 허용, 본 학습과 Sim2Real은 결정 2건 대기**

## ELI5 한 장 요약

로봇에게 걷기를 가르칠 때, 시험 중인 학생(actor)은 실물 로봇에서도 볼 수 있는 계기판만
보게 한다. 정답을 채점하는 코치(critic)는 시뮬레이터의 숨은 정보까지 볼 수 있다. 이번
스모크에서는 이 계약이 실제로 `학생 45개 입력 / 코치 82개 입력 / 다리 모터 12개 출력`으로
구성되고 GPU에서 학습되는 것까지 확인했다.

다만 현재 코드는 **privileged critic 방식**이다. 숨은 정보를 보고 행동 정답을 만드는 별도
Teacher policy를 먼저 학습한 뒤 Student에게 증류하는 방식은 아니다. 또한 최신 부품명은
한쪽 CAD에서 9개 바디를 가리켜, 과거 설명의 “양쪽 합계 10개”와 맞지 않는다. 이 두 의미를
확정하기 전에는 장시간 본 학습을 시작하지 않는다.

## 1. 질량 모델과 URDF

태그는 `prototype-tempmass`이고 loop 모델 파일명은 `prototype-tempmass_loop`이다.

- 원본: `/home/syaro/pyg_fea/fusion/bodies_v4_printed.json`
- 질량 근거: `docs/aluminium_parts_3dprint_masses.xlsx`, `알루미늄 부품` 시트 D열
- D열의 뜻: 실제 알루미늄 개별 계측값이 아니라 **CAD 형상에 Al 6061 밀도를 적용한 값**
- 예외: Arm_A/B는 시트에 기존 알루미늄 가공품이며 CAD 질량을 그대로 쓴다고 기록됨
- 적용 범위: `AnkleA2B_*` 6개 + `FEET`, `Arm_A`, `Arm_B` = 한쪽 CAD 9개 바디
- 빌더가 이 한쪽 CAD를 좌우 대칭으로 조립함
- 전체 질량: 기존 v4 printed 31.316 kg → prototype 32.470 kg

| CAD body | 임시 Al 질량 |
|---|---:|
| AnkleA2B_A_Support / (1) | 40.82 / 40.82 g |
| AnkleA2B_A_R / (1) | 31.84 / 32.27 g |
| AnkleA2B_A_L / (1) | 32.03 / 32.27 g |
| FEET | 617.72 g |
| Arm_A / Arm_B | 54.76 / 41.55 g |

생성 모델은 32.470 kg, hip-to-sole 0.9030 m, stance width 0.2474 m, 12개 actuator로
컴파일·검증됐다. 상체와 waist yaw의 질량/관성은 포함하지만 해당 5개 관절은 런타임에서
용접되어 action에 들어가지 않는다. IMU site는 `base_link` 기준
`(-0.000001, 0.007078, -0.069500) m`로 유지됐다.

주의: 빌드 출력 중 “python total 32.037 kg”은 loop 독립 바디를 출력용 부분합에서 빠뜨린
표시값이다. MuJoCo 컴파일 및 validation이 읽은 전체 subtreemass 32.470 kg이 실제 모델값이다.

## 2. Policy 계약

### Student actor — 45D, 실물 배포 입력

| 순서 | 신호 | 차원 | 실물 출처/이유 |
|---:|---|---:|---|
| 1 | body angular velocity | 3 | EBIMU 보정 gyro raw |
| 2 | projected gravity | 3 | EBIMU quaternion에서 회전한 중력벡터; yaw/magnetometer 불필요 |
| 3 | motor position history `[q(t-1), q(t)]` | 24 | 12개 모터 encoder 2 frame, oldest→newest |
| 4 | previous action `a(t-1)` | 12 | controller가 이미 알고 있는 직전 명령 |
| 5 | velocity command `[vx, vy, wz]` | 3 | 상위 제어기의 목표값(센서가 아니라 과제 입력) |

actor에서 금지한 것: true base linear velocity, joint velocity, passive ankle pitch/roll,
rod hinge, contact/force, foot height/air-time, terrain scan, DR draw. 센서 이력은 속도 미분보다
encoder noise에 덜 취약하면서 짧은 동역학 정보를 제공한다. 이전 action은 actuator 지연과
정책 자신의 최근 의도를 복원하게 한다.

### Privileged critic — 82D

base linear/angular velocity, projected gravity, loop에서 관측 가능한 좌표 16개와 속도 16개,
last action, command, 양발 height/air-time/contact/contact-force, 마찰·질량·COM·push·DR factor를
사용한다. 이 신호는 value estimation과 advantage 계산에만 쓰며 배포 actor로 전달하지 않는다.

### Output — 12D

각 action은 default pose에 더하는 joint-position target이고 scale은 0.25 rad/action-unit이다.
모델/actuator 순서는 다음과 같이 고정한다.

`L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_crank_A, L_crank_B,`
`R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_crank_A, R_crank_B`

최종 exporter에는 이 이름/순서/scale과 observation normalization 통계를 함께 넣고, 실기에서
checksum/assert로 불일치를 거부해야 한다.

### 학습기

PPO actor/critic MLP는 각각 `512-256-128`, ELU, observation normalization이다. actor는 Gaussian
scalar std 1.0으로 시작한다. PPO는 clip 0.2, lr 1e-3 adaptive, gamma 0.99, lambda 0.95,
5 epochs, 4 minibatches, desired KL 0.01, grad norm 1.0, 24 steps/env이다.

## 3. 현재 선택한 Reward — 스모크에서 실제 로드된 22개

| 목적 | 항목(가중치) | 선정 이유와 주요 기준 |
|---|---|---|
| 명령 추종 | linear `+2`, angular `+2`, linear progress `+1` | exp 추종의 먼 오차 구간 gradient 소실을 progress가 보완 |
| 자세/생존 | upright `+1`, pose `+1`, low-base `<0.7 m` 종료 | 넘어짐·무릎 꿇기·비정상 자세 방지; pose 허용폭은 속도별 변화 |
| stall 방지 | stand-still `-1` | nonzero command를 무시하고 서 있는 local optimum 제거 |
| 보행 구조 | air-time `+1`, clearance `-2`, landing swing-height `-0.25` | 고정 gait clock 없이 실제 command에 따라 swing 생성; target height 0.10 m |
| 접촉 품질 | slip `-0.1`, soft-landing `-1e-5`, impact velocity `-0.5`, force cap `-0.01` | 미끄럼과 발 내리찍기 억제; cap threshold 420 N, excess clip 560 N |
| 하드웨어 실현성 | knee overspeed `-0.5`, thermal effort `-0.02`, joint limit `-1` | RS04 knee 19.9 rad/s 초과와 연속토크 비율 제곱, ROM 위반 억제 |
| 자연스러움/안전 | stance knee extension `-2`, body ang vel `-0.05`, angular momentum `-0.02`, action rate `-0.1`, self-collision `-1` | stance knee 25° 초과 굴곡, 흔들림, 급격한 명령, 충돌 억제 |
| 비활성 | torque limit `-0.0` | T-N actuator가 peak envelope를 직접 clamp; 별도 penalty는 현재 ablation상 꺼짐 |

보상은 “더 많을수록 좋다”가 아니다. 성공 판정은 total reward가 아니라 명령 추종, fall/low-base,
air-time, foot-impact/GRF, slip, knee/torque/speed envelope를 독립 metric으로 함께 통과하는 것이다.
발 접촉 센서는 decimation=4에 맞춰 물리 substep 4개(20 ms) history를 유지하도록 수정했다.
그러나 현재 active `contact_force_cap`과 `soft_landing`은 아직 `data.force`의 마지막 substep을
읽고 `force_history` peak를 쓰지 않는다. 즉 기록 누락은 막았지만 reward 연결은 남아 있다.
본 학습 전 `max-over-4-substeps`를 별도 ablation으로 보정하지 않으면 순간 충격 reward에는
여전히 50 Hz aliasing이 남는다.

## 4. DR — 정확한 순서와 범위

### P1: clean gait formation

- command curriculum만 사용: vx `[-0.8,0.8] → [-1.2,1.2] → [-1.6,1.6] → [-2,2] → [-2,2.5]`
- vy는 `±0.5 → ±0.6 → ±0.7 → ±0.85 → ±1.0`, wz는 최대 `±1.0`
- push/friction/encoder는 `dr_factor=0`으로 중립화
- 관성 DR은 P1에서 제거하도록 런처를 수정함
- critic DR 채널은 폭 유지를 위해 남지만 nominal/zero를 보고함

### P2: P1 checkpoint full-resume

- push velocity 최대: xyz `±0.7, ±0.7, ±0.4 m/s`, angular `±0.52, ±0.52, ±0.78 rad/s`, 1–3 s 간격
- foot friction: `0.3–1.2`
- encoder bias: `±0.015 rad`
- 위 세 항목은 P1 종료 iteration부터 10k iteration 동안 factor 0→1 ramp
- mass/COM/inertia: body별 `mass_dr_prototype-tempmass_loop.json`; 일반 link 질량 최소 `0.95–1.05`,
  pelvis/torso `0.95–1.15`, COM floor 일반 ±5 mm/trunk ±20 mm, inertia shape 제한 포함

중요한 정직성: 현재 `pseudo_inertia`는 P2 시작 때 full range를 한 번 샘플하며 위 ramp의 대상이
아니다. 최적해는 reset-safe pseudo-inertia를 factor와 함께 점진적으로 적용하는 것, 차선책은
현재처럼 clean P1 뒤 P2 경계에서 full 관성 DR을 넣고 첫 100 iteration의 fall/value-loss를
rollback gate로 감시하는 것이다.

## 5. EBIMU-9DOFV6 적용

공식 사양: 3축 gyro/accelerometer/magnetometer, Euler/quaternion/calibrated raw,
RHA2/RAA/AGC/AVC, 1–1000 Hz output, accel LPF 21–493 Hz, gyro ±125–2000 dps,
accel ±2–16 g, timestamp, ASCII/HEX/polling, UART 9600–921600 bps, 3.3–6 V이다.

정책에는 다음만 쓴다.

- gyro calibrated raw → angular velocity 3D
- quaternion → projected gravity 3D
- timestamp → 50 Hz control tick에 가장 가까운 sample 선택 및 stale/drop 검출

정책에 쓰지 않는다: vendor short-term velocity/position, magnetometer heading. 전자는 적분 drift,
후자는 모터/철 구조물의 자기 교란으로 Sim2Real gap이 커질 수 있다. 초기 권장값은 HEX binary,
1000 Hz output 후 timestamp 기반 50 Hz downsample, gyro ±2000 dps, accel ±16 g이며, LPF/AVC는
실제 진동 PSD와 end-to-end latency를 측정한 뒤 확정한다.

공식 자료: <https://www.e2box.co.kr/entry/AHRS-EBIMU-9DOFV6>

## 6. MCP 준비 상태

프로젝트 범위 `.codex/config.toml`에 Streamable HTTP 서버를 등록했다.

```toml
[mcp_servers.pygmalion_fusion]
url = "http://192.18.20.161:27182/mcp"
enabled = true
required = false
startup_timeout_sec = 5
tool_timeout_sec = 30
default_tools_approval_mode = "prompt"
```

기존 `tools/fusion/mcp_client.py`도 같은 `/mcp` URL에서 `initialize → initialized → tools/list`
순서를 지원한다. 이번에는 쓰기/execute 요청을 하지 않았고, `192.18.20.161:27182`와 오타 가능성을
검사한 `192.168.20.161:27182` 모두 5초 connect timeout이었다. 따라서 설정은 적용됐지만 서버
도달성은 **UNKNOWN/NO CONNECTIVITY**다. Codex/앱 재시작 뒤 활성화되며, 네트워크가 열리면 첫 요청은
`tools/list`까지만 수행한다.

OpenAI 공식 MCP 설정: <https://developers.openai.com/codex/mcp/>

## 7. GPU 스모크 결과

- GPU: RTX 5060 Ti 16 GB; 시작 전 free 15,699 MiB, utilization 0%, 50°C
- host: available RAM 약 18 GiB, workspace free 약 153 GiB, 다른 trainer 없음
- 1,024 env, 약 32k steps/s, actor 45 / critic 82 / action 12 확인
- P1 checkpoint: iteration 200
- P2: DR factor 200→320에서 0→1, iteration 399 checkpoint 생성
- 최종 스모크 metric: mean reward 33.89, episode length 593.6 steps, steady xy error 0.723 m/s,
  air-time 0.140 s, stance knee 16.9°, impact mean 0.191 m/s, slip 0.280 m/s

400-iteration 스모크는 성능 합격 시험이 아니다. stage promotion 4회가 강제로 일어났으며 실제
명령 추종과 안정성을 평가하기엔 너무 짧다. 의미는 “모델/관측/역전파/DR/full-resume/checkpoint가
끊기지 않았다”뿐이다.

## 8. 가장 큰 실패 조건과 실행 로드맵

1. **정책 의미 확정**: privileged critic이면 현 45/82 PPO를 유지. action-producing Teacher와
   distillation이면 현 구조로 본 학습 금지, Teacher→DAgger/distillation→Student evaluation 설계.
2. **부품 scope 확정**: 과거 10개와 최신 9 CAD bodies/side의 대응표를 받기. 그 전에는 temp tag 유지.
3. **접촉 reward 200 Hz 연결**: 4-substep history의 per-foot peak를 cap/landing metric에 연결하고
   기존 50 Hz 기준 weight·threshold를 재보정. shuffle/air-time 퇴행이 없을 때만 채택.
4. **P1 nominal 장기학습**: 12-action/45-observation checksum, seed 3개, command gate를 forced가 아닌
   성능으로 통과.
5. **P2 DR**: 최적안은 관성 DR도 ramp. 차선안은 boundary step + rollback gate.
6. **sim evaluation**: 정지/저속/전후좌우/yaw grid, push, friction corners, mass corners, sensor delay/drop.
7. **HIL**: IMU timestamp/axis/sign/latency, motor order/sign/scale, 50 Hz deadline, e-stop 검증.
8. **tethered real**: 낮은 command/torque부터 승격. temp-mass checkpoint는 최종 무테더 배포 금지.

NO-GO 조건: action/encoder 순서 불일치, IMU axis·timestamp 미검증, 센서 stale을 정상값으로 처리,
수동 ankle 좌표가 actor에 재유입, P2 경계 fall/value loss 급증, foot impact/GRF가 정상화되지 않은 채
reward만 상승, 단일 seed 성공, 질량 scope 모순 미해결, e-stop/torque-speed limit 미검증.

## 9. 근거 문헌

- Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*:
  <https://arxiv.org/abs/1710.06542>
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*:
  <https://arxiv.org/abs/2109.11978>
- Radosavovic et al., *Real-World Humanoid Locomotion with Reinforcement Learning*:
  <https://arxiv.org/abs/2303.03381>
- Siekmann et al., *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition*:
  <https://arxiv.org/abs/2011.01387>

## 10. 본 학습 전에 필요한 사용자 답변

1. “Teacher”는 **privileged critic**인가, 아니면 privileged observation으로 행동을 내는
   **별도 Teacher policy + Student distillation**인가?
2. 과거 “양쪽 합계 10개”와 최신 목록의 `AnkleA2B` 6 + `FEET/Arm_A/Arm_B` 3개(한쪽 CAD)의
   실제 부품 대응을 알려 달라. 특히 한쪽 다리의 “4개 연결부품”이 어느 CAD body/assembly인지 필요하다.
3. MCP 주소 `192.18.20.161`이 정확한지, 그리고 이 호스트에서 그 subnet으로 가는 route/VPN이
   열려 있는지 확인이 필요하다.
