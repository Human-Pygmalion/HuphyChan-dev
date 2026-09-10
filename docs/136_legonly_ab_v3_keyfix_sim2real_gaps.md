# 136. `legonly_ab_v3_keyfix` Sim2Real 대응/격차 + numpy 검증 결과

> `legonly_ab_v3_keyfix` 정책의 각 시뮬레이션 요소가 실제 하드웨어의 무엇에 대응하는지,
> 그리고 어디서 격차가 있는지를 정리한다. 스키마 자체는
> [[135_legonly_ab_v3_keyfix_schema_reference]], 개념 설명은
> [[134_legonly_ab_v3_keyfix_policy_eli5]] 참고.
>
> **범위**: 이 문서는 대응관계와 격차를 **기록**만 한다. `schema.py`나
> `tools/pygviewer/pygviewer/bridge/` 아래 파일은 건드리지 않았고, 실기 wire에 정책
> 출력을 실제로 연결하는 작업은 이번 범위 밖이다.

## 0. 이 정책이 실제로 상체가 없는 것에 대하여

`legonly_ab_v3_keyfix`는 12관절(다리만) 정책이다. "전신(fullbody)" 액추에이션
정책은 이 프로젝트에 **존재하지 않는다** — 코드 조사로 확인(2026-09-10): 완주한
학습 런 중 `PYG_UPPER_DOF=1`로 실행된 것은 하나도 없고, 유일하게 그 플래그를 쓴
`v2u1_contract_test`는 인프라 검증용 미완주 스모크다. FullDoF/SemiFullDoF는 상체가
**용접(고정)**된 채 무게만 얹힌 모델이며, 액션/관측 스키마는 이 문서가 다루는
12-DOF와 완전히 동일하다. 따라서 이 문서는 상체를 다루지 않는다.

## 1. 관측 항목별 실기 대응표

| sim 관측 항목 | sim 단위/원천 | 실기 대응 | 실기 단위/원천 | 격차 |
|---|---|---|---|---|
| `base_ang_vel` | rad/s, `robot/imu_ang_vel` 센서 | `ImuState.gyro_dps` | **deg/s**(HUPHY `sensors/base.py`) | **단위 변환 필수**(×π/180 실기→sim 방향) |
| `projected_gravity` | 단위벡터, `robot/imu_upvector`(framezaxis) 부호반전 | `ImuState.gravity` | 단위벡터, "정책이 쓰는 유일한 자세 값"(HUPHY 자체 docstring) | 개념적으로 동일. sim은 특정 IMU 사이트의 framezaxis, 실기는 `Imu.read()` 백엔드(벤더별 EBIMU 등)가 자체 계산 — 부착 좌표계 정렬이 실측 필요 |
| `motor_pos_history` | rad, `joint_pos - default_q`, 관절엔코더 | `MotorState.position_deg` | **deg, "raw 공간"**(HUPHY 자체 docstring — 캘리브 전 원시각) | **단위 변환 필수** + `wrap180`(±180° 접힘, §5) + `joint_map_huphy.json`의 `sign`/`offset_rad`로 raw→sim 좌표 변환 필요 |
| `actions` | 무차원, 신경망 raw 출력 | 없음(정책 내부 상태) | — | 센서 아님 — 실기가 직접 잴 필요 없이 정책 실행기가 자기 상태로 들고 있으면 됨 |
| `command` | (vx, vy, ωz) | 사용자/상위 명령 채널 | — | 정책 외부 입력, 센서 아님 |

## 2. PD 법칙 비교 — 가장 큰 단일 격차

| | Sim(`TnPdActuatorCfg`, 이 체크포인트의 12관절 전부) | 실기(HUPHY MIT 모드) |
|---|---|---|
| 공식 | $\tau = K_p(q_{tgt}-q) + K_d(\dot q_{tgt}-\dot q)$, 그 다음 실측 토크-속도 곡선으로 clamp | $\tau = k_p(q_{tgt}-q) + k_d(0-\dot q) + \tau_{ff}$ (`motors/base.py::Gains`) |
| 계산 위치 | mjlab Python(`IdealPdActuator.compute`), 매 물리 서브스텝(200Hz) | 모터 펌웨어 내부, 명령마다 실림(레지스터 저장 아님) |
| 학습된 게인($K_p$/$K_d$) | hip_pitch/roll **150/6**, knee **220/6**, hip_yaw **150/6**, crank **22.3/1.4** (`pygmalion_constants.py`) | **현재 벤치 기본값 kp≈10/kd≈1**(`config/robot_v1.0.yaml`, "튜닝 전 시작값"), 브링업 초기엔 안전상 kp≤5로 더 낮춤(docs/123) |
| **격차** | — | **게인이 정책이 학습된 값과 전혀 다르다.** 이 게인 차이 하나가 실기 배포 전 가장 먼저 메워야 할 구멍이다 — 낮은 게인으로는 정책이 학습한 반응성을 재현할 수 없고(정지마찰도 못 이김, 벤치 실측: kp=20까지도 자유로터 정지마찰 미돌파), 학습 게인을 그대로 실기에 걸면 안전성 검증이 안 된 상태로 큰 토크가 나갈 수 있다. |

## 3. 모터 모델 불일치(요주의, 확정 아님)

`tools/pygviewer/pygviewer/bridge/joint_map_huphy.json`은 12관절 전부를 `RS03`으로
기록하고 있다. 하지만 sim 액추에이터 설정(`pygmalion_constants.py`)은
hip_pitch/hip_roll/knee에 **RS04급**(정격 120 N·m) 게인·행동을 가정하고, hip_yaw·
크랭크에만 RS03급(정격 60 N·m)을 쓴다. 이 매핑 파일 자체가
`"side_mapping_verified": false`, `"rom_deg": null`로 스스로 "미검증 잠정본"임을
표시하고 있어 **최종 하드웨어 BOM인지 현재 벤치의 임시 구성인지 이 문서만으로는
확정할 수 없다** — 실기 배포 전 실물 모터 구성을 다시 확인할 것.

## 4. `encoder_bias` — 실기 대응 없음

sim에서만 존재하는 도메인 무작위화 항(관절마다 ±0.015 rad 규모, 학습 후반부
`dr_factor=1.0`에서 최대로 적용됨). 목표각에서 빼는 형태로 들어가 있다
([[135_legonly_ab_v3_keyfix_schema_reference]] §6). **실기에는 대응하는 개념이
없다 — 항상 0으로 취급**해야 한다(이번 검증에 쓴 라이브 트레이스도 이 항을
꺼놓은 "정책 자체를 보여주는" 깨끗한 상태로 기록했다, §7).

## 5. `origin`/wire 격차 — 정책 출력은 아직 wire에 닿을 수 없다

`tools/pygviewer/pygviewer/schema.py`의 `JointTarget.origin`은
`Literal["manual", "script"]`로 타입이 고정돼 있다:

> "``origin``이 두 번째, 독립된 관문이다 — 미래의 어떤 호출자가 시도해도 정책
> 출력이 통과할 수 없다: `Literal` 타입 자체가 생성 시점에 `origin="policy"`를
> `pydantic.ValidationError`로 거부한다." (`schema.py`, `JointTarget` 클래스 docstring)

즉 **정책 출력이 wire에 도달할 수 없는 것은 현재 설계상 의도된 상태**다(관례가
아니라 타입 시스템 자체가 막음). 이 게이트를 여는 것은 이 작업의 범위 밖이며,
`schema.py`는 여러 세션이 공유하는 계약 파일이라 변경 전 `session_bus`에 사전
공지가 필요하다.

## 6. ±180° 감김(wrap) 버그 — 이 정책엔 도달 불가능

HUPHY 모터는 각도를 `[-180°, 180°)`로만 보고하며, 관절이 그 경계를 넘으면 표시값이
360° 건너뛴다(`wrap180`, `motors/base.py`). 이 체크포인트의 실제 목표각 범위
(§`safe_clip`, 도 단위로 환산):

| 관절 | 범위(도) | 최대 크기 |
|---|---|--:|
| hip_pitch | −112.8 ~ +17.7 (좌우 미러) | 112.8° |
| hip_roll | −19.5 ~ +79.5 (좌우 미러) | 79.5° |
| hip_yaw | −40.5 ~ +40.5 | 40.5° |
| knee | 6.0 ~ 114.0 (좌우 미러) | 114.0° |
| crank_A/B | −61.9 ~ +61.9 | 61.9° |

**전 관절이 ±180°에서 충분히 멀다(최대 114°)** — 이 정책의 정상적인 목표각이
그 경계를 넘을 일은 없다. 다만 이건 "이 정책엔 안전하다"는 뜻일 뿐, 이 버그 자체가
고쳐졌다는 뜻은 아니다(그 버그는 애초에 HUPHY의 raw 다회전 처리 특성이며, §7의
`clamp_jump`/`clamp_rate`처럼 코드로 고쳐진 것과는 다른 종류).

## 7. `max_delta_deg`/`clamp_jump` — 이미 수정됨(과거 문서 대비 정정)

과거 `docs/124_huphy_handoff_findings.md`가 적발한 문제("`max_delta_deg`가 속도
제한이라 문서화됐지만 실제로는 위치오차, 즉 토크를 제한한다")는 **이미 고쳐져
`biped` 브랜치에 병합됐다**(브랜치 `rate-limit-from-datasheet`, 병합 커밋
`b4582b8`, 2026-09-07). 현재 `guards.py`의 `clamp_jump`는 자기 docstring에서
스스로 "이것은 속도 제한이 아니다"라고 명시하고, 진짜 속도 제한은 별도의
`clamp_rate`(기준점이 "직전 명령"인 함수)로 분리돼 있다 — 코드로 직접 재확인
(2026-09-10). 이 문서를 읽는 사람은 과거 handoff 문서의 해당 절을 "이미 해결된
과거 이력"으로 읽어야 한다.

## 8. numpy 독립 재구현 — 검증 결과

`tools/policy_numpy_ref/`에 PyTorch를 쓰지 않는 순수 numpy 추론 재구현을 처음부터
독립적으로 작성했다(`tools/pygviewer/pygviewer/policy.py`는 재사용하지 않음 —
검증용 오라클로만 사용). 세 갈래로 교차검증했다:

### 8a. 단일 스텝, 32개 확률 관측치, 3자 대조

| 대조 | 최대 절대오차 | 평균 절대오차 |
|---|--:|--:|
| numpy(float64) vs 원본 `.pt`(PyTorch) | 7.60×10⁻⁶ | 3.20×10⁻⁶ |
| numpy(float64) vs 기존 ONNX(제2 오라클) | 7.60×10⁻⁶ | 3.13×10⁻⁶ |
| `.pt` vs ONNX(참고 — 기존 두 오라클끼리) | 2.15×10⁻⁶ | 8.97×10⁻⁷ |

세 구현 모두 이 프로젝트가 이미 쓰던 판정선(ONNX vs `.pt` ≤1×10⁻⁴,
`tools/pygviewer/tests/test_policy_parity.py`)보다 한 자릿수 이상 여유있게
일치한다.

**"바이트 단위로 완전히 동일"이 아니라 이 기준으로 판정하는 이유**: 실제 배포되는
신경망은 float32로 돈다. PyTorch의 BLAS 백엔드와 손으로 짠 numpy 행렬곱은 덧셈
순서가 다를 수밖에 없어, float32 반올림 차이가 나는 건 버그가 아니라 부동소수점
연산의 근본적 특성이다. numpy 쪽을 float64로 돌린 건 "더 정확해서"가 아니라(실제
배포망은 float32로 돈다) — 연산 순서 차이와 float32 반올림 차이를 분리해서 보기
위한 장치다.

### 8b. 연속 40스텝 실제 롤아웃 — 오차가 누적되는지

`motor_pos_history`가 직전 스텝의 실제 값을 물고 있는 구조라, 단일 스텝 검증만으론
"내 관측 조립 코드가 그 과거값을 스스로 잘 이어가는지"는 증명이 안 된다. 실제
40스텝 연속 롤아웃(`obs_parity.npz`)으로 두 가지를 검증:

1. **env가 이미 만든 관측을 그대로 신경망에 흘렸을 때**: 매 스텝 오차 최대
   6.53×10⁻⁷, 앞 20스텝 평균 3.18×10⁻⁷ vs 뒤 20스텝 평균 3.00×10⁻⁷ —
   **뒤쪽이 더 나쁘지 않다(오히려 미세하게 낮다)**. 누적되는 오차 없음.
2. **우리 자신의 관측 조립 코드(`obs.py`)가 raw 재료(자이로·중력벡터·관절각·
   직전행동·명령)만 가지고 40스텝을 스스로 재구성했을 때**: env가 기록한 진짜
   관측값과 매 스텝 최대오차 2.98×10⁻⁸(이건 `obs_parity.npz` 자체가 float32에서
   저장된 데이터라 float32 저장 반올림 수준일 뿐, 로직 오차가 아님).

### 8c. 행동 후처리 검증

합성 극단값(전부 0, 전부 ±1e6)으로 클립·오프셋 공식이 정확한지 단위 검증하고,
§7의 라이브 롤아웃 트레이스가 실제 sim이 계산한 `processed_target`/
`clipped_target`/`final_target`을 기록해 두었으므로 이것과도 직접 대조 —
전 스텝 일치.

**재현**: `pytest tools/policy_numpy_ref/tests/`(18개 테스트, 전부 통과, 2026-09-10).
숫자 원장: `tools/policy_numpy_ref/artifacts/tolerance_report.json`.

## 9. 문서 인프라 등록

이 3개 문서 + `tools/policy_numpy_ref/`는 `docs/66_experiment_registry.md` §11
"문서 인프라 (런 아님)"에 등록돼 있다.
