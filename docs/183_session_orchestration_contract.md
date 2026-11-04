# 183. Project Pygmalion 세션 오케스트레이션 계약

> 작성: 2026-11-02  
> 목적: 한 세션에 CAD, 학습, 발바닥, 배포, 하드웨어 컨텍스트가 섞여 판단 오류가 생기는 것을 막는다.  
> 이 문서는 세션별 **책임 경계와 인계 형식**만 정의한다. 로봇 수치의 원본은 각 담당 문서와 현재 artifact이다.

## 1. 운영 원칙

1. `Pygmalion | Main Research Orchestrator`가 사용자와 소통하고, 우선순위·상충 판단·GO/NO-GO를 중재한다.
2. 전문 세션은 자기 영역의 조사·구현·검증만 수행한다. 다른 영역의 전제를 임의로 바꾸지 않는다.
3. 전문 세션은 전체 채팅을 복제하지 않고 아래 인계 형식으로 메인에 결과를 돌려준다.
4. 동일 파일이나 동일 worktree를 두 세션이 동시에 수정하지 않는다. 겹칠 때는 메인이 순서를 정한다.
5. 현재 측정·artifact·로그가 과거 설명 문서보다 우선한다. 핵심 주장은 `VERIFIED`, `DERIVED`, `UNKNOWN`, `CONTRADICTED`로 표시한다.
6. 학습/실험은 전용 보고서, 설정·checkpoint·hash, 지표, 실패 조건, 영상과 ELI5 요약을 남긴다.
7. 메인 세션은 긴 구현과 대량 로그 분석을 직접 떠안지 않는다. 결정에 필요한 요약과 증거만 유지한다.

## 2. 영구 세션 구성

### A. Pygmalion | Main Research Orchestrator

- 역할: 사용자 인터페이스, 연구 질문 정제, 작업 배정, 충돌 조정, 최종 설계 판단, 로드맵과 문서 상태 관리.
- 보유 컨텍스트: 최종 목표, 현재 최고 policy, 하드 게이트, 세션별 최근 handoff, 열린 사용자 결정.
- 하지 않는 일: 장시간 학습 실행, 대량 CAD dump 분석, 배포 코드 장기 구현, 개별 실험 로그 원문 보관.
- 주요 소유 파일: `docs/183_session_orchestration_contract.md`, 전역 decision/handoff 문서, 작업 추적 상태.
- 현재 상태: **이 Codex 작업**.

### B. Pygmalion | CAD·MJCF·URDF·IMU·Joint Convention

- 역할: Fusion MCP 읽기, body/occurrence 매핑, 질량·COM·관성, mesh, joint axis/sign/ROM, AB closed loop, IMU frame/site/noise, 3개 variant 생성·검증.
- 이름 계약: `FullDoF_`, `SemiFullDoF_`, `LegOnly_`; AB만 `_loop`; D455와 foot overlay는 별도 suffix.
- 기준: 물리 motor+는 로터면을 바깥에서 볼 때 CW. 이것과 URDF q+ attachment 방향을 분리 검증한다.
- 소유 경로: robot builder, `tools/robot_model/`, 생성 URDF/MJCF 및 `docs/181_mjcf_urdf_models.md` 계열.
- 현재 상태: 새 전용 작업 `01a197c1-0810-7853-828a-f644c07d8f93`으로 초기화함. 기존 IMU 작업은 `Pygmalion | Legacy IMU Dev (Reference)`로 보존한다.

### C. Pygmalion | RL Policy·Training·Evaluation

- 역할: observation/action 계약, reward, DR, curriculum, launcher-resolved config, GPU 자원, checkpoint, 평가와 재학습 결정.
- 기준 policy: `terrain10_union_dr_49999`; actor 44, critic 268, action 12, 50 Hz, action scale 0.25.
- 중요 구분: 현재 구현은 별도 teacher distillation이 아니라 privileged critic을 쓰는 asymmetric PPO actor-critic이다.
- 소유 경로: 기본 `mujoco-sim/mjlab` 학습 코드와 일반 policy/config/evaluation. 발바닥 전용 worktree는 D 세션 소유.
- 현재 작업: 49999 계약 동결, 새 발바닥 후보가 확정되기 전 불필요한 본학습 금지, 후보 확정 후 동일 조건 fine-tune/retrain 비교.

### D. Pygmalion | Foot Sole Design Study

- 역할: 발바닥 CAD/충돌 형상, v3/v5/v5fullm, heel/toe/arch/rubber/solref sweep, CPU/GPU 평가 큐, 접촉·충격·slip·energy·낙상 분석.
- 소유 경로: `Feet_design/Feet_Design_Workspace/`와 그 아래 `mjcf_sim/union_branch`, `study/`, foot 전용 보고서·영상.
- 현재 작업: #184/#189. 감사 시점의 CPU 평가를 보존하고, `rc=-9` pebbles 누락을 낮은 병렬도로 재실행하며 원격 GPU 결과를 hash와 함께 동기화한다.
- 정밀 복구 기준: `Feet_design/Feet_Design_Workspace/mjcf_sim/study/CONTEXT_RECOVERY_2026-11-02.md`. 이 파일의 연구 계약, evidence state, 재개 DAG와 자동 중단 조건을 우선한다.
- 경계: 전체 로봇 mass/variant 규약은 B 세션에서 받고, reward/학습 계약은 C 세션에서 받는다. 발바닥 연구가 이 둘을 임의 변경하지 않는다.

### E. Pygmalion | Sim2Real Deploy·HUPHY·CAN·Safety

- 역할: 실제 runtime, motor ID/sign/zero/clamp, actor observation parity, 50 Hz timing, HUPHY 통합, CAN 패킷/손실, staged hardware test와 안전 중단 조건.
- 소유 경로: 배포 adapter/runtime, `System_ID/can_link/`, Sim2Real 배포 보고서. 외부 HUPHY 저장소는 별도 허가 없이는 read-only.
- 현재 작업: HUPHY `2fdc139` 기준 CAN 패치 갱신, ROS crank A/B observation 문제 구분, hip-roll Kp×1.2 단일변인 시험 준비, 오른쪽 Crank A 추종 점검.
- 경계: 모터 파라미터 식별은 F 세션 결과를 사용하고, policy 계약은 C 세션 값을 그대로 소비한다.

### F. Pygmalion | Hardware System ID·Power·Battery

- 역할: RS03/RS04 식별, 온도·마찰·관성, 전원·회생·BMS·배터리·PSU, 실측 질량과 fastener/cable/payload 불확실성.
- 기준 실측: RS04 1.514 kg, RS03 0.9195 kg, PSU RSP-2000-48 1.95 kg.
- 소유 경로: CAN 통신을 제외한 `System_ID/`, battery/power/thermal 문서와 측정 산출물.
- 현재 작업: 14S 58.8 V와 모터 60 V 제한의 1.2 V margin 검토, 양방향 bus-current 측정, vendor CAD 질문, zero/hard-stop 및 실물 stance/sole tilt 측정 지원.

## 3. 메인으로 보내는 표준 인계

전문 세션은 종료 또는 중요한 판단점마다 아래 8개 항목만 보낸다.

```text
목표:
결론/판정: GO | CONDITIONAL GO | NO-GO
VERIFIED 사실:
DERIVED 해석:
UNKNOWN/CONTRADICTED:
변경 파일·commit·artifact hash:
실패 조건과 되돌리기:
메인이 결정할 질문 1~3개:
```

원문 로그, 긴 표와 영상은 파일 링크로만 전달한다. 메인은 숫자를 다시 베껴 적기보다 canonical artifact를 링크한다.

### 현재 Codex 작업 ID

| 역할 | 작업 ID |
|---|---|
| A. Main Research Orchestrator | `01a05b6e-c216-7580-8424-af4c0aa69599` |
| B. CAD·MJCF·URDF·IMU·Joint Convention | `01a197c1-0810-7853-828a-f644c07d8f93` |
| C. RL Policy·Training·Evaluation | `01a197be-d96d-7ca1-aa11-ca7fe31c100d` |
| D. Foot Sole Design Study | `01a197be-d964-7e42-8129-70165c42c33d` |
| E. Sim2Real Deploy·HUPHY·CAN·Safety | `01a197be-d96c-7533-b81f-4ed7dfa48336` |
| F. Hardware System ID·Power·Battery | `01a197be-d959-7e52-bc5f-d9b278c0117b` |
| Legacy IMU reference | `01a060d9-9bf1-74d3-9e01-664abee79a8e` |

## 4. 새 세션용 시작 프롬프트

새 작업 생성 시 아래 공통 머리말 뒤에 해당 세션의 §2 역할을 붙인다.

```text
Project Pygmalion의 전문 작업 세션이다. 먼저 docs/183_session_orchestration_contract.md와
자기 역할에 직접 연결된 최신 handoff/artifact만 읽어라. 메인 세션의 우선순위를 바꾸거나
다른 세션 소유 파일을 수정하지 마라. 현재 증거를 과거 문서보다 우선하고, 모든 중요 주장을
VERIFIED/DERIVED/UNKNOWN/CONTRADICTED로 구분하라. 결과는 §3 표준 인계 형식으로
Pygmalion | Main Research Orchestrator에 보고하라. ELI5 요약을 포함하라.
```

## 5. 현재 우선순위와 라우팅

1. D가 진행 중인 foot 평가를 보존·완결한다.
2. C가 동일 device/terrain 안에서만 후보를 비교하고 다음 학습의 단일 변인을 고정한다.
3. B가 채택 후보를 현재 variant/mass/inertia 계약에 통합한다.
4. E가 observation/action parity와 staged hardware test를 수행한다.
5. F가 실물 불확실성과 전원 안전 근거를 제공한다.
6. A가 결과를 종합해 최종 하체 시험과 full-mass welded-upper-body 단계의 진행 여부를 사용자와 결정한다.

## 6. 즉시 중단 조건

- action 12 또는 actor observation 44 계약이 설명 없이 바뀜.
- `LegOnly_` 결과를 full-mass welded-upper-body 최종 모델 결과로 오인함.
- 서로 다른 device/terrain 결과를 같은 통계 표본처럼 비교함.
- raw MJCF를 runtime motor-parameter injection 없이 독립 실행해 결론을 냄.
- 동일 worktree 또는 동일 실험 queue를 두 세션이 동시에 변경함.
- checkpoint/config/model hash 없는 학습·평가를 최종 근거로 사용함.
- Fusion의 rotor-face CW motor+와 URDF q+를 같은 부호라고 가정함.
