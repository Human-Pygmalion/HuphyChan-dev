# 111. Fusion v30 → Pygmalion URDF 3종 재구축

> 상태: **최신 v30 메시 89/89 및 URDF 3종 staging 완료, 사용자 렌더 검사 대기**  
> 기준일: 2026-09-01 · 질량 태그: `prototype-tempmass`

## ELI5: 지금 무엇을 만드는가

CAD는 부품 창고이고 URDF는 로봇 사용 설명서다. CAD에 있는 나사 하나까지 URDF의 관절로 만들면
시뮬레이터가 지나치게 무거워진다. 그래서 CAD 부품을 “같이 움직이는 덩어리(link)”로 묶고, 덩어리
사이의 실제 회전축만 joint로 만든다. 질량은 버리지 않되 나사·모터의 복잡한 겉모양은 단순한 원통이나
충돌 도형으로 바꾼다.

이번에는 같은 하드웨어에서 세 설명서를 만든다.

| 모델 | 움직이는 관절 | 상체 처리 | 용도 |
|---|---:|---|---|
| `FullDoF_` | 17 | 허리 1 + 양팔 2×2, 목은 torso에 용접 | 전체기구 검증 |
| `SemiFullDoF_` | 13 | 허리만 가동, 양팔 roll joint 좌표 −15°에 용접, 목 용접 | 상체 질량을 가진 하체 보행 |
| `LegOnly_` | 12 | waist flange부터 위쪽 link·mass·visual·collision 모두 없음 | 순수 하체 정책/디버깅 |

파일명 계약은 접두사를 포함한다. 공통 revision tag가 `<revision>`이면 정식 파일명은
`FullDoF_<revision>`, `SemiFullDoF_<revision>`, `LegOnly_<revision>`이다. 2-RSU 폐루프
sidecar는 해당 Variant 이름 뒤에 `_loop`를 붙인다. 과거 Full 모델만 접두사 없이 생성된
`prototype-tempmass-...` 이름은 호환용 구명칭이며 새 산출물·뷰어·학습 실행에는 사용하지 않는다.

`FullDoF_`에 관절이 17개 있어도 현 보행 Policy의 action은 **항상 하체 12개**다. 자유도가 존재하는
것과 Policy가 그것을 명령하는 것은 별개의 계약이다.

## 1. Fusion 원본 보호와 출처

| 항목 | 값 |
|---|---|
| 원본 | `260819_HumanMesh_wUpper_OMAKASE_RrecoverFromCrash v30` |
| 원본 lineage | `GWnH8zg_RXG-AvybBDXcbQ` |
| 원본 상태 | saved, `isModified=False`, 외부 참조 35개 |
| 원본 저장 시각 | 2026-09-01 21:12:09 KST |
| 작업 사본 | `260819_HumanMesh_wUpper_URDFexport_v30 v1` |
| 사본 lineage | `lNAVK_XkSE2q1eyNqGIu9w` |
| 사본 폴더 | `Pygmalion_URDF_export_v30_20260901` |
| 사본 생성 시각 | 2026-09-01 21:50:46 KST |

Fusion의 preview deep-copy API는 이 설치에서 `Copy With External References Feature is disabled`로
거절됐다. 그래서 상위 설계를 새 lineage로 복사했다. 사본이 참조하는 외부 occurrence 544개를 종류별로
감사한 결과 전부 나사 34종이었다. 구조물 외부 참조는 없고 나사는 visual mesh에서 제외하므로 원본
부품을 편집할 이유가 없다.

`fusion2urdf`는 사본의 독립 비교 출력에만 사용한다. 현재 CAD는 occurrence 627개, 중첩 604개,
최대 깊이 5이고 `base_link` 컴포넌트가 없다. 평면 컴포넌트 트리를 요구하고 joint의 Component2를
parent, Component1을 child로 그대로 해석하는 구형 exporter를 최종 생성기로 사용하면 축·부모·폐루프가
손실된다. 최종 출력은 검증된 Pygmalion 빌더가 생성한다.

## 2. 좌표계와 관절 부호

CAD root에서 simulator로의 회전은 다음 하나만 사용한다.

```text
sim = (-CAD_y, CAD_x, CAD_z)
sim +X = 발끝/전방, +Y = 로봇 왼쪽, +Z = 머리/상방
```

CAD에는 한쪽 다리만 있고 `CAD x=-123.7 mm`다. 빌더가 이것을 좌우로 미러링한다. 현재 역사적 이름
계약 때문에 이 CAD 다리가 model `L_`에 배정된다는 점을 기하학적 좌/우와 혼동하면 안 된다.

| 관절 | L축(sim) | R축 차이 |
|---|---|---|
| hip_pitch | `+Y` | 동일 |
| hip_roll | `+X` | 부호 반전 |
| hip_yaw | `−Z` | 동일 |
| knee | `−Y` | 동일 |
| ankle_pitch | `−Y` | 동일 |
| ankle_roll | `−X` | 부호 반전 |
| waist_yaw | `+Z` | 중앙축 |
| shoulder_pitch | `+Y` | 동일 |
| shoulder_roll | `+X` | R은 부호 반전 |

모터 양의 회전은 rotor 바깥쪽 법선을 화면 안쪽으로 보았을 때 시계방향이다. 과거 독립 감사에서는
CAD의 `hip_pitch`와 `shoulder_pitch` 표기가 이 물리 convention과 반대일 가능성이 확인됐으므로 실제
모터 명령 매핑에서 다시 gate한다.

## 3. ROM 계약

| joint | limit [deg] | 근거 |
|---|---:|---|
| hip_pitch | −120 / +25 | CAD stop 안쪽 design cap |
| hip_roll | −85 / +25 | CAD stop 및 gen21 |
| hip_yaw | −45 / +45 | CAD stop 여유 |
| knee | −120 / 0 | 과신전 금지 |
| ankle_pitch | −50 / +30 | 2-RSU 기구 envelope |
| ankle_roll | −20 / +20 | JS6 cone/기구 envelope |
| waist_yaw | −45 / +45 | 보행용 trim, 기하는 더 넓음 |
| shoulder_pitch | −90 / +60 | 보행용 trim |
| shoulder_roll | −32 / +30 | CAD 금속 stop |

SemiFull의 “팔을 위로 15°”는 이전 CAD-COM 대조에서 양쪽 `shoulder_roll q=-15°`가 오차 0.000 mm였고,
0°는 66.8 mm, +15°는 132.5 mm였다. R축이 `−X`이므로 URDF fixed origin의 물리 회전은 L `−15°`,
R `+15°`지만 두 joint coordinate는 모두 `−15°`다.

## 4. 질량 처리 순서와 현재 결과

질량은 순서가 중요하다.

1. Fusion v30 raw body mass/COM/origin inertia
2. docs/89 출력물 밀도 survey를 JSON에 오프라인 적용 — CAD는 수정하지 않음
3. 기존 알루미늄 기록을 `prototype-tempmass`로 재적용
4. PSU `RSP-2000-48`: CAD 1.677251 kg → 사양 1.95 kg
5. 같은 형상·균일 밀도 가정으로 origin inertia 전체를 질량비 스케일, COM 유지

| 결과 | Full/Semi | LegOnly |
|---|---:|---:|
| pelvis | 4.841 kg | 3.357 kg |
| 한쪽 다리 | 8.865 kg | 8.865 kg |
| torso | 7.827 kg | 없음 |
| 한쪽 shoulder+arm | 2.129 kg | 없음 |
| 전체 | **34.654 kg** | **21.086 kg** |

LegOnly의 1.484 kg 감소는 `Baselink_toWaistYaw` flange와 RS04 waist-yaw motor를 pelvis에서 제거한
결과다. 상체 link만 삭제하고 pelvis mass를 그대로 두는 조용한 오류를 금지한다.

알루미늄 scope에는 아직 표현 차이가 있다. 사용자 설명은 실물 10개(foot 2 + ankle 연결 4×2)지만
현재 CAD manifest는 한쪽 CAD에서 `AnkleA2B` 6 body + `FEET/Arm_A/Arm_B` 3 body를 찾고 이를 좌우
미러링한다. 기존 값은 그대로 사용하되 이 CAD-body↔실물-part 대응은 최종 Sim2Real 전 확인 항목이다.

## 5. 메시 계약과 현재 진행

- Fusion viewport 기준 렌더 7방향 저장 완료.
- structural mesh 89/89 body 최신 tessellation 및 hash manifest 완료.
- motors/bearings/fasteners는 visual STL에서 제외한다. 질량은 각 link에 남고 motor는 해석 원통으로 그린다.
- 모든 STL은 staging에만 쓰며 사용자 렌더 확인 전 training asset으로 승격하지 않는다.
- MCP extractor에서 exception/traceback payload fallback을 제거했다. 정상 stdout만 쓰고 빈 응답 3회면
  캐시를 보존한 채 안전 중단한다. 수정 후 남은 12개를 취득했으며 MCP는 정상 유지됐다.
- Full/Semi는 최신 어깨 재작업 형상을 `torso.stl`에 합쳤고, LegOnly는 flange 위 42 body를 제외해
  50 body만 사용했다.

중요: Fusion 전체 렌더에 한쪽 기계 다리만 보이는 것은 누락이 아니라 CAD 구조다. 최종 URDF 렌더에서는
빌더가 그 다리를 좌우 미러링한 완전한 두 다리를 반드시 보여야 한다.

## 6. 자동 검증 결과

최신 v30 staging 메시와 `prototype-tempmass` 질량으로 생성기 계약을 검증했다.

| 모델 | links | joints | movable | fixed | 판정 |
|---|---:|---:|---:|---:|---|
| FullDoF | 18 | 17 | 17 | 0 | PASS |
| SemiFullDoF | 18 | 17 | 13 | 4 | PASS |
| LegOnly | 13 | 12 | 12 | 0 | PASS |

검사 항목: 단일 `base_link` root, link/joint 이름 중복, 모든 질량 양수, 관성텐서 양의 정부호 및
삼각부등식, ROM 일치, SemiFull ±15° 물리 회전, LegOnly upper link/mesh 부재.

## 7. 전문가라면 마지막까지 의심할 것

| 우선순위 | 의심점 | 실패 모습 | gate |
|---:|---|---|---|
| 1 | body가 두 link에 중복 귀속 | 총질량 정상처럼 보여도 관성·COM 오류 | body path 유일성 |
| 2 | 좌우 roll축 미러 부호 | 한쪽만 반대로 넘어짐 | FK/ROM 좌우 overlay |
| 3 | CAD→sim +90° 누락/중복 | 전진 명령에 옆걸음 | 발끝 +X regression |
| 4 | 숨긴 alternative branch 포함 | 수백 kg 또는 두 팔 중복 | 이름 기반 exclusion |
| 5 | light bulb를 suppression으로 오해 | 숨겨 둔 실제 나사 질량 소실 | view 상태와 설계 상태 분리 |
| 6 | Steel placeholder를 실재 재질로 신뢰 | 출력물 질량 3–9배 | survey provenance |
| 7 | 2-RSU를 plain URDF tree로 신뢰 | 발목이 비물리적으로 독립 회전 | MJCF equality sidecar |
| 8 | LegOnly에 flange/motor 관성 잔류 | CoM이 위로 치우침 | cut manifest |
| 9 | torso rework의 포함 여부 | visual과 inertia 불일치 | 사용자 렌더 gate |
| 10 | RS03 0.880 kg vs 실측 0.9195 kg | link별 약 40 g 편차 | hardware BOM 확정 |
| 11 | PSU 관성의 균일밀도 스케일 | 내부 부품 배치와 다른 tensor | DR 또는 실측 pendulum |
| 12 | visual mesh를 collision에 그대로 사용 | 접촉 폭증/GPU fallback | convex/primitives 별도 |

## 8. 완료 및 남은 실행 순서

완료: v30 사본 provenance 확인 → 89/89 cache → Full/LegOnly mesh+hull → 3종 URDF/MJCF →
hard gate → 전면/사시/측면 전체 렌더.

남음:

1. 사용자 visual inspection
2. `Torso2ShoulderP`를 visual뿐 아니라 torso mass/inertia에도 넣을지 기존 “ignore” 결정을 재확인
3. 승인된 staging만 training assets로 승격
4. MJCF 2-RSU equality 모델 및 12-action motor-space mapping 재검증
5. Student 센서 실기 raw packet·지연·축부호 확인 후 학습 재개

## 9. 80점 감사 후 보강 결과

| 항목 | 점수 | 판정 |
|---|---:|---|
| CAD 출처/사본 보호 | 20/20 | lineage·시각·저장 상태 확인 |
| 최신 visual geometry | 19/20 | 89/89, 3방향 렌더 완료; 사용자 검사만 남음 |
| 3종 kinematic contract | 20/20 | 17/13/12 movable hard gate PASS |
| mass/COM/inertia | 14/20 | PSU 반영; 알루미늄 매핑·torso rework·RS03 편차 미확정 |
| Sim2Real deployment contract | 12/20 | 12-action 분리는 유지; closed-loop/실기 축·지연 gate 남음 |
| **합계** | **85/100** | staging 검사용 GO, 학습 asset 승격은 아직 NO-GO |

80점에서 가장 크게 보강한 것은 최신 메시 완결성, LegOnly의 진짜 flange 절단, 세 변형 자동검증,
MCP 재발 방지다. 남은 15점은 렌더 승인이 아니라 물리 전제다. 특히 `Torso2ShoulderP` visual을 넣고
질량을 빼는 상태는 최종 모델에서 허용하지 않는다. 기존 제외 결정을 유지하면 visual에서도 빼고,
최신 구조물로 채택하면 mass/COM/inertia에도 넣어야 한다.
