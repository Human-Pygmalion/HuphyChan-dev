# 로봇 모델 파일(URDF · MJCF) 버전 대장

> 이 파일은 모델 파일과 같은 폴더(`pygmalion_locomotion/assets/pygmalion_v2/`)에 있습니다. 2026-11-04 에 `docs/186` 에서 옮겼습니다(`docs/186` 에는 안내만 남음).

2026-11-04 작성 · 소급 적용(처음 모델부터 지금까지) · 프로젝트를 몰라도 **"어떤 파일이 있고, 무엇이 왜 바뀌었나"** 를 알 수 있게 쓴 문서입니다.

> 이 문서의 규칙: 한 번 바뀔 때마다 **항목 1개**를 맨 아래에 더합니다(§6). 파일 이름이 이 문서에 없으면 점검 스크립트가 알려 줍니다(§7).
> **확인하지 못한 것은 "확인 못 함"이라고 적었습니다.** 추측으로 채우지 않았습니다.

---

# 1. 30초 요약

- **지금 쓸 최신 모델**: `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_loop` (항목 M19, 2026-11-04). 다리만 있는 로봇이고, 질량 26.91 kg입니다.
- **이 모델이 앞 모델과 다른 점 3가지**: ① 발바닥을 새 설계(v5)로, ② 엉덩이·무릎 보강재와 골반 카메라(D455)를 더함, ③ **다리 구조 부품 11종(좌우 22개)을 프린트 질량에서 알루미늄 질량(CAD 체적 × 2.70 g/cm³)으로** 바꿈(M18, M19). 보강재 이후 질량이 1.80 kg 늘었습니다.
- **원본은 지우지 않습니다.** 바꿀 때마다 새 이름의 새 파일을 만듭니다. 그래서 파일이 많습니다(§5에 전부 있습니다).
- **모든 파일은 손으로 고치지 않고 만드는 코드로 생성합니다.** 코드 위치는 각 항목에 적었습니다.

# 2. 용어 (이 문서에 나오는 말)

| 말 | 뜻 |
|---|---|
| **URDF**(`.urdf`) | 다른 프로그램(ROS, Isaac Sim 등)이 읽는 로봇 설명서. |
| **MJCF**(`.xml`) | MuJoCo와 mjlab(학습용 시뮬레이터)이 읽는 로봇 설명서. 모터 모양과 닫힌 발목 연결이 들어 있음. |
| **링크** | 같이 움직이는 부품 덩어리 하나. 예: `hip_roll_link`(엉덩이 회전 부분), `shin_link`(정강이). 부품마다 질량·무게중심·관성(돌리기 어려운 정도)을 하나로 합쳐 적습니다. |
| **CAD** | Fusion 360으로 그린 설계 파일. 모델의 숫자는 여기서 읽습니다. **CAD에는 다리가 한쪽만 있고**, 반대쪽은 좌우를 뒤집어 만듭니다. |
| **AB 발목 / `_loop`** | 발목을 모터 2개(크랭크 A, B)와 막대 2개로 닫힌 고리로 만든 실제 구조. 파일 이름 끝이 `_loop`이면 이것입니다. 없으면 **RP**(위아래+좌우 관절 2개를 직렬로 이은 단순판). |
| **FullDoF / SemiFullDoF / LegOnly** | 몸 범위. 전신(허리+양팔) / 허리까지 / **다리만**. |
| **`prototype-tempmass`** | 질량 규약. 알루미늄으로 설계했지만 실제로는 3D 프린터로 뽑은 부품은 **프린트 밀도(0.8874 g/cm³ 안팎)** 로 계산하고, 일부 부품(`AnkleA2B_*`, 팔 A/B, 발 `FEET`)은 임시로 알루미늄 질량을 씀. |
| **`motormeasured`** | 모터 질량을 저울로 잰 값으로: RS03 0.9195 kg, RS04 1.514 kg(CAD 자리표시자는 RS04 1.558 kg). `rs03measured`는 RS03만 실측한 앞 단계. |
| **`armfix`** | 팔 하우징 부품을 고친 판([[112_v30_armfix_rp_ab_part_review]]). 다리만 판에도 이름이 붙어 있으나 다리 값이 어떻게 달라지는지는 **확인 못 함**. |
| **`v30`** | 근거가 된 CAD 버전. 2026-09-01에 v30으로 다시 만들었음([[111_fusion_v30_urdf_rebuild]]). |
| **`proxyfix`** | 엉덩이 피치 모터 위치가 75.8 mm 틀렸던 것을 바로잡은 판(2026-09-02). |
| **`_d455`, `_footv5`, `_reinf`, `_alparts`** | 이 문서 M14~M18의 덧붙임. 뜻은 해당 항목에. |
| **Al 6061** | 알루미늄 합금. 밀도 2.70 g/cm³. |

# 3. 계보도 (위가 오래됨)

```
옛 계통(CAD 이전) pygmalion.xml, cant20/30, rolloff30, v1 ......... M01
  └ v2 (CAD에서 재구축, 직렬 발목)  v2.1 폐루프 ................... M02
      └ v3 (+Fusion 실측 질량) → v3_printed (프린트 밀도) .......... M03
          └ v3 + 폐루프 발목(_loop) ............................... M04
              └ v4_printed (상체 재측정, IMU 위치 수정) ........... M05
                  └ prototype-tempmass (일부 부품 알루미늄 질량) .. M06
                      └ v30 재구축(CAD v30) + rs03measured ........ M07
                          └ motormeasured + armfix ................ M08
                              └ proxyfix (엉덩이 피치 모터 위치) .. M09
                                  └ 9/3 수정 6종 + LegOnly ........ M10
                                      └ 9/12 좌우 이름 맞바꿈 ..... M11
                                          └ 9/15 모터 값을 URDF에도  M12
                                              ├ shell(겉껍데기) .... M13   (전신·허리까지만, 다리만 판과 별개 가지)
                                              └ _d455 (카메라) ..... M14
                                                  └ _footv5 (새 발) .. M15
                                                      └ _footv5_d455 (합본) ... M16
                                                          └ _reinf (보강재+카메라 질량) ... M17
                                                              └ _alparts (알루미늄 크랭크·마운트) ... M18
                                                              └ _alparts2 (알루미늄 부품 추가 8종) ... M19  ← 최신
```

(번호는 §4 항목 번호입니다.)

# 4. 버전 항목

각 항목의 칸: **날짜 · 파일 · 바탕 · 바뀐 것(전→후) · 왜 · 근거 · 질량 · 상태**.

## M01 · 옛 계통 (CAD 이전, 손으로 만든 모델)
- **날짜**: 2026년 7~8월에 쓰이다가 2026-09-12에 저장소로 옮겨 담김.
- **파일**: `pygmalion.xml`, `pygmalion_cant20.xml`, `pygmalion_cant30.xml`, `pygmalion_rolloff30.xml`, `pygmalion_v1.xml`.
- **바뀐 것**: CAD가 없던 시절의 로봇 설명서. `cant20/30`은 엉덩이를 안쪽으로 기울인 구성(20°, 30°), `rolloff30`은 또 다른 비교 구성. 정확한 정의는 **확인 못 함**.
- **왜**: 보행 학습을 시작하기 위한 초기 모델. 근거: [[122_spec_table_backfill]], [[65_design_value_uncertainty]].
- **질량**: 51.527 kg (MJCF 4개 동일).
- **상태**: **쓰지 않음**. `pygmalion_v1.xml`은 다리가 y가 아니라 x로 갈라지는 **다른 좌표 규약**이라 지금 도구에서 열리지 않음([[136_left_right_naming_history]]).

## M02 · v2, v2.1 — 최종 CAD에서 재구축
- **날짜**: 2026-08-20~22 · 커밋 `732df34`, `a356bbd`, `b0be612`, `bb0a19a`.
- **파일**: `pygmalion_v2.{xml,urdf}`, `pygmalion_v21_loop.xml`.
- **바뀐 것**: STEP 파일에서 부품별 질량·관성을 계산해 처음으로 CAD 기반 모델을 만듦. 무릎 모터를 허벅지에 귀속하는 등 재분류. v2.1은 발목을 닫힌 고리로 시도.
- **왜**: 최종 CAD가 나와서 손으로 만든 모델을 대체하려고.
- **근거**: [[87_robot_model_v2]].
- **질량**: v2 43.701 kg, v2.1 43.590 kg. (문서상 Fusion 실측 반영 뒤 v3에서 50.31 kg이라고 적혀 있으나, 이 저장소의 `pygmalion_v2.xml`은 43.7 kg임. 두 숫자의 관계는 **확인 못 함**.)
- **상태**: 보관.

## M03 · v3_printed — 프린트 부품을 프린트 밀도로
- **날짜**: 2026-08-22~23 · 커밋 `3a66335`, `355d87b`, `f7be394`.
- **파일**: `pygmalion_v3_printed.{xml,urdf}` (+`_bench20260910` MJCF).
- **바뀐 것**: 질량 출처를 STEP에서 **Fusion 360 실시간 문서**로 바꿈. 알루미늄으로 그렸지만 실제로는 3D 프린터로 뽑은 하체 부품 50개를 프린트 밀도(실측 질량÷CAD 알루미늄 질량 ≈ 0.329)로 낮춤.
- **왜**: CAD 질량(알루미늄)이 실제 로봇보다 훨씬 무거워서.
- **근거**: [[89_printed_parts_density_ratio]].
- **질량**: 35.348 kg.
- **상태**: 보관. 이 판으로 학습한 정책 15개가 있음([[131_motor_params_handoff]]).

## M04 · 폐루프 발목 `_loop` 추가
- **날짜**: 2026-08-23 · 커밋 `c1dadc2`, `aad3e28`.
- **파일**: `pygmalion_v3_printed_loop.*` 등 `_loop`가 붙은 모든 판.
- **바뀐 것**: 발목을 위아래·좌우 관절 2개(직렬) 대신 **크랭크 2개 + 막대 2개로 닫힌 고리**로. MJCF는 연결 구속으로 진짜 닫힘, URDF는 표현 불가(주석으로만).
- **왜**: 실제 발목이 모터 2개로 당기는 2-RSU 구조라서, 직렬 근사로는 토크·속도 한계가 틀림.
- **근거**: [[91_closed_loop_ankle_rl]].
- **질량**: v3 폐루프 35.347 kg(질량 변화 없음, 구조 변화).
- **상태**: 이후 모든 학습 모델의 기본 구조.

## M05 · v4_printed — 상체 재측정, IMU 위치 수정
- **날짜**: 2026-08-26 · 커밋 `834f226`, `a1d5b53`.
- **파일**: `pygmalion_v4_printed.{xml,urdf}`, `pygmalion_v4_printed_loop.*`.
- **바뀐 것**: 팔 질량 4.021 → 2.054 kg(한쪽). 팔이 용접되어 있고 DummyHand가 출력물이라서. IMU 센서 위치가 **310 mm 틀렸던 것**을 실측해 수정. 전체 35.347 → 31.316 kg.
- **왜**: 상체 질량이 11 % 틀렸고, v3 정책은 v4로 그대로 옮길 수 없다는 판단(재학습 필요).
- **근거**: [[105_imu_and_shoulder_cad_update]] §4.
- **상태**: 보관.

## M06 · `prototype-tempmass` — 일부 부품만 알루미늄 질량
- **날짜**: 2026-08-23~09-01 · `tools/robot_model/prototype_tempmass_manifest.json`.
- **파일**: `prototype-tempmass_loop.{xml,urdf}`.
- **바뀐 것**: v4 바탕에서 발 `FEET`(0.182 → 0.618 kg), 발목 `AnkleA2B_*` 6개(약 0.011~0.013 → 0.032~0.041 kg), 팔 B(0.039 → 0.042 kg)를 알루미늄 6061 질량으로. 표 열 D(CAD 알루미늄 질량)를 그대로 씀.
- **왜**: 이 부품들을 알루미늄으로 가공하기로 해서의 **임시 가정**. 문서의 상태 표시는 "조건부, 최종 Sim2Real 근거 아님".
- **근거**: `tools/robot_model/prototype_tempmass_manifest.json`, [[110_prototype_tempmass_student_teacher_report]].
- **질량**: 32.470 kg.
- **상태**: 보관.

## M07 · v30 재구축 + `rs03measured`
- **날짜**: 2026-09-01.
- **파일**: `prototype-tempmass-rs03measured-armfix_v30{,_loop}.xml`.
- **바뀐 것**: 최신 CAD v30으로 89개 메시를 다시 뽑아 3종(전신/허리까지/다리만)을 새로 만듦. RS03 모터 질량을 저울로 잰 0.9195 kg으로(카탈로그 0.880 kg).
- **왜**: CAD가 v30으로 갱신되었고, 모터 질량을 카탈로그 대신 실측하기로 해서.
- **근거**: [[111_fusion_v30_urdf_rebuild]], [[112_v30_armfix_rp_ab_part_review]].
- **질량**: 35.049 kg.
- **상태**: 보관.

## M08 · `motormeasured` + `armfix`
- **날짜**: 2026-09-01~02.
- **파일**: `prototype-tempmass-motormeasured-armfix_v30{,_loop}.xml`, `SemiFullDoF_…_v30.xml`, `LegOnly_prototype-tempmass-motormeasured_v30.*`(armfix 없는 다리만 판).
- **바뀐 것**: RS04 모터도 저울 실측 1.514 kg으로(자리표시자 1.558 kg). 팔 하우징 수정(armfix).
- **왜**: 실측 모터 질량을 모델에 반영하려고.
- **근거**: `tools/robot_model/massprops_fusion.py` 머리말(모터 실측 출처), [[112_v30_armfix_rp_ab_part_review]].
- **질량**: 전신/허리까지 35.707 kg. 다리만 armfix 없는 판 21.888 kg, 있는 판 23.630 kg(차이 1.742 kg의 이유는 **확인 못 함**).
- **상태**: 보관.

## M09 · `proxyfix`
- **날짜**: 2026-09-02 · [[114_huphy_proxyfix_rotation_dr_audit]], [[117_model_finalization_and_oneleg_training_plan]].
- **파일**: `…_v30_proxyfix{,_loop}.*` (전신, 허리까지, 다리만).
- **바뀐 것**: 엉덩이 피치 모터의 위치가 **75.8 mm 틀렸던 것**을 바로잡음.
- **왜**: 모터 배치 검토에서 위치 오류를 발견해서.
- **질량**: 전신 35.674 kg, 다리만 23.630 kg.
- **상태**: 이후 모든 판의 바탕.

## M10 · 2026-09-03 수정 6종 + 다리만(LegOnly) 확정
- **날짜**: 2026-09-03 · 커밋 `7cb25dc`, `bc2526d`.
- **바뀐 것**: 관절 부호 9건, 가동범위, 부품 귀속 오류를 고침. 발바닥을 상자 3개로. 다리만 판은 **허리 회전 모터의 질량을 골반에 남김**(관절은 없음). 6종 재생성.
- **왜**: 관절을 하나씩 돌려 보며 6번에 걸쳐 확인했더니 부호·범위·귀속이 틀린 곳이 여럿 있어서([[117_model_finalization_and_oneleg_training_plan]] §0).
- **상태**: 현재 파생 모델들이 이 규약을 이어받음.

## M11 · 2026-09-12 좌우 이름 맞바꿈
- **커밋** `a98d0d9`, [[136_left_right_naming_history]].
- **바뀐 것**: `L_`이 붙은 사슬이 실제로는 **오른쪽 다리**였던 것을, `L_` = 진짜 왼쪽(+y)으로 바꿈. 기하·질량·축은 한 글자도 안 바꾸고 이름과 참조만.
- **왜**: 실물 로봇 연결 때 왼쪽·오른쪽이 뒤바뀌어 하루를 썼음.
- **주의**: 메시 파일 이름은 그대로라서 `L_` 자산이 `R_…stl` 파일을 가리키는 것처럼 보임. **자산 이름이 기준**.
- **상태**: 현재 규약.

## M12 · 2026-09-15 실측 모터 값을 URDF에도
- **커밋** `987e2ab`, `2a18235`, [[144_urdf_motor_params_portability]].
- **바뀐 것**: URDF 옆에 `*.motor_params.json`을 둬서 회전자 관성·마찰 같은 실측값을 다른 시뮬레이터도 읽게 함(URDF에는 그 칸이 없음).
- **왜**: 다른 시뮬레이터에서도 같은 로봇이 되게.
- **상태**: 현재 규약.

## M13 · `_shell` 겉껍데기판
- **날짜**: 2026-10-06~07 · [[157_shell_lidar_upper_body_overlay]].
- **파일**: `FullDoF_…_shell{,_loop}.*`, `SemiFullDoF_…_shell{,_loop}.*` (다리만 판에는 없음).
- **바뀐 것**: 머리·몸통·팔 겉껍데기, 라이다, D435 카메라, 팔꿈치를 **덧붙임**. 원본 메시는 지우지 않음. URDF에는 모터 원통도 추가.
- **질량**: 41.162 kg.
- **상태**: 보관. 겉껍데기판 URDF·MJCF는 git에 아직 올라가 있지 않음(확인: 저장소에 첫 커밋 기록 없음).

## M14 · `_d455` 골반 카메라
- **날짜**: 2026-10-27~28 · [[180_d455_camera_on_legonly]].
- **파일**: `LegOnly_…_proxyfix_d455_loop.*`.
- **바뀐 것**: 골반에 Intel RealSense D455 카메라(0.116 kg, 60° 아래로) + PLA 마운트 브래킷(0.0620 kg)을 고정. 전체 +0.178 kg. 카메라 깊이 영상이 수식 정답과 0.023 mm 이내로 일치.
- **왜**: 계단·경사 같은 지형을 보는 정책에 쓰려고.
- **질량**: 23.808 kg.
- **상태**: 보관(M16이 대체).

## M15 · `_footv5` 새 발
- **날짜**: 2026-10-31 · `tools/robot_model/foot_overlay/build_footv5.py`.
- **파일**: `LegOnly_…_proxyfix_footv5_loop.{urdf,motor_params.json}`(MJCF는 합본에 있음).
- **바뀐 것**: 발 링크 2개만. v5 밑창(캐리어·테이프·고무 4 mm) 시각 메시, 볼록 조각 29개 충돌, 질량 0.845 → 1.186 kg/발(캐리어와 발등 덮개 80 g 추가).
- **왜**: 발바닥 형상 연구([[184_foot_v5_model_review]], 할 일 #184·#189).
- **상태**: 보관(M16이 대체).

## M16 · `_footv5_d455` 합본
- **날짜**: 2026-11-04 · 커밋 `07ddaf5` · [[185_footv5_d455_combined_model]].
- **파일**: `LegOnly_…_footv5_d455_loop.{urdf,xml,motor_params.json}`, `…_report.json`.
- **바뀐 것**: M14와 M15를 이어 붙임. 이전 두 판은 그대로 둠. 전체 24.494 kg(URDF), 검사 30개 통과.
- **왜**: 동료가 `stairs_slopes12_union_dr_99998` 정책을 새 발 + 카메라 모델에서 돌려 보려고.
- **상태**: M17이 대체. 동료의 `robot.py` 충돌 이름 정규식을 `\d+`로 고쳐야 함(발 충돌 모양이 32개).

## M17 · `_reinf` 보강재
- **날짜**: 2026-11-04 · 커밋 `3de5a29` · [[185_footv5_d455_combined_model]] §10 · `tools/robot_model/reinforce_overlay/`.
- **파일**: `…_footv5_d455_reinf_loop.{urdf,xml,motor_params.json}`, `…_reinf_loop_report.json`, `…_reinf_loop_verify.json`, 메시 `meshes_reinforce/`.
- **바탕**: M16.
- **바뀐 것**:

| 부품 | 질량 | 붙는 링크 | 개수 |
|---|---|---|---|
| Hip-Reinforcement(엉덩이 보강재) | 160.0 g | `hip_roll_link` | 다리마다 1 |
| Knee-Reinforcement(무릎 보강재) | 158.2 g | `shin_link` | 다리마다 1 |
| Pelvis_Camera(카메라+브래킷) | 166 g(카메라 116 + 브래킷 50) | `base_link` | 1 |

- **왜**: 실물 로봇에 보강재가 볼트로 붙었고(엉덩이: HipRoll2Yaw에, 무릎: Knee2Ankle에), 골반 카메라 브래킷 설계가 바뀌었음(체적 69.83 → 63.57 cm³).
- **방법**: 질량은 사용자가 준 값. 모양·무게중심·관성은 CAD에서 읽고 질량 비율로 줄임. 링크의 질량·관성에 합쳤고 새 링크는 없음. 보강재는 눈에 보이는 메시만 있고 **충돌 모양은 없음**(기존 링크 껍질 밖으로 최대 51 mm, 57 mm 튀어나옴).
- **질량**: +0.6244 kg → 25.115 kg(MJCF) / 25.119 kg(URDF). 무게중심 3.3 mm 위로.
- **검증**: 60/60 통과.
- **가정**: 카메라 166 g = 116 + 50 g은 가정(사용자가 합계만 줌).
- **상태**: M18이 대체.

## M18 · `_alparts` 알루미늄 크랭크 4개 + 엉덩이 마운트 2개
- **날짜**: 2026-11-04 · [[185_footv5_d455_combined_model]] §11 · `tools/robot_model/al_parts_overlay/`.
- **파일**: `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts_loop.{urdf,xml,motor_params.json}`, `…_alparts_loop_report.json`, `…_alparts_loop_verify.json`.
- **바탕**: M17.
- **바뀐 것** (질량은 전부 `CAD 체적 × 2.70 g/cm³`, 관성은 그 부품의 CAD 관성에 질량 비율을 곱함):

| 부품 | 전 | 후 | 링크 |
|---|---|---|---|
| 크랭크 A(다리마다 1) | 프린트 24.28 g | 알루미늄 73.89 g | `crank_A` |
| 크랭크 B(다리마다 1) | 프린트 23.72 g | 알루미늄 72.18 g | `crank_B` |
| 크랭크 커버 A, B | 프린트 | **그대로 프린트**(사용자: 커버 제외) | 위와 같음 |
| HipRoll2Yaw_Mount(다리마다 1) | 프린트 67.0 g | 알루미늄 164.59 g (+97.60 g) | `hip_roll_link` |

- **왜**: 실물 크랭크 A·B(좌우 합 4개)와 엉덩이 마운트(좌우 2개)가 프린트에서 알루미늄 가공품으로 바뀌었음(사용자 2026-11-04).
- **질량**: +0.3913 kg(크랭크 +0.1961, 마운트 +0.1952) → 25.510 kg(URDF) / 25.506 kg(MJCF).
- **검증**: 53/53 통과. 크랭크 링크는 "옛 값을 같은 CAD 숫자로 다시 만들어 보면 맞는가"를 먼저 확인했음(질량 0.01 %, 무게중심 0.000 mm, 관성 0.07~0.13 % 차이).
- **한계**: **CAD에는 다리가 한쪽만 있어서 반대쪽 다리의 부품이 실제로 알루미늄인지는 CAD로 확인할 수 없음.** 모델은 좌우가 같다고 가정하고 뒤집어 만들었음. 실물 양다리 확인이 필요함.
- **참고**: 같은 `HipRoll2Yaw` 안의 다른 부품(`HipRoll2YawA` 48.6 g, `HipRoll2YawB` 44.5 g, `Body4` 13.4 g)은 CAD 재질이 알루미늄 6061로 표시되지만 모델에서는 아직 프린트 질량임. 사용자가 이 부품들을 말하지 않아서 **바꾸지 않았음**.
- **상태**: M19가 대체.

## M19 · `_alparts2` 알루미늄 부품 추가 (엉덩이·허벅지·정강이·발목·발) ← **최신**
- **날짜**: 2026-11-04 · [[185_footv5_d455_combined_model]] §12 · `tools/robot_model/al_parts_overlay/` (`build_al_round2.py`, `verify_al_round2.py`, `make_al_vis.py`).
- **파일**: `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_loop.{urdf,xml,motor_params.json}`, `…_alparts2_loop_report.json`, `…_alparts2_loop_verify.json`, 메시 `meshes_reinforce/{R,L}_ankle_core_revised.stl`.
- **바탕**: M18.
- **바뀐 것**: **사용자가 이름을 말한 부품만** 알루미늄 6061로(질량 = Fusion CAD 체적 × 2.70 g/cm³, 관성 = 그 부품의 CAD 관성). 좌우 둘 다(CAD는 한쪽만 있어 반대쪽은 y축 대칭). 표는 다리 한쪽, "전"은 M16~M18 모델이 가진 값입니다.

| 부품 | 붙는 링크 | 전 | 후 | 증가 |
|---|---|---|---|---|
| Yaw2KneePlateA (**원본 모양·체적 119.24 cm³**) | `thigh_link` | 98.10 g | 321.95 g | +223.85 g |
| HipRoll2YawA | `hip_roll_link` | 48.58 g | 165.48 g | +116.90 g |
| HipRoll2YawB | `hip_roll_link` | 44.53 g | 135.53 g | +91.00 g |
| Knee2Ankle_RotorSide_1 (**`_1`만**) | `shin_link` | 55.25 g | 159.34 g | +104.09 g |
| Knee2Ankle_IdleSide_1 (**`_1`만**) | `shin_link` | 42.90 g | 130.57 g | +87.67 g |
| AnkleFeetPillowA / B | `foot_link` | 13.43 g 각각 | 40.89 g 각각 | +27.46 g 각각 |
| AnkleUniversalJointCore (**Revised**, 새 모양: 24.8 → 17.1 cm³) | `ankle_pitch_link` | 22.02 g | 46.21 g | +24.19 g |

- **바꾸지 않은 것**: Knee2Ankle 컴포넌트 아래 몸체 중 **`Knee2Ankle_IdleSide_1`, `Knee2Ankle_RotorSide_1` 두 개만 알루미늄**입니다(사용자 2026-11-04). 나머지 `Knee2Ankle_IdleSide_2`, `Knee2Ankle_RotorSide_2`, `Knee2Ankle_Connector_A/B`, 나사 64개는 **알루미늄이 아니라서 손대지 않았습니다**(Fusion의 재질 표시가 "Aluminum 6061"이어도 사용자가 아니라고 함: 표시는 믿을 수 없음). `AnkleA2B_*` 6개는 원래 알루미늄이라 모델의 값 그대로 둠. 이전 커밋 `44d6222`, `2f26d61`, `60a3062` 의 모델은 `_2`나 Connector 를 바꾸는 **실수**를 했고 이 항목에서 바로잡았습니다(같은 파일 이름, 이후 커밋이 정본).
- **링크별 증가(다리 한쪽)**: 엉덩이 회전 +207.9 g, 허벅지 +223.8 g, 정강이 +191.8 g, 발목 피치 +24.2 g, 발 +54.9 g. 합 **+702.6 g**.
- **왜**: 실물 부품이 알루미늄 가공품으로 바뀜(사용자 2026-11-04). AnkleFeetPillow A/B는 사용자가 알루미늄이라고 확인. AnkleUniversalJointCore는 새 모양(`-Revised`)으로 교체됨.
- **Knee2Ankle 엄격 재확인(몸체 76개를 원본 값과 비교)**: 모양은 그대로(체적 차이 ≤ 0.03 %, 무게중심 이동 ≤ 0.008 mm). 나사 64개는 질량·체적·위치가 전부 같음. Fusion의 재질 표시만 바뀌었고(Connector A/B: 프린트 → 알루미늄, AnkleA2B: 임시 알루미늄 → 알루미늄 6061) 이는 모델에 반영하지 않음.
- **Yaw2KneePlateA는 원본 모양을 씀**: Fusion에서 이 부품이 수정되어 무릎 베어링 구멍 둘레에 호 모양 포켓 2곳(+x 면에서 약 5.5 mm 깊이)이 새로 파였고 체적이 119.24 → 115.76 cm³가 됐는데, 사용자가 **"이전 버전이 맞고 지금 수정본이 잘못이다"** 라고 확인함. 그래서 **원본 메시와 체적(119.2408 cm³ × 2.70 = 321.95 g)**을 씀(Fusion v5 보관본, 관성은 원본 모델 덤프의 텐서를 알루미늄 질량으로 비례 환산). 파일 `fusion_snapshots/alparts_20261104/original_geometry.json`, `cad_mm/Yaw2KneePlateA.stl`. 수정본과의 차이 그림: `docs/img/al_parts/yaw2knee_plate_diff.png`.
- **방법**: 프로젝트의 질량 합산 도구(`massprops_fusion.py`)로 "라운드 1 상태 A"와 "라운드 2 상태 B"를 각각 계산해 링크마다 차이만 모델에 더함(상태 A가 이전 모델을 0.04 g 안에서 재현하는지 먼저 확인). 그와 별개로, 부품 각각의 CAD 값으로 질량·무게중심·관성 변화를 다시 계산해 일치를 확인(독립 검증).
- **질량**: 라운드 2 +1.4052 kg → **26.915 kg(URDF) / 26.911 kg(MJCF)**. M17(보강재) 이후 누적 +1.797 kg.
- **학습에 쓴 판(M10)과 비교**(MJCF, 영점 자세·`base_link` 기준): 질량 23.630 → **26.911 kg (+3.281 kg, +13.9 %)**. 로봇 전체 무게중심은 (−28.1, −0.3, −286.3) → (−26.3, −0.2, −299.2) mm로 **12.9 mm 낮아지고 1.9 mm 앞으로** 갔음(좌우 변화 0.03 mm). 학습 초기 자세(무릎 굽힘)에서 발목 기준으로는 **높이 574.0 → 561.4 mm(−12.6 mm), 앞뒤 +47.3 → +49.3 mm(+2.0 mm)**. 한쪽 다리 9.136 → 10.693 kg. 링크별: 엉덩이 회전 +465.5 g, 허벅지 +223.9 g, 정강이 +350.0 g, 발 +395.9 g, 크랭크 A/B +49.6/+48.5 g, 발목 피치 +24.2 g, 골반 변화 없음. 단계별 질량: M16 24.490 → M17 25.115 → M18 25.506 → M19 26.911 kg. 학습 때 질량 무작위화 설계 범위는 링크마다 약 ±2.5 %(`mass_dr_prototype-tempmass_loop.json`의 권장값)라, 새 링크 질량(엉덩이 회전 +34 %, 발 +47 %, 정강이 +13 %)은 그 범위 **밖**임(학습에 이 무작위화를 실제로 켰는지는 확인 못 함).
- **검증**: 107/107 통과(라운드 2), 라운드 1·보강재 검증도 그대로 통과.
- **시각화**: [[185_footv5_d455_combined_model]] §12의 그림. 알루미늄은 빨강, 아닌 것은 반투명(3090 서버에서 렌더).
- **한계**: ① 반대쪽 다리 부품이 실제로 알루미늄인지는 CAD로 확인할 수 없음(CAD에 다리가 한쪽뿐). ② 새 부품 모양은 눈에 보이는 메시만 있고 충돌 모양은 없음. ③ 발 링크(v5 발)에 AnkleFeetPillow 증가분을 더한 것은 v5 발이 "옛 발 + 새 부품"으로 만들어졌기 때문(`build_footv5.py` 머리말).
- **상태**: **현재 최신**.

# 5. 파일 전체 목록과 상태

(점검 스크립트가 이 표에 모든 파일 이름이 있는지 확인합니다.)

| 파일(확장자 `.urdf`/`.xml` 제외) | MJCF 질량(kg) | 항목 | 상태 |
|---|---|---|---|
| `pygmalion`, `pygmalion_cant20`, `pygmalion_cant30`, `pygmalion_rolloff30` | 51.527 | M01 | 쓰지 않음 |
| `pygmalion_v1` | 열리지 않음 | M01 | 쓰지 않음(좌표 규약 다름) |
| `pygmalion_v2` | 43.701 | M02 | 보관 |
| `pygmalion_v21_loop` | 43.590 | M02 | 보관 |
| `pygmalion_v3_printed`, `pygmalion_v3_printed_loop`, `pygmalion_v3_printed_loop_bench20260910` | 35.347~35.348 | M03·M04 | 보관(정책 15개가 이 판) |
| `pygmalion_v4_printed`, `pygmalion_v4_printed_loop`, `pygmalion_v4_printed_loop_bench20260910` | 31.316 | M05 | 보관 |
| `prototype-tempmass_loop` | 32.470 | M06 | 보관 |
| `prototype-tempmass-rs03measured-armfix_v30`, `prototype-tempmass-rs03measured-armfix_v30_loop` | 35.049 | M07 | 보관 |
| `prototype-tempmass-motormeasured-armfix_v30`, `prototype-tempmass-motormeasured-armfix_v30_loop` | 35.707 | M08 | 보관 |
| `prototype-tempmass-motormeasured-armfix_v30_proxyfix`, `prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop` | 35.707 | M09 | 보관 |
| `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix`, `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop` | 35.674 | M09·M10 | 보관 |
| `SemiFullDoF_prototype-tempmass-motormeasured-armfix_v30`, `SemiFullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix`, `SemiFullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop`, `SemiFullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop_bench20260910` | 35.674~35.707 | M08~M10 | 보관 |
| `LegOnly_prototype-tempmass-motormeasured_v30`, `LegOnly_prototype-tempmass-motormeasured_v30_proxyfix` | 21.888 | M08·M09 | 보관 |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix` | 23.630 | M10 | 보관(RP 직렬 발목) |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop` | 23.630 | M10 | **학습에 쓰인 판**(99998 이전 정책들) |
| `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_shell`, `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_shell_loop`, `SemiFullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_shell`, `SemiFullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_shell_loop` | 41.162 | M13 | 보관(git 미추적) |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_d455_loop` | 23.808 | M14 | 보관 |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_loop` | (URDF만) | M15 | 보관 |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_loop` | 24.490 | M16 | 보관 |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_loop` | 25.115 | M17 | 보관 |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts_loop` | 25.506 | M18 | 보관 |
| `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_loop` | 26.911 | M19 | **최신** |

# 6. 새 판을 만들 때 쓰는 칸

항목 1개에 아래를 모두 채웁니다. 한 칸이라도 모르면 "확인 못 함"이라고 씁니다.

1. 날짜, 번호(M19…), 파일 이름 전부.
2. **바탕**이 된 판.
3. **무엇이 바뀌었나**: 부품 이름, 전 값 → 후 값(질량은 g 또는 kg, 단위 포함), 붙는 링크.
4. **왜 바꿨나**: 한 문장. 누가 요청했나(사용자 날짜) 또는 어떤 측정이 근거인가.
5. 근거 문서와 만드는 코드 위치, 커밋.
6. 전체 질량 변화와 검증 결과(몇 개 중 몇 개 통과).
7. 알려진 한계(가정, 못 확인한 것).
8. 상태(최신 / 보관 / 쓰지 않음).

# 7. 점검

`python3 tools/robot_model/check_model_ledger.py` 는 이 폴더와 mjlab `xmls/` 의 모든 `.urdf`/`.xml` 이름이 이 문서 §5에 있는지 검사합니다. 빠진 파일이 있으면 목록을 출력하고 종료 코드 1을 돌려줍니다.

# 8. 이 대장에서 확인하지 못한 것 (소급 작성의 한계)

- 옛 계통(M01)의 `cant`, `rolloff`가 정확히 무엇인지.
- v2(43.7 kg)와 문서의 "v3 50.31 kg" 사이 관계, `armfix`가 다리만 판의 값을 1.742 kg 올리는 이유.
- M06~M10의 정확한 날짜는 문서의 기록일 기준이며, 파일 하나하나의 생성 시각이 아님(git 기록은 파일을 옮겨 담은 날짜가 섞여 있음).
- 파일 `_bench20260910`은 "관절에 측정값을 써 넣은 판"으로, 학습에 쓸 필요가 없다고 문서에 적혀 있음([[137_training_handoff_motor_params]]).
