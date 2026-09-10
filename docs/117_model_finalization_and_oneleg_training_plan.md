# 117. 모델 확정 + LegOnly 학습 계획

작성일: 2026-09-03 KST
스냅샷 커밋: `7cb25dc` (`tools/robot_model/build_robot.py` + `massprops_fusion.py` +
`meshes_fusion_full.py` + `tools/viewer/mjcf_joint_viewer.py`) — ROM·부호가 처음으로 전부
맞다고 확인된 시점. 이 문서의 모든 이후 작업은 이 스냅샷을 기준으로 diff-review한다.

## 0. 배경 — 무엇이 지금 맞는가

이번 세션에서 `tools/viewer/mjcf_joint_viewer.py`로 매 관절을 직접 돌려보며 6라운드에 걸쳐
확정한 것:

- 부호 반전 9건(hip_roll/hip_yaw/knee/waist_yaw 공유축 반전, R-only hip_pitch/shoulder_pitch,
  L-only shoulder_roll, per-tag crank_A(L)/crank_B(R))
- ROM negate-swap 8건(부호 반전과 별도로 range도 반전해야 하는 경우, hip_roll은 축은 안 바뀌고
  range만 R전용)
- 부품귀속 6건(HipPitchFlange→pelvis, HipYaw2Knee_outer→hip_roll_link, Shoulder_Stopper_P→torso,
  E2Box-IMU→pelvis, Torso2ShoulderP 복원→torso, 숄더-롤 브라켓 재분할)
- 좌우 미러 스플라이스 버그 2건(HipPitchFlange, Torso2ShoulderP — 한쪽 CAD 데이터를 공유바디에
  넣을 때 `link_mesh` 좌표변환 **이후**에 미러해야 함, 그 전에 하면 깨진 지오메트리가 나옴)
- 어깨 ROM을 사용자 기계설계표로 교체(pitch −170~+60/roll −15~+130, L 기준. rom_measured.json은
  09월 숄더 리워크 이전 측정이라 stale 판정, DESIGN_CAP_SUPERSEDES_MEASURED로 assert 우회)
- motor_proxies_fusion.json이 8/22 스냅샷(v30 리빌드 이전)이라 Hip Pitch 모터 위치가 75.8mm
  틀려 있던 것 확인 → `motor_proxies_live_20260902.json`(Fusion 재연결로 확보)로 교체
- LegOnly: waist yaw **액추에이터 질량은 유지**(실물은 위에 아무것도 없어도 모터가 펠비스에
  박혀있음), flange(`Baselink_toWaistYaw`)부터 그 위는 전부 컷, waist_yaw **관절(DOF)은 없음**

산출물: `FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix{,_loop}.{xml,urdf}`,
전체질량 35.601→35.675kg(Torso2ShoulderP 미러 포함, IMU 포함).

Fusion MCP 재연결법(향후 세션용): `FUSION_MCP=http://192.168.20.161:27182/mcp
FUSION_MCP_HOST=127.0.0.1:27182` + `M.connect()` 먼저 호출.
[[reference-fusion-mcp-lan-address]] 메모리 참고.

## 1. Collision capsule 재작성

- 현재 이미 존재: `fitted_capsule()`이 hip_pitch_link/hip_roll_link/thigh/shin(+shoulder_pitch_link
  가 메시 있을 때/arm)에 캡슐을 적용 중, `<contact><exclude>`가 CHAIN 인접쌍(pelvis~hip_pitch,
  hip_pitch~hip_roll, ... shin~foot, shin~foot_noloop 등)과 루프 바디를 이미 제외 중.
- 남은 일: 발바닥을 단일 박스(`foot1_collision`)에서 **직사각형 메시 여러 개 배열**로 교체.
  실물 발바닥 CAD 풋프린트(전면/중족/후면 등 접지 구간)를 참고해 3~4개 박스로 분할, 각 박스는
  `class="foot_box"` 유지, 접촉 사이트(`left_foot`/`right_foot`)는 그대로.
- 검증: `resolve_zero_pose_overlaps`가 여전히 비인접쌍 겹침을 잡아주는지 재확인, 제로포즈 렌더로
  캡슐이 실제 메시를 벗어나지 않는지 확인.
- **완료(2026-09-03)**: 발바닥 3-box(`foot1/2/3_collision`, X축 0.0867m 간격 연속 배열, 총 풋프린트
  X −0.08~+0.18m) 전 변형에 반영 확인. `resolve_zero_pose_overlaps`를 최신 6변형 XML(FullDoF/
  SemiFullDoF/LegOnly × serial/loop) 전체에 재실행 — **8개 파일 전부 CLEAN**(비인접쌍 잔여 겹침
  0건, SemiFullDoF는 이번에 처음 재확인). Task 1 종료.

## 2. Inertia/COM 검증 + DR 확정

- Fusion 대조: 이번 세션 수정(HipPitchFlange/HipYaw2Knee_outer/IMU/Torso2ShoulderP/숄더분할)이
  전부 Fusion 원본 CAD 바디 귀속을 근거로 한 것 — 소스는
  `tools/robot_model/fusion_snapshots/v30_inspection/bodies_prototype_tempmass.json`(라이브 재확인).
- huphy_mjcf 대조: L측 링크별 질량 비교 완료(이번 세션 앞선 응답) — shin/foot/ankle_pitch_link가
  huphy 대비 −12~−43% 무거움/가벼움. huphy 쪽 ankle_pitch_link 관성이 (0,0,0)이라 placeholder로
  보여 그 항목은 신뢰도 낮음. COM 절대값은 두 모델의 관절원점 기준이 달라 직접 비교 불가 — 축
  배치만 참고.
- 나사/모터 실측: RS03 0.9195kg/RS04 1.514kg 실측 이미 반영(`RS0{3,4}_MEASURED_KG`). 나사는
  CAD 집계 2.431547kg 포함이나 SCM 실제 나사의 절반만 모델링됐을 가능성 있음
  (docs/114 §5) — `PYG_FASTENER_COMPLETENESS_MIN`으로 처리.
- **DR 범위 재확정 1차**(이번 세션, 수정된 body 귀속만 반영):
  `mass_dr.py --bodies=.../bodies_printed_prototype_tempmass.json` →
  일반 링크 mass 0.95–1.05·COM ±5mm, pelvis/torso 0.95–1.15·COM ±20mm,
  ankle_pitch_link만 넓음(0.891–1.115, 표본 3개뿐이라 상대불확실성 큼).
  그림: `docs/img/mass_dr_round4_v30proxyfix.png`. **주의**: 이 1차본은
  `PYG_FASTENER_COMPLETENESS_MIN` 기본값(1.0, 나사 완전성 가정)으로 돌아 docs/114 §5의 50% 나사
  불확실성 방법론이 **빠져 있었다** — 기존 `mass_dr_fastener50_prototype-tempmass.json`은 반대로
  나사 방법론은 맞지만 이번 세션 이전(HipPitchFlange/IMU/Torso2ShoulderP 수정 전)의 **낡은 body
  귀속**을 쓰고 있어, 두 파일 중 어느 쪽도 단독으로 완전하지 않았다.
- **DR 범위 재확정 2차(완결, 2026-09-03)**: `PYG_FASTENER_COMPLETENESS_MIN=0.5`로 재실행해 두
  보정을 합쳤다 —
  `PYG_FASTENER_COMPLETENESS_MIN=0.5 mass_dr.py --bodies=.../bodies_printed_prototype_tempmass.json`
  → `mass_dr_fastener50_v30proxyfix_corrected.json` / `docs/img/mass_dr_fastener50_v30proxyfix_corrected.png`.
  전 링크(pelvis/hip_pitch/hip_roll/thigh/shin/ankle_pitch/foot/torso/shoulder_pitch/arm) mass·COM·
  관성 95% 구간 전부 갱신. **이 파일이 이제 권위본**이며, 이전 두 파일은 각각 방법론 절반만 맞아
  superseded. LegOnly 훈련용으로 torso/shoulder_pitch_link/arm(모델에 없는 바디)만 제외한 서브셋
  `mass_dr_legonly_fastener50_prototype-tempmass.json`도 함께 생성 — §5 실제 본학습(스모크 이후)은
  이 파일을 `PYG_MASS_DR_JSON`으로 써야 한다(스모크 자체는 방법론 검증용 구버전 서브셋으로
  이미 launch됐고 재launch하지 않음, [[2026-09-03_legonly_ab_smoke_test]] §3 참고).
- 불확실성이 큰 부분: (a) ankle_pitch_link(3바디, universal-joint cross만) (b) 나사 완전성
  가정 c~U(0.5,1) (c) huphy 대비 큰 질량차 링크(shin/foot) — CAD 소스가 다른 세대일 가능성,
  향후 huphy 쪽 BOM 확인 필요.

## 3. Viser COM/질량 시각화

`tools/viewer/mjcf_joint_viewer.py` 확장: 각 바디의 `body_ipos`(COM, 바디 로컬프레임)에
좌표계(RGB axis triad, mjviser의 프레임 API 재사용) 표시 + 바디별 질량을 클릭/hover 팝업
(viser `add_icosphere`+`on_click` 콜백으로 텍스트 갱신, 혹은 상시 라벨).

## 4. 3종 변형 재생성

FullDoF/SemiFullDoF/LegOnly 전부 이번 세션 수정 반영해서 재생성. LegOnly는 §0의 waist-yaw
질량-유지 계약대로.

## 5. LegOnly 비대칭 Student-Teacher 학습

기반: `docs/110_prototype_tempmass_student_teacher_report.md`(2026-09-01, GO 조건부).

- Actor(Student) 45D: ang_vel 3 + projected_gravity 3 + motor_pos_history[q(t-1),q(t)] 24
  + prev_action 12 + command(vx,vy,wz) 3. LegOnly는 크랭크(AB) 또는 ankle_pitch/roll(RP)
  12액션 — AB(현재 본학습과 동일 기구, v2s1_AB 계승)로 진행.
- Critic(privileged) 68~82D(상체 없으니 재계산 필요 — LegOnly는 upper 관련 관측 자체가 없음).
- Reward: docs/110 §3의 22개 항목 번들 그대로(현재 v2s1_AB와 동일 레시피).
- DR: P1 커맨드 커리큘럼만(dr_factor=0), P2에서 push/friction/encoder 10k iter ramp +
  §2에서 재확정한 mass/COM/inertia DR.
- 학습 런처: 실제로는 `run_v2_scratch.py`가 v2s1 이후 확정된 P1(게이트 커리큘럼)→P2(DR 램프)
  2단계 오케스트레이터다(`scripts/run_training.sh`는 구식 단일런 방식, v2s1/AB-RP 계열은 전부
  run_v2_scratch 사용 — 2026-09-03 확인).
- **2026-09-03 진행**: `legonly_ab_smoke_test`(1024 env 스모크)로 오케스트레이션 자체를 먼저
  검증 — P1 게이트 top-stage 도달(iter 200, 낙상 0)·P2 checkpoint full-resume·DR 0→1 램프까지
  크래시 없이 확인 중([[2026-09-03_legonly_ab_smoke_test]]). 핵심 리스크였던 "PYG_INERTIAL_DR의
  질량DR JSON이 LegOnly에 없는 torso/shoulder/arm 바디명을 참조해 P2에서 깨질 것"은 leg-only
  서브셋 DR JSON으로 사전 회피 확인.
- 스모크 PASS 확정 시 다음 명령으로 본학습 착수(§2의 fastener50 합본 DR 사용, 16384 env 풀스택):
  ```bash
  bash analysis/run_v2_scratch.sh \
    --run legonly_ab_v1 --ankle AB --vy-stages --model-v4 \
    --env PYG_MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix \
    --env PYG_MASS_DR_JSON=/home/syaro/MikuchanRemote/Human-Pygmalion/tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json
  ```
  (`--model-v4`는 no-op이나 v2s1과의 명령 대칭을 위해 유지; PYG_MODEL_TAG가 파일 선택을 전담)
- 게이트: 1000 iter/60분마다 스냅샷+판정(§2c 표), `gate_watch.sh` 백그라운드 필수.
- 실기 배포는 이 학습의 범위 밖(docs/114 §6·§7의 H/W P0 항목들은 여기서 다루지 않음) —
  목표는 시뮬레이션에서 상체 없이 걷는 정책 확보.
