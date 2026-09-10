# legonly_ab_v4_motorparams — 다시 잰 모터 값으로 재학습 (2026-09-11 발사)

> 이름 풀어쓰기: `legonly`(하체만) · `ab`(발목이 두 크랭크로 도는 폐루프 방식) · `v4`(이 계열 네 번째 본학습) · `motorparams`(유일 변인 = 모터 값 넷).
> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.

## §0 왜 이 학습을 하는가

직전 런 [[2026-09-05_legonly_ab_v3_keyfix]](27,999회 완주, 2026-09-07 09:43)는 로봇의 두 모터 종류(RS03·RS04)에
2026-08-23에 측정한 값을 쓰고 있었다. 그런데 이후 벤치에서 **다시 잰 값**이 훨씬 무겁고 뻑뻑했다. [[137_training_handoff_motor_params]]에
정리된 값 넷:

| | 회전자 관성 | 마찰 [뉴턴미터] | 점성 [뉴턴미터·초] |
|---|---|---|---|
| **RS04**(엉덩이 앞뒤·좌우, 무릎) | 0.016333 → **0.0773**(4.7배) | 0.269 → **0.519**(1.9배) | 0.0095 → **0.0275**(2.9배) |
| **RS03**(엉덩이 회전, 발목 크랭크 둘) | 0.015265 → **0.0638**(4.2배) | 0.285 → **0.251**(0.9배) | 0.0223 → **0.0252**(1.1배) |

거기에 더해 2026-09-10에 새 값을 반영할 때 **발목 포락표를 만드는 프로그램이 옛 값을 손으로 들고 있어서** 따라오지 않는 결함이 있었다.
엉덩이 회전(RS03)은 새 값을 쓰는데 발목 크랭크(같은 RS03)는 옛 값을 써서 **한 모형 안에서 4.2배 어긋난 채** 아무 오류 없이 지나갈 뻔했다.
이 결함과 재검산은 [[2026-09-11_training_readiness]]에 기록됐고, 2026-09-11에 발목 포락표 생성기를 고쳐(`b3af5168`) 네 관절 모두 같은 출처의 값을 쓰도록 만들었다.

**말로 풀면**: 모터가 예전 모형보다 **무겁고 뻑뻑해졌다.** 같은 동작에 토크가 더 들 것이고, 학습이 느려지거나 최고 속도가 낮게 나올 수 있다.
그것은 더 정확해진 것이지 나빠진 것이 아니다 — 직전 정책은 실제 하드웨어보다 가벼운 모터로 배웠다는 뜻이기 때문이다.

**이 런의 질문 하나**: 모터를 실제 무게·마찰로 바꾸면 **정책이 다르게 걷는가**(토크·지면 반력·속도 추종·무릎 굽힘이 어떻게 달라지는가)?

## §1 설계 — v3_keyfix와 유일 변인만 다르다

| 항목 | v3_keyfix | **이 런** |
|---|---|---|
| 모터 값(회전자 관성·마찰·점성) | 2026-08-23 측정값(카탈로그에 가까움) | **2026-09-11 재측정값**(위 표) |
| 값이 들어가는 코드 | `pygmalion_constants.py`의 `MOTOR_MEAS` 상수 | 〃(같은 자리, 값만 갱신 — 커밋 `e3c67da2`, `b3af5168`) |
| 발목 방식 | AB(두 크랭크 폐루프) | AB(동일) |
| 시작 자세·무릎 명령 창 수정 | 반영됨 | 반영됨(코드 변경 없음) |
| 세기(Kp/Kd) | 무릎 220/엉덩이 150, Kd 6 | **동일**(§4의 근거로 바꾸지 않음 — 로봇은 링크 관성이 지배해 감쇠비가 거의 안 바뀜, [[137_training_handoff_motor_params]] §5 철회1) |
| 보상·커리큘럼·DR·엔트로피 일정 | — | **전부 동일**(env_cfgs.py·run_v2_scratch.py 학습 로직 무변경, 2026-09-05 이후 diff로 확인) |

보상 변경 0, 게인 변경 0. 값이 코드 상수라 환경변수로 켜고 끄는 것이 아니라 **모터 값 자체가 코드에서 갱신된 상태로 이 런을 발사**한다.
그래서 v3_keyfix와 완전히 같은 발사 명령을 쓰는 것만으로 이 런의 유일한 변인이 성립한다.

## §1a 실행 명령 (v3_keyfix와 완전히 같은 명령, 이름만 변경)

```
cd mujoco-sim/mjlab
bash analysis/run_v2_scratch.sh --run legonly_ab_v4_motorparams --ankle AB --seed 42 --num-envs 16384 \
  --p1-max-iters 16000 --ramp-iters 10000 --digest-iters 4000 --settle-hold 10 --n-stages 5 \
  --gate-min-dwell 800 --gate-max-dwell 3000 --gate-window 100 --gate-err-ratio 1.1 --gate-fell-max 0.005 --gate-min-episodes 64 \
  --entropy-start 0.01 --entropy-end 0.002 --video-interval 8000 --video-length 500 --logger wandb --vy-stages \
  --env PYG_MODEL_TAG=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix \
  --env PYG_MASS_DR_JSON=/home/syaro/MikuchanRemote/Human-Pygmalion/tools/robot_model/fusion_snapshots/v30_inspection/mass_dr_legonly_fastener50_prototype-tempmass.json
```

발사 전 점검(사용자 지시, [[137_training_handoff_motor_params]] §2):

```
PYG_ANKLE_MODE=AB mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/check_motor_params.py
```

"모든 관절이 같은 출처의 값을 씁니다. 학습을 걸어도 됩니다." 확인됨(2026-09-11, §1a-1 발사 기록 앞).

런처가 자동으로: 액션 창 사전 점검 → 1단계(커리큘럼 게이트) → 2단계(도메인 무작위화 램프) → 매시 게이트 스냅샷(§2c) → 스펙 표 삽입(§1b~) → 누적 영상.

## §1a-1 발사 기록
- **2026-09-11 02:38:47** 1단계 발사(처음부터, 16,000회 상한, 16,384 환경). 트레이너 pid 2385280, 로그 `logs/legonly_ab_v4_motorparams_p1.log`, 런 디렉터리 `logs/rsl_rl/pygmalion_velocity/2026-09-11_02-39-21_legonly_ab_v4_motorparams_p1`(재현 스냅샷 `repro/` 포함).
- 발사 전 점검(`check_motor_params.py`): "모든 관절이 같은 출처의 값을 씁니다. 학습을 걸어도 됩니다." 통과(exit 0).
- 사전 점검 통과: 액션 창 12관절 전부 ok, 좌우 완전 대칭(v3_keyfix와 동일한 표) — 이번 런의 변인이 모터 값뿐임을 재확인.
- 사용 코드: mjlab HEAD `b3af5168`(발목 포락표 생성기가 MOTOR_MEAS를 읽도록 수정한 커밋). 자원: RAM 12.7 GiB 여유, GPU 15.3 GiB 여유, GPU 사용률 0%.
- 감시: 매시 게이트 스냅샷(review_loop), 워치독(자동 재개·RAM 보호), 게이트 알림(1,000회/60분).

## §1c 완주 판정 기준 (v3_keyfix 측정과 같은 조건으로 비교)

[[137_training_handoff_motor_params]] §7 지시대로 볼 것:

| 지표 | v3_keyfix(완주본, 미측정) | 목표 | 판정 |
|---|---|---|---|
| 관절 토크(§7 모터 사용률: 제곱평균·상위 5%·최대) | (완주 후 정식 측정 대기) | 정격 안, 새 모터 값 반영 후에도 안전권 | 신규 측정 |
| 지면 반력 봉우리 | 〃 | 몸무게 배수 기준 v3_keyfix 대비 서술 | 신규 측정 |
| 속도 추종 | 〃 | 학습 자체의 회귀 없음(낙상 0 유지) | 신규 측정 |
| 무릎 흔들림 각도 | 〃(4,100회 지점 세대비교 57~59°) | 사람 대역 55~65° 유지 | 신규 측정 |

**주의**: v3_keyfix는 완주본 정식 측정이 아직 없다(§2c-2 미완 항목). 이 런의 완주 후에는 **두 런을 나란히** 같은 프로토콜로 재는 것을 우선한다
(모터 값 하나만 다른 깨끗한 대조가 되므로, 완주 후 측정 계획에 v3_keyfix 완주본 측정을 함께 넣는다).

<!-- SPEC-TABLES:BEGIN -->
(런처가 §1b~§1b-4 보상·게인·액추에이터·가동범위 표를 자동 삽입한다)
<!-- SPEC-TABLES:END -->

## §2c 학습 중 리뷰 (게이트마다 스냅샷, docs/27 체크리스트)

| 시각 | iter | reward | ep_len | noise σ | value loss | entropy | surrogate / LR | fell / low_base | err_vxy / err_wz | DR / vx_max | thermal | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
(런처·리뷰루프가 채운다)

## §R 참조
- 직전 런: [[2026-09-05_legonly_ab_v3_keyfix]] · 핸드오프 문서: [[137_training_handoff_motor_params]]
- 값이 어떻게 나왔나: [[130_system_id_eli5]] · 값 세 벌과 이견: [[131_motor_params_handoff]] · 값 타당성 검산: [[2026-09-10_motor_value_validity]]
- 이번 발사 전 점검 도구·발목 포락표 결함 발견: [[2026-09-11_training_readiness]] · `tools/robot_model/check_motor_params.py`
- Kd A/B(기각된 방향): [[53_bc_kd_controlled_ab]]
- 모터 값 코드 변경: mjlab `e3c67da2`(RS04 행 정정) · `b3af5168`(발목 포락표 생성기가 MOTOR_MEAS를 읽도록 수정, 단일 출처화)
