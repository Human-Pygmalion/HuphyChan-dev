# 학습 실험 대장 (Trial-and-Error Ledger)

> 모든 학습 run의 **가설 → 변경 → 명령 → 결과(지표) → 판정**을 한 곳에서 관리한다.
> 새 run을 돌릴 때마다 여기 한 줄 추가 + 중요한 건 `EXP-NNN_*.md` 상세노트. (reward 세부는 [[04_reward_experiments]])
> ★**계보(era)별 정량 비교 총괄 = [[66_experiment_registry]]** (변인·핵심수치·권위정책·fc/fcp 데이터 대응표) · **그래프 뷰 = [[experiment_map.canvas]]** (2026-07-11 신설. 이 대장=시간순 원장, 레지스트리=비교용 정제본).

## 규칙 (어떻게 기록)
- **ID**: `EXP-NNN` 순번. **run**: `logs/rsl_rl/<exp>/<timestamp>_<run_name>`.
- 지표: `reward`(Train/mean), `ep_len`(Mean episode length, /1000), `err_vxy`(Metrics/base_velocity/error_vel_xy↓좋음).
- 학습 중 영상: `<run>/videos/train/*.mp4` + 누적 `<run>/videos/accumulated_progress.mp4`.
- 판정: ✅수렴/유효 · ⚠️문제발견 · ❌폐기.

## 대장 (시간순)
| ID | 날짜 | task | envs/iter | 핵심 변경 (가설) | reward | ep_len | err_vxy | 판정 |
|---|---|---|---|---|---|---|---|---|
| EXP-001 | 06-20 | Flat | CPU 1024 / 330 | G1레시피+사람다움4종 시작점 | +7.8 | 814 | **0.9** | ⚠️ 균형OK·속도추종 부족(=학습량 부족) |
| EXP-002 | 06-20 | Flat | GPU 2048 / 1500 | #001을 충분히 학습 (가중치 동일) | **+41.9** | 990 | **0.25** | ⚠️ 수렴했으나 **방향버그(게걸음)**·DR꺼짐·자기충돌X 발견 (run `…17-08-09_gpu_flat_v1`) |
| EXP-003 | 06-20 | Flat | GPU 16384 / 800 | omnidirectional cmd + 체중/마찰/COM DR + feet_distance↑ | +42 | ~1000 | 0.28 | ⚠️ DR 추가, 단 방향버그·자기충돌 여전 (run `…gpu_flat_v2dr`) + stale-pyc 삽질 |
| EXP-004 | 06-20 | Flat | GPU 16384 / 800 | **방향 −90°회전 + 자기충돌 ON + 토크리밋 reward** | +39 | **1000** | **0.28** | ✅ **전진 보행 정상**(MuJoCo FK 검증). 발목 병목 확인 (run `…18-36-47_gpu_flat_v3fix`) |
| EXP-005 | 06-20 | Rough | GPU 16384 / 1000 | EXP-004 설정 + 계단/경사 커리큘럼 + **학습중 영상(조망+명령화살표)** | +? | — | — | ⚠️ rough 초기 (run `…gpu_rough_v3vid/toe150`) |
| EXP-006 | 06-21 | Flat | 16384 / 999 | **방향버그 완전수정**(flat_fwd_fixed) — 전진 보행 확립 | +? | 999 | — | ✅ 평지 전진 base (run `…00-38-22_flat_fwd_fixed`) |
| EXP-007 | 06-21 | Flat | 16384 / 1499 | **넓은 DR**(vx 2.5·yaw 1.57·마찰·외력↑) | +? | — | — | ✅ stage-2 [[2026-06-21_01-52-57_flat_wide_dr]] |
| EXP-008 | 06-21 | Flat | 16384 / 2499 | **발목 offload**(torque_soft_limit) — 포화 발목 완화 | +? | — | — | ✅ stage-3 [[2026-06-21_03-46-50_stage3_ankle_offload]] |
| EXP-009 | 06-21 | Rough | 16384 / 1999 | 평지→**rough 이전**(warm-start) | +? | — | — | ⚠️ stage-4 [[2026-06-21_06-41-42_stage4_rough]] |
| EXP-010 | 06-21 | Rough | 16384 / 1300 | rough **수렴 시도** | +5.3 | 879 | **0.918** | ❌ **미수렴**(낙상20%) [[2026-06-21_10-33-47_stage5_rough_converge]] |
| EXP-011 | 06-21 | Flat | 16384 / 2499 | **forefoot CoP 간접보상**(H-A: toe 적재 유도) | +? | — | — | ⚠️ **H-A 음성**(toe 미적재) [[2026-06-21_12-22-03_forefoot_cop]] |
| EXP-012 | 06-21 | Flat | 16384 / 300 | **ankle push-off 일 보상** scale=0.1 | **+484** | 975 | **1.73** | ❌ **reward-HACK** [[2026-06-21_15-40-30_forefoot_pushoff]] |
| EXP-013 | 06-21 | Flat | 16384 / 900 | push-off **수정**(scale0.02·cap80·w0.5) | +41 | 994 | **0.59** | ✅ **건강·H-A양성**(ankle_pushoff↑), 영상수정위해 중도종료→재개 [[2026-06-21_16-30-58_forefoot_pushoff2]] |
| EXP-013b | 06-21 | Flat | 16384 / 500 | pushoff2 이어서(평지영상 밀도수정·재개) `…18-00-41_forefoot_pushoff3` | +43 | — | 0.50 | ⚠️ 영상수정 후 재개, iter500서 **측정 위해 종료**(이 npz가 무릎 감속비 분석의 주 데이터 → [[35_knee_gear_ratio_analysis]]) |
| EXP-014 | 06-21 | Flat | 16384 / 1919 | **impact reward**(foot_landing_vel w−2.0 + foot_impact_force w−0.01), pushoff2 warm-start | +34.5 | — | **0.72** | ❌ **OOM**(iter1919 RAM kill) + **추종 악화**(0.50→0.72 plateau, 가중치 과대→−1.0/−0.005 하향) [[2026-06-21_19-03-51_softcontact]] |
| EXP-015 | 06-21 | Flat | 16384 / ~50 | softcontact **완주 시도**(model_1950 warm-start) | — | — | — | ❌ **OOM/INCOMPLETE**(iter~50 SIGKILL exit137, warm-start dip 중 사망) [[2026-06-21_21-59-35_softcontact2]] |

> [!note] GPU 성능 튜닝 / 초기 탐색 run (reward 무관, 노트 생략)
> `gpu_rough_v1/v2/v3/v3fix/v3vid/toe150`=envs 스윕·rough 초기 탐색([[10_gpu_perf_tuning]]) · `gpu_flat_v1/v2dr/teacher/curric/toe150`=초기 탐색(EXP-002~006으로 수렴) · `*_test/_configtest`=설정검증(미기록).

## 배운 것 (누적 교훈)
1. **학습량부터**: reward 튜닝 전에 충분한 iter 확보 (EXP-001→002: err_vxy 0.9→0.25).
2. **영상 리뷰 필수**: 지표(err_vxy 0.25)는 좋아도 **방향버그(게걸음)**는 영상으로만 발견됨 (EXP-002).
3. **"서있기" local optimum**: 제약(자기충돌·토크리밋) 추가 시 ep_len 먼저 차고(서있기) err_vxy 나중에 하락(보행). 중간 과민반응 금지 (EXP-004).
4. **토크리밋 reward 효과**: 고관절 util 100%→70%. 단 발목은 진짜 병목(못 피함).
5. **편집설치 stale .pyc**: 소스 고쳐도 옛 동작 → `find -name '*.pyc' -delete` + `PYTHONDONTWRITEBYTECODE=1`.
6. ★ **까치발=발목 과부하** (2026-06-28 g1is_dm4340_flat): 속도추종+저충격 reward만으론 정책이 **까치발-shuffle** 학습 → ankle_roll/pitch RMS **~200%rated 포화**. 모터를 키워도(DM-J4340 27) 정책이 늘어난 토크를 다 씀. **plantigrade heel-toe 쉐이핑(foot-flat+cop_progression+air_time)이 인간형 *그리고* 저하중 둘 다의 열쇠** — 단순 모터 상향 불충분. HW 사이징은 *gait 정상화 후* 재측정해야 유효.
7. **feet_air_time threshold 함정**: threshold(0.4s)를 못 넘기면 보상이 0으로 미발화 → 발 안 드는 shuffle=짧은 보폭. 보폭엔 threshold↓(~0.25)·weight↑ 필요 (블로그 air_time+0.5의 anti-shuffle 의도와 일치).
8. ★★ **까치발 근본원인 = base_height 회귀** (2026-06-29, gaitfix↔G1 회귀분석 + 워크플로 wbpisjawi, high-conf): gaitfix(figure-8지만 **평발**)→G1(**까치발**) 전환서 **base_height(-1.0@0.85) 제거**가 주원인(~75%). ★ 시간증거: 첫 까치발 run(g1vanilla)이 gaitfix와 **동일 발(옛 mesh)** → reward 바뀐 순간 발생 = morphology 아님. base_height 없으면 PPO가 속도추종 reach 위해 다리 신전(base 0.95)→발목 plantarflex=까치발; gaitfix는 base 0.80-0.83(굽은 다리)=평발. **FIX = base_height를 `_apply_g1_impact_stable`(전 계통) 복원**. ★ human-ref(gait_reference) **단독**으론 base 0.926(까치발 지속)+불안정(GRF 7045N) → reference도 base_height 없이는 부족 = **근본 제약 복원 > reward 덧칠**(약한 foot_flat -0.5는 증상만 침). [[2026-06-29_tiptoe_regression]]. **§2b reward 테이블 룰**(이름/가중치/무엇/왜) 적용.


## 📚 노트 구조 (권위 → 설계 → 실험계보 → 근거)

### ★ 현행 권위 앵커 (설계값은 여기서만)
- **flat**: [[2026-07-13_gen21_bent_p2]] (`gen21p2_fc/fcp`) — 게이트 전항목 통과. knee 열 114% rated가 worst
- **rough**: [[2026-07-15_gen21_rough_uneven2_p2b]] (`p2b_v2_fc`, tile 88.6%) — **ankle_roll RS00 126% peak·GRF 1.74BW**가 worst
- 설계 하중 세트 = **flat∪rough 관절별 max** · SF: 열=RMS×1.15, 순시=P99×1.25 ([[65_design_value_uncertainty]])

### 설계 문서 (하드웨어 의사결정)
- [[65_design_value_uncertainty]] — 설계값+CI+SF 독트린 · [[64_joint_bearing_design_inputs]] — 베어링/wrench
- [[67_hip_cant_and_roll_motor_review]] — **캔트/roll-offset 종합**(§10 캔트 하중논의 종결·§11 정량비교표)
- [[68_hip_geometry_variants_viz]] — 기하변형 회전축·모션·bent init 가시화 · [[69_scaled_test_rig_design]] — 스케일 테스트rig(s=0.50)
- [[70_sim2real_pd_gains]] — ★sim2real PD게인 판정(현행 유지·모터펌웨어 1kHz 필수·DR범위 권고)
- [[62_policy_reward_design_review]] — 12 확정원칙+실패카탈로그 · [[66_experiment_registry]] — era별 정량표(전 런)

### Era-9 · 하드웨어 기하 co-design (2026-07-14~, 현행)
**캔트 계보** (하중논의 종결 — 순이득 없음→패키징 단독):
- [[2026-07-14_hip_cant30_variant]] → [[2026-07-14_cant30_p1]] → [[2026-07-14_cant30_p2]] (A/B: 재분배 발견) → 발벌림 적발 → [[2026-07-15_cant30fp_p1]] → [[2026-07-15_cant30fp_p2]] (★3-way A/B 종결 + 좌우영상) → [[2026-07-20_cant20fp_p1]](α=20, P2 학습중)
**roll-offset 계보**: [[2026-07-15_rolloff30_p1]] → [[2026-07-16_rolloff30_p2]] (fc 측정중)
**rough 소생 계보** (장님×불가지형 진단): [[2026-07-13_gen21_rough_p1]](❌) → [[2026-07-14_gen21_rough_uneven_p1]](부분) → [[2026-07-14_gen21_rough_uneven2_p1]](fell→0) → [[2026-07-15_gen21_rough_uneven2_p2b]](★rough 앵커)

### Era-8 · Gen-2 캠페인 → flat 앵커 (2026-07-10~13)
- [[2026-07-10_flat25_p1]](freeze) → [[2026-07-10_flat25b_prog_p1]](진행보상) → init A/B([[2026-07-11_bentinit_ab_plan]]·[[2026-07-12_bentinit_ab_result]] bent 승) → [[2026-07-11_flat25b_prog_p2]]/[[2026-07-11_flat25b_bentinit_p2]] → [[2026-07-12_gen2_bent_p1]]/[[2026-07-12_gen2_bent_p2]](creep 기각) → [[2026-07-13_gen21_bent_p1]] → ★[[2026-07-13_gen21_bent_p2]]

### 근거·진단 노트 (WHY — 실험 방향을 바꾼 것들)
- [[2026-06-29_tiptoe_regression]] 까치발=base_height · [[2026-07-13_stall_relative_threshold]] creep→상대임계
- [[2026-07-14_rough_p1_blind_stairs_diagnosis]] 장님×계단 · [[2026-07-12_bentinit_ab_result]] init 반전
- [[53_bc_kd_controlled_ab]] Kd A/B(link-critical 기각) · [[55_init_pose_straight_vs_bent]] 구 init A/B
- [[2026-06-30to07-07_pre-flat25_backfill]] 익명런 14건 소급

### 이전 Era (5~7 · mjlab 이행/B캠페인/구 rough)
- Era-5: [[2026-07-02_00-54-07_mjlab_A0a-actionscale]] · [[2026-07-02_03-56-30_mjlab_A0b-resume]] · [[2026-07-02_10-15-49_mjlab_A0-lowbase-term]] · [[2026-07-02_13-43-35_mjlab_A1-gains-knee800]] · [[2026-07-02_23-03-04_mjlab_A1b]] · [[2026-07-03_04-03-01_mjlab_B1]] · [[2026-07-03_05-14-12_mjlab_B1w2]] · [[2026-07-03_06-23-45_mjlab_B2]] · [[2026-07-03_07-34-12_mjlab_B3]]
- Era-6/7: [[2026-07-07_P2_final_analysis]] · [[2026-07-03_12-54-18_mjlab_R1]] · [[2026-07-03_16-32-57_mjlab_R1b]] · [[2026-07-05_04-29-10_mjlab_R2]] · [[2026-07-09_rough_p2_final]] · [[2026-07-09to10_superseded_runs]]

### 인터랙티브 도구
- `tools/wrench_studio/` — ★★**Wrench Studio v4 (서버판)**: FastAPI+three.js, 전 30개 측정정책·실메시 모션재생·브라켓하중 벡터·on-demand 집계. `server.py`(:8091) 또는 `docker compose up`
- `docs/tools/joint_wrench_explorer.html` — ★관절별 wrench 탐색기(4구성×6관절×36레짐, M⊥/Fr/Fa/τ+6성분, 브라우저 로컬 오픈)

### 그래프 뷰
- [[experiment_map.canvas]] — 계보 트리(WHY 노트 부착) · [[experiment_tree.canvas]] — ★증거 연결판(실험+근거+결과 이미지/영상, 2026-07-20 신설)
- [[2026-08-23_ankleAB_c2]] · [[2026-08-23_ankleRP_c2]] — ★Era-10 발목 기구 A/B(폐루프 AB vs 직렬 RP), 프린트 질량·실측 모터·T-N, 단일런 커리큘럼 32k (학습 중)
- [[2026-08-24_ankleAB_c3]] · [[2026-08-24_ankleRP_c3]] — c2 model_3100 warm-start + soft-landing 보상(제곱 접지속도 벌점, GRF 캡 1.2 BW) — [[95_soft_landing_prescription]]
- [[2026-08-26_ankleAB_vs_RP_comparison]] — 발목 AB/RP 단일변인 A/B 완주 비교 (성능 무승부·에너지 RP −12 - [[2026-08-26_bundleD1_AB]]
- [[2026-08-26_bundleCTL_RP]]
- [[2026-08-26_bundleD1_RP]]
- [[2026-08-26_bundleE1_AB_aborted]]
- [[2026-08-26_bundleV4_AB_aborted]]
- [[2026-08-27_bundleE1_AB]]
- [[2026-08-27_bundleV4_AB]]
- [[2026-08-27_bundleP1_AB]]
- [[2026-08-27_bundleP1s2_AB]]
- [[2026-08-27_bundleP1s3_AB]]
- [[2026-08-27_bundleD1s2_AB]]
- [[2026-08-28_v2s1_AB]]
- [[2026-08-28_ab_ankle_usage_audit]]
- [[2026-09-02_v30proxyfix_AB_st45_imuclip_idrsmoke_test]]
- [[2026-09-03_legonly_ab_smoke_test]] — ★Era-11 LegOnly(상체 제거 12-DOF) 첫 오케스트레이션 스모크, v2s1 레시피 상속
- [[2026-09-03_legonly_ab_v1]] — LegOnly 본학습(16384 env), 스모크 PASS 후 launch
- [[2026-09-03_legonly_ab_sideaware_smoke]] — ★미러축 버그픽스 검증 스모크: 프리플라이트 게이트 수정전 FAIL(L_knee 사용창 0.00°)→수정후 12관절 PASS, 발목 폐루프 리셋 closure 37.27→0.001 mm, L_knee qtarget 진폭 0.00→27~33°·사용ROM 0.21→30~37°·스톱 상시토크 21.8→6.4 N·m. v2/v3/v4 회귀 Δdefault 0.0
- [[2026-09-03_legonly_gait_kinematics]] — ★legonly_ab_v1 iter5600 보행 운동학 측정(순수 측정, 학습 미개입): 무릎 스윙 굴곡 8~10°(사람 60°)·왼무릎 사용ROM 0.2%·qtarget이 클립에 100% 붙음(정책 문제, PD 아님)·왼발 toe-off 부재(발피치 −0.1°)·**v30 모델 좌우 관절 부호 불일치로 L_knee 사용가능폭 0°** 발견

- [[2026-09-03_legonly_ab_v2]] — LegOnly 본학습 재발사(v1 중단 후): 유일 변인=설정 side-aware 수정(L_knee 창 0°→108°), 프리플라이트 PASS 12/12, 09-03 13:21 P1 16384 env
- [[2026-09-13_legonly_ab_v5_heading]] — ★Era-11 v5: v4와 **유일 변인 = 방향을 붙잡는 환경 비율 0.3 → 0.8** (몸통 비틀기·방향 표류를 줄이려는 시도). 실험 전체의 설계·게이트 기록
- [[2026-09-13_03-02-08_legonly_ab_v5_heading_p1]] — v5 **1단계** 런 리포트(반복 4,100): 커리큘럼 4/4 도달, 낙상 0, 보상 99.72, 방향 오차 0.8328
- [[2026-09-13_08-07-54_legonly_ab_v5_heading_p2]] — v5 **2단계 완주본** 런 리포트(반복 18,099): 무작위화 1.0 완주, 낙상 0, 보상 115.12, **방향 오차 0.5410 (1단계 대비 −35 %)**. 정식 측정(부하·비틀기 비율)은 아직 안 돌림
- [[2026-09-14_v5_stand_zero_command]] — v5 완주본에 정지 명령(0,0,0)을 준 측정. 서 있는 상태에서 시작하면 20초 동안 몸통 변화 폭 2 mm 로 **가만히 섬**(관절 흔들림 0.8도 이하), 최대 토크 엉덩이앞뒤 15.3 N·m(정격 38 %). ★엉덩이회전이 양쪽 다 액션 한계 ±40.5도에 상시 포화. 학습 환경 조건(리셋 흐트러짐+흔들기)에서는 복구하느라 제자리걸음 — 두 조건을 나눠 적음
- [[2026-09-14_v5_walk_then_stop]] — 3 m 를 0.8 m/s 로 걷고 20초 정지, **10회 비교**. 걷기 정상(0.866 m/s · 3.24 m)이지만 ★**정지 자세가 두 무리로 갈림** — 다리 벌린 자세만 조용하고(10회 중 2회) 나머지는 계속 발을 고쳐 디딤. 엉덩이회전 런간 폭 44도. 관절각 그림을 넣은 영상 포함(좌우 겹침·q_target 점선)
- [[2026-09-14_legonly_ab_v6_standing]] — v5와 유일 변인 = 명령 0 구간 비율 0.1 → 0.3. 원인 진단: 정지 자세 추종 항의 허용오차가 전 관절 2.86도 한 줄이라 항도 기울기도 1.7e−04 로 죽어 있음
- [[ABORTED_2026-09-14_19-37-39_legonly_ab_v6_standing_p1]] — v6 1단계 **중단 기록**(반복 2,434). 목표 웅크림 자세의 발목 값이 깊은 웅크림 것이라 발바닥이 10.6도 들려 접지 귀퉁이 4/24·지지면 폭 0 mm. 변인 자체는 효과 있었음(자세 항 1.7e−04 → 0.53)
- [[2026-09-14_legonly_ab_v6b_standing_flatfoot]] — 중단된 v6 을 대체. 같은 변인에 **발목 키프레임 정정**(발바닥 기울기 10.6도 → 0.000도, 접지 귀퉁이 4/24 → 24/24, 지지면 폭 0 → 260 mm)
- [[2026-09-15_legonly_ab_v7_heading_revert]] — v6과 유일 변인 = 방향 유지 비율 0.8 → 0.3. ★세대별 정지 측정이 범인을 지목: **v4는 발을 한 번도 안 떼고(0.00초) v5는 7.28초** — v5의 변인 하나가 제자리 서기를 망침
- [[2026-09-14_motor_bus_silent]] — 실물 모터 12개가 통신에 무응답(TX 10만 · RX 0). 소프트웨어 경로·설정 한계는 정상 확인, 원인 후보와 확인 순서 정리

<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->

**§1b-2 / §1b-3 / §1b-4 (설정 명세 표)** — 이 노트는 학습 실험 대장(전체 목록)이라 단일 런의 config가 없다. 리워드 가중치·모터 게인·토크 한계·ROM/액션 창·`PYG_*` 플래그는 **각 런 노트의 §1b~§1b-4**에 있다(모두 그 런의 `params/env.yaml`에서 기계 생성).

<!-- SPEC-TABLES:END -->
- [[2026-09-18_cnx_real_0918_1913]] — 파리 가운뎃다리 커넥톰 배우 2,048환경×1,500반복. 반복 858 에 정지 탈출, model_1000 이 0.48 m/s 로 걷는다(MLP 0.34). 표본은 MLP 의 8.4배. 배선 기여는 미판정
