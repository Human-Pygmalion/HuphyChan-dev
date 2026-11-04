# index — LLM Wiki 카탈로그

> 전 wiki 페이지의 카탈로그(임베딩 없이 검색). 매 ingest마다 갱신([[SCHEMA]]). 운영 메모리는 `memory/MEMORY.md`.

## 운영 (먼저 읽기)
- [[SCHEMA]] — 위키 규약·층·ingest 워크플로우 · [[log]] — ingest 타임라인 · [[raw/README]] — 검증 원문층 · [[RESUME]] — 현재 상태/재개점

## 주제 노트 (concepts)
- [[00_overview]] 개요 · [[01_install]] 설치 · [[02_asset_conversion]] MJCF→USD · [[03_environment]] env · [[05_sensing_logging]] 센싱·로깅 · [[06_teleop]] 키보드(WASD·가속·댐핑) · [[07_measurement]] 측정 · [[08_robot_hotswap]] 로봇 교체 · [[09_gpu_driver_fix]] 드라이버 · [[10_gpu_perf_tuning]] GPU 성능
- **보상/gait**: [[04_reward_experiments]] 보상 레퍼런스 · [[11_research_gait_rewards]] · [[22_energy_toe_reward]] 에너지·toe · [[27_training_review_loop]] 검토·보수적중단 · [[28_reward_actuator_fidelity]] 열/torque-dist·Shin·RobStride · [[29_natural_gait_reward_hw]] ★자연gait 레시피+HW(4질문)
- **toe/발**: [[12_toe_stiffness]] · [[15_toe_joint_research]] · [[17_toe_usage_vibration]] · [[19_toe_ablation]] · [[23_toe_use_methods]] toe-use(산업/최신/직접보상)
- **HW/사이징·생체역학**: [[20_fusion360_isaaclab]] · [[21_motor_power_weight]] 모터·파워·무게 · [[30_knee_biomechanics]] 무릎(과신전·굽힘자세) · [[31_humanoid_hw_comparison]] 실제로봇 HW비교(질량·댐핑→흐물거림 판정) · [[32_actuator_damping]] 액추에이터 댐핑 · [[33_knee_actuator_landscape]] 무릎 액추 landscape · [[34_cop_contact_rewards]] CoP/contact 보상공식 · [[35_knee_gear_ratio_analysis]] ★무릎 감속비 분석(sweep) · [[36_all_actuator_tn_envelopes]] ★전 액추에이터 T-N 봉투(교차검증·ankle_roll 바인딩) · [[37_ankle_linkage_fidelity]] 발목 링크 충실도(roll직결/pitch 1:1링크 반영여부) · [[38_parallel_ankle_sim2real]] ★2모터 병렬발목 sim2real(G1/Booster T1: 정책 joint-space + J^T 배포변환·DR·특이점) · [[40_foot_edge_ankle_roll]] ★발-edge보행+ankle_roll과부하 근본원인(stance-width 결핍→측방CoP를 ankle로; feet_separation+lateral foot-placement 처방, flat 단독증량 안티패턴) · [[41_ankle_pitch_pushoff_hw]] ★ankle_pitch RS03(60N·m) push-off under-spec 판정(인간 ~72~83N·m·130~210W; v4서 −60 클립) + fix(링크 감속 N1.3~1.5 1순위·RS04급 조건부·parallel/SEA spring·passive toe는 대체불가) + 발목 전체(pitch+roll RS00) HW 권고. 심층 수요: [[2026-06-22_ankle_pitch_pushoff_demand_vs_RS03]] · [[64_joint_bearing_design_inputs]] ★관절·베어링 설계입력 총괄(§10 FEA 하중) · [[71_ankle_2rsu_optimization_setup]] ★발목 2-RSU 치수최적화. ★★★★★2026-08-13 현재결론(§10~§12, HANDOFF_2026-08-13_ankle2rsu 참조): §8g/§8h(v3c)까지 결과 전부 폐기 — pitch 부호버그 + swing_foot 여각(90°−진짜각)버그 2개 발견·수정 후, 확정박스로는 swing_foot(발측볼조인트 요동각)≤20°가 ROM 전역에서 물리적으로 불가능함을 100시드 교차검증(std 0.02pp)으로 확정. 로드축력·모터토크/속도는 여유충분, 병목은 순전히 요동각. 사용자 결정 대기: ①박스완화 ②SWL_F 완화 ③배치재설계 ④모터 앞뒤트림(ax_y) 추가(9%개선 확인, 미채택). 참고논문(Cervettini+ IEEE Humanoids 2025) 기반 재매개변수화(v4) 시도했으나 개선 없음 결론.
- **지형/전이**: [[13_sim2real_height_scan]] · [[14_heightmap_survey]] · [[16_dr_expansion]] DR확장 · [[190_m19_policy_transfer_and_dr_judgement]] 학습 모델→최신 모델 정책 전이·DR 안/밖 판정
- **분석/계획**: [[24_training_health_analysis]] 학습건강도+모터viz(RMS/p95/peak) · [[18_research_roadmap]] 로드맵 · [[experiment_queue]] ★실험큐(toe-roll/충격/gait) · [[25_dayplan_2026-06-21]] 일일계획 · [[dayplan_2026-06-22_overnight]] ★야간계획 · [[26_reading_list]] 리딩리스트v2(검증) · [[99_troubleshooting]]
- **방법론(연구 워크플로)**: [[42_research_methodology]] ★multi-agent 구조 vs 직급 페르소나(구조=YES·근거 / 직급라벨=효과없음~음의효과; 채택템플릿 orchestrator→다각도finder(4필드spec)→adversarial verify→synthesis, 바꿀점5)

## 논문 리뷰 (verified) — [[Paperreview/INDEX]]
- [[Paperreview/kuo-donelan-dynamic-walking]] Kuo 전이-에너지 캐논 · [[Paperreview/siekmann-periodic-reward]] 주기보상 · [[Paperreview/margolis-rapid-locomotion]] · [[Paperreview/radosavovic-humanoid-transformer]] · [[Paperreview/caps-smooth-control]] · [[Paperreview/toe-stiffness-optimization]]

## 실험 (runs) — [[experiments/INDEX]]
- [[experiments/2026-06-21_01-52-57_flat_wide_dr]] stage-2 넓은DR · [[experiments/2026-06-21_03-46-50_stage3_ankle_offload]] stage-3 발목offload · [[experiments/2026-06-21_06-41-42_stage4_rough]] stage-4 rough · [[experiments/2026-06-21_12-22-03_forefoot_cop]] forefoot CoP(H-A 음성)
- 모니터로그: [[experiments/2026-06-21_12-22-03_forefoot_cop_monitor]] · [[experiments/2026-06-21_16-30-58_forefoot_pushoff2_monitor]] (Kuo push-off, 진행중)
- 보고: [[MIDREPORT_2026-06-21_1100]] 중간보고

## 운영 메모리 (별도 — memory/MEMORY.md)
샌드박스·리포트규칙·중간검토규칙·누적영상+영상프레젠테이션+메타룰(구현→rule→자동화→verify)·GitHub·RAM22GB.
