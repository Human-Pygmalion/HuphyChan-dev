# log — ingest 타임라인 (append-only)

> 형식 `## [YYYY-MM-DD] <kind> | <제목> → <페이지>`. kind = research / experiment / decision / fix. 매 ingest마다 append([[SCHEMA]]). 최신이 위.

## [2026-06-22] research | base reward 과구속 종합 → gaitfix_v6(토우 결합). 판정: **base reward(특히 base_height_l2 −1.0 고정target)가 골반 swing·push-off 과구속 맞음** — 삼중수렴(우리 측정 amp 55%+flat toe / 생체역학 vault=대사최적·push-off↔CoM 86–96% / 필드 RL: IsaacLab G1·H1이 lin_vel_z 죽이고 base_height *뺌*=안티패턴). 제안: base_height **−1.0→0(drop) 또는 비대칭/deadband(δ0.025)**, lin_vel_z **−0.2→−0.05**(2파일), flat_orientation **−1.0 유지**(동시완화 금지=낙상), ang_vel_xy/upright 유지. ★결합: base_height 완화가 골반 bob·토우 롤오버를 *동시에* 푸는 레버(공유 antagonist). ★로깅 정정: base euler/lin_vel은 **이미 wrench_logger.py:82-90에 기록됨** → 할 일은 analyze.py에 골반-swing 통계 추가. 검증: 수직 amp 1.4→~2.5cm(>4=bounce), 골반 ROM, M-shape, 낙상감시 → [[reward_research/2026-06-22_11-30_base_overconstrain_pelvis_swing_gaitfix_v6]]
## [2026-06-22] research | 사람 골반/CoM 보행 운동학 — base 과rigid 가설. 골반 ROM tilt 2–5°(mean4.3)·obliq 6–11°(7.4)·rot 3–14°(9.5), 수직 CoM amp **3–5cm**(우리 1.4cm=55%), M-shape(midstance vault 최고/DS 최저). ★Ortega&Farley 2005: CoM 평탄화=대사 **+6%**. ★push-off −50%=대사 +50%(Huang/Kuo). 결론: **base_height_l2(−1.0, 고정target)가 vault·push-off·토우를 동시억제** → 완화 제안 → [[reward_research/2026-06-22_10-35_base_rigid_pelvis_com_oscillation]] · [[raw/human-pelvis-com-kinematics]]
## [2026-06-21] research | RS03 공식 매뉴얼 T-N·과부하·스톨 곡선 직접 판독(`RS03User Manual260428.pdf` p.9-11) + 5 리셀러/wiki 교차. 헤드라인 20/60·무부하200·Kt2.36·9:1·880g 전부 일치. **T-N: 120rpm=peak60 → 단조감소 → 190rpm~13, corner≤120rpm(저속평탄 잘림)**, 연속정격=20N·m(회전)/13N·m(스톨). ⚠불일치: 정격속도 공식100 vs 리셀러180, 무부하전류 공식2A vs 리셀러0.6A, 전압 24–60 vs 15–60 → [[robstride-datasheet]]. 곡선→`assets/rs03_tn_curve_official.png`·`rs03_overload_thermal_official.png`
## [2026-06-21] research | RS04 2차출처 교차검증(OpenELAB·AIFITLAB·Seeed wiki·공식X) — 120/40·200/167rpm·Kt2.1·9:1 전부 일치. ⚠리셀러 T-N "100rpm까지 평탄" 요약은 공식곡선(95rpm peak서 곧장 감소)과 불일치 재확인 → [[robstride-datasheet]]
## [2026-06-21] analysis | 무릎 감속비 통합분석 — 토크-속도 분리·RS04 실측 T-N(사다리꼴)·다조건(rough 2배)·박스모델/순환성→sweep → [[35_knee_gear_ratio_analysis]]

## [2026-06-21] research | knee 액추에이터 landscape(womafnnro) — 저감속 QDD(Cheetah 5.8~7.67·Berkeley 9·MIT Humanoid 12) vs SEA(Cassie 16); J_ref=J_r·N²(Wensing 2017); RS04 직결=9:1 정통, M107만 저감속 토크밀도 우위(단품X) → [[33_knee_actuator_landscape]] · [[raw/knee-actuator-landscape]]
## [2026-06-21] research | CoP/contact 보상공식 검증(wyvmh4gpv) — ★ Siekmann/Walk-These-Ways 모두 CoP/ZMP/heel→toe 보상 없음(foot-force·velocity norm을 위상클록 게이트), heel→toe CoP-progression은 표준공식 아님 → [[34_cop_contact_rewards]] · [[raw/cop-contact-reward-formulas]]
## [2026-06-21] research | 실제 휴머노이드 HW 질량/세그먼트/PD 비교(wr1cz09bp) — 다리질량 정상(=H1), 흐물거림=무릎 underdamped → [[31_humanoid_hw_comparison]] · [[raw/humanoid-hw-specs]]
## [2026-06-21] research | RobStride 컨트롤러 Kd·감속비 댐핑 산정(w2pkt68gl) — RS03/04 Kd≤100, joint_Kd=motor_Kd×g², 무릎 belt 1:2.5 → [[32_actuator_damping]] · [[raw/robstride-datasheet]]

## [2026-06-21] decision | LLM Wiki(Karpathy) 채택 → [[SCHEMA]] · [[index]] · [[log]] · [[raw/README]] + lint_docs.sh
## [2026-06-21] research | knee biomechanics 학술조사(wsx14ecd0) — ★ live spec엔 이미 flexed nominal(knee −22.9°); 과신전+10°/L-R 비대칭은 결함 → [[30_knee_biomechanics]] · [[raw/knee-biomechanics]]
## [2026-06-21] research | passive-toe 자연gait deep(w3g1xw9oq, reward레시피·HW·toe효율·둥근뒤꿈치) → [[29_natural_gait_reward_hw]]
## [2026-06-21] experiment | forefoot_pushoff2 (Kuo push-off w0.5/scale0.02/cap80, 해킹수정) → [[experiments/2026-06-21_16-30-58_forefoot_pushoff2_monitor]]
## [2026-06-21] fix | push-off reward-hacking(scale0.1→0.02+cap) + 영상 로봇수(env_spacing 8→15+origin_env, 프레임검증) + debugging loop verify_run.sh
## [2026-06-21] research | reward(power/thermal/torque-dist)·Shin actuator-clip·RobStride 데이터시트(wyilgvpyj) → [[28_reward_actuator_fidelity]]
## [2026-06-21] decision | RMS·p95·peak 토크 메트릭(ankle_roll RMS%rated 151% 열과부하) → [[24_training_health_analysis]]
## [2026-06-21] research | Kuo/Donelan/Ruina dynamic-walking 캐논(검증) → [[Paperreview/kuo-donelan-dynamic-walking]]
## [2026-06-21] research | 리딩리스트 v2(19편 fetch검증, 대가랩) → [[26_reading_list]]
## [2026-06-21] research | 건전 학습메트릭+보수적 중단(we12yjosz) → [[27_training_review_loop]]
## [2026-06-21] experiment | forefoot_cop (CoP@0.5, H-A 음성=toe 미적재) → [[experiments/2026-06-21_12-22-03_forefoot_cop]]
## [2026-06-21] research | 직접vs간접 toe보상(whirkj8ws: 직접=anti-pattern) → [[23_toe_use_methods]]
## [2026-06-21] experiment | stage-2~4 (넓은DR·발목offload·rough) → [[experiments/INDEX]]
## [2026-06-21] report | 11시 중간보고 → [[MIDREPORT_2026-06-21_1100]]

## 2026-06-21 · RS04 공식 T-N 곡선 검증 (리서치)
- 공식 RobStride GitHub `Product_Information/Product Literature/RS04/RS04User Manual260428.pdf`(85pp) 직접 다운로드·pdfimages 추출.
- **§12 T-N curve**: 출력축 곡선 90~190rpm 표시, 전구간 단조감소(95rpm=120N·m peak → 150rpm 변곡 → 190rpm ~10N·m, no-load 200). 평탄(constant-torque) corner ≤~95rpm. **OpenELAB 리셀러 "flat to 100rpm" 요약 부정확 확정.**
- **§13 회전 과부하 듀티표**(50rpm, 345mm 방열판): 120N·m=3s … 50=324s, 40=rated. **§14 스톨 듀티표**(1.414× 발열): 120=1s … 50=74s, 28.5=rated.
- 정격(연속) 40N·m@100rpm(345mm) / 35(220mm), peak 120N·m@90Apk. 무게 1420g, Φ120×56. peak 토크밀도 84.5 N·m/kg.
- ❌ 효율맵 미공개(공식 매뉴얼·스펙PDF 전수 grep). 3rd-party bench 공개데이터 없음.
- 곡선 이미지 → `assets/rs04_tn_curve_official.png`·`rs04_overload_{rotating,stall}_official.png`. 정리 → [[robstride-datasheet]], 함의 → [[21_motor_power_weight]].

## [2026-06-22] research | 전 액추에이터 T-N 교차검증 (workflow wgk8q2vo3, RS00/03/04 각 2+ 출처) → [[36_all_actuator_tn_envelopes]]
- **RS03**(verified yes, 6출처): 20/60 N·m·Kt2.36·9:1·880g. T-N corner ≤~120rpm 후 roll-off. 연속 20(회전)/13(스톨). ⚠ **정격속도 공식 100 vs 리셀러 180** → 공식 채택. `assets/rs03_tn_curve_official.png`.
- **RS00**(verified yes, 5출처): 5/14 N·m·무부하315·Kt1.48·R1.5Ω·10:1·310g. corner ≈100rpm. ⚠ **정격속도 공식 100 vs 리셀러 260** → 공식(우리 "260"=리셀러값, 정정). 과부하 5=무한·14=5초. `assets/rs00_tn_overload_official.png`.
- 공통: 리셀러 정격속도 over-optimistic. 3모터 다 전압제한 봉투 → "peak×maxspeed 박스" 틀림. ⚠ rotor inertia·효율맵·열시상수 전 모터 미공개(bench 필요).
- 설계분석(★ankle_roll RS00 바인딩·상향후보) → [[36_all_actuator_tn_envelopes]]. 원자료 → [[robstride-datasheet]].

## [2026-06-21] research | Agibot X2 발목/다리 actuation 재조사 (보강) → [[37_ankle_linkage_fidelity]]
- 사용자 주장("roll직결+pitch 1:1 원격링크 = Agibot X2와 동일") 재확증. X2 자체는 기계 디테일 미공개 → 증거는 주로 X1 오픈소스 + X2-N 논문.
- 신규: X1 BOM 확정(R86-2×9/R86-3×6/R52×10/**L28추杆×4**), L28=자체개발 선형 추杆 ~200g. X1 teardown(지후): 발목="拉杆구동 + 万向节 재사용", 모터 shank탑재, shank 304.94mm.
- ★ 뉘앙스: X1 teardown은 발목 **양축 차동(2-rod)** ↔ 사용자/우리 sim·X2-N 논문은 **roll직결+pitch단일링크**. 사용자 설계는 X2-N 논문과 정합, X1 차동발목과는 상이(세대차 추정). X2 마케팅 "병렬 미사용"과 일관(추杆=직렬).
- 출처: zhuanlan.zhihu.com/p/1895593089554436375 · zhiyuan-robot.com/DOCS/PM/PFL28 · github.com/AgibotTech/agibot_x1_hardware · zhuanlan.zhihu.com/p/1909304508170875310 · arxiv.org/html/2604.21541v1

## [2026-06-22] research | 2-모터 병렬발목 sim2real (Unitree G1/Booster T1/LiPS/BRUCE) → [[38_parallel_ankle_sim2real]]
- 사용자 제안: 양 발목 포화(노트36/37) → **2×RS03 병렬발목**(2모터→pitch+roll 연성구동, G1/Tesla식). 질문: sim2real 방법(정책 출력공간/Jacobian/DR/gotcha).
- ★ **두 아키텍처 캠프**: (A) 정책=joint-space(가상직렬 pitch/roll), 저수준 SDK가 **J^T(joint토크→모터토크)+PD**로 배포변환 — **Booster Gym/T1·LiPS/Tien Kung·Unitree G1 PR mode**(업계 다수, ★우리 권장). (B) 정책=모터 A/B 직접, 학습 sim에 closed-chain(MuJoCo eq constraint) — **BRUCE·G1 AB mode·TopA**(전이충실도 최고·sim느림).
- **Jacobian**: 링크기하서 해석적(LiPS Eq.12 `J=1/((BC×CP)_y)·CP^T·R`). 캘리브=CAD 레버암+엔코더 홈잉. τ변환 `τ_motor=J^T τ_joint`, 로드힘 `F=τ_motor/r_m`.
- **DR gap**: 4편 전부 friction/mass/latency/Kp만, **링크기하/Jacobian-error/백래시/로드컴플라이언스 거의 미보고** → 우리가 추가하면 이득(±5-10% J·deadband 백래시·로드 stiffness → [[16_dr_expansion]]).
- **gotcha**: ★특이점 near ROM(로드 공선→J폭발; 우리 pitch 70-90° 큼=위험), 백래시(구면조인트 유격), PR↔AB 모드일치, ★**모터 동시포화**(pitch+roll worst가 한 RS03에 합산 → 하중측정 반드시 모터공간 J^T 후에). open-chain baseline은 병렬HW서 실패(TopA).
- 출처: arxiv 2506.15132(Booster) · 2503.08349(LiPS) · 2507.00273(BRUCE) · 2507.10164(TopA) · G1 dev guide · github booster_gym/unitree_rl_gym. 원자료 → [[raw/parallel-ankle-sim2real]].

## [2026-06-22] research | 발목 상향-토크 QDD 시장조사 (RS00/RS03 대체) → [[39_ankle_qdd_uptorque_survey]]
- ROLL(RS00 14/310g/\$125 직결·포화): ★승자 = **DAMIAO DM-J4340** (27 N·m peak / 9 cont / 362g / Φ57 / 40:1 / \$155) — 같은 외형·peak 1.9×·+52g·+\$30, 디커플드 직결 드롭인. 목표 25-30 N·m 정확 충족. 단 40:1 저속 → roll-rate sim검증 필요. 차점 AK70-10(24.8/610g)·AK80-8(25/570g)=토크는 닿으나 무게·Φ 초과.
- PITCH/MID(RS03 60/880g/\$225): RS03이 가격·토크밀도(66.67 N.m/kg) 챔피언, off-the-shelf로 못 이김. 동급 Φ98 후보 전부 토크 ↓+가격 1.7-3× (DM-J8009 40/\$385, AK10-9 48/\$699, RMD-X10-40 40/\$625). 60 초과는 고감속(AK80-64 120/64:1, RMD-X10 100/35:1)뿐=저속·고가·부적합. → **RS03 유지+링크감속 fix가 정답**.
- 출처: seeed RS00/RS03 · aifitlab DM-J4340/RMD-X10 · cubemars AK10-9/AK80-64/AK70-10 · dronegearup DM-J8009 · robotshop RMD-X10. 원자료 → [[raw/ankle-qdd-uptorque-survey]].

## [2026-06-22] research | 연구 방법론 — multi-agent 구조 vs 직급 페르소나 → [[42_research_methodology]]
- 질문: 우리 워크플로(orchestrator→병렬 finder→synthesis)에 학술 직급위계(post-doc/박사/석사)를 프롬프트하면 성능↑? 구조 vs 직급라벨 분리.
- ★ 증거 방향 정반대: **구조(분해·병렬·격리·검증·종합·effort스케일)=YES**(Anthropic orchestrator-worker BrowseComp +90.2%, 단 분산의 80%는 토큰사용만으로 설명, ~95%=토큰+toolcall+모델; 레버=병렬breadth+컨텍스트격리, 페르소나 수 아님). **직급 라벨 자체=NO**(Zheng EMNLP24: 162페르소나×2410문항, no-persona 못이김·role선택=random; PRISM: expert persona MMLU 71.6→68.0→66.3 정확도 깎음).
- 깊이: 2-tier 기본(Anthropic 프로덕션도 2-tier). 3-tier는 진짜 broad일 때만 — Silo-Bench k=2에 조정손실 15-49%·tree토폴로지 최악. 검증=단일 구조적 verifier(LLM-as-judge, hallucination −9.7~53.3%), 무차별 debate는 cargo-cult(self-consistency 거의 못이김).
- cargo-cult: 맨 직급라벨·불필요한 위계깊이·무차별 debate·"에이전트=많을수록 추론↑"·공유컨텍스트 작업에 fan-out.
- 바꿀점 5: ①직급라벨→기능서술 ②finder 4필드 spec+깊이지정 ③전용 adversarial verify 신설 ④쿼리타입 effort 트리아지 ⑤2-tier 기본·3-tier 예외.
- ⚠ 일부 arXiv id(2603/2604/2605.xxxxx) 미래표기 → 수치 추정, 결론방향은 verified출처와 정합. 원자료 → [[raw/research-methodology-multiagent]].

## 2026-06-29 · Adversarial verify — ankle normative angles + peak torque (human-gait-data claim)
- Verdict: PARTIAL (supported=true w/ load-bearing caveat). Torque ~1.5 N·m/kg→~78 N·m + heel→toe rollover = RIGHT (matches our [[41_ankle_pitch_pushoff_rs03_underspec]]). Phase-by-phase ANGLE timeline = WRONG in 3 places: (1) IC is ~neutral not 7-10° DF; (2) ★mid-stance SIGN REVERSED — humans DORSIFLEX ~+5..+10° in mid-stance, claim says plantarflex; (3) omits terminal-stance peak DF ~+10° @~48%, push-off peak is ~15-20° PF @~60-62% not 20-25° @50-60%.
- REF correction: Huang 2015 PMC4664043 actual title = "Mechanical and energetic consequences of reduced ankle plantar-flexion in human walking" (NOT "Mechanical reduction of internal friction…"); paper does NOT report 1.5 N·m/kg (that's Winter). Cited title fabricated.
- Action: use torque budget (78 N·m, already in §41/§49) + rollover reward direction; DISCARD prose angle numbers — digitize a real Winter/Perry normative curve before encoding task #11 tracking target. Full note: docs/reward_research/2026-06-29_verify_ankle_normative_angles_torque.md

## 2026-06-29 · ADVERSARIAL VERIFY — biomech-toe-pushoff (Hicks WINDLASS stiffness gain / CoP-progression lever)
- Verdict: supported=FALSE → use-with-care. CLAIM: windlass +10-20% arch stiffness, fascia strain 2-4%, energy 0.05-0.1J → IMPLICATION: toe-stiffness NOT the lever, ankle 40-60% dominant, cop_progression/forefoot_cop is the only lever.
- ★ PREMISE refuted by its OWN cited source: claim attributes "+10-20% stiffness" to "Ker JRSI 2009" but that JRSI paper is Welte et al. 2018 (PMC6127178) which found the OPPOSITE — arch MORE flexible when windlass engaged, energy return dorsi 15.6 vs plantar 14.5 mJ/kg (within error). Energy "0.05-0.1J" is a magnitude/unit error (Welte normalizes per body-mass → ~1.1J return for 70kg). Windlass = ~1/3 of rigidity in cadaver, active muscle dominates (Farris, PMC8848977 ~5/6 active).
- ★ Our robot has NO windlass (linear torsional spring k60, no fascia/arch/intrinsic muscle) → premise is moot; toe bends ONLY from external GRF moment θ=Fz·d_cop/k → CoP-progression is the only physically available lever (premise being weak STRENGTHENS the prescription).
- IMPLICATION is SOUND but ALREADY ADOPTED: cop_progression weight +1.2 (forefoot_cop static was +0.0251=0.06%, useless). ζ=2.89 verified (armature 0.008 = dominant inertia); M=610N·0.07-0.10m=43-61 N·m verified, but sister-note geometry ceiling (toe 6.5cm, measured Fz 340N → θ_max~21°) overrides claim's 41-58°.
- Sister note: docs/reward_research/2026-06-29_verify_biomech_toe_pushoff_mtp_angle.md (angle-target variant, also FALSE). Full: docs/reward_research/2026-06-29_verify_biomech_windlass_stiffness_cop.md

## [2026-10-31] survey | 험지 발바닥 아치 역할과 깊이 기준
- 노트: `Feet_design/Feet_Design_Workspace/reports/험지 발바닥 아치 역할과 깊이 기준.md` (Obsidian `ARILab 관련/Project Pygmalion/Research-AI_Assist/` 같은 제목). 원문 발췌 `Feet_design/Feet_Design_Workspace/research_notes/험지 발바닥 아치 역할과 깊이 기준/excerpts.md`, 계산 `arch_terrain_calc.py`.
- 결론: 중족부 돌이 지렛목이 되는 문제는 실재하지만 수 mm 아치로는 못 막는다. v5(중앙 3.1 mm)는 5 mm 이상 돌을 어디서도 넘기지 못하고, 최대 6 mm도 아치 구간 5 mm 돌 3 %(고무 눌림 포함 17 %). 20 mm 돌 시소 각도 평평 23.4° / v5 19.9° / 6 mm 18.7°. 사람 발 험지 이득은 변형(연조직 7–8 mm, 아치 하강 5–13 mm)이고 틈새 효과 측정은 없음. QDD 로봇은 평면 또는 G1 1.6 mm 요홈. → v5 깊이 유지, 험지 대응은 발보다 작은 돌 학습 지형 + 밑창 실측.

## [2026-11-01] survey | 트레일 런닝화 밑창 + 시뮬레이션 지면 모델
- 노트: `Feet_design/Feet_Design_Workspace/reports/트레일 런닝화 밑창과 시뮬레이션 지면 모델 서베이.md` (Obsidian Research-AI_Assist 동일 제목). 결론: 트레일화 중족부 컷아웃은 무게·유연성 때문(돌 끼임 근거 없음, 리뷰어는 오히려 진흙 쌓임). 모래 RL 학습은 있음(GM-Loco: 강체 0 % → RFT 100 %), 돌·자갈 강체 다수 학습은 없음. 가장 싼 보완 = 발 solref 0.02–0.4 무작위화(Singh 2025).

## [2026-11-04] analysis | 학습 모델(M10) 정책 49999 를 최신 모델(M19)에서 시험 + DR 안/밖 판정법
- 노트: [[190_m19_policy_transfer_and_dr_judgement]]. 도구 `tools/robot_model/m19_transfer/`, 숫자 `docs/sim2real/data/2026-11-04_*.json`, 그림 `docs/img/m19_transfer/`.
- 결론: 무작위화 끈 시뮬 넘어짐 0/32(M10~M19). 켠 시뮬 첫 넘어짐 9/64 대 14/64(p 0.36). M19 는 총질량은 DR 안, 다리 질량·관성은 밖(마할라노비스 9.0 대 99 백분위 3.6). 모터 토크로 잰 실기 지지 질량 23~25 kg(±1.5)로 로그 시점 실물은 M19 보다 가벼움.
- 출처: DROPO(arXiv 2201.08434), Chebotar 2019(1810.05687), ASAP(2502.01143), Muratore 2022 리뷰, Hwangbo 2019(1901.08652) — 검색으로 존재·요지 확인.
