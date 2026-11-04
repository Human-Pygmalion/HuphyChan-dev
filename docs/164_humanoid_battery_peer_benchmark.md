# 164. 휴머노이드 배터리·소비전력 동종 비교 — [[163_battery_bms_design]] 설계값이 타당한가

작성 2026-10-13 · 검증 대상: [[163_battery_bms_design]] (13S, 전신 47~51 kg, 평지 0.6~0.8 m/s 평균 523~663 W, Minimum 13S9P 1.77 kWh). 참고한 기존 노트: [[31_humanoid_hw_comparison]] (질량·PD 비교, 배터리 정보 없음), [[21_motor_power_weight]] (전기 전력식), [[82_final_design_mass_review]] (PSU만 있고 배터리 행 없음), `raw/robstride-datasheet.md` (R 정의).
방법: 공개 자료에서 로봇별 질량·팩 에너지·전압·런타임·소비전력을 모으고, 로봇이 주장하는 런타임에서 **평균 전력 = 팩 Wh ÷ 런타임**, **비전력 = W/kg**, **로봇 질량 대비 에너지 = Wh/kg** 을 계산해 우리 값과 같은 축에 놓았다.

## 0. 결론 먼저

1. **전압 13S/48 V급은 타당하다.** 조사한 거의 모든 휴머노이드가 48 V 계열이다 (G1 13S 54.6 V 만충, Booster T1 13S3P, Optimus 52 V, 대부분 48 V). 60 V 이하는 저전압(ELV) 한계이기도 하다. Unitree 큰 모델(H1 67.2 V, H2 75.6 V 만충)만 높은 전압을 쓰는데 이들은 자체 모터다. 우리는 Robstride 입력 한계 60 V 때문에 13S가 맞다.
2. **에너지 용량은 Minimum~Phase 1 범위가 타당하고, Recommended·Good 는 동종 어디에도 없는 값이다.** 로봇 질량당 탑재 에너지: 소형 Unitree·Booster 12~18 Wh/kg, NEO 28, Figure 02 32, Apollo 약 34(추정), Optimus 40, Atlas 41. 우리 Phase 1(13S6P) 25, Minimum(13S9P) 35 Wh/kg 는 이 범위 안이고, Recommended 56·Good 약 100 Wh/kg 는 누구도 안 가진 값이다. 즉 "마진까지 얹으면 불가능"이라는 163 결론은 시장과도 일치한다.
3. **우리 소비전력은 동종보다 높다 — 중심 모델이 이미 상한에 가깝고, 보수 모델은 비현실적이다.**
   - 걷기: 우리 중심 모델 11~14 W/kg(0.6~0.8 m/s). 같은 축의 동종 걷기 값은 T1 8.4, Optimus 8.8, Cassie 6.5, Digit 7(가벼운 사용)~14(무거운 사용), NEO 7, Figure 6.4 W/kg.
   - 서 있기: 우리 9.3 W/kg. 동종은 T1 4.2, H1 4.6, H2 4.6, Optimus(휴지) 1.8 W/kg → **우리가 약 2배**.
   - 163 의 보수 모델(16~24 W/kg)은 ASIMO급(12~18 W/kg)보다도 높아 현대 QDD 로봇으로는 신뢰하기 어렵다. **보수 모델은 폐기하고 중심 모델을 상한으로 보는 것이 맞다.**
4. **모터 저항 해석(163 이 "가장 큰 불확실성"으로 적은 것)은 기존 노트로 이미 답이 나와 있었다.** `raw/robstride-datasheet.md` 는 0.16/0.39 Ω 을 **line R** 로 기록하고 "per-phase R = line/2 (wye)" 라고 적었고, [[21_motor_power_weight]] 식(1.5·I²·R)도 같은 해석이다 → 163 의 **중심 모델(r_scale 0.5)이 올바른 해석**이다. 남는 불확실성은 리셀러 단일 출처의 ±10 % 와 P_idle·인버터 효율이다. 동종 비교가 이 해석을 뒷받침한다 (보수 해석이면 동종의 2~3배가 되어 맞지 않는다).
5. **그래서 우리 로봇은 "동종 최상위 효율"이 아니라 "동종보다 약 1.3~2배 전력을 쓰는 1세대 QDD" 로 보는 것이 정직하다.** 이 차이의 대부분은 서 있기·굽은 무릎 자세(§4) 에서 나오고, 이는 163 §6.4 의 토크 감소 지렛대가 동종 수준(8.6 W/kg)까지 가면 실제로 달성 가능한 범위임을 뜻한다.
6. **판정**: 163 의 *전압·셀·팩 질량 환산·Phase 1 팩·Minimum 팩* 은 타당. *Recommended/Good* 는 모델 문제가 아니라 목표가 동종 대비 과하다(평지 2 h + 험지 1 h + 마진). 163 의 결론은 유지하되 불확실성 서술(R 해석)을 고치고, 보수 모델을 설계 기준에서 뺀다. 아래 §5.

## 1. 수집 자료 (질량·팩·런타임)

신뢰 등급: **A** 제조사 공식 페이지 직접 확인 · **B** 제조사 자료를 재인용한 판매·보도 · **C** 집계 사이트·블로그·미확인. 표의 평균 전력은 *로봇이 주장하는 런타임에서 역산한 값* 이며 측정값이 아니다. 런타임 주장의 부하 조건(서 있기/걷기/혼합)은 대부분 공개되지 않았다.

| 로봇 | 질량 [kg] | 팩 | 전압 | 에너지 [Wh] | 런타임 주장 | 역산 평균 [W] | 비전력 [W/kg] | Wh/kg(로봇) | 등급 |
|---|---|---|---|---|---|---|---|---|---|
| Unitree G1 | 약 35 | 9 Ah, 13S | 46.8 공칭 / 54.6 만충 | 421 | 약 2 h | 210 | 6.0 | 12.0 | A (Unitree 페이지: 9000 mAh, 13S, 약 2 h) · 공칭 전압·Wh는 B |
| Booster T1 | 약 30 | 13S3P 10.5 Ah | 48 급 | 504 | 걷기 약 2 h / 서 있기 약 4 h | 252 / 126 | 8.4 / 4.2 | 16.8 | B (팀 사양서 다수 일치) |
| Unitree H1 | 약 47 | 15 Ah | 최대 67.2 | 864 | 정지 상태 4 h 미만 (보도), 리셀러 2~3 h | 약 216 / 288~432 | 4.6 / 6~9 | 18.4 | 팩: A (Unitree 페이지) · 런타임: C |
| Unitree H2 | 약 70 | 15 Ah | 최대 75.6 | 972 | 약 3 h | 324 | 4.6 | 13.9 | B |
| Unitree R1 | (미확인, 25 가정) | 6 Ah | — | 199 | 약 1 h | 199 | 약 8 | 약 8 | B |
| 1X NEO | 약 30 | — | — | 842 | 약 4 h (일반 사용) | 210 | 7.0 | 28.1 | B (제조사 값 재인용) |
| Figure 02 | 약 70 | 토르소 | 48 급(미확인) | 2,250 | 약 5 h | 450 | 6.4 | 32.1 | B (제조사 보도자료 2.25 kWh) |
| Tesla Optimus (Gen 2) | 약 57 | — | 52 | 2,300 | 휴지 약 100 W, 걷기 약 500 W (보도) | 100 / 500 | 1.8 / 8.8 | 40.4 | C (언론·분석 재인용, Gen 1/2 구분 불명확) |
| Boston Dynamics Atlas (전동) | 약 90 | 자동 교환 | — | 3,700 (단일 2차 출처) | 보통 약 4 h, 무거운 작업 약 2 h | 925 / 1,850 | 10.3 / 20.6 | 41.1 | 에너지: C · 런타임·질량: B |
| Agility Digit (구형) | 48 (연구판) | — | 48 급 | 1,000 (판매 목록) | 가벼운 사용 3 h / 무거운 사용 1.5 h | 333 / 667 | 6.9 / 13.9 | 20.8 | C |
| Apptronik Apollo | 73 | 교환식 | 48 급 | 2,000~3,000 (MPS 자료 범위) | 약 4 h | 500~750 | 6.9~10.3 | 27~41 | C (kWh 미공개) |
| Kepler K2 | (미확인) | — | — | 2,330 | 약 8 h | 291 | — | — | C |
| Berkeley Humanoid Lite | 약 16 | 6S 4 Ah LiPo | 22.2 공칭 | 약 89 (계산) | 약 30 min | 약 178 | 약 11 | 5.6 | C |
| Fourier GR-1 | 약 55 | — | 46.2 | (미공개) | 약 2 h (C) | — | — | — | A (사양서: 공급 46.2 V, **최대 전력 약 550 W**) |
| UBTech Walker S2 | (미확인) | 48 V 모듈 2개 교환식 | 48 | (미공개) | 걷기 약 2 h / 서 있기 약 4 h (팩 하나) | — | — | — | B |
| PAL TALOS | 95~100 | — | — | (미확인) | 최대 약 3 h | — | — | — | B |
| 우리 (Huphy 1.0, 163 중심 모델) | 47~51 | 13S6~9P | 46.8 / 54.6 | 1,179~1,769 | 0.6 m/s 1.9 h, 0.8 m/s 1.5 h (6P) | 서기 436 · 523 · 663 | 9.3 · 11.1 · 14.1 | 25~35 | 계산 |

옛 설계 참고(전기 전력 측정이 있는 문헌값, Frontiers Robot. AI 2023 리뷰 표): Cassie 31 kg, 1 m/s 에서 약 200 W (6.5 W/kg), CoT 0.7 · Durus 350 W, CoT 1.5 · Walk-Man 510 W, CoT 1.35 · ASIMO 600~1,800 W (출처마다 다름), CoT 1.6~2 · 인간 걷기 300~450 W, CoT 0.38~0.47.

**자료의 한계 (먼저 말해 둠)**
- 대부분이 *제조사 홍보 런타임* 이다. 부하 조건 불명이고 서 있기 비중이 큰 시연이 섞여 있어 "걷기 평균 전력" 으로 쓰면 낙관적으로 편향될 수 있다. 그래서 걷기 값은 *측정된 문헌값(Cassie 200 W)과 명시적으로 걷기라고 적은 값(T1 2 h, Optimus 500 W)* 을 중심으로 본다.
- Atlas 3.7 kWh, Optimus 질량·Gen 구분, Apollo kWh 는 단일 2차 출처 또는 범위 추정. Cassie 배터리 주장(48 V 20.5 Ah, 4 kg)은 팩 수준 약 250 Wh/kg 이라 비현실적이어서 사용하지 않았다.
- 공개된 *실제 측정 걷기 전력* 은 Cassie 한 건뿐이다. G1/H1 걷기 전력 측정 논문은 찾지 못했다 (G1 팔 전력 모델 논문 arXiv 2606.15915 만 존재).

## 2. 비교 ① — 질량 대비 에너지 (팩 크기의 타당성)

| 구분 | Wh/kg(로봇) | 대표 |
|---|---|---|
| 소형·연구용, 2 h 안팎 | 12~18 | G1 12, H2 14, T1 17, H1 18 |
| 중간 | 20~28 | Digit 21, NEO 28 |
| 4~5 h 목표 상용 | 32~41 | Figure 02 32, Apollo 약 34, Optimus 40, Atlas 41 |
| **우리 Phase 1 (13S6P P42A 1.18 kWh, 47 kg)** | **25** | 중간 |
| **우리 Minimum (13S9P P42A 1.77 kWh, 51 kg)** | **35** | 4~5 h 급 상용과 같은 대 |
| 우리 Recommended (3.5 kWh, 63 kg) | 56 | **동종 최대(41)의 1.4배** |
| 우리 Good (50E 10 kWh, 95 kg) | 약 105 | 비현실 |

→ 상용 로봇들이 2~5 h 를 내려고 쓰는 에너지가 질량의 12~41 Wh/kg 이고, 우리 Minimum 35 Wh/kg 은 상용 최상위권과 같다. **"우리가 필요하다고 계산한 에너지"가 동종이 실제로 싣는 에너지의 상한과 같은 수준에서 멈췄다는 점은 계산이 터무니없지 않다는 증거다.** (반대로 Recommended 이상은 아무도 싣지 않는다.)

참고: 같은 에너지를 쓰는 로봇의 런타임이 길다는 사실은 전력 차이에서 나온다.
- Optimus 40 Wh/kg 로 걷기 약 4.6 h (500 W) ← 비전력 8.8 W/kg
- Atlas 41 Wh/kg 로 약 4 h ← 10 W/kg
- 우리 35 Wh/kg 로 평지 0.8 m/s **약 2.4 h** (13S9P) ← 14 W/kg

## 3. 비교 ② — 소비전력 (모델의 타당성)

우리 중심 모델 값(163 §3.2, 47 kg, 전자 38 W·상체 대기 21 W 포함): 서 있기 436 W, 0.6 m/s 523 W, 0.8 m/s 663 W, 1.0 m/s 760 W, 험지 약 584 W.

| 상황 | 우리 [W/kg] | 동종 [W/kg] | 우리 / 동종 |
|---|---|---|---|
| 서 있기 | 9.3 | T1 4.2, H1 4.6, H2 4.6, Optimus 1.8 | **약 2~2.2배** (Optimus 대비 5배) |
| 걷기 0.6 m/s | 11.1 | Cassie 6.5(1 m/s), T1 8.4, Optimus 8.8, NEO 7.0, Figure 6.4 | 1.3~1.7배 |
| 걷기 0.8 m/s | 14.1 | 위와 같음 + Digit 무거운 사용 13.9, Atlas 10.3 | 1.6~2.2배 (Digit 무거운 사용과 동급) |
| 보수 모델 0.8 m/s | 약 24 | ASIMO 12~18, HRP-4 피크 약 19 | 동종 상한 초과 |

**운반비(CoT = P/(m·g·v), 전기 전력 기준)**
| 로봇 | CoT | 비고 |
|---|---|---|
| 인간 걷기 | 0.38~0.47 | 문헌 |
| Cassie | 0.7 | 1 m/s, 약 200 W |
| Walk-Man | 1.35 | |
| Durus | 1.5 | |
| ASIMO | 1.6~2 | 출처에 따라 |
| **우리 (중심)** | **1.65 (1.0 m/s) / 1.80 (0.8) / 1.89 (0.6)** | 전신, 전자 38 W 포함 (다리만이면 0.8 m/s 1.64) |
| 우리 험지 (0.5 m/s) | 약 2.5 | |

→ 우리 CoT 는 **ASIMO·Durus·Walk-Man 급**이고 Cassie 의 약 2.5배다. 나쁘지만 이상치는 아니다. Cassie 는 다리 질량이 가볍고 스프링으로 걷는 설계라 대표 *최상* 값이고, 풀 휴머노이드 QDD 에는 비교 가능한 측정이 없다. **우리 값이 "물리적으로 말이 안 되는" 수준은 아니고 "아직 최적화 안 된 1세대 휴머노이드" 수준**이라고 읽는 것이 타당하다.

**서 있기가 동종의 2배인 이유(추정, 확인 필요)**
1. 기본 자세가 무릎을 굽힌 `knees_bent` 라 중력 토크를 계속 받친다 (163 §3: 구리 손실이 지배). 동종 Unitree 는 대형 모터의 큰 감속비·토크 여유와 곧은 서 있기 자세 위주.
2. Robstride 는 QDD 라 *토크당 전류가 크다* (RS04 Kt 2.1 N·m/A 가 출력 기준). 무릎이 굽은 자세에서 무릎 토크 40~60 N·m 이면 20~28 A_rms.
3. 우리 모델은 P_idle 3 W × 19 개 + 인버터 3 % 를 둔 것이라 동종 대비 약간 보수적일 수 있다.
4. 로그가 23.6 kg 하체 모델에서 얻은 토크를 질량비로 외삽한 값이라 *과대 방향* 일 수 있다 (163 §3: 0.8 m/s 이상 최대 +50 %).

**이 비교가 알려주는 것**
- 보수 모델(구리 손실 ×2)은 동종 대비 지나치다 → 설계 기준에서 뺀다. 중심 모델은 *상한에 가깝다*.
- 동종이 달성한 8.4~8.8 W/kg(걷기)가 우리의 *현실적인 하한 목표* 다 (47~51 kg → 약 400~440 W).
- 따라서 실제 값은 **400~660 W (0.6~0.8 m/s, 47~50 kg)** 구간에 있을 가능성이 크다. 이 구간에서 팩별 런타임(사용 창 85 %):

| 팩 | 8.6 W/kg (동종 걷기 수준) | 11.1 W/kg (0.6 m/s 중심) | 14.1 W/kg (0.8 m/s 중심) |
|---|---|---|---|
| 13S6P P42A (1.18 kWh, 47 kg) | 2.5 h | 1.9 h | 1.5 h |
| 13S8P P42A (1.57 kWh, 50 kg) | 3.1 h | 2.4 h | 1.9 h |
| 13S9P P42A (1.77 kWh, 51 kg) | 3.4 h | 2.7 h | 2.1 h |

## 4. 비교 ③ — 전류·전압·BMS (정격의 타당성)

| 항목 | 동종 자료 | 우리 (163) | 판단 |
|---|---|---|---|
| 버스 전압 | 48 V 계열이 표준 (MPS 발표: Optimus·Figure·Digit·Apollo 48 V급 · Zenodo 사전공개: 60 V DC 이하가 안전 저전압 한계) | 13S: 46.8 공칭, 54.6 만충, 하한 39 V | 타당 |
| 최대 전력 설계 | MPS 발표 자료는 각 로봇의 피크 전력 예산을 약 2 kW (48 V에서 약 42 A) 로 둠 | 중심 모델 100 ms 피크 1.1~2.4 kW, 5 ms 최대 3.7 kW | **같은 크기** (험지와 5 ms 만 2 kW 초과) |
| 지속 전류 | 2 kWh 팩, 1 kW 에서 약 21 A (예시) | 평균 500~660 W → 약 12~16 A, 163 의 설계 rms 28~48 A (=보수×1.25) | 중심 평균은 동종 범위, 설계 값은 보수적 |
| 방전율 | technologies.org: 동적 플랫폼 5C~15C 목표 | 중심 100 ms 피크 약 1.5~3C (13S9P), 5 ms 4C 미만 | 여유 |
| Phase 1 BMS 정격 (지속 60 A / 1 s 120 A / 100 ms 160 A) | 공개 정격 없음 (G1 충전기 54 V 5 A 만 확인) | 약 2.5~7 kW | **동종 피크 예산(2 kW) 의 3배** — 과대, 줄여도 됨 |
| 회생·BMS | 동종의 회생 처리 방식은 공개 자료 없음 | 제동 초퍼 55.0/54.2 V | 비교 불가 (보수 설계 유지) |
| 팩 질량 비율 | 일반 논평: 로봇 질량의 15~25 % (C), 제조사 공개 없음 | Phase 1 17 %, Minimum 23 % | 범위 안 |
| 팩 에너지 밀도 | 상용 로봇 팩 광고 145~250 Wh/kg (C) | P42A 152, 50E 180 Wh/kg | 보수적(낮은 쪽) |

→ BMS 100 ms 정격은 **120 A (≈5 kW)**, 5 ms 이하는 버스 커패시터와 FET 단시간 정격으로 받는 것으로 충분하다 (중심 100 ms 최대 2.4 kW, 보수 4.4 kW). 163 의 160 A 는 *상한 마진* 으로 두거나 120 A 로 낮춰도 동종 대비 여유가 남는다. 이 값은 셀·FET·커넥터 비용에 직결되므로 Phase 1 에서는 120 A 안을 권한다.

## 5. 163 에 반영할 변경

1. **저항 해석 불확실성 재서술**: "가장 큰 불확실성" → 기존 노트가 line R, phase=R/2 로 이미 해석함. 중심 모델이 정본. 남는 오차는 리셀러 단일 출처 ±10 %, P_idle, 외삽. (163 §0-5, §2.2, §8.3-1 수정함)
2. **보수 모델은 설계 기준에서 제외** — 동종 대비 상한 초과 (§3). 163 의 보수 전류(100 ms 피크, rms×1.25)는 *마진 설계* 로만 유지.
3. **동종 대비 요약을 163 §0 에 추가** — 전압 일치, 에너지 Minimum 은 상용 상위권과 같음, 전력은 동종의 1.3~2배.
4. **권장 우선순위**: Phase 1 의 13S6P(1.18 kWh)는 G1(0.42 kWh/2 h)·T1(0.50 kWh/2 h)·H1(0.86 kWh)보다 크고 Optimus·Figure(2.2~2.3 kWh) 보다 작은 중간 크기로 적절하다. 서 있기 전력을 실측하는 순간 동종 대비 위치(4~5 W/kg 인지 9 W/kg 인지)가 확정된다.

## 6. 한계 (정직하게)
- 동종 값 대부분이 **홍보 런타임 역산** 이다. 부하·사용 창(100 % 방전 여부)·노화 정보가 없어 ±30 % 이상 흔들릴 수 있다.
- 걷기 전력의 *측정* 근거는 Cassie 단 한 건이고, 휴머노이드(풀 사이즈, 전신) 측정은 Optimus 500 W 보도뿐이다.
- 로봇 질량·팩 질량은 기종·세대에 따라 달라 일부는 근사(R1, Optimus, Digit). 위 W/kg 는 *크기 정도를 보는 값* 이다.
- Atlas 3.7 kWh, Apollo 2~3 kWh, Optimus 2.3 kWh/52 V 는 단일 2차 출처. 이 문서의 결론(§0-2, §2)은 이 값들이 ±30 % 어긋나도 유지된다 (우리 Minimum 35 Wh/kg 은 상용 상위권 32~41 과 겹침).
- Frontiers 리뷰는 앞쪽 100,000자만 확인했다 (HRP, TALOS 표는 읽지 못함). arXiv 2103.04675 (이족 하드웨어 리뷰) 는 본문 추출 실패.
- 제조사 데이터시트(Molicel, Samsung 등)와 Robstride 공식 매뉴얼은 여전히 미확인 (163 §9).

## 7. 출처 (2026-10-13 확인)
- **A (제조사 페이지 직접 확인)**: Unitree G1 https://www.unitree.com/g1 (9000 mAh, 13S, 약 35 kg, 약 2 h) · Unitree H1/H1-2 https://www.unitree.com/h1 (15 Ah 0.864 kWh, 최대 67.2 V, 47/70 kg) · Fourier GR-1 사양서 https://fftai.com/uploads/upload/files/20250515/3f8dfafee28ab3d5c6688016d8e36eaf.pdf (공급 46.2 V, 최대 전력 약 550 W).
- **B (판매·보도, 제조사 값 재인용)**: G1 421.2 Wh 46.8 V https://www.camera-warehouse.com.au/unitree-g1-battery-9000-mah · H2 0.972 kWh 최대 75.6 V, 70 kg, 3 h https://megadron.pl/en/products/unitree-h2-standard-battery-33427.html , https://www.generationrobots.com/en/404449-unitree-robotics-h2-humanoid-robot.html · Unitree R1 199 Wh https://www.4gltemall.com/unitree-r1-battery.html · Booster T1 13S3P 10.5 Ah, 걷기 2 h https://quakelogic.net/datasheets/QL-Booster-T1-Humanoid-Robot-for-Developers.pdf , https://cs.utexas.edu/~AustinVilla/humanoid/assets/qualification_materials/2025/UT Austin Villa AdultSize Team Specification.pdf · 1X NEO 842 Wh 4 h https://humanoid.guide/neo-what-1x-just-launched/ · Figure 02 2.25 kWh 5 h https://www.iotworldtoday.com/robotics/humanoid-robot-startup-figure-ai-unveils-next-generation-robot · Digit 구형 1 kWh, 3 h/1.5 h, Digit v5 90 min https://www.newequipment.com/home/product/55099224/73197-agility-robotics-develops-digit-the-bipedal-humanoid-robot , https://www.automate.org/robotics/industry-insights/agilitys-digit-v5-update-brings-more-safety-features-better-battery-and-new-knees · Apollo 73 kg, 4 h 교환식 https://apptronik.com/apollo , https://spectrum.ieee.org/humanoid-robot · Atlas 4 h/2 h 교환 https://www.techrepublic.com/article/news-boston-dynamics-atlas-robot-battery-swap/ · UBTech Walker S2 https://heise.de/-10498335 , https://www.yankodesign.com/?p=567362 · TALOS 3 h https://spectrum.ieee.org/talos-humanoid-now-available-from-pal-robotics · Kepler K2 2.33 kWh 8 h https://inspenet.com/en/?p=251631 .
- **C (집계·분석·블로그)**: Optimus 2.3 kWh/52 V, 100 W/500 W https://www.tomsguide.com/news/elon-musk-demos-the-human-like-optimus-tesla-bot-and-it-walks-on-its-own , https://liveinthefuture.org/stories/optimus-battery-irrelevance-43-robot-model-s-math · MPS(Monolithic Power Systems) 2025 발표 48 V·약 2 kW 피크 비교 https://media.monolithicpower.cn/mps_cms_document/m/p/mps_powers_humanoid_robots_9-23-2025.pdf (직접 다운로드 불가, 검색 요약 인용) · 평균 2.5 kWh 미만, 2~4 h https://technologies.org/the-humanoid-robot-bottleneck-is-the-battery-why-two-kilowatt-hours-caps-the-whole-industry/ · Zenodo 사전공개 "Power Systems … Humanoid Robots" https://zenodo.org/record/19152566 (48 V 표준, 팩 15~25 %, 일반론 — 근거 약함) · Berkeley Humanoid Lite 6S 4 Ah 30 min https://aiwiki.ai/wiki/berkeley_humanoid_lite · Booster K1 https://www.robotshop.com/products/booster-robotics-k1-geek-humanoid-robot .
- **문헌(측정/CoT)**: Frontiers in Robotics and AI 2023, 이족 로봇 에너지 효율 리뷰 https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1223946/full (Cassie 31 kg·약 200 W·CoT 0.7, Durus 350 W·1.5, Walk-Man 510 W·1.35, 인간 0.38~0.47) · Cassie 논문 https://arxiv.org/pdf/1809.07279 · HRP-4 750 W https://arxiv.org/pdf/1809.07073 · G1 팔 전력 모델 https://arxiv.org/abs/2606.15915 .
- **내부**: [[31_humanoid_hw_comparison]], [[21_motor_power_weight]], [[82_final_design_mass_review]], `docs/raw/robstride-datasheet.md` (line R 정의, 라인 27~32).
- 계산: `tools/power_budget/` (163 §11). 비교용 계산은 이 문서 표 안에서 재현 가능(W/kg = Wh÷런타임÷질량).

## 8. 추가 조사 — 학술논문·특허·사양서 (2026-10-13)

### 8.1 학술논문
| 논문 | 내용 | 우리에게 주는 뜻 |
|---|---|---|
| Frontiers Robot. AI 2023 이족 에너지 리뷰 | Cassie 31 kg, 1 m/s 에서 약 200 W, CoT 0.7. Durus 350 W (CoT 1.5). Walk-Man 510 W (CoT 1.35). 사람 300~450 W | §3 비교의 측정 기준점 |
| Cassie 사양 요약 (robotsguide, 등급 C) | 48 V 20.5 Ah, 서 있기 약 100 W, 걷기 약 300 W, 약 5 h | Cassie 걷기 200~300 W (6.5~9.7 W/kg), 서 있기 3.2 W/kg. 우리 서 있기 9.3 W/kg 는 Cassie 의 약 3배 |
| Duke Humanoid, arXiv 2409.19795 | 30 kg 이족. 0.3 m/s 에서 기본 정책 CoT 1.13, 수동 동역학 정책 0.77 (실물 −31 %, 시뮬 최대 −50 %). **CoT 는 전기 전력이 아니라 모터 기계 일로 계산** | 정책만 바꿔도 에너지가 30 % 줄 수 있다 → 163 §6.4 의 토크 30 % 감소 가정이 근거 없는 숫자가 아니다. 단, 기계 CoT 라 우리 전기 CoT 와 직접 비교 불가 |
| ToddlerBot, arXiv 2502.00893 | 소형. 모터 60 W (듀티 70 %), 1 h 걷기에 75 Wh. "전신 휴머노이드 전원은 흔히 2 kW·60 V 를 넘는다" | 60 V 한계와 2 kW 피크 예산이 업계 통념임을 뒷받침 |
| 스프링-힘줄 이족, arXiv 2203.01588 | 전체 전력 17~20 W 중 약 9 W 가 드라이버·제어 보드 대기 전력. CoT 1.5~2.9 | 대기 전력이 작은 로봇에서는 절반을 차지함. 우리 P_idle 가정(모터당 3 W, 합 57 W, 전체의 약 10 %)은 실측 확인 대상 |
| G1 팔 전력 모델, arXiv 2606.15915 | P = a·τ·ω + b·τ² + c·|ω| + d·ω², R² 0.93~0.97 | 다리에 같은 식을 적용해 P_idle·마찰항을 실측으로 식별할 수 있음 |
| Wiley Advanced Science 2026 "Leading the Pack" 리뷰 | 접근 불가 (HTTP 403). **읽지 못함** | 후속 확인 필요 |
| Zenodo 사전공개 (Ivchenko 2026) | 48 V 표준, 팩 15~25 %, 걷기 회생 15~25 W (5~8 %) | 일반론. 근거 약함. 163 의 "정상 걷기 회생 없음"과 같은 방향(우리는 순 회생 0) |

### 8.2 특허
| 특허 | 내용 | 우리에게 주는 뜻 |
|---|---|---|
| Figure AI 계열 WO2025179236A1, US 12611766, US 12638859, US 12707556 (검색 발췌, 본문 스캔본은 읽지 못함) | 배터리 2.5 kWh 초과, 런타임 4 h 초과(바람직 6 h 초과). 다른 발췌: 1.5~5 kWh, 바람직 2~3 kWh. 토르소 내부 높이 250 mm 초과, 전지 체적 2 L 초과·바람직 4 L 초과·최선 6 L 초과. 배터리는 토르소 뒤쪽, 연산 장치는 앞쪽, 사이에 냉각. 모듈식 | 우리 13S9P 는 셀 117개 × 약 25 cm³ + 30 % ≈ 3.8 L (13S6P 는 2.5 L). Figure 의 "바람직 4 L" 안에 들어옴. 토르소 체적은 CAD 로 확인 필요 |
| Tesla "Vertical Energy Storage Device Enclosure … for a Robot" (2차 요약, 미확인) | 전지를 세로로 배열해 무게중심을 낮춤 | 질량 위치 고려 (163 §6.7) |
| Kawasaki US 12113469 B2 "Robot, humanoid robot, and robot fall control method" (2024-10-08) | 모터 전체가 회생 저항 하나를 트랜지스터로 공유. 정상 때 기준 전압 약 200 V 에서 회생 에너지를 흡수해 과전압 방지. **비상정지 때는 기준을 30~60 V 로 낮춰 약한 동적 제동**. 기계식 브레이크 없음. 이유: 관절이 굳은 채로 넘어져 손상되는 것을 막음 | ① 제동 저항을 모터들이 공유하는 구조는 우리 초퍼 설계와 같다. ② 비상정지를 "토크 오프"로 정한 163 의 안은 **다리가 풀려 무너진다**. 동적 제동으로 속도를 줄여 천천히 주저앉히는 방식이 낫다. Robstride 드라이버가 지원하는지는 확인 필요 |

### 8.3 사양서·판매 자료
| 자료 | 값 | 우리에게 주는 뜻 |
|---|---|---|
| Robstride 제동(bleeder) 모듈 (rcdrone, OpenELAB) | 동작 15~60 V, 24/48 V 모드, **최대 클램프 80 V (100 ms 미만)**, 30 A 연속 / 200 A 피크 | 이미 팔리는 부품이 있다. 200 A × 55 V ≈ 11 kW 피크라 우리 5 kW 설계 범위 안. 임계 전압은 우리 값(55 V)으로 설정 가능한지 확인 |
| Robstride Power-Board (AIFITLAB) | 방전 저항 3~10 Ω / 50 W 이상, **기본 ON 64 V / OFF 60 V** (설정 가능) | 기본 ON 이 모터 입력 한계 60 V 보다 높다 → **그대로 쓰면 안 되고 55 V 로 낮춰야 한다**. 3~10 Ω 은 55 V 에서 0.3~1.0 kW 급이라 낙하 5 kW 설계(0.6~1.0 Ω)와 크기가 다르다 |
| Molicel P42A 판매 목록 | 교류 10 mΩ (1 kHz), **직류 16 mΩ (10 A, 1 s)**, 펄스 약 70 A (시간 미표기), 연속 45 A (80 °C 차단) / 30 A | 163 의 DCIR 가정(14 mΩ)을 16 mΩ 로 갱신. 13S8P 팩 저항 13×16/8 + 5 = 31 mΩ, 100 A 에서 전압 강하 3.1 V |
| Unitree G1/H1 제품 페이지 | G1: 9000 mAh, 13S, 약 2 h, 약 35 kg. H1: 15 Ah (0.864 kWh), 최대 67.2 V | §1 표와 같음 |

### 8.4 못 구한 것
- Robstride 공식 매뉴얼(과전압·회생 동작), Molicel·Samsung 공식 데이터시트, MPS 발표 원문, Figure 특허 본문, Wiley 리뷰 — 접근 불가 또는 스캔본. 이 문서의 해당 값은 모두 검색 발췌·판매 자료다.
- 풀 사이즈 휴머노이드의 **전기 전력 측정 논문**은 이번에도 찾지 못했다 (Cassie 외). 공개 벤치마크 공백이므로 우리 서 있기·걷기 실측값을 기록하면 그 자체로 기여 자료가 된다.

### 8.5 163 에 반영한 변경 (이번 조사)
1. 제동 초퍼는 Robstride 모듈(80 V 클램프, 200 A 피크)로 시작할 수 있다. 임계 55/54.2 V 로 재설정 가능한지만 확인한다.
2. 셀 DCIR 가정 14 → 16 mΩ.
3. 비상정지: 토크 오프만 쓰지 말고 동적 제동(회생 저항 경유 약한 제동)을 검토한다 (Kawasaki 특허).
