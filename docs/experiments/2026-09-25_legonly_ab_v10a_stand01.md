# legonly_ab_v10a_stand01 — A/B v10 의 대조 팔 (2026-09-25)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다.
> 짝: [[2026-09-25_ab_v10_stand_l1_compare]] (비교노트 — 가설·사전 등록 예측은 거기)

## §0 왜 이 학습을 하는가

서 있는 환경 비율을 0.1 로 되돌려도 **서기 전용 L1 편차 보상**으로 서기를 지킬 수 있는가.
0.3 으로 올렸을 때 커리큘럼이 5,400 회에 강제 승급하고(v6b·v7·v8 모두) 걷기가 후퇴했습니다(v6b).
근거: [[../reward_research/2026-09-25_isaac_asimov_teardown]] §3-2.

## §1 무엇이 다른가

| 설정 | v7 | **이 런** |
|---|---|---|
| 서 있는 환경 비율 | 0.3 | **0.1** |
| 방향유지 비율 | 0.3 | 0.3 |
| 서기 전용 L1 편차 | 없음 | 없음 |

나머지는 v7 과 같습니다(모델 `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix`,
질량 불확실성 JSON, 좌우 단계 명령, 게이트 최대 체류 3,000, 1단계 16,000 · 2단계 10,000+4,000,
씨앗 42, 16,384 환경). 짝 팔과의 차이는 `PYG_STAND_L1` **한 줄뿐**입니다.

발사: `run_v2_scratch.sh --run legonly_ab_v10a_stand01 --ankle AB --vy-stages --gate-max-dwell 3000 --allow-concurrent`
+ `PYG_REL_STANDING=0.1 PYG_REL_HEADING=0.3`. 짝 팔과 **같은 그래픽카드에서 동시에**.

## §1c 판정 기준

비교노트 §2 의 사전 등록 예측 P1~P5 와 같은 도구·같은 조건(§3).

## §2 최종 지표

(완주 후)

## §2c 학습 중 리뷰 (게이트마다 스냅샷, docs/27 체크리스트)

| 시각 | iter | reward | ep_len | noise σ | value loss | entropy | surrogate / LR | fell / low_base | err_vel xy / yaw | dr_factor / vx_max | thermal | 판정(docs/27) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
