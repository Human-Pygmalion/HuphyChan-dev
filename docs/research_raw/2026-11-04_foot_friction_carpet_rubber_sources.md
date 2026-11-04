# 원문 발췌: 신발 고무와 카펫의 마찰 계수, 발 마찰 무작위화 범위 (2026-11-04, 작업 #200)

등급: **S** = 검색 결과 요약만 읽음(원문 미정독). **L** = 이 저장소의 코드·설정에서 직접 읽음.
**결론: 신발 고무 밑창과 카펫 사이의 마찰 계수 실측값은 검색 6회에서 찾지 못했습니다.**

## 검색한 것과 나온 것

| # | 검색 | 얻은 것 | 등급 |
|---|---|---|---|
| 1 | rubber shoe sole carpet friction static dynamic | 정지/운동 마찰 정의, 시험 방법(ASTM F489 제임스 기계, F609 수평 당김 슬립미터). 카펫 수치 없음 | S |
| 2 | humanoid foot friction randomization range rubber sole | legged_gym 계열 가이드: 기본 [0.5, 1.25], 배포용 넓힘 [0.2, 1.7], 1.5 초과는 드묾. 고무-콘크리트 약 0.8~1.0, 고무-나무 약 0.6~0.8. Isaac 계열 논문 표: 발 마찰 (0.4, 1.1) 또는 U(0.4, 1.0) | S |
| 3 | 고무 밑창 + 카펫 + 트레드 + 털 끼임 | 트레드 방향이 미끄럼 방향과 수직·비스듬하면 동적 마찰 계수가 높고 평행이면 낮음. 홈이 넓을수록 평행 미끄럼에서 낮음. 젖으면 2~3배 낮음. 카펫 수치 없음 | S |
| 4 | 영국 안전보건연구소(HSL) 카펫 + 신발 마찰 | HSE 안내: 거칠기 측정기는 카펫에 부적합, 진자 시험(BS 7976)이 선호됨. 카펫 수치 없음 | S |
| 5 | 고무 마찰 이론(낮은 접촉 압력) | Persson 이론은 높은 압력 극한이라 낮은 압력 질문에 답하지 못함. 실접촉 면적은 겉보기의 약 1 % | S |
| 6 | ASTM F609 카펫 + 고무 뒤꿈치 값 | 방법 표준만. 수치 없음 | S |

## 링크

- legged_gym 마찰 무작위화 가이드: https://leggedgym-ex-doc.readthedocs.io/en/latest/developer_guide/how_to/domain_randomization.html (제3자 문서, 공식 아님)
- 고무 밑창과 고무 바닥 타일 마찰: https://www.tribology.rs/journals/2022/2022-2/5-1188.pdf
- 고무 바닥 매트 트레드 무늬와 마찰: https://www.tandfonline.com/doi/abs/10.1080/17515831.2017.1397311
- 신발 트레드 유한요소 해석: https://eprints.whiterose.ac.uk/id/eprint/164367/1/Trib%20Int%202020%20Shoe%20Tread%20FEA%20ARC.pdf
- 신발-바닥 미끄럼 시험 방법 비교(HSL 공저): https://eprints.whiterose.ac.uk/id/eprint/90813/1/J_2015-Understanding%20the%20friction%20measured%20by%20standardised%20test%20methodologies%20used%20to%20assess%20shoe-surface%20slip%20risk%20-PRE-PROOF%20OPEN%20ACCESS%20VERSION.pdf
- ASTM F609 개요: https://webstore.ansi.org/standards/astm/astmf60996e1
- ASTM F2913(전체 신발 동적 마찰): https://www.dinmedia.de/en/standard/astm-f-2913/383226757
- 고무 마찰 이론(Persson): https://www.arxiv.org/pdf/cond-mat/0502495

## 이 저장소에서 직접 읽은 것 (L)

- 학습 마찰 무작위화: `mujoco-sim/mjlab/src/mjlab/envs/mdp/dr/geom.py` 의 `geom_friction`(기본 축 0, `shared_random`), 정책 49999 의 `checkpoints/terrain10_union_dr_49999/env.yaml` 의 `foot_friction`: 범위 0.3~1.2, 연산 abs, `shared_random: true`.
- 접촉 덮어쓰기: `pygmalion_mjlab/robot.py` 의 `CollisionCfg`(condim 3, friction 0.6, priority 1).
- 접촉 우선순위 실험(MuJoCo 직접 실행): 발 0.3·바닥 1.0 이면 접촉 마찰 0.3, 발 1.2·바닥 0.2 이면 1.2, 우선순위가 같으면 큰 쪽(1.0).
