# 149. 정책의 떨림·안전·안정성 — 일반화 가능한 방법 정리

작성일: 2026-09-17 KST
발단: 사용자 지시 — *"떨림을 방지하는 '일반화가능한 방법' 도 정리해서, 정책의 '안전',
'안정성' 을 확보하는 방법에 대해 정리해 둬. 논문레퍼런스와 내용정리 포함해."*

## ELI5 — 왜 시뮬에서 잘 되던 정책이 실물에서 위험해지나

시뮬레이터의 모터는 **명령을 즉시, 무한히 빠르게** 따릅니다. 그래서 정책은 매 순간
명령을 크게 흔들어도 벌을 안 받고, 오히려 그렇게 하는 쪽이 점수가 높을 때가 많습니다.
실물 모터는 그럴 수 없습니다. 같은 명령을 주면 **덜덜 떨고, 뜨거워지고, 닳습니다.**

여기에 두 번째 문제가 겹칩니다. 실물 센서는 잡음을 냅니다. 정책이 **입력에 민감할수록**
그 잡음이 출력으로 증폭됩니다. 우리 v7b 가 정확히 이 경우였습니다 — 외력을 잘 버티게
만드느라 민감해졌고, 같은 잡음에 v7 보다 **19 배** 크게 반응합니다.

세 번째는 **겪어 본 적 없는 상황**입니다. 학습에 없던 상태(크레인에 매달림, 넘어지는
중)에 들어가면 정책은 "모른다" 고 말하지 않습니다. **자신 있게 엉뚱한 명령**을 냅니다.

이 문서는 그 셋을 각각 어떻게 다루는지 — 그리고 **어느 층에서 다루는지** 정리합니다.

---

## 0. 먼저, 어느 층에서 고칠 것인가

같은 증상도 층이 다르면 처방이 다릅니다. 순서대로 아래로 내려갑니다.

```
  ┌─ 1층  학습 목적함수      매끄러움을 목적에 넣는다 (보상 / 정규화 / 손실)
  ├─ 2층  학습 제약          "넘지 마라" 를 보상이 아니라 제약으로 건다
  ├─ 3층  신경망 구조        구조적으로 민감해질 수 없게 만든다
  ├─ 4층  배포 시 필터       정책 출력을 실행 전에 검사·수정한다
  ├─ 5층  상황 감지와 전환   학습 밖 상황을 알아채고 다른 모드로 넘어간다
  └─ 6층  하드웨어 가드      마지막 방어선. 토크·속도·한계
```

**위층일수록 근본적이고, 아래층일수록 확실합니다.** 위층만 하면 예외에 뚫리고,
아래층만 하면 원인은 그대로인 채 증상만 막습니다. 실제 시스템은 둘 다 씁니다.

---

## 1층. 학습 목적함수에서 매끄럽게

### 1a. 가장 흔한 것 — 행동 변화율 벌점 (그리고 그 함정)

연속한 두 명령의 차이에 벌점을 답니다. 우리도 `action_rate_l2` 가중치 **−0.1** 로
쓰고 있고, 문헌의 흔한 값도 같습니다. 관절 각도의 **1차·2차 차분**에 각각 벌점을 다는
변형도 씁니다.

**함정은 가중치입니다.** 립시츠 논문이 이걸 숫자로 보여 줍니다 — 매끄러움 계수를
0.002 에서 0.01 로 올렸더니 **과제 점수가 26.03 → 16.11 (−38%)**, 원문 표현으로
*"overly smooth and sluggish behaviors"*. 즉 **떨림을 누르면 과제 성능을 잃습니다.**
우리 경우 그 "과제 성능" 이 방금 얻은 외력 강건성입니다.

그래서 이 방법은 **가장 쉽지만 가장 위험**합니다. 반드시 A/B 로 원래 성능이 안 깎였는지
확인해야 합니다.

### 1b. CAPS — 시간축 + 상태축 두 방향으로 매끄럽게

**Regularizing Action Policies for Smooth Control with Reinforcement Learning**
(arXiv 2012.06644, 이하 CAPS). 매끄러움을 **두 종류**로 나눈 것이 핵심입니다.

* **시간축 매끄러움** — 연속한 두 상태에서 낸 명령이 비슷해야 한다

  `L_T = E_{s_t, s_{t+1}} [ ‖ π(s_t) − π(s_{t+1}) ‖₂ ]`

* **상태축 매끄러움** — **비슷한 상태**에서는 비슷한 명령이 나와야 한다

  `L_S = E_{s_t, s̄ ~ N(s_t, σ)} [ ‖ π(s_t) − π(s̄) ‖₂ ]`

* 합쳐서 `J_CAPS = J − λ_T · L_T − λ_S · L_S`

**둘의 차이가 중요합니다.** 시간축만 누르면 "천천히 움직여라" 가 되어 반응이 굼떠집니다
(1a 의 함정). 상태축은 **반응 속도를 안 건드리고** "입력이 조금 달라졌다고 출력이 확
달라지지 마라" 만 요구합니다 — **잡음 증폭에 정확히 겨눈 항**입니다. 우리 v7b 문제가
바로 상태축 문제입니다.

**실물 결과**: 쿼드로터에서 **소비전력 약 80% 감소**, 게다가 학습이 **8 배 빨라지고
표본이 90% 덜** 들었습니다. 매끄러움을 넣어서 성능이 **올라간** 사례입니다 — 1a 의
맞바꿈이 필연이 아니라는 증거입니다.

확장으로 **Grad-CAPS** (arXiv 2407.04315) 가 명령 변화량 자체를 줄이도록 항을 고쳤고,
이미지 입력용 변형(arXiv 2205.09658)도 있습니다.

### 1c. LCP — 기울기에 직접 벌점

**Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies**
(arXiv 2410.11825).

`max_π  J(π) − λ_gp · E[ ‖ ∇_s log π(a|s) ‖² ]`,  λ_gp = **0.002**

뜻은 CAPS 의 상태축 항과 같습니다 — "관측이 조금 바뀌었다고 행동이 확 바뀌지 마라" —
그런데 **표본 두 개를 비교하는 대신 기울기를 직접** 벌합니다. 미분 가능하고 항이 하나뿐
이라 조절할 손잡이가 적습니다.

논문이 기존 방법의 문제를 이렇게 적습니다: 매끄러움 보상은
*"typically require careful manual design and tuning"*, 저역통과 필터는
*"often dampen or limit exploration"*.

### 1d. 세 방법 비교

| | 무엇을 누르나 | 손잡이 | 대가 |
|---|---|---|---|
| 행동 변화율 벌점 | 시간축 변화 | 가중치 1개 | **반응이 굼떠짐** (검증된 −38% 사례) |
| CAPS | 시간축 + **상태축** | λ_T, λ_S | 상태축은 반응속도를 안 건드림 |
| LCP | **상태축**(기울기) | λ_gp 1개 | 기울기 계산으로 **메모리 증가** |

---

## 2층. 보상이 아니라 **제약**으로 걸기

**Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion**
(arXiv 2308.12517).

핵심 주장: 관절 속도·토크·접촉 같은 것은 **"최대한 작게" 가 아니라 "이 선을 넘지 마라"**
입니다. 보상으로 표현하면 가중치를 로봇마다 다시 맞춰야 하지만, 제약으로 표현하면
*"constraints can be used as generalizable conditions that the controller should satisfy
even when the robot changes"* 이고, *"set automatically from the robot description files"*
— **로봇 설명 파일에서 자동으로 뽑을 수 있습니다.**

효과가 극적입니다: **보상항이 14 개에서 3 개로** 줄었고, 로봇을 바꿔도
*"only a single reward coefficient was adjusted"*.

### 이들이 쓰는 제약과 문턱값

**확률 제약** (위반 확률이 이 값 이하여야 함): 관절 위치 **0.025** · 관절 속도 0.025 ·
관절 토크 0.025 · 몸통 접촉 0.025 · 무게중심 좌표계 0.025 · 걸음 패턴 **0.25**

**평균 제약** (기댓값이 이 값 이하): 옆방향 속도 **0.35 m/s** · 접촉 속도(로봇마다) ·
발 들림 **−0.07 ~ −0.2 m** · 발 높이 **0.09 ~ 0.2 m** · 좌우대칭 **0.1**

알고리즘은 **내부점 정책 최적화(IPO)** — 제약을 로그 장벽 함수로 바꿔 제약 없는 문제로
푸는 방식입니다.

**우리에게 주는 뜻**: 우리 보상은 22 개이고, 그중 여럿이 사실은 "넘지 마라" 입니다
(`torque_limit`, `dof_pos_limits`, `knee_overspeed`, `contact_force_cap`, `thermal_effort`).
이들은 제약으로 옮길 후보입니다. 옮기면 **가중치 조절이 사라집니다.**

---

## 3층. 신경망 구조로 못 민감해지게

**Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid
Locomotion** (arXiv 2504.08246).

가중치 행렬의 **스펙트럼 노름**(가장 큰 특이값)을 제한하면 신경망 전체의 립시츠 상수가
묶입니다. 그러면 **입력이 조금 바뀔 때 출력이 바뀔 수 있는 양의 상한**이 구조적으로
정해집니다 — 학습이 어떻게 되든 그 이상은 못 민감해집니다.

*"By constraining the spectral norm of network weights, SN effectively limits high-frequency
policy fluctuations while significantly reducing GPU memory usage."*

LCP 대비 장점은 **메모리**입니다 (기울기 벌점은 2차 미분이 필요해 비쌈). 논문이
기존 방식을 이렇게 요약합니다: 관절 속도·가속도·에너지 벌점은
*"require extensive hyperparameter tuning"*.

같은 계열: **L2C2** (국소 립시츠 연속 제약), **Lipschitz-Bounded Policy Networks**
(arXiv 2405.11432), **Stability-Certified RL via Spectral Normalization**
(arXiv 2012.13744) — 마지막 것은 스펙트럼 정규화로 **제어이론적 안정성 증명**까지 갑니다.

**왜 이게 사이투리얼 문제인가**: 시뮬은 *"infinite actuator bandwidth and the absence of
torque limits"* 를 가정하므로 정책이 **고주파 토크 변화에 기대는 해**를 찾아냅니다.
실물 액추에이터는 대역폭이 유한해서 그 해가 성립하지 않습니다.

---

## 4층. 배포 시 안전 필터 — 실행 전에 검사한다

정책은 그대로 두고, **출력을 실행하기 전에 가로채서** 위험하면 고칩니다.

**Shield-Loco: Shielding Locomotion Policies with Predictive Safety Filtering**
(arXiv 2606.07193). 4족 로봇에서:

1. 정책이 낸 계획(발 디딜 자리)을 **전체 동역학으로 미리 굴려 봅니다**
2. 위반이 없으면 **그대로 통과**
3. 위반이 예측되면 **표본 기반 최적화**로 가장 가까운 안전한 대안을 찾습니다

핵심은 *"minimally invasive"* — 보수적인 회복 정책으로 갈아타는 것이 아니라 **원래
계획에 최대한 가깝게** 고칩니다. 결과: 위반이 51+ 스텝에서 **29~30 스텝**으로,
추종 비용은 0.009 ± 0.001.

**제어장벽함수(CBF)** 계열도 같은 층입니다 — 안전을 런타임 하드 제약으로 강제합니다.
다만 조사 자료가 한계를 분명히 적습니다: 다리 로봇의 **간헐적 다중 접촉** 동역학에
유효한 CBF 를 만드는 것은 *"exceptionally difficult"* 이고, 보장이 **정확한 모형 지식에
의존**하는데 축소차수 모형의 미모형 접촉이 그걸 쉽게 깬다는 것입니다.

---

## 5층. 학습 밖 상황을 알아채고 모드를 바꾸기

### 5a. 왜 필요한가 — "조용한 실패"

조사 자료의 표현이 정확합니다: 정책은 학습 밖 상태에서 *"fail silently, producing
high-confidence actions"* 합니다. **모른다고 말하지 않고, 자신 있게 틀립니다.**

### 5b. RAPT — 사이투리얼 휴머노이드용 런타임 감시기

**RAPT: Model-Predictive Out-of-Distribution Detection and Failure Diagnosis for
Sim-to-Real Humanoid Robots** (arXiv 2602.01515). 우리 상황에 가장 가깝습니다.

* 시뮬에서 **정상 행동의 확률 모형**을 학습합니다 (GRU 부호기–복호기, 대각 가우시안)
* 배포 중에는 **차원별 음의 로그가능도**로 "지금 관측이 정상에서 얼마나 벗어났나" 를 잽니다
* **이분산(heteroscedastic)** 이라, 원래 변동이 큰 채널은 큰 분산을 받고 일정한 채널은
  조금만 벗어나도 크게 벌점 — 즉 **채널마다 다른 잣대**를 자동으로 갖습니다
* 결과: 제어 50 Hz 에서 **스텝당 1.63 ms**, 실물 78 회 시험에서 **참양성 88.9% ·
  에피소드 오탐 6.7%**, 실패 원인 분류 정확도 75% (21 범주)

**우리가 배울 점 두 가지.** 첫째, 우리가 시도한 **단일 문턱값**(토크 합)과 달리
**차원별·이분산** 으로 보면 갈립니다 — 우리 토크 감지기가 실패한 이유가 여기 있을 수
있습니다. 둘째, 오탐을 **고정 오탐률(0.5%)** 에 맞춰 보정하고 성능을 그 점에서
보고합니다. 우리도 *"서 있는데 잘못 켜질 확률"* 을 먼저 고정하고 나머지를 재야 합니다.

### 5c. 이중 정책 — 위험하면 다른 정책으로

**SafeFall** (arXiv 2602.13656) 은 과제 수행과 낙상 보호를 **두 정책**으로 나누고,
예기치 못한 낙상에서 보호 모드로 자동 전환합니다. 낙상 회복 학습의 표준 방식은 초기
상태를 넓게 뿌리는 것입니다 — 예: **1.1 m 높이에서 2 초 자유낙하, 자세·관절각 무작위**
(arXiv 2506.05516, 2303.05486).

**우리 "다음 목표"(넘어질 때 발작 방지)가 정확히 이 구조**입니다.

---

## 6층. 하드웨어 가드 — 마지막 방어선

학습과 무관하게 **항상** 걸려 있어야 하는 것들입니다. 우리 것으로 확인된 상태:

| | 우리 상태 |
|---|---|
| 토크 한계 | ✅ RS04 120 / RS03 60 N·m, 모형과 실물 일치 |
| 속도 한계 | ✅ 실측 T-N 곡선이 암묵 강제 (200 rpm 에서 토크 0) |
| 관절 한계 | ✅ 보정 좌표에서 검사, 여유 3° |
| 열 | ✅ 과부하 지속시간 표 기반 `thermal_effort` |
| 명령 변화율 제한 | ⚠️ **없음** |
| 통신 끊김 감시 | ⚠️ 확인 안 함 |

---

## 7. 우리에게 적용할 순서 (권고)

지금 우리 문제는 §1c·§1b 가 겨냥하는 **상태축 민감도**입니다 — 잡음을 끄면 v7b 도
완벽히 멈추므로(0.00) 자체 진동이 아니라 **증폭**입니다.

| 순서 | 무엇 | 왜 이 순서인가 | 위험 |
|---|---|---|---|
| **1** | **CAPS 상태축 항** 또는 **LCP** | 증폭에 정확히 겨눔. 반응속도를 안 건드림 | 손잡이 1~2 개, A/B 필요 |
| 2 | 스펙트럼 정규화 | 구조적 상한. 메모리 유리 | 학습 전체에 영향 |
| 3 | 행동 변화율 벌점 올리기 | 가장 쉬움 | **−38% 선례. 마지막 수단** |
| 4 | 들린 상태를 학습에 넣기 | 공중 발작은 증폭이 아니라 OOD | 별건 (#17) |
| 5 | 한계 항목들을 제약으로 이관 | 보상 22 개 중 5 개가 후보 | 큰 개편 |
| 6 | 명령 변화율 제한 (6층) | 확실한 마지막 방어선 | 정책 밖 층 추가 |

**어느 것을 하든 §2 의 밀기 표를 다시 재서 강건성이 안 깎였는지 확인합니다.**
그 확인 없이는 채택하지 않습니다.

---

## 우리 자신의 증거 (이 정리의 출발점)

| | v7 (서기만) | v7b (강건화) |
|---|---:|---:|
| 관찰 잡음 끄고 땅에서 떨림 | 0.001 | **0.00** |
| 잡음 켜고 (실물 조건) | 0.04 | **0.77** (19배) |
| 서 있을 때 토크 실효값 | 3.42 | **4.39 (+28%)** |
| 1.5 m/s 밀기 넘어짐/로봇·분 | 16.51 | **0.00** |

그림: `docs/data/airborne_noise_sweep.png` (명령 떨림 → 다리 움직임, 단조 증가)

---

## 출처

**매끄러움**
* [Regularizing Action Policies for Smooth Control with RL — CAPS (arXiv 2012.06644)](https://arxiv.org/abs/2012.06644) · [프로젝트 페이지](http://ai.bu.edu/caps/)
* [Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies (arXiv 2410.11825)](https://arxiv.org/html/2410.11825)
* [Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion (arXiv 2504.08246)](https://arxiv.org/abs/2504.08246)
* [Gradient-based Regularization for Action Smoothness (Grad-CAPS, arXiv 2407.04315)](https://arxiv.org/abs/2407.04315)
* [Image-Based Conditioning for Action Policy Smoothness (arXiv 2205.09658)](https://arxiv.org/abs/2205.09658)
* [On Robust RL with Lipschitz-Bounded Policy Networks (arXiv 2405.11432)](https://arxiv.org/html/2405.11432v1)
* [Stability-Certified RL via Spectral Normalization (arXiv 2012.13744)](https://arxiv.org/pdf/2012.13744)

**제약·안전**
* [Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion (arXiv 2308.12517)](https://arxiv.org/html/2308.12517v3)
* [Shield-Loco: Shielding Locomotion Policies with Predictive Safety Filtering (arXiv 2606.07193)](https://arxiv.org/html/2606.07193)
* [CBF-RL: Safety Filtering RL in Training with Control Barrier Functions (arXiv 2510.14959)](https://arxiv.org/pdf/2510.14959)
* [RL for Safe Robot Control using Control Lyapunov Barrier Functions (arXiv 2305.09793)](https://arxiv.org/pdf/2305.09793)

**학습 밖 상황·낙상**
* [RAPT: Model-Predictive OOD Detection and Failure Diagnosis for Sim-to-Real Humanoid Robots (arXiv 2602.01515)](https://arxiv.org/html/2602.01515)
* [SafeFall / Kung Fu Athlete Bot (arXiv 2602.13656)](https://arxiv.org/pdf/2602.13656)
* [Learning to Recover: Dynamic Reward Shaping for Fallen Robots (arXiv 2506.05516)](https://arxiv.org/pdf/2506.05516)
* [Learning Arm-Assisted Fall Damage Reduction and Recovery (arXiv 2303.05486)](https://arxiv.org/pdf/2303.05486)
* [A System-Level View on Out-of-Distribution Data in Robotics (arXiv 2212.14020)](https://arxiv.org/pdf/2212.14020)

**사이투리얼 무작위화**
* [Humanoid-Gym (arXiv 2404.05695)](https://arxiv.org/abs/2404.05695) · [roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym)
* [Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (arXiv 2512.01996)](https://arxiv.org/html/2512.01996)
* [unitree_rl_gym G1 config](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py)

관련 문서: [[148_air_shake_and_action_delay]] · [[reward_research/2026-09-17_standing_tremble]]
