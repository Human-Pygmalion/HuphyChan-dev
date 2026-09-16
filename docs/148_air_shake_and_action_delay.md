# 148. 공중에 들면 다리가 흔들리는 이유 — 명령 지연

작성일: 2026-09-17 KST
발단: 사용자 지시 — *"정책 실행하다가 로봇을 공중에 들 때가 있는데, 이러면 다리가 막 흔들리네.
가만히 있도록 할수있겠나?"* (열린 일 #9)
함께 닫은 것: 열린 일 #11 (명령 지연 + PD 게인 무작위화)

## ELI5 — 세 문장

로봇을 들어 올리면 발이 땅을 못 딛는데, 정책은 여전히 "몸을 바로잡으려고" 다리를 움직입니다.
땅이 없으면 같은 명령에도 다리가 훨씬 크게 움직이고, 거기에 **명령이 조금 늦게 도착하는 것**이
겹치면 되먹임이 한 박자씩 밀려서 진동이 됩니다. 그래서 땅에서는 멀쩡하던 로봇이 공중에서만
다리를 흔듭니다.

---

## 1. 무엇을 했나

같은 정책(`legonly_ab_v7_standonly_p2` 의 `model_9000`)으로 **두 상황**을 시뮬레이터에서
돌리고 관절의 힘·속도를 비교했습니다.

* **땅** — 평소대로 바닥 위에서 명령 속도 0 으로 서 있게 둠
* **공중** — 매 스텝 몸통의 **자리만** 붙잡아 0.6 m 들어 올림. 자세(기울기)는 놓아 줍니다.
  사람이 몸통을 잡고 든 것과 같습니다.

도구: [`tools/robot_model/airborne_signature.py`](../tools/robot_model/airborne_signature.py)

### 1a. 처음 두 번은 제 설정이 틀렸습니다

~~첫 결과는 "땅에서 다리가 1 N·m 밖에 안 쓴다" 였습니다.~~ **이름표가 어긋나 있었습니다.**
`actuator_force` 는 **액추에이터(ctrl) 순서**인데 저는 **행동 항목의 관절 순서**로 이름을
붙였습니다. 실제 순서는 크랭크 4개가 먼저입니다. 고친 뒤에야 "발목 크랭크가 가장 큰 힘을
낸다" 는 맞는 그림이 나왔습니다.

~~두 번째 결과는 "정책이 2초 만에 주저앉는다" 였습니다.~~ **환경 토글이 빠져 있었습니다.**
손으로 `PYG_ANKLE_MODE` 같은 것 몇 개만 적었더니 `PYG_INIT_BENT`(기준 자세), `PYG_KNEE_EXT`,
`PYG_MASS_DR_JSON` 이 빠졌습니다. **기준 자세가 다르면 행동의 오프셋이 달라져서** 정책이 내는
숫자가 전혀 다른 관절 목표가 됩니다 — 정책은 멀쩡한데 환경이 딴 로봇입니다.

고친 방법: 목록을 손으로 적지 않고, 학습이 스스로 남긴 `repro/launch_manifest.json` 의
`environment` 를 **통째로** 적용합니다. 손으로 적는 목록은 반드시 또 빠집니다.

## 2. 핵심 결과 — 지연이 원인이다

지연이 **없으면 공중에서도 안 흔들립니다.** 지연을 넣으면 흔들립니다. 땅에서는 지연이 있어도
멀쩡합니다.

| 명령 지연 | 땅에서 다리 움직임 | 공중에서 다리 움직임 |
|---:|---:|---:|
| 0 ms | 0.00 rad/s | 0.01 rad/s |
| 20 ms (1 제어 스텝) | 0.00 rad/s | **0.35 rad/s (최대 1.01)** |

(`sum |관절속도|`, 가라앉힌 뒤 8초 평균. 자세한 쓸기 결과는 §2a.)

**35~100 배**입니다. 그리고 땅에서는 변화가 없습니다. 이것이 사용자가 본 현상과 정확히
같은 모양입니다 — 세워 두면 멀쩡, 들면 흔들림.

### 왜 공중에서만 그런가

땅을 딛고 있으면 다리는 몸을 떠받치는 **기둥**입니다. 관절을 조금 움직여도 바닥이 버티므로
다리가 실제로 움직이는 양은 작습니다. 들어 올리면 버틸 것이 없어져 **같은 명령이 훨씬 큰
움직임**을 냅니다 — 되먹임의 실효 이득이 올라간 것입니다. 이득이 올라간 고리에 지연이
끼면 위상이 밀려 진동합니다. 제어에서 아주 흔한 조합입니다.

## 3. 그래서 한 일 — 학습에 지연과 게인 흔들기를 넣었다 (#11)

### 3a. 조사한 것 (원문 발췌)

**roboterax/humanoid-gym** — `humanoid/envs/custom/humanoid_env.py:194-196`:

```python
delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
actions = (1 - delay) * actions + delay * self.actions
actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
```

같은 저장소 `humanoid_config.py` 의 기본값: `action_delay = 0.5`, `action_noise = 0.02`.
친화 값들: `friction_range = [0.1, 2.0]`, `max_push_vel_xy = 0.2`, `push_interval_s = 4`.

**문헌 요약** (검색 2026-09-17): 제어 지연 무작위화는 **0~2 제어 스텝** 또는 **0~10 ms**,
모터 지연은 **0~20 ms**. PD 게인 배율은 **Kp [0.9, 1.1] · Kd [0.5, 1.5]**, 다른 구현은
**강성·감쇠 [0.75, 1.25]**. 한 연구는 *"using a constant 1-step delay on a fixed portion of
environments improves training speed and can prevent the robot from jittering caused by
uncertainty delay during policy execution"* 라고 명시합니다 — **지연을 학습에 넣는 것이
떨림을 막는다**는 우리 결론과 같은 말입니다.

### 3b. 구현

[`delayed_action.py`](../mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/pygmalion/delayed_action.py)
— humanoid-gym 의 방식 그대로 **이번 명령과 지난 명령을 섞습니다.** 섞는 비율이 곧 지연이고,
한 스텝보다 짧은 지연도 표현됩니다(정수 버퍼로는 못 하는 일). 비율은 환경마다 다르고 매 스텝
새로 뽑습니다 — 실물 지연은 일정하지 않기 때문입니다.

PD 게인은 **새로 만들지 않았습니다.** mjlab 에 이미 `dr.pd_gains` 가 있어 그걸 씁니다
(프로젝트 규칙: 새로 짜기 전에 있는 것부터 찾는다).

| 토글 | 기본값 | 뜻 |
|---|---|---|
| `PYG_ACTION_DELAY` | `0` (끔) | 섞는 비율의 최댓값 [제어 스텝]. 우리는 50 Hz 라 1 스텝 = 20 ms |
| `PYG_ACTION_NOISE` | `0` (끔) | 명령에 곱해지는 상대 잡음 표준편차 |
| `PYG_PD_GAIN_DR` | 끔 / `1` = Kp [0.8,1.2] Kd [0.7,1.3] / `"lo,hi,lo,hi"` | PD 게인 배율 범위 |

셋 다 **다른 DR 과 같은 창에서 0 → 전체로 서서히** 켜집니다(`dr_levels`). 게인 배율만은
**1 을 중심으로** 넓힙니다 — 마찰과 달리 0 이 "없음" 이 아니라 1 이 "없음" 이고, 0 을 곱하면
모터가 죽기 때문입니다.

**끄면 예전과 완전히 같습니다** — 확인함: 토글 없이 만든 설정은 `JointPositionActionCfg`
그대로이고 `pd_gains` 이벤트도 커리큘럼 항목도 생기지 않습니다. 진행 중인 학습의 재현성이
깨지지 않습니다.

## 4. 남은 것 — 실물에서 재야 하는 값

지금 답을 못 내는 것은 **우리 실물의 지연이 실제로 몇 ms 인가** 입니다. 시뮬레이션은
"지연이 원인" 까지는 보였지만, 우리 로봇의 지연 값은 시뮬레이션이 알려 줄 수 없습니다.
파이 → CAN → 드라이버 경로를 실측해야 합니다. 이건 모터를 움직이므로 **영상 기록이
필요합니다.**

그 전까지 `PYG_ACTION_DELAY` 는 조사값(0.5~1.0 스텝 = 10~20 ms)을 씁니다.

## 5. 공중 감지를 지금 넣지 않은 이유

처음 계획은 "토크로 공중을 감지해 자세를 고정" 이었습니다. 재 보니 **문턱값을 안전하게
그을 만한 신호가 마땅치 않습니다.**

| 후보 | 땅 | 공중 | 판정 |
|---|---:|---:|---|
| 발목 크랭크 토크 합 | 22.0 | 33.9 | 잘 갈리지만 **자세에 크게 의존** — 걷는 중엔 무의미 |
| 엉덩이 롤 토크 합 | 1.8 | 0.5 | 방향은 맞지만 **절대값이 모터 마찰 수준** |
| 무릎 토크 합 | 1.8 | 0.3 | 같은 문제 |
| 관절 속도 합 | 0.00 | 0.35 | 지연이 있을 때만 갈림 — **증상을 원인으로 쓰는 셈** |

우리 서기 자세는 다리가 거의 곧아서 바닥반력이 관절축을 거의 지나갑니다(모멘트 팔이 짧다).
그래서 **서 있을 때 관절 토크가 원래 작습니다** — 들었을 때와 크게 다르지 않습니다.
제가 처음에 "땅이 당연히 토크가 크다" 고 놓고 한 방향만 쟀던 것은 틀렸습니다.

지연을 학습에 넣는 쪽이 **문턱값 없이** 같은 문제를 없앱니다. 감지·고정 장치는 그래도
필요하다고 판단되면 그때 실물 토크를 재고 넣겠습니다.

---

## 출처

* [roboterax/humanoid-gym — humanoid_env.py](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_env.py)
* [roboterax/humanoid-gym — humanoid_config.py](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_config.py)
* [Humanoid-Gym: RL for Humanoid Robot with Zero-Shot Sim2Real Transfer (arXiv 2404.05695)](https://arxiv.org/abs/2404.05695)
* [Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (arXiv 2512.01996)](https://arxiv.org/html/2512.01996)
* [unitree_rl_gym — G1 config](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py)
