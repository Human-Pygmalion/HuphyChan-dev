# 정체(stagnation) 조기종료 — 문헌·코드 조사

작성일: 2026-10-09 KST
발단: 지도교수 피드백 *"거친 지형에서 명령 방향으로 몇 cm 이상 몇 초간 못 움직이면 정체(stagnation) 종료를 넣어라. 지금은 roll/넘어짐 종료뿐이다."*
성격: 문헌·코드 조사 전용. 보상·종료 코드는 건드리지 않았다. 설계 결정은 계획자 몫이므로 §6 의 "권고"는 근거 정리일 뿐이다.
원문 확인 방법: arXiv PDF 를 내려받아 `pdftotext` 로 본문 직접 확인, 코드는 GitHub raw 파일 직접 확인. 아래 인용은 모두 그 원문에서 복사했다. 웹 검색 요약문만 보고 쓴 항목은 "(검색 요약만, 원문 미확인)" 으로 표시했다.

---

## 0. 한 줄 결론

1. **"일정 시간 못 움직이면 에피소드를 끝낸다" 는 형태의 정체 종료를, 조사한 어떤 논문·코드에서도 찾지 못했다.** (legged_gym, Isaac Lab 의 속도 과제·H1·G1, mjlab 속도 과제, unitree_rl_gym G1, Extreme Parkour 코드, Robot Parkour Learning 코드, Walk These Ways, ANYmal Parkour, Radosavovic 2024, Berkeley Humanoid, HumanoidBench, Booster Gym, Humanoid-Gym, Miki 2022, Agarwal 2022.) 이것은 "없다는 증명"이 아니라 "찾은 범위에서 없다" 이다. 비공개 코드는 못 봤다.
2. **같은 문제는 세 가지 다른 도구로 다뤄진다.** (a) 에피소드 끝의 걸은 거리로 지형 난이도를 내리는 **지형 커리큘럼 강등**(Rudin 2021 이후 표준), (b) **보상 벌점**(Rudin 2022 의 stalling penalty, ANYmal Parkour 의 "Don't wait"), (c) 추종 보상 모양을 바꾸는 방법(선형 항, 허용 대역 보너스).
3. **정체 종료를 넣는다면 "실패(bootstrap 없음)" 로 둘지 "절단(truncation, bootstrap 함)" 으로 둘지가 핵심 설계 선택**이고, 선택에 따라 효과가 정반대다. 절단으로 두면 *데이터를 잘라내는 효과만* 있고 정책에게 정체를 피하라는 신호는 주지 못한다(§3).
4. 이륜(biped) 에서 "서서 살아있음 보상을 거둔다" 는 로컬 옵티멈은 **여러 곳에 문서화돼 있다**(§5). 특히 Light-Loco-Parkour(2026-08)는 휴머노이드가 *지수 추종 보상만 있으면 장애물 앞에서 멈춘다* 고 직접 서술한다.

---

## 1. 정체 종료를 쓰는 곳은 있나 — 감사 표

"종료 조건" 열은 해당 논문·코드가 실제로 쓰는 것. "정체 처리" 열은 정체(안 움직임)를 어떻게 다루는가.

| 작업 | 종료 조건 (원문) | 정체 처리 | 근거 |
|---|---|---|---|
| legged_gym (Rudin 2021 구현) | `check_termination`: 몸통 접촉력 > 1 N, 시간 초과 | **지형 커리큘럼 강등** (아래 §2). 종료는 아님. 보상 `stand_still` 은 *명령이 0일 때만* 작동 | 코드 |
| Isaac Lab 속도 과제 | `time_out`, `base_contact`(임계 1.0 N) 둘뿐 | 커리큘럼 `terrain_levels_vel` 강등 | 코드 |
| Isaac Lab H1·G1 거친 지형 | 위와 같음 + 몸통 링크로 바꿈. 보상 `termination_penalty = -200` | 종료 벌점 -200 (`is_terminated`, 시간초과는 제외) | 코드 |
| mjlab 속도 과제 (기본값) | `time_out`, `fell_over`(기울기 70°), `out_of_terrain_bounds`(**time_out=True**) | 커리큘럼 `terrain_levels_vel`; 보상 `variable_posture`·`feet_air_time`(명령 게이트) | 코드 |
| unitree_rl_gym G1 | `terminate_after_contacts_on = ["pelvis"]`; 보상 `alive=0.15`, `contact_no_vel=-0.2` | 정체 종료 없음 | 코드 |
| Extreme Parkour (Cheng 2023) | roll>1.5, pitch>1.5, 높이<-0.25, 시간초과/목표도달 | 커리큘럼 강등 | 논문+코드 |
| Robot Parkour Learning (Zhuang 2023) | roll, pitch, z_low, z_high (+ 선택 항 `abnormal_dynamics`, `timeout_at_border`) | 커리큘럼 강등 (명령 크기 게이트 포함) | 코드 |
| ANYmal Parkour (Hoeller 2023) | 몸통 충돌, 발 충격력 > 임계; 종료 벌점 -200 | **보상 벌점 "Don't wait"** | 논문 표 S2 |
| Walk These Ways (Margolis 2022) | (본문에 정체 종료 없음) | 보상을 `r_task·exp(c_aux·r_aux)` 로 합성해 *조기종료 유도 방지* | 논문 |
| Miki 2022 (ANYmal 지각 보행) | 몸통 지면 충돌, 큰 기울기, 관절 토크 한계 초과 | 정체 종료 없음 | 논문 |
| Agarwal 2022 | pitch>90° 또는 몸통·머리 충돌 | 정체 종료 없음 | 논문 |
| Rapid Motor Adaptation | 높이 하락 시 조기종료 | 정체 종료 없음 | 논문 |
| Radosavovic 2024 (인간형 험지) | 본문에 종료 조건 서술 없음. 보상에 **alive reward = 1** + 지수 추종(σ=0.2) | 정체 처리 서술 없음 | 논문 부록 |
| Berkeley Humanoid (Liao 2024) | 본문에 종료 조건 서술 없음 | 정체 처리 서술 없음 | 논문 부록 A |
| HumanoidBench | 보행: 1000 스텝 또는 골반 높이 < 0.2 | 정체 종료 없음; 보상이 `stable × tol(vx,(1,∞),1)` 곱셈 형태 | 논문 부록 |
| Booster Gym (2025, 인간형) | 시간초과 *(bootstrap)*; 낮은 몸통 높이·과속 시 종료(이후 보상 없음); **프레임 보상을 0 이상으로 clip** | 명령 일정 확률로 "stand still" | 논문 |
| Light-Loco-Parkour (2026-08, 인간형) | 시간초과·경계이탈·관절속도(→ *bootstrap*); 몸통접촉·가속도·전복(→ *실패*) | 정체 종료 없음; **속도 여유 보너스**로 해결 | 논문 |
| LEEPS (IROS 2024) | — | stalling penalty 를 쓴다는 *검색 요약*만 확인, 원문 PDF 접근 실패(서버 응답 없음) | (검색 요약만, 원문 미확인) |

핵심 직접 인용 (종료가 "넘어짐·충돌·한계 초과" 위주임을 보여주는 것):

- Miki et al. 2022, *Learning robust perceptive locomotion for quadrupedal robots in the wild*, Methods "Termination": "Termination criteria are: body collision with the ground, large body tilt, and exceeding the joint torque limit of the actuators."
- Agarwal et al. 2022, *Legged Locomotion in Challenging Terrains using Egocentric Vision*, 부록: "We terminate if the pitch exceeds 90° or if the base or head of the robot collides with an object."
- Booster Gym (arXiv 2506.15132), Episode Design: "...we implement additional early termination conditions, including excessively low base height or base velocity exceeds a threshold. Under these conditions, the environment is reset without granting subsequent rewards..." (※ 속도 *상한* 초과이지 *하한* 이 아니다.)

---

## 2. 표준 대안 — 걸은 거리 기반 지형 커리큘럼

### 2-1. 원 논문 (Rudin et al. 2021)

*Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*, arXiv 2109.11978, §3.1 Game-Inspired Curriculum:

> "If a robot manages to walk past the borders of its terrain, its level is increased, and at the next reset, it will start on more difficult terrain. However, if at the end of an episode it moved by less than half of the distance required by its target velocity, its level is reduced again."

같은 절에서 명령 크기와 지형 한 변의 길이: 지형은 8 m 정사각 타일, 명령은 "kept constant for the duration of an episode", 에피소드 길이 20 s.

### 2-2. 코드로 본 정확한 규칙

legged_gym `legged_robot.py::_update_terrain_curriculum` (leggedrobotics/legged_gym, master):

```python
distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
# robots that walked far enough progress to harder terains
move_up = distance > self.terrain.env_length / 2
# robots that walked less than half of their required distance go to simpler terrains
move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
```

- 승급: 걸은 거리 > 지형 길이의 절반 (명령과 무관한 *절대* 기준).
- 강등: 걸은 거리 < 0.5 × |명령 속도| × 에피소드 길이 (명령에 *상대적* 기준). 승급이 우선.
- **이 규칙은 "에피소드가 끝났을 때(종료든 시간초과든) 한 번" 평가**되며, 도중에 에피소드를 끊지 않는다.
- 약점: 명령이 작거나 0이면 강등 임계가 0 에 가깝고, 표류만으로도 "승급" 할 수 있다(지형 길이의 절반만 넘으면 됨).

Isaac Lab `terrain_levels_vel` (isaac-sim/IsaacLab main) 도 같은 규칙. docstring: "decrease the difficulty when the robot walks less than half of the distance required by the commanded velocity."

### 2-3. 변형들 (실제 코드)

| 출처 | 승급 | 강등 | 비고 |
|---|---|---|---|
| legged_gym | 거리 > 지형길이/2 | 거리 < 0.5·|cmd|·T | 위와 동일 |
| Extreme Parkour 코드 (`chengxuxin/extreme-parkour`) | 거리 > **0.8**·cmd_x·T | 거리 < **0.4**·cmd_x·T | 논문 본문은 "traverse more than half the length" / "travel less than half the expected distance vcmd T" 로 서술하나 코드는 0.8/0.4. **논문과 코드가 다르다** |
| Robot Parkour Learning 코드 (`ZiwenZhuang/parkour`, `_get_terrain_curriculum_move`) | 거리 > 1.5 × 블록길이 (선택: 시간초과로 끝났을 때만) | 거리 < 블록길이 **그리고 |cmd| > lin_cmd_cutoff** | 명령이 작은 환경은 강등하지 않는다 |
| mjlab main (`terrain_levels_vel`, 2026) | 거리 > min(0.8·명령거리, 지형길이/2), **그리고 명령거리 ≥ 1.0 m** | 거리 < 0.5·명령거리 | 에피소드가 일찍 끝나면 남은 시간만큼 현재 명령을 *빚으로* 더해 명령거리를 계산(넘어져도 요구치가 줄지 않음). 주석: "drift cannot promote an env that was asked to stand still." |

**프로젝트 주의**: 이 저장소의 vendored 사본 `mujoco-sim/mjlab/src/mjlab/tasks/velocity/mdp/curriculums.py` 는 위 표의 *legged_gym 식 옛 규칙*(시작 위치 `env_origins` 기준, 명령거리 보정 없음, `min_commanded_distance` 없음)이다. mjlab main 의 개량 버전과 다르다. 정체 문제를 커리큘럼으로 다루려면 이 차이가 먼저 걸린다.

### 2-4. 이 규칙의 의미: "정체 종료" 와의 차이

커리큘럼 강등은 *정책이 정체했다* 는 사실을 **다음 리셋 때의 지형 난이도** 로 반영한다. 정체한 에피소드 자체는 시간초과까지 계속되고, 그 사이의 보상은 그대로 학습 데이터가 된다. 즉 정체 행동에 대한 *즉시* 패널티는 없다. 지도교수 피드백(정체 종료)은 이 부분을 앞당기자는 것으로 읽힌다.

---

## 3. 조기종료의 함정 (Proximal Policy Optimization 기준)

### 3-1. 종료는 암묵적 보상이다

종료는 "남은 보상의 합을 0 으로 만든다". 스텝 보상의 부호에 따라 에이전트의 유인이 정반대가 된다.

- Kostrikov et al. 2019, *Discriminator-Actor-Critic* (arXiv 1809.02925): "a strictly positive reward function prevents the agent from solving tasks in a minimal number of steps and a strictly negative reward function is not able to emulate a survival bonus." 또 "since the reward function is strictly positive it implicitly provides a survival bonus. In other words, ... it always rewards the policy for avoiding absorbing states."
- Booster Gym: "Furthermore, the total reward for each frame is clipped to zero to avoid incentivizing early termination with negative rewards." (스텝 보상을 0 이상으로 clip)
- Walk These Ways (arXiv 2212.03238): "...the robot might abandon its task or choose an early termination when the task reward is overwhelmed by penalties from the auxiliary objectives. To resolve this, ... we force the total reward to be a positive linear function of the task reward by computing it as r_task exp(c_aux r_aux)..." (c_aux = 0.02)
- DeepMimic (arXiv 1804.02717): "Once early termination has been triggered, the character is left with zero reward for the remainder of the episode. This instantiation of early termination provides another means of shaping the reward function to discourage undesirable behaviors."
- Isaac Lab H1·G1: `termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)`. 그리고 `is_terminated` 의 docstring: "Penalize terminated episodes that don't correspond to episodic timeouts."
- ANYmal Parkour 표 S2·S3 의 `Termination` 행: 가중치 -200 (이동 정책), -0.5 (항법).

**정체 종료에 대한 함의**: 이 프로젝트처럼 스텝 보상의 합이 *양수*(추종·직립·자세 보상이 큼)이면, 실패로 처리한 정체 종료는 "남은 시간 × 스텝 보상" 만큼의 암묵적 벌점이다. 에피소드 초반에 걸릴수록 크다. 반대로 스텝 보상이 *음수*(벌점이 지배)인 정책에게는 종료가 *보상* 이 되어 일부러 정체·넘어짐을 택할 수 있다.

### 3-2. 시간초과(절단)와 실패(종료)의 bootstrap 구분

- Rudin 2021 §2.2.2 Reset Handling: "Resets based on failure or reaching a goal are not a problem because the critic can predict them. However, a reset based on a time out can not be predicted (we do not provide episode time in the observations). The solution is to distinguish the two termination modes and augment the reward with the expected infinite sum of discounted future rewards in a time-out case." 부록 A.2: bootstrap 이 총보상을 "approximately 10 % to 20 %" 개선.
- Pardo et al. 2018, *Time Limits in Reinforcement Learning* (arXiv 1712.00378), §1: "...for bootstrapping methods, we argue to bootstrap at states where termination is due to time limits, or more generally any other causes than the environmental ones." 같은 논문 §3: "A common mistake, however, is to then consider the terminations due to such time limits as environmental ones."
- Pardo (시간 한도가 과제의 일부인 경우의 실험, Figure 5): 시간 한도를 모르는 에이전트는 "the time-unaware PPO ... learns to actively stay in place to at least accumulate the rewards for staying alive." (γ=1 실험) — 종료/비종료 구분이 틀리면 *정체 자체가 유인* 이 될 수 있다는 직접 사례.
- 구현: rsl_rl `ppo.py`: `self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * extras["time_outs"].unsqueeze(1)...)` — `time_outs` 플래그가 켜진 전이만 bootstrap. mjlab 의 `TerminationTermCfg(time_out=True)` 가 이 플래그로 이어진다(`_truncated_buf`). mjlab 기본 설정이 이미 `out_of_terrain_bounds` 를 `time_out=True` 로 둔다.
- Light-Loco-Parkour (arXiv 2608.02653) 는 종료를 항목별로 갈랐다: "Time-out. ... marked as a time-out so the value function bootstraps from the final state rather than receiving a failure return." / "Out of bounds. ... It bootstraps rather than penalizes, signaling the end of usable terrain rather than a failure." / "Torso contact. A contact above 1 N on the torso resets the episode, penalizing the collisions and falls we want to avoid."
- Rudin 2022 (arXiv 2209.12827): 시간이 관측에 들어가는 위치 과제에서는 오히려 bootstrap 을 *뺐다*: "since there is no termination condition for this task [velocity tracking]. However, in our case, the time until task termination is clearly defined and provided as input to both the actor and the critic. We, therefore, remove the bootstrapping and obtain more stable training."

### 3-3. 정체 종료를 실패로 볼까 절단으로 볼까 (논리 정리, 직접 출처 없음)

Pardo 의 원칙을 적용하면: bootstrap 해야 하는 것은 "환경의 참 종료가 아닌 원인" 이다. 정체는 환경이 끝난 것이 아니라 *우리가 데이터 품질을 위해 끊는 것* 이다. 그러나 정체 판정은 **정책의 행동에 의존** 하므로 시간초과와 성격이 다르다.

| 처리 | 학습 신호 | 위험 |
|---|---|---|
| 절단(bootstrap) | 끊긴 상태의 가치 V(s) 를 이어 쓰므로, 정체가 계속된다고 예측한 가치가 그대로 이어진다. 정책은 정체를 피할 유인이 *없다*. 효과는 "정체 에피소드의 낭비된 표본을 줄이고 지형 리셋을 앞당김" 뿐 (DeepMimic 이 말한 "curating mechanism") | 신호가 없어서 정체가 줄지 않을 수 있음 |
| 실패(no bootstrap) | 남은 보상 전부를 잃는 암묵적 벌점. 신호가 있음 | 스텝 보상이 양수일 때만 의도대로. 임계가 어긋나면 정상 행동(발 딛기 직전의 정지, 장애물 앞 숙고)도 벌줌. 값 함수가 "정체 직전" 상태를 구분하려면 *정체 타이머가 관측에 있어야* 예측 가능 — 없으면 Rudin 이 말한 "can not be predicted" 문제가 생김 |
| 확률적 실패 | CaT(Constraints as Terminations, arXiv 2403.18765): 위반 시 즉시가 아니라 확률 δ 로 미래 보상을 자름. "terminating the episode if any constraint is violated might be overly conservative with respect to the constraints and impair exploration and learning." Light-Loco-Parkour 의 전복 종료도 스텝당 확률 0.01 | 구현 복잡도 증가 |

### 3-4. 그 밖의 함정 (출처 있는 것만)

- 이른 종료가 탐색을 막는다: BeamDojo (arXiv 2502.10363): "conventional early termination disrupts entire trajectories, making it extremely difficult to acquire safe foothold samples when learning from scratch." (희소 디딤돌 과제의 *넘어짐 종료* 에 대한 서술이나, 정체 종료도 같은 위험을 공유한다.)
- 종료 조건이 너무 엄격하면 학습이 안 된다: CaT: "once the agent enters a region of constraint violation, the episode always terminates and the agent does not learn anything."
- 종료의 실제 효용: ANYmal Parkour: 발 충격 종료는 "essential to get a transferable motion. Without it, the robot learns to jump down from the top." (종료는 *원치 않는 행동을 막는* 도구로 효과가 확인됨.)

---

## 4. 보상 쪽 대안 — "명령이 0 이 아닌데 서 있기" 억제

| 이름·출처 | 식 (원문) | 가중치 | 성격 |
|---|---|---|---|
| **Stalling penalty**, Rudin et al. 2022, *Advanced Skills by Learning Locomotion and Local Navigation End-to-End* (arXiv 2209.12827) §II-B.4 | r_stall = -1 if ‖ẋ_b‖ < 0.1 m/s and ‖x_b − x*_b‖ > 0.5 m; 0 otherwise | 서술상 "a small penalty" | 이진 벌점. 동기: 음의 벌점 + 양의 최종 보상 구조에서 할인 때문에 "the policy learns to wait until the last moment" |
| **"Don't wait"**, ANYmal Parkour 표 S2; Zhang et al. 2023 *Learning Agile Locomotion on Risky Terrains* (arXiv 2311.10484) 표 IV 도 사용 | 1(‖v_b‖ < 0.2) | -1 (ANYmal Parkour) | 이진 벌점. 같은 표에 "Move in direction" = cos⟨v_b, r* − r⟩ (가중치 +1) |
| **Zero-command 전용 stand still**, legged_gym `_reward_stand_still` | Σ|q − q_default| · 1(‖cmd_xy‖ < 0.1) | (설정값) | *명령 0 일 때만* 자세 유지. 명령이 있으면 0 이므로 정체 억제 효과 **없음** |
| Isaac Lab `stand_still_joint_deviation_l1` | joint_deviation_l1 · 1(‖cmd_xy‖ < command_threshold=0.06) | — | 위와 같은 성격 |
| BeamDojo (인간형) | ‖θ‖² · 1(‖v_cmd_xy‖² < ε), 가중치 -0.05 | -0.05 | zero-command 전용 |
| 명령 게이트 발 공중시간, legged_gym | Σ (t_air − 0.5)·1(첫 접지); `rew *= ‖cmd_xy‖ > 0.1` | 설정값 | 명령이 없으면 보상 0 |
| Isaac Lab 이륜 `feet_air_time_positive_biped` | 한 발 지지 중 `min(in_mode_time)` 을 threshold 로 clip; `*= ‖cmd_xy‖ > 0.1` | H1: 0.25 | 한 발씩 들기를 유도 (제자리 서 있기에서는 0) |
| mjlab `feet_air_time` | 보상 = Σ 1(0.05 < t_air < 0.5); `command_threshold=0.5` 로 게이트 | — | 명령 합(선형+각) > 0.5 일 때만 |
| **지수 추종**, legged_gym·Isaac Lab·mjlab | r = exp(−‖v_cmd − v‖² / σ²); mjlab 기본 σ² = 0.25 (std 0.5), 가중치 2.0 | — | 서 있을 때(v=0) r = exp(−‖v_cmd‖²/σ²) (§5 계산 표) |
| **선형(비제곱) 추종**, Zhuang parkour 코드 `world_vel_l2norm` | ‖v_cmd − v_world‖₂ (**제곱 없음**), 가중치 -1.0; alive = +2.0 | — | 오차가 작을 때도 기울기가 유지된다 |
| **내적 추종**, Extreme Parkour 식 (2) | r_tracking = min(⟨v, d̂_w⟩, v_cmd) | — | 명령 방향 성분의 속도 그 자체를 보상. 서 있으면 0, 상한 v_cmd 에서 포화. 논문: "world frame is used. This is done to prevent the robot from exploiting the reward and learning the unintended behavior of turning around the obstacle." |
| **속도 여유 보너스**, Light-Loco-Parkour 식 (3) | r_slack = 1[ v_x^c / v_x ∈ [0.3, 1.5] ] | — | 명령 속도의 30~150% 이면 보너스. 동기: "a policy trained from scratch tends to halt in front of an obstacle to avoid falling. Under the exponential tracking reward of Eq. (1) alone, this timidity is never overcome and the robot never crosses." |
| **변위·목표 근접 보상**, Light-Loco-Parkour 식 (9) | r_task = exp(−‖p* − p_t‖²/σ_c) | — | "keeps the robot committed to crossing rather than stalling in front of the obstacle" (주행계 없이 목표가 있을 때) |
| **보상 합성**, Walk These Ways | r_task · exp(c_aux · r_aux), c_aux = 0.02 | — | 벌점이 과제 보상을 압도하지 못하게 곱셈 형태 |
| 한 곳에 모은 제약, CaT / SoloParkour | `c_still = (‖q − q*‖² − ε_still) · 1[v_des = 0]` | — | *명령 0 일 때만* |
| 수동 단계 제거 | Rudin 2022: r_bias 는 "removed once r_task reaches 50 % of its maximum value" | — | 탐색 보조 보상을 학습 중 끄는 방식 |

**관찰**: 명령이 0 이 아닌 상태의 정체를 직접 겨냥한 것은 Rudin 2022 의 stalling penalty, ANYmal Parkour 의 "Don't wait", Light-Loco-Parkour 의 속도 여유 보너스 셋이다. 나머지 "stand still" 류는 모두 **명령 0 일 때만** 켜지는 것이라 이 문제와 반대 방향이다.

---

## 5. 이륜·인간형 특유의 증거 — "서서 보상 수확" 은 문서화된 로컬 옵티멈인가

**예, 여러 곳에 있다.** 다만 대부분 "서 있기가 안정적이라서" 라기보다 "발을 떼는 순간 넘어질 위험 + 서 있는 동안 받는 양의 보상" 구조로 설명된다.

- **Light-Loco-Parkour** (2026-08, 90 cm 인간형, arXiv 2608.02653): "On the hardest terrains, such as stepping stones and balance beams, a policy trained from scratch tends to halt in front of an obstacle to avoid falling. Under the exponential tracking reward of Eq. (1) alone, this timidity is never overcome and the robot never crosses." — 가장 직접적인 증거. 해결: §4 의 속도 여유 보너스.
- **Learning to Locomote** (Reda et al., SIGGRAPH Asia 2020, arXiv 2010.04304) — 생존 보너스(alive bonus): "If the survival bonus term is too large, however, the algorithm exploits the survival bonus reward while neglecting other reward terms. This results in a character that balances but never steps forward." 요약: "Values that are too small or too large leads to local minima corresponding to falling-forward and standing still, respectively." 또한 위치 목표(PD 타깃) 행동공간에서: "it is relatively easy to learn a constant set of PD-targets which maintain a static standing posture, which is rewarded via the survival bonus." (이륜 Walker2D 가 가장 심했다: "Walker2D suffers more severely from the problem of a local minimum.") — 우리 로봇도 위치 목표(PD) 행동공간이다.
- **Walk the PLANC** (arXiv 2601.06286, 인간형 디딤돌): "The optimization landscape for such precise, sparse-foothold locomotion is too difficult for PPO to traverse from scratch; without a structured prior, the policy consistently collapses to a local optimum (e.g., standing still)." 해결: 교사-학생 증류로 시작점 제공.
- **Isaac Lab issue #458** (isaac-sim/IsaacLab, 4족 Solo12, 평지): "solo12 prefer to stand still after falling several time instead of moving." 유지보수자 답: "The robot may learn to stay still if the penalty for moving is too high compared to the reward for the task." 사용자 KyleM73: "even for standard robots like Go1 ive seen it learn to stand still and it wasn't until iteration ~2000 that it started learning to walk" 및 "make the standard deviation param for velocity tracking smaller. I find that the defaults are too permissive and rewards the robot even for somewhat large velocity errors." (공식 논문이 아니라 이슈 댓글이므로 증거 등급은 낮다.)
- **Radosavovic 2024** (arXiv 2410.03654, 인간형 험지 미세조정): 보상 = 지수 선형·각속도 추종(σ=0.2) + **alive reward r_a := 1** + 다수 정규화 항. 정체 문제는 서술하지 않았다. 지수 σ=0.2 는 좁아서(서 있을 때 거의 0) 추종 신호 자체는 날카롭고, alive=1 은 서 있는 정책에게 스텝당 상수 보상을 준다 — 즉 "살아있음 보상 + 지수 추종" 조합이 *문서화된 성공 사례* 이기도 하다.
- **Booster Gym** (인간형, 2025): 명령 재추첨 때 "with a certain probability, the command is set to 'stand still' to encourage the robot to learn transitions between standing and walking", 보행 보상은 기준 보행주기에 따라 두 발로 걷기를 보상. 정체 종료는 없다.
- **unitree_rl_gym G1**: `alive = 0.15`, `contact_no_vel = -0.2`, `contact = 0.18`(보행주기 접지 일치 보상). 정체 종료 없음. 발바닥이 접지한 채 미끄러지는 것을 벌하는 항(`contact_no_vel`)이 있다.
- 4족 쪽 참고: Rudin 2022 stalling penalty 의 동기는 "서서 기다리다 마지막에 달린다" 인데, 이는 *최종 보상 + 지속 벌점 + 할인* 구조의 위치 과제에서 나온 것이다. 속도 추종 과제는 구조가 다르다.

**이 프로젝트의 보상 구조에서 서 있기가 얼마나 남는가 (계산, 직접 출처 아님)**: mjlab 속도 과제 기본값(`track_linear_velocity` 가중치 2.0, σ² = 0.25; `upright` 1.0; `pose` 1.0)을 그대로 쓴다고 가정한 계산이다. **실제 프로젝트 설정은 감사하지 않았다.**

| 명령 속도 ‖v_cmd‖ [m/s] | 서 있을 때 exp(−‖v_cmd‖²/0.25) | ×2.0 |
|---|---|---|
| 0.3 | 0.70 | 1.40 |
| 0.5 | 0.37 | 0.74 |
| 0.8 | 0.077 | 0.15 |
| 1.0 | 0.018 | 0.04 |

즉 명령이 0.5 m/s 이하이면 서 있는 것만으로 추종 보상의 37~70% 가 나오고, `upright`·`pose`(가만히 서 있을수록 최대)는 그대로 다 받는다. 거친 지형에서 한 걸음이 넘어짐 위험을 동반하면 서 있기가 매력적인 로컬 옵티멈이 되는 구조다. 반대로 명령이 1 m/s 쯤이면 서 있기의 추종 보상은 사실상 0 이라 이 문제가 약하다. 명령 분포의 하한(작은 명령 비율)이 중요한 이유다.

---

## 6. 권고를 뒷받침하는 근거 요약 (설계 결정 아님 — 계획자 판단용)

1. **정체 종료를 넣는다면 반드시 명령 크기로 게이트한다.** 근거: Zhuang parkour 코드(`less_moved` 에 `lin_cmd_cutoff`), mjlab main(`min_commanded_distance = 1.0`), legged_gym `feet_air_time`(`‖cmd‖ > 0.1`). 명령이 작은 환경에서 서 있는 것은 정답이다.
2. **판정량은 "명령 방향 변위", 창은 슬라이딩 윈도우, 명령이 재추첨되면 타이머 리셋.** 근거: Extreme Parkour 는 목표 방향 성분만 보상(식 2)한다. 재추첨 직후 몇 초는 정당하게 가속 중이다. (직접 출처 없음 — 논리)
3. **실패/절단 선택**: §3-3 표. 절단으로 두면 신호가 없고, 실패로 두면 스텝 보상이 양수일 때만 의도대로이며 정체 타이머를 관측에 넣지 않으면 값 함수가 예측하지 못한다(Rudin 2021 §2.2.2 와 같은 논리).
4. **종료 대신(또는 함께) 쓸 수 있는 문헌 선례**: (i) 커리큘럼 강등(이미 있는 도구, 단 이 저장소 사본은 옛 규칙), (ii) stalling penalty / "Don't wait" 계열의 이진 벌점(‖v‖<0.1~0.2 m/s), (iii) 속도 여유 보너스(Light-Loco-Parkour, 인간형 직접 증거).
5. **측정 먼저**: 로그에서 "명령 ≠ 0 인데 변위 < 임계" 인 에피소드의 비율, 그 에피소드의 지형 레벨 분포를 보면 정체가 실제 문제인지(지형 난이도 대비) 판별된다. 이 분포가 특정 레벨 이상에 몰려 있으면 종료 이전에 커리큘럼 승급 임계가 문제일 수 있다.

---

## 7. 한계·주의

1. "정체 종료를 쓰는 곳 없음" 은 **조사 범위 내 결론**이다. 비공개 코드(Radosavovic, Berkeley Humanoid 학습 코드 등)와 비공개 산업 코드는 확인하지 못했다. 웹 검색(확장 모드)에서도 직접 선례는 나오지 않았다.
2. LEEPS(IROS 2024) 원문은 서버 접근이 안 돼 확인하지 못했다. "stalling penalty + termination penalty" 는 검색 요약에서만 확인.
3. 논문 서술과 공개 코드가 다른 경우가 있다(Extreme Parkour 의 0.5 vs 0.8/0.4). 코드는 현재 main 브랜치 기준이며 논문 시점 코드와 다를 수 있다.
4. §5 의 증거 중 Isaac Lab issue #458 은 공식 문헌이 아니라 커뮤니티 댓글이다. 수치 계산 표는 mjlab 기본값을 가정한 것이다.
5. MDPI 논문(*Reinforcement Learning-Based Adaptive Motion Control of Humanoid Robots on Multi-Terrain*, Applied Sciences 2026)은 검색 요약에 "정지 상태로 수렴" 사례가 있다고 나왔으나 사이트가 열리지 않아(HTTP 403) **인용하지 않았다.**

---

## 8. 출처 목록

논문 (모두 arXiv PDF 본문 직접 확인)
- Rudin et al. 2021, Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning — https://arxiv.org/abs/2109.11978
- Rudin et al. 2022, Advanced Skills by Learning Locomotion and Local Navigation End-to-End — https://arxiv.org/abs/2209.12827
- Pardo et al. 2018, Time Limits in Reinforcement Learning — https://arxiv.org/abs/1712.00378
- Cheng et al. 2023, Extreme Parkour with Legged Robots — https://arxiv.org/abs/2309.14341
- Zhuang et al. 2023, Robot Parkour Learning — https://arxiv.org/abs/2309.05665
- Hoeller et al. 2023, ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots — https://arxiv.org/abs/2306.14874
- Zhang et al. 2023, Learning Agile Locomotion on Risky Terrains — https://arxiv.org/abs/2311.10484
- Margolis and Agrawal 2022, Walk These Ways — https://arxiv.org/abs/2212.03238
- Miki et al. 2022, Learning robust perceptive locomotion for quadrupedal robots in the wild — https://arxiv.org/abs/2201.08117
- Agarwal et al. 2022, Legged Locomotion in Challenging Terrains using Egocentric Vision — https://arxiv.org/abs/2211.07638
- Kumar et al. 2021, RMA: Rapid Motor Adaptation for Legged Robots — https://arxiv.org/abs/2107.04034
- Radosavovic et al. 2024, Learning Humanoid Locomotion over Challenging Terrain — https://arxiv.org/abs/2410.03654
- Liao et al. 2024, Berkeley Humanoid: A Research Platform for Learning-based Control — https://arxiv.org/abs/2407.21781
- Sferrazza et al. 2024, HumanoidBench — https://arxiv.org/abs/2403.10506
- Gu et al. 2024, Humanoid-Gym — https://arxiv.org/abs/2404.05695
- Booster Gym 2025 — https://arxiv.org/abs/2506.15132
- Chane-Sane et al. 2024, CaT: Constraints as Terminations for Legged Locomotion — https://arxiv.org/abs/2403.18765
- Wang et al. 2025, BeamDojo — https://arxiv.org/abs/2502.10363
- Chen et al. 2026, Light-Loco-Parkour — https://arxiv.org/abs/2608.02653
- Reda et al. 2020, Learning to Locomote: Understanding How Environment Design Matters for Deep Reinforcement Learning — https://arxiv.org/abs/2010.04304
- Dai et al. 2026, Walk the PLANC — https://arxiv.org/abs/2601.06286
- Kostrikov et al. 2019, Discriminator-Actor-Critic — https://arxiv.org/abs/1809.02925
- Peng et al. 2018, DeepMimic — https://arxiv.org/abs/1804.02717
- (원문 미확인) Qian et al. 2024, LEEPS — https://ieeexplore.ieee.org/abstract/document/10801925

코드 (2026-10-09 시점 main/master 직접 확인)
- legged_gym `legged_robot.py` — https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py
- Isaac Lab 커리큘럼 — https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py
- Isaac Lab 보상 — https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
- Isaac Lab 속도 과제 설정 — https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
- Isaac Lab H1 거친 지형 — https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py
- Isaac Lab G1 거친 지형 — https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py
- mjlab 커리큘럼 — https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/mdp/curriculums.py
- mjlab 속도 과제 설정·보상·종료 관리자 — https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/velocity_env_cfg.py , https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/mdp/rewards.py , https://github.com/mujocolab/mjlab/blob/main/src/mjlab/managers/termination_manager.py
- rsl_rl `ppo.py` (time_outs bootstrap) — https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
- unitree_rl_gym G1 — https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
- Extreme Parkour 코드 — https://github.com/chengxuxin/extreme-parkour
- Robot Parkour Learning 코드 — https://github.com/ZiwenZhuang/parkour
- Isaac Lab issue #458 — https://github.com/isaac-sim/IsaacLab/issues/458

이 저장소 내부 참조
- vendored 커리큘럼(옛 규칙): `/home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/src/mjlab/tasks/velocity/mdp/curriculums.py`
