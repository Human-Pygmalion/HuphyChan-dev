# 제자리 선회 상실 — 평지에서 험지로 옮길 때 요(yaw) 추종이 사라지는가: 문헌·코드 조사 (2026-10-11)

작성일: 2026-10-11 KST
발단: 평지 전용 정책 v11a 는 제자리 선회(turning in place, 명령 $v_x = v_y = 0,\ \omega_z \neq 0$)에 중앙값 51 % 성공한다. v11a 에서 출발해 험지(rough terrain)에서 3,000 반복(iteration) 더 학습하자 1 %로 무너졌고, 각속도 추종 보상도 학습 중 내려갔다. 전진 속도 과속(명령 1.2 m/s 에 1.43 m/s)과 좌우 비대칭도 나빠졌다.
성격: 문헌·코드 조사 전용. 보상·커리큘럼·학습 코드는 건드리지 않았다. 설계 결정은 계획자 몫이므로 §4 의 "권고"는 근거 정리일 뿐이다.
확인 방법: 코드는 GitHub 원문(raw)을 직접 읽었다. 논문은 arXiv HTML 을 가져와 본문을 읽었다. 숫자·인용 옆의 "(요약)" 표시는 웹 도구가 돌려준 요약문만 근거이고 원문 대조를 못 한 항목이다. "(추론)" 표시는 근거 문헌이 아니라 내(조사자) 해석이다.

용어 한 번씩 풀어 둔다. PPO = Proximal Policy Optimization(근접 정책 최적화). KL = Kullback–Leibler divergence(두 확률분포의 차이). DAgger = Dataset Aggregation(학생이 직접 굴린 상태를 전문가가 정답으로 라벨링해 지도학습). 커리큘럼 = 쉬운 지형부터 어려운 지형으로 올리는 학습 순서 설계.

---

## 1. 요약

- **"평지에서 험지로 옮기면 제자리 선회가 사라진다"를 직접 보고한 논문·이슈는 찾지 못했다.** legged_gym, rsl_rl, Isaac Lab, humanoid-gym, unitree_rl_gym, unitree_rl_lab, mjlab 의 이슈 검색(`gh search issues`)에도 해당 증상은 없었다. 이것은 "없다는 증명"이 아니라 "찾은 범위에서 없다"이다. 다만 원인 후보를 뒷받침하는 코드·이슈·논문은 있다(아래).
- **가장 강한 코드 근거: 표준 지형 커리큘럼은 요(yaw)를 전혀 보지 않는다.** legged_gym · Isaac Lab · mjlab 의 승급/강등 규칙은 모두 "에피소드 끝에 출발점에서 몇 m 떨어졌나"만 쓴다. 순수 요 명령(선속도 0)은 기대 이동거리가 0 이라 승급도 강등도 일으키지 않는다. 선속도+요 혼합 명령은 로봇이 호를 그리며 돌아와 부당하게 강등된다. mjlab 유지관리자 본인이 이슈 #934 에서 이를 인정했다("Yaw is also ignored when computing expected travel").
- **우리 코드 읽기에서 나온 더 직접적인 원인 후보(추론): 학습 분포에 순수 요 명령이 거의 없다.** 우리 명령 표집은 축마다 독립 균등분포이고, 서 있기(standing) 환경은 요까지 포함해 명령을 전부 0 으로 만든다. 그래서 $(0, 0, \omega_z \neq 0)$ 점은 학습 중 "서 있기 $(0,0,0)$ 덩어리"의 바로 곁에 있지만 정작 그 점의 밀도는 거의 0 이다(§3 계산: 서 있기가 순수 요 근방보다 약 8배 많다). 평지 51 %는 이 구역으로의 일반화였을 가능성이 크고, 험지 미세조정 중 "명령 선속도 ≈ 0 이면 가만히 있기"가 강화되면 쉽게 무너질 수 있다. 이 가설은 문헌이 아니라 코드와 우리 설정에서 나온 것이므로 시험이 필요하다.
- **파국적 망각(catastrophic forgetting)은 미세조정에서 실제로 보고되고, 대응은 숫자까지 나와 있다.** 거친 지형을 마지막에 도입한 Solo12(PPO 클립 0.2 → 0.05, 학습률 5e-3 → 1e-3), 인간형 지형 미세조정(행위자 학습률 1e-5, 초기 표준편차 0.135, 평지 12.5 % 유지), 굴착 로봇(동결 정책에 대한 KL 벌점 $\beta = 0.02$), 다전문가 증류(Parkour in the Wild)가 그것이다. 단, **이들 중 어느 것도 "제자리 선회"를 따로 재서 망각을 보인 것은 아니다.** 우리 문제에 옮기는 것은 외삽이다.
- **~1,000 반복 단일 변인 시험으로 깨끗한 것: (1) 순수 요 명령 비율 추가, (2) 보수적 미세조정(학습률·클립 축소), (3) 평지 타일 비율 고정.** 커리큘럼 승급 규칙 수정과 증류는 효과가 나타나는 데 1,000 반복보다 오래 걸리거나 구현이 크다. 가장 싼 선행 단계는 학습이 아니라 진단이다(§4 단계 0).

---

## 2. 원문 발췌

### 2-1. 지형 커리큘럼 코드 (승급 규칙이 무엇을 보는가)

**legged_gym** — `legged_robot.py`, `_update_terrain_curriculum`, `_resample_commands`
링크: https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py (원문 raw 확인)
- "robots that walked far enough progress to harder terains" (코드 주석)
- "robots that walked less than half of their required distance go to simpler terrains" (코드 주석)
- 코드 본문: `move_down = (distance < norm(commands[:, :2]) * max_episode_length_s * 0.5) * ~move_up`. **요 명령 `commands[:, 2]` 는 이 식에 들어가지 않는다.**
- 같은 파일에서 명령 표집은 선속도 크기 0.2 이하를 0 으로 만든다(`commands[:, :2] *= (norm > 0.2)`). 요는 그대로 두므로 legged_gym 에서는 선속도 0 + 요 ≠ 0 인 명령이 **자동으로 생긴다**(기본 상자 $\pm 1 \times \pm 1$ 에서 약 3.1 %, §3).
- 보상 설정(`legged_robot_config.py`): `tracking_lin_vel = 1.0`, `tracking_ang_vel = 0.5`, `tracking_sigma = 0.25`($\exp(-e^2/\sigma)$). 헤딩 명령 기본 켜짐.
- 우리에게 주는 의미: 승급 규칙이 요를 무시하는 것은 표준 설계다. 단, legged_gym 은 0.2 m/s 이하 선속도 불감대 덕에 순수 요 표본이 일정량 생기는데, mjlab 에는 이 불감대가 없다(§2-2 와 §3).

**Isaac Lab** — `terrain_levels_vel`
링크: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py (원문 raw 확인)
- "based on the distance the robot walked when commanded to move" (독스트링 일부)
- 본문은 legged_gym 과 동일하게 `command[env_ids, :2]` 의 크기만 쓴다. 요는 없다.
- 명령 설정(`velocity_env_cfg.py`): `rel_standing_envs = 0.02`, `rel_heading_envs = 1.0`, `heading_command = True`, `ang_vel_z = (-1, 1)`. 보상 `track_lin_vel_xy_exp` 1.0, `track_ang_vel_z_exp` 0.5, 둘 다 std $=\sqrt{0.25}$.
- 서 있기 환경 처리(`velocity_command.py`, 요약): 서 있기로 뽑힌 환경은 **명령 전체를 0 으로 덮어쓴다**(요 포함).
- 우리에게 주는 의미: Isaac Lab 의 서 있기 비율은 2 %, 우리 v11a 는 10 %. 서 있기 환경은 "선속도 0 → 요도 0" 상관을 가르친다(§3).

**mjlab (현재 main)** — `velocity_env_cfg.py`, `mdp/curriculums.py`
링크: https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/ (원문 raw 확인, 일부는 "요약")
- 기본 명령: `rel_standing_envs = 0.1`, `rel_heading_envs = 0.3`, `ang_vel_z = (-0.5, 0.5)`. 보상 `track_linear_velocity` 2.0 (std 0.5), `track_angular_velocity` 2.0 (std $\sqrt{0.5} \approx 0.71$).
- main 의 `terrain_levels_vel` 독스트링(요약): "Envs commanded less than min_commanded_distance meters are never promoted". 즉 상위 저장소는 이미 "표류(drift)로 승급" 문제를 고쳤다. 우리가 쓰는 mjlab 사본(`mujoco-sim/mjlab`, 커밋 27d16f82)은 **고치기 전 Isaac Lab 이식본**이다(로컬 `curriculums.py` 48~56행 확인).
- 우리에게 주는 의미: 상위 수정은 "순수 요 명령 환경이 승급하지 않게" 하지만, 순수 요 명령 환경에 *요 추종 기준*의 승강급을 주지는 않는다(요약 기준).

**mjlab 이슈 #934** — "Fairness of `terrain_levels_vel` curriculum progression function for low speed locomotion task"
링크: https://github.com/mujocolab/mjlab/issues/934 (유지관리자 kevinzakka 의 답변 원문 확인)
- "Yaw is also ignored when computing expected travel"
- "mixed yaw and forward commands cause unfair demotions"
- 같은 답변: 로봇이 호를 그리며 출발점 쪽으로 돌아오는데 임계값은 순수 전진 이동을 기대한다. 또 에피소드 끝의 *마지막* 명령만 읽으므로 명령이 바뀐 에피소드는 부당하게 판정된다.
- 대응 PR #1072(닫힘, 병합 안 됨): 기대 이동거리에 요 명령에 따른 현(chord) 보정 $\sin(\theta/2)/(\theta/2)$ 을 곱하고 승급 비율을 0.8 로 한다. 링크: https://github.com/mujocolab/mjlab/pull/1072
- 우리에게 주는 의미: 커리큘럼이 요 명령을 가진 환경을 **불공정하게 강등**한다는 것은 유지관리자가 확인한 사실이다. 다만 이것이 *요 추종이 사라지는* 직접 원인이라는 보고는 아니다. 요가 섞인 환경이 낮은 지형에 머물면 오히려 평지 요 연습이 늘어나므로, 방향은 단정할 수 없다(추론).

### 2-2. 명령 분포·커리큘럼에 요를 넣는 문헌

**Margolis & Agrawal, "Walk These Ways" (arXiv 2212.03238)** — 부록 표 7
링크: https://arxiv.org/abs/2212.03238 (HTML 본문 확인; 격자 적응 커리큘럼은 Rapid Locomotion 에서 가져왔다고 명시)
- 요 각속도 $\omega_z$ 초기 범위 [-1, 1], 최대 [-5, 5], 0.5 간격. 범위는 **에피소드 보상이 문턱을 넘을 때만** 넓어진다: $v_x, v_y$ 추종 0.8, **$\omega_z$ 추종 0.7**.
- 지형 커리큘럼은 없고 평지에서만 학습한다.
- 우리에게 주는 의미: 요 명령 난이도를 *요 추종 성적*으로 게이트하는 선례. 지형 커리큘럼 승급에 요 추종을 넣는 직접 선례는 아니다.

**Angarola 외, "Learning Terrain-Specialized Policies…" (arXiv 2509.20635)**
링크: https://arxiv.org/html/2509.20635v2 (HTML 본문 확인)
- 격자 커리큘럼의 한 칸이 $(v_x, \omega_z)$ 쌍이다. 범위 [-3, 3] 을 625 칸으로 나누고, 추종 오차가 문턱 아래이면 이웃 칸을 연다. **오차에 $\omega_z$ 가 들어간다.** 실험 문턱은 선속도 0.15, 각속도 0.25.
- 우리에게 주는 의미: "승급 기준에 각속도 추종 포함"의 직접 선례(명령 범위에 대해서이지 지형 레벨에 대해서는 아니다). 망각 측정은 없다.

**Li, Li, Hutter, "Scaling Rough Terrain Locomotion with Automatic Curriculum RL" (arXiv 2601.17428)**
링크: https://arxiv.org/html/2601.17428v1 (HTML 본문 확인)
- 요 각속도 최대 3.0 rad/s 를 6단계로 나누고 선속도 단계·지형 종류·난이도와 곱해 **600개 과제**를 만든다. 과제 선택은 보상 변화량(learning progress)의 softmax. 성공 판정은 선·각속도 추종 오차(EPTE-SP) 모두 30 % 미만.
- 우리에게 주는 의미: 요를 지형과 *같은 과제 격자*에 넣어 지형별로 요 성적을 따로 관리하는 방식. 구현이 크다.

**SSR: Scaling Surefooted and Symmetric Humanoid Traversal to the Open World (arXiv 2605.30770)**
링크: https://arxiv.org/html/2605.30770 (HTML 본문 확인)
- 표 4, 지형별 명령 범위: 경사 전진 $U(-1, 1)$ m/s·헤딩 $U(-80°, 80°)$, **계단 전진 $U(0, 1.2)$·헤딩 $U(-30°, 30°)$**, 틈·단 전진 $U(0, 1.2)$·헤딩 $U(-15°, 15°)$.
- 표 5: 가장 어려운 지형을 벗어난 뒤에야 전방위 명령(전진 $U(-1, 1)$, 측면 $U(-0.4, 0.4)$, 요 각속도 $U(-1.2, 1.2)$ rad/s)을 따로 표집한다.
- "The yaw angular velocity command is computed from the heading error."
- 우리에게 주는 의미: **계단·장애물 지형에서는 선회를 일부러 제한하고, 전방위 명령은 별도 단계에서 학습한다**는 지형별 명령 분포의 선례. "계단에서 회전은 위험하니 정책이 버린다"는 우리 가설과 정반대 방향의 설계(처음부터 요구하지 않음)다.

### 2-3. 미세조정에서의 망각과 대응

**Radosavovic 외(인간형 지형 미세조정), "Learning Humanoid Locomotion over Challenging Terrain" (arXiv 2410.03654)**
링크: https://arxiv.org/html/2410.03654 (HTML 본문 확인)
- "smaller than those typically used for RL from scratch" (초기 공분산).
- 설정: 행위자 학습률 1e-5(100 반복 선형 워밍업 1e-8 → 1e-5, 이후 코사인), 비평자 5e-4 고정, 초기 표준편차 0.135, 2,000 반복, 클립 0.2, 엔트로피 0. 지형 구성은 평지 12.5 %·거친 지면 12.5 %·완경사 12.5 %·거친 경사 12.5 %·장애물 25 %·언덕 25 %. 명령: 전진 [-1, 1], 측면 [-0.5, 0.5], 회전 [-1, 1] rad/s, **명령의 각 차원을 확률 0.5 로 독립적으로 0 으로 만든다**(10 s 마다 재표집).
- 이 독립 0화는 $P(v_x{=}0,\ v_y{=}0,\ \omega_z{\neq}0) = 0.5^3 = 0.125$ 즉 **순수 요 명령이 12.5 %** 라는 뜻이다(추론; 논문이 이를 목적으로 쓴다고 하지는 않는다).
- 평지를 넣은 이유: 평지 조건은 사전학습 정책이 "retains a good flat walking gait" 하도록 하려는 것.
- 우리에게 주는 의미: 위 세 가지(작은 학습률·작은 초기 소음·평지 비율)와 순수 요 12.5 % 가 한 레시피에 들어 있다. 망각을 직접 재지는 않았다.

**"Controlling the Solo12 Quadruped Robot with Deep Reinforcement Learning" (arXiv 2309.16683)**
링크: https://arxiv.org/html/2309.16683v1 (HTML 본문 확인; 표 숫자는 요약)
- 거친 지형(무작위 높이지도)을 **마지막(1,000번째) 반복에서 도입**하고 병렬 환경의 80 %가 비평지에서 시작한다. 이때 PPO 를 "more conservative updates" 로 바꾼다고 서술하고, 목적이 평지 보행 망각 방지라고 쓴다.
- 표 2(요약): 클립 비율 0.200 → 0.050, 기울기 노름 클립 0.500 → 0.300, 엔트로피 계수 0.010 → 0.000, 학습률 0.005 → 0.001.
- 우리에게 주는 의미: 가장 가까운 "평지 → 험지" 단일 변경 사례이며 보수적 PPO 설정이 숫자로 나와 있다. 20 %가 평지로 남는다는 뜻의 구성. 보행 속도 추종이 유지됐다는 보고지 선회를 따로 잰 것은 아니다.

**Parkour in the Wild (arXiv 2505.11164)** — 다전문가 증류 + 강화학습 미세조정
링크: https://arxiv.org/html/2505.11164 (HTML 본문 확인)
- "the performance of the policy steadily degrades once the RL loss is applied"
- "Keeping a diversity of training terrains seems important"
- 안정화: 증류 때 행동 잡음, 줄인 초기 정책 표준편차, 보수적 RL 하이퍼파라미터, 정책을 얼린 채 비평자 사전학습. 반복 미세조정에서 새 지형은 수집 표본의 약 3 %. 이전 지형 성능이 크게 변하지 않았다고 보고.
- 증류는 DAgger 방식(학생이 굴린 상태를 전문가가 라벨, 제곱오차 회귀).
- 우리에게 주는 의미: "RL 만 켜면 성능이 계속 떨어진다"는 일반 관찰과, 옛 지형 섞기·보수적 설정·증류라는 세 가지 해법. 증류는 "평지 선회 전문가 + 험지 전문가 → 하나" 로 옮길 수 있으나 구현이 크다.

**굴착기 사전학습 미세조정 (arXiv 2509.14992)**
링크: https://arxiv.org/html/2509.14992 (HTML 본문 확인, 로봇 보행이 아님)
- 목적함수에 $-\beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_0)$, $\beta = 0.02$($\pi_0$ 는 얼린 사전학습 정책). 행위자 학습률 1e-5 → 1e-7 코사인, 비평자 1e-4, 비평자는 사전학습 정책의 롤아웃으로 100 스텝 따로 예열.
- 우리에게 주는 의미: 기법은 이식 가능하나 보행 근거는 아니다. 또 얼린 정책에 묶으면 험지에서 새로 배울 것까지 막는다는 긴장이 있다.

**Pre-training and Preservative Fine-tuning (PPF, arXiv 2504.09833)**
링크: https://arxiv.org/html/2504.09833v2 (HTML 본문 확인)
- "catastrophic forgetting during the fine-tuning phase" 를 막으려 PPO 손실에 전문가(모델 기반 제어기) 행동과의 제곱오차를 더하되, 모델 가정이 성립하는 상태에서만 가중치 $w(s) = w_0 \exp(-\dot z^2/\delta)$ ($w_0 = 5$, $\delta = 0.0159$)를 준다.
- 우리에게 주는 의미: "옛 능력을 지키는 항을 *해당 상태에서만*" 거는 방식. 우리라면 "평지 타일에서만 v11a 행동에 묶기" 로 옮길 수 있다(추론).

### 2-4. 학습률 일정 (rsl_rl)

**rsl_rl `ppo.py` 적응형 학습률**
링크: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py (원문 raw 확인)
- KL 이 `desired_kl` 의 2 배보다 크면 학습률 ÷1.5(하한 1e-5), 절반보다 작으면 ×1.5(상한 1e-2).
- 이 KL 은 *직전 롤아웃 정책*과의 KL 이지 고정 기준 정책과의 KL 이 아니다. 즉 rsl_rl 에는 "v11a 에 묶는" 기능이 없다.
- 우리에게 주는 의미: 웜스타트 후 학습률이 최대 1e-2 까지 올라갈 수 있다. 미세조정 문헌의 1e-5 와 두 자릿수 이상 차이가 날 수 있으므로 3,000 반복 동안 실제 학습률 기록을 확인해야 한다(§4 단계 0).

### 2-5. 원문 대조를 못 한 항목

- "Physical Imitation Learning" (arXiv 2604.00611): 검색 요약은 "승급이 거리만 쓰고 요 [-1, 1] rad/s 를 같이 표집"이라고 한다. 원문 초록만 확인했고 본문은 못 봤다. 인용하지 않는다.
- Light-Loco-Parkour (arXiv 2608.02653) 의 다중 기술 증류: 검색 요약만.
- Rapid Locomotion (arXiv 2205.02824): PDF 가 너무 커서 본문을 못 열었다. Walk These Ways 가 인용하는 격자 커리큘럼의 출처로만 쓴다.

---

## 3. 우리 설정에 대한 계산 (조사자 계산, 추론)

우리 `velocity_command.py` 의 `_resample_command` 는 세 축을 각자 균등 표집한다(`r.uniform_(*ranges.lin_vel_x)` 등). 서 있기 환경(10 %)은 `_update_command` 에서 명령 전체를 0 으로 덮어쓰고, 전진 전용 환경(`rel_forward_envs`)은 요를 0 으로 만든다. 선속도 불감대는 없다(기본값 상태. 환경변수 `PYG_CMD_NORM_CAP` 는 크기 상한이며 불감대가 아니다).

명령 상자 $v_x \in [-2, 2.5]$ (폭 4.5), $v_y \in [-1, 1]$ (폭 2) 에서 선속도 크기가 0.2 m/s 미만일 확률은 (원 면적 / 상자 면적)

$$P(\lVert v_{xy} \rVert < 0.2) \approx \frac{\pi \cdot 0.2^2}{4.5 \times 2} \approx 0.014$$

이다. 서 있기 10 %는 정확히 $(0,0,0)$ 이므로, 서 있기 질량 대 선속도가 거의 0 인 이동 명령 질량의 비는 $0.10 / (0.9 \times 0.014) \approx 8$ 즉 **약 8배**이다(전진 전용 환경 비율은 무시). 선속도 크기가 0.05 m/s 미만으로 엄격히 잡으면 더 줄어 $\pi \cdot 0.05^2 / 9 \approx 8.7 \times 10^{-4}$ 이고 비율은 100배를 넘는다.

비교: 인간형 지형 미세조정 논문의 독립 0화는 순수 요 12.5 %, legged_gym 의 0.2 불감대는 기본 상자($2 \times 2$)의 $\pi \cdot 0.04 / 4 \approx 3.1\,\%$ 를 선속도 0 으로 만들어 순수 요 명령으로 바꾼다. 두 경우 모두 "선속도 0 + 요 ≠ 0" 이 학습 분포에 명시적으로 존재한다. 우리는 서 있기가 그 자리를 요 = 0 으로 차지한다.

주의: 위는 명령 분포만의 계산이다. 평지 v11a 가 51 % 성공한 것은 연속성 덕에 근방 일반화가 어느 정도 되었음을 뜻한다. 험지 학습에서 그 일반화가 무너졌다는 것이 증상이다. 이 계산만으로 원인을 확정할 수 없다.

---

## 4. 후보 대책 표

난이도는 우리 코드 기준 구현 추정(낮음 = 설정값/환경변수 하나, 중간 = 명령·커리큘럼 코드 수정, 높음 = 새 학습 파이프라인). "단일 변인 시험"은 v11a 에서 출발하는 1,000 반복 험지 미세조정을 같은 시드·같은 설정의 기준(baseline) 1,000 반복과 비교하는 형태를 뜻한다.

| 대책 | 근거 | 우리 구현 난이도 | 단일 변인 시험 가능? |
|---|---|---|---|
| 순수 요 명령 비율 추가 (예: 선속도 $=0$, 요 $\sim U(-1, 1)$ 인 환경을 10~15 % 별도 지정; 서 있기와 구분) | 인간형 지형 미세조정 논문의 독립 0화가 순수 요 12.5 %를 만든다(§2-3). legged_gym 의 0.2 불감대도 같은 효과(§2-1). 우리 분포의 밀도 부족은 §3 계산(추론) | 중간 (`_resample_command` 에 분기 한 단락. 환경변수로 끌 수 있음) | 가능. 변인은 "순수 요 명령 비율" 하나. 승급 규칙이 순수 요를 중립으로 처리하므로 지형 분포 교란이 작다 |
| 보수적 미세조정: 학습률 상한 낮춤(적응형 상한 1e-2 → 1e-3 이하 또는 고정), 클립 0.2 → 0.05, 초기 표준편차 축소 | Solo12 표 2 (클립 0.2 → 0.05, 학습률 5e-3 → 1e-3); 인간형 지형 미세조정(행위자 1e-5, 초기 표준편차 0.135); Parkour in the Wild 가 "RL 손실을 켜면 성능이 계속 떨어진다" 보고 | 낮음 (설정값) | 가능. 단 "학습률/클립/초기 표준편차" 를 한꺼번에 바꾸면 변인이 셋이다. 한 번에 하나씩 하거나 "보수적 묶음"을 한 변인으로 선언해야 한다. 험지 적응이 느려지는 부작용 확인 필요 |
| 평지 타일 비율을 커리큘럼과 무관하게 고정 (예: 20~25 % 환경은 항상 평지 행) | Solo12: 80 % 비평지, 즉 20 % 평지 유지; 인간형 지형 논문: 평지 12.5 % 유지가 목적("retains a good flat walking gait"); Parkour in the Wild: 지형 다양성이 중요 | 중간 (지형 생성기 비율 / 일부 환경의 레벨 고정) | 가능. 변인은 "평지 환경 비율". 다만 험지 성능 저하와 맞바꾸는 면이 있고 레벨 분포가 바뀌므로 `terrain_levels` 로그로 교란 확인 |
| 지형별 명령 분포: 계단·장애물 타일은 요/헤딩 제한, 평지·완경사에서 전방위 요 허용 | SSR 표 4 (계단 ±30°, 경사 ±80°) 와 표 5 (전방위는 별도 단계) | 중간~높음 (타일 종류별 명령 범위; 우리는 지형 종류를 환경별로 알아야 함) | 가능하나 변인이 둘(계단 요 제한 + 평지 요 허용)이 되기 쉽다. 순수 요 비율 시험과 병행하지 말 것 |
| 승급 기준 수정: 기대 이동거리에 요 보정(현 보정), 최소 명령 거리 문턱, 또는 요 추종 오차 포함 | mjlab #934, PR #1072(미병합), mjlab main 의 `min_commanded_distance`(요약); 격자 커리큘럼에서 각속도 오차 포함(2509.20635, 문턱 0.25), Walk These Ways (각속도 문턱 0.7) | 중간 (`terrain_levels_vel` 수정, 상위 변경 이식) | 가능하나 1,000 반복으로는 레벨 분포 변화가 늦다. 승강급 변경은 지형 노출 자체를 바꿔 교란이 큼. 장기 시험용 |
| 각속도 추종 보상 가중치·σ 변경 | 가중치 비: legged_gym 1.0 : 0.5, Isaac Lab 1.0 : 0.5 (std 0.5 둘 다), mjlab 2.0 : 2.0 (선속도 std 0.5, 각속도 std 0.71) | 낮음 (설정값) | 가능. 그러나 평지 v11a 가 같은 보상으로 51 % 성공했으므로 원인으로서 사전확률은 낮다. 먼저 §5 단계 0 의 가중치 확인 (아래 주의 참조) |
| 헤딩 명령 비율 변경 (현재 30 %) | Isaac Lab 기본 1.0, mjlab 기본 0.3, SSR 은 지형 학습 전체를 헤딩 오차 기반으로 요를 만든다 | 낮음 (설정값) | 가능. 헤딩 환경은 요 명령을 *선속도와 독립* 이 아니라 *오차에서 생성*하므로 큰 초기 오차에서 포화된 요(최대값 고정)가 길게 나온다. 사전확률은 낮음 |
| 동결 정책에 대한 KL 벌점 / 평지 상태에서만 v11a 행동 모방 항 | 굴착기 논문 ($\beta = 0.02$), PPF (모델 가정이 성립하는 상태에서만 모방) | 높음 (rsl_rl 에 기준 정책 KL 이 없음; 손실 항과 동결 사본 필요) | 가능. 변인은 "KL/모방 항 유무". 구현량이 커서 1,000 반복 시험의 가성비는 낮다 |
| 다전문가 증류 (평지 선회 전문가 + 험지 전문가 → 하나) | Parkour in the Wild (DAgger 증류 + 보수적 RL), Light-Loco-Parkour (요약) | 높음 (파이프라인 신설) | 아니오 (여러 단계, 1,000 반복 안에 끝나지 않음) |
| 요 명령 범위의 성적 게이트(요 추종 > 0.7 이면 범위 확대) | Walk These Ways 표 7, 2509.20635 | 중간~높음 | 아니오 (커리큘럼 동역학이 1,000 반복보다 느림) |

---

## 5. 권고 순서 (근거 정리, 결정은 계획자)

**단계 0 — 학습 없이 먼저 확인 (몇 분~한 시간).** 어떤 대책을 고르든 원인 후보를 좁혀 준다.
1. 저장된 중간 체크포인트(있으면 500 반복 간격)를 **평지 제자리 선회**로 평가해 51 % → 1 % 가 *언제* 시작되는지 본다. 첫 500~1,000 반복 안에 이미 무너졌다면 1,000 반복 시험으로 충분히 관찰된다. 서서히 무너졌다면 시험 길이를 늘려야 한다.
2. 험지 3,000 반복 동안의 **실제 학습률** 기록(적응형 일정이 1e-2 까지 올렸는지)과 정책 표준편차 기록을 확인한다. 문헌 미세조정은 1e-5 대다(§2-3, §2-4).
3. **험지 실행의 각속도 추종 가중치가 평지와 같은지** 확인한다. 우리 `velocity_env_cfg.py` 기본값은 `track_ang_vel_z_exp.weight = 2.0`, `flat_env_cfg.py` 는 1.0 으로 덮어쓴다(소스 확인). "보상 동일"이라고 알려졌더라도 v11a(평지)와 험지 실행의 실제 값이 같은지 설정 파일로 대조한다.
4. 학습 분포 안의 순수 요 질량을 *실측*한다(표본 수천 개로 $\lVert v_{xy} \rVert < 0.2$ 이면서 $|\omega_z| > 0.3$ 인 비율, 서 있기 비율 대비). §3 계산의 검증이다.
5. 평가용 지형별(레벨별) 요 추종 오차를 로그에서 분리해, 요 추종이 어느 지형 종류에서 먼저 떨어지는지 본다. 계단에서만 떨어지고 평지 행에서는 유지된다면 "계단에서의 선회 포기" 쪽이고, 평지 행에서도 같이 떨어진다면 "망각/분포" 쪽이다.

**단계 1 — 1,000 반복 단일 변인 시험 두 팔(arm).** 같은 시작점(v11a), 같은 시드, 같은 설정에서 각각 한 가지만 바꾼다.
- 팔 1: **순수 요 명령 비율 추가** (10~15 %, 서 있기와 별도). 가장 사전확률이 높고 구현이 한 단락이다. 성공 판정은 평지 제자리 선회 성공률(중앙값)과 험지 지형별 요 추종 보상이다. 우리 기존 A/B 노트(`docs/experiments/2026-10-10_ab_rough_stagnation_compare.md`)의 기준 팔과 비교할 수 있다.
- 팔 2: **보수적 미세조정** (학습률 상한 낮춤 + 클립 축소를 한 묶음 변인으로 선언). 설정값만 바꾸므로 구현이 없다.
- 두 팔의 해석: 팔 2 만 선회를 지키면 "망각", 팔 1 만 지키면 "분포 부족", 둘 다 지키면 두 요인이 겹침, 둘 다 못 지키면 계단 위험 회피 같은 다른 원인(단계 0 의 지형별 분해로 구분)이다.

**단계 2 — 단계 1 에서 부분적으로만 회복될 때.** 평지 타일 비율 고정(20~25 %)을 같은 형식의 단일 변인으로 추가한다.

**단계 3 — 장기 과제(1,000 반복 시험 아님).** 승급 규칙 수정(요 보정·최소 명령 거리·요 추종 포함), 계단 구간 요 제한 + 평지 전방위 요(지형별 명령 분포), 증류. 승급 규칙은 상위 mjlab main 의 수정을 이식하는 것이 출발점이다.

---

## 6. 주의 (최대 5개)

1. **직접 보고가 없다.** "평지에서 험지로 가면 제자리 선회가 사라진다"는 현상을 명시한 논문·이슈를 찾지 못했다. 위 원인은 (a) 코드 읽기, (b) mjlab 유지관리자 확인(#934), (c) 우리 분포 계산(추론)의 합성이다. 대책의 효과는 모두 우리 시험으로만 검증된다.
2. **미세조정 문헌은 선회를 재지 않았다.** Solo12·인간형 지형·Parkour in the Wild 는 평지 보행 또는 이전 지형 성능 유지를 보고하지, 요 추종 성공률을 따로 재지 않았다. 일반 원리의 외삽이다.
3. **일부 수치는 요약 기준이다.** Solo12 표 2 숫자, mjlab main 의 `min_commanded_distance` 설명, Light-Loco-Parkour·PIL 항목은 웹 도구 요약이다. 시험 설계에 쓰기 전에 원문 표를 한 번 대조할 것.
4. **명령 상자 가정.** 사용자 설명(vx −2~2.5, vy ±1, wz ±1)을 그대로 썼다. 코드의 PLAY 구성은 `ang_vel_z = (-1.57, 1.57)` 도 있으므로 §3 계산 전에 학습 구성의 실제 상자를 확인할 것. 요 폭은 계산(선속도 크기 기준)에 영향이 없으나 "요 > 0.3" 조건의 비율에는 영향이 있다.
5. **전진 과속·좌우 비대칭은 이 조사 범위 밖이다.** 같은 미세조정 망각의 징후일 수 있으나(추론) 이를 다룬 근거를 따로 찾지 않았다. 팔 2 결과에서 함께 보이면 같은 원인의 증거가 된다.

---

## 7. 우리 학습 기록 대조 (2026-10-11, 계획자 확인 — 학습 없이)

권고 순서 단계 0 의 항목을 학습 기록(텐서보드)과 설정 파일로 확인했다.

| 확인 항목 | 결과 |
|---|---|
| 각속도 추종 가중치·폭 | 평지 v11a 와 험지 A·B 모두 가중치 2.0, 폭 0.707 — **같음**(설정 파일 대조) |
| 명령 상자 | 험지 A·B 학습 내내 단계 4(전진 −2~2.5, 좌우 ±1, 회전 ±1) — 평지와 **같음** |
| 학습률 | 험지 0.00005 → 0.0004 (평지 2단계 0.00018 → 0.00009). 적응형이라 2~4 배 높음 |
| **행동 잡음 표준편차** | **평지 1단계 최대 0.28, 2단계 끝 0.13 → 험지 A·B 0.13 → 0.60** (300 회 0.40, 750 회 0.57) |
| 엔트로피 계수 | 평지 2단계는 0.01 → 0.002 로 줄였다(반복 15,999~29,998). 험지는 반복 번호가 0 부터라 이 창이 시작조차 안 되어 **0.01 고정** |
| 회전 추종 몫 (회전 보상 / 전진 보상) | 평지 끝 ≈ 1.0. 험지: 100 회 0.68, 200 회 0.67, 400 회 0.21, 800 회 0.08, 끝 0.07 |

해석: 잡음이 0.23 → 0.57 로 커지는 구간(200~800 회)과 회전 추종이 무너지는 구간이 겹친다. 관절 목표에 표준편차 0.6 rad 의 잡음이 섞이면 몸통 회전 속도가 잡음에 묻혀 회전 보상을 얻기 어렵고, 정책이 회전 시도를 포기하는 쪽으로 간다고 본다(가설).
잡음이 커진 이유는 ① 엔트로피 계수가 부모의 끝값(0.002)이 아니라 0.01 로 돌아간 것, ② 새로 시작한 가치 함수(행위자만 이어받음)의 부정확한 이점 추정, 둘 다일 수 있다. 문헌의 "보수적 미세조정"(엔트로피 0 또는 축소, §2-3)과 같은 방향이다.

→ **단일 변인 시험 발사 (2026-10-11 13:48)**: 험지 B 와 모든 설정이 같고 엔트로피 계수만 0.002 고정 (`analysis/launch_rough_turn_ent.py`, 실행 `2026-10-11_13-48-08_rough_turn_ent002_v11a`, 1,000 회). 판정: 같은 반복(1,000)의 험지 B 저장본과 제자리 회전 중앙값 비교, 잡음 표준편차 곡선. 성공 기준(계획 162 단계 4): 회전 중앙값 ≥ 50 %.
순수 회전 명령 비율(§4 의 1 순위 후보)은 이 결과를 본 뒤 두 번째 변인으로 둔다.

## 출처 요약

- legged_gym 원문: https://github.com/leggedrobotics/legged_gym (`legged_robot.py`, `legged_robot_config.py`)
- Isaac Lab 원문: https://github.com/isaac-sim/IsaacLab (`mdp/curriculums.py`, `velocity_env_cfg.py`, `velocity_command.py`)
- mjlab: https://github.com/mujocolab/mjlab, 이슈 #934 https://github.com/mujocolab/mjlab/issues/934, PR #1072 https://github.com/mujocolab/mjlab/pull/1072
- rsl_rl: https://github.com/leggedrobotics/rsl_rl (`algorithms/ppo.py`)
- Margolis & Agrawal, Walk These Ways, https://arxiv.org/abs/2212.03238
- Angarola 외, https://arxiv.org/abs/2509.20635
- Li, Li, Hutter, https://arxiv.org/abs/2601.17428
- SSR, https://arxiv.org/abs/2605.30770
- Radosavovic 외, https://arxiv.org/abs/2410.03654
- Solo12, https://arxiv.org/abs/2309.16683
- Parkour in the Wild, https://arxiv.org/abs/2505.11164
- 굴착기 미세조정, https://arxiv.org/abs/2509.14992
- PPF, https://arxiv.org/abs/2504.09833
- 우리 코드: `/home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/src/mjlab/tasks/velocity/mdp/velocity_command.py`, `mdp/curriculums.py`; `/home/syaro/MikuchanRemote/Human-Pygmalion/pygmalion_locomotion/source/pygmalion_locomotion/tasks/locomotion/velocity_env_cfg.py`, `flat_env_cfg.py`
