# reward_research — reward 변경 전 *원인 연구·분석* 노트 (HARD RULE, user 2026-06-22)

> **규칙**: 학습이 끝난 뒤 reward 항(weight/term)을 바꾸기 *전에*, **반드시** 아래 과정을 거쳐 이 폴더에 노트를 남긴다. 반응적 "그냥 weight 조정" 금지. **PreToolUse 훅(`require_reward_research.sh`)이 강제** — 노트가 *최신 학습 체크포인트보다 newer*가 아니면 reward 파일(flat_env_cfg/velocity_env_cfg/rewards.py) 수정이 차단된다(매 run 후 새 분석 강제, 옛 노트 재사용 불가).
>
> ⚠ **workflow로 연구 시 노트가 중복 생성될 수 있음**(finder 개별 + synthesis). **canonical = synthesis 노트 1개**만 정본으로 본다(아래 인덱스). finder 노트는 superseded.

---

## ★ 연구 계보 인덱스 (gaitfix 시리즈)

| 주제 | canonical 노트 | 1줄 결론 | superseded |
|---|---|---|---|
| **ankle_roll 측방 edge** (v3→v4) | [[2026-06-22_03-50_foot_edge_ankle_roll_gaitfix_v4]] | wide-stance+foot-body-flat → stance 넓혀도 RMS만 −20%, **PEAK 안 줄음** | [[2026-06-22_foot_flat_ankle_roll_stance]] (내 원본 stub, v4 리포트 참조) |
| **ankle_roll PEAK = HW?** (v5) | [[2026-06-22_11-00_ankle_roll_peak_gaitfix_v5_or_hw]] | ★ SPLIT: RMS=artifact / **PEAK=물리바닥**(mg×발반폭). routing(foot-placement+hip)으로 판별 → 실패 시 RS00 under-spec(DM-J4340/2-RSU/발폭) | — |
| **toe 롤오버** (v6) | [[2026-06-22_12-30_toe_rollover_cop_progression_gaitfix_v6]] | 스프링 정상(~60=human MTP); **CoP 전진 부재**가 원인. CoP-progression+push-off 복원 필요(단일 forefoot_cop는 정적이라 실패) | [[2026-06-22_toe_moment_rollover_plantar_surface]] (finder) |
| **골반/base rigidity** (v6) | [[2026-06-22_11-30_base_overconstrain_pelvis_swing_gaitfix_v6]] | ★ **`base_height_l2`(−1.0)가 vault/push-off 억제 = 골반·토우 공통 범인**. 비대칭/완화 | [[2026-06-22_10-35_base_rigid_pelvis_com_oscillation]] (finder) |

★ **공통 발견**: `base_height_l2(−1.0)`가 single-support CoM vault를 페널티 → push-off를 막아 **토우 미사용 + 골반 swing 부족을 동시 유발**. gaitfix_v6에서 base 완화가 두 문제의 공통 해법.
*(HW 분석 노트 — reward 아님 — 은 `docs/NN_*.md`: 36 TN봉투·37 링크·38 병렬·39 QDD survey·40 발edge·41 ankle_pitch push-off)*

---

## 외부 코드 분석

| 주제 | 노트 | 1줄 결론 |
|---|---|---|
| **Isaac Asimov (Menlo Research) 학습 코드 전수 분석** (2026-09-25) | [[2026-09-25_isaac_asimov_teardown]] | 뿌리가 mjlab 으로 같음. ★**서기를 표본 대신 보상(서기 전용 L1 편차)으로 해결** — 우리가 서 있는 비율로 풀다 얻은 게이트 정체·보행 후퇴를 피하는 길. ★**시간 오차 세 겹 무작위화**(구동기 0~25 ms·관측 0~40 ms·이득 ±20 %)는 v7·v8 에 꺼져 있음. 참조 동작은 **자기 정책 녹화** |

---

## 순서 (무조건)
1. **직전 학습 결과 분석** — 지표(error_vel·낙상·reward항 기여) + **영상 audit**(클로즈업) + **§7 모터 활용**(RMS/p95/peak·포화) + 측정. *무엇이 문제인가*.
2. **이전 이력 활용** — `docs/experiments/`·`docs/log.md`·관련 docs 노트에서 *전에 뭘 시도했고 어떻게 됐나* (같은 실수 반복 금지).
3. **학술/자료조사** — 관련 논문·레퍼런스를 **면밀히** 조사(workflow/web). 하이퍼링크 + 검증 원문(`docs/raw/`).
4. **원인·문제 규명** — 위를 종합해 *근본 원인*이 무엇인지 결론.
5. **제안** — 그래서 어떤 reward를 *왜* 어떻게 바꾸는지(또는 reward 아닌 다른 해법인지).
6. 그 다음에만 reward 수정 → 학습.

## 파일명
`docs/reward_research/<YYYY-MM-DD_HH-MM>_<주제>.md` (예: `2026-06-22_03-10_toe_loading.md`)

## 템플릿
```markdown
# reward 연구 — <주제> (<날짜>)
> 트리거: <어느 run 결과>. 바꾸려는 reward: <무엇>.

## 1. 직전 결과 분석
- 지표: error_vel … / 낙상 … / 기여 큰 항 …
- 영상 audit: …  · §7 모터: RMS/p95/peak …  · 측정: …
- → 관측된 문제: …

## 2. 이전 이력
- [[experiments/<run>]] / docs/log: 전에 <무엇> 시도 → <결과>. 교훈: …

## 3. 학술/자료조사 (출처 하이퍼링크)
- <논문/레퍼런스> — <핵심>. [[raw/<slug>]]

## 4. 원인·문제 규명
- 근본 원인: …

## 5. 제안 (reward 변경 + 왜)
- <항> <기존>→<신규> 이유: …  (또는 reward 아닌 해법: …)
```
