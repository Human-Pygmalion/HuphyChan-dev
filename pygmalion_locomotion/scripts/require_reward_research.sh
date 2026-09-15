#!/bin/bash
# PreToolUse(Edit|Write) HOOK — user 2026-06-22 (HARD RULE): BEFORE adjusting any REWARD term/weight, FIRST
# research the cause (training result + past history docs/experiments+log + literature/academic survey),
# determine the ROOT CAUSE / problem, and RECORD the process in a research note. This BLOCKS a reward-file edit
# unless a FRESH reward-research note exists (docs/reward_research/*.md modified in the last 90 min).
# Stops the reactive "just tweak the weight" failure mode. Robust: any parse error -> exit 0 (never wedge).
ROOT=/home/syaro/MikuchanRemote/Human-Pygmalion
input=$(cat 2>/dev/null)

# --- 세션별 끄기 (hook_optout.sh) — .claude/hooks_off/<세션번호> 가 있으면 조용히 통과
printf '%s' "$input" | bash "${CLAUDE_PROJECT_DIR:-/home/syaro/MikuchanRemote/Human-Pygmalion}"/pygmalion_locomotion/scripts/hook_optout.sh && exit 0
fp=$(printf '%s' "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null) || exit 0
[ -z "$fp" ] && exit 0
# only police the REWARD-definition files
case "$fp" in *flat_env_cfg.py|*velocity_env_cfg.py|*/rewards.py) ;; *) exit 0 ;; esac
# only when the edit actually CHANGES a reward weight / term (not a comment / unrelated line)
new=$(printf '%s' "$input" | python3 -c "import sys,json; d=json.load(sys.stdin).get('tool_input',{}); print(str(d.get('new_string',''))+str(d.get('content','')))" 2>/dev/null)
printf '%s' "$new" | grep -qE '\.rewards\.|weight *=|RewTerm|func=pyg_rewards' || exit 0
# require a FRESH reward-research note = the cause analysis recorded BEFORE the reward change
# require a reward_research note NEWER than the LATEST training checkpoint (= you analyzed the LATEST run's
# result). A note written BEFORE the last run no longer counts -> can't reuse a stale note for a new tweak.
LATEST=$(find "$ROOT/pygmalion_locomotion/logs/rsl_rl" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST" ]; then
  [ -n "$(find "$ROOT/docs/reward_research" -name '*.md' ! -name 'README.md' -newer "$LATEST" 2>/dev/null)" ] && exit 0
else
  [ -n "$(find "$ROOT/docs/reward_research" -name '*.md' ! -name 'README.md' -mmin -90 2>/dev/null)" ] && exit 0
fi
echo "★★ BLOCKED (PreToolUse): reward 항(weight/term)을 바꾸기 전에 RESEARCH+분석을 먼저 하고 기록해야 합니다 (user 2026-06-22 HARD RULE). 순서: (1) 직전 학습 결과 + 이전 이력(docs/experiments·log) 분석 (2) 학술/자료조사로 *원인·문제* 규명 (3) docs/reward_research/<YYYY-MM-DD_HH-MM>_<주제>.md 에 [결과분석 · 이력 · 연구/출처 · 원인·문제 · 제안] 기록 (4) 그 다음 reward 수정. (90분 내 reward_research 노트가 없어 차단 — 반응적 튜닝 금지.)" >&2
exit 2
