#!/bin/bash
# PreToolUse(Bash) HOOK — force ALL training through scripts/run_training.sh (the protocol launcher:
# OVERVIEW + CLOSE-UP video + formatted report, every run incl. tests). The model kept launching train.py
# directly and skipping video (even via a self-made config-test exemption). This blocks ANY direct
# `python ... scripts/train.py` so the launcher is the ONLY path. The launcher calls train.py as a SUBPROCESS
# (not seen here), so `bash scripts/run_training.sh ...` is allowed. Robust: any parse error -> exit 0.
input=$(cat 2>/dev/null)

# --- 세션별 끄기 (hook_optout.sh) — .claude/hooks_off/<세션번호> 가 있으면 조용히 통과
printf '%s' "$input" | bash "${CLAUDE_PROJECT_DIR:-/home/syaro/MikuchanRemote/Human-Pygmalion}"/pygmalion_locomotion/scripts/hook_optout.sh && exit 0
cmd=$(printf '%s' "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null) || exit 0
printf '%s' "$cmd" | grep -qE 'python[^|&;]*scripts/train\.py' || exit 0
echo "★★ BLOCKED (PreToolUse): a DIRECT 'python scripts/train.py' launch is not allowed. EVERY training (incl. config/tests) MUST go through the launcher so video (OVERVIEW + CLOSE-UP) + report are ALWAYS produced and cannot be skipped:
    bash scripts/run_training.sh <task> <run_name> <max_iter> <num_envs> [extra train args]
(user rule 2026-06-22: 모든 학습은 테스트던 뭐던 영상+노트 양식대로. no exemptions.)" >&2
exit 2
