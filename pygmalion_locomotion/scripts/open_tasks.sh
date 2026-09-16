#!/bin/bash
# 매 프롬프트마다 **아직 안 끝난 일**을 앞에 꽂는다. (UserPromptSubmit 훅)
#
# 2026-09-16. 왜 필요한가: 사용자가 일을 시키고 곧이어 다른 일로 끼어들면, 앞의 일이
# 대화 속에만 남고 아무도 다시 안 꺼냅니다. 실제로 "외력/초기자세 강건화 학습" 이
# 그렇게 사라졌습니다. 이 리포의 훅은 전부 Stop(세션 끝)/PostToolUse 에만 있어서
# **턴 중간에는 아무도 안 깨웁니다.**
#
# ★ 이 훅은 `hooks_off` 를 **따르지 않습니다.** 학습노트 훅이 시끄러워서 끄더라도,
# "사용자가 시킨 일" 목록까지 같이 꺼지면 안 됩니다. 그것이 이 사고의 원인이었습니다.
set -u
ROOT="${CLAUDE_PROJECT_DIR:-/home/syaro/MikuchanRemote/Human-Pygmalion}"
OUT=$(python3 "$ROOT/tools/tasks/task.py" list 2>/dev/null)
[ -z "$OUT" ] && exit 0
case "$OUT" in *"열린 일 없음"*) exit 0 ;; esac
printf '아직 안 끝난 일 (tools/tasks/task.py):\n%s\n' "$OUT"
printf '끝났으면 `task.py done <번호>`, 막혔으면 `task.py block <번호> "이유"`.\n'
printf '사용자가 새 일을 시키면 시작 전에 `task.py add` 로 먼저 적을 것.\n'
exit 0
