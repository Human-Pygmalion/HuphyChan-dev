#!/bin/bash
# 이 훅 호출이 **꺼 둔 세션**에서 온 것인지 본다. 꺼져 있으면 exit 0, 아니면 exit 1.
#
# 2026-09-15 · 사용자 지시: *"학습 돌리는 세션은 훅 살려두고, 지금 이 세션만 학습관련 훅
# 모두 없앨수있나? 거슬리고, 맥락이 계속 섞여서 불편해."*
#
# 훅 설정(.claude/settings.json)은 **프로젝트 단위**라 세션마다 다르게 둘 수 없다. 그런데
# 훅은 호출될 때 표준입력으로 JSON 을 받고 그 안에 `session_id` 가 들어 있다. 그래서
# 설정을 건드리지 않고 **세션 번호로** 가를 수 있다.
#
#   끄기:  touch .claude/hooks_off/<세션번호>
#   켜기:  rm     .claude/hooks_off/<세션번호>
#
# 쓰는 쪽(각 훅 맨 위):
#   printf '%s' "$input" | bash "$CLAUDE_PROJECT_DIR/pygmalion_locomotion/scripts/hook_optout.sh" && exit 0
#
# ★ 파일이 남아 있는 한 그 세션에서는 훅이 아무것도 안 합니다. 세션 번호는 세션마다
# 다르므로 **다른 세션(학습 돌리는 쪽)은 그대로 걸립니다.**
set -u
ROOT="${CLAUDE_PROJECT_DIR:-/home/syaro/MikuchanRemote/Human-Pygmalion}"
OFF="$ROOT/.claude/hooks_off"
[ -d "$OFF" ] || exit 1

sid=$(python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('session_id',''))
except Exception: print('')
" 2>/dev/null) || exit 1

[ -n "$sid" ] || exit 1
[ -f "$OFF/$sid" ] || exit 1
exit 0
