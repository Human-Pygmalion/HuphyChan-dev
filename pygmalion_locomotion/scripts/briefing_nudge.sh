#!/bin/bash
# briefing_nudge.sh — PostToolUse(Bash): when a task BOUNDARY just happened, remind the model to
# stamp it in the live briefing. Non-blocking by design (always exit 0) - the Stop hook is what
# enforces; this one only makes the update happen at the right moment instead of all at the end,
# which is the whole point of a "real-time" page.
set -uo pipefail
payload=$(cat 2>/dev/null) || exit 0

# --- 세션별 끄기 (hook_optout.sh) — .claude/hooks_off/<세션번호> 가 있으면 조용히 통과
printf '%s' "$payload" | bash "${CLAUDE_PROJECT_DIR:-/home/syaro/MikuchanRemote/Human-Pygmalion}"/pygmalion_locomotion/scripts/hook_optout.sh && exit 0
cmd=$(printf '%s' "$payload" | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('tool_input',{}).get('command','')[:600])
except Exception: print('')
" 2>/dev/null) || exit 0
[ -z "$cmd" ] && exit 0

case "$cmd" in
  *briefing.py*) exit 0 ;;                       # do not nag about the nagging
esac

hit=""
case "$cmd" in
  *train_wandb_video.py*|*run_training.sh*)  hit="학습을 시작했습니다" ;;
  *scripts.evaluate*|*impact_probe_multi*|*measure_loads.py*) hit="측정/평가를 돌렸습니다" ;;
  *render_loads.py*|*rom_sweep_video*|*material_diagram*)     hit="영상/그림 산출물이 나왔습니다" ;;
  *"git commit"*) hit="커밋했습니다" ;;
esac
[ -z "$hit" ] && exit 0

echo "[briefing] $hit — docs/000.Real-time Brefing.md 를 지금 갱신하세요: python3 tools/briefing/briefing.py now|done|block ..."
exit 0
