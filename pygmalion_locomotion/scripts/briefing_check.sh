#!/bin/bash
# briefing_check.sh — Stop-HOOK: BLOCK (exit 2) if docs/000.Real-time Brefing.md has fallen behind
# what actually happened. Same reasoning as audit_notes.sh: the rule kept depending on the model
# remembering, so the HARNESS enforces it. stderr is fed back to the model as the reason to keep going.
#
# Stale means any of:
#   (1) a training run started or finished since the briefing state was last touched
#   (2) an experiment note was written/changed since then
#   (3) a git commit landed since then
#   (4) "지금 하고 있는 일" points at a training run that is no longer running
#
# Robust by construction: any internal error -> exit 0. A status page must never wedge a session.
__HOOK_IN=$(cat 2>/dev/null)
# --- 세션별 끄기 (hook_optout.sh) — .claude/hooks_off/<세션번호> 가 있으면 조용히 통과
printf '%s' "$__HOOK_IN" | bash "${CLAUDE_PROJECT_DIR:-/home/syaro/MikuchanRemote/Human-Pygmalion}"/pygmalion_locomotion/scripts/hook_optout.sh && exit 0

set -uo pipefail
R="${CLAUDE_PROJECT_DIR:-/home/syaro/MikuchanRemote/Human-Pygmalion}"
STATE="$R/docs/.briefing_state.json"
PAGE="$R/docs/000.Real-time Brefing.md"
LOGS="$R/mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity"
SLACK_MIN=20        # a briefing updated within N minutes of the activity counts as current

msg=""

if [ ! -f "$STATE" ]; then
  echo "★ 실시간 브리핑이 아직 없습니다 — python3 tools/briefing/briefing.py now \"<지금 하는 일>\"" >&2
  exit 2
fi

sref=$(stat -c %Y "$STATE" 2>/dev/null) || exit 0
newer_than_state() {   # $1 = path; true if modified more than SLACK_MIN after the briefing
  local t; t=$(stat -c %Y "$1" 2>/dev/null) || return 1
  [ "$t" -gt "$((sref + SLACK_MIN * 60))" ]
}

# (1) training runs that appeared or advanced after the briefing was last written
if [ -d "$LOGS" ]; then
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    n=$(basename "$d")
    # only runs with real training in them
    cnt=$(find "$d" -maxdepth 1 -name 'model_*.pt' 2>/dev/null | wc -l)
    [ "${cnt:-0}" -lt 2 ] && continue
    if newer_than_state "$d"; then
      msg+="  - 학습 런 '$n'이 브리핑 이후에 움직였습니다 → done/now 로 반영\n"
    fi
  done < <(find "$LOGS" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
fi

# (2) experiment notes written after the briefing
while IFS= read -r f; do
  [ -z "$f" ] && continue
  newer_than_state "$f" && msg+="  - 실험 노트 '$(basename "$f")'가 브리핑보다 최신입니다 → 결과를 done 으로\n"
done < <(find "$R/docs/experiments" -maxdepth 1 -name '*.md' -newermt "-2 days" 2>/dev/null)

# (3) commits after the briefing
last_commit=$(cd "$R" && git log -1 --format=%ct 2>/dev/null)
if [ -n "${last_commit:-}" ] && [ "$last_commit" -gt "$((sref + SLACK_MIN * 60))" ]; then
  subj=$(cd "$R" && git log -1 --format=%s 2>/dev/null | cut -c1-60)
  msg+="  - 브리핑 이후 커밋됨: \"$subj\" → 산출물이면 done 으로\n"
fi

# (4) "now" claims a run that is not running any more
cur=$(python3 - "$STATE" <<'PY' 2>/dev/null
import json, sys
try:
    print((json.load(open(sys.argv[1], encoding='utf-8')).get('now') or {}).get('title', ''))
except Exception:
    print('')
PY
)
# Fire only when the title claims training is RUNNING and neither a trainer nor any other
# measurement/eval process exists. The first version matched the bare word "학습", which
# flagged "학습 완료, 채점 중" while the evaluator was busily running - a false positive
# from our own heuristic, twice in one evening.
if echo "$cur" | grep -qiE '학습 중|학습중|training|본런|스모크 실행'; then
  # Runs on the remote 3090 (title says 원격) are checked there; an unreachable host is not flagged here
  # (the run's own watcher reports that).
  if echo "$cur" | grep -q '원격'; then
    alive() { timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=5 syaro@10.8.0.14 \
      'pgrep -f "[t]rain_wandb_video|[r]un_v2_scratch|[t]rain_actor_warmstart" >/dev/null' 2>/dev/null; [ $? -ne 1 ]; }
  else
    alive() { pgrep -f "train_wandb_video|run_training|run_v2_scratch" >/dev/null 2>&1; }
  fi
  if ! alive; then
    msg+="  - '지금 하고 있는 일'은 진행 중인 학습을 가리키는데 학습 프로세스가 없습니다 → now 갱신\n"
  fi
fi

if [ -n "$msg" ]; then
  {
    echo "★★ BRIEFING STALE — docs/000.Real-time Brefing.md 가 실제 상황보다 뒤쳐졌습니다:"
    printf "%b" "$msg"
    echo "  갱신: python3 tools/briefing/briefing.py {now|done|block|correct|add} ...  (realtime-briefing 스킬 참조)"
    echo "  감사자가 읽는 페이지입니다 — 결과 먼저, 숫자 포함, 전문용어는 풀어서, 산출물은 --media 로."
  } >&2
  exit 2
fi
exit 0
