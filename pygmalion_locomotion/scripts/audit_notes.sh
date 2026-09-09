#!/bin/bash
# audit_notes.sh — Stop-HOOK enforcement: BLOCK (exit 2) if substantial training runs / analysis are
# un-recorded as notes. The memory rules kept being forgotten; this makes the HARNESS enforce them so it
# does not depend on the model remembering. stderr is fed back to the model as the reason to keep working.
# Wired as a Claude Code Stop hook (settings.json). Robust: any internal error -> exit 0 (never wedge).
ROOT=/home/syaro/MikuchanRemote/Human-Pygmalion
cd "$ROOT" 2>/dev/null || exit 0
g=""

# (1) DONE substantial runs (>=500 iter, inactive >5min) whose name is referenced NOWHERE in docs/
for d in pygmalion_locomotion/logs/rsl_rl/*/*/; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  # skip runs matching the explicit skip-list (exploratory / config-test runs not individually noted)
  [ -f docs/experiments/.audit_skip ] && grep -qFf docs/experiments/.audit_skip <<<"$run" 2>/dev/null && continue
  l=$(ls "$d"model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
  [ -z "$l" ] && continue
  [ "$l" -lt 500 ] 2>/dev/null && continue
  # skip ACTIVE runs (a checkpoint written in the last 5 min)
  [ -n "$(find "$d" -maxdepth 1 -name 'model_*.pt' -mmin -5 2>/dev/null)" ] && continue
  short=$(echo "$run" | cut -d_ -f3-)   # strip YYYY-MM-DD_HH-MM-SS -> run_name (INDEX refs use short name)
  { grep -rqF "$run" docs/ 2>/dev/null || { [ -n "$short" ] && grep -rqF "$short" docs/ 2>/dev/null; }; } && continue
  g="$g\n  - run '$run' (iter $l): NO docs note  -> docs/experiments/$run.md + INDEX"
done

# (2) experiment notes left with [작성 필요] placeholders
for f in $(grep -rlF "작성 필요" docs/experiments/*.md 2>/dev/null); do
  g="$g\n  - ${f##*/}: [작성 필요] 미작성 (정량/정성 분석 채우기)"
done

# (3) ad-hoc ANALYSIS plots in docs/assets not embedded in any note
for p in docs/assets/*.png; do
  [ -e "$p" ] || continue
  n=${p##*/}
  case "$n" in *knee_*|*analysis*|*demand*|*envelope*|*_split*|*compare*|*ratio*) ;; *) continue ;; esac
  grep -rqF "$n" docs/ --include="*.md" 2>/dev/null && continue
  g="$g\n  - plot '$n': 어떤 노트에도 미임베딩 (분석 노트에 Obsidian 임베드 ![[$n]] 추가)"
done

# (4) substantial TRAINING runs with NO in-training VIDEO (--no_video used -> gait 디버깅 불가; user rule:
#     항상 영상 ON, FLAT=밀도유지 / ROUGH=spacing조정해 로봇 분간). accumulate 또는 train clip 둘 다 없으면 flag.
for d in pygmalion_locomotion/logs/rsl_rl/*/*/; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  [ -f docs/experiments/.audit_skip ] && grep -qFf docs/experiments/.audit_skip <<<"$run" 2>/dev/null && continue
  l=$(ls "$d"model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
  [ -z "$l" ] && continue
  [ "$l" -lt 500 ] 2>/dev/null && continue
  [ -n "$(find "$d" -maxdepth 1 -name 'model_*.pt' -mmin -5 2>/dev/null)" ] && continue
  # ★ require BOTH (user 2026-06-22): OVERVIEW (train clips / accumulate) AND CLOSE-UP (play single-robot).
  #   GRANDFATHER runs that PREDATE this rule (date < 2026-06-22): overview alone satisfied the old rule, so
  #   we do NOT retroactively demand a close-up for historical experiments (that would just burn GPU).
  ov=0; { [ -f "${d}videos/accumulated_progress.mp4" ] || ls "${d}videos/train/"*.mp4 >/dev/null 2>&1; } && ov=1
  cu=0; ls "${d}videos/play/"*.mp4 >/dev/null 2>&1 && cu=1
  need_cu=1; [[ "${run:0:10}" < "2026-06-22" ]] && need_cu=0
  [ "$ov" = 1 ] && { [ "$need_cu" = 0 ] || [ "$cu" = 1 ]; } && continue
  miss=""; [ "$ov" = 0 ] && miss="overview"; { [ "$need_cu" = 1 ] && [ "$cu" = 0 ]; } && miss="$miss close-up"
  g="$g\n  - run '$run' (iter $l): 영상 누락 ($miss) -> 학습 영상ON(overview) + 끝나고 play.py --video(단일로봇 클로즈업). gait 디버깅용"
done

# (4b) ★mjlab 런 노트 감사 (2026-07-12 사용자 적발: mjlab 경로가 스캔에서 빠져 4개 런이
#      정식 리포트 없이 통과) — mjlab logs의 substantial 런(>=1000 iter, 비활성)도 (1)과
#      동일하게 docs/ 참조 여부를 확인. 비교/통합 노트는 대체 불가이므로 run 이름이
#      docs/experiments/*.md "파일명"에 있어야 통과(본문 언급만으로는 부족).
for d in mujoco-sim/mjlab/logs/rsl_rl/*/*/; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  [ -f docs/experiments/.audit_skip ] && grep -qFf docs/experiments/.audit_skip <<<"$run" 2>/dev/null && continue
  l=$(ls "$d"model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
  [ -z "$l" ] && continue
  [ "$l" -lt 1000 ] 2>/dev/null && continue
  [ -n "$(find "$d" -maxdepth 1 -name 'model_*.pt' -mmin -10 2>/dev/null)" ] && continue
  short=$(echo "$run" | cut -d_ -f3-)
  ok=0
  for f in docs/experiments/*.md; do
    bn=$(basename "$f")
    case "$bn" in *"$short"*|*"$run"*) ok=1; break;; esac
  done
  # dedicated report may use the short run_name in its body header (legacy runs) —
  # accept a file whose FIRST line mentions the run timestamp (report title format).
  [ "$ok" = 0 ] && grep -l "^# 학습 리포트.*$(echo "$run" | cut -d_ -f1-2)" docs/experiments/*.md >/dev/null 2>&1 && ok=1
  [ "$ok" = 1 ] && continue
  g="$g\n  - mjlab run '$run' (iter $l): 정식 per-run 리포트 없음 -> docs/experiments/<run>.md (비교노트 대체불가, 템플릿 §1~§12+§R)"
done

# (5) ★registry/canvas 동기화 (user 2026-07-11: "이 정리는 매 실험마다 반드시 수행해") —
#     새 실험노트가 생겼는데 66_experiment_registry.md / experiment_map.canvas가 그보다 오래되면 BLOCK.
#     30분 슬랙: 노트 직후 같은 세션에서 레지스트리를 곧 갱신하는 정상 흐름은 통과.
REG=docs/66_experiment_registry.md; CAN=docs/experiment_map.canvas
if [ -f "$REG" ] && [ -f "$CAN" ]; then
  newest_note=$(ls -t docs/experiments/2026-*.md 2>/dev/null | head -1)
  if [ -n "$newest_note" ]; then
    nt=$(stat -c %Y "$newest_note" 2>/dev/null || echo 0)
    rt=$(stat -c %Y "$REG" 2>/dev/null || echo 0)
    ct=$(stat -c %Y "$CAN" 2>/dev/null || echo 0)
    slack=1800
    [ $((nt - rt)) -gt $slack ] && g="$g\n  - registry 미갱신: '$(basename "$newest_note")'가 66_experiment_registry.md보다 새로움 -> era표에 런/변인/정량 행 추가"
    [ $((nt - ct)) -gt $slack ] && g="$g\n  - canvas 미갱신: experiment_map.canvas에 새 실험 노드/엣지 추가 (계보 연결)"
  fi
fi

# (6) ★실시간 브리핑 갱신 (user 2026-08-27: "이거 항상 발동하도록 ... 언제나 반응하도록 해줘") —
#     docs/000.Real-time Brefing.md 는 외부 감사자가 읽는 현황 페이지다. 실험 노트나 산출물이
#     생겼는데 브리핑 상태파일이 그보다 오래되면 BLOCK. 손으로 페이지를 고치지 말고
#     tools/briefing/briefing.py 로 갱신할 것(페이지는 상태파일에서 렌더된다).
#     30분 슬랙: 작업 직후 같은 세션에서 곧 갱신하는 정상 흐름은 통과.
BST=docs/.briefing_state.json
if [ -f "$BST" ]; then
  bt=$(stat -c %Y "$BST" 2>/dev/null || echo 0)
  newest_out=$(ls -t docs/experiments/2026-*.md docs/img/*.png docs/video/*.mp4 2>/dev/null | head -1)
  if [ -n "$newest_out" ]; then
    ot=$(stat -c %Y "$newest_out" 2>/dev/null || echo 0)
    if [ $((ot - bt)) -gt 1800 ]; then
      g="$g\n  - 실시간 브리핑 미갱신: '$(basename "$newest_out")'가 docs/000.Real-time Brefing.md보다 새로움 -> tools/briefing/briefing.py 로 now/done/block/correct 반영 (realtime-briefing 스킬)"
    fi
  fi
else
  g="$g\n  - 실시간 브리핑 상태파일이 없습니다 -> tools/briefing/briefing.py 로 생성 (realtime-briefing 스킬)"
fi

# (7) WARN (not a block) — 학습 런 노트에 설정 명세 표(§1b-2 액추에이터/§1b-3 ROM/§1b-4 플래그)가 없음.
#     사용자 지시 2026-09-03 "이건 모든 Docs에 다 넣으라고". 생성은 손이 아니라 도구로:
#       python3 tools/notes/backfill_spec_tables.py --only <노트>
#     경고로만 두는 이유: 신규 노트는 런처(run_v2_scratch.py / run_training.sh)가 자동으로
#     채우므로, 여기서 막으면 아직 학습이 안 끝난 노트에 대해 헛되이 세션을 붙잡는다.
w=""
for n in docs/experiments/2026-*.md; do
  [ -f "$n" ] || continue
  grep -q "SPEC-TABLES:BEGIN" "$n" 2>/dev/null && continue
  grep -q "1b-2" "$n" 2>/dev/null && continue
  # 학습 런이 아닌 노트(벤치 모터 측정 등)는 스스로 그렇게 밝힌다. §1b-2/3/4 는 학습 설정에서
  # 나오는 표라, 대응하는 런 디렉터리가 없는 노트에는 채울 근거가 없다 — 억지로 넣으면
  # 근거 없는 표가 남는다. 2026-09-10 추가.
  grep -q "SPEC-TABLES: 해당 없음" "$n" 2>/dev/null && continue
  w="$w\n  - $(basename "$n"): §1b-2/§1b-3/§1b-4 설정 명세 표 없음"
done
if [ -n "$w" ]; then
  printf "NOTE-AUDIT WARN — 학습 런 노트에 설정 명세 표(§1b-2 액추에이터·§1b-3 ROM·§1b-4 플래그)가 빠졌습니다 (블록 아님):%b\n  고치기: python3 tools/notes/backfill_spec_tables.py --only <노트명>\n" "$w" >&2
fi

if [ -n "$g" ]; then
  printf "★★ NOTE-AUDIT BLOCK — 미기록 실험/분석이 있습니다. 종료 전 노트화하세요 (feedback-training-report-rule / feedback-research-recording-rule):%b\n(진짜 끝내려면: 노트를 만들거나, config-test면 무시 가능 — 단 반복 누락은 사용자가 싫어함)\n" "$g" >&2
  exit 2
fi
exit 0
