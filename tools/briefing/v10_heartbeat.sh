#!/usr/bin/env bash
# Refresh the standing "A/B v10 정기 점검" briefing entry with live numbers from both arms
# (v10a local launcher log, v10b over ssh on 10.8.0.14) and commit it under the git lock.
#   tools/briefing/v10_heartbeat.sh ["extra sentence for the finding"]
set -euo pipefail
cd "$(dirname "$0")/../.."
M=mujoco-sim/mjlab
f() { sed -E -e 's/.*\[(P1)\] iter ([0-9]+) stage ([0-9.]+) dwell ([0-9]+) err ([0-9.]+) .*fell ([0-9.]+).*/\1 \2 회 · 단계 \3 · 체류 \4 · 오차 \5 · 넘어짐 \6/' \
             -e 's/.*\[(P2)\] iter ([0-9]+)\/([0-9]+) dr_factor ([0-9.]+).*/\1 \2 \/ \3 회 · 무작위화 세기 \4/'; }
A=$(grep -E "\[P[12]\] iter" $M/analysis/out/v10a_launch.out | tail -1 | f)
B=$(ssh -o BatchMode=yes -o ConnectTimeout=15 syaro@10.8.0.14 \
  'grep -E "\[P[12]\] iter" ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/analysis/out/v10b_launch.out | tail -1' | f || echo "원격 응답 없음")
python3 tools/briefing/briefing.py done v10-heartbeat-0229 "A/B v10 정기 점검 ($(date +%H:%M)) — 두 팔 학습 중" \
  --finding "두 팔 모두 학습 중입니다. ${1:-}" --number "v10a: $A · v10b: $B" | tail -1
( flock -w 300 9 && git add "docs/000.Real-time Brefing.md" docs/.briefing_state.json \
  && git commit -q -m "brief: v10 heartbeat" && git log --oneline -1 ) 9>/home/syaro/pyg_fea/locks/git.lock
