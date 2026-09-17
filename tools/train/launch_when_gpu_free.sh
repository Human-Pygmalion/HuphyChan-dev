#!/bin/bash
# 그래픽카드가 비면 **그때** 학습을 발사한다.
#
# 2026-09-18 · 사용자 지시: *"GPU 비면 바로 A/B 걸어."*
#
# 왜 필요한가 — 지금 두 대 다 다른 세션의 학습이 물고 있습니다. 사람이 지켜보고 있다가
# 거는 대신, 비는 순간을 기계가 잡아서 겁니다. 늦게 발사되면 그만큼 밤이 낭비됩니다.
#
# "비었다" 의 정의: **학습 프로세스가 없다.** 사용률만 보면 잠깐 내려간 순간에 잘못
# 발사해 남의 학습을 느리게 만듭니다. 프로세스가 사라진 것이 확실한 신호입니다.
# 그리고 한 번 비었다고 바로 쏘지 않고 **연속 3 회**(90 초) 비어 있어야 쏩니다 —
# 학습이 단계 사이에 잠깐 죽었다 살아나는 경우가 있습니다.
#
#   launch_when_gpu_free.sh <host|local> <이름> <발사명령...>
#
# 호스트가 `local` 이면 이 기계에서, 아니면 그 기계에 ssh 로 겁니다.

set -u
HOST="$1"; NAME="$2"; shift 2
CMD="$*"
LOG="/tmp/wait_gpu_${NAME}.log"
NEED=3          # 연속 몇 번 비어 있어야 쏘나
INTERVAL=30     # 몇 초마다 보나
: > "$LOG"

# 2026-09-18: 처음엔 그냥 `pgrep -f "run_v2_scratch"` 로 봤는데, **이 감시기 자신의
# 명령줄에 그 글자가 들어 있습니다** — 발사할 명령을 인자로 들고 있으니까요. 그래서
# 자기 자신을 학습으로 착각해 영영 안 쏩니다. 실제로 그렇게 났습니다.
#
# 그래서 **명령줄이 파이썬으로 시작하는 것만** 셉니다. 진짜 학습은
# `.venv/bin/python3 -u analysis/...` 로 시작하고, 이 감시기는 `/bin/bash ...`,
# 원격 점검용 껍데기는 `bash -c ...` 로 시작합니다.
PAT='^[^ ]*python3.*(train_wandb_video|run_v2_scratch)'

busy() {
  if [ "$HOST" = local ]; then
    pgrep -f "$PAT" >/dev/null 2>&1
  else
    timeout 20 ssh -o BatchMode=yes "$HOST" "pgrep -f '$PAT' >/dev/null" 2>/dev/null
  fi
}

free_count=0
while true; do
  if busy; then
    free_count=0
  else
    free_count=$((free_count + 1))
    echo "$(date +%H:%M:%S) $HOST 비어 있음 ($free_count/$NEED)" >> "$LOG"
  fi
  if [ "$free_count" -ge "$NEED" ]; then
    echo "$(date +%H:%M:%S) 발사: $CMD" >> "$LOG"
    if [ "$HOST" = local ]; then
      cd /home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab || exit 1
      setsid nohup bash -c "$CMD" < /dev/null >> "$LOG" 2>&1 &
    else
      ssh -o BatchMode=yes "$HOST" "cd ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab && setsid nohup bash -c '$CMD' < /dev/null > /tmp/${NAME}_launch.log 2>&1 & disown" >> "$LOG" 2>&1
    fi
    echo "$(date +%H:%M:%S) 발사 완료" >> "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
