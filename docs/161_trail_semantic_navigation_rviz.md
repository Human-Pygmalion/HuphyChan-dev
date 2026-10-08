# 161. 산길 지도에서의 자율 경로 — RViz 그림 (2026-10-08)

> 약어를 쓰지 않는다. 처음 보는 사람도 읽을 수 있게 쓴다. 앞선 창고 그림은 [[159_nav_article_layout_proposals]], 인수인계는 [[160_rviz_isaac_visualization_handoff]].

## 0. 결과

![[layout_trail_autonomy.png]]

- 지도: `Map_Data/all_raw_points.ply` — 색이 있는 라이다 지도, 점 1,747만 개, 약 120 m × 200 m, 높이 차 약 17 m. 나무 사이로 굽이치는 산길.
- **계산한 것 (그림용으로 그린 것 없음)**: 지면 높이 지도, 지형 분류(산길·거친 땅·가파른 비탈·나무), 분류별 비용, 그 비용 위의 A* 경로.
  경로 182.4 m, 높이 차 9.0 m, 경로의 99 % 가 산길 분류 위.
- 그림: 3090 의 진짜 RViz2 (가상 화면) 캡처 두 장 — 로봇 뒤 근접 장면, 전체 경로를 위에서 본 장면 — 을 합성. 아래 줄은 움직이는 방식:
  `LiDAR map → terrain classes → A* path → velocity command → RL walking policy`.

## 1. 아직 진짜가 아닌 것 (정직하게)

| 항목 | 지금 | 진짜로 하려면 |
|---|---|---|
| 로봇 위치 | 계획 경로 위 42 % 지점에 **놓은 것** | 이 지형을 높이 지도로 시뮬레이터에 넣고 정책으로 실제로 걷게 해서 그 궤적을 재생 |
| 로봇 관절 각도 | 창고(평지)에서 v11a 정책이 걷던 한 순간 | 위와 같이 이 지형에서 걷는 순간 |
| 라이다 점 (흰색) | 지도 점 중 로봇 7 m 안, 지면 0.3~3 m 위 | 센서 모델로 그 자리에서 스캔 |
| 지형 분류 | 기하만 (기울기·거칠기·지면 위 점 밀도) | 성민 팀의 험지 분류(색·학습 기반)가 있으면 그것으로 |
| 속도 명령 → 정책 | 그림의 화살표뿐 | 경로 추종기(pure pursuit)가 v11a 에 명령을 주는 폐루프 |

## 2. 분류 규칙 (`tools/terrain_nav/prep_terrain.py`)

| 분류 | 규칙 | 비용 |
|---|---|---|
| 산길 | 그 외 지면 (완만·매끈) | 1 (가운데일수록 더 쌈) |
| 거친 땅 | 기울기 > 10° 또는 지면 거칠기 > 0.12 m | 4 |
| 가파른 비탈 | 기울기 > 20° | 40 |
| 나무·덤불 | 지면 위 0.3~2.2 m 점이 칸당 12 개 이상이고 20 % 이상, 1~2 칸 폭 선 모양은 제거 | 300 (벽이 아닌 매우 비싼 칸 — 선 모양 오검출이 길을 끊지 않게) |
| 모름 | 지면이 안 보임 | 지나갈 수 없음 |

격자 0.25 m, 지면 = 칸의 5 번째 백분위 높이(튀는 값 제거·구멍 메움·평활). 첫 시도에서는 나무를 벽으로 두었더니 산길 가운데의 얇은 선
(지도 만들 때 지나간 사람·장비로 추정)이 길을 끊어 경로가 없었다 — 위의 비용·제거 규칙은 그래서 생겼다.

## 3. 다시 만드는 법

```bash
# 이 PC
mujoco-sim/mjlab/.venv/bin/python3 tools/terrain_nav/prep_terrain.py Map_Data/all_raw_points.ply terrain.npz   # 약 20 초
rsync -a tools/terrain_nav/{publish_terrain.py,capture_terrain.sh,terrain.rviz,terrain_top.rviz} terrain.npz syaro@10.8.0.14:pyg_demo/terrain/
# 3090
cd ~/pyg_demo/terrain && bash capture_terrain.sh terrain.rviz hero.png 45                               # 로봇 뒤 근접 (PUB_ARGS="--at 0.6" 로 위치)
PUB_ARGS="--no-text --dots --corridor 6" bash capture_terrain.sh terrain_top.rviz top.png 45            # 전체 경로 위에서
# 합성 (이 PC 또는 Windows)
python3 tools/terrain_nav/compose_terrain.py hero.png top.png layout_trail_autonomy.png --stats "182 m"
```

카메라는 캡처 스크립트가 로봇 위치·방향을 읽어 로봇 뒤 10 m, 옆으로 30° 에 둔다 (`CAM_D`, `CAM_P`, `CAM_YAW` 로 바꿈).
원본: `docs/img/terrain_nav/rviz_trail_hero.png`, `rviz_trail_top.png`.
