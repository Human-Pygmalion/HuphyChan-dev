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

## 4. 지도 전체를 직접 보고 경로 찍어 보기 (대화형 RViz, 2026-10-08)

3090 의 가상 화면에서 RViz2 가 계속 떠 있고, VNC 로 붙어서 직접 돌려 볼 수 있다.

| 항목 | 값 |
|---|---|
| 접속 | VNC 뷰어(TightVNC·RealVNC 등) → `10.8.0.14:5901` (0.0.0.0 에 묶임) |
| 비밀번호 | 3090 의 `~/pyg_demo/nav/VNC_PASSWORD.txt` (syaro 만 읽을 수 있음): `ssh syaro@10.8.0.14 cat ~/pyg_demo/nav/VNC_PASSWORD.txt` |
| 켜기·끄기 | `ssh syaro@10.8.0.14 'bash ~/pyg_demo/nav/run_nav_viewer.sh'` / `... run_nav_viewer.sh stop` |
| 출발점 | 위 도구줄 **2D Pose Estimate** 를 누르고 지도를 클릭 |
| 목적지 | **2D Goal Pose** 를 누르고 클릭 → 0.4 초 안에 A* 가 다시 계획 (경로 길이·높이 차·산길 비율이 목적지 위에 뜬다) |
| 지도 층 | `Map 10 cm`(217만 점, 기본 켬) · `Map 5 cm`(526만) · `Map full`(1,747만, 원본 전부) — 왼쪽 목록에서 켜고 끈다 |
| 지형 분류 | `Terrain classes`: 노랑 산길 · 주황 거친 땅 · 빨강 가파른 비탈 · 초록 나무 (25 cm 칸) |

- 그래픽 가속이 없는 소프트웨어 렌더라 10 cm 층에서 약 4 fps 다. 5 cm·원본 층은 더 느리니 볼 때만 켠다.
- 로봇은 출발점에 경로 방향으로 서 있다(관절은 평지 걷기의 한 순간). 경로를 따라 걷는 동작은 아직 없다 — §1.
- 고친 함정: 예전 캡처 스크립트가 `ros2 run` 겉껍데기만 끄고 안의 `robot_state_publisher` 를 남겨 3090 에 29 개가 쌓여 있었다
  (로봇 좌표가 서로 충돌). 모두 정리했고, 세 스크립트(`run_nav_viewer.sh`, `capture_terrain.sh`, `isaac_demo/rviz/capture_rviz.sh`)가
  이제 자식 프로세스까지 끈다. 로봇 표시는 2 초마다 다시 보내서, 시작 순간의 "좌표 없음" 오류가 남지 않게 했다.

파일: `tools/terrain_nav/{nav_server.py, nav.rviz, run_nav_viewer.sh, prep_clouds.py}`. 데이터는 `prep_terrain.py`·`prep_clouds.py` 로 만들어
3090 `~/pyg_demo/nav/` 에 둔다 (약 400 MB).
