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
| 로봇 위치 (RViz 그림) | 계획 경로 위 42 % 지점에 **놓은 것** | 지형은 이제 MuJoCo 에 들어갔고 정책이 그 위를 직접 걷는다(§5, 사람이 조종). 남은 것: 경로 추종기가 명령을 주는 자율 주행의 궤적을 RViz 에 재생 |
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
| 목적지 | **2D Goal Pose** 를 누르고 클릭 → 0.4 초 안에 A* 가 다시 계획 (화면의 큰 글씨는 뺐다. 경로 길이 등은 `nav_server.py` 로그에 찍힌다) |
| 지도 층 | `Map 10 cm`(217만 점, 기본 켬) · `Map 5 cm`(526만) · `Map full`(1,747만, 원본 전부) — 왼쪽 목록에서 켜고 끈다 |
| 지형 분류 | `Terrain classes`: 노랑 산길 · 주황 거친 땅 · 빨강 가파른 비탈 · 초록 나무 (25 cm 칸) |
| 지형 면 | `Terrain mesh (= collider)` (기본 켬): 지도 색을 입힌 매끈한 삼각형 면 — §5 |
| 카메라 | 로봇(`base_link`)을 중심으로 도는 시점, 거리 6.5 m. 마우스 왼쪽 끌기 = 회전, 휠 = 확대 |

- 그래픽 가속이 없는 소프트웨어 렌더라 10 cm 층에서 약 4 fps 다. 5 cm·원본 층은 더 느리니 볼 때만 켠다.
- 로봇은 출발점에 경로 방향으로 서 있다(관절은 평지 걷기의 한 순간). 경로를 따라 걷는 동작은 아직 없다 — §1.
- 고친 함정: 예전 캡처 스크립트가 `ros2 run` 겉껍데기만 끄고 안의 `robot_state_publisher` 를 남겨 3090 에 29 개가 쌓여 있었다
  (로봇 좌표가 서로 충돌). 모두 정리했고, 세 스크립트(`run_nav_viewer.sh`, `capture_terrain.sh`, `isaac_demo/rviz/capture_rviz.sh`)가
  이제 자식 프로세스까지 끈다. 로봇 표시는 2 초마다 다시 보내서, 시작 순간의 "좌표 없음" 오류가 남지 않게 했다.

파일: `tools/terrain_nav/{nav_server.py, nav.rviz, run_nav_viewer.sh, prep_clouds.py}`. 데이터는 `prep_terrain.py`·`prep_clouds.py` 로 만들어
3090 `~/pyg_demo/nav/` 에 둔다 (약 400 MB).

## 5. 매끈한 지형 면 + 충돌체, 그리고 직접 조종해서 걷기 (MuJoCo, 2026-10-08)

![[rviz_terrain_mesh_view.png]]

위: 3090 RViz (VNC). 지도 색을 입힌 지형 면, 로봇 중심 카메라, 오른쪽 껍데기(`shell_*_R`)만 숨김, 화면의 큰 글씨 없음.

![[teleop_browser_view.png]]

위: 브라우저 조종 화면 (이 PC 에서 자동으로 눌러 본 화면). 앞으로 0.4 m/s 명령 → 10 초에 4.4 m 걸음, 상태 "walking (RL policy v11a)".

### 5-1. 지형 면과 충돌체 — 같은 높이에서 만든 두 가지 (`tools/terrain_nav/prep_terrain_mesh.py`)

| 것 | 해상도 | 쓰는 곳 | 만드는 법 |
|---|---|---|---|
| 충돌체 (높이 지도, heightfield) | 10 cm, 1232 × 2022 칸 | MuJoCo 물리 | 칸마다 지도 점 높이의 아래 5 % 값 → 튀는 점 제거 → 약한 평활. 스캔이 못 본 칸은 1 m 안이면 주변 값으로 채우고, 더 멀면 **2 m 벽**으로 올려 허공으로 걸어 나가지 못하게 |
| 나무 기둥 | 268 개 원기둥, 반지름 0.08~0.6 m | MuJoCo 물리 | 나무 분류 칸 덩어리마다 하나. 반지름 = 덩어리 넓이 원의 0.6 배 (줄기는 잎보다 가늘다) |
| 보이는 면 (mesh) | 25 cm, 정점 47,939 · 삼각형 67,779 | RViz, 브라우저, 다른 시뮬레이터 (`terrain_mesh.ply`) | 칸 중앙을 정점으로 한 삼각형 면. 색 = 지면 근처 점들의 실제 색 55 % + 분류 색 45 %. 한 칸에 0.5 m 넘게 솟는 거의 수직 삼각형(나무 밑에서 지면을 잘못 잡은 얇은 지느러미)은 **보이는 면에서만** 뺐다 |

- 예전의 "사각형 칸" 은 분류 표시(`Terrain classes`, 25 cm 정육면체)였다. 그건 그대로 두고 꺼 두었다.
- 보이는 면은 25 cm, 충돌체는 10 cm 라서 둘은 완전히 같지 않다. 발이 실제로 닿는 건 10 cm 쪽이다.

### 5-2. 이 지형을 MuJoCo 에 넣기

- `tools/terrain_nav/make_sim_terrain.py`: 계획 경로의 첫 점(산길 위)을 시뮬레이션 원점으로, 그 자리 지면 높이를 0 으로 옮긴다.
  지도 좌표 = 시뮬레이션 좌표 + (3.40, 0.43, −0.38) m.
- `tools/pygviewer/pygviewer/bake.py`: 환경변수 `PYG_BAKE_TERRAIN=<trail_sim_terrain.npz>` 를 줄 때만 높이 지도 + 나무 기둥을 모델에 더한다.
  마찰·접촉 설정은 원래 바닥과 같고, 원래 평평한 바닥은 지형 아래로 내린다. 변수를 안 주면 예전과 똑같다 (로봇 설정 파일은 건드리지 않았다).
- 모델이 바뀌면 정책 계약 파일의 모델 지문(`model_contract_sha`)이 달라진다. 그래서 별도 폴더 `~/pyg_demo/cache_trail` 에 굽고,
  그 폴더의 v11a 계약 사본만 새 지문으로 바꿨다 (원래 지문은 `model_contract_sha_original` 에 남김). 원래 캐시는 그대로다.
- 확인: 로봇 질량 23.63 kg 그대로. 이 지형에서 v11a 를 0.6 m/s 로 15 초 → 7.87 m, 넘어지지 않음, 평균 0.58 m/s (`tools/isaac_demo/walk_probe.py`).

### 5-3. 직접 조종하기 (`tools/terrain_nav/trail_teleop.py`, 3090 에서 실행)

| 항목 | 값 |
|---|---|
| 3D 화면 + 조종판 | 브라우저로 `http://10.8.0.14:8095/` (0.0.0.0 에 묶임) |
| 키보드 조종 | 다른 탭에 `http://10.8.0.14:8096/` 를 열고 그 탭을 누른 채 W/↑ 빠르게 · S/↓ 느리게(뒤로) · A/D 회전 · Space 정지 · R 넘어졌을 때 그 자리에서 다시 세우기 |
| 조종판 | 앞 속도 −0.4~1.2 m/s, 회전 −0.6~0.6 rad/s, 카메라 로봇 따라가기, 지형 면·점 지도 켜고 끄기, 출발점으로 되돌리기 |
| 켜기 | `ssh syaro@10.8.0.14` 후 `cd ~/MikuchanRemote/Human-Pygmalion/tools/terrain_nav && CUDA_VISIBLE_DEVICES="" setsid nohup nice -n 5 ../../mujoco-sim/mjlab/.venv/bin/python3 trail_teleop.py --cache $HOME/pyg_demo/cache_trail --nav $HOME/pyg_demo/nav > $HOME/pyg_demo/nav/teleop.log 2>&1 &` |
| 끄기 | `ps -eo pid,args \| grep [t]rail_teleop` 로 번호를 보고 `kill <번호>` |
| 점검 | 이 PC 에서 `mujoco-sim/mjlab/.venv/bin/python3 tools/terrain_nav/ux_check_teleop.py 10.8.0.14 <out.png>` — W 4 번 → 10 초 걷기 → Space 를 실제로 눌러 본다 |

- 정책이 받는 것은 **속도 명령(앞 속도, 회전 속도)뿐**이다. 사람이 주든 경로 계획기가 주든 같은 입력이고, 다리 동작·균형·울퉁불퉁한 땅의 발 디딤은 정책이 스스로 한다.
  "원격 조종으로 다리를 움직이는 것이 아니다" 를 보여 주는 데 쓸 수 있다.
- 물리 200 Hz, 정책 50 Hz, 실시간. 그래픽 카드 없이 CPU 로 돈다. 3D 그림은 보는 사람 브라우저의 그래픽 카드가 그린다.
- 넘어짐 판정: 몸통이 그 자리 지면보다 0.45 m 아래로 내려가면 "넘어짐" 으로 표시하고 멈춘다 → R 로 다시 세운다.

### 5-4. 한계 (정직하게)

- RViz 의 로봇과 조종 시뮬레이션의 로봇은 **아직 연결돼 있지 않다** (RViz 는 출발점에 서 있는 그림).
- v11a 는 제자리 회전이 약하다 (회전 추종 약 17 %). 걸으면서 도는 것이 훨씬 잘 된다.
- 나무 기둥은 나무 분류 덩어리로 추정한 것이다. 실제 줄기 위치를 잰 것이 아니다.
- 지형 면 바깥 테두리가 들쭉날쭉한 것은 스캔이 본 영역의 경계이고, 군데군데 떠 보이는 조각은 비탈 쪽 스캔이 듬성한 곳이다.

파일: `tools/terrain_nav/{prep_terrain_mesh.py, make_sim_terrain.py, trail_teleop.py, ux_check_teleop.py}`, `tools/pygviewer/pygviewer/bake.py`.
3090 의 데이터: `~/pyg_demo/nav/{terrain_mesh.npz, terrain_mesh.ply, trail_sim_terrain.npz}`, 구운 모델 `~/pyg_demo/cache_trail/`.
