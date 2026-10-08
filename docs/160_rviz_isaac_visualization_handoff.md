# 160. 인수인계 — RViz · Isaac Sim 시각화 작업 (2026-10-08)

> 약어를 쓰지 않는다. 처음 보는 사람도 이 문서 하나로 이어서 작업할 수 있게 쓴다.
> 이 작업을 한 세션(Windows 쪽 보조 세션, 2026-10-06~07)이 사용 한도로 끊겨서, 그 대화 기록에서 상태를 복구해 정리했다.
> 자세한 근거는 [[157_shell_lidar_upper_body_overlay]](모델), [[158_isaac_warehouse_lidar_demo]](Isaac 데모),
> [[159_nav_article_layout_proposals]](기사용 그림) 에 있다. 이 문서는 "어디에 무엇이 있고, 어떻게 다시 돌리고, 무엇이 남았나"만 모은다.

## 0. 한눈에

| 만든 것 | 파일 | 상태 |
|---|---|---|
| Isaac Sim 창고에서 걷는 Huphy 1.0 영상 (쉘 + Ouster 라이다 + D435) | `docs/media/huphy_warehouse_walk.mp4` (46 초, 1920×1080, 30 fps) | 완료 |
| 기사용 그림 A안 — RViz 판 (연구 느낌) | `docs/img/nav_layouts/layout_A_rviz.png` | 완료 (시안 표시 있음) |
| 기사용 그림 A안 — 사진 판 | `docs/img/nav_layouts/layout_A_hero.png` | 완료 (시안 표시 있음) |
| RViz 근접 장면 + 그림 속 그림 2개 (지도·경로 / D435 깊이) + 캡션 | `docs/img/nav_layouts/rviz/rviz_close_view.png` | 완료 (포스터용) |
| 이전 시안 B·C·D, 한 장 모음 | `docs/img/nav_layouts/layout_{B,C,D}_*.png`, `layouts_overview.png` | 이전 피드백 이전 버전 — 최신 문구 미반영 |
| 글자 없는 Isaac 원본 정지 화면 | `docs/img/nav_layouts/stills/`, `stills2/` (장애물 있는 최신 장면) | 완료 |

**ELI5**: 로봇 모델에 겉껍데기와 센서를 붙였고(157), 그 모델을 엔비디아의 시뮬레이터(Isaac Sim) 창고 안에서 걷게 해서
라이다 점과 카메라 영상을 함께 보여 주는 영상을 만들었다(158). 그다음 교수님 피드백("원격 조종이 아니라 스스로 길을 찾는 것으로
보여야 한다")에 맞춰, 로봇 연구에서 쓰는 화면 도구(RViz)로 지도·경로·목적지가 보이는 기사용 그림을 만들었다(159).
**지도와 경로는 아직 진짜 길찾기 결과가 아니라 그림용이다.**

## 1. 기계와 역할

| 기계 | 주소 | 이 작업에서 하는 일 |
|---|---|---|
| 이 PC (Linux, "177 서버") | `192.168.20.177` / 터널 `10.8.0.16` | 저장소 원본 (`~/MikuchanRemote/Human-Pygmalion`), 문서·커밋, 동영상 압축(ffmpeg). ROS 2 Humble·rviz2 는 있으나 가상 화면이 없고, **그래픽카드가 가상 머신에 연결되어 있지 않다** (할 일 #88) |
| Windows PC | `P:\Human-Pygmalion` = 위와 같은 폴더 | 그림 합성(Python + Pillow), 문서 확인 |
| 원격 RTX 3090 가상 머신 | `ssh syaro@10.8.0.14` | **Isaac Sim 실행과 RViz 캡처 모두 여기서.** 자기 디스크의 사본이며 git 저장소가 아니다 |
| Fusion 360 PC | `192.168.20.161:27182` | CAD 원본 (모델 작업 때만, 157) |

3090 쪽 구성:

| 위치 | 내용 |
|---|---|
| `~/isaac/venv311` | Isaac Sim 5.0.0 (pip 설치, Python 3.11). 설치 기록 `~/isaac/install.log` |
| `~/pyg_demo/` | 데모 작업 폴더 (1.1 GB): `warehouse_demo.py` 등 스크립트 사본, `assets/`(쉘 URDF·메시·MJCF·라이다 가림 마스크), `pyg_demo_cache/`(정리된 하체 모델 + v11a 정책), `tools/pygviewer/`, `world_dump.npz`(RViz 용 장면 자료), `huphy_warehouse_walk.mp4`(압축 전 원본) |
| `~/pyg_demo/rviz/` | RViz 캡처 사본: `capture_rviz.sh`, `publish_scene.py`, `scene*.rviz`, `grid.npz`, `huphy_rviz.urdf`, `rsp_params.yaml` |
| 시스템 | ROS 2 Humble 데스크톱 + Nav2 + SLAM Toolbox + Xvfb(가상 화면) + ImageMagick + x11vnc (2026-10-07 설치, `tools/isaac_demo/install_ros_rviz.sh`) |
| 환경 파일 | RViz 쪽 `~/ros_env_rviz.sh` (시스템 Python 3.10), Isaac 쪽 `~/pyg_demo/ros_env_isaac.sh` — **두 쪽은 서로 다른 셸에서 쓴다** |

## 2. 다시 돌리는 법

### 2.1 Isaac Sim 창고 데모 (3090)

```bash
ssh syaro@10.8.0.14
cd ~/pyg_demo && source ~/isaac/venv311/bin/activate
python warehouse_demo.py --mode video --seconds 46 --out demo.mp4     # 화면 없이, 단순 직진, 약 8 분
python warehouse_demo.py --mode web --port 8080                      # 브라우저로 조이스틱·키보드 운전, http://10.8.0.14:8080/
```

- 걷기는 viser(pygviewer)와 같은 계산 경로다: 하체 정책 v11a(`deploy_asset/pyg_v11a_stand01/`)가 MuJoCo 에서 걷고, 그 자세를 Isaac 의
  전신 쉘 모델에 그대로 옮긴다(팔은 고정). 명령 1.2 m/s → 실제 1.205 m/s, 쓰러짐 0.
- 기사용 장면: `--obstacles`(통로에 팔레트+상자 3개), `--path-ref "y:x,..."`(그 장애물을 돌아가는 걸음), `--dump-world world_dump.npz --dump-world-at <초>`(RViz 용 자료 저장).
- 키·창 목록·시작 위치는 158 §5·§9.
- 3090 에는 ffmpeg 가 없다. OpenCV 로 쓴 mp4 는 커서(386 MB) 이 PC 에서 `ffmpeg -c:v libx264 -crf 24` 로 줄였다.

### 2.2 RViz 그림 (3090 → 그림 합성)

1. **장면 자료**: Isaac 데모를 `--obstacles --path-ref … --dump-world ~/pyg_demo/world_dump.npz` 로 돌린다. 누적 라이다 점구름(6 cm 격자), 그 순간의 스캔, 로봇 자세·관절이 들어간다.
2. **점유 격자**: `tools/isaac_demo/isaac_map.py out.png` 로 지붕을 뺀 창고 위 그림을 뽑고 `rviz/make_grid.py <그림> grid.npz` 로 격자(0.05 m)를 만든다. **SLAM 결과가 아니라 렌더에서 만든 것**이다.
3. **캡처**: `bash ~/pyg_demo/rviz/capture_rviz.sh <설정.rviz> <out.png> [기다릴 초]` — 가상 화면(Xvfb :99, 2560×1440)에 rviz2 를 띄우고
   `publish_scene.py` 가 점구름·스캔·로봇(URDF 메시 마커)·격자·경로·목적지·라벨을 발행한 뒤 화면을 찍는다.
   설정: `scene.rviz`(전체 장면), `scene_close.rviz`(로봇 근접), `scene_ogm.rviz`(위에서 본 지도, `--dots` 로 현재 위치·목적지 점).
4. **합성** (Windows 또는 Pillow 가 있는 어디서든, `tools/isaac_demo/rviz/`):
   - `python make_close_view.py <docs/img/nav_layouts>` → `rviz/rviz_close_view.png` (글자 없는 근접 장면 + 좌상단 지도 그림 속 그림 + 우하단 D435 깊이, 각 캡션은 흰 반투명 상자, Arial 44 px)
   - `python compose_rviz_layout.py rviz/rviz_view.png rviz/rviz_close_view.png layout_A_rviz.png` → 상단 `CMD :` 막대 + 전체 장면 + 근접 삽입창
   - 사진 판 A안·이전 시안: `tools/isaac_demo/nav_layout_mockups.py`

## 3. 무엇이 진짜이고 무엇이 그림용인가

| 진짜 (시뮬레이션에서 나온 값) | 그림용 (아직 계산하지 않음) |
|---|---|
| 로봇 모양·자세·관절, 창고, 라이다 점(Isaac 내장 RTX 라이다), D435 컬러·깊이, 장애물 위치, 로봇이 실제로 걸은 길 | **계획 경로**(걸은 길을 그려 넣은 것), 목적지 깃발, 의미 라벨(PALLET·TABLE·RACK·AISLE·Dock A), `CMD :` 문장, 남은 거리 값, 점유 격자(렌더에서 만듦) |

그래서 기사용 그림에는 `LAYOUT MOCKUP` 표시를 남겨 두었다. 포스터에 쓸 때는 "In development (simulation)" 같은 캡션을 권한다.
현재 정책은 제자리 회전이 약하다(v11a 중앙값 17 %) — 완만한 곡선 경로만 실제로 걸을 수 있다.

## 4. 남은 일

| 일 | 메모 |
|---|---|
| 진짜 자율 주행 결과로 바꾸기 | ① 누적 라이다로 SLAM 지도 (SLAM Toolbox 설치됨) ② 지도 위 경로 계획(A* 또는 Nav2) ③ 장소 이름 → 목적지 사전 ④ 경로를 속도 명령으로 주행 ⑤ 그림을 진짜 결과로 다시 뽑기 (159 §4) |
| 라이다 기울기 확인 | CAD 메시 축으로 앞으로 40° 기울였더니 앞쪽은 바닥만 본다(정면 광선이 수평선 아래 18.8°~61.2°). 실제 장착 방향과 맞는지 사용자 확인 필요. 바꿀 곳: `tools/robot_model/shell_overlay/manifest.json` 의 `lidar.tilt_deg` 와 기울기 부호 |
| `--lidar-pitch-deg -40` 옵션 | 수평 스캔 비교용으로 넣었으나 점이 거의 안 나옴(0~2.8만 개) — 원인 미상, 쓰지 말 것 |
| 머리 확대 창 | 광선이 화면을 덮어 머리 쉘이 잘 안 보임 |
| 웹 모드 속도 | 렌더 약 5 fps 라 실시간이 아님. 가벼운 모드 없음 |
| 시안 B·C·D | 최신 피드백(`CMD :` 문구 등) 미반영 |
| RViz 글자 간격 | 소프트웨어 OpenGL 이라 라벨 간격이 넓게 나온다 |
| 전신 정책 | 없음. 하체 정책 + 상체 고정 |
| D435·라이다 세부 | D435 광학창 위치와 컬러 렌즈 좌우 비껴섬은 CAD 로 미확인, 라이다 줄 각도 표는 Isaac 의 OS1 REV7 설정 그대로 |
| Fusion 문서의 저장 안 된 수정 | `v27~recovered` 에 저장 안 된 수정이 있다는 사실만 확인됨, 내용은 API 로 못 봄 — 저장해 주면 다시 읽어 버전 기록 |

## 5. 함정 (이 작업에서 실제로 겪은 것)

- **`pkill -f <패턴>` 을 같은 명령줄에서 쓰면 자기 자신(셸)도 죽는다** — 프로세스 번호로 끄거나 패턴을 따로 실행.
- RViz(시스템 Python 3.10)와 Isaac(가상 환경 3.11)은 **같은 셸에서 섞지 않는다**.
- 설치 스크립트의 마지막 점검이 `set -u` 와 ROS 설정 파일의 미정의 변수 때문에 조용히 죽었다 — 저장소 판은 고쳐져 있다.
  다시 점검만: `bash ~/install_ros_rviz.sh --check`. 설치 자체는 sudo 암호가 필요해 사람이 했다.
- RViz 는 URDF 메시를 **절대 경로**로, 색을 **각 시각 요소 안에** 넣어야 그린다 — `capture_rviz.sh` 가 변환한다.
- RViz 첫 시험에서 현재 스캔의 바닥 점이 하늘색 판처럼 시야를 덮었다 — 높이 0.2 m 아래·먼 점을 빼고 흰 작은 점으로 그린다.
- 모델 파일은 손으로 고치지 않고 `tools/robot_model/shell_overlay/build_overlay.py` 로 다시 만든다(157). 한 번 팔 메시를 지우고 CAD 에서 다시 만들어
  어깨→팔꿈치 옆판이 사라진 적이 있다 — 지금은 원본 팔 메시를 팔꿈치 평면에서 자르기만 하고 쉘·팔꿈치 부품만 얹는다(157 §12.5).
- Isaac 의 쉘 URDF 에는 원래 모터가 시각 요소로 없었다(원본 MJCF 에만 원통 19 개) — URDF 에 같은 위치로 추가해 21 개(팔꿈치 2 포함).

## 6. 관련 파일

- 스크립트: `tools/isaac_demo/` (`warehouse_demo.py`, `walk_probe.py`, `selfhit_check.py`, `isaac_map.py`, `isaac_probe*.py`, `nav_layout_mockups.py`, `install_ros_rviz.sh`), `tools/isaac_demo/rviz/`
- 모델: `pygmalion_locomotion/assets/pygmalion_v2/*_shell*.urdf`, `meshes_v30_armfix_shell/`, 단일 숫자 출처 `tools/robot_model/shell_overlay/manifest.json`
- 센서 자료: `docs/datasheets/ouster/` (공식 PDF 3 개 — 라이다 원점 높이 38.195 mm, 세로 시야 ±21.2°)
- 정책 배포 묶음: `deploy_asset/pyg_v11a_stand01/` (입력 45, 출력 12, 50 Hz)
- 대화 기록 원본: `handoff_Rvizvisualize/c429e0b6-…jsonl` (git 제외, 100 MB)
