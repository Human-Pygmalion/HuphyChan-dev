---
title: Intel RealSense D455 URDF 센서 통합 가이드 (depth image 시뮬레이션용)
date: 2026-10-27
tags: [realsense, D455, URDF, MJCF, IsaacLab, MuJoCo, depth-camera, HUPHY, 설계자료]
---

## 요약

- 로봇 모델에 D455를 넣는 방법은 두 층으로 나누는 것이 안전합니다. URDF/MJCF에는 **프레임(링크+fixed joint)과 형상·질량**만 두고, **depth 센서 자체(해상도·FOV·clipping·출력 타입)는 시뮬레이터 설정에서 그 프레임에 붙입니다.** 이 구성은 아래 공식 문서들의 동작에서 도출한 설계 권고입니다.
- 필요한 입력은 세 가지입니다: ① 설계 원점(뒷판 M4 중점) 기준 프레임 위치, ② FOV에서 구한 intrinsics, ③ 시뮬레이터별 부착 설정.
- 사양·좌표계 근거는 [[Intel Realsense D455]]에 있고, 이 노트는 그 값을 URDF·시뮬 설정으로 옮기는 절차와 원본 자료 위치를 모았습니다.
- 값 중 **공칭값(데이터시트/URDF 기반)과 직접 읽어야 하는 값(실기 intrinsics/extrinsics)** 을 구분해서 적었습니다. 실기 값으로 교체하는 방법은 5장에 있습니다.

## 1. 원본 자료 위치

### 1.1 내 vault 안

| 자료 | 위치 | 내용 |
|---|---|---|
| D455 사양·좌표계 정리 | [[Intel Realsense D455]] | 치수, FOV, 원점, IMU 위치, 센서 위치표(3장), 설계 원점 변환 |
| 로봇 설계 노트 | [[Project Human Pygmalion - 설계 Scratchpad]] | 허리/어깨/Neck/URDF 완성 섹션, 끝에 "Intel Realsense D455 Pelvis 장착" 제목(본문 비어 있음) |
| 토르소 개정·배터리 | [[Human Pygmalion - Battery]] | Torso 2차 개정 등 Pelvis 주변 설계 변경 |
| 생성한 도면 이미지 | `attachFiles/D455_front_layout.svg`, `D455_top_view_depth_axis.svg`, `D455_fov_topview.svg` | 데이터시트 수치로 다시 그린 전면도·평면도·FOV |

### 1.2 로컬/저장소 (이번 세션에서 위치를 확인하지 못함)

- 현재 로봇 URDF/MJCF 파일 경로, Pelvis 링크 이름, 카메라 장착 위치·자세(CAD 값)는 vault와 대화에서 확인되지 않았습니다. 6장의 확인 목록에 넣었습니다.
- 사용자 메모상 소스 저장소는 `Human-Pygmalion/HUPHY`(biped 브랜치)이고 URDF/MJCF는 Fusion 360에서 ACDC4Robot·fusion2urdf로 내보내는 파이프라인입니다. 링크는 직접 확인하지 않았습니다.

### 1.3 웹 원본

| 자료 | 링크 | 이 노트에서 쓴 곳 |
|---|---|---|
| D400 데이터시트 (Doc 337029-017, 2026-03) | [PDF](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf) | p.65 FOV·baseline, p.67 치수·무게, p.75 스트림, p.84 Min-Z, p.89 Z 원점, p.91–92 Fig 4-8, p.142 Fig 10-12 |
| realsense-ros description 패키지 | [realsense2_description](https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description) | 공식 D455 URDF(xacro), STL 메시, 테스트 xacro |
| ↳ D455 xacro | [`urdf/_d455.urdf.xacro`](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d455.urdf.xacro) | 프레임 오프셋, 메시/충돌 상자 위치 |
| ↳ D455 메시 | [`meshes/d455.stl`](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/meshes/d455.stl) | visual 형상 (xacro에서 scale 0.001) |
| ↳ D455 테스트 xacro | [`urdf/test_d455_camera.urdf.xacro`](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/test_d455_camera.urdf.xacro) | 로봇에 붙이는 사용 예 |
| librealsense 좌표계 문서 | [doc/d435i.md](https://github.com/realsenseai/librealsense/blob/development/doc/d435i.md) | 축 규약, IMU 좌표 |
| Isaac Lab 카메라 개요 | [Camera 문서 (v2.3.2)](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/core-concepts/sensors/camera.html) | TiledCameraCfg 예시, data_types |
| Isaac Lab CameraCfg 소스 | [camera_cfg.py (main)](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py) | OffsetCfg convention |
| Isaac Lab Pinhole 소스 | [sensors_cfg.py (main)](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sim/spawners/sensors/sensors_cfg.py) | `PinholeCameraCfg`, `from_intrinsic_matrix` |
| Isaac Lab URDF 변환 옵션 | [converters API (v2.3.2)](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.converters.html) | `merge_fixed_joints` |
| Isaac Sim D455 에셋 | [Depth Sensors → RealSense D455](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_depth_sensors.html#realsense-d455-certified) | NVIDIA 제공 D455 모델 |
| MuJoCo camera 요소 | [XML Reference – body/camera](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera) | 축 규약, fovy, depth 출력 |
| MuJoCo URDF 정적 링크 병합 | [compiler/fusestatic](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-fusestatic) | URDF 가져올 때 프레임 링크 처리 |

## 2. URDF에 넣을 프레임과 형상

### 2.1 프레임 트리

```
pelvis (실제 Pelvis 링크명으로 교체)
 └─ d455_mount            설계 원점 = 뒷판 M4 두 홀의 중점 (CAD에서 측정한 장착 포즈)
     └─ d455_link         depth 프레임 = Left IR 중심, ROS body 축 (x 전, y 좌, z 상)
         ├─ d455_depth_optical_frame   rpy(−π/2, 0, −π/2): x 우, y 하, z 전
         ├─ d455_color_frame / _optical_frame
         ├─ d455_infra2_frame (Right IR)
         └─ d455_imu_frame
```

realsense-ros는 `camera_link`를 depth 프레임으로 두고, optical 프레임은 rpy(−π/2, 0, −π/2)로 만들며, 공칭 오프셋은 infra1 0, infra2 −0.095 m, color −0.059 m(y축)입니다. [^urdf]

### 2.2 오프셋 값 (설계 원점 `d455_mount` 기준, ROS body 축, m)

`d455_mount → d455_link` = (0.02135, 0.0475, 0), 회전 0. 이 값은 [[Intel Realsense D455]] 3장 위치표에서 가져온 것이며, M4 홀 중점이 본체 좌우·높이 중앙이라는 가정이 들어 있습니다. [^ds]

| 프레임 | 부모 | xyz (m) | 비고 |
|---|---|---|---|
| `d455_link` | `d455_mount` | 0.02135 0.0475 0 | 뒷면에서 Z=0 면까지 21.35 mm [^ds] |
| `d455_color_frame` | `d455_link` | 0 −0.059 0 | 공칭값, 실제는 extrinsic으로 교체 [^urdf][^ds] |
| `d455_infra2_frame` | `d455_link` | 0 −0.095 0 | baseline 95 mm [^ds] |
| `d455_imu_frame` | `d455_link` | −0.01602 −0.03022 0.0074 | 도면 IMU 표기와 xacro가 일치 [^urdf][^ds] |
| `*_optical_frame` | 각 frame | 0 0 0, rpy −π/2 0 −π/2 | [^urdf] |

### 2.3 URDF 스니펫

```xml
<!-- 설계 원점: 뒷판 M4 두 홀 중점. 부모 링크와 origin은 CAD 값으로 교체 -->
<link name="d455_mount"/>
<joint name="pelvis_to_d455_mount" type="fixed">
  <parent link="pelvis"/>
  <child  link="d455_mount"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>

<link name="d455_link">
  <inertial>
    <!-- 데이터시트 116 g, 균일 상자 근사 (아래 주석 참고) -->
    <origin xyz="-0.00835 -0.0475 0" rpy="0 0 0"/>
    <mass value="0.116"/>
    <inertia ixx="1.57e-4" iyy="1.47e-5" izz="1.55e-4" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <origin xyz="0.00465 -0.0475 0" rpy="1.5707963 0 1.5707963"/>
    <geometry><mesh filename="meshes/d455.stl" scale="0.001 0.001 0.001"/></geometry>
  </visual>
  <collision>
    <origin xyz="-0.00835 -0.0475 0" rpy="0 0 0"/>
    <geometry><box size="0.026 0.124 0.029"/></geometry>
  </collision>
</link>
<joint name="d455_mount_to_link" type="fixed">
  <parent link="d455_mount"/>
  <child  link="d455_link"/>
  <origin xyz="0.02135 0.0475 0" rpy="0 0 0"/>
</joint>

<link name="d455_depth_optical_frame"/>
<joint name="d455_depth_optical_joint" type="fixed">
  <parent link="d455_link"/>
  <child  link="d455_depth_optical_frame"/>
  <origin xyz="0 0 0" rpy="-1.5707963 0 -1.5707963"/>
</joint>

<link name="d455_color_frame"/>
<joint name="d455_color_joint" type="fixed">
  <parent link="d455_link"/>
  <child  link="d455_color_frame"/>
  <origin xyz="0 -0.059 0" rpy="0 0 0"/>
</joint>
<link name="d455_color_optical_frame"/>
<joint name="d455_color_optical_joint" type="fixed">
  <parent link="d455_color_frame"/>
  <child  link="d455_color_optical_frame"/>
  <origin xyz="0 0 0" rpy="-1.5707963 0 -1.5707963"/>
</joint>
```

- 메시 visual 원점은 xacro 값(전면 플레이트 +4.65 mm, 좌우 −47.5 mm, rpy(π/2, 0, π/2), scale 0.001)을 옮긴 것입니다. [^urdf]
- collision 상자는 xacro와 같은 크기(0.026×0.124×0.029)입니다. 위치는 xacro가 유리 기준(+4.55 mm)으로 중심을 잡은 것과 달리 전면 플레이트(+4.65)·후면(−21.35) 중점인 −8.35 mm를 썼습니다 (0.1 mm 차이). [^urdf][^ds]
- 질량은 xacro의 0.072 kg이 아니라 데이터시트 116 g(±10 %)을 썼습니다. xacro 주석이 자체 질량·관성 값을 "신뢰할 수 없다"고 밝혔기 때문입니다. [^urdf][^ds]
- 관성은 균일 상자 근사로 제가 계산한 값입니다 (Ixx = m(w²+h²)/12 등, 폭 0.124·깊이 0.026·높이 0.029 m).
- Pelvis CAD에 이미 카메라 질량이 들어 있다면 이중으로 세지 않도록 `d455_link`의 inertial을 빼거나 Pelvis 쪽에서 빼야 합니다.
- IMU, infra2 프레임은 필요할 때 같은 방식으로 추가합니다 (2.2 표).

## 3. depth 센서 파라미터

### 3.1 FOV → intrinsics (공칭 근사)

데이터시트 값은 Depth H 87° / V 58° (±3°, 2 m 기준), 색 H 90° / V 65°입니다. [^ds] 아래는 정사각 픽셀·주점 중앙을 가정하고 `fx = (W/2) / tan(HFOV/2)`로 제가 계산한 근사값입니다.

| 모드 | HFOV | fx = fy (px) | cx, cy | 정사각 픽셀 가정 시 VFOV |
|---|---|---|---|---|
| Depth 848×480 | 87° | 446.8 | 424, 240 | 약 56.5° |
| Depth 1280×720 | 87° | 674.4 | 640, 360 | 약 56.2° |
| RGB 1280×800 | 90° | 640.0 | 640, 400 | — |

- 가능한 depth 스트림은 1280×720이 5/15/30 fps, 848×480 이하는 최대 90 fps입니다. [^ds]
- 정사각 픽셀로 계산한 VFOV(56.5°)가 스펙 V 58°보다 작지만 ±3° 허용 안입니다. 시뮬은 HFOV 87°를 맞추는 쪽으로 잡았습니다.
- 실제 intrinsics는 개체별이므로 5장 방법으로 읽어서 교체해야 합니다.

### 3.2 clipping·출력 타입 (설계 선택)

| 항목 | 값 | 근거/비고 |
|---|---|---|
| near | 0.35 m (848×480), 0.52 m (1280×720) | 해상도별 Min-Z [^ds] |
| far | 6 m | 권장 동작 범위 0.6–6 m [^prod] |
| 출력 | 이미지 평면 기준 Z 거리 | Isaac Lab `distance_to_image_plane`(= `depth` 별칭), MuJoCo `depth` [^ilcam][^mjcam] |

- Min-Z는 광학 한계가 아니라 스테레오 매칭의 최소 거리이므로, 시뮬에서 near를 작게 두고 후처리로 Min-Z 안쪽을 무효 처리하는 방법도 가능합니다. 어느 쪽을 쓸지는 설계 선택입니다.
- 실기 `Z16` 포맷이 Z 거리라는 점에 맞춰 평면 거리를 골랐습니다. 이는 포맷 이름에서 한 해석이고, 실기 값과 직접 비교해 확인하세요. [^ds]
- 시뮬은 미터 단위 float로 내보내므로, 실기 파이프라인과 맞추려면 실기 depth scale로 변환할지 정해야 합니다.

## 4. 시뮬레이터별 부착

### 4.1 Isaac Lab

- 카메라 오프셋 convention은 세 가지입니다. `ros`는 전방 +Z·위 −Y, `opengl`은 전방 −Z·위 +Y, `world`는 전방 +X·위 +Z입니다. [^ilcfg]
- `d455_link`는 x 전방·z 위(ROS body)이므로 **`convention="world"`** 와 같은 축입니다. 따라서 오프셋은 회전 없이 위치만 주면 됩니다. [^ilcfg][^urdf]
- 쿼터니언 순서가 버전에 따라 다릅니다. main의 `camera_cfg.py` 주석은 (x, y, z, w)이고, v2.3.2 문서 예시 `rot=(0.9945, 0.0, 0.1045, 0.0)`는 (w, x, y, z) 형태입니다. 사용 중인 버전에서 항등 회전을 어떻게 쓰는지 먼저 확인하세요. [^ilcfg][^ilcam]
- `PinholeCameraCfg.from_intrinsic_matrix`로 intrinsics를 직접 줄 수 있습니다. 행렬은 `[fx,0,cx, 0,fy,cy, 0,0,1]` 순서이고 `width`, `height`, `clipping_range`를 함께 받습니다. [^ilpin]
- URDF → USD 변환에서 `merge_fixed_joints` 기본값이 True이고 "fixed joint로 연결된 링크를 통합"한다고 적혀 있습니다. 질량 없는 프레임 링크(`d455_link` 등)는 병합돼 USD에 별도 prim이 없을 수 있으므로, **카메라는 병합되지 않는 부모 링크(Pelvis) 아래에 오프셋으로 붙이는 방식이 안전합니다.** 병합 여부는 변환 후 USD에서 확인하세요 (이 부분은 문서의 한 줄 설명에서 도출한 판단입니다). [^ilurdf]

```python
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg   # 사용 버전의 import 경로 확인

fx = fy = 446.8; cx, cy = 424.0, 240.0       # 3.1 표의 공칭 근사 → 실기 값으로 교체
d455_depth = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/<pelvis 링크 prim>/d455_depth",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(PX, PY, PZ),          # pelvis 기준: mount 포즈 ∘ (0.02135, 0.0475, 0)
        rot=IDENTITY,              # 버전별 쿼터니언 순서 주의
        convention="world",
    ),
    data_types=["distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
        width=848, height=480, clipping_range=(0.35, 6.0),
    ),
    width=848, height=480,
)
```

위 코드는 문서·소스에서 확인한 필드명으로 구성했지만 이번 세션에서 실행해 보지는 않았습니다. NVIDIA가 제공하는 D455 에셋도 있습니다 (Isaac Sim 문서 "Realsense D455 (Certified)"). 다만 그 문서의 치수가 124×25×29 mm(깊이 25)로 적혀 데이터시트의 26 mm와 달라서, 기구 간섭 계산에는 데이터시트 값을 쓰고 에셋 경로는 별도로 확인해야 합니다. [^isimd455]

### 4.2 MuJoCo (MJCF)

- MuJoCo 카메라는 프레임의 −Z 방향을 보고 +X가 오른쪽, +Y가 위입니다. ROS optical(z 전방)과는 다릅니다. [^mjcam]
- `fovy`는 **수직** FOV(도)이고 가로는 해상도 비율로 자동 계산됩니다. `resolution`은 값을 저장하는 용도일 뿐 렌더링 크기를 정하지 않습니다. `output`에 `depth`를 적으면 "카메라 평면에서의 거리" 이미지를 뜻합니다. [^mjcam]
- 더 정확한 intrinsics가 필요하면 `sensorsize`를 주고 `focalpixel`, `principalpixel`을 쓰며, 이때 `fovy`는 무시됩니다. [^mjcam]
- URDF를 MuJoCo로 읽으면 `fusestatic` 기본값이 true라서, 참조되지 않은 정적 body는 부모에 병합됩니다. 카메라는 URDF 링크에서 만들어지지 않으므로 **변환된 MJCF의 Pelvis body 안에 `<camera>`를 직접 추가**합니다. [^mjfuse]
- ROS body 축(x 전, y 좌, z 상) 기준으로 카메라를 정면(+x)으로 보게 하려면 오른쪽 = −y, 위 = +z이므로 `xyaxes="0 -1 0 0 0 1"`입니다. `xyaxes`의 X·Y축이 픽셀 공간의 "오른쪽"과 "위"에 해당한다는 문서 설명에서 도출했고, 렌더링으로 방향을 확인하세요. [^mjcam]

```xml
<body name="pelvis">
  ...
  <!-- pos = pelvis 기준 mount 포즈 ∘ (0.02135, 0.0475, 0) -->
  <camera name="d455_depth" pos="PX PY PZ" xyaxes="0 -1 0 0 0 1"
          fovy="56.5" resolution="848 480" output="depth"/>
</body>
```

`fovy="56.5"`는 3.1 표에서 HFOV 87°·848×480을 유지하는 정사각 픽셀 값입니다. MuJoCo Python 쪽 depth 렌더링 API는 이번 세션에서 문서를 확인하지 못했으므로, 사용하는 MuJoCo 버전의 Python 문서에서 확인하세요.

## 5. 실기 값으로 교체·검증

1. 실기 D455에서 intrinsics/extrinsics를 읽어 3.1·2.2의 공칭값을 교체합니다. `get_extrinsics_to`, `get_intrinsics` 사용 예는 [[Intel Realsense D455]] 5장에 있습니다 (실행 미검증).
2. 프레임 방향 확인: 시뮬 카메라 프레임을 축으로 그려 +x 우, +y 하, +z 전방(optical)이 맞는지 봅니다. [^lrs]
3. 거리 확인: 평평한 벽을 알려진 거리에 두고 실기와 시뮬의 같은 픽셀 depth를 비교합니다 (Min-Z 이상 거리에서).
4. 화각 확인: 폭을 아는 물체를 같은 거리에 두고 양쪽 영상에서 차지하는 픽셀 폭을 비교합니다.
5. 형상 확인: `d455.stl`을 올려 놓고 본체 윤곽이 Pelvis CAD의 카메라 홈/브래킷과 맞는지 봅니다. 메시 라이선스는 realsense-ros 저장소 설명을 확인하세요.

## 6. 확인 필요

- 현재 URDF/MJCF 파일 경로, Pelvis 링크 이름, `d455_mount`의 부모 기준 위치·자세 (CAD에서 측정)
- M4 홀 중점이 본체 좌우·높이 중앙이라는 가정 (STEP으로 확인)
- 사용 중인 Isaac Lab/MuJoCo 버전과 그에 맞는 쿼터니언 순서·import 경로
- 실기 intrinsics/extrinsics, depth scale
- 프로젝터·RGB 렌즈의 z 위치 (데이터시트에서 확인하지 못함)
- Isaac Sim D455 에셋 경로와 내부 FOV 설정

## 출처

[^ds]: RealSense, *RealSense D400 Series Datasheet* (Doc 337029-017, 2026-03): p.65 Table 3-49 (FOV, baseline), p.67 Table 3-53 (치수·무게), p.75 Table 4-3 (스트림 모드), p.84 Table 4-12 (Min-Z), p.89 Table 4-17 (Z′ = −4.55 mm), p.91 Fig 4-8 및 p.92 Table 4-21, p.142 Fig 10-12 (IMU 위치). https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf
[^urdf]: RealSense, *realsense-ros* `realsense2_description/urdf/_d455.urdf.xacro` (ros2-master): 프레임 오프셋, optical rpy, 메시·충돌 원점, 질량/관성 주석. https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d455.urdf.xacro
[^lrs]: RealSense, *librealsense* `doc/d435i.md`, "Sensor origin and coordinate system" (+x 우, +y 하, +z 전방). https://github.com/realsenseai/librealsense/blob/development/doc/d435i.md
[^prod]: RealSense, "D455" 제품 페이지 (Ideal Range 0.6–6 m, 깊이 오차 < 2 % @ 4 m). https://www.realsenseai.com/products/real-sense-depth-camera-d455f/
[^ilcam]: Isaac Lab 문서 v2.3.2, "Camera" (TiledCameraCfg 예시, data_types: `distance_to_camera`, `distance_to_image_plane`, `depth`는 `distance_to_image_plane`의 별칭). https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/core-concepts/sensors/camera.html
[^ilcfg]: Isaac Lab 소스 `camera_cfg.py` (main): `OffsetCfg.convention`(opengl/ros/world 축 정의), `rot` (x, y, z, w). https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py
[^ilpin]: Isaac Lab 소스 `sensors_cfg.py` (main): `PinholeCameraCfg`, `from_intrinsic_matrix`. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sim/spawners/sensors/sensors_cfg.py
[^ilurdf]: Isaac Lab 문서 v2.3.2, `UrdfConverterCfg` (`merge_fixed_joints` 기본 True, "Consolidate links that are connected by fixed joints."). https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.converters.html
[^isimd455]: NVIDIA, Isaac Sim 문서 "Depth Sensors – Realsense D455 (Certified)". https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_depth_sensors.html#realsense-d455-certified
[^mjcam]: MuJoCo 문서 "XML Reference – body/camera" (−Z 방향 시선, +X 우, +Y 상, `fovy` 수직 FOV, `resolution`, `output` depth = 카메라 평면 거리, `sensorsize`/`focalpixel`/`principalpixel`, `xyaxes` 의미). https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
[^mjfuse]: MuJoCo 문서 "XML Reference – compiler/fusestatic" (URDF 기본 true, 참조되지 않은 정적 body는 부모에 병합). https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-fusestatic
