#!/usr/bin/env python3
"""영상에 쓸 **모터 한 대짜리 MuJoCo 장면**. 진짜 3D 렌더러로 그린다.

2026-09-10. ★ 앞선 기록 정정: "이 컴퓨터에 오프스크린 OpenGL 이 없다"고 적었는데 **틀렸습니다.**
`MUJOCO_GL=egl` 로 잘 됩니다(OpenGL 4.6 NVIDIA). 실패로 본 것은 출력 꼬리만 보고 판단한
것이었고, 그 EGL 오류는 **렌더가 끝난 뒤 소멸자에서 나는 뒷정리 잡음**이었습니다.

장면 구성
---------
* 회색 원통 = 모터 몸통 (고정)
* 은색 원판 + **초록 막대** = MuJoCo 가 계산한 각도로 도는 축 (실물에 붙인 나무 막대에 대응)
* 검정 막대 = 실물이 실제로 간 각도 (기록에서 읽어 그대로 돌림)
* 회색 막대 = 우리가 시킨 목표 각도

세 막대는 높이를 달리해 서로 안 가립니다. **전부 질량 0** 이라 물리에 안 들어갑니다.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

DEG = math.pi / 180.0

SCENE = """
<mujoco model="bench_motor">
  <compiler angle="radian" autolimits="true"/>
  <option><flag contact="disable" gravity="disable"/></option>
  <visual>
    <global offwidth="1200" offheight="1200" azimuth="90" elevation="-46"/>
    <headlight diffuse="0.30 0.30 0.32" ambient="0.40 0.40 0.44" specular="0.08 0.08 0.08"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.02"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.95 0.97"
             rgb2="0.82 0.84 0.88" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.88"
             rgb2="0.80 0.80 0.83" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="14 14" reflectance="0.05"/>
    <material name="body" rgba="0.42 0.43 0.47 1" specular="0.4" shininess="0.4"/>
    <material name="flange" rgba="0.78 0.79 0.82 1" specular="0.75" shininess="0.7"/>
    <material name="ptr_sim" rgba="0.00 0.66 0.47 1" specular="0.3"/>
    <material name="ptr_real" rgba="0.10 0.10 0.12 0.92"/>
    <material name="ptr_cmd" rgba="0.58 0.58 0.60 0.85"/>
  </asset>
  <worldbody>
    <light pos="0.30 -0.45 0.55" dir="-0.4 0.6 -1" directional="true"
           diffuse="0.32 0.32 0.34" specular="0.10 0.10 0.10"/>
    <geom name="floor" type="plane" size="0.6 0.6 0.02" pos="0 0 -0.062" material="floor"/>
    <geom name="case" type="cylinder" size="0.088 0.030" pos="0 0 -0.032" material="body"/>
    <body name="shaft">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j_sim" type="hinge" axis="0 0 1" limited="false"/>
      <geom type="cylinder" size="0.086 0.006" pos="0 0 0" material="flange" mass="0"/>
      <geom type="cylinder" size="0.030 0.009" pos="0 0 0.004" material="body" mass="0"/>
      <geom name="ptr_sim" type="box" size="0.062 0.011 0.007" pos="0.062 0 0.018"
            material="ptr_sim" mass="0"/>
      <geom type="box" size="0.011 0.011 0.007" pos="0 0 0.018" material="ptr_sim" mass="0"/>
    </body>
    <body name="ghost_real">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j_real" type="hinge" axis="0 0 1" limited="false"/>
      <geom type="box" size="0.046 0.008 0.005" pos="0.046 0 0.036" material="ptr_real" mass="0"/>
    </body>
    <body name="ghost_cmd">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j_cmd" type="hinge" axis="0 0 1" limited="false"/>
      <geom type="box" size="0.078 0.005 0.004" pos="0.078 0 0.050" material="ptr_cmd" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class MotorScene:
  """세 각도를 넣으면 그 자세의 그림을 돌려준다."""

  def __init__(self, width=560, height=560, azimuth=90.0, elevation=-46.0,
               distance=0.352, lookat=(0.0, 0.0, 0.012)):
    self.model = mujoco.MjModel.from_xml_string(SCENE)
    self.data = mujoco.MjData(self.model)
    self.renderer = mujoco.Renderer(self.model, height, width)
    self.cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(self.cam)
    self.cam.azimuth, self.cam.elevation = azimuth, elevation
    self.cam.distance = distance
    self.cam.lookat[:] = lookat
    self.opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(self.opt)

  def render(self, sim_deg, real_deg, cmd_deg):
    self.data.qpos[:] = [sim_deg * DEG, real_deg * DEG, cmd_deg * DEG]
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)
    self.renderer.update_scene(self.data, self.cam, self.opt)
    return self.renderer.render()


if __name__ == "__main__":
  import os
  from PIL import Image
  os.environ.setdefault("MUJOCO_GL", "egl")
  sc = MotorScene()
  Image.fromarray(sc.render(25.0, 21.0, 30.0)).save("/tmp/scene_test.png")
  print("그림: /tmp/scene_test.png")


# =====================================================================================
# 실물 사진 **위에 겹쳐 그리기** 위한 장면 (2026-09-10, 사용자 지시:
# "Sim / Real 을 같은 스케일로, 겹쳐서 보이게 알파 줘서 시각화하고")
# =====================================================================================
#
# 겹쳐 놓으려면 두 가지가 맞아야 합니다.
#
#   ① 투영 방식 — 사진은 위에서 거의 수직으로 내려다본 것입니다. 그래서 시뮬레이터도
#      **직교 투영**(멀고 가까움에 따라 크기가 안 변하는 방식)으로 위에서 내려다봅니다.
#      원근 투영이면 가장자리가 벌어져 아무리 크기를 맞춰도 안 겹칩니다.
#   ② 크기 — 시뮬레이터 원판의 반지름이 **사진 속 원판 반지름과 같은 화소 수**가 되게
#      그림을 늘리거나 줄입니다. 그 배율은 프로그램이 시작할 때 한 번 재서 씁니다.
#      (직교 투영에서 화면 크기가 어떻게 정해지는지 문서로 따지는 대신, **직접 재는** 편이
#      확실합니다.)
#
# 배경은 남기지 않습니다. MuJoCo 의 물체 구분 그림(segmentation)으로 "물체가 있는 화소"만
# 골라내어, 그 자리에만 알파를 주고 얹습니다.

OVERLAY = """
<mujoco model="overlay_motor">
  <compiler angle="radian" autolimits="true"/>
  <option><flag contact="disable" gravity="disable"/></option>
  <visual>
    <global offwidth="1400" offheight="1400" orthographic="true" fovy="0.30"/>
    <headlight diffuse="0.78 0.78 0.78" ambient="0.48 0.48 0.50" specular="0.15 0.15 0.15"/>
    <quality offsamples="8"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <material name="disc" rgba="0.55 0.62 0.78 1"/>
    <material name="p_sim" rgba="0.00 0.80 0.55 1" specular="0.25"/>
    <material name="p_real" rgba="0.06 0.06 0.08 1"/>
    <material name="p_cmd" rgba="1.0 0.72 0.10 1"/>
  </asset>
  <worldbody>
    <body name="ring">
      <geom name="disc" type="cylinder" size="{disc_r} 0.001" pos="0 0 -0.006"
            material="disc" mass="0"/>
    </body>
    <body name="shaft">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j_sim" type="hinge" axis="0 0 1" limited="false"/>
      <geom name="ptr_sim" type="box" size="{sim_half} 0.011 0.004" pos="{sim_half} 0 0.006"
            material="p_sim" mass="0"/>
      <geom name="hub_sim" type="cylinder" size="0.017 0.005" pos="0 0 0.006"
            material="p_sim" mass="0"/>
    </body>
    <body name="ghost_real">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j_real" type="hinge" axis="0 0 1" limited="false"/>
      <geom name="ptr_real" type="box" size="{sim_half} 0.005 0.003" pos="{sim_half} 0 0.016"
            material="p_real" mass="0"/>
    </body>
    <body name="ghost_cmd">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-9 1e-9 1e-9"/>
      <joint name="j_cmd" type="hinge" axis="0 0 1" limited="false"/>
      <geom name="ptr_cmd" type="box" size="{cmd_half} 0.003 0.002" pos="{cmd_half} 0 0.024"
            material="p_cmd" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""

DISC_R = 0.086          # 시뮬레이터 원판 반지름 (미터). 사진 속 반지름과 맞출 기준이 된다.


class OverlayScene:
  """사진 **위에 얹을** 그림을 만든다. 배경은 안 돌려주고, 물체가 있는 자리만 알려준다.

  돌려주는 것은 셋입니다 — 그림, **원판 자리**, **막대 자리**. 원판은 옅게, 막대는 진하게
  얹으려고 자리를 따로 알려줍니다. 원판까지 진하게 얹으면 정작 봐야 할 실물 모터가 가려집니다.
  """

  def __init__(self, size=900, ptr_reach=1.45, cmd_reach=1.62):
    """``ptr_reach`` 는 **원판 반지름의 몇 배**까지 막대가 뻗는가. 실물 사진에서 잰 값을
    넣으면 시뮬레이터 막대가 실물 막대와 같은 길이로 그려져, 겹칠 때 바로 견줘집니다."""
    xml = OVERLAY.format(disc_r=DISC_R, sim_half=DISC_R * ptr_reach / 2.0,
                         cmd_half=DISC_R * cmd_reach / 2.0)
    self.model = mujoco.MjModel.from_xml_string(xml)
    self.data = mujoco.MjData(self.model)
    self.size = size
    self.renderer = mujoco.Renderer(self.model, size, size)
    self.cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(self.cam)
    self.cam.azimuth, self.cam.elevation = 90.0, -90.0   # 바로 위에서 내려다본다
    self.cam.distance = 0.6
    self.cam.lookat[:] = (0.0, 0.0, 0.0)
    self.opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(self.opt)
    self.disc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "disc")
    self.disc_px = self._measure_disc()

  def _pose(self, sim_deg, real_deg, cmd_deg):
    self.data.qpos[:] = [sim_deg * DEG, real_deg * DEG, cmd_deg * DEG]
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def _seg(self):
    self.renderer.enable_segmentation_rendering()
    self.renderer.update_scene(self.data, self.cam, self.opt)
    seg = self.renderer.render()[:, :, 0].copy()
    self.renderer.disable_segmentation_rendering()
    return seg

  def _measure_disc(self):
    """원판이 **몇 화소 반지름**으로 그려지는지 직접 잰다.

    직교 투영에서 화면 크기가 어떤 식으로 정해지는지 문서로 따지는 대신 재는 편이 확실합니다.
    이 값과 사진 속 반지름의 비가 곧 그림을 늘릴 배율입니다."""
    self._pose(0.0, 0.0, 0.0)
    seg = self._seg()
    xs = np.nonzero((seg == self.disc_id).any(axis=0))[0]
    return (xs.max() - xs.min() + 1) / 2.0

  def render(self, sim_deg, real_deg, cmd_deg):
    self._pose(sim_deg, real_deg, cmd_deg)
    self.renderer.update_scene(self.data, self.cam, self.opt)
    rgb = self.renderer.render()
    seg = self._seg()
    return rgb, seg == self.disc_id, (seg >= 0) & (seg != self.disc_id)
