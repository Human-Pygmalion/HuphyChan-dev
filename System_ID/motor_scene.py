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
