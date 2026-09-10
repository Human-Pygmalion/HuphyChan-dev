#!/usr/bin/env python3
"""**mjlab 로봇 모델 그 자체**에서 관절 하나를 꺼내 3D 로 그린다. 실물 사진 위에 얹기 위한 것.

2026-09-10 · 사용자 지시: **"mujocolab 이랑 겹치라는 말은, mujocolab 의 3d 렌더랑 겹치라는거야."**

앞 판은 제가 따로 만든 **대체 원판**을 그렸습니다. 모양은 비슷해도 그건 시뮬레이터가 실제로
푸는 그 물체가 아닙니다. 이 판은 학습에 쓰는 로봇 파일을 **그대로 열어서**, 그 안의 무릎(또는
엉덩이 회전) 관절과 그 모터를 꺼내 그립니다. 그러니 화면에 보이는 원통은 시뮬레이터가 정말
들고 있는 그 모터이고, 그 관절에 붙은 관성·마찰·점성도 그 파일에 적힌 값입니다.

무엇을 보이게 두는가
--------------------
* 모터 원통 하나 (예: ``L_knee_motor``) — 관절의 **고정 쪽**에 붙어 있습니다
* 관절의 **도는 쪽**에 달아 준 **질량 0 인 막대** — 원통은 아무리 돌려도 그림이 안 바뀌므로,
  실물에 나무 막대를 붙인 것과 같은 이유로 여기도 막대를 답니다
* 견주기 위한 막대 둘 — 실물이 실제로 간 각도(검정), 우리가 시킨 각도(주황)

나머지 다리·골반은 **투명하게** 만들어 안 보이게 합니다. 지우지 않고 투명하게만 하는 이유는,
파일을 손대지 않고 그대로 쓰기 위해서입니다.

어느 방향에서 보는가
--------------------
관절 축을 **정면으로** 내려다봅니다(축을 따라 보면 회전이 각도 그대로 보입니다). 사진도 거의
수직으로 내려다본 것이라 **직교 투영**을 씁니다.

화면에서 몇 도로 보이는가는 카메라가 어떻게 돌아 있느냐에 따라 달라집니다. 그래서 이 프로그램은
**자기가 그린 그림을 스스로 재서** 그 관계를 구합니다 — 막대를 0도와 90도에 놓고 그려 보고,
그림에서 막대가 어느 쪽을 가리키는지 재서 "장면 각도 -> 그림 각도" 식을 얻습니다. 그러면
"사진에서 이 각도로 보이게 하라"는 요구를 정확히 맞출 수 있습니다.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

DEG = math.pi / 180.0

# 어느 관절을 꺼낼 것인가. (관절 이름, 그 관절의 모터 원통 이름)
JOINTS = {
  "RS04": ("L_knee_joint", "L_knee_motor"),
  "RS03": ("L_hip_yaw_joint", "L_hip_yaw_motor"),
}

RGBA = {
  "sim": (0.00, 0.82, 0.55, 1.0),
  "real": (0.06, 0.06, 0.08, 1.0),
  "cmd": (1.00, 0.70, 0.10, 1.0),
  "motor": (0.42, 0.50, 0.62, 1.0),
}


def _perp(axis):
  """축에 수직인 단위 벡터 하나. 막대를 뻗을 방향이다."""
  a = np.asarray(axis, dtype=float)
  a = a / np.linalg.norm(a)
  t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
  u = np.cross(a, t)
  return u / np.linalg.norm(u)


class MjlabJointScene:
  """학습용 로봇 파일에서 관절 하나만 꺼내 그리는 장면."""

  def __init__(self, xml_path, family="RS04", size=900, ptr_reach=1.45, cmd_reach=1.62,
               ptr_width=0.010):
    joint_name, motor_geom = JOINTS[family]
    spec = mujoco.MjSpec.from_file(xml_path)
    jnt = spec.joint(joint_name)
    rotor = jnt.parent                    # 관절이 붙은 body = **도는 쪽**
    stator = rotor.parent                 # 그 부모 = **고정 쪽**, 모터 원통이 여기 있다
    mg = spec.geom(motor_geom)
    self.motor_r = float(mg.size[0])      # 모터 원통의 반지름 [m] — 크기를 맞출 기준
    axis = np.asarray(jnt.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    anchor = np.asarray(jnt.pos, dtype=float)
    u = _perp(axis)

    L = self.motor_r * ptr_reach
    Lc = self.motor_r * cmd_reach
    # 막대는 **캡슐**로 만든다. 캡슐은 두 점을 주면 그 사이를 잇기 때문에, 회전을 나타내는
    # 사원수를 손으로 계산할 필요가 없다.
    def bar(body, name, start, length, rad, rgba):
      g = body.add_geom()
      g.name = name
      g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
      p0 = anchor + axis * start
      p1 = p0 + u * length
      g.fromto = [*p0, *p1]
      g.size = [rad, 0.0, 0.0]
      g.rgba = rgba
      g.mass = 0.0
      g.contype = 0
      g.conaffinity = 0
      g.group = 0
      return g

    self.g_sim = bar(rotor, "bench_ptr_sim", self.motor_r * 0.55, L, ptr_width, RGBA["sim"])

    # 견주기용 막대 둘. 같은 자리·같은 축으로 도는 **따로 도는 몸통**을 고정 쪽에 붙인다.
    self.ghosts = []
    for nm, key, off, ln, rad in (("bench_real", "real", 0.80, L, ptr_width * 0.55),
                                  ("bench_cmd", "cmd", 1.05, Lc, ptr_width * 0.40)):
      b = stator.add_body()
      b.name = nm
      b.pos = rotor.pos
      b.quat = rotor.quat
      b.mass = 1e-6
      b.inertia = [1e-9, 1e-9, 1e-9]
      j = b.add_joint()
      j.name = nm + "_joint"
      j.type = mujoco.mjtJoint.mjJNT_HINGE
      j.pos = jnt.pos
      j.axis = jnt.axis
      j.limited = 0
      self.ghosts.append(bar(b, nm + "_ptr", self.motor_r * off, ln, rad, RGBA[key]))

    # 나머지는 **투명하게**. 파일은 손대지 않고 보이기만 끈다.
    keep = {motor_geom, "bench_ptr_sim", "bench_real_ptr", "bench_cmd_ptr"}
    for g in spec.geoms:
      if g.name not in keep:
        g.rgba = [0.0, 0.0, 0.0, 0.0]
    mg.rgba = RGBA["motor"]

    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.visual.global_.orthographic = 1
    # 직교 투영에서 `fovy` 는 각도가 아니라 **화면에 담을 세로 길이(미터)** 다. 기본값 45 를
    # 그대로 두면 45 미터를 담으려 해서 모터가 한 화소도 안 된다(처음에 그렇게 비어 나왔다).
    spec.visual.global_.fovy = float(mg.size[0]) * 6.0
    spec.visual.global_.offwidth = max(1400, size)
    spec.visual.global_.offheight = max(1400, size)
    spec.visual.headlight.diffuse = [0.75, 0.75, 0.75]
    spec.visual.headlight.ambient = [0.45, 0.45, 0.48]
    spec.visual.headlight.specular = [0.12, 0.12, 0.12]
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    self.size = size
    self.renderer = mujoco.Renderer(self.model, size, size)

    self.jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    self.q_sim = self.model.jnt_qposadr[self.jid]
    self.q_real = self.model.jnt_qposadr[
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bench_real_joint")]
    self.q_cmd = self.model.jnt_qposadr[
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bench_cmd_joint")]
    self.gid_motor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, motor_geom)
    self.gid_sim = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bench_ptr_sim")

    self.opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(self.opt)
    self.cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(self.cam)
    self._aim(axis)
    self.motor_px = self._measure_motor()
    self.img_of_scene = self._calibrate()

  # ------------------------------------------------------------------ 카메라
  def _aim(self, axis_local):
    """관절 축을 **정면으로 보는** 자리에 카메라를 놓는다."""
    self._pose(0.0, 0.0, 0.0)
    ax = np.asarray(self.data.xaxis[self.jid], dtype=float)
    ax = ax / np.linalg.norm(ax)
    f = -ax                                   # 카메라가 바라볼 방향
    self.cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
    self.cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
    self.cam.distance = 1.0
    self.cam.lookat[:] = self.data.xanchor[self.jid]

  def _pose(self, sim_deg, real_deg, cmd_deg):
    self.data.qpos[:] = self.model.qpos0
    self.data.qpos[self.q_sim] = sim_deg * DEG
    self.data.qpos[self.q_real] = real_deg * DEG
    self.data.qpos[self.q_cmd] = cmd_deg * DEG
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def _seg(self):
    self.renderer.enable_segmentation_rendering()
    self.renderer.update_scene(self.data, self.cam, self.opt)
    seg = self.renderer.render()[:, :, 0].copy()
    self.renderer.disable_segmentation_rendering()
    return seg

  def _measure_motor(self):
    """모터 원통이 **몇 화소 반지름**으로 그려지는지 잰다. 사진과 크기를 맞출 기준."""
    self._pose(0.0, 0.0, 0.0)
    seg = self._seg()
    xs = np.nonzero((seg == self.gid_motor).any(axis=0))[0]
    ys = np.nonzero((seg == self.gid_motor).any(axis=1))[0]
    return min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / 2.0

  def _calibrate(self):
    """**자기가 그린 그림을 스스로 재서** 장면 각도와 그림 각도의 관계를 구한다.

    카메라가 어떻게 돌아 있느냐에 따라 화면에서의 방향과 부호가 달라지므로, 문서로 따지는
    대신 두 각도(0도, 90도)에 놓고 그려 보고 잰다. 셋째 각도로 확인까지 한다."""
    def img_angle(scene_deg):
      self._pose(scene_deg, scene_deg, scene_deg)
      seg = self._seg()
      ys, xs = np.nonzero(seg == self.gid_sim)
      cy, cx = self.size / 2.0, self.size / 2.0
      # 회전 중심에서 가장 먼 쪽이 막대 끝이다
      d = (xs - cx) ** 2 + (ys - cy) ** 2
      sel = d >= np.percentile(d, 80)
      return math.degrees(math.atan2(ys[sel].mean() - cy, xs[sel].mean() - cx))
    a0, a90 = img_angle(0.0), img_angle(90.0)
    diff = (a90 - a0 + 180.0) % 360.0 - 180.0
    s = 1.0 if diff > 0 else -1.0             # 부호 (그림 좌표는 아래로 갈수록 y 가 크다)
    chk = img_angle(45.0)
    pred = a0 + s * 45.0
    err = (chk - pred + 180.0) % 360.0 - 180.0
    if abs(err) > 4.0 or abs(abs(diff) - 90.0) > 6.0:
      raise SystemExit(f"[mjlab_scene] 각도 눈금 확인 실패: 0도->{a0:.1f} 90도->{a90:.1f} "
                       f"45도->{chk:.1f} (예상 {pred:.1f})")
    return a0, s

  # ------------------------------------------------------------------ 그리기
  def scene_deg_for_img(self, img_deg):
    """"사진에서 이 각도로 보이게 하라" 를 장면 각도로 바꾼다."""
    a0, s = self.img_of_scene
    return (img_deg - a0) / s

  def render(self, sim_img_deg, real_img_deg, cmd_img_deg):
    """세 각도는 **사진에서 보일 각도**로 준다. (그림, 모터 자리, 막대 자리)"""
    self._pose(self.scene_deg_for_img(sim_img_deg), self.scene_deg_for_img(real_img_deg),
               self.scene_deg_for_img(cmd_img_deg))
    self.renderer.update_scene(self.data, self.cam, self.opt)
    rgb = self.renderer.render()
    seg = self._seg()
    return rgb, seg == self.gid_motor, (seg >= 0) & (seg != self.gid_motor)
