#!/usr/bin/env python3
"""**서기 정책을 브라우저에서 직접 눌러 보고 판정한다.** 배포용 코드 그대로 돌린다.

2026-09-21 · 사용자 지시: *"viser 띄워서 내가 검증할 수 있게 준비해줘."*

## 무엇이 다른가 — 보이는 것이 곧 실물에 올라갈 코드입니다

평소 재생 뷰어는 학습 때 쓰던 **PyTorch 정책**을 그대로 부릅니다. 그러면 "시뮬레이터
안에서는 잘 선다" 까지만 압니다. 이 뷰어는 대신 **로봇에 올릴 코드 경로**로 돕니다.

    센서값 → obs_adapter (numpy) → 정책.onnx → 관절 목표각 → 시뮬레이터

즉 `tools/deploy/policy_runner.py` 가 라즈베리파이에서 할 일을 여기서 그대로 합니다.
화면이 멀쩡하면 **그 코드가 멀쩡한 것**입니다. 두 경로가 같은 관측을 낸다는 것은
`verify_obs_adapter.py` 가 이미 200 스텝 최대 차이 0.000e+00 으로 확인했고, 이 뷰어는
그 위에 "그래서 실제로 서 있나" 를 사람 눈으로 보태는 자리입니다.

## 손으로 해 볼 수 있는 것

* **명령** — 앞뒤·좌우·돌기 속도. 서기 판정은 전부 0 에서 봅니다.
* **밀기** — 버튼을 누르면 몸통에 그 세기의 속도를 한 번 꽂습니다. 측정 도구
  (`stand_robustness.py`)가 쓰는 것과 **같은 함수**라 숫자와 눈이 어긋나지 않습니다.
* **들어올리기** — 자리만 붙잡고 자세는 놓아 줍니다. 사람이 몸통을 들고 있는 것과
  같습니다. 2026-09-17 에 사용자가 본 *"공중에 들면 다리가 떨린다"* 가 이 상태입니다.
* **관측 잡음** — 학습 때 켜 두는 센서 잡음. 켜야 실물과 같은 조건입니다. 떨림은
  잡음이 있어야 나타나므로, **끄고 보면 안 떨리는 것처럼 보입니다.**

## 오른쪽 숫자 네 개

정책을 판정하는 양입니다. 그냥 보기 좋으라고 띄운 것이 아닙니다.

* **기울기** — 똑바로 선 상태에서 몇 도 기울었나.
* **떨림** — 관절속도 크기의 합. 가만히 서 있는데 이 값이 크면 그게 떨림입니다.
* **토크 RMS** — 서 있느라 쓰는 힘. 작을수록 덜 뜨겁습니다.
* **자세오차** — 기준자세에서 몇 도 벗어나 서 있나. CAPS 정책이 이 값에서 대가를
  치르고 있어(v7b 0.64° 대 caps 1.82°) 특히 봐야 합니다.

## 띄우는 법

    .venv/bin/python3 tools/robot_model/viser_stand_viewer.py \
        --contract <deploy_contract_*.json> --onnx <정책.onnx>

주소는 `0.0.0.0` 에 붙습니다 — 다른 PC 에서 봅니다. 뜨면 LAN 주소를 찍어 줍니다.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/robot_model"))
sys.path.insert(0, str(REPO / "tools/deploy"))

from airborne_signature import apply_run_env, build, zero_command   # noqa: E402


def lan_address() -> str:
  """다른 PC 에서 칠 주소. 밖으로 패킷을 보내지는 않는다(UDP 연결은 주소만 고른다)."""
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("192.168.20.1", 1))
    return s.getsockname()[0]
  except OSError:
    return socket.gethostbyname(socket.gethostname())
  finally:
    s.close()


class DeployDriver:
  """시뮬레이터에서 **로봇이 읽을 수 있는 값만** 꺼내 배포 코드에 넣는다."""

  def __init__(self, env, contract_path, onnx_path):
    import mujoco

    from policy_runner import StandingPolicy                       # noqa: E402

    self.pol = StandingPolicy.from_files(contract_path, onnx_path,
                                         warn_period=False)
    self.pol.self_test()
    self.env = env
    self.robot = env.scene["robot"]
    m = env.sim.mj_model
    self.sens = {}
    for t in self.pol.adapter.terms:
      sn = t.get("sensor_name")
      if not sn:
        continue
      sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sn)
      if sid < 0:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sn.split("/")[-1])
      if sid < 0:
        raise SystemExit(f"센서 {sn} 를 모형에서 못 찾았습니다")
      self.sens[t["term"]] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))
    alljoints = list(self.robot.joint_names)
    self.jidx = [alljoints.index(j) for j in self.pol.adapter.hist_joints]
    self.act_jidx = [alljoints.index(j) for j in self.pol.joint_names]

    # ★ 센서 잡음은 **우리가 직접 넣어야 합니다.**
    #
    # 2026-09-21 · 처음 띄웠을 때 떨림이 0.000 으로 나왔습니다. 배선이 틀린 것이
    # 아니라, 이 뷰어가 **관측 관리자를 건너뛰기 때문**입니다. mjlab 에서 관측 잡음은
    # 센서가 아니라 관측 관리자가 얹습니다. 우리는 실물처럼 센서를 직접 읽으므로
    # 완벽하게 깨끗한 값을 받고, 정책은 떨 이유가 없습니다.
    #
    # 그런데 **떨림은 잡음이 있어야 나타납니다**(docs/148). 잡음 없이 보면 어떤
    # 정책도 안 떠는 것처럼 보이고, 그건 실물에 대해 아무 말도 못 하는 화면입니다.
    # 그래서 학습이 쓰던 **그 잡음 설정 그대로**를 센서값에 얹습니다 — 우리가 지어낸
    # 잡음이 아니라 같은 객체입니다.
    self.noise = {}
    for name, t in env.cfg.observations["actor"].terms.items():
      if getattr(t, "noise", None) is not None:
        self.noise[name] = t.noise
    self.noise_on = True
    self.reset()

  def _noisy(self, name, arr, torch):
    """학습과 같은 잡음 모형을 값에 얹는다. 설정이 없으면 그대로 돌려준다."""
    cfg = self.noise.get(name)
    if cfg is None or not self.noise_on:
      return arr
    x = torch.as_tensor(np.asarray(arr, dtype=np.float32)).unsqueeze(0)
    return np.asarray(cfg.apply(x)[0].cpu(), dtype=np.float64)

  def reset(self) -> None:
    self.pol.reset()

  def _slice(self, term):
    a, d = self.sens[term]
    return slice(a, a + d)

  def step(self, command, torch) -> np.ndarray:
    """한 주기. 정책이 낸 **날 것의 행동** 을 낸다 (시뮬레이터가 먹는 단위)."""
    sd = np.asarray(self.env.sim.data.sensordata[0].clone().cpu())
    q = np.asarray(self.robot.data.joint_pos[0].clone().cpu())
    gyro = self._noisy("base_ang_vel", sd[self._slice("base_ang_vel")], torch)
    # 센서는 **위 방향**을 낸다. 관측은 그 부호를 뒤집은 **중력 방향**이다.
    grav = self._noisy("projected_gravity",
                       -sd[self._slice("projected_gravity")], torch)
    qn = self._noisy("motor_pos_history", q[self.jidx], torch)
    self.pol.inference(
      gyro=gyro, projected_gravity=grav,
      joint_pos=dict(zip(self.pol.adapter.hist_joints, map(float, qn))),
      command=command,
    )
    return self.pol.last_raw_action


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--contract", required=True)
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--checkpoint", default=None,
                  help="학습 환경설정을 가져올 저장본. 없으면 명세에 적힌 것을 쓴다")
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument("--obs-noise", action="store_true", default=True,
                  help="관측 잡음을 켜고 시작 (기본 켜짐 — 실물과 같은 조건)")
  ap.add_argument("--no-obs-noise", dest="obs_noise", action="store_false")
  n = ap.parse_args(argv)

  import json

  import torch
  import viser

  contract = json.loads(Path(n.contract).read_text(encoding="utf-8"))
  ckpt = n.checkpoint or contract.get("checkpoint")
  if not ckpt:
    raise SystemExit("--checkpoint 이 필요합니다 (명세에 checkpoint 가 없습니다)")
  ckpt = str((REPO / "mujoco-sim/mjlab" / ckpt) if not Path(ckpt).is_absolute()
             and not Path(ckpt).exists() else ckpt)
  if not Path(ckpt).exists():
    raise SystemExit(f"저장본을 못 찾았습니다: {ckpt}")

  # 학습과 **같은 환경설정**으로 세운다. 손으로 고르지 않는다 — 기준자세가 다르면
  # 정책이 내는 숫자가 전혀 다른 관절 목표가 된다.
  apply_run_env(ckpt)
  env, envw, _torch_policy, torch = build(ckpt, obs_noise=n.obs_noise)[:4]
  env.cfg.observations["actor"].enable_corruption = bool(n.obs_noise)

  drv = DeployDriver(env, n.contract, n.onnx)
  drv.noise_on = bool(n.obs_noise)
  robot = env.scene["robot"]
  aterm = env.action_manager.get_term("joint_pos")
  jids = np.asarray(aterm._target_ids).reshape(-1)
  default_q = np.array([drv.pol.default_pose[j] for j in drv.pol.joint_names])

  from mjlab.envs.mdp.events import push_by_setting_velocity
  from mjlab.viewer.viser import MjlabViserScene

  server = viser.ViserServer(host="0.0.0.0", port=n.port, label="Pygmalion 서기 검증")
  scene = MjlabViserScene(server=server, mj_model=env.sim.mj_model, num_envs=1,
                          sim_model=env.sim.model,
                          expanded_fields=env.sim.expanded_fields)
  scene.env_idx = 0

  # 기본 시점이 너무 멀어 로봇이 점으로 보입니다. 붙는 사람마다 로봇 앞 2 m 에
  # 카메라를 놓아 줍니다 — 처음 열었을 때 바로 보이는 것이 중요합니다.
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (1.3, -1.0, 0.95)
    client.camera.look_at = (0.0, 0.0, 0.55)

  with server.gui.add_folder("명령 (서기 판정은 전부 0)"):
    s_vx = server.gui.add_slider("앞뒤 [m/s]", -1.0, 1.0, 0.05, 0.0)
    s_vy = server.gui.add_slider("좌우 [m/s]", -1.0, 1.0, 0.05, 0.0)
    s_wz = server.gui.add_slider("돌기 [rad/s]", -1.0, 1.0, 0.05, 0.0)

  with server.gui.add_folder("흔들어 보기"):
    s_push = server.gui.add_slider("미는 세기 [m/s]", 0.5, 3.0, 0.1, 1.5)
    b_push = server.gui.add_button("밀기")
    c_lift = server.gui.add_checkbox("들어올리기 (자리만 잡음)", False)
    c_noise = server.gui.add_checkbox("관측 잡음 (실물과 같은 조건)", bool(n.obs_noise))
    b_reset = server.gui.add_button("기준자세로 리셋")

  with server.gui.add_folder("지금 상태"):
    h_stat = server.gui.add_html("")
  with server.gui.add_folder("무엇을 보고 있나"):
    server.gui.add_markdown(
      "이 화면을 도는 것은 **로봇에 올릴 코드 그대로**입니다 — "
      "`obs_adapter`(numpy) → `정책.onnx` → 관절 목표각. 학습 때 쓰던 PyTorch "
      "정책이 아닙니다.\n\n"
      "**떨림**이 가만히 선 상태에서 크면 그것이 고치려는 증상입니다. "
      "**관측 잡음을 끄면 안 떨리는 것처럼 보입니다** — 잡음이 있어야 나타납니다.\n\n"
      "**밀기** 는 측정 도구가 쓰는 것과 같은 함수입니다. 1.5 m/s 까지 안 넘어지는 "
      "것이 합격선입니다."
    )

  state = {"push": False, "reset": False}
  b_push.on_click(lambda _: state.__setitem__("push", True))
  b_reset.on_click(lambda _: state.__setitem__("reset", True))

  @c_noise.on_update
  def _(_):
    drv.noise_on = bool(c_noise.value)

  # viser 는 요청한 문이 차 있으면 **말없이 다음 문으로 옮깁니다.** 요청한 번호를
  # 그대로 찍으면 열리지 않는 주소를 알려 주게 되므로, 실제로 열린 문을 물어봅니다.
  port = int(server.get_port())
  ip = lan_address()
  if port != n.port:
    print(f"  (요청한 문 {n.port} 가 차 있어 {port} 로 열렸습니다)", flush=True)
  print(f"\n  ★ 브라우저에서 여십시오:  http://{ip}:{port}   "
        f"(이 기계에서는 http://localhost:{port})\n", flush=True)

  obs, _ = envw.reset()
  drv.reset()
  pose_hold = None
  dt = 1.0 / float(drv.pol.adapter.control_rate_hz)
  qd_hist: list[float] = []
  t_next = time.perf_counter()

  try:
    while True:
      if state.pop("reset", False) or state.get("reset"):
        state["reset"] = False
        obs, _ = envw.reset()
        drv.reset()
        pose_hold = None
        qd_hist.clear()

      # 들어올리기: **자리만** 고정하고 자세는 놓아 준다(사람이 몸통을 든 것과 같다).
      if c_lift.value:
        if pose_hold is None:
          pose_hold = robot.data.root_link_pose_w.clone()
          pose_hold[:, 2] += 0.35
          pose_hold[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cur = robot.data.root_link_pose_w.clone()
        cur[:, :3] = pose_hold[:, :3]
        robot.write_root_link_pose_to_sim(cur)
        v = robot.data.root_link_vel_w.clone()
        v[:, :3] = 0.0
        robot.write_root_link_velocity_to_sim(v)
      else:
        pose_hold = None

      if state.pop("push", False):
        p = float(s_push.value)
        push_by_setting_velocity(
          env, None, {"x": (-p, p), "y": (-p, p), "z": (0.0, 0.0),
                      "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)})

      cmd = (float(s_vx.value), float(s_vy.value), float(s_wz.value))
      raw = drv.step(cmd, torch)
      act = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0)
      obs, _, _, _ = envw.step(act)
      # 명령은 사용자가 미끄럼단추로 정한다 — 환경이 무작위로 새 명령을 뽑지 못하게.
      if cmd == (0.0, 0.0, 0.0):
        zero_command(env)
      else:
        for cname in ("twist", "base_velocity"):
          try:
            t = env.command_manager.get_term(cname)
            t.vel_command_b[:] = torch.tensor(cmd, device=t.vel_command_b.device)
            break
          except Exception:                                        # noqa: BLE001
            continue

      # ---- 숫자 네 개 -------------------------------------------------
      up = np.asarray(robot.data.projected_gravity_b[0].cpu()) \
        if hasattr(robot.data, "projected_gravity_b") else None
      tilt = (float(np.degrees(np.arccos(np.clip(-up[2], -1, 1)))) if up is not None
              else float("nan"))
      qd = np.asarray(robot.data.joint_vel[0].clone().cpu())[jids]
      tau = np.asarray(robot.data.actuator_force[0].clone().cpu())
      q = np.asarray(robot.data.joint_pos[0].clone().cpu())[jids]
      jit = float(np.abs(qd).sum())
      qd_hist.append(jit)
      if len(qd_hist) > 100:
        qd_hist.pop(0)
      pose_err = float(np.degrees(np.abs(q - default_q).mean()))
      h_stat.content = (
        f"<div style='font-family:monospace;line-height:1.6'>"
        f"기울기 <b>{tilt:5.2f}°</b><br>"
        f"떨림 |qd| <b>{np.mean(qd_hist):6.3f}</b> "
        f"<span style='opacity:.6'>(지금 {jit:.3f})</span><br>"
        f"토크 RMS <b>{float(np.sqrt((tau ** 2).mean())):6.2f} Nm</b><br>"
        f"자세오차 <b>{pose_err:5.2f}°</b><br>"
        f"높이 {float(robot.data.root_link_pos_w[0, 2]):.3f} m<br>"
        f"주기 {drv.pol.steps}"
        f"</div>")

      scene.update(env.sim.data)
      t_next += dt
      sleep = t_next - time.perf_counter()
      if sleep > 0:
        time.sleep(sleep)
      else:                                    # 뒤처지면 따라잡지 않고 시계를 맞춘다
        t_next = time.perf_counter()
  except KeyboardInterrupt:
    print("\n뷰어를 닫습니다.")
  return 0


if __name__ == "__main__":
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
  raise SystemExit(main())
