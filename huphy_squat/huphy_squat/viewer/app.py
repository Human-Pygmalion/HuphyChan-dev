"""HuphyViewer: mjlab's ViserPlayViewer + Squat / DR / Disturb tabs.

Threading: GUI callbacks run in viser's thread and only change Python-side values or queue
work with `self._run_in_sim(fn)`; everything that touches sim tensors runs in the viewer's
main loop (between policy steps), like the stock viewer's own buttons.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import viser
from typing_extensions import override

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import UpdateReason, ViserPlayViewer

from ..env_cfg import HEIGHT_RANGE, PUSH
from ..mdp.commands import base_height
from ..mdp.events import get_motor
from ..mdp.observations import foot_friction, loop_distances
from .dr import DrControl
from .scene import BOX, BOX_PARK, box_body

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

DIRECTIONS = {"front": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}
G = 9.81


def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _direction(name: str, n: int, device) -> torch.Tensor:
  """Angle [rad] in the robot frame, one per env ("random" draws per env)."""
  if name == "random":
    return torch.rand(n, device=device) * 2 * math.pi
  return torch.full((n,), math.radians(DIRECTIONS[name]), device=device)


class HuphyViewer(ViserPlayViewer):
  def __init__(self, env, policy, *, box_edge: float, **kw: Any) -> None:
    super().__init__(env, policy, **kw)
    self.box_edge = box_edge
    u: ManagerBasedRlEnv = env.unwrapped
    self.u = u
    self.dr = DrControl(u)
    self.cmd = u.command_manager.get_term("height")
    self.push = u.huphy_push  # push_pelvis instance (step event)
    self.h_mode = "Manual"
    self.h_value = float(self.cmd.cfg.height_range[1])
    self.sweep_period = 8.0
    self._sweep_t = 0.0
    self.falls = 0
    self._wrap_height_resample()
    self.set_random_push(False, PUSH["force_range"][1])

  # ------------------------------------------------------------------ plumbing

  def _run_in_sim(self, fn: Callable[[], None]) -> None:
    self.request_action("CUSTOM", {"type": "huphy", "fn": fn})

  @override
  def _handle_custom_action(self, action: ViewerAction, payload: Any) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "huphy":
      with self._sim_lock:
        payload["fn"]()
      self._pending_update_reasons.add(UpdateReason.ACTION)
      return True
    return super()._handle_custom_action(action, payload)

  def _env_ids(self, all_envs: bool) -> torch.Tensor:
    if all_envs:
      return torch.arange(self.u.num_envs, device=self.u.device)
    return torch.tensor([self._scene.env_idx], device=self.u.device)

  @override
  def reset_environment(self) -> None:
    """Every reset (GUI Reset buttons, checkpoint swap) spawns with the GUI DR values."""
    self.dr.write_params()
    self._sweep_t = 0.0
    super().reset_environment()
    self._refresh_dr_status()

  @override
  def _execute_step(self) -> bool:
    if self.h_mode == "Sweep":
      lo, hi = self.cmd.cfg.height_range
      mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
      self.cmd.target[:, 0] = mid + amp * math.cos(2 * math.pi * self._sweep_t / self.sweep_period)
    elif self.h_mode == "Manual":
      self.cmd.target[:, 0] = self.h_value
    ok = super()._execute_step()
    if ok:
      self._sweep_t += self.u.step_dt
      tm = self.u.termination_manager
      self.falls += int((tm.terminated & ~tm.time_outs).sum())
    return ok

  # ------------------------------------------------------------------ height command

  def _wrap_height_resample(self) -> None:
    orig = self.cmd._resample_command  # noqa: SLF001

    def resample(env_ids: torch.Tensor) -> None:
      if self.h_mode == "Train":
        orig(env_ids)
      else:  # Manual / Sweep: the viewer owns h*; never resample on its own
        self.cmd.target[env_ids, 0] = self.h_value
        self.cmd.time_left[env_ids] = 1e9

    self.cmd._resample_command = resample  # noqa: SLF001

  def set_height_mode(self, mode: str) -> None:
    self.h_mode = mode

    def apply() -> None:
      if mode == "Train":
        self.cmd.time_left[:] = 0.0  # resample at the next step
      else:
        self.cmd.time_left[:] = 1e9
        self._sweep_t = 0.0

    self._run_in_sim(apply)

  # ------------------------------------------------------------------ disturbances

  def set_random_push(self, on: bool, f_max: float) -> None:
    """Training-style random pelvis pushes (U(0, f_max) N, 0.1-0.2 s, every 2-5 s)."""
    p = self.push
    self.u.event_manager.get_term_cfg("push").params["force_range"] = (0.0, float(f_max))
    p.interval = PUSH["interval_s"] if on else (1e9, 1e9)
    idle = p.remaining <= 0
    p.pause = torch.where(idle, p._u(p.interval, len(idle)), p.pause)  # noqa: SLF001

  def push_now(self, force: float, duration: float, direction: str, all_envs: bool) -> None:
    ids = self._env_ids(all_envs)
    robot = self.u.scene["robot"]
    ang = _yaw(robot.data.root_link_quat_w[ids]) + _direction(direction, len(ids), self.u.device)
    p = self.push
    p.force[ids, 0] = force * torch.cos(ang)
    p.force[ids, 1] = force * torch.sin(ang)
    p.force[ids, 2] = 0.0
    p.remaining[ids] = duration
    p.count[ids] += 1

  def throw_box(self, mass: float, speed: float, dist: float, z_off: float, direction: str, all_envs: bool) -> None:
    """Launch the box from `dist` m away (direction in the robot frame) so that its ballistic
    path hits the pelvis + z_off at horizontal speed `speed`."""
    u, ids = self.u, self._env_ids(all_envs)
    n, b, s = len(ids), box_body(u), self.box_edge
    u.sim.model.body_mass[ids, b] = mass
    u.sim.model.body_inertia[ids, b] = torch.full((n, 3), mass * s * s / 6.0, device=u.device)
    u.sim.recompute_constants(RecomputeLevel.set_const)
    robot = u.scene["robot"]
    target = robot.data.root_link_pos_w[ids].clone()
    target[:, 2] += z_off
    ang = _yaw(robot.data.root_link_quat_w[ids]) + _direction(direction, n, u.device)
    unit = torch.stack((torch.cos(ang), torch.sin(ang)), dim=1)
    t = dist / speed
    root = torch.zeros(n, 13, device=u.device)
    root[:, :2] = target[:, :2] + dist * unit
    root[:, 2] = target[:, 2]
    root[:, 3] = 1.0
    root[:, 7:9] = -speed * unit
    root[:, 9] = 0.5 * G * t  # same start / end height
    u.scene[BOX].write_root_state_to_sim(root, env_ids=ids)

  def park_box(self, all_envs: bool) -> None:
    ids = self._env_ids(all_envs)
    root = torch.zeros(len(ids), 13, device=self.u.device)
    root[:, :3] = self.u.scene.env_origins[ids]
    root[:, 0] += BOX_PARK[0]
    root[:, 1] += BOX_PARK[1]
    root[:, 2] += self.box_edge / 2
    root[:, 3] = 1.0
    self.u.scene[BOX].write_root_state_to_sim(root, env_ids=ids)

  # ------------------------------------------------------------------ GUI

  @override
  def setup(self) -> None:
    gui = self._server.gui
    orig = gui.add_tab_group
    captured: dict[str, Any] = {}

    def capture(*a, **k):
      captured["tabs"] = orig(*a, **k)
      return captured["tabs"]

    gui.add_tab_group = capture  # type: ignore[method-assign]
    try:
      super().setup()
    finally:
      gui.add_tab_group = orig  # type: ignore[method-assign]
    tabs = captured["tabs"]
    with tabs.add_tab("Squat", icon=viser.Icon.ARROWS_VERTICAL):
      self._build_height_gui()
      self._build_monitor_gui()
    with tabs.add_tab("DR", icon=viser.Icon.ADJUSTMENTS):
      self._build_dr_gui()
    with tabs.add_tab("Disturb", icon=viser.Icon.HAND_FINGER):
      self._build_disturb_gui()
    self.request_reset()  # first spawn: clean DR

  def _build_height_gui(self) -> None:
    gui = self._server.gui
    lo, hi = HEIGHT_RANGE
    with gui.add_folder("Height command h*"):
      mode = gui.add_dropdown("mode", options=("Manual", "Train", "Sweep"), initial_value=self.h_mode,
                              hint="Manual: 슬라이더 값 / Train: 학습처럼 3-6 s마다 무작위 / Sweep: 학습 범위를 cos으로 왕복")
      h = gui.add_slider("h* [m]", min=0.60, max=0.95, step=0.001, initial_value=self.h_value,
                         hint=f"학습 범위 {lo:.3f}-{hi:.3f} m. 바로 반영됨 (Manual)")
      presets = gui.add_button_group("preset", options=(f"low {lo:.2f}", "mid", f"high {hi:.3f}"))
      period = gui.add_slider("sweep period [s]", min=2.0, max=30.0, step=0.5, initial_value=self.sweep_period)

    @mode.on_update
    def _(_) -> None:
      self.set_height_mode(mode.value)

    @h.on_update
    def _(_) -> None:
      self.h_value = float(h.value)

    @presets.on_click
    def _(ev) -> None:
      v = ev.target.value
      h.value = lo if v.startswith("low") else hi if v.startswith("high") else round(0.5 * (lo + hi), 3)
      self.h_value = float(h.value)
      if mode.value != "Manual":
        mode.value = "Manual"
        self.set_height_mode("Manual")

    @period.on_update
    def _(_) -> None:
      self.sweep_period = float(period.value)

  def _build_monitor_gui(self) -> None:
    with self._server.gui.add_folder("Monitor (selected env)"):
      self._monitor_html = self._server.gui.add_html("")
      clear = self._server.gui.add_button("Clear fall count")

    @clear.on_click
    def _(_) -> None:
      self.falls = 0

  def _build_dr_gui(self) -> None:
    gui = self._server.gui
    self._dr_status = gui.add_html("")
    buttons = gui.add_button_group("", options=("Random", "Clean", "Apply & Reset"))
    self._dr_handles: dict[str, Any] = {}
    folders: dict[str, list] = {}
    for k in self.dr.knobs:
      folders.setdefault(k.folder, []).append(k)
    inertia = [f for f in folders if f.startswith("Inertia/")]
    for folder, knobs in folders.items():
      if folder in inertia:
        continue
      with gui.add_folder(folder, expand_by_default=folder in ("Mass", "Contact", "Latency")):
        for k in knobs:
          self._add_knob(k)
    with gui.add_folder("Inertia (pseudo-inertia)", expand_by_default=False):
      for folder in inertia:
        with gui.add_folder(folder.split("/", 1)[1], expand_by_default=False):
          for k in folders[folder]:
            self._add_knob(k)

    @buttons.on_click
    def _(ev) -> None:
      if ev.target.value == "Apply & Reset":
        self.request_reset()
        return
      new = self.dr.random() if ev.target.value == "Random" else self.dr.clean()
      for key, v in new.items():
        self.dr.values[key] = v
        self._dr_handles[key].value = v
      self._refresh_dr_status()

    self._refresh_dr_status()

  def _add_knob(self, k) -> None:
    lo, hi = k.train
    hint = f"학습 범위 {lo:g} ~ {hi:g}, clean {k.clean:g}" + (f". {k.hint}" if k.hint else "")
    handle = self._server.gui.add_slider(k.label, min=k.lo, max=k.hi, step=k.step, initial_value=k.clean, hint=hint)
    self._dr_handles[k.key] = handle

    @handle.on_update
    def _(_) -> None:
      self.dr.values[k.key] = float(handle.value)
      self._refresh_dr_status()

  def _refresh_dr_status(self) -> None:
    if not hasattr(self, "_dr_status"):
      return
    pend = self.dr.pending()
    if pend:
      msg = (f'<span style="color:#d97706;"><strong>대기 중 {len(pend)}개</strong></span> — '
             "<strong>Apply &amp; Reset</strong> (또는 Controls의 Reset)을 눌러야 반영됩니다.")
    else:
      msg = '<span style="color:#16a34a;"><strong>적용됨</strong></span> — 현재 로봇이 이 값으로 스폰되어 있습니다.'
    self._dr_status.content = f'<div style="font-size:0.85em;padding:0 1em 0.5em 1em;">{msg}</div>'

  def _build_disturb_gui(self) -> None:
    gui = self._server.gui
    with gui.add_folder("Push (pelvis force)"):
      pf = gui.add_slider("force [N]", min=0.0, max=200.0, step=1.0, initial_value=60.0,
                          hint=f"학습: U(0, {PUSH['force_range'][1]:g}) N")
      pd = gui.add_slider("duration [s]", min=0.05, max=1.0, step=0.05, initial_value=0.2)
      pdir = gui.add_dropdown("direction", options=(*DIRECTIONS, "random"), initial_value="random",
                              hint="로봇이 밀리는 방향 (로봇 기준)")
      pall = gui.add_checkbox("all envs", initial_value=False)
      pbtn = gui.add_button("Push", color="orange")
      rnd = gui.add_checkbox("random pushes (train style)", initial_value=False,
                             hint="학습처럼 2-5 s마다 U(0, max) N, 0.1-0.2 s")
      rmax = gui.add_slider("random max [N]", min=0.0, max=200.0, step=1.0, initial_value=PUSH["force_range"][1])
    with gui.add_folder("Throw box"):
      gui.add_html(f'<div style="font-size:0.8em;padding:0 1em;">상자 한 변 {self.box_edge:g} m '
                   "(실행 인자 --box-edge). 골반을 향해 포물선으로 던짐.</div>")
      bm = gui.add_slider("mass [kg]", min=0.1, max=30.0, step=0.1, initial_value=3.0,
                          hint="관성은 균일 정육면체 m·s²/6")
      bv = gui.add_slider("speed [m/s]", min=0.5, max=15.0, step=0.1, initial_value=4.0, hint="수평 속도")
      bd = gui.add_slider("start distance [m]", min=0.5, max=4.0, step=0.1, initial_value=1.5)
      bz = gui.add_slider("aim height [m]", min=-0.6, max=0.4, step=0.01, initial_value=0.0,
                          hint="골반 원점 기준 위(+)/아래(-)")
      bdir = gui.add_dropdown("from", options=(*DIRECTIONS, "random"), initial_value="front",
                              hint="상자가 날아오는 쪽 (로봇 기준)")
      ball = gui.add_checkbox("all envs ", initial_value=False)
      bbtn = gui.add_button("Throw", color="red")
      bpark = gui.add_button("Remove box")

    @pbtn.on_click
    def _(_) -> None:
      f, d, dr, a = float(pf.value), float(pd.value), pdir.value, pall.value
      self._run_in_sim(lambda: self.push_now(f, d, dr, a))

    def _random_push(_) -> None:
      on, fm = rnd.value, float(rmax.value)
      self._run_in_sim(lambda: self.set_random_push(on, fm))

    rnd.on_update(_random_push)
    rmax.on_update(_random_push)

    @bbtn.on_click
    def _(_) -> None:
      args = (float(bm.value), float(bv.value), float(bd.value), float(bz.value), bdir.value, ball.value)
      self._run_in_sim(lambda: self.throw_box(*args))

    @bpark.on_click
    def _(_) -> None:
      self._run_in_sim(lambda: self.park_box(True))

  # ------------------------------------------------------------------ monitor

  @override
  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    if self._counter % 6 == 0:
      self._update_monitor()

  def _update_monitor(self) -> None:
    u, i = self.u, self._scene.env_idx
    robot = u.scene["robot"]
    m = get_motor(u)
    with self._sim_lock:
      h = float(base_height(u)[i])
      hs = float(self.cmd.target[i, 0])
      vz = float(robot.data.root_link_lin_vel_w[i, 2])
      g = robot.data.projected_gravity_b[i]
      tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))
      ratio = float(m.step_max_ratio[i].max())
      loop = float(loop_distances(u)[i].max()) * 1000
      pushing = float(self.push.remaining[i]) > 0
      fr = float(foot_friction(u, u.event_manager.get_term_cfg("foot_friction").params["asset_cfg"])[i].mean())
      payload = float(getattr(u, "huphy_payload", torch.zeros(u.num_envs))[i])
      kp = float((m.stiffness[i] / m.default_stiffness[i]).mean())
      kd = float((m.damping[i] / m.default_damping[i]).mean())
      buf = m._delay_buffer  # noqa: SLF001
      cmd_ms = 5 * int(buf.current_lags[i]) if buf is not None else 0
      obs_ms = 20 * int(u.huphy_obs_lag[i])
    err_col = "#16a34a" if abs(h - hs) < 0.02 else "#d97706"
    rat_col = "#dc2626" if ratio > 0.9 else "inherit"
    self._monitor_html.content = f"""
      <div style="font-size:0.85em;line-height:1.35;padding:0 1em 0.5em 1em;">
        <strong>h</strong> {h:.3f} m &nbsp; <strong>h*</strong> {hs:.3f} m &nbsp;
        <span style="color:{err_col};">err {1000 * (h - hs):+.0f} mm</span><br/>
        <strong>vz</strong> {vz:+.3f} m/s &nbsp; <strong>tilt</strong> {tilt:.1f}°<br/>
        <span style="color:{rat_col};"><strong>max torque / T-N</strong> {ratio:.2f}</span> &nbsp;
        <strong>loop</strong> {loop:.2f} mm<br/>
        <strong>falls</strong> {self.falls} &nbsp; {'<strong style="color:#d97706;">PUSH</strong>' if pushing else ''}<br/>
        <span style="opacity:0.75;">applied: payload {payload:+.1f} kg, μ {fr:.2f}, Kp×{kp:.2f}, Kd×{kd:.2f},
        cmd {cmd_ms} ms, obs {obs_ms} ms</span>
      </div>"""
