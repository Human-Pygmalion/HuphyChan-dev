"""viser UI: 3D scene (mjviser) + the control panel.

Panel folders: Model / Base / Joints / Plots / Status.

Two rules the panel follows everywhere:
  * every number shown for a joint comes from the CONTRACT (default, clip window, travel
    sign, mirrored flag) or from the model's own state - never from a regex over the name;
  * a GUI callback only enqueues a ``Command``.  It never touches MjData.  The sim thread
    owns the model.

Mirror handling: on the v30 build the two legs have opposite joint axes for the knee, the
hips, the cranks and the ankle roll.  ``+0.35 rad`` therefore means flexion on the left leg
and extension on the right.  Every mirrored joint's readout carries a second number,
``phys``, which is ``travel_sign * q`` - the same physical motion reads the same on both
legs.  The slider stays in raw q, because that is what the API, the policy and the
telemetry all speak.
"""

from __future__ import annotations

import os
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import viser
from mjviser import ViserMujocoScene

from . import CACHE_DIR, VARIANTS
from .contract import list_baked, load_contract

PLOT_WINDOW_S = 10.0
PLOT_HZ = 50.0
PLOT_PUSH_HZ = 10.0
RENDER_HZ = 30.0
READOUT_HZ = 10.0
MAX_CHANNELS = 8

_COLORS = [
  "#4c78a8", "#f58518", "#54a24b", "#e45756",
  "#72b7b2", "#b279a2", "#eeca3b", "#9d755d",
]


class RingPlot:
  """Fixed-size circular buffer, one row per channel.  Never grows."""

  def __init__(self, channels: list[str], n: int):
    self.channels = channels
    self.n = n
    self.t = np.zeros(n)
    self.y = np.zeros((len(channels), n))
    self.i = 0
    self.filled = 0
    self._idx = {c: k for k, c in enumerate(channels)}

  def push(self, t: float, values: dict[str, float]) -> None:
    self.t[self.i] = t
    for c, v in values.items():
      self.y[self._idx[c], self.i] = v
    self.i = (self.i + 1) % self.n
    self.filled = min(self.filled + 1, self.n)

  def view(self, chans: list[str]):
    if self.filled == 0:
      return None
    order = np.arange(self.i - self.filled, self.i) % self.n
    return self.t[order], [self.y[self._idx[c]][order] for c in chans]


def build_ui(core, host: str, port: int, freshness: dict, base: str = "free") -> viser.ViserServer:
  server = viser.ViserServer(host=host, port=port, label="Pygmalion sim<->real viewer")
  state = {"core": core, "scene": None, "stop": False, "freshness": freshness, "base": base}
  _mount(server, state)

  def render_loop():
    last_r = last_p = last_o = 0.0
    while not state["stop"]:
      now = time.perf_counter()
      c = state["core"]
      if now - last_r >= 1.0 / RENDER_HZ:
        try:
          state["scene"].update_from_mjdata(c.d)
        except Exception:
          pass
        last_r = now
      if now - last_o >= 1.0 / READOUT_HZ:
        try:
          state["readout"]()
        except Exception:
          pass
        last_o = now
      if now - last_p >= 1.0 / PLOT_PUSH_HZ:
        try:
          state["push_plot"]()
        except Exception:
          pass
        last_p = now
      time.sleep(0.005)

  t = threading.Thread(target=render_loop, name="pygviewer-ui", daemon=True)
  t.start()
  state["thread"] = t
  return server


# --------------------------------------------------------------------------- panel
_PANEL_RESIZE_JS = (
  "if(window.__pygGrip)return;window.__pygGrip=1;"
  "var K='pygPanelWidth';function find(el){var n=el;while(n&&n.parentElement){var r=n.getBoundingCl"
  'ientRect();if(r.width>150&&r.width<window.innerWidth*0.8&&r.height>window.innerHeight*0.5&&r.rig'
  "ht>window.innerWidth-80)return n;n=n.parentElement;}return null;}var panel=find(this);"
  "if(!panel)return;var g=document.createElem"
  "ent('div');g.title='\\uB04C\\uC5B4\\uC11C \\uB108\\uBE44 \\uC870\\uC808';g.style.cssText='position:fixe"
  "d;top:0;bottom:0;width:10px;z-index:10000;cursor:col-resize;background:rgba(128,128,128,.25)';g."
  "onmouseenter=function(){g.style.background='rgba(90,150,210,.55)'};g.onmouseleave=function(){g.s"
  "tyle.background='rgba(128,128,128,.25)'};document.body.appendChild(g);function put(w){w=Math.max"
  "(240,Math.min(window.innerWidth-120,w));panel.style.width=w+'px';panel.style.maxWidth='none';g.s"
  "tyle.right=w+'px';try{localStorage.setItem(K,String(w))}catch(e){}}var drag=false;g.addEventList"
  "ener('pointerdown',function(e){drag=true;g.setPointerCapture(e.pointerId);e.preventDefault()});g"
  ".addEventListener('pointermove',function(e){if(drag)put(window.innerWidth-e.clientX)});g.addEven"
  "tListener('pointerup',function(){drag=false});g.addEventListener('pointercancel',function(){drag"
  "=false});g.addEventListener('dblclick',function(){put(420)});window.addEventListener('resize',fu"
  'nction(){put(panel.getBoundingClientRect().width)});var w0=420;try{w0=parseInt(localStorage.getI'
  "tem(K)||'420',10)||420}catch(e){}put(w0);"
)
"""조작판 왼쪽에 끌기 손잡이를 붙이는 코드. 브라우저에서 실행된다."""


def _mount(server: viser.ViserServer, state: dict) -> None:
  core = state["core"]
  c = core.c
  gui = server.gui
  state["scene"] = ViserMujocoScene(server, core.m, num_envs=1)
  state["scene"].update_from_mjdata(core.d)

  # --------------------------------------------------------- Panel width
  # 2026-09-12, user: "viser 우측 창 내가 크기 슬라이드가능하게 못하나? 너무작아".
  # viser's own theme only offers three fixed widths, so ``large`` is the new baseline and
  # the block below adds a drag handle on top of it.  Everything else in ``configure_theme``
  # keeps its default, which is what this app was already getting.
  gui.configure_theme(control_width="large")

  # The handle is built in the browser, because only the browser knows where the panel is.
  # It walks UP from this element - the panel is whatever ancestor is right-anchored and
  # full-height - instead of matching a CSS class, because Mantine's class names carry a
  # build hash that changes when viser is upgraded.  Width is remembered per browser;
  # double-click resets it.
  #
  # ★ The code rides on an ``onerror`` attribute, not a ``<script>`` tag: HTML inserted
  # through ``innerHTML`` (which is what ``add_html`` ends up doing) never executes script
  # tags, so the first attempt here silently did nothing - confirmed by driving the page
  # with a real browser, which is the only way this class of bug shows up.
  gui.add_html(
    '<img src="data:," style="display:none" alt="" onerror="'
    + _PANEL_RESIZE_JS.replace('"', '&quot;')
    + '">'
  )

  # ------------------------------------------------------------- Bench camera
  # 2026-09-12, user: "viser Wrapper 뷰어에 카메라 화면뜨게 할수있나? 지연 최대한 적게".
  #
  # The browser fetches the MJPEG stream DIRECTLY from the camera server, so the frames
  # never pass through this process or viser's websocket: no decode, no re-encode, no
  # extra hop.  That is the lowest-latency option available here - ``gui.add_image`` would
  # mean pulling every frame into this process and pushing it back out again.
  #
  # Off unless ``PYG_CAM_URL`` is set, so a viewer started without it is byte-for-byte the
  # viewer that existed before this block.
  cam_url = os.environ.get("PYG_CAM_URL", "").strip()
  if cam_url:
    with gui.add_folder("Bench camera"):
      gui.add_html(
        f'<img src="{cam_url}" alt="bench" '
        f'style="width:100%;border-radius:6px;display:block" />'
        f'<div style="font-size:11px;opacity:.6;margin-top:4px">{cam_url}</div>'
      )

  # ------------------------------------------------------------------ Model
  with gui.add_folder("Model"):
    baked = list_baked(CACHE_DIR) or [c.variant]
    dd = gui.add_dropdown("variant", tuple(baked), initial_value=c.variant)
    stale = "STALE" if state["freshness"].get("stale") else "fresh"
    gui.add_markdown(
      f"**{c.variant}** &nbsp; `{c.contract_sha[:12]}` &nbsp; ({stale})\n\n"
      f"{c.raw['total_mass_kg']} kg &middot; nu {c.raw['nu']} &middot; "
      f"{c.raw['n_dof']} joints &middot; physics {1 / c.raw['physics_dt']:.0f} Hz / "
      f"control {1 / c.raw['step_dt']:.0f} Hz\n\n"
      f"xml `{c.raw['model_xml'].split('/')[-1]}`  \n"
      f"mjlab `{c.raw['mjlab_git']}` &middot; baked {c.raw['bake_utc']}"
    )
    reload_btn = gui.add_button("load variant")

    @reload_btn.on_click
    def _(_evt):
      want = dd.value
      if want == state["core"].c.variant:
        return
      _swap_variant(server, state, want)

  # ------------------------------------------------------------------ Base
  with gui.add_folder("Base link"):
    gui.add_markdown(
      "`free` gravity only &middot; `fixed` welded to the anchor &middot; "
      "`pivot` the offset point is held, rotation free &middot; `string` a safety tether: "
      "slack above `Z_set`, taut (holds the base up) at or below it - horizontal motion is "
      "always free, like a real overhead harness.\n\n"
      "**Gravity is never modified by any of these.**"
    )
    mode = gui.add_dropdown(
      "mode", ("free", "fixed", "pivot", "string"), initial_value=state.get("base", core.base_mode)
    )
    px = gui.add_number("anchor x [m]", initial_value=0.0, step=0.005)
    py = gui.add_number("anchor y [m]", initial_value=0.0, step=0.005)
    height = gui.add_slider(
      "height z [m]", min=0.0, max=1.6, step=0.005, initial_value=float(core.base_pos[2])
    )
    roll = gui.add_slider("roll [deg]", min=-90, max=90, step=0.5, initial_value=0.0)
    pitch = gui.add_slider("pitch [deg]", min=-90, max=90, step=0.5, initial_value=0.0)
    yaw = gui.add_slider("yaw [deg]", min=-180, max=180, step=0.5, initial_value=0.0)
    gui.add_markdown("offset xyz below is shared by `pivot` (rotation centre) and `string` "
                      "(tether attachment point) - both are \"a point in the BASE frame\".")
    ox = gui.add_number("pivot/hook offset x [m]", initial_value=0.0, step=0.005)
    oy = gui.add_number("pivot/hook offset y [m]", initial_value=0.0, step=0.005)
    oz = gui.add_number("pivot/hook offset z [m]", initial_value=0.0, step=0.005)
    ground = gui.add_checkbox("ground contact", initial_value=True)
    z_set = gui.add_slider(
      "string Z_set [m]", min=0.3, max=1.2, step=0.005, initial_value=float(core.string_z_set)
    )
    follow_xy = gui.add_checkbox(
      "string follow XY (no swing)", initial_value=core.string_follow_xy
    )
    string_md = gui.add_markdown("")
    kf = gui.add_button_group("reset to (also restores the base pose)",
                              ("home", "knees_bent"))

  def push_base(_evt=None):
    state["core"].submit(
      {
        "op": "base",
        "mode": mode.value,
        "pos": [px.value, py.value, height.value],
        "rpy": [math.radians(roll.value), math.radians(pitch.value), math.radians(yaw.value)],
        "pivot_offset": [ox.value, oy.value, oz.value],
        "hook_offset": [ox.value, oy.value, oz.value],
        "ground": ground.value,
        "z_set": z_set.value,
        "follow_xy": follow_xy.value,
      }
    )

  for h in (mode, px, py, height, roll, pitch, yaw, ox, oy, oz, ground, z_set, follow_xy):
    h.on_update(push_base)

  @kf.on_click
  def _(_evt):
    state["core"].submit({"op": "reset", "keyframe": kf.value})

  # ------------------------------------------------------------------ Joints
  sliders: dict[str, tuple] = {}
  readouts: dict[str, object] = {}
  guard = {"busy": False}

  def send(name: str, v: float):
    if guard["busy"]:
      return
    state["core"].submit({"op": "target", "values": {name: float(v)}})

  with gui.add_folder("Joints (targets, rad)"):
    gui.add_markdown(
      "Slider range = the contract's `safe_clip` (centre &plusmn; 0.5&middot;range&middot;"
      "factor) - the same clamp the trainer applies to a policy target. `phys` on a mirrored "
      "joint is `travel_sign &times; q`, so both legs read alike."
    )
    for n in c.action_joint_names:
      lo, hi = c.clip(n)
      d0 = c.default_q(n)
      row = gui.add_slider(
        n.replace("_joint", ""),
        min=round(lo, 4),
        max=round(hi, 4),
        step=0.002,
        initial_value=float(np.clip(d0, lo, hi)),
      )
      num = gui.add_number(f"{n.replace('_joint', '')} =", initial_value=float(d0), step=0.002)
      txt = gui.add_text(f"{n.replace('_joint', '')} state", initial_value="", disabled=True)
      sliders[n] = (row, num)
      readouts[n] = txt

      def _mk(nn, rr, nn2):
        @rr.on_update
        def _(_evt):
          guard["busy"] = True
          nn2.value = rr.value
          guard["busy"] = False
          send(nn, rr.value)

        @nn2.on_update
        def _(_evt):
          lo2, hi2 = c.clip(nn)
          v = float(np.clip(nn2.value, lo2, hi2))
          guard["busy"] = True
          rr.value = v
          guard["busy"] = False
          send(nn, v)

      _mk(n, row, num)

    home_btn = gui.add_button("targets -> default pose")

    @home_btn.on_click
    def _(_evt):
      vals = {n: c.default_q(n) for n in c.action_joint_names}
      state["core"].submit({"op": "target", "values": vals})
      guard["busy"] = True
      for n, (r, nu) in sliders.items():
        lo2, hi2 = c.clip(n)
        r.value = float(np.clip(vals[n], lo2, hi2))
        nu.value = vals[n]
      guard["busy"] = False

  # ------------------------------------------------------------------ Ankle (AB)
  ankle_ctl = {}
  if core.ankle_inverse is not None:
    meta = c.raw["ankle_inverse"]
    (pl, ph), (rl, rh) = core.ankle_inverse.bounds()
    with gui.add_folder("Ankle foot-space (AB)"):
      gui.add_markdown(
        f"method **{meta['method']}** from `{meta['envelope_tag']}`, per-leg sign map fitted "
        f"at bake, worst probe residual **{meta['worst_residual_rad']} rad**.\n\n"
        "These sliders command the two CRANKS through the inverse. The pitch/roll shown in "
        "Status is the model's own state, so any inverse error shows as target-vs-actual."
      )
      for s in ("L", "R"):
        sp = gui.add_slider(
          f"{s} ankle pitch [rad]", min=round(pl, 3), max=round(ph, 3), step=0.005,
          initial_value=0.0,
        )
        sr = gui.add_slider(
          f"{s} ankle roll [rad]", min=round(rl, 3), max=round(rh, 3), step=0.005,
          initial_value=0.0,
        )
        ankle_ctl[s] = (sp, sr)

        def _mk_ankle(side, a, b):
          def _cb(_evt):
            state["core"].submit(
              {"op": "ankle", "side": side, "pitch": a.value, "roll": b.value}
            )
            cr = state["core"].ankle_inverse(side, a.value, b.value)
            guard["busy"] = True
            for jn, v in zip((f"{side}_crank_A_joint", f"{side}_crank_B_joint"), cr):
              if jn in sliders:
                lo2, hi2 = c.clip(jn)
                sliders[jn][0].value = float(np.clip(v, lo2, hi2))
                sliders[jn][1].value = float(v)
            guard["busy"] = False

          a.on_update(_cb)
          b.on_update(_cb)

        _mk_ankle(s, sp, sr)

  # ------------------------------------------------------------------ Policy (P2)
  pol_state: dict = {}
  with gui.add_folder("Policy"):
    baked_pol = sorted(
      p.name[: -len(".policy_contract.json")]
      for p in Path(CACHE_DIR).glob("*.policy_contract.json")
    )
    compat = []
    for n in baked_pol:
      try:
        pcj = json.loads((Path(CACHE_DIR) / f"{n}.policy_contract.json").read_text())
        if pcj.get("model_contract_sha") == c.contract_sha:
          compat.append(n)
      except Exception:
        pass
    gui.add_markdown(
      f"{len(compat)} of {len(baked_pol)} baked policies match this model's contract "
      f"`{c.contract_sha[:12]}`. A policy baked against another model is REFUSED: its "
      "default pose is the action offset, so loading it would bias every joint target.\n\n"
      "Bake one with `run.py bake policy --pt <run_dir>/model_N.pt --variant "
      f"{c.variant}`."
    )
    pol_dd = gui.add_dropdown(
      "policy", tuple(compat) or ("(none baked for this model)",),
      initial_value=(compat[0] if compat else "(none baked for this model)"),
    )
    pol_load = gui.add_button("load policy")
    pol_unload = gui.add_button("unload")
    # real_replay 가 빠져 있었습니다 (2026-09-15). 그래서 실물 값을 받아도 3D 로봇을
    # 그 값으로 움직이게 만들 방법이 화면에 없었고, 시뮬은 idle 로 제 물리를 돌며
    # 중력에 처졌습니다 -- 옆 숫자판은 맞는데 그림만 엉뚱하게 움직이는 증상입니다.
    # initial_value 는 core 의 지금 상태를 따릅니다 (huphy_viser 는 real_replay 로 켭니다).
    run_mode = gui.add_dropdown(
      "mode", ("idle", "manual", "policy_sim", "policy_shadow", "real_replay"),
      initial_value=core.mode,
    )
    vx = gui.add_slider("vx [m/s]", min=-1.0, max=2.5, step=0.05, initial_value=0.0)
    vy = gui.add_slider("vy [m/s]", min=-0.5, max=0.5, step=0.05, initial_value=0.0)
    wz = gui.add_slider("wz [rad/s]", min=-1.5, max=1.5, step=0.05, initial_value=0.0)
    shadow_follow_cb = gui.add_checkbox(
      "shadow_follow (policy_shadow only, local sim)", initial_value=core.shadow_follow
    )
    pol_info = gui.add_markdown("no policy loaded")
    mask_md = gui.add_markdown("")  # always-visible per-term source strip (design item 1)
    with gui.add_folder("observation sources", expand_by_default=False):
      src_md = gui.add_markdown(
        "Per observation TERM, sim or real. `real` is only ever READ during `policy_shadow` "
        "(`policy_sim` always uses sim); a stale/missing real value falls back to sim for "
        "that tick and shows up in the strip above and in `shadow_warnings`."
      )
      src_dd: dict = {}

    @shadow_follow_cb.on_update
    def _(_evt):
      state["core"].submit({"op": "shadow_follow", "value": shadow_follow_cb.value})

    def _refresh_policy_panel():
      core_ = state["core"]
      if core_.policy is None:
        pol_info.content = "no policy loaded"
        return
      pc = core_.policy_contract or {}
      layout = core_.obs_builder.describe()
      pol_info.content = (
        f"**{pc.get('name', '(uncontracted)')}**  \n"
        f"`{core_.policy.name}`  obs {core_.obs_builder.obs_dim}  act "
        f"{len(core_.act_names)}  \n"
        f"terms: " + ", ".join(f"{d['name']}({d['dim']})" for d in layout) + "  \n"
        f"ckpt `{str(pc.get('checkpoint', '?')).split('/')[-1]}`  "
        f"clip_actions {pc.get('clip_actions')}"
      )
      if not src_dd:
        for d in layout:
          h = gui.add_dropdown(d["name"], ("sim", "real"), initial_value="sim")
          src_dd[d["name"]] = h

          def _mk_src(nm, hh):
            @hh.on_update
            def _(_e):
              try:
                state["core"].obs_mux.set({nm: hh.value})
              except (KeyError, ValueError) as exc:
                hh.value = "sim"
                src_md.content = f"**{exc}**"

          _mk_src(d["name"], h)

    @pol_load.on_click
    def _(_evt):
      core_ = state["core"]
      name = pol_dd.value
      f = Path(CACHE_DIR) / f"{name}.policy_contract.json"
      if not f.exists():
        pol_info.content = f"**no such baked policy: {name}**"
        return
      pc = json.loads(f.read_text())
      try:
        core_.load_policy(onnx=pc["onnx"], policy_contract=pc)
      except Exception as exc:
        pol_info.content = f"**REFUSED - {type(exc).__name__}: {exc}**"
        return
      _refresh_policy_panel()

    @pol_unload.on_click
    def _(_evt):
      state["core"].clear_policy()
      run_mode.value = "manual"
      _refresh_policy_panel()

    @run_mode.on_update
    def _(_evt):
      try:
        state["core"]._apply_cmd({"op": "mode", "value": run_mode.value})
      except Exception as exc:
        pol_info.content = f"**{exc}**"
        run_mode.value = state["core"].mode

    def _push_cmd(_evt=None):
      state["core"].submit({"op": "cmd", "value": [vx.value, vy.value, wz.value]})

    for h in (vx, vy, wz):
      h.on_update(_push_cmd)

  state["refresh_policy"] = _refresh_policy_panel

  # ------------------------------------------------------------------ Gains
  with gui.add_folder("Gains", expand_by_default=False):
    gui.add_markdown(
      "`train` = the contract's kp/kd, the gains the policy was optimised against. "
      "`real` needs a hardware gain table in the contract (`real_gains`); the viewer will "
      "not invent one, because a sim-vs-robot response overlay is meaningless until the "
      "gains match. Hardware also encodes them differently per motor family (RS03/RS04 "
      "0-5000 vs 0-500 registers), which is why the table is per joint."
    )
    gsrc = gui.add_dropdown("source", ("train", "real"), initial_value="train")
    gtxt = gui.add_markdown("")

    def _gains_md():
      t = state["core"].gains_table()
      have_real = any(g.get("real_kp") is not None for g in t.values())
      if not have_real:
        rows = ["| joint | motor | kp | kd | kp train | kd train |", "|---|---|---|---|---|---|"]
        for n, g in t.items():
          rows.append(
            f"| {n.replace('_joint', '')} | {g.get('motor', '?')} | {g['kp']:.1f} | "
            f"{g['kd']:.2f} | {g['kp_train']:.1f} | {g['kd_train']:.2f} |"
          )
        return "\n".join(rows) + "\n\n*(no hardware gains received yet - real columns appear once telemetry reports them, R7)*"
      # P4/R7: real gains have arrived - show them alongside sim with a ratio, red when >5% off.
      rows = ["| joint | motor | kp sim | kp real | kp ratio | kd sim | kd real | kd ratio |",
              "|---|---|---|---|---|---|---|---|"]
      for n, g in t.items():
        def _cell(sim_v, real_v, ratio, flag):
          if real_v is None:
            return "-", "-"
          txt = f"{ratio:.3f}" if ratio is not None else "-"
          if flag:
            txt = f'<span style="color:#e45756;font-weight:bold">{txt}</span>'
          return f"{real_v:.2f}", txt
        rp, rrp = _cell(g["kp"], g.get("real_kp"), g.get("real_ratio_kp"), g.get("real_flag_kp"))
        rd, rrd = _cell(g["kd"], g.get("real_kd"), g.get("real_ratio_kd"), g.get("real_flag_kd"))
        rows.append(
          f"| {n.replace('_joint', '')} | {g.get('motor', '?')} | {g['kp']:.1f} | {rp} | {rrp} | "
          f"{g['kd']:.2f} | {rd} | {rrd} |"
        )
      return "\n".join(rows)

    gtxt.content = _gains_md()

    @gsrc.on_update
    def _(_evt):
      try:
        state["core"].set_gains(gsrc.value)
      except Exception as exc:
        gsrc.value = state["core"].gains_source
        gtxt.content = f"**{exc}**\n\n" + _gains_md()
        return
      gtxt.content = _gains_md()

  # ------------------------------------------------------------------ Script player (P4)
  with gui.add_folder("Script (same-target sequence)", expand_by_default=False):
    gui.add_markdown(
      "`{joint_names, rows:[[t_s, q...], ...]}` played in `manual` mode. The same file, "
      "played through the robot's own bridge (not implemented here - the viewer never "
      "transmits), gives a second recording `compare.py` can overlay against this one."
    )
    script_path = gui.add_text("script path", initial_value="")
    script_run_btn = gui.add_button("run")
    script_stop_btn = gui.add_button("stop")
    script_status = gui.add_markdown("no script running")

    @script_run_btn.on_click
    def _(_evt):
      try:
        info = state["core"].run_script(script_path.value)
        script_status.content = f"running `{info['run_id']}` - {info['duration_s']:.1f}s"
      except Exception as exc:
        script_status.content = f"**{type(exc).__name__}: {exc}**"

    @script_stop_btn.on_click
    def _(_evt):
      try:
        state["core"].stop_script()
        script_status.content = "stopped"
      except Exception as exc:
        script_status.content = f"**{type(exc).__name__}: {exc}**"

  # ------------------------------------------------------------------ Telemetry (P3)
  with gui.add_folder("Telemetry"):
    jmap_path = Path(__file__).parent / "bridge" / "joint_map_huphy.json"
    try:
      side_verified = bool(json.loads(jmap_path.read_text()).get("side_mapping_verified"))
    except Exception:
      side_verified = False
    if not side_verified:
      gui.add_markdown(
        "**UNVERIFIED** - the HUPHY limb/motor->sim map is USER-ASSERTED (L=left), not yet "
        "confirmed by the verification protocol's sign-sweep step (docs/121 section 5, "
        "step 2). Do not trust a sim<->real overlay until that step passes."
      )
    tele_md = gui.add_markdown("no telemetry received yet")

    gui.add_markdown("**Record**")
    rec_path = gui.add_text("path (blank = auto)", initial_value="")
    rec_start = gui.add_button("start recording")
    rec_stop = gui.add_button("stop recording")
    rec_status = gui.add_markdown("not recording")

    @rec_start.on_click
    def _(_evt):
      try:
        info = state["core"].start_recording(rec_path.value or None)
        rec_status.content = f"recording -> `{info['path']}`"
      except Exception as exc:
        rec_status.content = f"**{type(exc).__name__}: {exc}**"

    @rec_stop.on_click
    def _(_evt):
      try:
        info = state["core"].stop_recording()
        rec_status.content = f"stopped - {info['n_lines']} lines, {info['errors']} errors -> `{info['path']}`"
      except Exception as exc:
        rec_status.content = f"**{type(exc).__name__}: {exc}**"

    gui.add_markdown("**Replay** (file_replay mode)")
    replay_path = gui.add_text("recording path", initial_value="")
    replay_load = gui.add_button("load")
    replay_run = gui.add_dropdown("run mode", ("idle", "manual", "file_replay"), initial_value="idle")
    replay_speed = gui.add_slider("speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    replay_status = gui.add_markdown("no recording loaded")

    @replay_load.on_click
    def _(_evt):
      try:
        info = state["core"].load_replay(replay_path.value)
        replay_status.content = f"loaded {info['n_rows']} rows, {info['duration_s']:.1f} s"
      except Exception as exc:
        replay_status.content = f"**{type(exc).__name__}: {exc}**"

    @replay_run.on_update
    def _(_evt):
      try:
        state["core"]._apply_cmd({"op": "mode", "value": replay_run.value})
      except Exception as exc:
        replay_status.content = f"**{exc}**"
        replay_run.value = state["core"].mode

    @replay_speed.on_update
    def _(_evt):
      state["core"].submit({"op": "replay_speed", "speed": replay_speed.value})

  def _telemetry_md() -> str:
    s = state["core"].snapshot()
    if not s:
      return "no telemetry received yet"
    t = s.get("telemetry", {})
    if not t.get("rx_count"):
      return "no telemetry received yet (no /ws/in or UDP bridge traffic)"
    lines = [
      f"rx **{t['rx_hz']:.1f} Hz**  age {t['age_s']}s  {'**STALE**' if t['stale'] else 'fresh'}  "
      f"seq gaps {t['seq_gaps']}",
      f"clock offset {t['clock_offset_ms']} ms  jitter {t['clock_jitter_ms']} ms "
      f"{'**(grey: >15 ms)**' if t.get('jitter_grey') else ''}",
      f"contract mismatches {t['contract_mismatches']}  wrap events {t['wrap_events']}  "
      f"bridge errors {t['bridge_errors']}",
    ]
    if t.get("range_violations"):
      lines.append(f"range violations: {t['range_violations']}")
    if t.get("bridge_last_error"):
      lines.append(f"last bridge error: `{t['bridge_last_error']}`")
    ss = t.get("sign_sanity") or {}
    red = {n: v for n, v in ss.items() if v.get("red")}
    if red:
      lines.append("**SIGN SANITY RED**: " + ", ".join(f"{n} ({v['disagree_frac']:.0%})" for n, v in red.items()))
    if "replay" in t:
      r = t["replay"]
      if r.get("playing"):
        lines.append(f"replay: {r['cursor']}/{r['n_rows']} rows ({r['frac']:.0%}) speed {r['speed']:.1f}x")
    return "  \n".join(lines)

  state["telemetry_md"] = lambda: setattr(tele_md, "content", _telemetry_md())

  # ------------------------------------------------------------------ Plots
  chan_names = []
  for n in c.action_joint_names:
    chan_names += [f"{n}|q", f"{n}|target", f"{n}|tau", f"{n}|real"]
  ring = RingPlot(chan_names, int(PLOT_WINDOW_S * PLOT_HZ))
  sel_joint: dict[str, object] = {}
  with gui.add_folder("Plots"):
    gui.add_markdown(
      f"ring buffer {PLOT_WINDOW_S:.0f} s @ {PLOT_HZ:.0f} Hz, pushed at {PLOT_PUSH_HZ:.0f} Hz, "
      f"max {MAX_CHANNELS} channels. Nothing accumulates."
    )
    sig_q = gui.add_checkbox("q", initial_value=True)
    sig_t = gui.add_checkbox("target", initial_value=True)
    sig_u = gui.add_checkbox("tau [N*m]", initial_value=False)
    sig_r = gui.add_checkbox("real (P3)", initial_value=False)
    with gui.add_folder("joints", expand_by_default=False):
      for n in c.action_joint_names:
        sel_joint[n] = gui.add_checkbox(
          n.replace("_joint", ""), initial_value=(n == c.action_joint_names[3])
        )
    warn = gui.add_markdown("")
    plot = gui.add_uplot(
      data=(np.zeros(2), np.zeros(2)),
      series=(
        {"label": "t [s]"},
        {"label": "-", "stroke": _COLORS[0]},
      ),
      title="joint traces",
      aspect=1.4,
    )

  def selected_channels() -> list[str]:
    sigs = [s for s, on in (("q", sig_q.value), ("target", sig_t.value), ("tau", sig_u.value), ("real", sig_r.value)) if on]
    out = [f"{n}|{s}" for n in c.action_joint_names if sel_joint[n].value for s in sigs]
    return out[:MAX_CHANNELS], len(out)

  def push_plot():
    chans, total = selected_channels()
    warn.content = (
      f"showing {len(chans)} of {total} selected channels (cap {MAX_CHANNELS})"
      if total > MAX_CHANNELS
      else ""
    )
    if not chans:
      return
    v = ring.view(chans)
    if v is None:
      return
    t, ys = v
    plot.data = (t, *ys)
    plot.series = (
      {"label": "t [s]"},
      *(
        {"label": ch.replace("_joint", ""), "stroke": _COLORS[k % len(_COLORS)]}
        for k, ch in enumerate(chans)
      ),
    )

  state["push_plot"] = push_plot

  # ------------------------------------------------------------------ Status
  with gui.add_folder("Status"):
    status_md = gui.add_markdown("")

  jc = c.raw["joint_contract"]

  def readout():
    s = state["core"].snapshot()
    if not s:
      return
    qi = {n: i for i, n in enumerate(s["joint_names"])}
    for k, n in enumerate(s["act_names"]):
      q = s["q"][qi[n]]
      tau = s["tau"][k]
      tgt = s["target"][k]
      extra = ""
      if jc[n]["mirrored"] and jc[n]["travel_sign"]:
        extra = f"  phys {jc[n]['travel_sign'] * q:+.3f}"
      if n in readouts:
        readouts[n].value = f"q {q:+.3f}  tgt {tgt:+.3f}  tau {tau:+7.2f}{extra}"
    b = s["base"]
    lines = [
      f"**{s['variant']}**  mode `{s['mode']}`  sim t {s['t']:.2f} s",
      "",
      f"physics **{s['rates']['phys_hz']:.0f} Hz**  control **{s['rates']['ctrl_hz']:.0f} Hz**  "
      f"drops {s['rates']['drops']}",
      "",
      f"base `{b['mode']}` pos ({b['pos'][0]:+.3f}, {b['pos'][1]:+.3f}, {b['pos'][2]:+.3f}) "
      f"rpy ({math.degrees(b['rpy'][0]):+.1f}, {math.degrees(b['rpy'][1]):+.1f}, "
      f"{math.degrees(b['rpy'][2]):+.1f}) deg  ground {'on' if b['ground'] else 'OFF'}",
    ]
    st = s.get("string")
    if st and b["mode"] == "string":
      state_word = "**TAUT**" if st["taut"] else "slack"
      string_md.content = (
        f"z_set **{st['z_set']:.3f} m**  length {st['ten_length']:.3f}/{st['length']:.3f} m  "
        f"{state_word}  tension **{st['tension_N']:.1f} N**"
      )
    elif st:
      string_md.content = f"(not in `string` mode; last z_set {st['z_set']:.3f} m)"
    if "ankle_derived" in s:
      ad = " ".join(
        f"{k}: pitch {v['pitch']:+.3f} roll {v['roll']:+.3f}" for k, v in s["ankle_derived"].items()
      )
      lines += ["", f"ankle (model state) {ad}", f"loop closure {s['closure_mm']:.4f} mm"]
    if s.get("imu"):
      g = s["imu"].get("imu_upvector")
      w = s["imu"].get("imu_ang_vel")
      if g and w:
        lines += [
          "",
          f"imu up ({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f})  gyro "
          f"({w[0]:+.2f}, {w[1]:+.2f}, {w[2]:+.2f}) rad/s",
        ]
    p = s.get("policy")
    if p:
      lines += [
        "",
        f"policy `{p['name'] or p['kind']}` cmd ({p['cmd'][0]:+.2f}, {p['cmd'][1]:+.2f}, "
        f"{p['cmd'][2]:+.2f})  driving {'YES' if p['driving'] else 'no'}  "
        f"obs sources `{p['source_mask']}`",
      ]
      # Always-visible 5-cell source strip (design item 1): green=sim, orange=real, and a
      # cell shown red when the operator ASKED for real but a stale/missing value forced a
      # sim fallback THIS tick - the one state that must never be silently invisible.
      req = p.get("obs_sources", {})
      eff = p.get("obs_sources_effective", {}) or req
      cells = []
      for name in req:
        r, e = req.get(name, "sim"), eff.get(name, "sim")
        if r == "real" and e == "sim":
          color, label = "#e45756", f"{name}: R->S (stale)"
        elif e == "real":
          color, label = "#f58518", f"{name}: real"
        else:
          color, label = "#54a24b", f"{name}: sim"
        cells.append(f'<span style="background:{color};color:white;padding:1px 6px;'
                     f'border-radius:3px;margin-right:3px;font-size:11px">{label}</span>')
      warns = p.get("shadow_warnings") or []
      mask_md.content = "".join(cells) + ("\n\n**" + "; ".join(warns) + "**" if warns else "")
    from .api import rss_mb

    lines += ["", f"RSS {rss_mb():.0f} MB  gains `{s.get('gains_source', 'train')}`"]
    if s.get("warnings"):
      lines += ["", "**" + " / ".join(s["warnings"]) + "**"]
    status_md.content = "\n".join(lines)
    state["telemetry_md"]()
    gtxt.content = _gains_md()  # R7: refresh so a newly-received hardware gains report shows

  state["readout"] = readout

  # -------------------------------------------------- sim hook -> ring buffer
  last = {"t": -1.0}

  def hook(snap: dict):
    if snap["t"] - last["t"] < 1.0 / PLOT_HZ:
      return
    last["t"] = snap["t"]
    qi = {n: i for i, n in enumerate(snap["joint_names"])}
    vals = {}
    for k, n in enumerate(snap["act_names"]):
      vals[f"{n}|q"] = snap["q"][qi[n]]
      vals[f"{n}|target"] = snap["target"][k]
      vals[f"{n}|tau"] = snap["tau"][k]
    # P3: the real channel, when present - omitted (not zeroed) for a joint with no data
    # yet, so the ring buffer holds its last real value rather than showing a false zero.
    real_q = state["core"].real.snapshot_joints()
    for n, v in real_q.items():
      if v["q"] is not None:
        vals[f"{n}|real"] = v["q"]
    ring.push(snap["t"], vals)

  core.add_hook(hook)
  state["ring"] = ring


def _swap_variant(server: viser.ViserServer, state: dict, want: str) -> None:
  """Tear the whole panel down and rebuild it on another baked model."""
  from .sim_core import SimCore

  old = state["core"]
  old.stop()
  c = load_contract(CACHE_DIR, want)
  new = SimCore(c)
  server.scene.reset()
  server.gui.reset()
  state["core"] = new
  state["freshness"] = c.freshness()
  _mount(server, state)
  new.start()
