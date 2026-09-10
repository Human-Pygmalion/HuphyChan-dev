"""Interactive per-joint MJCF viewer for the Pygmalion robot, served over viser.

Lets a human move every joint with a +/- slider in a browser to eyeball ROM and motor
sign convention against the design intent. Read-only: never edits the XML/URDF.

Two robot-body variants x two ankle modes = 4 loadable combinations (docs/66 registry,
docs/117 3-variant regen, 2026-09-03 latest armfix/v30_proxyfix set):
  --variant fulldof   FullDoF_..._v30_proxyfix{,_loop}.xml   full body: legs + waist_yaw +
                       both arms.
  --variant legonly   LegOnly_..._v30_proxyfix{,_loop}.xml   legs only, waist-yaw actuator
                       MASS kept but no waist/arm JOINTS at all (they simply don't exist in
                       this XML -- the joint groups below skip themselves automatically).
  --model loop        2-RSU closed-loop ankle (crank A/B driven, ankle_pitch/roll PASSIVE).
                       This is the AB mode.
  --model proxyfix    serial ankle: ankle_pitch/roll are direct hinges, no crank. RP mode.

Why MJCF, not the URDF the model-build step also emits: the AB/loop ankle's closed
kinematic loop (rod end -> ball joint on the foot) cannot be expressed in URDF at all --
build_robot.py's URDF output leaves it as a plain comment ("loop closure, not expressible
in URDF ... see the <equality><connect> in the MJCF") and instead lists ankle_pitch/roll
AND crank_A/B AND the rod universal joints as independent, kinematically over-determined
tree joints. Driving any of those directly in the URDF would show a pose the real
mechanism can never reach. The MJCF's `<equality><connect>` is the only representation
that is physically correct for AB mode, so that is what this viewer loads and settles via
mj_step() (see the AB handling note below) for both --variant choices.

Closed-loop ankle handling (crank -> ankle_pitch/roll): mj_forward() alone does NOT
kinematically resolve the `equality/connect` constraints that close the rod ends onto the
foot's ball sites -- that only happens through the constraint solver during mj_step(). So
for the `loop` model this script adds two small PD position servos per leg on
L/R_crank_A/B_joint (Kp=22.3, Kd=1.41, forcerange +-60 N*m -- same numbers as
tools/robot_model/loop_ankle_verify.py's validated settle()), ramps their ctrl target over
half of a short step burst and holds for the rest, while every OTHER joint's qpos is
pinned (qpos overwritten, qvel zeroed) each step so only the crank/rod/ankle sub-chain
actually integrates. This reproduces loop_ankle_verify.py's settle() output exactly
(checked against tools/robot_model/loop_ankle_verify.json's jam grid before writing this
file: e.g. cA=cB=-40deg -> pitch 31.89 roll 0.38, closure_err 0.002 mm here vs
31.89/0.38 in the json). A naive hard qpos-snap of the crank angle (no actuator, no
ramp) was tried first and went numerically unstable (QACC NaN) -- do not "simplify" this
away.
Consequence for the UI: crank_A/crank_B are the sliders (they are the real RS03 DOFs);
ankle_pitch/ankle_roll are shown as a read-only derived readout, never a slider, for the
loop model. For the proxyfix model ankle_pitch/ankle_roll ARE ordinary sliders.

Sign convention cross-check: tools/robot_model/motor_sign_convention.json records, per
joint, whether the URDF/XML positive-qpos direction has been confirmed to match the real
motor's positive (CW-from-outside) rotation ("matches": true/false/null). This script
looks each joint up there and prints a badge next to the slider. It does NOT re-derive the
sign itself -- the badge is only as good as that file. Only L_-prefixed joints (+
waist_yaw, which is on the midline) have been audited; R_-prefixed joints fall back to the
L_ entry with a note, since the audit's own _meta says the right side "not synthesized".
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

REPO = Path(__file__).resolve().parents[2]
XMLS = REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
SIGN_JSON = REPO / "tools/robot_model/motor_sign_convention.json"

MODELS = {
  "fulldof": {
    "loop": "FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml",
    "proxyfix": "FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix.xml",
  },
  "legonly": {
    "loop": "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml",
    "proxyfix": "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix.xml",
  },
}

CRANK_KP, CRANK_KD, CRANK_FORCE = 22.3, 1.41, 60.0
SETTLE_STEPS = 400
SETTLE_DT = 0.001

# passive 2-RSU linkage DOFs: never exposed, not even read-only (meaningless to a user)
LOOP_HIDDEN = [f"{s}_{p}" for s in "LR" for p in ("rod_A_u1", "rod_A_u2", "rod_B_u1", "rod_B_u2")]
LOOP_PASSIVE_READOUT = [f"{s}_ankle_{a}_joint" for s in "LR" for a in ("pitch", "roll")]
CRANK_JOINTS = [f"{s}_crank_{t}_joint" for s in "LR" for t in ("A", "B")]

GROUPS = [
  ("Left leg", ["L_hip_pitch_joint", "L_hip_roll_joint", "L_hip_yaw_joint", "L_knee_joint",
                "L_crank_A_joint", "L_crank_B_joint", "L_ankle_pitch_joint", "L_ankle_roll_joint"]),
  ("Right leg", ["R_hip_pitch_joint", "R_hip_roll_joint", "R_hip_yaw_joint", "R_knee_joint",
                 "R_crank_A_joint", "R_crank_B_joint", "R_ankle_pitch_joint", "R_ankle_roll_joint"]),
  ("Torso", ["waist_yaw_joint"]),
  ("Arms", ["L_shoulder_pitch_joint", "L_shoulder_roll_joint",
            "R_shoulder_pitch_joint", "R_shoulder_roll_joint"]),
]


def load_sign_convention() -> dict:
  if not SIGN_JSON.exists():
    return {}
  return json.loads(SIGN_JSON.read_text())


def sign_badge(joint_name: str, sign_conv: dict) -> str:
  key = joint_name
  note = ""
  if key not in sign_conv and key.startswith("R_"):
    key = "L_" + key[2:]
    note = " (R inferred from L, not independently verified)"
  entry = sign_conv.get(key)
  if entry is None:
    return "no motor-sign audit for this joint"
  m = entry.get("matches")
  motor = entry.get("motor", "?")
  if m is True:
    return f"sign OK vs {motor}{note}"
  if m is False:
    return f"** SIGN INVERTED ** vs {motor}{note}"
  return f"sign UNVERIFIED (CAD-only) vs {motor}{note}"


def build_model(variant: str, kind: str):
  xml = XMLS / MODELS[variant][kind]
  assert xml.exists(), xml
  spec = mujoco.MjSpec.from_file(str(xml))
  spec.option.gravity[:] = [0, 0, 0]  # pure kinematic pose review, no gravity sag
  spec.option.timestep = SETTLE_DT
  crank_actuators = {}
  if kind == "loop":
    for s in "LR":
      for t in "AB":
        jn = f"{s}_crank_{t}_joint"
        a = spec.add_actuator()
        a.name = f"{jn}_servo"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = jn
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = CRANK_KP
        a.biasprm[1] = -CRANK_KP
        a.biasprm[2] = -CRANK_KD
        a.forcerange[:] = [-CRANK_FORCE, CRANK_FORCE]
        a.forcelimited = True
        crank_actuators[jn] = a.name
  m = spec.compile()
  return m, crank_actuators


class ComMarkers:
  """Per-body COM coordinate frame + a "name / mass" label, toggled by one GUI checkbox.
  Reads d.xipos/d.ximat (world-frame inertial-frame position/orientation MuJoCo already
  computes every mj_forward) so the markers track the live pose the same way the mesh
  scene does -- no extra kinematics here. Skipped for body 0 (world) and any body with
  zero mass (nothing meaningful to show; a couple of loop sub-bodies are like this)."""

  def __init__(self, server: viser.ViserServer, m: mujoco.MjModel):
    self.m = m
    self.frames: dict[int, viser.FrameHandle] = {}
    self.labels: dict[int, viser.LabelHandle] = {}
    with server.gui.add_folder("COM / mass"):
      self.toggle = server.gui.add_checkbox(
        "show COM markers", initial_value=False,
        hint="coordinate frame + mass label at each link's centre of mass")
      self.toggle.on_update(lambda _: self._set_visible(self.toggle.value))
      for b in range(1, m.nbody):
        if m.body_mass[b] <= 0:
          continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        self.frames[b] = server.scene.add_frame(
          f"/com_frame/{name}", axes_length=0.05, axes_radius=0.003, visible=False)
        self.labels[b] = server.scene.add_label(
          f"/com_label/{name}", f"{name}  {m.body_mass[b]:.3f} kg", visible=False)

  def _set_visible(self, v: bool):
    for h in (*self.frames.values(), *self.labels.values()):
      h.visible = v

  def update(self, d: mujoco.MjData):
    if not self.toggle.value:
      return
    quat = np.empty(4)
    for b, frame in self.frames.items():
      pos = d.xipos[b]
      mujoco.mju_mat2Quat(quat, d.ximat[b])
      frame.position = pos
      frame.wxyz = quat
      self.labels[b].position = pos


def run_generic(xml_path: str, host: str, port: int, label: str):
  """Generic joint viewer for an arbitrary flat-hinge MJCF (no closed loop / crank
  handling) -- e.g. a teammate's model that isn't part of this repo's build pipeline.
  Every hinge joint gets a direct slider (qpos set + mj_forward, no physics settle needed:
  there is no equality constraint to resolve). Joints are grouped by the L_/R_ name prefix
  the way this repo's own models are named; anything else falls into one 'Joints' group.
  """
  assert os.path.exists(xml_path), xml_path
  m = mujoco.MjModel.from_xml_path(xml_path)
  d = mujoco.MjData(m)
  sign_conv = load_sign_convention()  # will mostly say "no audit" for a foreign model's
                                       # joint names -- that is correct, not a bug

  def jid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

  joint_names = []
  for j in range(m.njnt):
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
      continue                       # skip the free/floating-base joint, slide joints etc.
    joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))

  groups: dict[str, list[str]] = {}
  for jn in joint_names:
    if jn.startswith("L_"):
      groups.setdefault("Left", []).append(jn)
    elif jn.startswith("R_"):
      groups.setdefault("Right", []).append(jn)
    else:
      groups.setdefault("Joints", []).append(jn)

  mujoco.mj_forward(m, d)
  server = viser.ViserServer(host=host, port=port, label=label)
  scene = ViserMujocoScene(server, m, num_envs=1)
  com = ComMarkers(server, m)

  readout_handles = {}

  def refresh_readouts():
    for n, h in readout_handles.items():
      h.content = f"**{float(np.degrees(d.qpos[m.jnt_qposadr[jid(n)]])):.1f} deg**"

  server.gui.add_markdown(
    f"# {label}\nFile: `{xml_path}`\n\n"
    "Generic flat-hinge viewer: every joint below is a direct slider (qpos set + "
    "mj_forward), no closed-loop mechanism handling -- this model has none. "
    "Sign badges only fire for joint names this repo's own "
    "`tools/robot_model/motor_sign_convention.json` audit recognizes; a foreign model's "
    "joint names will mostly read \"no motor-sign audit for this joint\", which is "
    "expected, not an error."
  )
  server.gui.add_markdown("---")

  slider_map: dict[str, viser.GuiSliderHandle] = {}

  def apply(jn, val_deg):
    d.qpos[m.jnt_qposadr[jid(jn)]] = np.radians(val_deg)
    mujoco.mj_forward(m, d)
    refresh_readouts()
    scene.update_from_mjdata(d)
    com.update(d)

  for group_name, joints in groups.items():
    with server.gui.add_folder(group_name):
      for jn in joints:
        j_id = jid(jn)
        lo, hi = np.degrees(m.jnt_range[j_id])
        badge = sign_badge(jn, sign_conv)
        init_deg = float(np.degrees(d.qpos[m.jnt_qposadr[j_id]]))
        server.gui.add_markdown(f"**{jn}** — range {lo:.1f} to {hi:.1f} deg — {badge}")
        readout_handles[jn] = server.gui.add_markdown(f"**{init_deg:.1f} deg**")
        slider = server.gui.add_slider(jn, min=float(lo), max=float(hi), step=0.1,
                                        initial_value=init_deg)
        slider_map[jn] = slider
        slider.on_update(lambda _, jn=jn: apply(jn, slider_map[jn].value))

  with server.gui.add_folder("Reset"):
    def reset_all(_):
      for s in slider_map.values():
        s.value = 0.0
      d.qpos[:] = 0.0
      mujoco.mj_forward(m, d)
      refresh_readouts()
      scene.update_from_mjdata(d)
      com.update(d)
    server.gui.add_button("Zero all joints").on_click(reset_all)

  refresh_readouts()
  scene.update_from_mjdata(d)
  com.update(d)
  print(f"[mjcf_joint_viewer] xml={xml_path}")
  print(f"[mjcf_joint_viewer] serving on http://{host}:{port}")
  while True:
    time.sleep(1.0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", choices=list(MODELS), default="fulldof",
                   help="fulldof = legs+waist_yaw+arms; legonly = legs only "
                        "(waist-yaw actuator mass kept, no waist/arm joints)")
  ap.add_argument("--model", choices=["loop", "proxyfix"], default="loop",
                   help="loop = AB 2-RSU closed-loop ankle (정본/default); "
                        "proxyfix = RP serial ankle")
  ap.add_argument("--xml", default=None,
                   help="load an arbitrary external flat-hinge MJCF instead of a Pygmalion "
                        "model (generic viewer, no closed-loop handling)")
  ap.add_argument("--label", default=None, help="viser server label (with --xml)")
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--port", type=int, default=8098)
  args = ap.parse_args()

  if args.xml:
    run_generic(args.xml, args.host, args.port,
                args.label or f"Joint viewer [{os.path.basename(args.xml)}]")
    return

  m, crank_actuators = build_model(args.variant, args.model)
  d = mujoco.MjData(m)
  sign_conv = load_sign_convention()

  def jid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

  def has_joint(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0

  is_loop = args.model == "loop"
  hidden = set(LOOP_HIDDEN) if is_loop else set()
  passive_readout = set(LOOP_PASSIVE_READOUT) if is_loop else set()
  crank_set = set(CRANK_JOINTS) if is_loop else set()

  driven_qpos_idx = set()
  if is_loop:
    for n in LOOP_HIDDEN + CRANK_JOINTS + LOOP_PASSIVE_READOUT:
      driven_qpos_idx.add(int(m.jnt_qposadr[jid(n)]))
  pinned_qpos_idx = sorted(set(range(m.nq)) - driven_qpos_idx)
  driven_dof_idx = set()
  if is_loop:
    for n in LOOP_HIDDEN + CRANK_JOINTS + LOOP_PASSIVE_READOUT:
      driven_dof_idx.add(int(m.jnt_dofadr[jid(n)]))
  pinned_dof_idx = sorted(set(range(m.nv)) - driven_dof_idx)

  d.qpos[:] = 0.0
  d.qpos[0:3] = [0, 0, 1.0]  # base pinned in the air, matches tools/robot_model precedent
  d.qpos[3:7] = [1, 0, 0, 0]
  target = d.qpos.copy()
  mujoco.mj_forward(m, d)

  server = viser.ViserServer(host=args.host, port=args.port,
                              label=f"Pygmalion joint viewer [{args.variant}/{args.model}]")
  scene = ViserMujocoScene(server, m, num_envs=1)
  com = ComMarkers(server, m)

  def settle_loop():
    """Ramp all 4 crank servo targets to `target` over half the burst, hold for the
    rest, while every other DOF's qpos is pinned each step (see module docstring)."""
    if not is_loop:
      return
    a_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, crank_actuators[jn]) for jn in CRANK_JOINTS]
    c0 = np.array([d.ctrl[a] for a in a_ids])
    c1 = np.array([target[m.jnt_qposadr[jid(jn)]] for jn in CRANK_JOINTS])
    for k in range(SETTLE_STEPS):
      f = min(1.0, k / (SETTLE_STEPS / 2))
      for a, v0, v1 in zip(a_ids, c0, c1):
        d.ctrl[a] = v0 + f * (v1 - v0)
      d.qpos[pinned_qpos_idx] = target[pinned_qpos_idx]
      d.qvel[pinned_dof_idx] = 0.0
      mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

  def apply(joint_name: str, val_deg: float):
    target[m.jnt_qposadr[jid(joint_name)]] = np.radians(val_deg)
    if joint_name in crank_set:
      settle_loop()
    else:
      d.qpos[m.jnt_qposadr[jid(joint_name)]] = np.radians(val_deg)
      mujoco.mj_forward(m, d)
    refresh_readouts()
    scene.update_from_mjdata(d)
    com.update(d)

  readout_handles = {}

  def refresh_readouts():
    for n, h in readout_handles.items():
      cur_deg = float(np.degrees(d.qpos[m.jnt_qposadr[jid(n)]]))
      if n in crank_set:
        cmd_deg = float(np.degrees(target[m.jnt_qposadr[jid(n)]]))
        h.content = f"**actual (servo-limited): {cur_deg:.1f} deg**  (commanded {cmd_deg:.1f} deg)"
      else:
        h.content = f"**{cur_deg:.1f} deg**"

  variant_label = "FullDoF (legs + waist_yaw + arms)" if args.variant == "fulldof" \
    else "LegOnly (legs only — no waist/arm joints)"
  server.gui.add_markdown(
    f"# Pygmalion joint viewer — `{args.variant}` / `{args.model}` model\n"
    f"File: `{MODELS[args.variant][args.model]}`\n"
    f"Robot body: **{variant_label}**\n\n"
    + ("This is the **AB / 2-RSU closed-loop** ankle mode — the one the current main "
       "training run (v2s1_AB / legonly_ab_*) uses. "
       "**Crank A/B sliders drive the real RS03 motors**; ankle pitch/roll below them are "
       "PASSIVE and shown read-only, resolved by stepping MuJoCo's constraint solver "
       "(not simple forward kinematics) each time a crank moves — see the module "
       "docstring for why. Switch to `--model proxyfix` for the RP serial-ankle mode "
       "where ankle pitch/roll are direct sliders instead."
       if is_loop else
       "This is the **RP / serial** ankle mode — ankle pitch/roll are ordinary direct "
       "hinges here, no closed-loop mechanism. Use `--model loop` for the AB mode that "
       "the current main training run actually uses.")
  )
  server.gui.add_markdown("---")

  slider_map: dict[str, viser.GuiSliderHandle] = {}
  for group_name, joints in GROUPS:
    joints = [j for j in joints if has_joint(j) and j not in hidden]
    if not joints:
      continue
    with server.gui.add_folder(group_name):
      for jn in joints:
        j_id = jid(jn)
        lo, hi = np.degrees(m.jnt_range[j_id])
        badge = sign_badge(jn, sign_conv)
        init_deg = float(np.degrees(d.qpos[m.jnt_qposadr[j_id]]))
        if jn in passive_readout:
          server.gui.add_markdown(f"**{jn}** (passive, follows crank A/B) — range "
                                   f"{lo:.1f} to {hi:.1f} deg — {badge}")
          readout_handles[jn] = server.gui.add_markdown(f"**{init_deg:.1f} deg**")
          continue
        label = jn + ("  [driven RS03 crank]" if jn in crank_set else "")
        server.gui.add_markdown(f"**{label}** — range {lo:.1f} to {hi:.1f} deg — {badge}")
        readout_handles[jn] = server.gui.add_markdown(f"**{init_deg:.1f} deg**")
        slider = server.gui.add_slider(jn, min=float(lo), max=float(hi), step=0.1,
                                        initial_value=init_deg)
        slider_map[jn] = slider
        slider.on_update(lambda _, jn=jn: apply(jn, slider_map[jn].value))

  with server.gui.add_folder("Reset"):
    def reset_all(_):
      for s in slider_map.values():
        s.value = 0.0  # fires on_update per-slider (redundant settle_loop calls, harmless)
      target[:] = 0.0
      target[0:3] = [0, 0, 1.0]
      target[3:7] = [1, 0, 0, 0]
      d.qpos[:] = target
      d.qvel[:] = 0.0
      mujoco.mj_forward(m, d)
      refresh_readouts()
      scene.update_from_mjdata(d)
      com.update(d)
    server.gui.add_button("Zero all joints").on_click(reset_all)

  refresh_readouts()
  scene.update_from_mjdata(d)
  com.update(d)

  print(f"[mjcf_joint_viewer] variant={args.variant} model={args.model} "
        f"({MODELS[args.variant][args.model]})")
  print(f"[mjcf_joint_viewer] serving on http://{args.host}:{args.port}  "
        f"(browse http://<this-machine-lan-ip>:{args.port} or use SSH -L {args.port}:localhost:{args.port})")
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
