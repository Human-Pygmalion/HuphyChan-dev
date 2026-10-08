"""Bake a runnable ``.mjb`` + contract out of the mjlab training environment.

Why bake at all: the Pygmalion MJCF in ``asset_zoo`` has **no actuators, no floor and no
keyframe** (nu=0).  All three are attached by mjlab at env-build time from the EntityCfg, so
a viewer that loads the XML directly would be driving a robot the trainer never trains
(``tools/sim2sim/mujoco_ab_loop_drift.py`` documents the same conclusion).  So we build the
env once per variant, take the *scene spec* mjlab compiled, add the base anchor to it, and
save the compiled model to disk.  After that the viewer needs neither mjlab nor torch.

What is added to the scene spec (and nothing else - the robot XML is never edited):
  * ``pyg_anchor``  mocap body, no geom (a geom would add mass; the bake asserts the total
    mass is bit-identical to the env's), one tiny visual site.
  * ``base_weld``   equality/weld  anchor <-> base_link, inactive.   -> base mode ``fixed``
  * ``base_pivot``  equality/connect base_link <-> anchor, inactive. -> base mode ``pivot``

Both equalities are given a stiff solref/solimp: with MuJoCo's default (0.02, 1) a 23 kg
robot hanging off the weld sags 3.7e-4 m and keeps creeping 1.9e-4 m per 2 s, which is not a
"fixed" base.  At solref (0.002, 1) / solimp (0.9999, 0.99999, 1e-5) the same test gives
2.6e-8 m of offset and 1.0e-8 m of drift.  Gravity is NEVER touched.

Usage (repo root, mjlab venv, CPU only - the GPU belongs to the trainer):

    CUDA_VISIBLE_DEVICES="" mujoco-sim/mjlab/.venv/bin/python3 tools/pygviewer/run.py \
        bake model --variant LegOnly-AB
    ... --all          # every variant, one subprocess each (one env in RAM at a time)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import CACHE_DIR, REPO, VARIANTS
from .contract import CONTRACT_VERSION, canonical_sha, sha256_file

XML_DIR = Path(REPO) / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
CONSTANTS_PY = (
  Path(REPO) / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py"
)
ENVELOPE = Path(REPO) / "pygmalion_locomotion/assets/pygmalion_v2/ankle_rp_envelope.json"
LOOP_VERIFY = Path(REPO) / "tools/robot_model/loop_ankle_verify.json"

# The v30 armfix/proxyfix family is the current model generation (docs/111, docs/112).
# AB selects the loop MJCF through PYG_MODEL_TAG (the loop branch appends "_loop.xml");
# RP selects the serial twin through PYG_V2_XML - docs/112 L44: the loop branch has no
# PYG_V2_XML override at all, so using the wrong variable silently loads pygmalion_v3.
MODEL_STEM = "{dof}_prototype-tempmass-motormeasured-armfix_v30_proxyfix"

# Flags the current runs train under (docs/experiments/2026-09-03_legonly_ab_v2.md
# section 1b-4, from that run's repro/launch_manifest.json).  PYG_INIT_MID is meaningless
# unless PYG_INIT_BENT selects the bent keyframe, so both are on: the viewer's zero-action
# pose must be the pose the policy's actions are offsets from.
BASE_TOGGLES = {
  "PYG_V2": "1",
  "PYG_INIT_BENT": "1",
  "PYG_INIT_MID": "1",
  "PYG_MOTOR_MEAS": "1",
  "PYG_TN": "1",
  "PYG_SAFE_TARGET_CLIP": "1",
  "PYG_STUDENT_TEACHER": "1",
  "PYG_ARM_ABD_DEG": "15",
}

TASK = "Mjlab-Velocity-Flat-Pygmalion"

# Anchor equality tuning - see module docstring.
EQ_SOLREF = (0.002, 1.0)
EQ_SOLIMP = (0.9999, 0.99999, 1.0e-5, 0.5, 2.0)

# Safety-tether ("string") mode: a 2-site spatial tendon between the mocap anchor and a site
# on base_link, LIMITED to length [0, STRING_L0].  A MuJoCo tendon limit is a one-directional
# (rope, not spring) constraint - slack below the limit, taut at it - which is exactly a real
# overhead harness: it catches a falling base, it never holds one up that is standing fine.
# STRING_L0 only sets how far above the commanded catch height (z_set, a SimCore runtime
# state, not baked) the mocap anchor is parked; the number itself is arbitrary as long as it
# is positive, since sim_core.py re-derives the anchor's world Z as z_set + STRING_L0 every
# tick. Baked DISABLED (tendon_limited=False) - same "inert until a runtime mode switch turns
# it on" pattern as base_weld/base_pivot above. solref_limit is deliberately 10x SOFTER than
# EQ_SOLREF: a rigid catch on a 23 kg falling body is a jolt a real safety line does not give;
# 0.02 s settling was verified analytically (tools/pygviewer/pygviewer/bake.py string rig
# test, docs/121 section 9) to land within 2e-4 m of the commanded z_set with zero steady
# state error, so there was no reason to go stiffer.
STRING_L0 = 1.0
STRING_SOLREF_LIMIT = (0.02, 1.0)


def variant_env(variant: str, init_bent: bool = True) -> dict[str, str]:
  """PYG_* environment for one variant.  Raises on an unknown variant."""
  if variant not in VARIANTS:
    raise ValueError(f"unknown variant {variant!r}; one of {VARIANTS}")
  dof, ankle = variant.split("-")
  env = dict(BASE_TOGGLES)
  if not init_bent:
    env.pop("PYG_INIT_BENT")
  stem = MODEL_STEM.format(dof=dof)
  env["PYG_ANKLE_MODE"] = ankle
  if ankle == "AB":
    env["PYG_MODEL_TAG"] = stem
    xml = XML_DIR / f"{stem}_loop.xml"
  else:
    env["PYG_V2_XML"] = f"{stem}.xml"
    xml = XML_DIR / f"{stem}.xml"
  if not xml.exists():
    raise FileNotFoundError(f"{variant}: model XML missing: {xml}")
  return env


def variant_xml(variant: str) -> Path:
  dof, ankle = variant.split("-")
  stem = MODEL_STEM.format(dof=dof)
  return XML_DIR / (f"{stem}_loop.xml" if ankle == "AB" else f"{stem}.xml")


# --------------------------------------------------------------------------- worker
def bake_one(variant: str, cache_dir: str, init_bent: bool = True) -> dict:
  """Build the env for ``variant``, add the anchor, save ``.mjb`` + contract.  In-process."""
  os.environ.update(variant_env(variant, init_bent=init_bent))
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

  t0 = time.time()
  import re

  import mujoco
  import numpy as np

  import mjlab.tasks  # noqa: F401
  from mjlab.asset_zoo.robots.pygmalion import pygmalion_constants as PC
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  build_s = time.time() - t0
  robot = env.scene["robot"]
  m_env = env.sim.mj_model

  names: list[str] = list(robot.joint_names)
  default_q = np.asarray(robot.data.default_joint_pos[0]).flatten()

  # --- action order: the action term's OWN resolved list, not model order ------------
  term = env.action_manager._terms["joint_pos"]
  act_names = list(term._target_names)
  # term.cfg.scale is keyed by REGEX (".*_hip_pitch_joint"), not by joint name. Use the
  # term's own resolved _scale tensor so the contract is per-joint, in action order - a
  # consumer must never have to re-run a regex to find a joint's action scale.
  raw_scale = term._scale
  if isinstance(raw_scale, (int, float)):
    action_scale = {n: float(raw_scale) for n in act_names}
  else:
    sv = np.asarray(raw_scale.detach().cpu()).reshape(-1, len(act_names))[0]
    action_scale = {n: float(sv[i]) for i, n in enumerate(act_names)}
  action_scale_cfg = (
    float(term.cfg.scale)
    if isinstance(term.cfg.scale, (int, float))
    else {k: float(v) for k, v in term.cfg.scale.items()}
  )
  off = term._offset
  action_offset = (
    {n: float(off) for n in act_names}
    if isinstance(off, (int, float))
    else {
      n: float(np.asarray(off.detach().cpu()).reshape(-1, len(act_names))[0][i])
      for i, n in enumerate(act_names)
    }
  )

  # --- observation layout: the ACTOR group's own resolved terms, in order ------------
  # The term names are not stable across toggles: with PYG_STUDENT_TEACHER=1 the actor's
  # joint block is `motor_pos_history` (2 stacked frames of the 12 motor shafts, obs 45),
  # without it it is `joint_pos`+`joint_vel` (16 joints, obs 53).  So record the layout
  # rather than assume a name, and let P2's obs builder read it back.
  om = env.observation_manager

  def _term_joints(cfg_):
    ac = cfg_.params.get("asset_cfg")
    ids = getattr(ac, "joint_ids", None) if ac is not None else None
    if ids is None:
      return None
    return names[ids] if isinstance(ids, slice) else [names[i] for i in ids]

  obs_layout = []
  obs_names: list[str] = []
  for nm, tcfg in zip(om._group_obs_term_names["actor"], om._group_obs_term_cfgs["actor"]):
    jn = _term_joints(tcfg)
    obs_layout.append(
      dict(
        name=nm,
        func=getattr(tcfg.func, "__name__", str(tcfg.func)),
        params={k: str(v) for k, v in tcfg.params.items() if k != "asset_cfg"},
        joint_names=jn,
        history_length=int(getattr(tcfg, "history_length", 0) or 0),
      )
    )
    if jn and not obs_names:
      obs_names = jn
  obs_dim = int(om.group_obs_dim["actor"][0])

  # --- gains: the PD lives in the mjlab actuator objects, NOT in mjModel -------------
  gains: dict[str, dict] = {}
  family: dict[str, str] = {}
  for act in robot.actuators:
    pats = list(getattr(act.cfg, "target_names_expr", []))
    kp = np.asarray(act.stiffness).flatten()
    kd = np.asarray(act.damping).flatten()
    matched = [n for n in names if any(re.fullmatch(p, n) for p in pats)]
    for k, jn in enumerate(matched):
      gains[jn] = dict(
        kp=round(float(kp[k] if kp.size > 1 else kp[0]), 6),
        kd=round(float(kd[k] if kd.size > 1 else kd[0]), 6),
      )
  for a in range(m_env.nu):
    jid = int(m_env.actuator_trnid[a, 0])
    jn = (mujoco.mj_id2name(m_env, mujoco.mjtObj.mjOBJ_JOINT, jid) or "").split("/")[-1]
    gains.setdefault(jn, {})["effort"] = round(float(m_env.actuator_forcerange[a, 1]), 4)
    gains[jn]["actuator_gainprm"] = [round(float(x), 6) for x in m_env.actuator_gainprm[a, :3]]
    gains[jn]["actuator_biasprm"] = [round(float(x), 6) for x in m_env.actuator_biasprm[a, :3]]
  for jn in act_names:
    family[jn] = "RS03" if ("crank" in jn or "hip_yaw" in jn) else "RS04"
  tn_curves = {
    f: [[round(float(w), 6), round(float(t), 4)] for w, t in PC.tn_curve(f)]
    for f in sorted(set(family.values()))
  }

  # --- per-DOF armature / damping / frictionloss for every joint ---------------------
  dof_props = {}
  jrange: dict[str, list[float]] = {}
  joint_contract: dict[str, dict] = {}
  for jn in names:
    j = m_env.joint(f"robot/{jn}")
    dof = int(m_env.jnt_dofadr[j.id])
    dof_props[jn] = dict(
      armature=round(float(m_env.dof_armature[dof]), 9),
      damping=round(float(m_env.dof_damping[dof]), 9),
      frictionloss=round(float(m_env.dof_frictionloss[dof]), 9),
    )
    lo, hi = (float(x) for x in m_env.jnt_range[j.id])
    jrange[jn] = [lo, hi]

  # mirrored / travel_sign come from pygmalion_constants and from the model's own axes,
  # never from a name regex.  BOTH tests are needed and they catch different things:
  #   range_mirrored  L_knee [0,+120] vs R_knee [-120,0]  (the v30 stiff-knee trap)
  #   axis_mirrored   L_ankle_roll axis -X vs R_ankle_roll axis +X with the SAME symmetric
  #                   range [-20,+20] - a range test alone calls that "not mirrored" and a
  #                   caller then sends both legs the same signed number for opposite tilts.
  axis_of = {jn: [float(v) for v in m_env.jnt_axis[m_env.joint(f"robot/{jn}").id]] for jn in names}
  for jn in names:
    stem = jn[2:] if jn[:2] in ("L_", "R_") else jn
    twin = ("R_" if jn.startswith("L_") else "L_") + stem if jn[:2] in ("L_", "R_") else None
    has_twin = bool(twin and twin in jrange)
    range_mirrored = bool(
      has_twin
      and (abs(jrange[jn][0] - jrange[twin][0]) > 1e-9 or abs(jrange[jn][1] - jrange[twin][1]) > 1e-9)
    )
    axis_mirrored = bool(
      has_twin and not np.allclose(axis_of[jn], axis_of[twin], atol=1e-6)
    )
    joint_contract[jn] = dict(
      axis=[round(v, 6) for v in axis_of[jn]],
      range=[round(v, 8) for v in jrange[jn]],
      travel_sign=(float(PC.joint_travel_sign(jn)) if jn in PC.PYG_JOINT_RANGE else None),
      mirrored=bool(range_mirrored or axis_mirrored),
      range_mirrored=range_mirrored,
      axis_mirrored=axis_mirrored,
      twin=twin if has_twin else None,
      limited=bool(m_env.jnt_limited[m_env.joint(f"robot/{jn}").id]),
    )

  safe_clip = {k: [float(v[0]), float(v[1])] for k, v in PC.safe_target_clip().items()}

  # --- spawn height: lowest sole point at the default pose --------------------------
  spawn_base_z = _spawn_base_z(mujoco, np, m_env, names, default_q)
  # ... and how far the keyframe mjlab actually resets to puts the soles UNDER the floor.
  # This is not a viewer artefact: pygmalion_constants._v2_standing_z() reads
  # `standing_base_z` out of the pygmalion_v2 validation file and the v30 build is a
  # different robot, so every training reset starts with the feet buried and the solver
  # pushes them out over the first ~20 steps.  Recorded, not corrected - the viewer must
  # reset exactly the way the trainer does.
  kf_pen = round(float(spawn_base_z - float(PC.KNEES_BENT_KEYFRAME.pos[2])), 6)

  # --- keyframes --------------------------------------------------------------------
  key_qpos = [float(x) for x in m_env.key_qpos[0]] if m_env.nkey else None
  keyframes = {
    "knees_bent": dict(
      base_z=float(PC.KNEES_BENT_KEYFRAME.pos[2]),
      joint_pos={n: round(float(q), 8) for n, q in zip(names, default_q)},
      source="env keyframe 0 (= the init state this bake's PYG_* toggles selected)",
      qpos=key_qpos,
      sole_penetration_m=kf_pen,
    ),
    "home": dict(
      base_z=float(PC.HOME_KEYFRAME.pos[2]),
      joint_pos={n: 0.0 for n in names},
      source="pygmalion_constants.HOME_KEYFRAME (all joints 0)",
      qpos=None,
    ),
  }

  # ----------------------------------------------------------------- spec surgery ----
  spec = env.scene.spec
  base_body_spec = next(b for b in spec.bodies if b.name.endswith("base_link"))
  base_body = base_body_spec.name
  anchor = spec.worldbody.add_body(name="pyg_anchor", mocap=True, pos=[0.0, 0.0, 1.0])
  # site, not geom: a geom on the anchor body adds 4.2 g to body_mass.sum() and the bake
  # asserts the mass is unchanged.
  anchor.add_site(
    name="pyg_anchor_site", size=[0.012, 0, 0], rgba=[1.0, 0.35, 0.1, 0.55]
  )
  for nm, typ in (("base_weld", mujoco.mjtEq.mjEQ_WELD), ("base_pivot", mujoco.mjtEq.mjEQ_CONNECT)):
    eq = spec.add_equality()
    eq.name = nm
    eq.type = typ
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = "pyg_anchor" if nm == "base_weld" else base_body
    eq.name2 = base_body if nm == "base_weld" else "pyg_anchor"
    eq.active = False
  # "string" base mode (safety tether): a 2-site spatial tendon, anchor site on the SAME
  # mocap body the weld/pivot use, hook site on base_link at the local origin (sim_core moves
  # it at runtime via `site_pos`, exactly the way `base_pivot`'s pivot_offset is rewritten on
  # the compiled model below - MjModel per-site position is not a compile-time-only field).
  anchor.add_site(name="pyg_string_anchor", size=[0.01, 0, 0], rgba=[0.3, 0.9, 0.3, 0.0])
  base_body_spec.add_site(
    name="pyg_string_hook", pos=[0.0, 0.0, 0.0], size=[0.01, 0, 0], rgba=[0.3, 0.9, 0.3, 0.0]
  )
  string_tendon = spec.add_tendon(name="pyg_string")
  string_tendon.limited = False  # inert at bake - sim_core turns this on only in mode "string"
  string_tendon.range = [0.0, STRING_L0]
  string_tendon.solref_limit = list(STRING_SOLREF_LIMIT)
  string_tendon.width = 0.006
  string_tendon.rgba = [0.5, 0.5, 0.5, 0.0]  # sim_core drives colour/alpha at runtime
  string_tendon.wrap_site("pyg_string_anchor")
  string_tendon.wrap_site("pyg_string_hook")
  # Opt-in static terrain (tools/terrain_nav/make_sim_terrain.py): heightfield + tree cylinders, for the trail teleop.
  # Unset = no change to the bake.  Only world geoms are added - robot bodies, mass and joints are untouched.
  if os.environ.get("PYG_BAKE_TERRAIN"):
    _add_terrain(mujoco, np, spec, os.environ["PYG_BAKE_TERRAIN"])
  m = spec.compile()

  # MjSpec leaves equality.data at a type-agnostic default ([0,1,0,0,1,...] - the
  # joint/tendon polycoef default), which for a weld/connect means "anchor at (0,1,0)".
  # That is not a typo we can afford: it drops the robot on the floor 1 m sideways.
  # Write the two rows explicitly on the compiled model instead of trusting the default.
  wi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "base_weld")
  ci = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "base_pivot")
  anchor_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pyg_anchor")
  base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body)
  # weld: body1 = anchor, body2 = base; anchor point (0,0,0), relpose identity, torquescale 1
  m.eq_obj1id[wi], m.eq_obj2id[wi] = anchor_bid, base_bid
  m.eq_data[wi] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
  # connect: body1 = base, body2 = anchor; data[0:3] = pivot point in the BASE frame
  # (runtime rewrites it), data[3:6] = the point on body2 = the anchor origin.
  m.eq_obj1id[ci], m.eq_obj2id[ci] = base_bid, anchor_bid
  m.eq_data[ci] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for e in (wi, ci):
    m.eq_solref[e] = EQ_SOLREF
    m.eq_solimp[e] = EQ_SOLIMP
    m.eq_active0[e] = 0

  string_tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "pyg_string")
  string_anchor_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pyg_string_anchor")
  string_hook_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pyg_string_hook")
  # Written explicitly rather than trusted from the spec compile, same reasoning as eq_data
  # above: this is the one row sim_core reads every tick, so a silent default here is a
  # silent wrong catch height.
  m.tendon_limited[string_tid] = 0
  m.tendon_range[string_tid] = [0.0, STRING_L0]
  m.tendon_solref_lim[string_tid] = STRING_SOLREF_LIMIT
  m.tendon_width[string_tid] = 0.006
  m.tendon_rgba[string_tid] = [0.5, 0.5, 0.5, 0.0]

  # mjlab applies SimulationCfg to the COMPILED model, not to the spec: the scene spec still
  # carries the XML's own <option> (timestep 0.002, default solver settings), so a
  # spec.compile() here silently produced a 500 Hz model with different solver iterations
  # than the trainer's 200 Hz one.  Copy the whole option block across and then assert it.
  for f in (
    "timestep", "apirate", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance",
    "ccd_tolerance", "density", "viscosity", "o_margin", "integrator", "cone", "jacobian",
    "solver", "iterations", "ls_iterations", "noslip_iterations", "ccd_iterations",
    "sdf_iterations", "disableflags", "enableflags", "disableactuator",
  ):
    if hasattr(m_env.opt, f):
      setattr(m.opt, f, getattr(m_env.opt, f))
  for f in ("gravity", "wind", "magnetic", "o_solref", "o_solimp", "o_friction"):
    if hasattr(m_env.opt, f):
      getattr(m.opt, f)[:] = getattr(m_env.opt, f)

  # --------------------------------------------------------------- contract checks ---
  errs = []
  for f in ("timestep", "integrator", "solver", "iterations", "ls_iterations", "cone", "impratio"):
    if getattr(m.opt, f) != getattr(m_env.opt, f):
      errs.append(f"opt.{f} {getattr(m.opt, f)} != env {getattr(m_env.opt, f)}")
  if m.nu != m_env.nu:
    errs.append(f"nu {m.nu} != env {m_env.nu}")
  if abs(float(m.body_mass.sum()) - float(m_env.body_mass.sum())) > 1e-9:
    errs.append(f"mass {m.body_mass.sum()} != env {m_env.body_mass.sum()}")
  if m.nkey != m_env.nkey:
    errs.append(f"nkey {m.nkey} != env {m_env.nkey}")
  if m.nq != m_env.nq or m.nv != m_env.nv:
    errs.append(f"nq/nv {m.nq}/{m.nv} != env {m_env.nq}/{m_env.nv}")
  if not np.allclose(m.actuator_forcerange[: m_env.nu], m_env.actuator_forcerange):
    errs.append("actuator forcerange differs from env")
  if not np.allclose(m.actuator_gainprm[: m_env.nu], m_env.actuator_gainprm):
    errs.append("actuator gainprm differs from env")
  if m.nkey and not np.allclose(m.key_qpos[0][: m_env.nq], m_env.key_qpos[0]):
    errs.append("keyframe qpos differs from env")
  if not np.allclose(m.opt.gravity, m_env.opt.gravity):
    errs.append("gravity differs from env")
  if errs:
    raise RuntimeError(f"{variant}: baked model does not match the env: " + "; ".join(errs))

  floor_geom = next(
    (
      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
      for g in range(m.ngeom)
      if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
    ),
    None,
  )
  if floor_geom is None:
    raise RuntimeError(f"{variant}: no plane geom in the baked scene - ground toggle impossible")

  sensors = {
    (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor{s}"): dict(
      adr=int(m.sensor_adr[s]), dim=int(m.sensor_dim[s]), type=int(m.sensor_type[s])
    )
    for s in range(m.nsensor)
  }

  Path(cache_dir).mkdir(parents=True, exist_ok=True)
  mjb = Path(cache_dir) / f"{variant}.mjb"
  mujoco.mj_saveModel(m, str(mjb), None)

  # --------------------------------------------- AB: measured loop transmission ------
  transmission = None
  if variant.endswith("-AB"):
    transmission = _measure_transmission(mujoco, np, m, names, act_names, gains, family, tn_curves)

  xml = variant_xml(variant)
  contract = dict(
    contract_version=CONTRACT_VERSION,
    variant=variant,
    dof_class=variant.split("-")[0],
    ankle_mode=variant.split("-")[1],
    task=TASK,
    model_xml=str(xml),
    xml_sha256=sha256_file(xml),
    constants_path=str(CONSTANTS_PY),
    constants_sha256=sha256_file(CONSTANTS_PY),
    mjlab_git=_git_head(Path(REPO) / "mujoco-sim/mjlab"),
    repo_git=_git_head(Path(REPO)),
    bake_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    bake_seconds=round(build_s, 1),
    env_toggles={k: v for k, v in os.environ.items() if k.startswith("PYG_")},
    total_mass_kg=round(float(m.body_mass.sum()), 6),
    n_dof=len(names),
    nu=int(m.nu),
    nq=int(m.nq),
    nv=int(m.nv),
    joint_names=names,
    action_joint_names=act_names,
    obs_joint_names=obs_names,
    obs_dim=obs_dim,
    obs_layout=obs_layout,
    critic_obs_dim=int(om.group_obs_dim["critic"][0]) if "critic" in om.group_obs_dim else None,
    default_q={n: round(float(q), 8) for n, q in zip(names, default_q)},
    action_scale=action_scale,
    action_scale_cfg=action_scale_cfg,
    action_offset=action_offset,
    decimation=int(env.cfg.decimation),
    physics_dt=float(m.opt.timestep),
    step_dt=float(env.step_dt),
    control_hz=round(1.0 / float(env.step_dt), 4),
    sim_options=dict(
      timestep=float(m.opt.timestep),
      integrator=int(m.opt.integrator),
      solver=int(m.opt.solver),
      iterations=int(m.opt.iterations),
      ls_iterations=int(m.opt.ls_iterations),
      cone=int(m.opt.cone),
      impratio=float(m.opt.impratio),
      jacobian=int(m.opt.jacobian),
    ),
    gravity=[float(x) for x in m.opt.gravity],
    gains=gains,
    joint_family=family,
    tn_curves=tn_curves,
    dof_props=dof_props,
    joint_contract=joint_contract,
    safe_clip=safe_clip,
    spawn_base_z=spawn_base_z,
    keyframe_sole_penetration_m=kf_pen,
    keyframes=keyframes,
    sensors=sensors,
    floor_geom=floor_geom,
    anchor_eq_ids=dict(
      mocap_body="pyg_anchor",
      mocap_id=int(m.body_mocapid[anchor_bid]),
      weld=int(wi),
      connect=int(ci),
      base_body=base_body,
      base_body_id=int(base_bid),
      solref=list(EQ_SOLREF),
      solimp=list(EQ_SOLIMP),
    ),
    string_rig=dict(
      tendon="pyg_string",
      tendon_id=int(string_tid),
      anchor_site="pyg_string_anchor",
      anchor_site_id=int(string_anchor_sid),
      hook_site="pyg_string_hook",
      hook_site_id=int(string_hook_sid),
      L0=STRING_L0,
      solref_limit=list(STRING_SOLREF_LIMIT),
    ),
    ankle_inverse=_ankle_inverse_meta(variant, transmission),
    loop_transmission=transmission,
    mjb_sha256=sha256_file(mjb),
  )
  contract["contract_sha"] = canonical_sha(contract)
  out = Path(cache_dir) / f"{variant}.model_contract.json"
  out.write_text(json.dumps(contract, indent=1, ensure_ascii=False))

  print(
    f"BAKE_OK {variant}  nu={m.nu} nq={m.nq} mass={contract['total_mass_kg']} kg  "
    f"joints={len(names)} act={len(act_names)} obs={len(obs_names)}  "
    f"sha={contract['contract_sha'][:12]}  {build_s:.1f}s"
  )
  return contract


def _add_terrain(mujoco, np, spec, path: str) -> None:
  """Heightfield + tree cylinders as static world geoms with the floor plane's contact settings; the plane is moved
  below the lowest terrain point (kept: sim_core needs a plane geom for its ground toggle)."""
  t = np.load(path)
  h = t["hfield"].astype(np.float64)
  zmin, zmax = float(h.min()), float(h.max())
  floor = next(g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE)  # lives in the "terrain" body, not the world body
  floor.pos = [0.0, 0.0, zmin - 2.0]
  ny, nx = h.shape
  hf = spec.add_hfield(name="trail_hfield", size=[float(t["half"][0]), float(t["half"][1]), max(zmax - zmin, 1e-3), 1.0],
                       nrow=ny, ncol=nx, userdata=((h - zmin) / max(zmax - zmin, 1e-3)).ravel().tolist())
  common = dict(contype=floor.contype, conaffinity=floor.conaffinity, friction=list(floor.friction),
                solref=list(floor.solref), solimp=list(floor.solimp), condim=floor.condim)
  spec.worldbody.add_geom(name="trail_ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hf.name,
                          pos=[float(t["center"][0]), float(t["center"][1]), zmin], rgba=[0.55, 0.5, 0.4, 1.0], **common)
  for i, tr in enumerate(t["trees"]):  # x, y, bottom, radius[, top]; without a top the trunk is 3 m tall
    x, y, z, r = (float(v) for v in tr[:4])
    top = float(tr[4]) if len(tr) > 4 else z + 3.0
    spec.worldbody.add_geom(name=f"trail_tree{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, (top - z) / 2, 0],
                            pos=[x, y, (z + top) / 2], rgba=[0.3, 0.55, 0.3, 1.0], **common)


def _spawn_base_z(mujoco, np, m, names, default_q) -> float:
  """Base height that puts the lowest sole corner exactly on z=0 in the default pose."""
  d = mujoco.MjData(m)
  d.qpos[:] = m.qpos0
  for jn, q in zip(names, default_q):
    d.qpos[m.jnt_qposadr[m.joint(f"robot/{jn}").id]] = q
  fa = int(m.jnt_qposadr[0])
  d.qpos[fa : fa + 3] = [0.0, 0.0, 1.0]
  d.qpos[fa + 3 : fa + 7] = [1.0, 0.0, 0.0, 0.0]
  mujoco.mj_forward(m, d)
  zmin = 1e9
  for g in range(m.ngeom):
    gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "foot" not in gn or "collision" not in gn or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
      continue
    p, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    corners = np.array(
      [[sx * s[0], sy * s[1], sz * s[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    zmin = min(zmin, float((p + corners @ R.T)[:, 2].min()))
  return round(1.0 - zmin, 6) if zmin < 1e8 else 0.0


def _measure_transmission(mujoco, np, m, names, act_names, gains, family, tn_curves) -> dict:
  """Crank -> ankle lever ratio, measured on THIS baked model with the base welded.

  The reference numbers in ``tools/robot_model/loop_ankle_verify.json`` were produced on
  ``pygmalion_v3_printed_loop`` with a shin weld, not on the v30 build, so they are a
  cross-check, not a spec.  Measuring here means the contract carries the number for the
  model the viewer actually runs.
  """
  d = mujoco.MjData(m)
  qadr = {n: int(m.jnt_qposadr[m.joint(f"robot/{n}").id]) for n in names}
  dadr = {n: int(m.jnt_dofadr[m.joint(f"robot/{n}").id]) for n in names}
  a_q = np.array([qadr[n] for n in act_names])
  a_d = np.array([dadr[n] for n in act_names])
  kp = np.array([gains[n]["kp"] for n in act_names])
  kd = np.array([gains[n]["kd"] for n in act_names])
  eff = np.array([gains[n]["effort"] for n in act_names])
  fam = [family[n] for n in act_names]
  tnw = {f: np.array([p[0] for p in tn_curves[f]]) for f in tn_curves}
  tnt = {f: np.array([p[1] for p in tn_curves[f]]) for f in tn_curves}
  wi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "base_weld")

  def run(c_deg: dict[str, float], steps: int = 1200) -> dict[str, float]:
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0
    fa = int(m.jnt_qposadr[0])
    d.qpos[fa : fa + 3] = [0, 0, 1.0]
    d.qpos[fa + 3 : fa + 7] = [1, 0, 0, 0]
    d.mocap_pos[0] = [0, 0, 1.0]
    d.mocap_quat[0] = [1, 0, 0, 0]
    d.eq_active[wi] = 1
    mujoco.mj_forward(m, d)
    q0 = d.qpos[a_q].copy()
    tgt = q0.copy()
    for i, n in enumerate(act_names):
      if n in c_deg:
        tgt[i] = np.radians(c_deg[n])
    for k in range(steps):
      f = min(1.0, 2.0 * k / steps)
      qt = q0 + f * (tgt - q0)
      qv = d.qvel[a_d]
      raw = np.clip(kp * (qt - d.qpos[a_q]) - kd * qv, -eff, eff)
      for i, fm in enumerate(fam):
        hi = np.interp(qv[i], tnw[fm], tnt[fm]) if qv[i] >= 0 else tnt[fm][0]
        lo = -(np.interp(-qv[i], tnw[fm], tnt[fm]) if qv[i] < 0 else tnt[fm][0])
        raw[i] = min(max(raw[i], lo), hi)
      d.qfrc_applied[:] = 0.0
      d.qfrc_applied[a_d] = raw
      mujoco.mj_step(m, d)
    return {n: float(np.degrees(d.qpos[qadr[n]])) for n in names if "ankle" in n or "crank" in n}

  def cr(**kw):
    return {f"{s}_crank_{t}_joint": v for t, v in kw.items() for s in ("L", "R")}

  # A trap this measurement exists to expose: in the v30 build crank_A and crank_B of the
  # SAME leg have OPPOSITE joint axes (L: A = -Y, B = +Y).  So "both cranks +10 deg" in
  # joint-q space is the differential mode, not the common one, and the v3-era
  # "pitch per common crank" number does not transfer.  Measure the full 2x2 Jacobian in
  # q-space instead and let the numbers say which combination is pitch.
  base = run({})
  ap = run(cr(A=10))
  am = run(cr(A=-10))
  bp = run(cr(B=10))
  bm = run(cr(B=-10))
  closure = _closure_mm(mujoco, np, m, d)
  out: dict = {
    "note": (
      "d(ankle)/d(crank) in JOINT-q degrees, measured on this baked model with the base "
      "welded and the legs hanging. J[i][j] = d(pitch,roll)[i] / d(crank A,B)[j]."
    ),
    "closure_worst_mm": closure,
  }
  for s in ("L", "R"):
    pj, rj = f"{s}_ankle_pitch_joint", f"{s}_ankle_roll_joint"
    J = [
      [(ap[pj] - am[pj]) / 20.0, (bp[pj] - bm[pj]) / 20.0],
      [(ap[rj] - am[rj]) / 20.0, (bp[rj] - bm[rj]) / 20.0],
    ]
    Jn = np.asarray(J)
    out[s] = dict(
      neutral_deg=dict(
        pitch=round(base[pj], 5),
        roll=round(base[rj], 5),
        crank_A=round(base[f"{s}_crank_A_joint"], 5),
        crank_B=round(base[f"{s}_crank_B_joint"], 5),
      ),
      J=[[round(float(x), 5) for x in row] for row in J],
      J_inv=(
        [[round(float(x), 5) for x in row] for row in np.linalg.inv(Jn)]
        if abs(np.linalg.det(Jn)) > 1e-6
        else None
      ),
      det=round(float(np.linalg.det(Jn)), 6),
      # v3-era summary numbers, recomputed in this model's q convention
      pitch_per_crank_common_deg=round(float(Jn[0, 0] + Jn[0, 1]) / 2.0, 5),
      pitch_per_crank_opposed_deg=round(float(Jn[0, 0] - Jn[0, 1]) / 2.0, 5),
      roll_per_crank_common_deg=round(float(Jn[1, 0] + Jn[1, 1]) / 2.0, 5),
      roll_per_crank_opposed_deg=round(float(Jn[1, 0] - Jn[1, 1]) / 2.0, 5),
      crank_tracking_deg=round(
        max(abs(ap[f"{s}_crank_A_joint"] - 10.0), abs(bp[f"{s}_crank_B_joint"] - 10.0)), 4
      ),
    )
  ref = json.loads(LOOP_VERIFY.read_text()) if LOOP_VERIFY.exists() else {}
  out["reference_v3"] = dict(
    source=str(LOOP_VERIFY),
    pitch_per_crank_deg=ref.get("pitch_per_crank_deg"),
    roll_per_crank_diff_deg=ref.get("roll_per_crank_diff_deg"),
    closure_worst_mm=ref.get("closure_worst_mm"),
    note=(
      "measured on pygmalion_v3_printed_loop with only the shin welded - a cross-check on "
      "MAGNITUDE, not a spec: the v30 crank q-axes differ, so signs need not match."
    ),
  )
  # ---------- does the saved v3 inverse grid work on THIS model, and under what sign? --
  # The v3 grid's MAGNITUDES survive the rebuild (its 1.21 deg pitch per common crank is
  # 2 x this model's 0.586 opposed number, 3 %), but the v30 generator flipped crank_B's
  # joint axis, so the grid's crank pair has to be re-signed before it means anything here.
  # Fit that re-signing instead of asserting it: try the 8 (swap, signA, signB) maps and
  # keep the one the MODEL says is right.
  if ENVELOPE.exists():
    envj = json.loads(ENVELOPE.read_text())
    probes = [(0.0, 0.0), (-0.35, 0.0), (0.17, 0.0), (0.0, 0.17), (0.0, -0.17), (-0.2, 0.1)]
    combos = [(sw, sa, sb) for sw in (0, 1) for sa in (1, -1) for sb in (1, -1)]
    worst_of: dict[tuple, dict[str, float]] = {}
    for combo in combos:
      w = {"L": 0.0, "R": 0.0}
      for tp, tr in probes:
        cmd = {}
        for s in ("L", "R"):
          a, b = _apply_combo(_grid_lookup(np, envj, s, tp, tr), combo)
          cmd[f"{s}_crank_A_joint"] = float(np.degrees(a))
          cmd[f"{s}_crank_B_joint"] = float(np.degrees(b))
        got = run(cmd)
        for s in ("L", "R"):
          w[s] = max(
            w[s],
            abs(float(np.radians(got[f"{s}_ankle_pitch_joint"]) - tp)),
            abs(float(np.radians(got[f"{s}_ankle_roll_joint"]) - tr)),
          )
      worst_of[combo] = w
    best = {s: min(combos, key=lambda c: worst_of[c][s]) for s in ("L", "R")}
    out["envelope_fit"] = dict(
      source=str(ENVELOPE),
      envelope_tag=envj.get("tag"),
      probes_rad=probes,
      per_side={
        s: dict(
          swap_AB=bool(best[s][0]),
          sign_A=best[s][1],
          sign_B=best[s][2],
          worst_rad=round(worst_of[best[s]][s], 5),
          identity_worst_rad=round(worst_of[(0, 1, 1)][s], 5),
        )
        for s in ("L", "R")
      },
      usable=bool(max(worst_of[best[s]][s] for s in ("L", "R")) < 0.05),
      note=(
        "crank = combo(grid_lookup(pitch,roll)). combo is fitted here because the grid was "
        "solved on a model whose crank q-axes differ; 'usable' false means the UI must fall "
        "back to the measured linear inverse loop_transmission[side].J_inv."
      ),
    )
  return out


def _apply_combo(ab, combo):
  swap, sa, sb = combo
  a, b = (ab[1], ab[0]) if swap else ab
  return sa * a, sb * b


def _grid_lookup(np, envj, side: str, pitch_rad: float, roll_rad: float):
  """Bilinear lookup of ``crank_rad`` in the saved envelope (no scipy needed here)."""
  P = np.radians(np.asarray(envj["grid"]["pitch_deg"], dtype=float))
  R = np.radians(np.asarray(envj["grid"]["roll_deg"], dtype=float))
  G = np.asarray(envj["legs"][side]["crank_rad"], dtype=float)
  p = float(np.clip(pitch_rad, P[0], P[-1]))
  r = float(np.clip(roll_rad, R[0], R[-1]))
  i = int(np.clip(np.searchsorted(P, p) - 1, 0, len(P) - 2))
  j = int(np.clip(np.searchsorted(R, r) - 1, 0, len(R) - 2))
  u = (p - P[i]) / (P[i + 1] - P[i])
  v = (r - R[j]) / (R[j + 1] - R[j])
  g = (
    (1 - u) * (1 - v) * G[i, j]
    + u * (1 - v) * G[i + 1, j]
    + (1 - u) * v * G[i, j + 1]
    + u * v * G[i + 1, j + 1]
  )
  return float(g[0]), float(g[1])


def _closure_mm(mujoco, np, m, d) -> float:
  worst = 0.0
  for s in "LR":
    for t in "AB":
      i1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"robot/{s}_rod_{t}_end")
      i2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"robot/{s}_ball_{t}")
      if i1 < 0 or i2 < 0:
        continue
      worst = max(worst, float(np.linalg.norm(d.site_xpos[i1] - d.site_xpos[i2])) * 1e3)
  return round(worst, 6)


def _ankle_inverse_meta(variant: str, transmission: dict | None) -> dict | None:
  """How the viewer turns an (ankle pitch, roll) command into a crank pair.

  Two methods, and the bake decides between them from measurement, not from belief:
    ``envelope`` the saved ``ankle_rp_envelope.json`` grid with the per-leg sign map that
                 ``loop_transmission.envelope_fit`` measured on this model.  Full ROM.
    ``linear``   crank = neutral + J^-1 . (target - neutral), from the measured 2x2
                 Jacobian.  Used only when the grid does not fit; valid near neutral.
  """
  if not variant.endswith("-AB") or not ENVELOPE.exists():
    return None
  env = json.loads(ENVELOPE.read_text())
  tag = env.get("tag")
  fit = (transmission or {}).get("envelope_fit") or {}
  usable = bool(fit.get("usable"))
  return dict(
    method="envelope" if usable else "linear",
    source=str(ENVELOPE),
    envelope_tag=tag,
    pitch_deg=env["grid"]["pitch_deg"],
    roll_deg=env["grid"]["roll_deg"],
    tag_matches_variant=False,
    sign_map={s: fit.get("per_side", {}).get(s) for s in ("L", "R")} if usable else None,
    worst_residual_rad=(
      max(fit["per_side"][s]["worst_rad"] for s in ("L", "R")) if fit.get("per_side") else None
    ),
    caveat=(
      f"the crank_rad grid was solved on {tag!r}, not on this v30 build, whose crank joint "
      "axes are re-signed. The bake FITS the per-leg sign map by commanding the grid and "
      "reading the ankle back, and records the residual. The ankle READOUT is always the "
      "model's own state, so any remaining grid error shows as target-vs-actual, never as a "
      "silent lie."
    ),
  )


def _git_head(repo: Path) -> str:
  try:
    return subprocess.run(
      ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
      capture_output=True,
      text=True,
      timeout=10,
    ).stdout.strip()
  except Exception:
    return "?"


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog="pygviewer bake", description=__doc__)
  sub = ap.add_subparsers(dest="what", required=True)
  mp = sub.add_parser("model", help="bake a model .mjb + contract")
  mp.add_argument("--variant", choices=list(VARIANTS))
  mp.add_argument("--all", action="store_true")
  mp.add_argument("--cache", default=CACHE_DIR)
  mp.add_argument("--no-init-bent", action="store_true", help="bake with the HOME keyframe")
  mp.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
  pp = sub.add_parser("policy", help="export a trained .pt to ONNX + policy contract")
  pp.add_argument("--pt", required=True, help="path to <run_dir>/model_N.pt")
  pp.add_argument("--variant", choices=list(VARIANTS), required=True)
  pp.add_argument("--cache", default=CACHE_DIR)
  pp.add_argument("--name", default=None)
  pp.add_argument("--no-init-bent", action="store_true")
  pp.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
  args = ap.parse_args(argv)

  if args.what == "policy":
    if args.in_process:
      from ._bake_policy import bake_policy

      bake_policy(
        args.pt, args.variant, args.cache, args.name, init_bent=not args.no_init_bent
      )
      return 0
    cmd = [
      sys.executable, str(Path(__file__).resolve().parents[1] / "run.py"), "bake", "policy",
      "--pt", args.pt, "--variant", args.variant, "--cache", args.cache, "--in-process",
    ]
    if args.name:
      cmd += ["--name", args.name]
    if args.no_init_bent:
      cmd.append("--no-init-bent")
    envv = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for k in list(envv):
      if k.startswith("PYG_"):
        envv.pop(k)
    return subprocess.run(cmd, cwd=REPO, env=envv).returncode

  if args.all:
    todo = list(VARIANTS)
  elif args.variant:
    todo = [args.variant]
  else:
    ap.error("--variant or --all")

  if args.in_process:
    if len(todo) != 1:
      ap.error("--in-process bakes exactly one variant")
    bake_one(todo[0], args.cache, init_bent=not args.no_init_bent)
    return 0

  # One env per subprocess: a ManagerBasedRlEnv holds ~1.3 GB and mjlab modules read PYG_*
  # at IMPORT time, so a second variant in the same interpreter would silently reuse the
  # first one's XML.
  rc = 0
  for v in todo:
    cmd = [
      sys.executable,
      "-m" if __package__ else str(Path(__file__).parent.parent / "run.py"),
    ]
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "run.py"), "bake", "model",
           "--variant", v, "--cache", args.cache, "--in-process"]
    if args.no_init_bent:
      cmd.append("--no-init-bent")
    envv = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for k in list(envv):
      if k.startswith("PYG_") and k != "PYG_BAKE_TERRAIN":  # the opt-in terrain is not a training toggle
        envv.pop(k)
    print(f"--- bake {v} ---", flush=True)
    p = subprocess.run(cmd, cwd=REPO, env=envv)
    rc = rc or p.returncode
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
