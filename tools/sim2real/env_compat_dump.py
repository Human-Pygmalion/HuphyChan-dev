#!/usr/bin/env python3
"""Dump what a policy sees and does in a built env, so two builds of the "same" env can be diffed.

Used to check that the audit package's task (stock mjlab, no PYG_* flags) reproduces the env a
fork-trained policy was trained in: actor observation terms and widths, action joints / scale /
offset, default pose, joint limits, per-actuator gains, and MuJoCo model fingerprints.

  # package side
  <python> env_compat_dump.py --task Pygmalion-Velocity-Flat-P2 --import pygmalion_mjlab --pkg <pkg> --out a.json
  # fork side (PYG_* flags from the training run's manifest)
  <python> env_compat_dump.py --task Mjlab-Velocity-Flat-Pygmalion --manifest <run>/repro/launch_manifest.json --out b.json
  python env_compat_dump.py --compare a.json b.json
"""
import argparse
import hashlib
import importlib
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--task")
ap.add_argument("--import", dest="imp", default="")
ap.add_argument("--pkg", default="")
ap.add_argument("--manifest", default="")
ap.add_argument("--drop", default="", help="comma list of PYG_* names to unset after reading the manifest")
ap.add_argument("--out")
ap.add_argument("--compare", nargs=2)
a = ap.parse_args()

if a.compare:
  x, y = (json.load(open(p)) for p in a.compare)
  bad = 0
  for k in sorted(set(x) | set(y)):
    if x.get(k) != y.get(k):
      bad += 1
      print(f"DIFF {k}:\n   A={json.dumps(x.get(k))[:300]}\n   B={json.dumps(y.get(k))[:300]}")
  print(f"{len(set(x) | set(y))} fields compared, {bad} differ")
  raise SystemExit(0)

if a.pkg:
  sys.path.insert(0, a.pkg)
if a.manifest:
  for k, v in json.load(open(a.manifest))["environment"].items():
    if k.startswith("PYG_"):
      os.environ[k] = str(v)
for k in filter(None, a.drop.split(",")):
  os.environ.pop(k, None)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

if a.imp:
  importlib.import_module(a.imp)
cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = 2
cfg.seed = 11
env = ManagerBasedRlEnv(cfg, device="cpu")
env.reset()
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
jid = term.target_ids
r = lambda t: [round(float(v), 7) for v in t.flatten().tolist()]  # noqa: E731
d = {}
om = env.observation_manager
d["actor_terms"] = {n: int(np.prod(s)) for n, s in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"])}
d["actor_dim"] = int(sum(d["actor_terms"].values()))
d["actor_term_order"] = list(om.active_terms["actor"])
d["action_dim"] = int(term.action_dim)
d["action_joints"] = list(term.target_names)
sc, of = term.scale, term.offset
d["action_scale"] = r(sc[0] if torch.is_tensor(sc) else torch.tensor([sc]))
d["action_offset"] = r(of[0] if torch.is_tensor(of) else torch.tensor([of]))
d["default_pose"] = r(robot.data.default_joint_pos[0, jid])
d["joint_limits_hard"] = r(robot.data.joint_pos_limits[0, jid])
d["decimation"] = int(cfg.decimation)
d["step_dt"] = round(float(env.step_dt), 6)
d["physics_dt"] = round(float(cfg.sim.mujoco.timestep), 6)
for i, act in enumerate(robot.actuators):
  k = f"actuator{i}"
  d[k + "_class"] = type(act).__name__
  d[k + "_targets"] = list(act.target_names)
  for attr in ("default_stiffness", "default_damping", "stiffness", "damping", "effort_limit",
               "armature", "frictionloss", "velocity_limit", "saturation_effort", "velocity_limit_motor"):
    v = getattr(act, attr, None)
    if torch.is_tensor(v):
      d[f"{k}_{attr}"] = r(v[0])
    elif isinstance(v, (int, float)):
      d[f"{k}_{attr}"] = float(v)
m = env.sim.mj_model
d["mj_counts"] = dict(nbody=int(m.nbody), njnt=int(m.njnt), ngeom=int(m.ngeom), neq=int(m.neq), nu=int(m.nu), nv=int(m.nv))
d["mj_total_mass"] = round(float(m.body_mass.sum()), 6)
for nm in ("dof_armature", "dof_damping", "dof_frictionloss", "body_mass", "body_ipos", "geom_size", "geom_friction", "eq_data", "jnt_range"):
  d["mj_hash_" + nm] = hashlib.sha1(np.round(getattr(m, nm), 6).tobytes()).hexdigest()[:12]
d["crank_joints"] = [n for n in term.target_names if "crank" in n]
d["env_reward_terms"] = sorted(env.reward_manager.active_terms)
json.dump(d, open(a.out, "w"), indent=1)
print("wrote", a.out, "actor_dim", d["actor_dim"], "joints", len(d["action_joints"]))
env.close()
