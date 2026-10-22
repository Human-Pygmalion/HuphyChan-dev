#!/usr/bin/env python3
"""Stage 6 lab: which simulator change reproduces the real robot's episodic 5 Hz oscillation? (docs/sim2real/2026-10-22_stage6_5hz_reproduction.md)

One process, many conditions: every (condition, repetition) is one environment of ONE batched stock-mjlab env, so all
conditions share the same physics step (5 ms), the same policy (terrain10_2048_model_37400.pt, mean action, command 0/0), the
same flat floor and the same timing model as tools/sim2real/walk/sim_walk_record.py (policy called on a schedule, action HELD
between calls, command reaches the motor 1 step late, Kp/Kd through the 12-bit MIT quantiser, joint angles 10 ms old,
q_prev = angle seen at the previous call). Nothing is edited in the task; the config is rewritten like sim_walk_record does.

Per-environment condition parameters (all default OFF; names are the keys of CONDITIONS below):
  period     'real' (call intervals = diff of column t of --real-timing-csv, circular, random start per env) | 'fixed' with period_ms
  kb, ka, kd Kp,Kd scale of the body joints (hip/knee), of the ankle cranks, extra Kd multiplier on everything
             (a "gain" of 0.6 in the log names scales Kp and Kd together; Kd x1.5 scales Kd only)
  imu_age_ms IMU (gyro + gravity) age at the policy call;  still_thr / still_s: gyro and gravity-free report (0,0,0) when |w| < still_thr
             rad/s for longer than still_s seconds
  tlag_ms    first-order lag of the applied motor torque (current loop);  plag_ms: first-order lag of the joint angle the motor PD reads
  imp_N / imp_ms / imp_h   torso impulse: horizontal force +x [N] for imp_ms at t = t_imp, acting at imp_h metres above the base CoM
  hold_k / hold_c / hold_h horizontal spring-damper (x and y) from a fixed world point to a point hold_h above the base CoM [N/m, N s/m, m]
  hold_kr / hold_cr  torsional spring-damper that keeps the base z axis upright (hand holding the torso level) [N m/rad, N m s/rad]
  gyro_scale scale of the gyro the policy sees (0 = gyro input forced to zero);  qage_ms: age of the joint angles in the observation (default --obs-age-ms)
  exc_Nm / exc_hz / exc_axis  sinusoidal torque on the base [N m] at exc_hz, about world y (axis 1, pitch) or x (axis 0, roll), from t = 3 s to 15 s
             (closed-loop frequency-response probe: how strongly does policy + robot answer a given frequency)
  push_N     constant horizontal +x force on the base [N] for the whole run, acting at push_h [m] above the CoM (a hand that leans on the robot)
  support    constant upward force on the base as a fraction of the robot weight
  loop_t     time constant [s] of the 4 ankle closed-loop (rod) connect constraints (structure compliance; sim default 0.002 -> clamped to 2 dt)
  fric       joint friction-loss scale of the 12 motors

Output: <out>.npz with per-call arrays (NaN padded, rows = policy calls) and per-step arrays, <out>.json with the condition table.
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl nice -n 10 mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/s6_oscillation_lab.py \
      --branch-root <gw_branch>/vibecoded-audit-20261012 --mjlab-src <stock_mjlab>/src --real-timing-csv "Expr_Data/vx0gain1walkob(진동확인용)1021.csv" \
      --set T1 --reps 8 --seconds 30 --out <dir>/T1
Sets are defined in SETS; --list prints them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--branch-root", default="")
ap.add_argument("--mjlab-src", default="")
ap.add_argument("--checkpoint", default="")
ap.add_argument("--real-timing-csv", default="")
ap.add_argument("--set", default="", help="name of a condition set in SETS, or comma list of condition names")
ap.add_argument("--conditions-json", default="", help="json file {name: {param: value}} instead of --set")
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--obs-age-ms", type=float, default=10.0, help="age of the joint angles in the observation (as sim_walk_record)")
ap.add_argument("--t-imp", type=float, default=5.0)
ap.add_argument("--out", default="")
ap.add_argument("--device", default="cpu")
ap.add_argument("--list", action="store_true")
ap.add_argument("--task", default="Pygmalion-Velocity-Generalized-Terrain10-2kResume19700")
args = ap.parse_args()


def C(**kw):
  return kw


REAL = dict(period="real")
# impulse used for T1.. (force [N], duration [ms], lever above the base CoM [m]); calibrated by the CAL set (see doc)
IMP = dict(imp_N=80.0, imp_ms=80.0, imp_h=0.25)
SETS = {}
# ---- calibration of the impulse: sweep of the force at the baseline
SETS["CAL"] = {f"CAL_F{int(f)}": C(**REAL, imp_N=float(f), imp_ms=40.0, imp_h=0.25) for f in (50, 100, 150, 200, 250, 300)}


SETS["CAL2"] = {f"CAL2_F{int(f)}_{int(ms)}ms": C(**REAL, imp_N=float(f), imp_ms=float(ms), imp_h=0.25)
                for f, ms in ((70, 90), (80, 80), (90, 80), (100, 60), (85, 70), (95, 70))}


def _imp(**kw):
  d = dict(REAL)
  d.update(IMP)
  d.update(kw)
  return d


def build_sets():
  S = SETS
  S["T0"] = {"T0 baseline, no disturbance": dict(REAL)}
  S["T1"] = {"T1 impulse": _imp()}
  hold = {}
  for k in (300, 1000, 3000):
    for c in (20, 60):
      hold[f"T2 hold k{k} c{c}"] = _imp(hold_k=float(k), hold_c=float(c), hold_h=0.1)
  hold["T2 support 20%"] = _imp(support=0.2)
  hold["T2 hold k1000 c20 + support 20%"] = _imp(hold_k=1000.0, hold_c=20.0, hold_h=0.1, support=0.2)
  S["T2"] = hold
  S["T3"] = {
    "T3a IMU age 10 ms": _imp(imu_age_ms=10.0),
    "T3b gyro still-zero (0.02 rad/s, 0.2 s)": _imp(still_thr=0.02, still_s=0.2),
    "T3c IMU age 10 ms + still-zero": _imp(imu_age_ms=10.0, still_thr=0.02, still_s=0.2),
    # the sim body never rests below 0.02 rad/s (its quiet gyro is ~0.14 rad/s rms), so 3b never triggers; 3d scales the threshold to the sim noise
    "T3d gyro still-zero (0.3 rad/s, 0.2 s)": _imp(still_thr=0.3, still_s=0.2),
  }
  S["T4"] = {
    "T4a torque lag 2 ms": _imp(tlag_ms=2.0),
    "T4b torque lag 5 ms": _imp(tlag_ms=5.0),
    "T4c torque lag 10 ms": _imp(tlag_ms=10.0),
    "T4d position-measurement lag 5 ms": _imp(plag_ms=5.0),
    "T4e torque lag 5 ms + position lag 5 ms": _imp(tlag_ms=5.0, plag_ms=5.0),
  }
  S["T5"] = {
    "T5a ankle loop time const 0.02 s": _imp(loop_t=0.02),
    "T5b ankle loop time const 0.05 s": _imp(loop_t=0.05),
  }
  S["T6"] = {"T6a joint friction x2": _imp(fric=2.0), "T6b joint friction x4": _imp(fric=4.0)}
  # ---- batch 2: what sets the baseline chatter, and the hold family in more detail
  S["DIAG"] = {
    "D0 gyro input forced to 0": dict(REAL, gyro_scale=0.0),
    "D1 interval fixed 20 ms": dict(period="fixed", period_ms=20.0),
    "D2 interval fixed 80 ms": dict(period="fixed", period_ms=80.0),
    "D3 gains x0.6 (ankle x1)": dict(REAL, kb=0.6),
    "D4 Kd x1.5": dict(REAL, kd=1.5),
    "D5 joint-angle age 0 ms": dict(REAL, qage_ms=0.0),
  }
  S["H2"] = {
    "T1 gains x0.6 (ankle x1)": _imp(kb=0.6),
    "T1 Kd x1.5": _imp(kd=1.5),
    "T1 interval fixed 20 ms": _imp(period="fixed", period_ms=20.0),
  }
  for kk in (1000, 3000):
    base = dict(hold_k=float(kk), hold_c=20.0, hold_h=0.1)
    S["H2"][f"H k{kk} c20 gains x0.6"] = _imp(**base, kb=0.6)
    S["H2"][f"H k{kk} c20 Kd x1.5"] = _imp(**base, kd=1.5)
    S["H2"][f"H k{kk} c20 interval 20 ms"] = _imp(**base, period="fixed", period_ms=20.0)
  S["H2"]["H k10000 c60"] = _imp(hold_k=10000.0, hold_c=60.0, hold_h=0.1)
  S["H2"]["H k30000 c200"] = _imp(hold_k=30000.0, hold_c=200.0, hold_h=0.1)
  S["H2"]["H k3000 c20 h0.0"] = _imp(hold_k=3000.0, hold_c=20.0, hold_h=0.0)
  S["H2"]["H k3000 c20 h0.3"] = _imp(hold_k=3000.0, hold_c=20.0, hold_h=0.3)
  for kr, cr in ((50, 5), (200, 20), (800, 80)):
    S["H2"][f"R torsion kr{kr} cr{cr}"] = _imp(hold_kr=float(kr), hold_cr=float(cr))
  S["EXC"] = {}
  for f in (2, 3, 4, 5, 6, 7, 8):
    S["EXC"][f"E pitch torque 3 Nm {f} Hz"] = dict(REAL, exc_Nm=3.0, exc_hz=float(f), exc_axis=1.0)
  for f in (3, 5, 8):
    S["EXC"][f"E roll torque 3 Nm {f} Hz"] = dict(REAL, exc_Nm=3.0, exc_hz=float(f), exc_axis=0.0)
  S["SEED1"] = {"T0 baseline, no disturbance": dict(REAL), "T1 impulse": _imp()}
  for kk in (1000, 3000):
    base = dict(hold_k=float(kk), hold_c=20.0, hold_h=0.1)
    S["SEED1"][f"T2 hold k{kk} c20"] = _imp(**base)
    S["SEED1"][f"H k{kk} c20 gains x0.6"] = _imp(**base, kb=0.6)
    S["SEED1"][f"H k{kk} c20 Kd x1.5"] = _imp(**base, kd=1.5)
    S["SEED1"][f"H k{kk} c20 interval 20 ms"] = _imp(**base, period="fixed", period_ms=20.0)
  S["BATCH3"] = dict(S["SEED1"])
  S["BATCH3"].update(S["EXC"])
  S["BATCH4"] = {}
  h3 = dict(hold_k=3000.0, hold_c=20.0, hold_h=0.1)
  for ms in (20, 31, 40, 52, 70):
    S["BATCH4"][f"I hold k3000 c20 interval {ms} ms"] = _imp(**h3, period="fixed", period_ms=float(ms))
  for k_, c_ in ((10000, 60), (1000, 60), (3000, 60)):
    S["BATCH4"][f"K hold k{k_} c{c_} interval 20 ms"] = _imp(hold_k=float(k_), hold_c=float(c_), hold_h=0.1, period="fixed", period_ms=20.0)
  for fN in (40, 80):
    S["BATCH4"][f"P push {fN} N, no spring, 20 ms"] = _imp(push_N=float(fN), push_h=0.1, period="fixed", period_ms=20.0)
    S["BATCH4"][f"P push {fN} N, no spring, real intervals"] = _imp(push_N=float(fN), push_h=0.1)
  S["BATCH2"] = dict(S["DIAG"])
  S["BATCH2"].update(S["H2"])
  S["MAIN"] = {}
  for k in ("T0", "T1", "T2", "T3", "T4", "T5", "T6"):
    S["MAIN"].update(S[k])


build_sets()

if args.list:
  for k, v in SETS.items():
    print(k, list(v))
  sys.exit(0)

if args.conditions_json:
  CONDITIONS = json.load(open(args.conditions_json))
else:
  CONDITIONS = {}
  for nm in args.set.split(","):
    nm = nm.strip()
    if nm in SETS:
      CONDITIONS.update(SETS[nm])
    else:  # a single condition name from any set
      hit = [v[nm] for v in SETS.values() if nm in v]
      assert hit, f"unknown set / condition {nm!r}"
      CONDITIONS[nm] = hit[0]
names = list(CONDITIONS)
grid = [(c, r) for c in names for r in range(args.reps)]
n = len(grid)

sys.path.insert(0, args.mjlab_src)  # stock mjlab must win over the fork's editable install
sys.path.insert(0, args.branch_root)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab  # noqa: E402
import pygmalion_mjlab  # noqa: E402,F401
from mjlab.actuator.actuator import ActuatorCmd  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg  # noqa: E402
from pygmalion_mjlab.terrain10 import StripTerrainCfg  # noqa: E402

DT_MS = 5.0
MOTORS = [f"{s}_{j}_joint" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]


@dataclass(kw_only=True)
class ZeroCommandCfg(UniformVelocityCommandCfg):
  def build(self, env):
    return ZeroCommand(self, env)


class ZeroCommand(UniformVelocityCommand):
  def _resample_command(self, env_ids):
    self.vel_command_b[env_ids] = 0.0

  def _update_command(self):
    self.vel_command_b[:] = 0.0


def build_cfg(num_envs: int, seed: int):
  cfg = load_env_cfg(args.task, play=False)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  side = 60.0
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=89500, curriculum=False, size=(side, side), num_rows=1, num_cols=1, border_width=0.0, color_scheme="none",
      sub_terrains={"level1_flat": StripTerrainCfg(size=(side, side), slope_deg=0.0, ridge_height_range=(0.0, 0.0))}),
    max_init_terrain_level=0)
  cfg.curriculum = {}
  cfg.episode_length_s = int(1e9)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  tw = cfg.commands["twist"]
  cfg.commands["twist"] = ZeroCommandCfg(**{f.name: getattr(tw, f.name) for f in fields(tw)})
  cfg.commands["twist"].rel_standing_envs = 0.0
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.05), "yaw": (0.0, 0.0)}
  cfg.events.pop("torso_push", None)
  cfg.events = {k: cfg.events[k] for k in ("reset_base", "reset_robot_joints")}
  cfg.observations["actor"].enable_corruption = False
  th = cfg.observations["actor"].terms["motor_pos_history"]
  th.delay_min_lag = th.delay_max_lag = 0
  ent = cfg.scene.entities["robot"]
  ent.articulation.actuators = tuple(replace(a, delay_min_lag=1, delay_max_lag=1) for a in ent.articulation.actuators)
  cfg.decimation = 1
  return cfg


def quant(x, xmax):
  return torch.floor(torch.clamp(x, 0, xmax) / xmax * 4095) * xmax / 4095


def load_real_intervals_ms():
  t = []
  for ln in open(args.real_timing_csv, encoding="utf-8"):
    if ln.startswith("#") or ln.startswith("t,"):
      continue
    t.append(float(ln.split(",", 1)[0]))
  return np.diff(np.array(t)) * 1000.0


def make_schedule(dur_ms, seed):
  """(n, K) scheduled call times [ms] per env, first call at 0."""
  dts = load_real_intervals_ms() if args.real_timing_csv else None
  K = int(dur_ms / 20.0) + 6
  rng = np.random.default_rng(seed + 11)
  iv = np.zeros((n, K))
  for i, (c, r) in enumerate(grid):
    cd = CONDITIONS[c]
    if cd.get("period", "real") == "fixed":
      iv[i] = float(cd.get("period_ms", 20.0))
    else:
      assert dts is not None, "--real-timing-csv needed for period=real"
      iv[i] = np.take(dts, rng.integers(0, len(dts)) + np.arange(K), mode="wrap")
  return np.concatenate([np.zeros((n, 1)), np.cumsum(iv, 1)[:, :-1]], 1)


def quat_rotate(q, v):  # q (n,4) w,x,y,z ; v (n,3)
  w, xyz = q[:, :1], q[:, 1:]
  t = 2.0 * torch.cross(xyz, v, dim=1)
  return v + w * t + torch.cross(xyz, t, dim=1)


def main():
  dur = args.seconds
  nsteps = int(round(dur * 1000.0 / DT_MS))
  ckpt = Path(args.checkpoint or Path(args.branch_root) / "deploy/generalized_walk/checkpoints/terrain10_2048_model_37400.pt")
  h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
  assert h == "7bbd5cbc1431fb8085b6533f457b4cbb7eb20b960f4c033a6907c3e7fe0fbe00", "unexpected checkpoint"
  cfg = build_cfg(n, args.seed)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  dev = env.device
  agent_cfg = load_rl_cfg(args.task)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(wenv, asdict(agent_cfg), device=args.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  torch.manual_seed(args.seed)
  env.seed(args.seed)
  robot = env.scene["robot"]
  at = env.action_manager.get_term("joint_pos")
  assert list(at.target_names) == MOTORS
  names16 = list(robot.joint_names)
  idx12 = torch.tensor([names16.index(m) for m in MOTORS], device=dev)
  P = lambda key, default: torch.tensor([float(CONDITIONS[c].get(key, default)) for c, _ in grid], device=dev)  # noqa: E731

  # ---- gains: body = hip/knee, ankle = crank; Kp and Kd scale together with kb/ka, Kd also with kd
  kb, ka, kdm = P("kb", 1.0), P("ka", 1.0), P("kd", 1.0)
  for a_ in robot.actuators:
    is_ank = torch.tensor(["crank" in t for t in a_.target_names], device=dev)
    ks = torch.where(is_ank[None, :], ka[:, None], kb[:, None])
    a_.stiffness[:] = quant(a_.default_stiffness * ks, 5000.0)
    a_.damping[:] = quant(a_.default_damping * ks * kdm[:, None], 100.0)

  # ---- motor-side models: torque lag, position-measurement lag
  tlag, plag = P("tlag_ms", 0.0), P("plag_ms", 0.0)
  al_t = (1.0 - torch.exp(-DT_MS / torch.clamp(tlag, min=1e-6)))[:, None]
  al_t = torch.where((tlag > 0)[:, None], al_t, torch.ones_like(al_t))
  al_p = (1.0 - torch.exp(-DT_MS / torch.clamp(plag, min=1e-6)))[:, None]
  al_p = torch.where((plag > 0)[:, None], al_p, torch.ones_like(al_p))
  lag_on = bool(((tlag > 0) | (plag > 0)).any())

  def wrap(act):
    orig = act.compute
    st = {"tau": None, "p": None}

    def compute(cmd):
      if st["p"] is None:
        st["p"] = cmd.pos.clone()
      st["p"] = st["p"] + al_p * (cmd.pos - st["p"])
      out = orig(ActuatorCmd(position_target=cmd.position_target, velocity_target=cmd.velocity_target,
                             effort_target=cmd.effort_target, pos=st["p"], vel=cmd.vel))
      if st["tau"] is None:
        st["tau"] = out.clone()
      st["tau"] = st["tau"] + al_t * (out - st["tau"])
      return st["tau"]

    act.compute = compute

  if lag_on:
    for a_ in robot.actuators:
      wrap(a_)

  # ---- per-env model fields: friction, loop constraint softness
  fr, loop_t = P("fric", 1.0), P("loop_t", 0.0)
  fields_needed = []
  if (fr != 1).any():
    fields_needed.append("dof_frictionloss")
  if (loop_t > 0).any():
    fields_needed.append("eq_solref")
  if fields_needed:
    env.sim.expand_model_fields(tuple(fields_needed))
    model = env.sim.model
    if "dof_frictionloss" in fields_needed:
      dof = robot.indexing.joint_v_adr[idx12]
      model.dof_frictionloss[:, dof] = model.dof_frictionloss[:, dof] * fr[:, None]
    if "eq_solref" in fields_needed:
      sr = model.eq_solref  # (n, neq, 2)
      sr[:, :, 0] = torch.where(loop_t[:, None] > 0, loop_t[:, None].expand(-1, sr.shape[1]), sr[:, :, 0])
      print("eq_solref time constant per env, loop 0:", sr[:, 0, 0].cpu().numpy().tolist(), flush=True)

  # ---- bookkeeping
  default = robot.data.default_joint_pos[:, idx12].clone()
  mass_tot = float(env.sim.mj_model.body_mass[1:].sum())
  weight = mass_tot * 9.81
  base_bid = robot.find_bodies("base_link")[0][0]
  print(f"robot mass {mass_tot:.2f} kg, n envs {n}, steps {nsteps}", flush=True)

  sched = make_schedule(dur * 1000.0, args.seed)
  K = sched.shape[1]
  ncalls = (sched < dur * 1000.0 - 1e-6).sum(1)
  schedt = torch.tensor(sched, device=dev, dtype=torch.float32)
  ncalls_t = torch.tensor(ncalls, device=dev)
  ar = torch.arange(n, device=dev)
  back = torch.round(P("qage_ms", args.obs_age_ms) / DT_MS).long()
  gyro_scale = P("gyro_scale", 1.0)[:, None]
  imu_back = torch.round(P("imu_age_ms", 0.0) / DT_MS).long()
  still_thr, still_n = P("still_thr", 0.0), torch.round(P("still_s", 0.0) * 1000.0 / DT_MS)
  still_on = bool((still_thr > 0).any())
  B = 64
  Qr = torch.zeros(n, B, 12, device=dev)
  GB = torch.zeros(n, B, 6, device=dev)
  q_prev = robot.data.joint_pos[:, idx12].clone()
  last_raw = torch.zeros(n, 12, device=dev)
  held = torch.zeros(n, 12, device=dev)
  ptr = torch.zeros(n, dtype=torch.long, device=dev)
  still_cnt = torch.zeros(n, device=dev)

  imp_N, imp_ms, imp_h = P("imp_N", 0.0), P("imp_ms", 0.0), P("imp_h", 0.0)
  hk, hc, hh = P("hold_k", 0.0), P("hold_c", 0.0), P("hold_h", 0.1)
  sup = P("support", 0.0)
  hkr, hcr = P("hold_kr", 0.0), P("hold_cr", 0.0)
  exc_A, exc_f, exc_ax = P("exc_Nm", 0.0), P("exc_hz", 0.0), P("exc_axis", 1.0)
  push_N, push_h = P("push_N", 0.0), P("push_h", 0.1)
  wrench_on = bool(((imp_N > 0) | (push_N > 0) | (hk > 0) | (sup > 0) | (hkr > 0) | (exc_A > 0)).any())

  # outputs
  f32 = lambda *s: np.full(s, np.nan, np.float32)  # noqa: E731
  ACT, TGT, POS, POSTRUE, OBSG, TCALL = f32(K, n, 12), f32(K, n, 12), f32(K, n, 12), f32(K, n, 12), f32(K, n, 6), f32(K, n)
  STEP_G, STEP_T, STEP_XY = f32(nsteps, n, 3), f32(nsteps, n), f32(nsteps, n, 2)
  fell_k = np.full(n, -1, np.int64)
  x0 = None
  t0 = time.time()
  with torch.inference_mode():
    obs, _ = wenv.reset()
    x0 = robot.data.root_com_pos_w.clone()
    hold_x0 = None
    att_loc = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), hh], 1)  # hold point, body frame, rel. CoM
    for k in range(nsteps):
      t_ms = k * DT_MS
      q_now = robot.data.joint_pos[:, idx12]
      Qr[:, k % B] = q_now
      GB[:, k % B] = obs["actor"][:, 0:6]
      if still_on:
        gnorm = obs["actor"][:, 0:3].norm(dim=1)
        still_cnt = torch.where(gnorm < still_thr, still_cnt + 1, torch.zeros_like(still_cnt))
      pk = ptr.clamp(max=K - 1)
      due = (ptr < ncalls_t) & (t_ms + 1e-6 >= schedt[ar, pk])
      o = obs["actor"].clone()
      if due.any():
        q_obs = Qr[ar, (k - back).clamp(min=0) % B]
        imu = GB[ar, (k - imu_back).clamp(min=0) % B]
        o[:, 0:6] = imu
        o[:, 0:3] = o[:, 0:3] * gyro_scale
        if still_on:
          zero = (still_thr > 0) & (still_cnt > still_n)
          o[:, 0:3] = torch.where(zero[:, None], torch.zeros_like(o[:, 0:3]), o[:, 0:3])
        o[:, 6:18] = q_prev - default
        o[:, 18:30] = q_obs - default
        o[:, 30:42] = last_raw
        obs["actor"] = o
        raw = policy(obs)
        d = due[:, None]
        e_i, p_i = ar[due].cpu().numpy(), ptr[due].cpu().numpy()
        ACT[p_i, e_i] = raw[due].cpu().numpy()
        TGT[p_i, e_i] = (default + 0.25 * raw)[due].cpu().numpy()
        POS[p_i, e_i] = q_obs[due].cpu().numpy()
        POSTRUE[p_i, e_i] = q_now[due].cpu().numpy()
        OBSG[p_i, e_i] = o[:, 0:6][due].cpu().numpy()
        TCALL[p_i, e_i] = t_ms / 1000.0
        held = torch.where(d, raw, held)
        last_raw = torch.where(d, raw, last_raw)
        q_prev = torch.where(d, q_obs, q_prev)
        ptr = ptr + due.long()
      if wrench_on:
        F = torch.zeros(n, 3, device=dev)
        t_s = t_ms / 1000.0
        on = (imp_N > 0) & (t_s >= args.t_imp - 1e-6) & (t_s < args.t_imp + imp_ms / 1000.0 - 1e-6)
        Fi = torch.zeros(n, 3, device=dev)
        Fi[:, 0] = torch.where(on, imp_N, torch.zeros_like(imp_N))
        com = robot.data.root_com_pos_w
        quat = robot.data.root_link_quat_w
        # impulse acts at imp_h above the CoM (world +z), pure translation + torque r x F
        ri = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), imp_h], 1)
        Tq = torch.cross(ri, Fi, dim=1)
        F = F + Fi
        Fp = torch.zeros(n, 3, device=dev)
        Fp[:, 0] = push_N
        Tq = Tq + torch.cross(torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), push_h], 1), Fp, dim=1)
        F = F + Fp
        if (hk > 0).any():
          r = quat_rotate(quat, att_loc)
          p_att = com + r
          if hold_x0 is None:
            hold_x0 = p_att.clone()  # fixed world anchor = where the hold point is at the first step
          v_att = robot.data.root_com_lin_vel_w + torch.cross(robot.data.root_com_ang_vel_w, r, dim=1)
          Fh = torch.zeros(n, 3, device=dev)
          Fh[:, :2] = -hk[:, None] * (p_att[:, :2] - hold_x0[:, :2]) - hc[:, None] * v_att[:, :2]
          Fh = torch.where((hk > 0)[:, None], Fh, torch.zeros_like(Fh))
          Tq = Tq + torch.cross(r, Fh, dim=1)
          F = F + Fh
        if (hkr > 0).any():
          ez = torch.zeros(n, 3, device=dev)
          ez[:, 2] = 1.0
          zb = quat_rotate(quat, ez)
          w_w = robot.data.root_com_ang_vel_w.clone()
          w_w[:, 2] = 0.0
          Tr = hkr[:, None] * torch.cross(zb, ez, dim=1) - hcr[:, None] * w_w
          Tq = Tq + torch.where((hkr > 0)[:, None], Tr, torch.zeros_like(Tr))
        if (exc_A > 0).any():
          on_e = (exc_A > 0) & (t_s >= 3.0) & (t_s < 15.0)
          amp = torch.where(on_e, exc_A * torch.sin(2 * math.pi * exc_f * (t_s - 3.0)), torch.zeros_like(exc_A))
          Te = torch.zeros(n, 3, device=dev)
          Te[:, 1] = torch.where(exc_ax == 1, amp, torch.zeros_like(amp))
          Te[:, 0] = torch.where(exc_ax == 0, amp, torch.zeros_like(amp))
          Tq = Tq + Te
        F[:, 2] += sup * weight
        robot.write_external_wrench_to_sim(F[:, None, :], Tq[:, None, :], body_ids=[base_bid])
      obs, _, dones, ext = wenv.step(held)
      tm = env.reset_terminated.cpu().numpy()
      newly = tm & (fell_k < 0)
      fell_k[newly] = k
      STEP_G[k] = robot.data.root_link_ang_vel_b.cpu().numpy()
      STEP_T[k] = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1, 1)).cpu().numpy()
      STEP_XY[k] = (robot.data.root_com_pos_w[:, :2] - x0[:, :2]).cpu().numpy()
      if k % 400 == 0:
        print(f"step {k}/{nsteps}  {time.time() - t0:.0f}s  falls {int((fell_k >= 0).sum())}", flush=True)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(str(out) + ".npz", act=ACT, tgt=TGT, pos=POS, pos_true=POSTRUE, obs_gyro_grav=OBSG, t_call=TCALL,
                      step_gyro=STEP_G, step_tilt=STEP_T, step_xy=STEP_XY, fell_k=fell_k, n_calls=ptr.cpu().numpy(),
                      default=default[0].cpu().numpy(), cond=np.array([c for c, _ in grid]), rep=np.array([r for _, r in grid]))
  json.dump({"conditions": CONDITIONS, "args": vars(args), "wall_s": time.time() - t0, "robot_mass_kg": mass_tot,
             "motors": MOTORS, "dt_ms": DT_MS,
             "layout": "per-call arrays (K, n, ...) rows = policy calls of that env (NaN padded); step arrays (nsteps, n, ...) at 5 ms; "
                       "tgt/pos/default in rad; pos = joint angle the policy saw (aged), pos_true = at the call instant; fell_k = first step with a termination (-1 none)"},
            open(str(out) + ".json", "w"), indent=1)
  print(f"done wall={time.time() - t0:.0f}s falls={int((fell_k >= 0).sum())}/{n} -> {out}.npz", flush=True)


if __name__ == "__main__":
  main()
