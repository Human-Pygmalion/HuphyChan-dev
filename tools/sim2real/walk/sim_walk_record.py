#!/usr/bin/env python3
"""Record simulator reference data for the deployed `generalized_walk` policy (terrain10_2048_model_37400.pt).

Purpose: give the real-robot logs something to be compared against (spectra, frequency response, gait, yaw
drift). The policy runs deterministically (mean action) in the training task
`Pygmalion-Velocity-Generalized-Terrain10-2kResume19700` on STOCK mjlab (upstream 20f10e96) with the branch
package `vibecoded-audit-20261012/pygmalion_mjlab`. Nothing here edits the task; it only rewrites the config.

Setup (stock mjlab is NOT the mujoco-sim/mjlab fork; the fork venv supplies torch / mujoco-warp / rsl-rl only):
  git worktree add --detach <dir>/gw_branch origin/vibecoded-audit-20261012
  git -C mujoco-sim/mjlab worktree add --detach <dir>/stock_mjlab 20f10e96
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl nice -n 10 mujoco-sim/mjlab/.venv/bin/python3 \
      tools/sim2real/walk/sim_walk_record.py --branch-root <dir>/gw_branch/vibecoded-audit-20261012 \
      --mjlab-src <dir>/stock_mjlab/src --runs R1,R2,R3,R4 --out mujoco-sim/mjlab/analysis/out/s2r_walk

Runs (all: flat ground, 8 envs, mean action, start from the default pose, joint offset +-0.05 rad and base
velocity +-0.2 m/s as in the training reset; start position and heading forced to 0 so world +x is forward):
  R1  profile vx 0.5 (0-5 s) -> 1.0 (5-10 s), wz 0   NOMINAL: no DR, no obs noise, no obs lag, no actuator delay, no push
  R2  same profile                                   TRAINING: all DR events, obs noise, obs lag 0-1 step, actuator delay 0-1 substep; push OFF
  R3  vx 0.5, wz 0, 20 s  nominal      R4  vx 1.0, wz 0, 20 s  nominal
  R9  = R3 with training DR/noise/lag/delay (extra)   R10 = R4 with training DR/noise/lag/delay (extra)
  R6  = R3 (vx 0.5) and R7 = R4 (vx 1.0) with the policy called every 52 ms (fixed) instead of every 20 ms   [TIMING runs]
  R8  = R3 with the policy-call intervals of the real robot (diff of column t of Expr_Data/vx0gain1walkob(...)1021.csv,
        circular, random start per env)                                                                       [TIMING run]
  V0/V3/W5 x {_20ms, _52ms, _real}: 20 s steady command vx 0 / vx 0.3 / wz 0.5 (others 0) with nominal 20 ms stepping,
        fixed 52 ms calls, real-interval calls (compare with real logs, command 0/0, 0.3/0, 0/0.5; vx 0.5 is R3/R6/R8).
"R5" (gain scaling to HUPHY robot.yaml) is not run: robot_v1.0.yaml gains equal the training gains.

TIMING runs mimic the real loop the way tools/sim2real/gw_jitter_sim.py does: physics and env step 5 ms (decimation 1); the
policy is called when the scheduled time has passed; the action is HELD between calls (the observation's last-action term is
the raw output of the last call); the command reaches the motor 1 step (5 ms) late (FIFO); Kp/Kd go through the 12-bit MIT
quantiser (Kp 0-5000, Kd 0-100); joint angles in the observation are --obs-age-ms old (default 10 ms ~ loop cycle 12.4 ms
minus 3.3 ms reply offset); q_prev is the angle seen at the previous call. No noise, no DR, no push in TIMING runs.

Terrain: one flat 60 x 60 m Terrain10 strip (same contact solref/solimp as training level 0) instead of the 10 x 10 m
strips, because the profile walks 5-20 m. The full Terrain10 mix is not run (10 m strips end the walk after ~5 m).

Output per run: <out>/<run>.npz + <run>.json (sidecar describing every key). Row k of a 50 Hz array is the instant the
observation of policy step k was built (state BEFORE action k). 200 Hz arrays hold one row per physics substep, taken
at the substep start (the same instant the PD law reads q/qd).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--branch-root", required=True, help="dir containing pygmalion_mjlab/ and deploy/ (vibecoded-audit-20261012)")
ap.add_argument("--mjlab-src", required=True, help="stock mjlab src dir (…/src, contains mjlab/)")
ap.add_argument("--checkpoint", default="", help="default: <branch-root>/deploy/generalized_walk/checkpoints/terrain10_2048_model_37400.pt")
ap.add_argument("--runs", default="R1,R2,R3,R4")
ap.add_argument("--out", required=True)
ap.add_argument("--num-envs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--real-timing-csv", default="", help="real policy log whose column t (s) gives the call times, for *_real / R8 runs")
ap.add_argument("--obs-age-ms", type=float, default=10.0, help="TIMING runs: age of the joint angles in the observation")
ap.add_argument("--tag", default="", help="suffix of the output file name, e.g. _s1 for a run with --seed 1")
ap.add_argument("--smoke-seconds", type=float, default=0.0, help="override duration for a quick test")
ap.add_argument("--task", default="Pygmalion-Velocity-Generalized-Terrain10-2kResume19700")
args = ap.parse_args()

sys.path.insert(0, args.mjlab_src)  # stock mjlab must win over the fork's editable install
sys.path.insert(0, args.branch_root)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab  # noqa: E402
import pygmalion_mjlab  # noqa: E402,F401  (registers the tasks)
from pygmalion_mjlab import robot as pyg_robot  # noqa: E402
from pygmalion_mjlab.terrain10 import StripTerrainCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg  # noqa: E402

DT_POLICY, SUBSTEPS = 0.02, 4
MOTORS = [f"{s}_{j}_joint" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
PASSIVE = [f"{s}_ankle_{j}_joint" for s in "RL" for j in ("pitch", "roll")]
JOINTS16 = MOTORS + PASSIVE
PROFILE = [(0.0, 5.0, 0.5, 0.0), (5.0, 10.0, 1.0, 0.0)]  # (t0, t1, vx, wz)
RUNS = {
  "R1": dict(sched=PROFILE, dur=10.0, mode="nominal", desc="deployment profile, nominal"),
  "R2": dict(sched=PROFILE, dur=10.0, mode="train", desc="deployment profile, training DR/noise/lag/delay, no push"),
  "R3": dict(sched=[(0.0, 20.0, 0.5, 0.0)], dur=20.0, mode="nominal", desc="steady vx 0.5, nominal"),
  "R4": dict(sched=[(0.0, 20.0, 1.0, 0.0)], dur=20.0, mode="nominal", desc="steady vx 1.0, nominal"),
  "R9": dict(sched=[(0.0, 20.0, 0.5, 0.0)], dur=20.0, mode="train", desc="steady vx 0.5, training DR/noise/lag/delay"),
  "R10": dict(sched=[(0.0, 20.0, 1.0, 0.0)], dur=20.0, mode="train", desc="steady vx 1.0, training DR/noise/lag/delay"),
  "R6": dict(sched=[(0.0, 20.0, 0.5, 0.0)], dur=20.0, mode="nominal", timing=("fixed", 52.0), desc="steady vx 0.5, policy every 52 ms (held action, 5 ms command delay)"),
  "R7": dict(sched=[(0.0, 20.0, 1.0, 0.0)], dur=20.0, mode="nominal", timing=("fixed", 52.0), desc="steady vx 1.0, policy every 52 ms (held action, 5 ms command delay)"),
  "R8": dict(sched=[(0.0, 20.0, 0.5, 0.0)], dur=20.0, mode="nominal", timing=("real", 0), desc="steady vx 0.5, real policy-call intervals (held action, 5 ms command delay)"),
}
for _nm, _vx, _wz in (("V0", 0.0, 0.0), ("V3", 0.3, 0.0), ("W5", 0.0, 0.5)):
  _s = [(0.0, 20.0, _vx, _wz)]
  RUNS[f"{_nm}_20ms"] = dict(sched=_s, dur=20.0, mode="nominal", desc=f"steady vx {_vx} wz {_wz}, nominal 20 ms stepping")
  RUNS[f"{_nm}_52ms"] = dict(sched=_s, dur=20.0, mode="nominal", timing=("fixed", 52.0), desc=f"steady vx {_vx} wz {_wz}, policy every 52 ms")
  RUNS[f"{_nm}_real"] = dict(sched=_s, dur=20.0, mode="nominal", timing=("real", 0), desc=f"steady vx {_vx} wz {_wz}, real policy-call intervals")


# ---- scheduled command -------------------------------------------------------------------------------------
@dataclass(kw_only=True)
class ScheduledCommandCfg(UniformVelocityCommandCfg):
  schedule: tuple = ()  # ((t0, t1, vx, wz), ...) seconds since episode start

  def build(self, env):
    return ScheduledCommand(self, env)


class ScheduledCommand(UniformVelocityCommand):
  """Command = piecewise-constant function of episode time; vy = 0. Replaces random resampling."""

  def _apply(self, ids):
    t = self._env.episode_length_buf[ids].float() * self._env.step_dt
    vx = torch.zeros_like(t)
    wz = torch.zeros_like(t)
    for t0, t1, cvx, cwz in self.cfg.schedule:
      m = (t >= t0 - 1e-6) & (t < t1 - 1e-6)
      vx[m], wz[m] = cvx, cwz
    self.vel_command_b[ids] = torch.stack([vx, torch.zeros_like(t), wz], dim=-1)

  def _resample_command(self, env_ids):
    self._apply(env_ids)

  def _update_command(self):
    self._apply(torch.arange(self.num_envs, device=self.device))


# ---- env configuration -------------------------------------------------------------------------------------
def build_cfg(mode: str, sched, dur: float, num_envs: int, seed: int, timing: bool = False):
  cfg = load_env_cfg(args.task, play=False)  # training config: all DR events + obs noise + lags
  assert cfg.observations["actor"].enable_corruption, "expected training config with actor corruption on"
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  side = 60.0
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=89500, curriculum=False, size=(side, side), num_rows=1, num_cols=1, border_width=0.0, color_scheme="none",
      sub_terrains={"level1_flat": StripTerrainCfg(size=(side, side), slope_deg=0.0, ridge_height_range=(0.0, 0.0))},
    ),
    max_init_terrain_level=0,
  )
  cfg.curriculum = {}
  cfg.episode_length_s = int(1e9)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  tw = cfg.commands["twist"]
  cfg.commands["twist"] = ScheduledCommandCfg(**{f.name: getattr(tw, f.name) for f in fields(tw)}, schedule=tuple(sched))
  cfg.commands["twist"].rel_standing_envs = 0.0
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  # reset: fixed start position and heading, keep the training joint / velocity offsets
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.05), "yaw": (0.0, 0.0)}
  cfg.events.pop("torso_push", None)  # pushes OFF in every run
  if mode == "nominal":
    cfg.events = {k: cfg.events[k] for k in ("reset_base", "reset_robot_joints")}
    cfg.observations["actor"].enable_corruption = False
    th = cfg.observations["actor"].terms["motor_pos_history"]
    th.delay_min_lag = th.delay_max_lag = 0
    ent = cfg.scene.entities["robot"]
    lag = 1 if timing else 0  # TIMING runs: command reaches the motor one 5 ms step late
    ent.articulation.actuators = tuple(replace(a, delay_min_lag=lag, delay_max_lag=lag) for a in ent.articulation.actuators)
  if timing:
    assert mode == "nominal"
    cfg.decimation = 1  # env step = physics step = 5 ms; the policy schedule is ours
  return cfg


# ---- recorder ----------------------------------------------------------------------------------------------
class Recorder:
  def __init__(self, env, T: int, sub: int = SUBSTEPS):
    self.env, self.T, self.sub = env, T, sub
    E = env.num_envs
    self.robot = env.scene["robot"]
    names = list(self.robot.joint_names)
    self.idx16 = torch.tensor([names.index(n) for n in JOINTS16], device=env.device)
    self.idx12 = self.idx16[:12]
    at = env.action_manager.get_term("joint_pos")
    assert list(at.target_names) == MOTORS, f"action order differs from expected: {at.target_names}"
    self.act_term = at
    self.default12 = self.robot.data.default_joint_pos[:, self.idx12].cpu().numpy()
    # actuator -> columns in the 12-motor order
    self.acts = []
    for a in self.robot.actuators:
      cols = [MOTORS.index(n) for n in a.target_names]
      self.acts.append((a, torch.tensor(cols, device=env.device)))
    # feet
    sensor = env.scene["feet_ground_contact"]
    self.sensor = sensor
    pn = [p.split("/")[-1] for p in sensor.primary_names]
    self.foot_order = [pn.index("R_foot_link"), pn.index("L_foot_link")]
    mjm = env.sim.mj_model
    import mujoco
    sites = {}
    for sid, s in zip(self.robot.indexing.site_ids.tolist(), self.robot.indexing.sites):
      body = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.site_bodyid[sid]))
      sites[body.split("/")[-1]] = (sid, s.name.split("/")[-1])
    self.foot_site_local = []
    self.foot_site_names = []
    for body in ("R_foot_link", "L_foot_link"):
      sid, nm = sites[body]
      self.foot_site_local.append(list(self.robot.indexing.site_ids.tolist()).index(sid))
      self.foot_site_names.append(f"{nm} (on {body})")
    f = lambda *s: np.full(s, np.nan, np.float32)  # noqa: E731
    self.d50 = dict(
      obs=f(T, E, 44), action=f(T, E, 12), target=f(T, E, 12), q=f(T, E, 16), qd=f(T, E, 16), q_enc=f(T, E, 12),
      tau_first=f(T, E, 12), tau_mean=f(T, E, 12), tau_req_mean=f(T, E, 12), command=f(T, E, 3),
      base_pos=f(T, E, 3), base_quat=f(T, E, 4), base_yaw=f(T, E), base_lin_vel_w=f(T, E, 3), base_lin_vel_b=f(T, E, 3),
      base_ang_vel_w=f(T, E, 3), base_ang_vel_b=f(T, E, 3), proj_gravity=f(T, E, 3), foot_contact=np.zeros((T, E, 2), bool),
      foot_n_contacts=f(T, E, 2), foot_force_w=f(T, E, 2, 3), foot_pos_w=f(T, E, 2, 3), terminated=np.zeros((T, E), bool),
      timeout=np.zeros((T, E), bool),
    )
    n = T * sub
    self.d200 = dict(q=f(n, E, 16), qd=f(n, E, 16), target_eff=f(n, E, 12), tau=f(n, E, 12), tau_req=f(n, E, 12))
    self.k = 0  # substep counter
    self.tau_acc = None
    self._hook()

  def _hook(self):
    am = self.env.action_manager
    orig_apply = am.apply_action
    rec = self

    def apply_action():
      orig_apply()
      k = rec.k
      if k < rec.d200["q"].shape[0]:
        d = rec.robot.data
        rec.d200["q"][k] = d.joint_pos[:, rec.idx16].cpu().numpy()
        rec.d200["qd"][k] = d.joint_vel[:, rec.idx16].cpu().numpy()

    am.apply_action = apply_action
    for a, cols in self.acts:
      orig = a.compute

      def compute(cmd, _orig=orig, _a=a, _cols=cols):
        out = _orig(cmd)
        k = rec.k
        if k < rec.d200["q"].shape[0]:
          rec.d200["target_eff"][k][:, _cols.cpu().numpy()] = cmd.position_target.cpu().numpy()
          rec.d200["tau"][k][:, _cols.cpu().numpy()] = _a.applied_effort.cpu().numpy()
          rec.d200["tau_req"][k][:, _cols.cpu().numpy()] = _a.unclipped_effort.cpu().numpy()
        return out

      a.compute = compute
    orig_update = self.env.scene.update

    def update(dt):
      orig_update(dt)
      rec.k += 1

    self.env.scene.update = update

  def snapshot(self, i, obs_actor):
    d, D = self.robot.data, self.d50
    D["obs"][i] = obs_actor.cpu().numpy() if torch.is_tensor(obs_actor) else obs_actor
    D["q"][i] = d.joint_pos[:, self.idx16].cpu().numpy()
    D["qd"][i] = d.joint_vel[:, self.idx16].cpu().numpy()
    D["q_enc"][i] = d.joint_pos_biased[:, self.idx12].cpu().numpy()
    D["command"][i] = self.env.command_manager.get_command("twist").cpu().numpy()
    D["base_pos"][i] = d.root_link_pos_w.cpu().numpy()
    D["base_quat"][i] = d.root_link_quat_w.cpu().numpy()
    D["base_yaw"][i] = d.heading_w.cpu().numpy()
    D["base_lin_vel_w"][i] = d.root_link_lin_vel_w.cpu().numpy()
    D["base_lin_vel_b"][i] = d.root_link_lin_vel_b.cpu().numpy()
    D["base_ang_vel_w"][i] = d.root_link_ang_vel_w.cpu().numpy()
    D["base_ang_vel_b"][i] = d.root_link_ang_vel_b.cpu().numpy()
    D["proj_gravity"][i] = d.projected_gravity_b.cpu().numpy()
    s = self.sensor.data
    fo = self.foot_order
    D["foot_n_contacts"][i] = s.found[:, fo].cpu().numpy()
    D["foot_contact"][i] = s.found[:, fo].cpu().numpy() > 0
    D["foot_force_w"][i] = s.force[:, fo].cpu().numpy()
    D["foot_pos_w"][i] = d.site_pos_w[:, self.foot_site_local].cpu().numpy()

  def after_step(self, i, action, term, tout):
    D = self.d50
    D["action"][i] = action.cpu().numpy()
    D["target"][i] = self.act_term._processed_actions.cpu().numpy()
    D["terminated"][i], D["timeout"][i] = term.cpu().numpy(), tout.cpu().numpy()
    sl = slice(i * self.sub, (i + 1) * self.sub)
    D["tau_first"][i] = self.d200["tau"][i * self.sub]
    D["tau_mean"][i] = self.d200["tau"][sl].mean(0)
    D["tau_req_mean"][i] = self.d200["tau_req"][sl].mean(0)

  def statics(self):
    E = self.env.num_envs
    out = {}
    kp, kd, st = (np.zeros((E, 12), np.float32) for _ in range(3))
    for a, cols in self.acts:
      c = cols.cpu().numpy()
      kp[:, c], kd[:, c], st[:, c] = a.stiffness.cpu().numpy(), a.damping.cpu().numpy(), a.strength_scale.cpu().numpy()
    out.update(kp=kp, kd=kd, strength_scale=st, default_q12=self.default12[0].astype(np.float32))
    out["encoder_bias"] = self.robot.data.encoder_bias[:, self.idx12].cpu().numpy()
    return out


def sha256(p):
  h = hashlib.sha256()
  h.update(Path(p).read_bytes())
  return h.hexdigest()


def git(*a, cwd):
  try:
    return subprocess.check_output(["git", *a], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
  except Exception:
    return None


KEYS50 = {
  "t": "s, policy time at the row instant (k*0.02)",
  "obs": "actor observation EXACTLY as fed to the network (before the checkpoint's internal normaliser); 44 = [0:3 gyro rad/s, 3:6 proj. gravity, 6:18 motor q old frame (q - default, rad), 18:30 motor q new frame, 30:42 previous raw action, 42 vx, 43 wz]. Includes noise/lag when the run has them",
  "action": "raw policy output (mean action, unitless), computed from obs[k]; applied during policy step k",
  "target": "rad, processed joint target = clip(default + 0.25*action), policy-intent in encoder coordinates, motor/crank space; the sim target also subtracts the encoder bias (see target_eff in the 200 Hz data)",
  "q": "rad, TRUE simulated joint positions at the row instant, 16 joints in JOINTS16 order (12 motors, then passive R_ankle_pitch, R_ankle_roll, L_ankle_pitch, L_ankle_roll). Crank joints are in crank (motor) space",
  "qd": "rad/s, true joint velocities, same order as q",
  "q_enc": "rad, motor joint position as the encoder reports it (q + persistent encoder bias; no noise, no lag). Equals q in nominal runs",
  "tau_first": "N*m, applied (torque-speed clamped) actuator torque in the first physics substep of policy step k, 12 motors",
  "tau_mean": "N*m, mean applied torque over the 4 substeps of policy step k",
  "tau_req_mean": "N*m, mean requested (unclamped PD) torque over the 4 substeps of policy step k",
  "command": "[vx m/s, vy m/s, wz rad/s] body-frame command at the row instant (vy always 0; the network sees only vx, wz)",
  "base_pos": "m, base_link position, world frame (start (30,30,~0.9) on the 60 m strip: coordinates are offset by the terrain origin; use differences)",
  "base_quat": "base_link orientation quaternion (w,x,y,z), world",
  "base_yaw": "rad, heading (yaw of the base x axis), wrapped to +-pi",
  "base_lin_vel_w": "m/s, base_link linear velocity, world frame", "base_lin_vel_b": "m/s, base_link linear velocity, body frame",
  "base_ang_vel_w": "rad/s, base angular velocity, world", "base_ang_vel_b": "rad/s, base angular velocity, body frame (noise-free; the actor sees the IMU sensor value, obs[0:3])",
  "proj_gravity": "unit gravity direction in body frame (noise-free); upright = (0,0,-1)",
  "foot_contact": "bool [R, L], foot touches the terrain (contact count > 0)",
  "foot_n_contacts": "number of active contact points per foot [R, L]",
  "foot_force_w": "N, net contact force on each foot [R, L] x (x,y,z), world frame",
  "foot_pos_w": "m, foot site positions [R, L] x (x,y,z), world frame",
  "terminated": "bool, the env terminated (fell) during policy step k and was auto-reset; rows after it belong to a new episode",
  "timeout": "bool, time-out reset during step k (never with the long episode used here)",
}
KEYS200 = {
  "t200": "s, k*0.005",
  "q200": "rad, true joint positions at substep start, 16 joints (JOINTS16 order)",
  "qd200": "rad/s, joint velocities at substep start, 16 joints",
  "target_eff200": "rad, position target the PD law actually used in that substep (after encoder-bias subtraction and the 0-1 substep actuator delay), 12 motors",
  "tau200": "N*m, applied torque after the torque-speed clamp, 12 motors",
  "tau_req200": "N*m, requested PD torque before the clamp, 12 motors",
}
KEYS_STATIC = {
  "kp": "per-env, per-motor Kp as simulated (DR scale in train mode)", "kd": "per-env per-motor Kd",
  "strength_scale": "per-env per-motor torque-speed envelope scale (1 nominal)",
  "default_q12": "rad, default pose (policy offset), 12 motors", "encoder_bias": "rad, per-env per-motor persistent encoder bias",
}


def load_real_intervals_ms():
  """Intervals between policy calls [ms] = diff of column t of a real policy log."""
  assert args.real_timing_csv, "--real-timing-csv is required for *_real / R8 runs"
  t = []
  for ln in open(args.real_timing_csv, encoding="utf-8"):
    if ln.startswith("#") or ln.startswith("t,"):
      continue
    t.append(float(ln.split(",", 1)[0]))
  return np.diff(np.array(t)) * 1000.0


def quant(x, xmax):  # 12-bit MIT field quantiser, as in gw_jitter_sim.py
  return torch.floor(torch.clamp(x, 0, xmax) / xmax * 4095) * xmax / 4095


def call_schedule(kind, E, dur_ms, seed):
  """Per-env scheduled call times [ms], first call at 0. Returns (E, K)."""
  if kind[0] == "fixed":
    iv = np.full((E, int(dur_ms / kind[1]) + 4), kind[1])
  else:
    dts = load_real_intervals_ms()
    K = int(dur_ms / max(dts.min(), 5.0)) + 4
    rng = np.random.default_rng(seed + 11)
    iv = np.stack([np.take(dts, np.arange(st, st + K), mode="wrap") for st in rng.integers(0, len(dts), E)])
  return np.concatenate([np.zeros((E, 1)), np.cumsum(iv, 1)[:, :-1]], 1)


def run_one(name: str, ckpt: Path, out: Path):
  spec = RUNS[name]
  if args.smoke_seconds:
    spec = {**spec, "dur": args.smoke_seconds}
  timing = spec.get("timing")
  sub = 1 if timing else SUBSTEPS
  T = int(round(spec["dur"] / DT_POLICY))  # policy steps (nominal stepping) or 4 x 5 ms steps per 20 ms
  nsteps = T * SUBSTEPS
  cfg = build_cfg(spec["mode"], spec["sched"], spec["dur"], args.num_envs, args.seed, timing=bool(timing))
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, asdict(agent_cfg), device=args.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  torch.manual_seed(args.seed)
  env.seed(args.seed)
  if timing:  # Kp/Kd through the 12-bit MIT quantiser
    for a_ in env.scene["robot"].actuators:
      a_.stiffness[:] = quant(a_.default_stiffness, 5000.0)
      a_.damping[:] = quant(a_.default_damping, 100.0)
  obs, _ = wenv.reset()
  rec = Recorder(env, nsteps if timing else T, sub)
  st = rec.statics()
  t0 = time.time()
  extra = {}
  with torch.inference_mode():
    if not timing:
      for i in range(T):
        rec.snapshot(i, obs["actor"])
        act = policy(obs)
        obs, _, dones, ext = wenv.step(act)
        rec.after_step(i, act, env.reset_terminated, env.reset_time_outs)
        if i % 100 == 0:
          print(f"[{name}] step {i}/{T}  {time.time() - t0:.0f}s", flush=True)
      assert rec.k == T * SUBSTEPS, (rec.k, T)
    else:
      E, dev = env.num_envs, env.device
      sched = call_schedule(timing, E, spec["dur"] * 1000.0, args.seed)  # ms
      ncalls = (sched < spec["dur"] * 1000.0 - 1e-6).sum(1)
      schedt = torch.tensor(sched, device=dev, dtype=torch.float32)
      ar = torch.arange(E, device=dev)
      B, dt_ms = 64, env.step_dt * 1000.0
      Qr = torch.zeros(E, B, 12, device=dev)
      default = rec.robot.data.default_joint_pos[:, rec.idx12].clone()
      q_prev = rec.robot.data.joint_pos[:, rec.idx12].clone()
      last_raw = torch.zeros(E, 12, device=dev)
      held = torch.zeros(E, 12, device=dev)
      ptr = torch.zeros(E, dtype=torch.long, device=dev)
      back = int(round(args.obs_age_ms / dt_ms))
      due_hist = np.zeros((nsteps, E), bool)
      nan_obs = np.full((E, 44), np.nan, np.float32)
      for k in range(nsteps):
        q_now = rec.robot.data.joint_pos[:, rec.idx12]
        Qr[:, k % B] = q_now
        due = (ptr < torch.tensor(ncalls, device=dev)) & (k * dt_ms + 1e-6 >= schedt[ar, ptr.clamp(max=schedt.shape[1] - 1)])
        o = obs["actor"].clone()
        if due.any():
          q_obs = Qr[:, (k - back) % B] if k >= back else q_now
          o[:, 6:18] = q_prev - default
          o[:, 18:30] = q_obs - default
          o[:, 30:42] = last_raw
          obs["actor"] = o
          raw = policy(obs)
          held = torch.where(due[:, None], raw, held)
          last_raw = torch.where(due[:, None], raw, last_raw)
          q_prev = torch.where(due[:, None], q_obs, q_prev)
          ptr = ptr + due.long()
        od = o.cpu().numpy()
        due_np = due.cpu().numpy()
        due_hist[k] = due_np
        rec.snapshot(k, np.where(due_np[:, None], od, nan_obs))
        obs, _, dones, ext = wenv.step(held)
        rec.after_step(k, held, env.reset_terminated, env.reset_time_outs)
        if k % 400 == 0:
          print(f"[{name}] step {k}/{nsteps}  {time.time() - t0:.0f}s", flush=True)
      assert rec.k == nsteps, (rec.k, nsteps)
      # gather per-call rows (padded with NaN / False)
      nmax = int(due_hist.sum(0).max())
      rows = [np.where(due_hist[:, e])[0] for e in range(E)]
      call = {}
      tau = rec.d200["tau"]
      for key, v in rec.d50.items():
        if key in ("tau_first", "tau_mean", "tau_req_mean", "timeout"):
          continue
        pad = np.zeros((nmax, E) + v.shape[2:], v.dtype) if v.dtype == bool else np.full((nmax, E) + v.shape[2:], np.nan, np.float32)
        for e in range(E):
          pad[: len(rows[e]), e] = v[rows[e], e]
        call[key] = pad
      tf = np.full((nmax, E, 12), np.nan, np.float32)
      tm = np.full((nmax, E, 12), np.nan, np.float32)
      tcall = np.full((nmax, E), np.nan, np.float32)
      for e in range(E):
        r = rows[e]
        tcall[: len(r), e] = r * dt_ms / 1000.0
        ends = np.append(r[1:], nsteps)
        tf[: len(r), e] = tau[r, e]
        tm[: len(r), e] = np.stack([tau[a_:b_, e].mean(0) for a_, b_ in zip(r, ends)])
      call.update(tau_first=tf, tau_mean=tm, t_call=tcall, n_calls=due_hist.sum(0).astype(np.int32),
                  t_sched=np.where(np.arange(nmax)[:, None] < ncalls[None], sched[:, :nmax].T, np.nan).astype(np.float32) / 1000.0)
      call = {k: v for k, v in call.items()}
      extra = call
      extra["_every_step"] = {k + "200": rec.d50[k] for k in (
        "base_pos", "base_quat", "base_yaw", "base_lin_vel_w", "base_lin_vel_b", "base_ang_vel_w", "base_ang_vel_b", "proj_gravity",
        "foot_contact", "foot_n_contacts", "foot_force_w", "foot_pos_w", "command", "action", "target", "terminated")}
      iv_all = np.concatenate([np.diff(tcall[: n_, e]) for e, n_ in enumerate(due_hist.sum(0))]) * 1000.0
      extra["_interval_stats_ms"] = {"mean": float(iv_all.mean()), "median": float(np.median(iv_all)), "p5": float(np.percentile(iv_all, 5)),
                                     "p95": float(np.percentile(iv_all, 95)), "max": float(iv_all.max())}
  out.mkdir(parents=True, exist_ok=True)
  if timing:
    arr = {k: v for k, v in extra.items() if not k.startswith("_")}
    arr.update(extra["_every_step"])
    arr["t"] = arr["t_call"]  # per-call times (NaN padded), see n_calls
  else:
    arr = {k: v for k, v in rec.d50.items()}
    arr["t"] = (np.arange(T) * DT_POLICY).astype(np.float32)
  for k, v in rec.d200.items():
    arr[k + "200"] = v
  arr["t200"] = (np.arange(nsteps) * DT_POLICY / SUBSTEPS).astype(np.float32)
  arr.update(st)
  np.savez_compressed(out / f"{name}{args.tag}.npz", **arr)
  n_fall = int((arr["terminated200"] if timing else arr["terminated"]).any(0).sum())
  repo = Path(__file__).resolve().parents[3]
  meta = {
    "run": name + args.tag, "description": spec["desc"], "mode": spec["mode"], "seed": args.seed, "num_envs": args.num_envs,
    "duration_s": spec["dur"], "policy_hz": 50 if not timing else None, "physics_hz": 200, "substeps": sub, "device": args.device,
    "policy": {"checkpoint": ckpt.name, "sha256": sha256(ckpt), "task": args.task, "deterministic_mean_action": True},
    "software": {"mjlab": getattr(mjlab, "__version__", "?"), "mjlab_src": args.mjlab_src, "branch_root": args.branch_root,
                 "branch_commit": git("rev-parse", "HEAD", cwd=args.branch_root), "script": "tools/sim2real/walk/sim_walk_record.py",
                 "repo_commit": git("rev-parse", "HEAD", cwd=repo)},
    "joint_order_motors12": MOTORS, "joint_order_16": JOINTS16,
    "joint_order_note": "right leg first; per leg hip_pitch, hip_roll, hip_yaw, knee, ankle_a(=crank_A), ankle_b(=crank_B). The last four of the 16 are the passive ankle pitch/roll hinges (R_pitch, R_roll, L_pitch, L_roll). Right/left follow the model names R_/L_; the sign convention of left joints is mirrored (see ACTION_CLIP in pygmalion_mjlab/robot.py)",
    "foot_order": ["R", "L"], "foot_sites": rec.foot_site_names,
    "action_transform": "target = default_pose + 0.25 * action, clipped to ACTION_CLIP; PD torque clamped by the measured torque-speed curve",
    "gains_nominal": {"hip_pitch/hip_roll": [135, 6], "hip_yaw": [75, 4], "knee": [130, 6], "crank_A/B": [60, 2.7]},
    "command_schedule": {"columns": ["t0_s", "t1_s", "vx_mps", "wz_radps"], "segments": spec["sched"], "vy": 0.0,
                         "note": "command at policy step k uses t = k*0.02 (TIMING runs: 5 ms env steps, t = k*0.005); vx switches at the first row with t >= t0"},
    "start": "default pose; reset joint offset U(-0.05,0.05) rad, base z 0.01-0.05 m, base velocity x/y/yaw U(-0.2,0.2); start xy offset and yaw forced to 0",
    "terrain": "single flat 60x60 m Terrain10 strip (training contact solref/solimp); base starts at its centre",
    "noise_and_dr": {
      "reset_joint_and_velocity_offsets": True,
      "domain_randomization": spec["mode"] == "train",
      "dr_list": ("link pseudo-inertia 0.9-1.1, base CoM +-3 cm, Kp/Kd x0.9-1.1, motor strength x0.8-1.2, joint friction/damping x0.8-1.2, foot friction U(0.3,1.2), encoder bias +-0.01 rad (all per env, sampled at reset)" if spec["mode"] == "train" else "none"),
      "actor_obs_noise": spec["mode"] == "train",
      "actor_obs_noise_detail": "gyro U(-0.2,0.2) rad/s, gravity U(-0.05,0.05), motor q U(-0.01,0.01) rad, every policy step" if spec["mode"] == "train" else "none",
      "motor_q_obs_lag_policy_steps": "0-1 per env (sampled once per reset)" if spec["mode"] == "train" else 0,
      "actuator_command_delay": "0-1 physics substep (0-5 ms), resampled every substep" if spec["mode"] == "train" else ("fixed 1 step = 5 ms" if timing else "none"),
      "push": False,
    },
    "timing": ({"kind": timing[0], "fixed_interval_ms": timing[1] if timing[0] == "fixed" else None, "real_csv": args.real_timing_csv if timing[0] == "real" else None,
                "env_step_ms": 5, "command_delay_steps": 1, "action_held_between_calls": True, "obs_joint_age_ms": args.obs_age_ms,
                "kp_kd_quantised_12bit": True, "interval_stats_ms": extra["_interval_stats_ms"], "n_calls_per_env": arr["n_calls"].tolist(),
                "layout": "per-call arrays (obs, action, target, q, ... ) have shape (Ncalls_max, E, ...), rows = policy calls of that env, padded with NaN/False; t_call (Ncalls_max,E) = call time [s] on the 5 ms grid; t_sched = scheduled time. Arrays with suffix 200 have one row per 5 ms step (action200/target200 = HELD values)"}
               if timing else None),
    "falls": {"envs_terminated_at_least_once": n_fall, "note": "if > 0 rows after the first terminated flag of that env are from a new episode; mask them"},
    "keys_50hz": {k: v for k, v in KEYS50.items() if k in arr},
    "keys_200hz": {k: v for k, v in KEYS200.items() if k in arr},
    "keys_note_timing": "TIMING runs: the 50 Hz key table describes per-CALL rows; extra per-call keys: t_call, t_sched, n_calls. Keys ending in 200 other than the table above are the same quantities sampled at every 5 ms step (see keys_50hz for meaning)." if timing else None,
    "keys_static": KEYS_STATIC,
    "shapes": {k: list(v.shape) for k, v in arr.items()},
    "wall_time_s": round(time.time() - t0, 1),
  }
  (out / f"{name}{args.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
  print(f"[{name}] done: falls={n_fall}  wall={time.time() - t0:.0f}s -> {out / (name + args.tag + '.npz')}", flush=True)
  return meta


def main():
  ckpt = Path(args.checkpoint or Path(args.branch_root) / "deploy/generalized_walk/checkpoints/terrain10_2048_model_37400.pt")
  assert sha256(ckpt) == "7bbd5cbc1431fb8085b6533f457b4cbb7eb20b960f4c033a6907c3e7fe0fbe00", "unexpected checkpoint"
  for name in args.runs.split(","):
    run_one(name.strip(), ckpt, Path(args.out))


if __name__ == "__main__":
  main()
