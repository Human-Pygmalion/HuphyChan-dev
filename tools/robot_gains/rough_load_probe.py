#!/usr/bin/env python3
"""Per-joint load probe on rough terrain (box set, slopes, stairs) with a trained policy.

Why: the PD-gain criterion "walking torque P99 within 30 deg position error" (docs/169 s7) used FLAT-walking
torques. The robot will also walk mountain trails, so this measures the same quantities on the terrain the
rough policies were trained on (box terrain, PYG_BOX_TERRAIN) and on slopes / stairs the policies never saw.
Results feed docs/169 s8. Nothing here touches training code; the policy is the MEAN action of a checkpoint.

Run on a CUDA machine, from mujoco-sim/mjlab, always with the mjlab venv (the policy's own PYG_* flags are
read from <run>/repro/launch_manifest.json, exactly like analysis/terrain_eval.py):

  MUJOCO_GL=egl nice -n 10 .venv/bin/python3 ../../tools/robot_gains/rough_load_probe.py \
      --run logs/rsl_rl/pygmalion_velocity/<run> --ckpt model_11999.pt --suite box \
      --levels 0,3,6,9 --vx 0.3,0.6,1.0 --out /tmp/box_main.json

Suites
  box     the trained BOX_TERRAINS_CFG curriculum grid, every env pinned to one difficulty row (level), the five
          terrain columns spread evenly (flat / pyramid stairs / inverted pyramid stairs / random grid / tilted
          grid), random start heading as in training. Neighbouring tiles are other terrain types (as in training).
  slope   straight ramps built from a tilted box (NOT a heightfield: warp heightfield contact misbehaves for the
          foot, see velocity_env_cfg BOX_TERRAINS_CFG note), up and down, --angles degrees, heading fixed to +x.
  stairs  straight stairs (tread 0.30 m), up and down, --risers metres, heading fixed to +x.
  Slope and stair statistics only use steps while the base is on the feature (base x >= start - 0.3 m, i.e. the front foot
  can already stand on it; a robot stalled at the first riser therefore still counts).

Recorded per joint group (hip_pitch, hip_roll, hip_yaw, knee, crank; L and R pooled), only for steps before a
termination (fall) or a time-out/out-of-bounds truncation:
  |tau| RMS, P50, P99, P99.9, max [N*m]; |q_target - q| P50, P99, P99.9 [deg]; fraction of steps with |tau| at
  98 % of the actuator effort limit; fraction of steps with the target on the action clip; the same torque stats
  split into "landing" (<= 0.1 s = 5 control steps after a touchdown of the same-side foot) and "other"; per-env
  RMS (thermal). Per foot: ground reaction (contact-only force from the feet-ground contact sensor, NOT the loop
  constraint forces) peak in body weights. Falls per condition and per terrain column, cause counts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="run directory (reads repro/launch_manifest.json for PYG_* flags)")
ap.add_argument("--manifest-run", help="run directory whose repro/launch_manifest.json holds the PYG_* flags (default: --run; "
                "needed for a resumed run, whose own directory has no repro/)")
ap.add_argument("--ckpt", default="model_11999.pt")
ap.add_argument("--suite", choices=["box", "slope", "stairs"], required=True)
ap.add_argument("--levels", default="0,3,6,9", help="box suite: difficulty rows (0..9)")
ap.add_argument("--vx", default="0.6", help="comma list of forward commands [m/s]")
ap.add_argument("--angles", default="10,20,30", help="slope suite [deg]")
ap.add_argument("--risers", default="0.10,0.15,0.20,0.25", help="stairs suite [m]")
ap.add_argument("--envs", type=int, default=256)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# GAIN_AB_PROBE_DROP="PYG_X PYG_Y": flags that the manifest must NOT set (2026-10-17: PYG_PD_GAIN_DR of arm C, so that the
# probe runs with the nominal gains; default empty = unchanged behaviour).
_drop = os.environ.get("GAIN_AB_PROBE_DROP", "").split()
for k, v in json.load(open(Path(a.manifest_run or a.run) / "repro/launch_manifest.json"))["environment"].items():
  if k.startswith("PYG_") and k not in _drop:
    os.environ[k] = str(v)
for _k in _drop:
  os.environ.pop(_k, None)
os.environ.pop("PYG_STAGNATION", None)  # judge walking, not the stagnation rule itself (terrain_eval convention)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab.tasks  # noqa: E402, F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.terrains.terrain_generator import (  # noqa: E402
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

G = 9.81
TILE = (16.0, 8.0)  # custom tile (x length, y width)
SPAWN_X = 1.5  # spawn x inside the custom tile
FEATURE_X0 = 3.0  # slope / stairs start x inside the custom tile (flat start before)
TREAD = 0.30
N_STEPS = 20
ON_FEATURE_MARGIN = 0.3  # base x may be this far before the feature start and the front foot is already on it


# --------------------------------------------------------------------------- custom terrains
def _box(body, center, half, quat=(1.0, 0.0, 0.0, 0.0)):
  g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=center, quat=quat)
  return TerrainGeometry(geom=g, color=(0.55, 0.55, 0.6, 1.0))


@dataclass(kw_only=True)
class RampTerrainCfg(SubTerrainCfg):
  """Straight ramp from boxes: flat start, 12 m ramp of angle_deg, flat top. direction up|down."""

  angle_deg: float = 10.0
  direction: str = "up"
  ramp_run: float = 12.0

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    sx, sy = self.size
    th = math.radians(self.angle_deg)
    H = self.ramp_run * math.tan(th)
    x0, t = FEATURE_X0, 1.0
    geoms = [_box(body, (sx / 2, sy / 2, -0.5), (sx / 2, sy / 2, 0.5))]  # floor, top z = 0
    phi = -th if self.direction == "up" else th  # rotation about +y: x axis -> (cos, 0, -sin phi)
    quat = (math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0)
    n = np.array([math.sin(phi), 0.0, math.cos(phi)])  # top-face normal
    top_c = np.array([x0 + self.ramp_run / 2, sy / 2, H / 2])
    ctr = top_c - 0.5 * t * n
    Lr = self.ramp_run / math.cos(th)
    geoms.append(_box(body, tuple(ctr), (Lr / 2, sy / 2, t / 2), quat))
    if self.direction == "up":
      xe = x0 + self.ramp_run
      geoms.append(_box(body, ((xe + sx) / 2, sy / 2, (H - 0.5) / 2), ((sx - xe) / 2, sy / 2, (H + 0.5) / 2)))
      origin = (SPAWN_X, sy / 2, 0.0)
    else:
      geoms.append(_box(body, (x0 / 2, sy / 2, (H - 0.5) / 2), (x0 / 2, sy / 2, (H + 0.5) / 2)))
      origin = (SPAWN_X, sy / 2, H)
    return TerrainOutput(origin=np.array(origin), geometries=geoms)


@dataclass(kw_only=True)
class LineStairsTerrainCfg(SubTerrainCfg):
  """Straight stairs: flat start, N_STEPS treads of TREAD m with the given riser, then a platform."""

  riser: float = 0.15
  direction: str = "up"

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    sx, sy = self.size
    x0, h = FEATURE_X0, self.riser
    geoms = [_box(body, (sx / 2, sy / 2, -0.5), (sx / 2, sy / 2, 0.5))]
    top_h = N_STEPS * h
    if self.direction == "down":
      geoms.append(_box(body, (x0 / 2, sy / 2, (top_h - 0.5) / 2), (x0 / 2, sy / 2, (top_h + 0.5) / 2)))
    for k in range(1, N_STEPS + 1):
      hk = k * h if self.direction == "up" else (N_STEPS - k) * h
      xa, xb = x0 + (k - 1) * TREAD, x0 + k * TREAD
      geoms.append(_box(body, ((xa + xb) / 2, sy / 2, (hk - 0.5) / 2), (TREAD / 2, sy / 2, (hk + 0.5) / 2)))
    if self.direction == "up":
      xe = x0 + N_STEPS * TREAD
      geoms.append(_box(body, ((xe + sx) / 2, sy / 2, (top_h - 0.5) / 2), ((sx - xe) / 2, sy / 2, (top_h + 0.5) / 2)))
    origin = (SPAWN_X, sy / 2, 0.0 if self.direction == "up" else top_h)
    return TerrainOutput(origin=np.array(origin), geometries=geoms)


# --------------------------------------------------------------------------- env
cfg = load_env_cfg("Mjlab-Velocity-Rough-Pygmalion", play=False)
cfg.scene.num_envs = a.envs
cfg.seed = a.seed
cfg.curriculum = {}
cfg.episode_length_s = a.seconds + 5.0
conds: list[dict] = []  # each: name, column (custom suites), level, vx, meta
vxs = [float(x) for x in a.vx.split(",")]
if a.suite == "box":
  for lvl in [int(x) for x in a.levels.split(",")]:
    for vx in vxs:
      conds.append(dict(name=f"box_L{lvl}_vx{vx}", level=lvl, col=None, vx=vx, meta=dict(level=lvl)))
else:
  gens = {}
  if a.suite == "slope":
    for ang in [float(x) for x in a.angles.split(",")]:
      for d in ("up", "down"):
        gens[f"slope_{d}_{ang:g}"] = (RampTerrainCfg(angle_deg=ang, direction=d), dict(angle_deg=ang, direction=d))
  else:
    for r in [float(x) for x in a.risers.split(",")]:
      for d in ("up", "down"):
        gens[f"stairs_{d}_{int(round(r * 100))}cm"] = (
          LineStairsTerrainCfg(riser=r, direction=d),
          dict(riser_m=r, direction=d, tread_m=TREAD, steps=N_STEPS),
        )
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=TILE, border_width=10.0, num_rows=1, num_cols=len(gens), curriculum=True, seed=0,
    sub_terrains={k: v[0] for k, v in gens.items()}, add_lights=True,
  )
  cfg.scene.terrain.max_init_terrain_level = 0
  pr = cfg.events["reset_base"].params["pose_range"]
  pr["yaw"] = (0.0, 0.0)
  for ci, (k, (_, meta)) in enumerate(gens.items()):
    for vx in vxs:
      conds.append(dict(name=f"{k}_vx{vx}", level=0, col=ci, vx=vx, meta=meta))

tw = cfg.commands["twist"]
tw.ranges.lin_vel_x = (vxs[0], vxs[0])
tw.ranges.lin_vel_y = (0.0, 0.0)
tw.ranges.ang_vel_z = (0.0, 0.0)
for attr, val in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0), ("heading_command", False),
                  ("resampling_time_range", (1e6, 1e6))):
  if hasattr(tw, attr):
    setattr(tw, attr, val)
for attr in ("rel_forward_envs", "rel_world_envs"):
  if hasattr(tw, attr):
    setattr(tw, attr, 0.0)
if hasattr(tw, "heading_command") and not tw.heading_command:
  tw.ranges.heading = None
cfg.events.pop("push_robot", None)

env = ManagerBasedRlEnv(cfg, device="cuda")
dev = env.device
terrain = env.scene.terrain
robot = env.scene["robot"]
tm = env.termination_manager
term = env.action_manager.get_term("joint_pos")
tnames = list(term.target_names)
tids = term.target_ids
NJ = len(tnames)
sensor = env.scene["feet_ground_contact"]
foot_names = list(sensor.primary_names)  # order of the contact columns
foot_side = ["R" if n.startswith("R_") else "L" for n in foot_names]
mj = env.sim.mj_model
robot_mass = float(sum(mj.body_mass[int(b)] for b in robot.indexing.body_ids))
BW = robot_mass * G

GROUPS = {"hip_pitch": "hip_pitch", "hip_roll": "hip_roll", "hip_yaw": "hip_yaw", "knee": "knee"}


def group_of(n: str) -> str:
  for k in GROUPS:
    if k in n:
      return k
  assert "crank" in n, n
  return "crank"


jgroup = [group_of(n) for n in tnames]
jside = [n[0] for n in tnames]
side_to_foot = {s: foot_side.index(s) for s in "LR"}
LIMIT = {"hip_pitch": 120.0, "hip_roll": 120.0, "knee": 120.0, "hip_yaw": 60.0, "crank": 60.0}
lim_j = torch.tensor([LIMIT[g] for g in jgroup], device=dev)
clip = env.cfg.actions["joint_pos"].clip
clo = torch.tensor([clip[n][0] for n in tnames], device=dev)
chi = torch.tensor([clip[n][1] for n in tnames], device=dev)
dof_ids = torch.as_tensor(tids, device=dev)
foot_of_j = torch.tensor([side_to_foot[s] for s in jside], device=dev)
print(f"[probe] robot mass {robot_mass:.2f} kg  BW {BW:.1f} N  feet {foot_names}  conds {len(conds)}", flush=True)


class TorchActor:
  """Mean action from a training checkpoint (same as analysis/terrain_eval.py TorchActor, batched)."""

  def __init__(self, pt: str):
    sd = torch.load(pt, map_location="cpu", weights_only=False)["actor_state_dict"]
    self.mean, self.std = sd["obs_normalizer._mean"].to(dev), sd["obs_normalizer._std"].to(dev)
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    self.layers = [(sd[f"mlp.{i}.weight"].to(dev), sd[f"mlp.{i}.bias"].to(dev)) for i in idx]

  @torch.no_grad()
  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    x = (obs - self.mean) / (self.std + 0.01)
    for j, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if j < len(self.layers) - 1:
        x = torch.nn.functional.elu(x)
    return x


actor = TorchActor(str(Path(a.run) / a.ckpt))


def pct(x: torch.Tensor, qs: list[float]) -> list[float]:
  if x.numel() == 0:
    return [float("nan")] * len(qs)
  s = torch.sort(x.flatten()).values
  n = s.numel()
  return [float(s[min(n - 1, int(math.ceil(q / 100.0 * n)) - 1)]) for q in qs]


def tau_stats(x: torch.Tensor) -> dict:
  if x.numel() == 0:
    return dict(n=0)
  p50, p99, p999 = pct(x, [50, 99, 99.9])
  return dict(n=int(x.numel()), rms=float(torch.sqrt((x**2).mean())), p50=p50, p99=p99, p999=p999, max=float(x.max()))


# --------------------------------------------------------------------------- one condition
def run_condition(c: dict) -> dict:
  N, T = a.envs, int(round(a.seconds / env.step_dt))
  cmd_term = env.command_manager.get_term("twist")
  cmd_term.cfg.ranges.lin_vel_x = (c["vx"], c["vx"])
  terrain.terrain_levels[:] = c["level"]
  if c["col"] is not None:
    terrain.terrain_types[:] = c["col"]
  terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
  torch.manual_seed(a.seed)
  obs, _ = env.reset()
  cmd0 = env.command_manager.get_command("twist")[:, 0]
  assert torch.allclose(cmd0, torch.full_like(cmd0, c["vx"]), atol=1e-4), (float(cmd0.min()), float(cmd0.max()))
  types = terrain.terrain_types.clone()
  origin = terrain.env_origins.clone()

  tau = torch.zeros(T, N, NJ, device=dev)
  err = torch.zeros(T, N, NJ, device=dev)
  onclip = torch.zeros(T, N, NJ, dtype=torch.bool, device=dev)
  found = torch.zeros(T, N, 2, dtype=torch.bool, device=dev)
  grf = torch.zeros(T, N, 2, device=dev)
  valid = torch.zeros(T, N, dtype=torch.bool, device=dev)
  onfeat = torch.ones(T, N, dtype=torch.bool, device=dev)
  vel_err = torch.zeros(T, N, device=dev)
  alive = torch.ones(N, dtype=torch.bool, device=dev)
  fell = torch.zeros(N, dtype=torch.bool, device=dev)
  fell_t = torch.full((N,), -1, dtype=torch.long, device=dev)
  trunc = torch.zeros(N, dtype=torch.bool, device=dev)
  cause = {}
  reached_feat = torch.zeros(N, dtype=torch.bool, device=dev)
  start_xy = robot.data.root_link_pos_w[:, :2].clone()
  dist = torch.zeros(N, device=dev)
  last_lx = torch.zeros(N, device=dev)
  t0 = time.time()
  for t in range(T):
    act = actor(torch.nan_to_num(obs["actor"]))
    cmd = env.command_manager.get_command("twist")[:, :2].clone()
    obs, _, _, _, _ = env.step(act)
    d = robot.data
    done = env.reset_buf.bool()
    term_now = tm.terminated & alive
    ok = alive & ~done
    valid[t] = ok
    tau[t] = d.qfrc_actuator[:, dof_ids]
    err[t] = d.joint_pos_target[:, dof_ids] - d.joint_pos[:, dof_ids]
    proc = term._processed_actions
    onclip[t] = (proc <= clo + 1e-4) | (proc >= chi - 1e-4)
    found[t] = sensor.data.found.view(N, -1)[:, :2] > 0
    fh = sensor.data.force_history  # [N, P, H, 3] contact-only, global frame
    grf[t] = torch.linalg.norm(fh, dim=-1).amax(dim=-1)[:, :2]
    lx = d.root_link_pos_w[:, 0] - origin[:, 0] + SPAWN_X
    if c["col"] is not None:
      onfeat[t] = (lx >= FEATURE_X0 - ON_FEATURE_MARGIN) & (lx <= FEATURE_X0 + (12.0 if a.suite == "slope" else N_STEPS * TREAD))
      reached_feat |= alive & ~done & onfeat[t]
    vel_err[t] = (d.root_link_lin_vel_b[:, :2] - cmd).norm(dim=1)
    dist = torch.where(ok, (d.root_link_pos_w[:, :2] - start_xy).norm(dim=1), dist)
    last_lx = torch.where(ok, lx, last_lx)
    newly = term_now & ~fell
    if newly.any():
      for name in tm.active_terms:
        hit = tm.get_term(name) & newly
        if name != "time_out" and hit.any():
          cause[name] = cause.get(name, 0) + int(hit.sum())
    fell |= term_now
    fell_t = torch.where(newly, torch.full_like(fell_t, t), fell_t)
    trunc |= done & ~term_now & alive
    alive &= ~done
  wall = time.time() - t0

  # touchdown windows (<= 0.1 s = 5 steps after a touchdown of the same-side foot)
  touch = found & ~torch.cat([torch.zeros_like(found[:1]), found[:-1]], dim=0)
  land_f = torch.zeros_like(touch)
  for k in range(5):
    land_f[k:] |= touch[: T - k]
  m = valid & onfeat  # [T, N]
  mj_ = m[:, :, None].expand(-1, -1, NJ)
  land = land_f[:, :, :].gather(2, foot_of_j.view(1, 1, NJ).expand(T, N, NJ))
  tabs = tau.abs()
  edeg = err.abs() * 180.0 / math.pi

  def scope_masks(sel_env: torch.Tensor | None):
    base = mj_
    if sel_env is not None:
      base = base & sel_env.view(1, N, 1)
    return base

  def group_stats(base: torch.Tensor) -> dict:
    out = {}
    for g in dict.fromkeys(jgroup):
      js = torch.tensor([i for i, x in enumerate(jgroup) if x == g], device=dev)
      b = base[:, :, js]
      x = tabs[:, :, js][b]
      e = edeg[:, :, js][b]
      ld = land[:, :, js][b]
      lim = lim_j[js].view(1, 1, -1).expand(T, N, -1)[b]
      oc = onclip[:, :, js][b]
      p50e, p99e, p999e = pct(e, [50, 99, 99.9]) if e.numel() else (float("nan"),) * 3
      # per-env RMS (thermal): only envs with >= 100 valid samples per joint
      sq = (tabs[:, :, js] ** 2) * b
      cnt = b.sum(dim=0).float()  # [N, J]
      rms_env = torch.sqrt(sq.sum(dim=0) / cnt.clamp(min=1))[cnt >= 100]
      out[g] = dict(
        tau=tau_stats(x),
        tau_landing=tau_stats(x[ld]),
        tau_other=tau_stats(x[~ld]),
        err_deg=dict(p50=p50e, p99=p99e, p999=p999e),
        sat_frac=float((x >= 0.98 * lim).float().mean()) if x.numel() else float("nan"),
        clip_frac=float(oc.float().mean()) if x.numel() else float("nan"),
        rms_env_p50=pct(rms_env, [50])[0] if rms_env.numel() else float("nan"),
        rms_env_p95=pct(rms_env, [95])[0] if rms_env.numel() else float("nan"),
        landing_frac=float(ld.float().mean()) if x.numel() else float("nan"),
      )
    return out

  def grf_stats(sel_env: torch.Tensor | None) -> dict:
    out = {}
    for fi, fn in enumerate(foot_names):
      b = m if sel_env is None else (m & sel_env.view(1, N))
      x = grf[:, :, fi][b] / BW
      pe = (grf[:, :, fi] * b).amax(dim=0) / BW  # per-env peak
      pe = pe[b.any(dim=0)]
      out[fn] = dict(p99_bw=pct(x, [99])[0] if x.numel() else float("nan"), max_bw=float(x.max()) if x.numel() else float("nan"),
                     env_peak_p50_bw=pct(pe, [50])[0] if pe.numel() else float("nan"))
    return out

  res = dict(cond=c["name"], vx=c["vx"], level=c["level"], meta=c["meta"], wall_s=wall,
             n_envs=N, steps=T, valid_steps=int(m.sum()),
             falls=float(fell.float().mean()), truncated_oob=float(trunc.float().mean()),
             mean_fall_time_s=float(fell_t[fell].float().mean() * env.step_dt) if fell.any() else None,
             fall_causes=cause, vel_err_mean=float(vel_err[m].mean()) if m.any() else float("nan"),
             dist_mean=float(dist.mean()))
  if c["col"] is not None:
    res["reached_feature"] = float(reached_feat.float().mean())
    res["end_x_mean_m"] = float(last_lx.mean())  # base x in the tile at the last valid step (feature starts at 3.0)
    res["fell_after_feature"] = float((fell & reached_feat).float().mean())
    res["fell_before_feature"] = float((fell & ~reached_feat).float().mean())
  res["all"] = group_stats(scope_masks(None))
  res["grf"] = grf_stats(None)
  if a.suite == "box":
    names = list(cfg.scene.terrain.terrain_generator.sub_terrains.keys())
    res["per_type"] = {}
    for ti, nm in enumerate(names):
      sel = types == ti
      if not sel.any():
        continue
      res["per_type"][nm] = dict(falls=float(fell[sel].float().mean()), valid_steps=int(m[:, sel].sum()),
                                 stats=group_stats(scope_masks(sel)), grf=grf_stats(sel))
  return res


results = dict(run=str(a.run), ckpt=a.ckpt, suite=a.suite, seconds=a.seconds, envs=a.envs, robot_mass_kg=robot_mass,
               body_weight_N=BW, pyg_flags={k: v for k, v in os.environ.items() if k.startswith("PYG_")},
               conditions=[])
# gains actually built into the env (verify the crank Kp that the manifest asked for)
gains = {}
for act in robot.actuators:
  names = list(act.target_names)
  kp = torch.as_tensor(act.stiffness).float()
  gains[",".join(names)] = dict(kp_env0=float(kp.flatten()[0]), kp_mean=float(kp.mean()))
results["built_gains_first_env"] = gains
print("[probe] built Kp per actuator group:", gains, flush=True)
for c in conds:
  r = run_condition(c)
  results["conditions"].append(r)
  print(f"[probe] {c['name']:<28} falls {r['falls'] * 100:5.1f} %  valid steps {r['valid_steps']:>7}  "
        f"vel err {r['vel_err_mean']:.3f}  crank|tau| P99 {r['all']['crank']['tau'].get('p99', float('nan')):5.1f}  "
        f"knee P99 {r['all']['knee']['tau'].get('p99', float('nan')):5.1f}  ({r['wall_s']:.0f} s)", flush=True)
  json.dump(results, open(a.out, "w"), indent=1)
env.close()
