#!/usr/bin/env python3
"""Real HUPHY standing logs vs the deployed policy in simulation (task #152, 2026-10-17).

Question: "IMU fixed, the robot stands, a small push makes the policy diverge - is the PD gain
too low, or is the policy out of its training range?"  Three stages, one file:

  sim      (mjlab python, CPU)  stand + push the deployed policy at given gain scales / timing and
                                 save per-step traces (observation 45, action, target, joint angle).
  compare  (any python with numpy+matplotlib)  read the real logs + the sim traces, write the tables
                                 (JSON) and the figures.
  conds    print the per-log conditions only (no sim needed).

Real logs: HUPHY `archivedlogs/` (policy log "...ob.csv" = 82 columns, latency log "...latency.csv").
Policy log columns (stand.OBS_NAMES): t, obs 45 (gyro 3, gravity 3, q(t-1)-default 12, q(t)-default 12,
last action 12, command 3), act 12 (raw model output), tgt 12 (deg, before the safety guards), pos 12 (deg).
Gain scale and push are NOT recorded; the scale comes from the file name (see conds output).

  # 1. simulation (one heavy process at a time; decimation is global per process)
  CUDA_VISIBLE_DEVICES="" nice -n 10 <mjlab python> tools/sim2real/real_log_vs_sim.py sim \
      --pkg <pkg dir> --checkpoint <model.pt> --logs <archivedlogs dir> --decimation 5 \
      --conds "0.4/1.0:real@041,0.4/0.4:real@041" --out <prefix>
  # 2. tables and figures
  <python> tools/sim2real/real_log_vs_sim.py compare --logs <archivedlogs dir> --sim <prefix1> <prefix2> ... \
      --out-json <summary.json> --img-dir docs/sim2real/img
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------------------------
# constants (HUPHY control/stand.py; deploy_asset/pyg_rough_main_20261012/deploy_contract.json)
ORDER = ["right_leg/hip_pitch", "right_leg/hip_roll", "right_leg/hip_yaw", "right_leg/knee", "right_leg/ankle_a", "right_leg/ankle_b",
         "left_leg/hip_pitch", "left_leg/hip_roll", "left_leg/hip_yaw", "left_leg/knee", "left_leg/ankle_a", "left_leg/ankle_b"]
SHORT = ["R hip_pitch", "R hip_roll", "R hip_yaw", "R knee", "R ankle_a", "R ankle_b",
         "L hip_pitch", "L hip_roll", "L hip_yaw", "L knee", "L ankle_a", "L ankle_b"]
DEFAULT_DEG = np.array([-10.0268, 0, 0, 20.0535, 8.2480, -8.4261, 10.0268, 0, 0, -20.0535, -8.2440, 8.4150])
ACTION_SCALE = 0.25
OBS_NAMES = (["gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z"] + [f"q_prev {s}" for s in SHORT] + [f"q {s}" for s in SHORT]
             + [f"last_act {s}" for s in SHORT] + ["cmd_vx", "cmd_vy", "cmd_wz"])
KNEE, ANKLE_A = [3, 9], [4, 10]
# gain scale of each log: (body, ankle).  File-name rule of the earlier logs: "00503" = 0.05 body / 0.3 ankle,
# "0051" = 0.05 / 1.0, "010" = 0.1 all.  So "031" = 0.3 / 1, "04 1" = 0.4 / 1, "051" = 0.5 / 1.
LOGS = {
    "031": dict(ob="031lstandob.csv", lat="031latency stand.csv", name_scale=(0.3, 1.0)),
    "041": dict(ob="04 1 stand ob.csv", lat="041standlatency.csv", name_scale=(0.4, 1.0)),
    "051": dict(ob="051stand ob.csv", lat="051standlatency.csv", name_scale=(0.5, 1.0)),
}


# ---------------------------------------------------------------------------------------------
# real logs
def load_ob(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return dict(t=d[:, 0], obs=d[:, 1:46], act=d[:, 46:58], tgt=d[:, 58:70], pos=d[:, 70:82])


def load_latency(path):
    rows = list(csv.reader(open(path)))
    head = rows[0][0]
    rows = rows[2:]
    hz = float(head.split("hz=")[1]); drain = float(head.split("drain_ms=")[1].split()[0])
    out = dict(hz=hz, drain_ms=drain, n=len(rows), joints={})
    t = np.array([float(r[0]) for r in rows if r[0]])
    out["span_s"] = float(t.max())
    c = Counter(r[3] for r in rows)
    out["late_pct"] = 100 * c["late"] / len(rows); out["lost_pct"] = 100 * c["lost"] / len(rows)
    by = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    for j in ORDER:
        rs = by[j]; cj = Counter(r[3] for r in rs)
        lat = np.array([float(r[2]) for r in rs if r[3] == "ontime"])
        out["joints"][j] = dict(n=len(rs), p50=float(np.median(lat)), p95=float(np.percentile(lat, 95)), p99=float(np.percentile(lat, 99)),
                                late_pct=100 * cj["late"] / len(rs), lost_pct=100 * cj["lost"] / len(rs))
    out["stale_by_joint"] = [(out["joints"][j]["late_pct"] + out["joints"][j]["lost_pct"]) / 100 for j in ORDER]
    return out


def rolling_rms(x, t, win):
    """RMS of x over the trailing window [t-win, t]."""
    out = np.zeros(len(x)); j = 0
    c = np.concatenate([[0], np.cumsum(x ** 2)])
    for i in range(len(x)):
        while t[i] - t[j] > win:
            j += 1
        out[i] = math.sqrt((c[i + 1] - c[j]) / (i + 1 - j))
    return out


def joint_vel(r):
    """Joint velocity in rad/s from the two history frames of the observation: (q - q_prev) / policy period."""
    dt = np.diff(r["t"], prepend=np.nan)
    dt[0] = np.nanmedian(dt)
    return (r["obs"][:, 18:30] - r["obs"][:, 6:18]) / dt[:, None]


def phases(r):
    """Split a real log: settle / idle / disturbed / diverged.  Returns dict of boolean masks + event list.

    gyro = |angular velocity|.  A disturbance event starts when gyro first exceeds 0.08 rad/s after the robot has been
    quiet (1 s window RMS < 0.02) or, for a log with no quiet window, after the first 4 s.  Divergence = 0.3 s RMS of gyro > 0.5 rad/s or a knee more than 12 deg from its default angle.
    """
    t = r["t"]; gn = np.linalg.norm(r["obs"][:, 0:3], axis=1)
    rms1 = rolling_rms(gn, t, 1.0); rms03 = rolling_rms(gn, t, 0.3)
    quiet = np.where(rms1 < 0.02)[0]
    quiet = quiet[t[quiet] > 1.0]
    t_quiet = float(t[quiet[0]]) if len(quiet) else None
    t_start = t_quiet if t_quiet is not None else 4.0
    knee_dev = np.abs(r["pos"][:, KNEE] - DEFAULT_DEG[KNEE]).max(1)
    div = np.where((rms03 > 0.5) | (knee_dev > 12.0))[0]
    t_div = float(t[div[0]]) if len(div) else None
    events, last = [], -9.0
    for i in np.where((gn > 0.08) & (t > t_start))[0]:
        if t[i] - last > 1.0:
            events.append(float(t[i]))
        last = t[i]
    end = t_div if t_div is not None else float(t[-1])
    m_settle = t < t_start
    m_idle = (t >= t_start) & (t < (events[0] if events else end))
    m_after = (t >= (events[0] if events else end + 1)) & (t < end)
    m_div = t >= end if t_div is not None else np.zeros(len(t), bool)
    return dict(t_quiet=t_quiet, t_div=t_div, events=events, masks=dict(settle=m_settle, idle=m_idle, after_push=m_after, diverged=m_div), gyro=gn)


def dominant_hz(x, t):
    x = np.asarray(x) - np.mean(x)
    if len(x) < 8:
        return float("nan")
    dt = float(np.median(np.diff(t))); f = np.fft.rfftfreq(len(x), dt); a = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    band = (f > 0.3) & (f < 12)
    return float(f[band][np.argmax(a[band])]) if band.any() else float("nan")


def real_summary(key, logs_dir):
    spec = LOGS[key]; r = load_ob(os.path.join(logs_dir, spec["ob"])); lat = load_latency(os.path.join(logs_dir, spec["lat"]))
    t, obs = r["t"], r["obs"]; ph = phases(r); dt = np.diff(t) * 1000
    pos, tgt, act = r["pos"], r["tgt"], r["act"]
    s = dict(key=key, file=spec["ob"], scale_from_name=spec["name_scale"], rows=len(t), duration_s=float(t[-1]),
             policy_dt_ms=dict(mean=float(dt.mean()), median=float(np.median(dt)), p95=float(np.percentile(dt, 95)), p99=float(np.percentile(dt, 99)), max=float(dt.max()),
                               frac_over_30ms=float((dt > 30).mean())),
             grav_start=obs[0, 3:6].tolist(), act_abs_max=float(np.abs(act).max()), act_p99=float(np.percentile(np.abs(act), 99)),
             act_frac_gt3=float((np.abs(act) > 3).mean()),
             t_quiet=ph["t_quiet"], t_div=ph["t_div"], events=ph["events"], latency=lat)
    # target limit clip and the guard clip that the robot applied afterwards (guards are not logged; recomputed)
    lo = np.array([-112.7503, -19.5, -40.5, 6.0, -61.8794, -61.8794, -17.75, -79.5, -40.5, -114.0003, -61.8794, -61.8794])
    hi = np.array([17.75, 79.5, 40.5, 114.0003, 61.8794, 61.8794, 112.7503, 19.5, 40.5, -6.0, 61.8794, 61.8794])
    lim = np.clip(tgt, lo + 3, hi - 3); sent = np.clip(lim, pos - 50, pos + 50)
    s["guard_limit_pct"] = float(100 * (np.abs(lim - tgt) > 1e-6).mean()); s["guard_jump_pct"] = float(100 * (np.abs(sent - lim) > 1e-6).mean())
    s["tgt_at_model_limit_pct"] = float(100 * ((np.abs(tgt - lo) < 1e-3) | (np.abs(tgt - hi) < 1e-3)).mean())
    # tracking error in the idle window (target - measured), degrees
    for name, m in ph["masks"].items():
        if m.sum() >= 5:
            e = (tgt - pos)[m]
            s[f"phase_{name}"] = dict(t0=float(t[m][0]), t1=float(t[m][-1]), n=int(m.sum()),
                                      grav_x_mean=float(obs[m, 3].mean()), grav_y_mean=float(obs[m, 4].mean()),
                                      gyro_rms=float(np.sqrt((ph["gyro"][m] ** 2).mean())), gyro_max=float(ph["gyro"][m].max()),
                                      tracking_err_deg=e.mean(0).tolist(), act_abs_max=float(np.abs(act[m]).max()),
                                      act_p99=float(np.percentile(np.abs(act[m]), 99)),
                                      knee_std_deg=[float(pos[m, k].std()) for k in KNEE], ankle_a_std_deg=[float(pos[m, k].std()) for k in ANKLE_A])
    # events: peak gyro, grav change, joint-velocity burst within 1.5 s after the start
    vel = joint_vel(r); ev = []
    base_vel = np.abs(vel[ph["masks"]["idle"]]).max() if ph["masks"]["idle"].sum() > 3 else float("nan")
    for te in ph["events"]:
        m = (t >= te) & (t < te + 1.5)
        ev.append(dict(t=te, gyro_peak=float(ph["gyro"][m].max()), grav_x_swing=float(np.ptp(obs[m, 3])), grav_y_swing=float(np.ptp(obs[m, 4])),
                       joint_vel_peak_rad_s=float(np.abs(vel[m]).max()), act_max=float(np.abs(act[m]).max()),
                       fell=bool(ph["t_div"] is not None and te + 1.5 >= ph["t_div"])))
    s["event_detail"] = ev; s["idle_joint_vel_max_rad_s"] = float(base_vel)
    # divergence: frequency of the knee angle in the last 1.5 s before the end
    m = t > t[-1] - 1.5
    s["end_knee_std_deg"] = [float(pos[m, k].std()) for k in KNEE]
    s["end_knee_hz"] = dominant_hz(pos[m, 3], t[m])
    return s, r, ph


# ---------------------------------------------------------------------------------------------
# simulation stage
def run_sim(a):
    sys.path.insert(0, a.pkg)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from dataclasses import asdict
    import pygmalion_mjlab  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1), "none": (0, 0)}
    pushes = [float(p) for p in a.pushes.split(",")]
    grid = []  # (cond, push, dir, rep)
    for c in a.conds.split(","):
        spec, _, rest = c.partition("@")
        tag, _, aoff = rest.partition("~")  # "~X": ankle_a zero-offset error of X deg, right leg +X, left leg -X (mirror)
        gains, _, mode = spec.partition(":")
        if not rest:  # no '@': an offset may follow the mode directly ("0.4/1.0:real~-4")
            mode, _, aoff = mode.partition("~")
        body, ankle = (float(x) for x in gains.split("/"))
        stale = np.array(load_latency(os.path.join(a.logs, LOGS[tag]["lat"]))["stale_by_joint"]) if tag else np.full(12, 0.02)
        for p in [0.0] + pushes:
            for d in (["+x", "-x", "+y", "-y"] if p > 0 else ["none"]):
                for r in range(a.reps):
                    grid.append(dict(cond=c, body=body, ankle=ankle, mode=mode, stale=stale, push=p, dir=d, rep=r, aoff=float(aoff or 0.0)))
    n = len(grid)
    T_PUSH, T_END = a.t_push, a.t_push + a.t_after
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = n; cfg.seed = 7; cfg.episode_length_s = T_END + 5
    if a.decimation:
        cfg.decimation = a.decimation
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    tw = cfg.commands["twist"]; tw.heading_command = False; tw.ranges.heading = None
    tw.rel_standing_envs = 1.0; tw.resampling_time_range = (1e6, 1e6)
    env = ManagerBasedRlEnv(cfg, device=a.device)
    ag = load_rl_cfg(a.task); wenv = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
    runner = load_runner_cls(a.task)(wenv, asdict(ag), device=a.device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    dev = env.device; robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos"); jid = term.target_ids
    dt = env.step_dt
    print(f"envs {n}, policy period {dt*1000:.1f} ms, joints {list(term.target_names)}", flush=True)

    def quant(x, xmax):  # MIT codec: floor(x / xmax * 4095) * xmax / 4095
        return torch.floor(torch.clamp(x, 0, xmax) / xmax * 4095) * xmax / 4095

    body = torch.tensor([g["body"] for g in grid], device=dev).float(); ankle = torch.tensor([g["ankle"] for g in grid], device=dev).float()
    gains_seen = {}
    for act in robot.actuators:
        if getattr(act, "stiffness", None) is None:
            continue
        is_ankle = torch.tensor(["crank" in t for t in act.target_names], device=dev)
        s = torch.where(is_ankle[None, :], ankle[:, None], body[:, None])
        act.stiffness[:] = quant(act.default_stiffness * s, 5000.0); act.damping[:] = quant(act.default_damping * s, 100.0)
        for i, tn in enumerate(act.target_names):
            gains_seen[tn] = (float(act.default_stiffness[0, i]), float(act.default_damping[0, i]))
    print("package Kp/Kd at scale 1.0:", gains_seen, flush=True)

    mode = [g["mode"] for g in grid]
    real = torch.tensor([m == "real" for m in mode], device=dev); noise = torch.tensor([m == "train" for m in mode], device=dev)
    stale = torch.tensor(np.stack([g["stale"] for g in grid]), device=dev).float()
    off = torch.zeros(n, 12, device=dev)  # joint zero-offset error (calibration): measured = true + off; the sent target is in the measured frame
    for i, g in enumerate(grid):
        off[i, 4] = math.radians(g["aoff"]); off[i, 10] = -math.radians(g["aoff"])
    dv = torch.tensor([[DIRS[g["dir"]][0] * g["push"], DIRS[g["dir"]][1] * g["push"]] for g in grid], device=dev).float()
    default = robot.data.default_joint_pos[:, jid]; hard = robot.data.joint_pos_limits[:, jid]; m3 = math.radians(3)
    steps = int(round(T_END / dt)); k_push = int(round(T_PUSH / dt))
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    OBS, RAW, SENT, Q, TILT, ALIVE = [], [], [], [], [], []
    with torch.inference_mode():
        wenv.reset(); obs = wenv.get_observations()
        q_seen = robot.data.joint_pos[:, jid].clone(); hist = [q_seen.clone()] * 3; raw_prev = torch.zeros(n, 12, device=dev)
        for k in range(steps):
            if k == k_push:
                v = robot.data.root_link_vel_w.clone(); v[:, 0:2] += dv; robot.write_root_link_velocity_to_sim(v)
            o = obs["actor"].clone()
            q_now = robot.data.joint_pos[:, jid] + off
            keep = (torch.rand(n, 12, device=dev) < stale) & real[:, None]  # reply late/lost: the cache keeps the old angle
            q_seen = torch.where(keep, q_seen, q_now)
            hist = [hist[1], hist[2], q_seen.clone()]
            o[:, 6:18] = torch.where(real[:, None], hist[0], hist[1]) - default
            o[:, 18:30] = torch.where(real[:, None], hist[1], hist[2]) - default
            o[:, 30:42] = raw_prev
            # training noise (env.yaml actor terms): gyro +-0.2, gravity +-0.05 (not renormalised), joint angle +-0.01
            nz = noise[:, None].float()
            o[:, 0:3] += nz * (torch.rand(n, 3, device=dev) * 0.4 - 0.2)
            o[:, 3:6] += nz * (torch.rand(n, 3, device=dev) * 0.1 - 0.05)
            o[:, 6:30] += nz * (torch.rand(n, 24, device=dev) * 0.02 - 0.01)
            obs["actor"] = o
            raw = policy(obs)
            tgt = default + ACTION_SCALE * raw
            t1 = torch.minimum(torch.maximum(tgt, hard[..., 0] + m3), hard[..., 1] - m3)
            t2 = torch.minimum(torch.maximum(t1, q_seen - math.radians(50)), q_seen + math.radians(50))
            sent = torch.where(real[:, None], t2, tgt)
            OBS.append(o.cpu().numpy().astype(np.float32)); RAW.append(raw.cpu().numpy().astype(np.float32))
            Q.append(robot.data.joint_pos[:, jid].cpu().numpy().astype(np.float32)); SENT.append(tgt.cpu().numpy().astype(np.float32))
            g = robot.data.projected_gravity_b; TILT.append(torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))).cpu().numpy())
            ALIVE.append(alive.cpu().numpy().copy())
            obs, _, dones, _ = wenv.step((sent - off - default) / ACTION_SCALE)
            raw_prev = raw.clone()
            alive &= ~(dones.bool() & ~env.termination_manager.time_outs)
    meta = [dict(cond=g["cond"], push=g["push"], dir=g["dir"], rep=g["rep"]) for g in grid]
    np.savez_compressed(a.out + ".npz", obs=np.stack(OBS), raw=np.stack(RAW), tgt=np.stack(SENT), q=np.stack(Q), tilt=np.stack(TILT),
                        alive=np.stack(ALIVE), alive_end=alive.cpu().numpy(), dt=dt, k_push=k_push, default=default[0].cpu().numpy())
    json.dump(dict(args=vars(a), meta=meta, package_gains=gains_seen), open(a.out + ".json", "w"), indent=1)
    print("saved", a.out, flush=True)
    env.close()


# ---------------------------------------------------------------------------------------------
# compare stage
_SEEN: set = set()


def load_sim(prefix):
    """Condition names repeat across runs with another decimation (policy period); the later run gets the suffix ' dec<N>'."""
    z = np.load(prefix + ".npz"); j = json.load(open(prefix + ".json"))
    dec = j["args"].get("decimation") or 4
    names = list(dict.fromkeys(m["cond"] for m in j["meta"]))
    ren = {c: (c + f" dec{dec}" if c in _SEEN else c) for c in names}
    _SEEN.update(ren.values())
    for m in j["meta"]:
        m["cond"] = ren[m["cond"]]
    return dict(z={k: z[k] for k in z.files}, meta=j["meta"], args=j["args"])


def sim_stats(sim):
    """Per-condition summary from the sim traces."""
    z, dt, kp = sim["z"], float(sim["z"]["dt"]), int(sim["z"]["k_push"])
    out = defaultdict(list)
    idx = defaultdict(list)
    for i, m in enumerate(sim["meta"]):
        idx[m["cond"]].append(i)
    for c, ii in idx.items():
        rows = []
        for i in ii:
            m = sim["meta"][i]; al = z["alive"][:, i]
            n_alive = int(al.sum()); fell = n_alive < len(al)
            t_fall = n_alive * dt if fell else None
            a0 = int(2.0 / dt); idle = slice(a0, kp)  # 2 s .. push
            post = slice(kp, max(n_alive, kp + 2))
            q = z["q"][:, i]; tgt = z["tgt"][:, i]; act = z["raw"][:, i]
            r = dict(push=m["push"], dir=m["dir"], fell=fell, t_fall=t_fall,
                     fell_before_push=bool(fell and n_alive <= kp),
                     idle_knee_std=float(np.degrees(q[idle][:, KNEE].std(0).mean())) if n_alive > kp else None,
                     idle_ankle_err=np.degrees((tgt - q)[idle].mean(0)).tolist() if n_alive > kp else None,
                     idle_act_p99=float(np.percentile(np.abs(act[idle]), 99)) if n_alive > kp else None,
                     idle_tilt_mean=float(z["tilt"][idle, i].mean()) if n_alive > kp else None)
            if n_alive > kp + 2:
                p = slice(kp, n_alive)
                w = slice(kp, min(n_alive, kp + int(1.5 / dt)))  # first 1.5 s: what the real event table measures
                gyro = np.linalg.norm(z["obs"][w, i, 0:3], axis=1); gx = z["obs"][w, i, 3]; gy = z["obs"][w, i, 4]
                r.update(gyro_peak=float(gyro.max()), grav_x_swing=float(np.ptp(gx)), grav_y_swing=float(np.ptp(gy)),
                         joint_vel_peak=float(np.abs((z["obs"][w, i, 18:30] - z["obs"][w, i, 6:18]) / dt).max()))
                r.update(post_knee_std=float(np.degrees(q[p][:, KNEE].std(0).mean())), post_act_p99=float(np.percentile(np.abs(act[p]), 99)),
                         post_act_max=float(np.abs(act[p]).max()), tilt_max=float(z["tilt"][p, i].max()),
                         post_knee_hz=dominant_hz(q[p][:, 3], np.arange(n_alive - kp) * dt),
                         last2s_knee_std=float(np.degrees(q[max(kp, n_alive - int(2 / dt)):n_alive][:, KNEE].std(0).mean())))
            rows.append(r)
        out[c] = rows
    return out


def sim_table(stats):
    """Per condition and push size: fall count, median peak gyro / gravity swing in the first 1.5 s, knee std after the push, action p99."""
    nz = lambda v: float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else None  # noqa: E731
    tab = {}
    for c, rows in stats.items():
        tab[c] = {}
        for p in sorted({r["push"] for r in rows}):
            rr = [r for r in rows if r["push"] == p]
            g = lambda k: [r.get(k, np.nan) if r.get(k) is not None else np.nan for r in rr]  # noqa: E731
            tab[c][str(p)] = dict(n=len(rr), fell=int(sum(r["fell"] for r in rr)), fell_before_push=int(sum(r["fell_before_push"] for r in rr)),
                                  gyro_peak=nz(g("gyro_peak")), grav_x_swing=nz(g("grav_x_swing")), grav_y_swing=nz(g("grav_y_swing")),
                                  joint_vel_peak=nz(g("joint_vel_peak")), tilt_max=nz(g("tilt_max")), post_knee_std=nz(g("post_knee_std")),
                                  post_act_p99=nz(g("post_act_p99")), post_knee_hz=nz(g("post_knee_hz")), idle_knee_std=nz(g("idle_knee_std")),
                                  idle_act_p99=nz(g("idle_act_p99")), idle_tilt_mean=nz(g("idle_tilt_mean")),
                                  idle_ankle_a_err=[nz([r["idle_ankle_err"][j] if r["idle_ankle_err"] else np.nan for r in rr]) for j in ANKLE_A])
    return tab


def band(sim, cond, skip_s=1.0):
    """Per-channel training-like reference from a sim condition: P1, P99, mean, std over alive steps (>= skip_s) of all envs.
    The 45 observation channels plus 12 joint velocities."""
    z, dt = sim["z"], float(sim["z"]["dt"])
    ii = [i for i, m in enumerate(sim["meta"]) if m["cond"] == cond]
    sel = z["alive"][int(skip_s / dt):][:, ii]
    obs = z["obs"][int(skip_s / dt):][:, ii]
    vel = (obs[..., 18:30] - obs[..., 6:18]) / dt
    x = np.concatenate([obs, vel], -1)[sel]
    return dict(p1=np.percentile(x, 1, 0), p99=np.percentile(x, 99, 0), mean=x.mean(0), std=x.std(0), n=int(len(x)))


def real_channels(r):
    return np.concatenate([r["obs"], joint_vel(r)], 1)


def out_of_band(x, b, mask):
    """Fraction of rows outside [P1, P99] per channel and the largest |z| (z = (x-mean)/std of the reference)."""
    xs = x[mask]
    if len(xs) == 0:
        return None
    out = ((xs < b["p1"]) | (xs > b["p99"])).mean(0)
    z = np.abs((xs - b["mean"]) / np.maximum(b["std"], 1e-3))
    return dict(frac=out, zmax=z.max(0), zmed=np.median(z, 0), n=len(xs))


def normalizer():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deploy_asset", "pyg_rough_main_20261012", "deploy_contract.json")
    d = json.load(open(p))
    return np.array(d["obs_normalizer_mean"]), np.array(d["obs_normalizer_std"])


def run_compare(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reals, raw, phs = {}, {}, {}
    for k in LOGS:
        reals[k], raw[k], phs[k] = real_summary(k, a.logs)
    sims = [load_sim(p) for p in a.sim]
    stats = {}
    for s in sims:
        stats.update(sim_stats(s))
    allconds = {}
    for s in sims:
        for c in dict.fromkeys(m["cond"] for m in s["meta"]):
            allconds[c] = s
    summary = dict(real=reals, sim={c: stats[c] for c in stats})
    summary["sim_table"] = sim_table(stats)
    summary["ladder"] = ladder(a.logs)
    nm, ns = normalizer()
    # --- observation distribution: real vs sim reference bands
    refs = {}
    for name, cond in (("train", a.train_cond),):
        if cond in allconds:
            refs[name] = band(allconds[cond], cond)
    oob = {}
    for k in LOGS:
        x = real_channels(raw[k]); m = phs[k]["masks"]
        # same-gain closed loop reference (the real-timing sim at this log's scale, no push) for the "gain" reading
        cond_same = f"{LOGS[k]['name_scale'][0]}/{LOGS[k]['name_scale'][1]}:real@{k}"
        refs_k = dict(refs)
        if cond_same in allconds:
            refs_k["same_gain"] = band(allconds[cond_same], cond_same)
        oob[k] = {}
        for rn, b in refs_k.items():
            oob[k][rn] = {}
            for ph in ("settle", "idle", "after_push", "diverged"):
                r_ = out_of_band(x, b, m[ph])
                if r_ is not None:
                    oob[k][rn][ph] = dict(n=r_["n"], any_channel_pct=float(100 * (r_["frac"][:45] > 0).mean()),
                                          frac=[float(v) for v in r_["frac"]], zmax=[float(v) for v in r_["zmax"]], zmed=[float(v) for v in r_["zmed"]])
        # policy normaliser z (whole training run incl. walking): worst channel
        zn = np.abs((raw[k]["obs"] - nm) / (ns + 0.01))
        summary.setdefault("normalizer_z", {})[k] = {ph: dict(zmax=[float(v) for v in zn[m[ph]].max(0)], zmed=[float(v) for v in np.median(zn[m[ph]], 0)])
                                                  for ph in m if m[ph].sum() > 0}
    summary["out_of_band"] = oob
    summary["refs"] = {n: {k: [float(v) for v in b[k]] for k in ("p1", "p99", "mean", "std")} | dict(n=b["n"]) for n, b in refs.items()}
    json.dump(summary, open(a.out_json, "w"), indent=1)
    os.makedirs(a.img_dir, exist_ok=True)
    make_figures(a, reals, raw, phs, stats, sims, refs, oob, plt)
    extra_figures(a, reals, raw, phs, stats, plt)
    print("wrote", a.out_json)


def main_cond(c):
    """Conditions drawn in the figures: the 25 ms real-timing runs; period and offset variants stay in the JSON."""
    return " dec" not in c and "~" not in c and ":train" not in c and ":ideal" not in c


def make_figures(a, reals, raw, phs, stats, sims, refs, oob, plt):
    day = a.date
    labels = {"031": "log 031 (name: body 0.3 / ankle 1)", "041": "log 04 1 (body 0.4 / ankle 1)", "051": "log 051 (body 0.5 / ankle 1)"}
    # figure 1: time series of the three real logs
    fig, ax = plt.subplots(5, 3, figsize=(17, 12), sharex="col")
    for c, k in enumerate(LOGS):
        r, ph = raw[k], phs[k]; t = r["t"]
        ax[0, c].set_title(labels[k], fontsize=10)
        ax[0, c].plot(t, r["obs"][:, 3], label="gravity x"); ax[0, c].plot(t, r["obs"][:, 4], label="gravity y"); ax[0, c].set_ylabel("gravity vector [-]")
        ax[1, c].plot(t, ph["gyro"], color="C2"); ax[1, c].set_yscale("symlog", linthresh=0.01); ax[1, c].set_ylabel("|gyro| [rad/s]")
        ax[2, c].plot(t, r["pos"][:, 3], label="R knee"); ax[2, c].plot(t, r["pos"][:, 9], label="L knee"); ax[2, c].set_ylabel("knee angle [deg]")
        ax[3, c].plot(t, r["pos"][:, 4], label="R ankle_a measured"); ax[3, c].plot(t, r["tgt"][:, 4], "--", label="R ankle_a target")
        ax[3, c].plot(t, r["pos"][:, 10], label="L ankle_a measured"); ax[3, c].plot(t, r["tgt"][:, 10], "--", label="L ankle_a target"); ax[3, c].set_ylabel("ankle_a [deg]")
        ax[4, c].plot(t, np.abs(r["act"]).max(1), color="C3"); ax[4, c].axhline(3, color="k", lw=0.5, ls=":"); ax[4, c].set_ylabel("max |action| [-]"); ax[4, c].set_xlabel("time [s]")
        for te in ph["events"]:
            for row in range(5):
                ax[row, c].axvline(te, color="k", alpha=0.25, lw=0.7)
        if ph["t_div"] is not None:
            for row in range(5):
                ax[row, c].axvline(ph["t_div"], color="r", alpha=0.5, lw=1)
        for row in (0, 2, 3):
            ax[row, c].legend(fontsize=7, loc="best")
    fig.suptitle("Real standing logs (grey lines: detected disturbance start, red line: divergence)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{a.img_dir}/{day}_real_stand_timeseries.png", dpi=80); plt.close(fig)

    # figure 2: out-of-band fraction per channel group, per phase, against the training-like reference
    if "train" in refs:
        groups = [("gyro", range(0, 3)), ("gravity", range(3, 6)), ("q(t-1)", range(6, 18)), ("q(t)", range(18, 30)), ("last action", range(30, 42)), ("joint vel", range(45, 57))]
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
        for c, k in enumerate(LOGS):
            w = 0.2
            for pi, ph in enumerate(("settle", "idle", "after_push", "diverged")):
                if ph not in oob[k].get("train", {}):
                    continue
                f = np.array(oob[k]["train"][ph]["frac"])
                ax[c].bar(np.arange(len(groups)) + pi * w, [100 * f[list(g)].mean() for _, g in groups], w, label=f"{ph} (n={oob[k]['train'][ph]['n']})")
            ax[c].set_xticks(np.arange(len(groups)) + 1.5 * w); ax[c].set_xticklabels([g for g, _ in groups], fontsize=8)
            ax[c].set_title(labels[k], fontsize=10); ax[c].legend(fontsize=7)
        ax[0].set_ylabel("share of time outside sim P1-P99 [%]")
        fig.suptitle("Real observation vs training-like simulation band (gain 1.0, noise on, pushes), mean over channels of each group", fontsize=11)
        fig.tight_layout(); fig.savefig(f"{a.img_dir}/{day}_real_obs_out_of_band.png", dpi=80); plt.close(fig)

    # figure 3: grav x/y distribution real vs sim band, and idle tracking error real vs sim
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k in LOGS:
        m = phs[k]["masks"]["idle"]
        if m.sum():
            ax[0].plot(raw[k]["t"][m], raw[k]["obs"][m, 3], label=f"{k} idle gravity x")
    ax[0].axhspan(-0.0, 0.0, color="k"); ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("gravity x [-]"); ax[0].legend(fontsize=8)
    ax[0].set_title("Gravity x while the robot is quiet (sim stands at 0.00)", fontsize=10)
    conds = [c for c in stats if ":real" in c or ":ideal" in c]
    names, real_e, sim_e = [], [], []
    ax[1].set_title("Ankle_a tracking error (target - measured), quiet standing", fontsize=10)
    for k in LOGS:
        m = phs[k]["masks"]["idle"] if phs[k]["masks"]["idle"].sum() > 5 else phs[k]["masks"]["settle"]
        e = (raw[k]["tgt"] - raw[k]["pos"])[m].mean(0)
        ax[1].scatter([k + " real R"], [e[4]], c="C3", marker="o"); ax[1].scatter([k + " real L"], [e[10]], c="C3", marker="s")
    for c in stats:
        v = [np.array(r["idle_ankle_err"]) for r in stats[c] if r.get("idle_ankle_err") is not None and r["push"] == 0.0]
        if v and ":real" in c and main_cond(c):
            v = np.mean(v, 0); ax[1].scatter([c + " R"], [v[4]], c="C0", marker="o"); ax[1].scatter([c + " L"], [v[10]], c="C0", marker="s")
    ax[1].tick_params(axis="x", rotation=70, labelsize=7); ax[1].set_ylabel("ankle_a target - measured [deg]")
    fig.tight_layout(); fig.savefig(f"{a.img_dir}/{day}_idle_ankle_gravity.png", dpi=80); plt.close(fig)

    # figure 4: push response in sim per condition (fall rate vs push size) with the real result marked
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for c in sorted(stats):
        if ":real" not in c or not main_cond(c):
            continue
        ps = sorted({r["push"] for r in stats[c] if r["push"] > 0})
        fr = [100 * np.mean([r["fell"] for r in stats[c] if r["push"] == p]) for p in ps]
        ax[0].plot(ps, fr, marker="o", label=c)
        base = np.mean([r["fell"] for r in stats[c] if r["push"] == 0.0])
    ax[0].set_xlabel("push size, added base velocity [m/s]"); ax[0].set_ylabel("fell within 6 s [%]"); ax[0].legend(fontsize=6, ncol=2); ax[0].set_title("Simulation, real timing", fontsize=10)
    for c in sorted(stats):
        if ":real" not in c or not main_cond(c):
            continue
        ps = sorted({r["push"] for r in stats[c] if r["push"] > 0})
        v = [np.nanmedian([r.get("post_knee_std", np.nan) for r in stats[c] if r["push"] == p]) for p in ps]
        ax[1].plot(ps, v, marker="o", label=c)
    ax[1].set_xlabel("push size [m/s]"); ax[1].set_ylabel("knee angle std after the push [deg] (median)"); ax[1].set_title("Simulation: shaking after the push", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.img_dir}/{day}_sim_push_response.png", dpi=80); plt.close(fig)


def extra_figures(a, reals, raw, phs, stats, plt):
    day = a.date
    # latency per motor
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    x = np.arange(12)
    for k in LOGS:
        L = reals[k]["latency"]["joints"]
        ax[0].plot(x, [L[j]["p50"] for j in ORDER], marker="o", label=f"log {k} (hz={reals[k]['latency']['hz']:g})")
        ax[1].plot(x, [L[j]["p99"] for j in ORDER], marker="o", label=f"log {k}")
        ax[2].plot(x, [L[j]["late_pct"] + L[j]["lost_pct"] for j in ORDER], marker="o", label=f"log {k}")
    for i, t in enumerate(("Reply delay, median [ms]", "Reply delay, 99th percentile [ms]", "Reply late or lost [% of cycles]")):
        ax[i].set_title(t, fontsize=10); ax[i].set_xticks(x); ax[i].set_xticklabels(SHORT, rotation=70, fontsize=7); ax[i].legend(fontsize=8)
    fig.suptitle("Latency logs: delay grows with the motor order inside one leg (burst signature, seq would be flat ~0.7 ms)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.img_dir}/{day}_latency_per_motor.png", dpi=80); plt.close(fig)
    # push size calibration: sim peak gyro / gravity swing vs push, real events as horizontal lines
    tab = sim_table(stats)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for c, v in tab.items():
        if ":real@" not in c or not main_cond(c) or c.split(":")[0].split("/")[0] == c.split(":")[0].split("/")[1]:
            continue
        ps = [float(p) for p in v if float(p) > 0]
        ax[0].plot(ps, [v[str(p)]["gyro_peak"] for p in ps], marker="o", label=c)
        ax[1].plot(ps, [max(v[str(p)]["grav_x_swing"] or 0, v[str(p)]["grav_y_swing"] or 0) for p in ps], marker="o", label=c)
    for k in LOGS:
        for e in reals[k]["event_detail"]:
            ax[0].axhline(e["gyro_peak"], color="k", alpha=0.3, lw=0.8, ls="--"); ax[1].axhline(max(e["grav_x_swing"], e["grav_y_swing"]), color="k", alpha=0.3, lw=0.8, ls="--")
    ax[0].set_xlabel("simulated push [m/s]"); ax[0].set_ylabel("peak |gyro| in 1.5 s [rad/s]"); ax[0].set_title("Dashed: real disturbance events", fontsize=10); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("simulated push [m/s]"); ax[1].set_ylabel("gravity vector swing in 1.5 s [-]"); ax[1].set_title("Dashed: real disturbance events", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.img_dir}/{day}_push_size_calibration.png", dpi=80); plt.close(fig)


OLDER = [  # (file, body, ankle) from docs/sim2real/2026-10-13_huphy_archivedlogs_conditions.md; burst mode, before the IMU axis fix (bb558c6)
    ("00503ob2ms.csv", 0.05, 0.3), ("00503ob4ms.csv", 0.05, 0.3), ("00505ob2ms.csv", 0.05, 0.5), ("00505ob4ms.csv", 0.05, 0.5),
    ("005054msgscan(r)ob.csv", 0.05, 0.5), ("0051ob4ms.csv", 0.05, 1.0), ("010\ubc14\ub2e5\uc11c\uae30ob.csv", 0.1, 0.1),
]


def ladder(logs_dir):
    """One row per policy log (older 0.05-0.1 ladder + the three new ones): period, start tilt, quiet-time gyro, action size, knee excursion."""
    rows = []
    for f, b, an in OLDER + [(v["ob"], *v["name_scale"]) for v in LOGS.values()]:
        r = load_ob(os.path.join(logs_dir, f)); t = r["t"]; dt = np.diff(t) * 1000
        gn = np.linalg.norm(r["obs"][:, 0:3], axis=1)
        kd = np.abs(r["pos"][:, KNEE] - DEFAULT_DEG[KNEE]).max(1)
        rows.append(dict(file=f, body=b, ankle=an, duration_s=float(t[-1]), rows=len(t), period_mean_ms=float(dt.mean()), period_p95_ms=float(np.percentile(dt, 95)),
                         period_max_ms=float(dt.max()), grav_x_start=float(r["obs"][0, 3]), grav_x_median=float(np.median(r["obs"][:, 3])),
                         gyro_rms=float(np.sqrt((gn ** 2).mean())), gyro_p50=float(np.median(gn)), act_p99=float(np.percentile(np.abs(r["act"]), 99)),
                         act_frac_gt3=float((np.abs(r["act"]) > 3).mean()), knee_dev_max_deg=float(kd.max()),
                         knee_std_deg=float(np.mean([r["pos"][:, k].std() for k in KNEE]))))
    return rows


def conds(a):
    for k in LOGS:
        s, _, _ = real_summary(k, a.logs)
        print(json.dumps({kk: v for kk, v in s.items() if kk not in ("latency", "event_detail")}, indent=1)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--pkg", required=True); s.add_argument("--checkpoint", required=True); s.add_argument("--logs", required=True)
    s.add_argument("--task", default="Pygmalion-Velocity-Flat-P2")
    s.add_argument("--conds", required=True, help='comma list "body/ankle:mode[@log]"; mode = ideal | real | train; @log = take the per-joint stale rate (late+lost) of that latency log')
    s.add_argument("--pushes", default="0.15,0.3,0.5"); s.add_argument("--reps", type=int, default=2)
    s.add_argument("--t-push", type=float, default=5.0); s.add_argument("--t-after", type=float, default=6.0)
    s.add_argument("--decimation", type=int, default=0, help="0 = task default (4 -> 20 ms); 5 -> 25 ms like the real 23-27 ms")
    s.add_argument("--device", default="cpu"); s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--logs", required=True); c.add_argument("--sim", nargs="+", required=True)
    c.add_argument("--out-json", required=True); c.add_argument("--img-dir", required=True)
    c.add_argument("--train-cond", default="1.0/1.0:train"); c.add_argument("--date", default="2026-10-17")
    d = sub.add_parser("conds"); d.add_argument("--logs", required=True)
    a = ap.parse_args()
    {"sim": run_sim, "compare": run_compare, "conds": conds}[a.stage](a)
