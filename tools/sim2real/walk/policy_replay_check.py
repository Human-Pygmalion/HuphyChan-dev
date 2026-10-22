#!/usr/bin/env python3
"""Policy-side check of the real `generalized_walk` logs (no spectra, no timing statistics beyond the policy step).

Four jobs, all on the checkpoint `terrain10_2048_model_37400.pt` (actor only, deterministic mean):
  1 replay        feed every logged 44-input row to the actor and compare with the logged `act/` row.
                  Also checks the log contract: last_act[t+1]==act[t], q_prev[t+1]==q[t], tgt==clip(default+0.25*act).
  2 input sanity  per input group: real mean/std/min/max vs the checkpoint's own observation normalizer
                  (= the training distribution) and vs the training noise; quiet-standing-segment offsets.
  3 sensitivity   Jacobian d(action)/d(input) of the actor at the real rows, in target degrees, plus
                  single-bias experiments (gyro z, gravity tilt, one joint +1 deg) and noise amplification.
  4 counterfactual open-loop re-run of the actor on the real rows with the quiet-segment gyro-z bias, gravity
                  tilt and joint offsets removed (and, as an extra, with the joint-history step rescaled to 20 ms).

Open loop means: the logged inputs are replaced, the robot does not react. It shows what the NETWORK would
have commanded for the same sensor history, not what the closed loop would have done.

Run (see the doc for the checkpoint extraction):
  CUDA_VISIBLE_DEVICES="" mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/policy_replay_check.py \
      --ckpt <terrain10_2048_model_37400.pt> --logs Expr_Data --json docs/sim2real/data/<name>.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from rsl_rl.modules.normalization import EmpiricalNormalization
from torch import nn
from torch.func import jacrev, vmap

torch.set_grad_enabled(False)
RAD = 57.29577951308232
ORDER = [f"{s}_leg/{j}" for s in ("right", "left") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")]
SHORT = [("R" if i < 6 else "L") + "_" + n.split("/")[1] for i, n in enumerate(ORDER)]
DEFAULT_DEG = np.array([-10.0268, 0, 0, 20.0535, 8.2480, -8.4261, 10.0268, 0, 0, -20.0535, -8.2440, 8.4150])  # HUPHY generalized_walk.py
OBS_NAMES = (["obs/gyro/x", "obs/gyro/y", "obs/gyro/z", "obs/grav/x", "obs/grav/y", "obs/grav/z"]
             + [f"obs/q_prev/{j}" for j in ORDER] + [f"obs/q/{j}" for j in ORDER]
             + [f"obs/last_act/{j}" for j in ORDER] + ["obs/cmd/vx", "obs/cmd/wz"])
SHORT_OBS = (["gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z"] + [f"q_prev {s}" for s in SHORT]
             + [f"q {s}" for s in SHORT] + [f"last_act {s}" for s in SHORT] + ["cmd_vx", "cmd_wz"])
GROUPS = {"gyro": slice(0, 3), "gravity": slice(3, 6), "q_prev": slice(6, 18), "q": slice(18, 30),
          "last_act": slice(30, 42), "cmd": slice(42, 44)}
# training actor corruption, README of the deploy package: U(-a, a) -> std a/sqrt(3)
NOISE_A = {"gyro": 0.2, "gravity": 0.05, "q_prev": 0.01, "q": 0.01}
YAW, ROLL = (2, 8), (1, 7)       # hip_yaw / hip_roll indices of (right, left)
STAND_SCORE = 0.05               # quiet window: sum of std(q 12 joints, rad) + std(gyro 3) below this


# ---------------------------------------------------------------- actor
class Actor(nn.Module):
    """rsl_rl actor: EmpiricalNormalization (x-mean)/(std+1e-2) -> MLP 44-512-256-128-12 with ELU."""

    def __init__(self, sd):
        super().__init__()
        self.norm = EmpiricalNormalization(44)
        self.norm.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("obs_normalizer.")})
        self.mlp = nn.Sequential(nn.Linear(44, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))
        self.mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
        self.eval()

    def forward(self, x, normalize=True):
        return self.mlp(self.norm(x) if normalize else x)


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return Actor(ck["actor_state_dict"]), ck


def jacobian(actor64, X):
    """d action / d raw input, shape (n, 12, 44). float64, torch.func."""
    return vmap(jacrev(lambda x: actor64(x[None])[0]))(X)


# ---------------------------------------------------------------- logs
def load_logs(folder):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        with open(p, encoding="utf-8") as f:
            head = f.readline()
        d = pd.read_csv(p, comment="#")
        if "obs/gyro/x" not in d.columns:
            continue  # motor-latency log (t, joint, latency_ms, status): no policy data
        lim = {}
        if head.startswith("# limits"):
            for m in re.finditer(r"(\S+/\S+)=(-?[\d.]+):(-?[\d.]+)", head):
                lim[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        lo = np.array([lim[j][0] for j in ORDER]) if lim else None
        hi = np.array([lim[j][1] for j in ORDER]) if lim else None
        has_vy = "obs/cmd/vy" in d.columns
        X = d[[c for c in OBS_NAMES if c in d.columns]].values.astype(np.float64)
        if X.shape[1] != 44:
            raise SystemExit(f"{p}: {X.shape[1]} obs columns, expected 44 (+optional vy)")
        out.append(dict(name=os.path.basename(p), t=d["t"].values, X=X, A=d[[f"act/{j}" for j in ORDER]].values.astype(np.float64),
                        T=d[[f"tgt/{j}" for j in ORDER]].values.astype(np.float64), lo=lo, hi=hi, has_vy=has_vy,
                        vy=d["obs/cmd/vy"].values if has_vy else None))
    return out


def quiet_window(L, seconds=2.0):
    """Window with the smallest (std of 12 joint angles + std of 3 gyro axes)."""
    X, t = L["X"], L["t"]
    n = min(len(X), max(10, int(round(seconds / np.median(np.diff(t))))))
    best = None
    for i in range(0, len(X) - n + 1, max(1, n // 10)):
        w = X[i:i + n]
        s = w[:, 18:30].std(0).sum() + w[:, 0:3].std(0).sum()
        if best is None or s < best[0]:
            best = (s, i)
    s, i = best
    w = X[i:i + n]
    return dict(i0=i, n=n, t0=float(t[i]), score=float(s), gyro_mean=w[:, 0:3].mean(0), gyro_std=w[:, 0:3].std(0),
                g_mean=w[:, 3:6].mean(0), g_std=w[:, 3:6].std(0), q_mean=w[:, 18:30].mean(0), q_std=w[:, 18:30].std(0))


def tilt_deg(g):
    g = np.asarray(g)
    return dict(pitch=float(np.degrees(np.arcsin(np.clip(g[0], -1, 1)))), roll=float(np.degrees(np.arcsin(np.clip(g[1], -1, 1)))),
                total=float(np.degrees(np.arccos(np.clip(-g[2] / np.linalg.norm(g), -1, 1)))))


def rot_to(a, b):
    """Rotation matrix taking unit vector a to unit vector b (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def rot_axis(axis, deg):
    th, c, s = np.radians(deg), np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ---------------------------------------------------------------- helpers
def tdeg(a):
    """action -> target offset from default, degrees."""
    return a * 0.25 * RAD


def asym(a_deg):
    """mean over rows of target offsets -> dict R/L/A for hip_yaw and hip_roll. A=(R+L)/2: both hip axes are shared
    in sign between the legs (build_robot.py; training-normalizer means are R=-L), so a mirror-symmetric gait has A=0."""
    m = a_deg.mean(0)
    return dict(yaw_R=m[YAW[0]], yaw_L=m[YAW[1]], yaw_A=(m[YAW[0]] + m[YAW[1]]) / 2,
                roll_R=m[ROLL[0]], roll_L=m[ROLL[1]], roll_A=(m[ROLL[0]] + m[ROLL[1]]) / 2)


def run(actor, X, normalize=True):
    return actor(torch.as_tensor(X, dtype=torch.float32), normalize).double().numpy()


def fmt(v, p=3):
    return f"{v:.{p}f}"


# ---------------------------------------------------------------- 1 replay
def task1(actor, logs, rep):
    print("\n## 1 replay (float32 actor, normalizer then ELU MLP; logged values carry 6 significant digits)\n")
    print("| file | rows | contract | max abs diff (act units) | mean abs diff | max diff in target deg | no-normalizer max diff | shifted-row max diff | last_act[t+1]==act[t] | q_prev[t+1]==q[t] | tgt-clip max dev (deg) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rep["replay"] = {}
    for L in logs:
        X, A = L["X"], L["A"]
        P = run(actor, X)
        d = np.abs(P - A)
        d_no = np.abs(run(actor, X, normalize=False) - A).max()
        d_sh = np.abs(P[:-1] - A[1:]).max()
        la = (np.abs(X[1:, 30:42] - A[:-1]).max(1) < 1e-5).mean()
        qp = (np.abs(X[1:, 6:18] - X[:-1, 18:30]).max(1) < 1e-5).mean()
        tg = DEFAULT_DEG + tdeg(A)
        if L["lo"] is not None:
            tg = np.clip(tg, L["lo"], L["hi"])
        dev = np.abs(L["T"] - tg).max()
        clipped = (np.abs(L["T"] - (DEFAULT_DEG + tdeg(A))) > 1e-3).mean()
        c = "45 in (extra obs/cmd/vy column = older stand contract; first 44 shown)" if L["has_vy"] else "44 in"
        rep["replay"][L["name"]] = dict(rows=len(X), has_vy=L["has_vy"], max=float(d.max()), mean=float(d.mean()), max_target_deg=float(tdeg(d.max())),
                                        no_norm_max=float(d_no), shifted_max=float(d_sh), last_act_ok=float(la), q_prev_ok=float(qp),
                                        tgt_dev_deg=float(dev), tgt_clipped_frac=float(clipped))
        print(f"| {L['name']} | {len(X)} | {c} | {d.max():.2e} | {d.mean():.2e} | {tdeg(d.max()):.2e} | {d_no:.2f} | {d_sh:.2f} | {la:.3f} | {qp:.3f} | {dev:.3f} |")


# ---------------------------------------------------------------- 2 input sanity
def task2(logs, norm_mean, norm_std, rep):
    print("\n## 2 input sanity\n")
    nm, ns = norm_mean, norm_std
    q = {}
    print("### quiet window per file (2 s with the smallest joint+gyro motion)\n")
    print("| file | cmd | t0 (s) | motion score | gyro mean x,y,z (rad/s) | gyro std max (rad/s) | gyro rows exactly 0 | joint std max (deg) | gravity std max | gravity mean x,y,z | tilt pitch/roll/total (deg) | gravity first row | tilt first row (total, deg) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for L in logs:
        if L["has_vy"]:
            continue
        w = quiet_window(L)
        L["quiet"] = w
        cmd = L["X"][:, 42:44]
        L["cmd_zero"] = bool((np.abs(cmd).max() < 1e-9))
        tl = tilt_deg(w["g_mean"])
        tf = tilt_deg(L["X"][0, 3:6])
        q[L["name"]] = dict(t0=w["t0"], score=w["score"], gyro_mean=w["gyro_mean"].tolist(), gyro_std=w["gyro_std"].tolist(),
                            g_mean=w["g_mean"].tolist(), tilt=tl, q_mean_deg=(w["q_mean"] * RAD).tolist(), first_g=L["X"][0, 3:6].tolist(), first_tilt=tf,
                            cmd_zero=L["cmd_zero"])
        print(f"| {L['name']} | {'0,0' if L['cmd_zero'] else np.unique(cmd, axis=0)[0].tolist()} | {w['t0']:.1f} | {w['score']:.3f} | "
              f"{w['gyro_mean'][0]:+.4f}, {w['gyro_mean'][1]:+.4f}, {w['gyro_mean'][2]:+.4f} | {w['gyro_std'].max():.4f} | {(np.abs(L['X'][w['i0']:w['i0']+w['n'], 0:3]).max(1) == 0).mean():.2f} | {w['q_std'].max()*RAD:.3f} | {w['g_std'].max():.5f} | "
              f"{w['g_mean'][0]:+.4f}, {w['g_mean'][1]:+.4f}, {w['g_mean'][2]:+.4f} | {tl['pitch']:+.2f} / {tl['roll']:+.2f} / {tl['total']:.2f} | "
              f"{L['X'][0,3]:+.3f}, {L['X'][0,4]:+.3f}, {L['X'][0,5]:+.3f} | {tf['total']:.2f} |")
    rep["quiet"] = q
    stand = [L for L in logs if not L["has_vy"] and L["cmd_zero"] and L["quiet"]["score"] < STAND_SCORE]
    ref = dict(files=[L["name"] for L in stand],
               gyro=np.mean([L["quiet"]["gyro_mean"] for L in stand], 0), gyro_sd=np.std([L["quiet"]["gyro_mean"] for L in stand], 0),
               g=np.mean([L["quiet"]["g_mean"] for L in stand], 0), g_sd=np.std([L["quiet"]["g_mean"] for L in stand], 0),
               q=np.mean([L["quiet"]["q_mean"] for L in stand], 0), q_sd=np.std([L["quiet"]["q_mean"] for L in stand], 0))
    ref["tilt"] = tilt_deg(ref["g"])
    rep["quiet_ref"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ref.items()}
    print(f"\nQuiet-standing reference = mean over {len(stand)} files with command 0 and motion score < {STAND_SCORE}: {', '.join(ref['files'])}\n")
    print(f"- gyro bias mean (x,y,z) = {ref['gyro'][0]:+.5f}, {ref['gyro'][1]:+.5f}, {ref['gyro'][2]:+.5f} rad/s; spread between files (std) = {ref['gyro_sd'][0]:.5f}, {ref['gyro_sd'][1]:.5f}, {ref['gyro_sd'][2]:.5f}")
    print(f"- gravity mean = {ref['g'][0]:+.4f}, {ref['g'][1]:+.4f}, {ref['g'][2]:+.4f}; spread = {ref['g_sd'][0]:.4f}, {ref['g_sd'][1]:.4f}, {ref['g_sd'][2]:.5f}; implied tilt pitch {ref['tilt']['pitch']:+.2f} deg, roll {ref['tilt']['roll']:+.2f} deg, total {ref['tilt']['total']:.2f} deg")
    print("\n### joint offset obs/q at the quiet window (deg, = measured minus HUPHY default pose)\n")
    print("| joint | " + " | ".join(L["name"][:14] for L in stand) + " | mean | spread (std) | training reset+bias range (deg) |")
    print("|---|" + "---|" * (len(stand) + 3))
    for j in range(12):
        row = [L["quiet"]["q_mean"][j] * RAD for L in stand]
        print(f"| {SHORT[j]} | " + " | ".join(f"{v:+.2f}" for v in row) + f" | {ref['q'][j]*RAD:+.2f} | {ref['q_sd'][j]*RAD:.2f} | +-{(0.05+0.01)*RAD:.1f} |")

    print("\n### input groups, real vs training (normalizer mean/std = the training distribution of that input)\n")
    print("Training noise (README): gyro U(-0.2,0.2) rad/s, gravity U(-0.05,0.05), joint angle U(-0.01,0.01) rad, no noise on last_act/cmd.\n")
    rep["groups"] = {}
    for regime in ("stand", "moving"):
        sel = [L for L in logs if not L["has_vy"] and (L["cmd_zero"] == (regime == "stand"))]
        X = np.vstack([L["X"] for L in sel])
        print(f"#### regime {regime}: {len(sel)} files, {len(X)} rows (all rows of the files, hand-held)\n")
        print("| input | real mean | real std | real min | real max | train mean | train std | z of real mean | rows beyond train mean+-3 std |")
        print("|---|---|---|---|---|---|---|---|---|")
        g = {}
        for i, name in enumerate(SHORT_OBS):
            x = X[:, i]
            z = (x.mean() - nm[i]) / (ns[i] + 1e-2)
            out = (np.abs(x - nm[i]) > 3 * ns[i]).mean()
            g[name] = dict(mean=x.mean(), std=x.std(), min=x.min(), max=x.max(), train_mean=nm[i], train_std=ns[i], z=z, out3=out)
            flag = " **" if abs(z) > 2 or out > 0.05 else ""
            print(f"| {name}{flag} | {x.mean():+.4f} | {x.std():.4f} | {x.min():+.4f} | {x.max():+.4f} | {nm[i]:+.4f} | {ns[i]:.4f} | {z:+.2f} | {out:.3f} |")
        rep["groups"][regime] = {k: {a: float(b) for a, b in v.items()} for k, v in g.items()}
        print()
    return ref


# ---------------------------------------------------------------- 3 sensitivity
def task3(actor, actor64, logs, rep):
    print("\n## 3 sensitivity (Jacobian of the actor at the real rows)\n")
    stdv = {"gyro": 0.2 / 3 ** 0.5, "gravity": 0.05 / 3 ** 0.5, "q_prev": 0.01 / 3 ** 0.5, "q": 0.01 / 3 ** 0.5}
    out = {}
    for regime in ("stand", "moving"):
        sel = [L for L in logs if not L["has_vy"] and (L["cmd_zero"] == (regime == "stand"))]
        X = np.vstack([L["X"] for L in sel])
        J = jacobian(actor64, torch.as_tensor(X)).numpy()      # (n,12,44) action per raw input unit
        Jm = J.mean(0)
        # finite-difference self-check of the Jacobian on 3 rows
        for r in (0, len(X) // 2, len(X) - 1):
            for i in (2, 20, 35):
                e = np.zeros(44); e[i] = 1e-6
                fd = (actor64(torch.as_tensor(X[r] + e)[None]) - actor64(torch.as_tensor(X[r] - e)[None]))[0].numpy() / 2e-6
                assert np.abs(fd - J[r, :, i]).max() < 1e-5 * (1 + np.abs(fd).max()), "jacobian self-check failed"
        print(f"### regime {regime}: {len(sel)} files, {len(X)} rows. Jacobian self-check against finite differences passed.\n")
        print("Target response in degrees: gyro per 1 rad/s, gravity per 1 rad of tilt, joint angles per 1 rad (= deg per deg), last_act per 1 action unit, cmd per 1 unit. Input-block norm = Frobenius norm of the 12 x n block of 0.25*57.3*J.\n")
        print("| group | inputs | block norm (target deg per input unit) | mean-J block norm | training noise std | target-deg rms (over 12 joints) caused by training noise on this group |")
        print("|---|---|---|---|---|---|")
        gres = {}
        for gname, sl in GROUPS.items():
            B = J[:, :, sl] * 0.25 * RAD                        # per row
            fro = np.sqrt((B ** 2).sum((1, 2))).mean()
            fro_m = np.sqrt(((Jm[:, sl] * 0.25 * RAD) ** 2).sum())
            if gname in stdv:
                n_rms = np.sqrt(((B * stdv[gname]) ** 2).sum((1, 2)) / 12).mean()
                ns_s = f"{stdv[gname]:.4f}"
                nr = f"{n_rms:.3f}"
            else:
                ns_s, nr, n_rms = "-", "-", None
            gres[gname] = dict(fro=float(fro), fro_mean_J=float(fro_m), noise_std=stdv.get(gname), noise_rms_deg=None if n_rms is None else float(n_rms))
            print(f"| {gname} | {sl.stop-sl.start} | {fro:.1f} | {fro_m:.1f} | {ns_s} | {nr} |")
        # per-input amplification
        colnorm = np.sqrt(((J * 0.25 * RAD) ** 2).sum(1)).mean(0)      # (44,) target-deg per input unit
        unit = np.array([0.1] * 3 + [np.radians(1)] * 3 + [np.radians(1)] * 24 + [1.0] * 14)[:44]
        unit[30:42] = 0.1
        unit[42:] = 0.1
        amp = colnorm * unit
        order = np.argsort(-amp)
        print("\nTop 12 inputs by target-deg response (12-joint 2-norm) to a typical small disturbance: gyro 0.1 rad/s, gravity/joint 1 deg, last_act 0.1, cmd 0.1:\n")
        print("| rank | input | norm of column (deg per input unit) | disturbance | target-deg 2-norm |")
        print("|---|---|---|---|---|")
        for k, i in enumerate(order[:12]):
            u = "0.1 rad/s" if i < 3 else ("1 deg" if i < 30 else "0.1")
            print(f"| {k+1} | {SHORT_OBS[i]} | {colnorm[i]:.1f} | {u} | {amp[i]:.3f} |")
        # joint DC / Nyquist gain
        print("\nJoint-angle channel: the network sees q(t) and q(t-1). A constant offset enters both (DC gain = |J_q + J_qprev|). A sign-alternating jitter enters with opposite sign (Nyquist gain = |J_q - J_qprev|, i.e. the network's finite-difference velocity cue). Numbers are target deg per 1 deg of joint-angle disturbance, 2-norm over the 12 target joints.\n")
        print("| joint | q column | q_prev column | DC (q+q_prev) | Nyquist (q-q_prev) | Nyquist/DC |")
        print("|---|---|---|---|---|---|")
        dcs, nys = [], []
        for j in range(12):
            a, b = J[:, :, 18 + j] * 0.25, J[:, :, 6 + j] * 0.25
            nq, npq = np.sqrt((a ** 2).sum(1)).mean(), np.sqrt((b ** 2).sum(1)).mean()
            dc, ny = np.sqrt(((a + b) ** 2).sum(1)).mean(), np.sqrt(((a - b) ** 2).sum(1)).mean()
            dcs.append(dc); nys.append(ny)
            print(f"| {SHORT[j]} | {nq:.2f} | {npq:.2f} | {dc:.2f} | {ny:.2f} | {ny/max(dc,1e-9):.1f} |")
        # last_act feedback
        Jla = J[:, :, 30:42]
        rho = np.array([np.abs(np.linalg.eigvals(m)).max() for m in Jla])
        print(f"\nlast_act feedback (a(t) = J_la a(t-1), network alone, no plant): spectral radius median {np.median(rho):.2f}, share of rows above 1: {(rho > 1).mean():.3f}, p95 {np.percentile(rho,95):.2f}, max {rho.max():.2f}\n")
        out[regime] = dict(groups=gres, colnorm=colnorm.tolist(), dc=dcs, nyq=nys, rho_gt1=float((rho > 1).mean()), rho_med=float(np.median(rho)), rho_p95=float(np.percentile(rho, 95)),
                           rho_max=float(rho.max()), top=[(SHORT_OBS[i], float(colnorm[i]), float(amp[i])) for i in order[:12]],
                           J_mean=Jm.tolist())
        out[regime]["X"] = X
    rep["jacobian"] = {k: {a: b for a, b in v.items() if a != "X"} for k, v in out.items()}

    # single-bias experiments (non-linear: re-run the actor), mean over rows
    print("### single-bias experiments (actor re-run, mean over rows, target degrees; A = (R+L)/2 asymmetric part)\n")
    print("A mirror-symmetric gait has A=0 for hip_yaw and hip_roll. A bias that moves A away from 0 pushes both hips the same way, which is the veering signature.\n")
    exps = []
    for b in (0.01, 0.05):
        exps.append((f"gyro_z +{b} rad/s", lambda X, b=b: _add(X, [2], b)))
        exps.append((f"gyro_z -{b} rad/s", lambda X, b=b: _add(X, [2], -b)))
    for ax, nm_ in (("y", "pitch tilt (gravity x)"), ("x", "roll tilt (gravity y)")):
        for dg in (1, 2, -1, -2):
            exps.append((f"gravity {nm_} {dg:+d} deg", lambda X, ax=ax, dg=dg: _rotg(X, rot_axis(ax, dg))))
    for j in range(12):
        exps.append((f"joint offset {SHORT[j]} +1 deg", lambda X, j=j: _add(X, [6 + j, 18 + j], np.radians(1))))
    res = {}
    for regime in ("stand", "moving"):
        X = out[regime]["X"]
        base = tdeg(run(actor, X))
        print(f"#### regime {regime}\n")
        print("| bias | hip_yaw R | hip_yaw L | A_yaw | hip_roll R | hip_roll L | A_roll | hip_pitch R | hip_pitch L | largest |change| joint |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        rr = {}
        for name, fn in exps:
            d = tdeg(run(actor, fn(X))) - base
            m = d.mean(0)
            a = asym(d)
            k = int(np.argmax(np.abs(m)))
            rr[name] = dict(mean_change_deg=m.tolist(), **{k_: float(v) for k_, v in a.items()})
            print(f"| {name} | {m[2]:+.3f} | {m[8]:+.3f} | {a['yaw_A']:+.3f} | {m[1]:+.3f} | {m[7]:+.3f} | {a['roll_A']:+.3f} | {m[0]:+.3f} | {m[6]:+.3f} | {SHORT[k]} {m[k]:+.2f} |")
        res[regime] = rr
        print()
    rep["bias_experiments"] = res


def _add(X, cols, v):
    X = X.copy(); X[:, cols] += v
    return X


def _rotg(X, R):
    X = X.copy(); X[:, 3:6] = X[:, 3:6] @ R.T
    return X


# ---------------------------------------------------------------- 4 counterfactual
def task4(actor, logs, ref, rep):
    print("\n## 4 counterfactual replay (open loop: actor re-run on the logged sensor rows, the robot does not react)\n")
    gb = np.array([0, 0, ref["gyro"][2]])
    R = rot_to(np.asarray(ref["g"]), np.array([0, 0, -1.0]))
    qoff = np.asarray(ref["q"])
    print(f"Removed quantities (global quiet-standing reference): gyro z bias {gb[2]:+.5f} rad/s; gravity rotation of {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))):.2f} deg (mean g {np.round(ref['g'],4).tolist()} -> [0,0,-1]); joint offsets (deg) " + ", ".join(f"{s} {v*RAD:+.2f}" for s, v in zip(SHORT, qoff)) + ".\n")

    def variants(L):
        X = L["X"]; out = {"baseline": X}
        a = X.copy(); a[:, 0:3] -= gb; out["a gyro z"] = a
        b = X.copy(); b[:, 3:6] = b[:, 3:6] @ R.T; out["b gravity"] = b
        c = X.copy(); c[:, 6:18] -= qoff; c[:, 18:30] -= qoff; out["c joints"] = c
        d = X.copy(); d[:, 0:3] -= gb; d[:, 3:6] = d[:, 3:6] @ R.T; d[:, 6:18] -= qoff; d[:, 18:30] -= qoff; out["d a+b+c"] = d
        dt = np.diff(L["t"], prepend=L["t"][0] - np.median(np.diff(L["t"])))
        e = X.copy(); e[:, 6:18] = X[:, 18:30] - (X[:, 18:30] - X[:, 6:18]) * (0.02 / dt)[:, None]; out["e dt 20 ms"] = e
        return out

    names = ["baseline", "a gyro z", "b gravity", "c joints", "d a+b+c", "e dt 20 ms"]
    rep["counterfactual"] = {}
    for key, lab in (("yaw_A", "hip_yaw"), ("roll_A", "hip_roll")):
        print(f"### {lab}: mean target offset asymmetry A=(R+L)/2 in degrees, per file (baseline = replay of the logged rows = logged action)\n")
        print("| file | cmd | policy dt median (ms) | " + " | ".join(names) + " |")
        print("|---|---|---|" + "---|" * len(names))
        for L in logs:
            if L["has_vy"]:
                continue
            V = variants(L)
            rr = {n: asym(tdeg(run(actor, V[n]))) for n in names}
            rep["counterfactual"].setdefault(L["name"], {n: {k: float(v) for k, v in rr[n].items()} for n in names})
            cmd = L["X"][0, 42:44].tolist()
            print(f"| {L['name']} | {cmd} | {np.median(np.diff(L['t']))*1000:.0f} | " + " | ".join(f"{rr[n][key]:+.3f}" for n in names) + " |")
        print()
    # (f) ideal stand: q=q_prev=0 (exactly the default pose), g=(0,0,-1), gyro 0, cmd 0, last_act iterated to a fixed point
    x = np.zeros((1, 44)); x[0, 5] = -1.0
    for _ in range(300):
        a_f = run(actor, x)
        x[0, 30:42] = a_f[0]
    af = tdeg(a_f)
    ideal = asym(af)
    rep["ideal_stand"] = dict(target_deg=af[0].tolist(), last_act_converged=bool(np.abs(run(actor, x) - a_f).max() < 1e-6), **{k: float(v) for k, v in ideal.items()})
    print(f"\n### (f) ideal-stand input (all joints at default, gravity [0,0,-1], gyro 0, command 0, last_act iterated to a fixed point; converged={rep['ideal_stand']['last_act_converged']})\n")
    print("Target offset from default pose (deg): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(SHORT, af[0])))
    print(f"hip_yaw R {ideal['yaw_R']:+.3f} L {ideal['yaw_L']:+.3f} A {ideal['yaw_A']:+.3f}; hip_roll R {ideal['roll_R']:+.3f} L {ideal['roll_L']:+.3f} A {ideal['roll_A']:+.3f}")
    # per-leg means for baseline and (d) on pooled rows, plus mean |action change| from each variant
    print("### pooled effect of each removal on the whole action vector (all 44-input files, row-weighted)\n")
    print("| variant | mean hip_yaw R | mean hip_yaw L | A_yaw | mean hip_roll R | mean hip_roll L | A_roll | mean abs change of all 12 targets vs baseline (deg) |")
    print("|---|---|---|---|---|---|---|---|")
    pooled = {n: [] for n in names}
    for L in logs:
        if L["has_vy"]:
            continue
        for n, X in variants(L).items():
            pooled[n].append(tdeg(run(actor, X)))
    base = np.vstack(pooled["baseline"])
    rep["counterfactual_pooled"] = {}
    for n in names:
        P = np.vstack(pooled[n]); a = asym(P)
        ch = np.abs(P - base).mean()
        rep["counterfactual_pooled"][n] = dict(**{k: float(v) for k, v in a.items()}, mean_abs_change_deg=float(ch))
        print(f"| {n} | {a['yaw_R']:+.3f} | {a['yaw_L']:+.3f} | {a['yaw_A']:+.3f} | {a['roll_R']:+.3f} | {a['roll_L']:+.3f} | {a['roll_A']:+.3f} | {ch:.3f} |")


def plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    stand = rep["quiet_ref"]["files"]
    for k, n in enumerate(stand):
        ax[0].scatter(range(12), rep["quiet"][n]["q_mean_deg"], s=22, alpha=0.7, label=re.sub(r"[^\x00-\x7f]", "", n).replace(".csv", "").strip()[:24])
    ax[0].plot(range(12), np.array(rep["quiet_ref"]["q"]) * RAD, "k_", ms=18, mew=2, label="mean")
    ax[0].axhspan(-3.4, 3.4, color="g", alpha=0.1, label="training reset+bias range")
    ax[0].set_xticks(range(12)); ax[0].set_xticklabels(SHORT, rotation=70, fontsize=7)
    ax[0].set_ylabel("obs/q offset from default at quiet standing (deg)"); ax[0].set_title("Joint offsets in the quietest 2 s"); ax[0].legend(fontsize=6)
    j = rep["jacobian"]["moving"]; js = rep["jacobian"]["stand"]
    w = 0.2
    for k, (v, lab) in enumerate(((js["dc"], "DC stand"), (js["nyq"], "Nyquist stand"), (j["dc"], "DC moving"), (j["nyq"], "Nyquist moving"))):
        ax[1].bar(np.arange(12) + (k - 1.5) * w, v, w, label=lab)
    ax[1].set_xticks(range(12)); ax[1].set_xticklabels(SHORT, rotation=70, fontsize=7)
    ax[1].set_ylabel("target deg per 1 deg input (2-norm, 12 targets)"); ax[1].set_title("Joint-angle input gain: constant offset vs alternating jitter"); ax[1].legend(fontsize=7)
    names = [n for n in rep["counterfactual"]]
    x = np.arange(len(names))
    for k, v in enumerate(("baseline", "c joints", "d a+b+c")):
        ax[2].bar(x + (k - 1) * 0.27, [rep["counterfactual"][n][v]["yaw_A"] for n in names], 0.27, label=v)
    ax[2].axhline(rep["ideal_stand"]["yaw_A"], color="k", ls="--", label="network at ideal default-pose input")
    ax[2].set_xticks(x); ax[2].set_xticklabels([re.sub(r"[^\x00-\x7f]", "", n).replace(".csv", "").strip()[:28] for n in names], rotation=75, fontsize=6)
    ax[2].set_ylabel("mean hip_yaw asymmetry A=(R+L)/2 of target (deg)"); ax[2].set_title("Open-loop counterfactual"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.27); fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--logs", default="Expr_Data")
    ap.add_argument("--json", default="")
    ap.add_argument("--png", default="")
    a = ap.parse_args()
    actor, ck = load_actor(a.ckpt)
    import copy
    actor64 = copy.deepcopy(actor).double()
    nm = actor.norm.mean.numpy().astype(np.float64)
    ns = actor.norm.std.numpy().astype(np.float64)
    logs = load_logs(a.logs)
    rep = dict(ckpt=a.ckpt, iter=int(ck["iter"]), norm_count=int(actor.norm.count), norm_mean=nm.tolist(), norm_std=ns.tolist())
    print(f"# policy replay check: checkpoint iter {ck['iter']}, normalizer count {int(actor.norm.count)}, {len(logs)} policy logs")
    task1(actor, logs, rep)
    ref = task2(logs, nm, ns, rep)
    task3(actor, actor64, [L for L in logs], rep)
    task4(actor, logs, ref, rep)
    if a.png:
        plot(rep, a.png)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


if __name__ == "__main__":
    main()
