#!/usr/bin/env python3
"""Heading-drift probe: why does a policy turn while it is commanded to walk straight?

Reads the walking rollouts (gait_kinematics_probe.py, 1.2 m/s forward, wz = 0, 17 s, 3 seeds) and the
standing rollouts (measure_stand.py) of several policies and tabulates, with NO interpretation:

  1. body yaw and path heading against time, yaw rate, rate per 3 s window, sign per seed
  2. left/right asymmetry per leg: joint angle, torque, target-minus-actual error (mean and 99th
     percentile of |x|), derived ankle torque, step time, stance time, step length, foot placement,
     ground reaction share
  3. policy output asymmetry: mean (qtarget - default) per joint
  4. what the probe commanded (yaw-rate command only; heading mode is switched off in the probe)
  5. standing: yaw change at wz = 0

Why the yaw is rebuilt from velocities: the walking rollouts record base_x, base_y (world) and the body
frame velocity base_vx, base_vy, but not the base quaternion. The world velocity is d(base_xy)/dt and the
body-frame velocity is R(psi)^T v_world, so psi = angle(v_world) - angle(v_body). The estimate is averaged
over a window (default 1.0 s, about one stride) as angle(sum v_world * conj(v_body)). It is checked against
the true quaternion yaw of the v5_walkstop files (`--validate`).

Sign conventions (printed in the tables):
  * yaw psi: counter-clockwise seen from above is positive = left turn.
  * Joint values are "mirrored": left value = physical angle about the joint axis, right value is
    multiplied by -1 for rotations about x or z so that a perfectly left/right symmetric gait gives
    equal left and right numbers. Display signs: hip yaw + = toe out, hip roll + = abduction,
    hip/ankle pitch + = flexion/dorsiflexion (thigh forward, toes up), knee + = flexion.
  * Mean values use an integer number of left strides (first to last left touchdown after 2 s), so a
    partial gait cycle does not bias one leg.

Run (CPU only):
  cd mujoco-sim/mjlab   # analysis/out lives there
  CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python3 ../../tools/robot_gains/heading_drift_probe.py \
      --md-out /tmp/heading_tables.md --json-out ../../tools/robot_gains/results/heading_drift.json \
      --plot ../../docs/img/pd_gain/2026-10-16_heading_drift_probe.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "mujoco-sim/mjlab/analysis/out"
DT = 0.02
DEG = 180.0 / np.pi
CONTACT_N = 5.0
WARM_S = 2.0
YAW_WIN = 51  # 1.02 s smoothing for the slow yaw track
YAW_WIN_SHORT = 11  # 0.22 s for the per-phase yaw increments

POLICIES = {  # label -> tag stem (files <stem>_walk_s{seed}_vx1.2.npz, <stem>_stand_s{seed}.npz)
    "baseline": "m_rbmv11a_final",
    "bundle_trial": "m_trialv11a_final",
    "armA": "m_gcrank60_final",
}
KO = {"baseline": "기준선", "bundle_trial": "묶음 시험", "armA": "팔 A"}
WALK_SEEDS = (11, 12, 13)
STAND_SEEDS = (0, 1, 2, 3)
META = OUT / "m_gcrank60_final_walk_s11_meta.json"
MJB = OUT / "stand10_v9caps_m8998_s1_model.mjb"
ENVELOPE = REPO / "pygmalion_locomotion/assets/pygmalion_v2/ankle_rp_envelope.json"

# channel -> (joint name stem, display sign applied after mirroring)
CHANNELS = {
    "hip_yaw": ("hip_yaw_joint", +1),
    "hip_roll": ("hip_roll_joint", +1),
    "hip_pitch": ("hip_pitch_joint", -1),
    "knee": ("knee_joint", +1),
    "crank_A": ("crank_A_joint", +1),
    "crank_B": ("crank_B_joint", +1),
}
KO_CH = {
    "hip_yaw": "엉덩이 비틀기",
    "hip_roll": "엉덩이 좌우",
    "hip_pitch": "엉덩이 앞뒤",
    "knee": "무릎",
    "crank_A": "크랭크 A",
    "crank_B": "크랭크 B",
    "ankle_pitch": "발목 앞뒤",
    "ankle_roll": "발목 좌우",
}


# ----------------------------------------------------------------------------- helpers
def load_meta():
    return json.load(open(META))


def mirror_factor(meta, joint_stem: str, side: str) -> float:
    """sign that turns the raw joint coordinate into the mirrored physical angle."""
    ax = np.asarray(meta["joints"][f"{side}_{joint_stem}"]["axis"])
    k = int(np.argmax(np.abs(ax)))
    s = float(np.sign(ax[k]))
    if k == 1:  # about y: symmetric motion has equal physical angles
        return s
    return s * (1.0 if side == "L" else -1.0)  # about x or z: symmetric motion has opposite angles


def side_is_left(s):
    """Jc^T (analysis/measure_loads.py tauank_eq_*) returns ankle torque in a mirrored convention: a
    symmetric standing pose gives equal magnitude and OPPOSITE sign on the two legs, for pitch and roll
    (checked on the standing rollouts, table 20). Mirroring therefore flips the right leg."""
    return s == "L"


def rel_pct(l, r):
    den = 0.5 * (abs(l) + abs(r))
    return float("nan") if den < 1e-9 else 100.0 * (l - r) / den


def smooth_c(z, win):
    return np.convolve(z, np.ones(win) / win, mode="same")


def yaw_track(x, y, vbx, vby, win):
    """psi(t) [rad, unwrapped], phi(t) path heading, beta(t) body-frame travel direction, valid mask."""
    vw = np.gradient(x, DT) + 1j * np.gradient(y, DT)
    vb = vbx + 1j * vby
    zs = smooth_c(vw * np.conj(vb), win)
    psi = np.unwrap(np.angle(zs))
    phi = np.unwrap(np.angle(smooth_c(vw, win)))
    beta = np.angle(smooth_c(vb, win))
    n = len(x)
    ok = np.zeros(n, bool)
    ok[win // 2 : n - win // 2] = True
    return psi, phi, beta, ok


def verdict_drift(d, warm=WARM_S):
    """Copy of analysis/walking_verdict.py heading drift (absolute) plus the signed value."""
    t = d["time"]
    m = t >= t[0] + warm
    x, y = d["base_x"][m], d["base_y"][m]
    n = max(3, len(x) // 10)
    h0 = np.arctan2(y[n] - y[0], x[n] - x[0])
    h1 = np.arctan2(y[-1] - y[-n], x[-1] - x[-n])
    s = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
    return float(s * DEG)


def debounce(c, n=3):
    """Fill contact gaps of at most n samples, then drop contact runs of at most n samples (force chatter)."""
    c = c.astype(bool).copy()
    for target in (False, True):  # first fill short gaps, then delete short contacts
        idx = np.flatnonzero(np.diff(np.r_[0, (c == target).astype(int), 0]))
        for st, en in zip(idx[::2], idx[1::2]):
            if en - st <= n and st > 0 and en < len(c):
                c[st:en] = not target
    return c


def edges(contact):
    c = contact.astype(int)
    d = np.diff(c)
    return np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1  # touchdown, liftoff indices


def fk_feet(mjm, d):
    """Foot position and yaw in the pelvis frame (base at origin, identity attitude) for each frame."""
    import mujoco

    md = mujoco.MjData(mjm)
    adr = {}
    for j in range(mjm.njnt):
        nm = (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or "").split("/")[-1]
        adr[nm] = int(mjm.jnt_qposadr[j])
    bid = {s: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"robot/{s}_foot_link") for s in "LR"}
    n = len(d["time"])
    out = {s: np.zeros((n, 4)) for s in "LR"}  # x, y, z, yaw
    for i in range(n):
        md.qpos[:] = 0.0
        md.qpos[3] = 1.0
        for nm, a in adr.items():
            if nm == "root":
                continue
            k = f"q_{nm}"
            if k in d:
                md.qpos[a] = d[k][i]
        mujoco.mj_kinematics(mjm, md)
        for s in "LR":
            p = md.xpos[bid[s]]
            R = md.xmat[bid[s]].reshape(3, 3)
            out[s][i] = (p[0], p[1], p[2], np.arctan2(R[1, 0], R[0, 0]))
    return out


# ----------------------------------------------------------------------------- one walk file
def analyse_walk(path, meta, mjm, jcT):
    d = np.load(path)
    t = d["time"]
    res = {}

    # ---- 1. yaw track
    psi, phi, beta, ok = yaw_track(d["base_x"], d["base_y"], d["base_vx"], d["base_vy"], YAW_WIN)
    sel = ok & (t >= WARM_S)
    i0 = int(np.argmax(sel))
    psi_rel = (psi - psi[i0]) * DEG
    phi_rel = (phi - phi[i0]) * DEG
    tt = t[sel]
    p = np.polyfit(tt, psi_rel[sel], 1)
    fit = np.polyval(p, tt)
    ss_res = float(np.sum((psi_rel[sel] - fit) ** 2))
    ss_tot = float(np.sum((psi_rel[sel] - psi_rel[sel].mean()) ** 2))
    res["yaw_rate_deg_s"] = float(p[0])
    res["yaw_linear_r2"] = 1.0 - ss_res / max(ss_tot, 1e-12)
    tmid = 0.5 * (tt[0] + tt[-1])
    h1, h2 = tt <= tmid, tt > tmid
    res["yaw_rate_first_half"] = float(np.polyfit(tt[h1], psi_rel[sel][h1], 1)[0])
    res["yaw_rate_second_half"] = float(np.polyfit(tt[h2], psi_rel[sel][h2], 1)[0])
    wins = []
    for a, b in ((2, 5), (5, 8), (8, 11), (11, 14), (14, 16.5)):
        w = sel & (t >= a) & (t < b)
        wins.append(float(np.polyfit(t[w], psi_rel[w], 1)[0]))
    res["yaw_rate_windows"] = wins
    res["yaw_total_deg"] = float(psi_rel[sel][-1])
    res["yaw_at_s"] = {str(s): float(np.interp(s, tt, psi_rel[sel])) for s in (5, 8, 11, 14, 16)}
    res["path_heading_change_deg"] = float(phi_rel[sel][-1])
    res["verdict_drift_signed_deg"] = verdict_drift(d)
    res["beta_mean_deg"] = float(np.degrees(np.angle(np.mean(np.exp(1j * beta[sel])))))
    res["vx_mean"] = float(d["base_vx"][t >= WARM_S].mean())
    res["vy_mean"] = float(d["base_vy"][t >= WARM_S].mean())
    res["cmd_vx"] = float(np.unique(d["cmd_vx"])[0])
    res["_t"], res["_psi_rel"] = tt, psi_rel[sel]

    # ---- gait events
    cL, cR = debounce(d["Fz_L"] > CONTACT_N), debounce(d["Fz_R"] > CONTACT_N)
    tdL, loL = edges(cL)
    tdR, loR = edges(cR)
    tdL = tdL[t[tdL] >= WARM_S]
    tdR = tdR[t[tdR] >= WARM_S]
    if len(tdL) < 4:
        raise RuntimeError(f"{path}: fewer than 4 left touchdowns")
    a, b = int(tdL[0]), int(tdL[-1])
    W = slice(a, b)  # integer number of left strides
    res["n_left_strides"] = int(len(tdL) - 1)
    res["window_s"] = float(t[b] - t[a])

    def stance_and_swing(td, lo):
        st, sw = [], []
        for k, i in enumerate(td):
            nxt = lo[lo > i]
            if len(nxt) and t[nxt[0]] <= t[-1]:
                st.append(t[nxt[0]] - t[i])
            nt = td[td > i]
            if len(nxt) and len(nt) and nt[0] > nxt[0]:
                sw.append(t[nt[0]] - t[nxt[0]])
        return np.array(st), np.array(sw)

    gait = {}
    for s, td, lo in (("L", tdL, loL), ("R", tdR, loR)):
        st, sw = stance_and_swing(td, lo)
        gait[s] = {
            "stride_time": float(np.diff(t[td]).mean()),
            "stance_time": float(st.mean()),
            "swing_time": float(sw.mean()),
            "n_td": int(len(td)),
        }
    # steps: alternate touchdown events
    ev = sorted([(i, "L") for i in tdL] + [(i, "R") for i in tdR])
    stp = {"L": [], "R": []}
    for (i0_, s0), (i1_, s1) in zip(ev[:-1], ev[1:]):
        if s0 == s1:
            continue
        dist = float(np.hypot(d["base_x"][i1_] - d["base_x"][i0_], d["base_y"][i1_] - d["base_y"][i0_]))
        stp[s1].append((t[i1_] - t[i0_], dist))  # step ending with a touchdown of foot s1
    for s in "LR":
        arr = np.array(stp[s])
        gait[s]["step_time"] = float(arr[:, 0].mean())
        gait[s]["step_len_base"] = float(arr[:, 1].mean())
    # single / double support shares and a foot-order check
    gait["both_stance_pct"] = float((cL & cR)[W].mean() * 100)
    gait["flight_pct"] = float((~cL & ~cR)[W].mean() * 100)
    gait["L_only_pct"] = float((cL & ~cR)[W].mean() * 100)
    gait["R_only_pct"] = float((~cL & cR)[W].mean() * 100)
    res["gait"] = gait

    # ---- ground reaction (vertical, summed contact normal force)
    BW = meta["BW_N"]
    FL, FR = d["Fz_L"][W] / BW, d["Fz_R"][W] / BW
    res["grf"] = {
        "mean_L": float(FL.mean()),
        "mean_R": float(FR.mean()),
        "p99_L": float(np.percentile(FL, 99)),
        "p99_R": float(np.percentile(FR, 99)),
        "impulse_share_L_pct": float(FL.sum() / (FL.sum() + FR.sum()) * 100),
        "sensor_p99_L": float(np.percentile(d["sensorF_L"][W] / BW, 99)),
        "sensor_p99_R": float(np.percentile(d["sensorF_R"][W] / BW, 99)),
    }

    # ---- per-joint channels (mirrored)
    chans = {}
    dflt = meta["default_joint_pos_deg"]
    for ch, (stem, disp) in CHANNELS.items():
        row = {}
        for s in "LR":
            f = mirror_factor(meta, stem, s) * disp
            q = d[f"q_{s}_{stem}"][W] * f * DEG
            tau = d[f"tau_{s}_{stem}"][W] * f
            err = (d[f"qtarget_{s}_{stem}"][W] - d[f"q_{s}_{stem}"][W]) * f * DEG
            act = (d[f"qtarget_{s}_{stem}"][W] * DEG - dflt[f"{s}_{stem}"]) * f
            row[s] = {
                "q_mean": float(q.mean()),
                "q_p99": float(np.percentile(np.abs(q), 99)),
                "q_ptp": float(np.percentile(q, 95) - np.percentile(q, 5)),
                "tau_mean": float(tau.mean()),
                "tau_p99": float(np.percentile(np.abs(tau), 99)),
                "tau_rms": float(np.sqrt(np.mean(tau**2))),
                "err_mean": float(err.mean()),
                "err_p99": float(np.percentile(np.abs(err), 99)),
                "act_mean": float(act.mean()),
                "raw_q_mean": float(d[f"q_{s}_{stem}"][W].mean() * DEG),
                "raw_tau_mean": float(d[f"tau_{s}_{stem}"][W].mean()),
            }
        chans[ch] = row
    # derived ankle: joint angle from the recorded passive coordinates, torque = Jc^T tau_crank
    for ch, stem, disp, idx in (("ankle_pitch", "ankle_pitch_joint", +1, 0), ("ankle_roll", "ankle_roll_joint", +1, 1)):
        row = {}
        for s in "LR":
            f = mirror_factor(meta, stem, s) * (-1 if ch == "ankle_pitch" else 1) * disp
            q = d[f"q_{s}_{stem}"][W] * f * DEG
            row[s] = {"q_mean": float(q.mean()), "q_p99": float(np.percentile(np.abs(q), 99)),
                      "q_ptp": float(np.percentile(q, 95) - np.percentile(q, 5))}
            if jcT is not None:
                qp = d[f"q_{s}_ankle_pitch_joint"]
                qr = d[f"q_{s}_ankle_roll_joint"]
                tc = np.stack([d[f"tau_{s}_crank_A_joint"], d[f"tau_{s}_crank_B_joint"]], 1)
                ta = np.array([jcT(s, qp[i], qr[i]) @ tc[i] for i in range(a, b)])[:, idx] * (1.0 if side_is_left(s) else -1.0)
                row[s].update({"tau_mean": float(ta.mean()), "tau_p99": float(np.percentile(np.abs(ta), 99)),
                               "tau_rms": float(np.sqrt(np.mean(ta**2)))})
        chans[ch] = row
    res["chan"] = chans

    # net hip-yaw torque on the pelvis (both axes are +z, raw coordinates add)
    res["net_hip_yaw_torque_mean"] = float(
        (d["tau_L_hip_yaw_joint"][W] + d["tau_R_hip_yaw_joint"][W]).mean()
    )
    res["net_hip_yaw_angle_mean_deg"] = float(
        (d["q_L_hip_yaw_joint"][W] + d["q_R_hip_yaw_joint"][W]).mean() / 2 * DEG
    )

    # ---- foot placement at touchdown (pelvis frame, forward kinematics)
    if mjm is not None:
        fk = fk_feet(mjm, d)
        place = {}
        for s, td in (("L", tdL), ("R", tdR)):
            td = td[(td >= a) & (td <= b)]
            sgn = 1.0 if s == "L" else -1.0  # mirror the lateral offset and the foot yaw
            place[s] = {
                "reach_x": float(fk[s][td, 0].mean()),
                "lateral_y": float((sgn * fk[s][td, 1]).mean()),
                "foot_yaw_deg": float((sgn * fk[s][td, 3]).mean() * DEG),
            }
        res["place"] = place

    # ---- yaw increment per support phase (short window, telescoping sum)
    psi_s, _, _, ok_s = yaw_track(d["base_x"], d["base_y"], d["base_vx"], d["base_vy"], YAW_WIN_SHORT)
    dpsi = np.diff(psi_s) * DEG
    ph = {
        "L_only": cL & ~cR,
        "R_only": ~cL & cR,
        "double": cL & cR,
        "flight": ~cL & ~cR,
    }
    inc = {}
    for k, m in ph.items():
        mm = m[:-1] & ok_s[:-1] & ok_s[1:]
        mm[: a] = False
        mm[b:] = False
        inc[k] = float(dpsi[mm].sum())
    inc["window_s"] = float(t[b] - t[a])
    inc["total"] = float(sum(inc[k] for k in ph))
    res["yaw_inc"] = inc
    return res


# ----------------------------------------------------------------------------- stand file
def analyse_stand(path, meta):
    d = np.load(path)
    t = d["time"]
    q = d["qpos_full"]
    w, z = q[:, 3], q[:, 6]
    x_, y_ = q[:, 4], q[:, 5]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x_ * y_), 1 - 2 * (y_**2 + z**2)))
    m = t >= 3.0
    tt = t[m]
    yy = (yaw[m] - yaw[m][0]) * DEG
    p = np.polyfit(tt, yy, 1)
    BW = meta["BW_N"]
    fl, fr = d["GRFc_L_foot_link_z"][m] / BW, d["GRFc_R_foot_link_z"][m] / BW
    qs = {}
    for s in "LR":
        f = mirror_factor(meta, "hip_yaw_joint", s)
        qs[s] = float(d[f"qpos_{s}_hip_yaw_joint"][m].mean() * f * DEG)
    mt = {}
    for ch, (stem, disp) in CHANNELS.items():
        mt[ch] = {sd: float(d[f"tau_{sd}_{stem}"][m].mean() * mirror_factor(meta, stem, sd) * disp) for sd in "LR"}
    for ch, ax in (("ankle_pitch", "pitch"), ("ankle_roll", "roll")):
        mt[ch] = {sd: float(d[f"tauank_eq_{sd}_{ax}"][m].mean() * (1.0 if side_is_left(sd) else -1.0)) for sd in "LR"}
    return {
        "seed": int(path.stem.split("_s")[-1]),
        "mirror_tau": mt,
        "yaw_change_deg": float(yy[-1]),
        "yaw_rate_deg_s": float(p[0]),
        "yaw_abs_max_deg": float(np.abs(yy).max()),
        "travel_m": float(np.hypot(d["base_vx"][m], d["base_vy"][m]).sum() * DT),
        "Fz_share_L_pct": float(fl.sum() / (fl.sum() + fr.sum()) * 100),
        "hip_yaw_mean_L": qs["L"],
        "hip_yaw_mean_R": qs["R"],
        "cmd_wz": float(np.unique(d["cmd_wz"])[0]),
        "duration_s": float(tt[-1] - tt[0]),
    }


# ----------------------------------------------------------------------------- validation
def validate():
    errs = {}
    for win in (YAW_WIN, YAW_WIN_SHORT):
        e = []
        for s in range(10):
            f = OUT / f"v5_walkstop_s{s}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            q = d["qpos_full"][:188]
            yaw_true = np.unwrap(np.arctan2(2 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]),
                                           1 - 2 * (q[:, 5] ** 2 + q[:, 6] ** 2)))
            psi, _, _, ok = yaw_track(q[:, 0], q[:, 1], d["base_vx"][:188], d["base_vy"][:188], win)
            sel = ok.copy()
            sel[: int(2.0 / DT)] = False  # the walk-stop file starts from rest
            if sel.sum() < 5:
                continue
            i0 = int(np.argmax(sel))
            r = (psi - psi[i0]) - (yaw_true - yaw_true[i0])
            e.append(float(np.abs(r[sel]).max() * DEG))
        errs[win] = e
    return errs


# ----------------------------------------------------------------------------- tables
def agg(runs, fn):
    v = np.array([fn(r) for r in runs], float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=0))


def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def table_channel_p99(R, key, unit, nd=2, chans=None):
    """one row per channel; columns per policy: left, right, (L-R)%."""
    lines = []
    hdr = "| 항목 | " + " | ".join(f"{KO[p]} 왼쪽 | {KO[p]} 오른쪽 | {KO[p]} 차이 [%]" for p in POLICIES) + " |"
    lines.append(hdr)
    lines.append("|---|" + "--:|" * (3 * len(POLICIES)))
    for ch in chans or KO_CH:
        cells = []
        for p in POLICIES:
            runs = R[p]
            if key not in runs[0]["chan"][ch]["L"]:
                cells += ["—"] * 3
                continue
            l, _ = agg(runs, lambda r: r["chan"][ch]["L"][key])
            rr, _ = agg(runs, lambda r: r["chan"][ch]["R"][key])
            pct, sd = agg(runs, lambda r: rel_pct(r["chan"][ch]["L"][key], r["chan"][ch]["R"][key]))
            cells += [fmt(l, nd), fmt(rr, nd), f"{pct:+.1f} ± {sd:.1f}"]
        lines.append(f"| {KO_CH[ch]} [{unit}] | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_channel_mean(R, key, unit, nd=2, chans=None):
    """signed means: left, right, difference in physical units."""
    lines = []
    hdr = "| 항목 | " + " | ".join(f"{KO[p]} 왼쪽 | {KO[p]} 오른쪽 | {KO[p]} 왼−오른" for p in POLICIES) + " |"
    lines.append(hdr)
    lines.append("|---|" + "--:|" * (3 * len(POLICIES)))
    for ch in chans or KO_CH:
        cells = []
        for p in POLICIES:
            runs = R[p]
            if key not in runs[0]["chan"][ch]["L"]:
                cells += ["—"] * 3
                continue
            l, _ = agg(runs, lambda r: r["chan"][ch]["L"][key])
            rr, _ = agg(runs, lambda r: r["chan"][ch]["R"][key])
            dd, sd = agg(runs, lambda r: r["chan"][ch]["L"][key] - r["chan"][ch]["R"][key])
            cells += [fmt(l, nd), fmt(rr, nd), f"{dd:+.{nd}f} ± {sd:.{nd}f}"]
        lines.append(f"| {KO_CH[ch]} [{unit}] | " + " | ".join(cells) + " |")
    return "\n".join(lines)


KO_FEAT = {
    "hip_yaw_angle_mean_L-R_deg": "엉덩이 비틀기 각도 평균, 왼−오른 [도]",
    "hip_yaw_action_mean_L-R_deg": "엉덩이 비틀기 정책 출력 평균, 왼−오른 [도]",
    "net_hip_yaw_torque_mean_Nm": "두 엉덩이 비틀기 토크 합의 평균 [N·m]",
    "hip_yaw_torque_p99_L-R_pct": "엉덩이 비틀기 토크 99 백분위수 차이 [%]",
    "hip_yaw_angle_ptp_L-R_pct": "엉덩이 비틀기 각도 진폭 차이 [%]",
    "hip_roll_angle_mean_L-R_deg": "엉덩이 좌우 각도 평균, 왼−오른 [도]",
    "hip_roll_torque_mean_L-R_Nm": "엉덩이 좌우 토크 평균, 왼−오른 [N·m]",
    "knee_torque_p99_L-R_pct": "무릎 토크 99 백분위수 차이 [%]",
    "crank_A_torque_p99_L-R_pct": "크랭크 A 토크 99 백분위수 차이 [%]",
    "crank_B_torque_p99_L-R_pct": "크랭크 B 토크 99 백분위수 차이 [%]",
    "crank_A_error_p99_L-R_pct": "크랭크 A 오차 99 백분위수 차이 [%]",
    "stance_time_L-R_pct": "지지 시간 차이 [%]",
    "step_time_L-R_pct": "한 걸음 시간 차이 [%]",
    "step_length_base_L-R_pct": "걸음 길이 차이 [%]",
    "vertical_impulse_share_L_minus_50_pct": "수직 반력 누적 중 왼발 몫 빼기 50 [%p]",
    "body_lateral_speed_vy_m_s": "몸통 좌우 속도 [m/s]",
    "yaw_gain_in_L_only_stance_deg": "왼발만 땅일 때 방향각 증가 [도]",
    "yaw_gain_in_R_only_stance_deg": "오른발만 땅일 때 방향각 증가 [도]",
    "yaw_gain_in_double_support_deg": "두 발 땅일 때 방향각 증가 [도]",
    "foot_yaw_at_touchdown_L-R_deg": "착지 발 방향각, 왼−오른 [도]",
    "foot_reach_x_L-R_m": "착지 발 앞쪽 거리, 왼−오른 [m]",
    "foot_lateral_L-R_m": "착지 발 옆쪽 거리, 왼−오른 [m]",
}


def feature_vector(r):
    g = r["gait"]
    f = {
        "hip_yaw_angle_mean_L-R_deg": r["chan"]["hip_yaw"]["L"]["q_mean"] - r["chan"]["hip_yaw"]["R"]["q_mean"],
        "hip_yaw_action_mean_L-R_deg": r["chan"]["hip_yaw"]["L"]["act_mean"] - r["chan"]["hip_yaw"]["R"]["act_mean"],
        "net_hip_yaw_torque_mean_Nm": r["net_hip_yaw_torque_mean"],
        "hip_yaw_torque_p99_L-R_pct": rel_pct(r["chan"]["hip_yaw"]["L"]["tau_p99"], r["chan"]["hip_yaw"]["R"]["tau_p99"]),
        "hip_yaw_angle_ptp_L-R_pct": rel_pct(r["chan"]["hip_yaw"]["L"]["q_ptp"], r["chan"]["hip_yaw"]["R"]["q_ptp"]),
        "hip_roll_angle_mean_L-R_deg": r["chan"]["hip_roll"]["L"]["q_mean"] - r["chan"]["hip_roll"]["R"]["q_mean"],
        "hip_roll_torque_mean_L-R_Nm": r["chan"]["hip_roll"]["L"]["tau_mean"] - r["chan"]["hip_roll"]["R"]["tau_mean"],
        "knee_torque_p99_L-R_pct": rel_pct(r["chan"]["knee"]["L"]["tau_p99"], r["chan"]["knee"]["R"]["tau_p99"]),
        "crank_A_torque_p99_L-R_pct": rel_pct(r["chan"]["crank_A"]["L"]["tau_p99"], r["chan"]["crank_A"]["R"]["tau_p99"]),
        "crank_B_torque_p99_L-R_pct": rel_pct(r["chan"]["crank_B"]["L"]["tau_p99"], r["chan"]["crank_B"]["R"]["tau_p99"]),
        "crank_A_error_p99_L-R_pct": rel_pct(r["chan"]["crank_A"]["L"]["err_p99"], r["chan"]["crank_A"]["R"]["err_p99"]),
        "stance_time_L-R_pct": rel_pct(g["L"]["stance_time"], g["R"]["stance_time"]),
        "step_time_L-R_pct": rel_pct(g["L"]["step_time"], g["R"]["step_time"]),
        "step_length_base_L-R_pct": rel_pct(g["L"]["step_len_base"], g["R"]["step_len_base"]),
        "vertical_impulse_share_L_minus_50_pct": r["grf"]["impulse_share_L_pct"] - 50.0,
        "body_lateral_speed_vy_m_s": r["vy_mean"],
        "yaw_gain_in_L_only_stance_deg": r["yaw_inc"]["L_only"],
        "yaw_gain_in_R_only_stance_deg": r["yaw_inc"]["R_only"],
        "yaw_gain_in_double_support_deg": r["yaw_inc"]["double"],
    }
    if "place" in r:
        f["foot_yaw_at_touchdown_L-R_deg"] = r["place"]["L"]["foot_yaw_deg"] - r["place"]["R"]["foot_yaw_deg"]
        f["foot_reach_x_L-R_m"] = r["place"]["L"]["reach_x"] - r["place"]["R"]["reach_x"]
        f["foot_lateral_L-R_m"] = r["place"]["L"]["lateral_y"] - r["place"]["R"]["lateral_y"]
    return f


def build_md(R, S, val):
    o = []
    P = list(POLICIES)

    # --- A. yaw
    o.append("### 표 1. 몸통 방향각(yaw) 변화와 방향\n")
    o.append("부호: 위에서 봤을 때 반시계 방향이 양수(왼쪽으로 돈다). 방향각은 2 초부터 16.5 초까지 잰다.\n")
    o.append("| 정책 | 씨앗 | 방향각 변화율 [도/초] | 직선 맞춤 결정계수 | 16.5 초까지 방향각 변화 [도] | 경로 방향 변화(판정 도구 값, 부호 있음) [도] | 몸통 기준 진행 방향 [도] | 몸통 좌우 속도 [m/s] |")
    o.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for p in P:
        for s, r in zip(WALK_SEEDS, R[p]):
            o.append(f"| {KO[p]} | {s} | {r['yaw_rate_deg_s']:+.2f} | {r['yaw_linear_r2']:.3f} | {r['yaw_total_deg']:+.1f} | {r['verdict_drift_signed_deg']:+.1f} | {r['beta_mean_deg']:+.1f} | {r['vy_mean']:+.3f} |")
        ym = np.mean([r["yaw_rate_deg_s"] for r in R[p]])
        ys = np.std([r["yaw_rate_deg_s"] for r in R[p]])
        tm = np.mean([r["yaw_total_deg"] for r in R[p]])
        ts = np.std([r["yaw_total_deg"] for r in R[p]])
        vm = np.mean([r["verdict_drift_signed_deg"] for r in R[p]])
        vs = np.std([r["verdict_drift_signed_deg"] for r in R[p]])
        o.append(f"| **{KO[p]} 평균 ± 표준편차** | 3 | **{ym:+.2f} ± {ys:.2f}** | | **{tm:+.1f} ± {ts:.1f}** | **{vm:+.1f} ± {vs:.1f}** | | |")
    o.append("")
    o.append("### 표 2. 시간 구간별 방향각 변화율 [도/초]\n")
    o.append("| 정책 | 씨앗 | 2~5 초 | 5~8 초 | 8~11 초 | 11~14 초 | 14~16.5 초 | 앞 절반 | 뒤 절반 |")
    o.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for p in P:
        for s, r in zip(WALK_SEEDS, R[p]):
            w = r["yaw_rate_windows"]
            o.append(f"| {KO[p]} | {s} | " + " | ".join(f"{x:+.2f}" for x in w) + f" | {r['yaw_rate_first_half']:+.2f} | {r['yaw_rate_second_half']:+.2f} |")
    o.append("")

    # --- B. command
    o.append("### 표 3. 측정이 준 명령과 실제 몸통 움직임\n")
    o.append("| 정책 | 명령 앞 속도 [m/s] | 명령 회전 속도 [rad/s] | 실제 앞 속도 [m/s] | 실제 방향각 변화율 [도/초] |")
    o.append("|---|--:|--:|--:|--:|")
    for p in P:
        o.append(f"| {KO[p]} | {np.mean([r['cmd_vx'] for r in R[p]]):.2f} | 0.00 | {np.mean([r['vx_mean'] for r in R[p]]):.3f} | {np.mean([r['yaw_rate_deg_s'] for r in R[p]]):+.2f} |")
    o.append("")

    # --- C. channels
    o.append("### 표 4. 관절 각도 — 99 백분위수 (절댓값) [도]\n")
    o.append(table_channel_p99(R, "q_p99", "도") + "\n")
    o.append("### 표 5. 관절 각도 — 평균 (부호 있음) [도]\n")
    o.append(table_channel_mean(R, "q_mean", "도") + "\n")
    o.append("### 표 6. 관절 각도 — 진폭 (5~95 백분위수 폭) [도]\n")
    o.append(table_channel_p99(R, "q_ptp", "도") + "\n")
    o.append("### 표 7. 관절 토크 — 99 백분위수 (절댓값) [N·m]\n")
    o.append(table_channel_p99(R, "tau_p99", "N·m", 1) + "\n")
    o.append("### 표 8. 관절 토크 — 평균 (부호 있음) [N·m]\n")
    o.append(table_channel_mean(R, "tau_mean", "N·m", 2) + "\n")
    o.append("### 표 9. 목표 빼기 실제 각도 오차 — 99 백분위수 (절댓값) [도]\n")
    o.append(table_channel_p99(R, "err_p99", "도", 2, chans=list(CHANNELS)) + "\n")
    o.append("### 표 10. 목표 빼기 실제 각도 오차 — 평균 (부호 있음) [도]\n")
    o.append(table_channel_mean(R, "err_mean", "도", 3, chans=list(CHANNELS)) + "\n")

    # --- D. actions
    o.append("### 표 11. 정책 출력의 평균 치우침 (목표 빼기 기본 자세, 부호 있음) [도]\n")
    o.append(table_channel_mean(R, "act_mean", "도", 2, chans=list(CHANNELS)) + "\n")

    # --- E. gait timing
    o.append("### 표 12. 걸음 시간과 걸음 길이 (왼발, 오른발)\n")
    o.append("걸음 길이는 몸통이 반대 발 착지에서 이 발 착지까지 움직인 거리이다. 걸음 시간은 같은 구간의 시간이다.\n")
    o.append("| 항목 | " + " | ".join(f"{KO[p]} 왼쪽 | {KO[p]} 오른쪽 | {KO[p]} 차이 [%]" for p in P) + " |")
    o.append("|---|" + "--:|" * (3 * len(P)))
    for nm, key, unit, nd in (("한 걸음 시간", "step_time", "초", 3), ("걸음 길이", "step_len_base", "m", 3),
                              ("발 한 바퀴(두 걸음) 시간", "stride_time", "초", 3), ("지지 시간", "stance_time", "초", 3),
                              ("스윙 시간", "swing_time", "초", 3)):
        cells = []
        for p in P:
            l, _ = agg(R[p], lambda r: r["gait"]["L"][key])
            rr, _ = agg(R[p], lambda r: r["gait"]["R"][key])
            pct, sd = agg(R[p], lambda r: rel_pct(r["gait"]["L"][key], r["gait"]["R"][key]))
            cells += [fmt(l, nd), fmt(rr, nd), f"{pct:+.1f} ± {sd:.1f}"]
        o.append(f"| {nm} [{unit}] | " + " | ".join(cells) + " |")
    o.append("")
    o.append("### 표 13. 지지 구간 비율과 접지 반력\n")
    o.append("| 항목 | " + " | ".join(KO[p] for p in P) + " |")
    o.append("|---|" + "--:|" * len(P))
    for nm, fn, nd in (
        ("두 발 모두 땅 [%]", lambda r: r["gait"]["both_stance_pct"], 1),
        ("두 발 모두 공중 [%]", lambda r: r["gait"]["flight_pct"], 1),
        ("왼발만 땅 [%]", lambda r: r["gait"]["L_only_pct"], 1),
        ("오른발만 땅 [%]", lambda r: r["gait"]["R_only_pct"], 1),
        ("수직 반력 평균, 왼발 [체중]", lambda r: r["grf"]["mean_L"], 3),
        ("수직 반력 평균, 오른발 [체중]", lambda r: r["grf"]["mean_R"], 3),
        ("수직 반력 99 백분위수, 왼발 [체중]", lambda r: r["grf"]["p99_L"], 2),
        ("수직 반력 99 백분위수, 오른발 [체중]", lambda r: r["grf"]["p99_R"], 2),
        ("수직 반력 누적 중 왼발 몫 [%]", lambda r: r["grf"]["impulse_share_L_pct"], 2),
    ):
        o.append(f"| {nm} | " + " | ".join(
            "{:.{n}f} ± {:.{n}f}".format(*agg(R[p], fn), n=nd) for p in P) + " |")
    o.append("")
    if "place" in R[P[0]][0]:
        o.append("### 표 14. 착지 순간 발 위치 (몸통 좌표계, 앞뒤 기울기는 미보정)\n")
        o.append("옆 방향 거리와 발 방향각은 왼쪽 기준으로 거울 반사했다. 발 방향각 양수는 발끝이 바깥을 향한다.\n")
        o.append("| 항목 | " + " | ".join(f"{KO[p]} 왼쪽 | {KO[p]} 오른쪽 | {KO[p]} 왼−오른" for p in P) + " |")
        o.append("|---|" + "--:|" * (3 * len(P)))
        for nm, key, nd in (("앞쪽 거리 [m]", "reach_x", 3), ("옆쪽 거리 [m]", "lateral_y", 3), ("발 방향각 [도]", "foot_yaw_deg", 2)):
            cells = []
            for p in P:
                l, _ = agg(R[p], lambda r: r["place"]["L"][key])
                rr, _ = agg(R[p], lambda r: r["place"]["R"][key])
                dd, sd = agg(R[p], lambda r: r["place"]["L"][key] - r["place"]["R"][key])
                cells += [fmt(l, nd), fmt(rr, nd), f"{dd:+.{nd}f} ± {sd:.{nd}f}"]
            o.append(f"| {nm} | " + " | ".join(cells) + " |")
        o.append("")
    o.append("### 표 15. 지지 구간별 방향각 증가 (분석 구간 전체 합) [도]\n")
    o.append("| 정책 | 씨앗 | 왼발만 땅 | 오른발만 땅 | 두 발 땅 | 두 발 공중 | 합 | 분석 구간 [초] |")
    o.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for p in P:
        for s, r in zip(WALK_SEEDS, R[p]):
            i = r["yaw_inc"]
            o.append(f"| {KO[p]} | {s} | {i['L_only']:+.1f} | {i['R_only']:+.1f} | {i['double']:+.1f} | {i['flight']:+.1f} | {i['total']:+.1f} | {i['window_s']:.1f} |")
    o.append("")
    o.append("### 표 16. 엉덩이 비틀기 순수 토크와 각도 (두 다리 합, 원래 좌표)\n")
    o.append("| 정책 | 씨앗 | 두 엉덩이 비틀기 토크 합의 평균 [N·m] | 두 엉덩이 비틀기 각도 평균 [도] |")
    o.append("|---|--:|--:|--:|")
    for p in P:
        for s, r in zip(WALK_SEEDS, R[p]):
            o.append(f"| {KO[p]} | {s} | {r['net_hip_yaw_torque_mean']:+.2f} | {r['net_hip_yaw_angle_mean_deg']:+.2f} |")
    o.append("")

    # --- F. stand
    o.append("### 표 17. 서 있기 (명령 0, 3 초 이후 27 초)\n")
    o.append("| 정책 | 씨앗 | 방향각 변화 [도] | 방향각 변화율 [도/초] | 흘러간 거리 [m] | 왼발 반력 몫 [%] | 엉덩이 비틀기 평균 왼쪽 [도] | 엉덩이 비틀기 평균 오른쪽 [도] |")
    o.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for p in P:
        for s, r in zip(STAND_SEEDS, S[p]):
            o.append(f"| {KO[p]} | {s} | {r['yaw_change_deg']:+.1f} | {r['yaw_rate_deg_s']:+.3f} | {r['travel_m']:.3f} | {r['Fz_share_L_pct']:.1f} | {r['hip_yaw_mean_L']:+.2f} | {r['hip_yaw_mean_R']:+.2f} |")
        o.append(f"| **{KO[p]} 평균 ± 표준편차** | 4 | **{np.mean([r['yaw_change_deg'] for r in S[p]]):+.1f} ± {np.std([r['yaw_change_deg'] for r in S[p]]):.1f}** | | | | | |")
    o.append("")

    # --- stand mirror check
    o.append("### 표 20. 거울 반사 부호 검증 — 서 있기 평균 토크 (거울 반사한 값) [N·m]\n")
    o.append("대칭 자세에서 왼쪽과 오른쪽 부호가 같아야 한다. 값은 서 있기 씨앗 0, 1, 3 의 평균이다 (씨앗 2 는 제자리에서 돈다).\n")
    o.append("| 정책 | " + " | ".join(f"{KO_CH[c]} 왼쪽 | {KO_CH[c]} 오른쪽" for c in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "crank_A", "crank_B")) + " | 발목 앞뒤 파생 왼쪽 | 발목 앞뒤 파생 오른쪽 | 발목 좌우 파생 왼쪽 | 발목 좌우 파생 오른쪽 |")
    o.append("|---|" + "--:|" * 16)
    for p in P:
        cells = []
        for c in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "crank_A", "crank_B"):
            for sd in "LR":
                cells.append("{:+.1f}".format(np.mean([r["mirror_tau"][c][sd] for r in S[p] if r["seed"] != 2])))
        for c in ("ankle_pitch", "ankle_roll"):
            for sd in "LR":
                cells.append("{:+.1f}".format(np.mean([r["mirror_tau"][c][sd] for r in S[p] if r["seed"] != 2])))
        o.append(f"| {KO[p]} | " + " | ".join(cells) + " |")
    o.append("")

    # --- G. correlations
    feats = [feature_vector(r) for p in P for r in R[p]]
    rate = np.array([r["yaw_rate_deg_s"] for p in P for r in R[p]])
    rows = []
    for k in feats[0]:
        v = np.array([f[k] for f in feats], float)
        if np.nanstd(v) < 1e-12:
            continue
        c = float(np.corrcoef(v, rate)[0, 1])
        rows.append((k, c, float(v.mean()), float(v.std())))
    rows.sort(key=lambda x: -abs(x[1]))
    o.append("### 표 18. 방향각 변화율과 비대칭 지표의 상관 (아홉 번 걷기 전체, 피어슨 상관계수)\n")
    o.append("| 지표 | 상관계수 | 아홉 값의 평균 | 표준편차 |")
    o.append("|---|--:|--:|--:|")
    for k, c, m_, s_ in rows:
        o.append(f"| {KO_FEAT.get(k, k)} | {c:+.2f} | {m_:+.3f} | {s_:.3f} |")
    o.append("")
    o.append("### 표 19. 방향각 추정 방법의 검증\n")
    o.append("v5 걷다 멈추기 파일(10 개)의 처음 3.75 초에서 속도로 복원한 방향각과 사원수에서 구한 진짜 방향각의 최대 차이 [도]이다.\n")
    o.append("| 평균 창 | 파일 수 | 최대 차이의 평균 [도] | 최대 차이의 최댓값 [도] |")
    o.append("|---|--:|--:|--:|")
    for win, e in val.items():
        if e:
            o.append(f"| {win} 표본 ({win * DT:.2f} 초) | {len(e)} | {np.mean(e):.2f} | {np.max(e):.2f} |")
    o.append("")
    return "\n".join(o), rows


# ----------------------------------------------------------------------------- plot
def make_plot(R, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"baseline": "#1b6ca8", "bundle_trial": "#d17a00", "armA": "#b0302a"}
    lab = {"baseline": "baseline (rough_box_main_v11a)", "bundle_trial": "bundle trial (crank 55 + randomization)",
           "armA": "arm A (crank 60/2.7 only)"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    for p in POLICIES:
        for k, r in enumerate(R[p]):
            a.plot(r["_t"], r["_psi_rel"], color=col[p], lw=1.6, alpha=0.9, label=lab[p] if k == 0 else None,
                   ls=("-", "--", ":")[k])
    a.axhline(0, color="#888", lw=0.8)
    a.set_xlabel("time [s]")
    a.set_ylabel("body yaw relative to t = 2 s [deg]   (+ = left turn)")
    a.set_title("Body yaw while commanded straight (wz = 0)\nline style = seed 11 / 12 / 13")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    a.grid(alpha=0.25)

    b = ax[1]
    def pct(ch, k):
        return lambda r: rel_pct(r["chan"][ch]["L"][k], r["chan"][ch]["R"][k])

    metrics = [
        ("hip yaw angle range", pct("hip_yaw", "q_ptp")),
        ("hip roll angle range", pct("hip_roll", "q_ptp")),
        ("hip pitch angle range", pct("hip_pitch", "q_ptp")),
        ("knee angle range", pct("knee", "q_ptp")),
        ("ankle roll angle range", pct("ankle_roll", "q_ptp")),
        ("hip yaw torque, 99th pct", pct("hip_yaw", "tau_p99")),
        ("hip roll torque, 99th pct", pct("hip_roll", "tau_p99")),
        ("knee torque, 99th pct", pct("knee", "tau_p99")),
        ("crank A torque, 99th pct", pct("crank_A", "tau_p99")),
        ("crank B torque, 99th pct", pct("crank_B", "tau_p99")),
        ("crank A error, 99th pct", pct("crank_A", "err_p99")),
        ("stance time", lambda r: rel_pct(r["gait"]["L"]["stance_time"], r["gait"]["R"]["stance_time"])),
        ("step length (base travel)", lambda r: rel_pct(r["gait"]["L"]["step_len_base"], r["gait"]["R"]["step_len_base"])),
        ("vertical force, 99th pct", lambda r: rel_pct(r["grf"]["p99_L"], r["grf"]["p99_R"])),
    ]
    n = len(metrics)
    y = np.arange(n)
    h = 0.26
    for i, p in enumerate(POLICIES):
        m = np.array([[f(r) for r in R[p]] for _, f in metrics])
        b.barh(y + (i - 1) * h, m.mean(1), h, xerr=m.std(1), color=col[p], alpha=0.9, error_kw={"lw": 1, "capsize": 2},
               label=lab[p])
    b.set_yticks(y)
    b.set_yticklabels([m for m, _ in metrics], fontsize=9)
    b.invert_yaxis()
    b.axvline(0, color="#444", lw=0.8)
    b.set_xlabel("(left - right) / mean of both, [%]   (mean of 3 seeds, bar = 1 s.d.)")
    b.set_title("Left/right asymmetry per metric (mirrored joint values)\ncolors as in the left panel")
    b.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md-out")
    ap.add_argument("--json-out")
    ap.add_argument("--plot")
    ap.add_argument("--no-fk", action="store_true")
    a = ap.parse_args()

    meta = load_meta()
    mjm = None
    if not a.no_fk and MJB.exists():
        import mujoco

        mjm = mujoco.MjModel.from_binary_path(str(MJB))
    jcT = None
    if ENVELOPE.exists():
        e = json.load(open(ENVELOPE))
        pa, ra = np.radians(e["grid"]["pitch_deg"]), np.radians(e["grid"]["roll_deg"])
        JC = {s: np.array(e["legs"][s]["JcT"]) for s in "LR"}

        def jcT(s, p_, r_):
            i = int(np.clip(np.argmin(abs(pa - p_)), 0, len(pa) - 1))
            k = int(np.clip(np.argmin(abs(ra - r_)), 0, len(ra) - 1))
            return JC[s][i, k]

    R = {p: [analyse_walk(OUT / f"{stem}_walk_s{s}_vx1.2.npz", meta, mjm, jcT) for s in WALK_SEEDS]
         for p, stem in POLICIES.items()}
    S = {p: [analyse_stand(OUT / f"{stem}_stand_s{s}.npz", meta) for s in STAND_SEEDS] for p, stem in POLICIES.items()}
    val = validate()
    md, corr = build_md(R, S, val)
    if a.md_out:
        Path(a.md_out).write_text(md)
    if a.json_out:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o

        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"walk": clean(R), "stand": clean(S), "validation_max_err_deg": {str(k): v for k, v in val.items()},
                   "correlations": corr}, open(a.json_out, "w"), indent=1)
    if a.plot:
        make_plot(R, a.plot)
    if not a.md_out:
        print(md)


if __name__ == "__main__":
    main()
