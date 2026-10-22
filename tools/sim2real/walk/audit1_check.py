#!/usr/bin/env python3
"""Independent audit of the 2026-10-22 real-walk log claims (C1..C9).

Written from the raw CSVs only (no import of real_walk_analyze.py / policy_replay_check.py).
Run (CPU, mjlab venv):
  CUDA_VISIBLE_DEVICES="" mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/audit1_check.py \
      --ckpt /tmp/ckpt/terrain10_2048_model_37400.pt [--data Expr_Data] [--only C3,C8]
Default pose is rebuilt from pygmalion_mjlab/robot.py INIT_STATE (hip 0.175, knee 0.35 rad, crank values).
Output: plain text tables on stdout.
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from scipy.signal import lombscargle, welch, coherence, csd, find_peaks
warnings.filterwarnings("ignore")

J = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"]
JN = [f"{l}/{j}" for l in ("right_leg", "left_leg") for j in J]
DEF = np.zeros(12)  # deg
DEF[[0, 3, 4, 5]] = np.degrees([-0.175, 0.35, 0.14395487369574003, -0.14706307530170212])
DEF[[6, 9, 10, 11]] = np.degrees([0.175, -0.35, -0.1438849965517705, 0.14687020555452607])
STAND = {"1020": "policy_generalized_walk_20261020_185916.csv", "A": "1021ob061vx0.csv",
         "vib": "vx0gain1walkob(진동확인용)1021.csv", "Kd": "vx0gain1kdx1.5walkob1021.csv",
         "B": "1021vx0obgain0.61.csv"}
MOVE = {"vx0.3": "1021vx0.3ob.csv", "vx0.5": "1021vx0.5ob.csv", "broken": "1021vx0.5gain0615ob_broken.csv",
        "wz": "1021vz0.5ob.csv", "wzsole": "1021vz0.5ob발바닥 붙임.csv", "wz0.6": "1021vz0.5obgain0.61.csv"}
OLD = {"1019 g0.8": "1019gain0.8 stand.csv", "1019 g1#1": "1019gain1stand1.csv"}
VIB_LAT = "vx0gain1walklatency(진동확인용)1021.csv"
D = "Expr_Data"


def load(name):
    p = os.path.join(D, name)
    with open(p, encoding="utf-8") as f:
        skip = 1 if f.readline().startswith("#") else 0
    return pd.read_csv(p, skiprows=skip)


def cols(df, pref):
    return df[[f"{pref}/{n}" for n in JN]].to_numpy()


def obs_of(df):
    return np.hstack([df[["obs/gyro/x", "obs/gyro/y", "obs/gyro/z"]].to_numpy(),
                      df[["obs/grav/x", "obs/grav/y", "obs/grav/z"]].to_numpy(),
                      cols(df, "obs/q_prev"), cols(df, "obs/q"), cols(df, "obs/last_act"),
                      df[["obs/cmd/vx", "obs/cmd/wz"]].to_numpy()])


def c1():
    print("== C1 timing")
    for nm, f in {**OLD, **STAND, **MOVE}.items():
        dt = np.diff(load(f).t.to_numpy()) * 1e3
        print(f"{nm:10s} policy dt median {np.median(dt):5.1f} p5 {np.percentile(dt,5):5.1f} min {dt.min():5.1f} max {dt.max():6.1f} ms")
    lat = pd.read_csv(os.path.join(D, VIB_LAT), skiprows=1)
    tl = lat[lat.joint == "right_leg/hip_pitch"].t.to_numpy()
    print("vib-check loop cycle median %.1f ms, mean %.2f ms (%.1f Hz)" % (np.median(np.diff(tl)) * 1e3, np.diff(tl).mean() * 1e3, 1 / np.diff(tl).mean()))
    dp = np.diff(load(STAND["vib"]).t.to_numpy())
    for N in (1, 2, 3, 4, 5, 6, 8):
        m = min(len(dp), len(tl) // N - 1)
        sp = np.array([tl[N * (k + 1)] - tl[N * k] for k in range(m)])
        print(f"  policy interval vs sum of {N} loop cycles: r={np.corrcoef(dp[:m], sp)[0,1]:+.3f}  MAE {np.mean(np.abs(dp[:m]-sp))*1e3:5.2f} ms")


def c2():
    print("== C2 log consistency (max abs diff)")
    for nm, f in {**OLD, **STAND, **MOVE}.items():
        df = load(f); q = cols(df, "obs/q")
        print(f"{nm:10s} q_prev(k)-q(k-1) {np.abs(cols(df,'obs/q_prev')[1:]-q[:-1]).max():.1g}  last_act(k)-act(k-1) "
              f"{np.abs(cols(df,'obs/last_act')[1:]-cols(df,'act')[:-1]).max():.1g}  pos-DEF-deg(q) {np.abs(cols(df,'pos')-DEF-np.degrees(q)).max():.2g} deg")


def ls(t, y, fr, chunk=1500):
    """Lomb-Scargle on the raw (non-uniform) timestamps; frequency grid is chunked (a 30k-frequency
    grid on 3.8k samples otherwise needs ~1 GB and was OOM-killed)."""
    y = y - y.mean()
    return np.concatenate([lombscargle(t, y, 2 * np.pi * fr[i:i + chunk], normalize=True) for i in range(0, len(fr), chunk)])


def c3():
    print("== C3 5 Hz oscillation, vib-check")
    df = load(STAND["vib"]); t = df.t.to_numpy(); t0 = t - t[0]; pos = cols(df, "pos"); act = cols(df, "act"); tgt = cols(df, "tgt")
    fr = np.arange(0.3, 60, 0.02)
    for nm, i in (("R hip_pitch", 0), ("R knee", 3)):
        for src, a in (("pos", pos), ("act", act)):
            p = ls(t0, a[:, i], fr); idx, _ = find_peaks(p); idx = idx[np.argsort(p[idx])[::-1]][:3]
            print(f"LS 0.3-60 Hz raw t, {nm} {src}: top", [(round(fr[k], 2), round(p[k], 3)) for k in idx])
    fine = np.arange(0.3, 60, 0.002)
    for f0 in (5.09, 13.32):
        p = ls(t0, np.sin(2 * np.pi * f0 * t0), fine); idx, _ = find_peaks(p); idx = idx[np.argsort(p[idx])[::-1]][:3]
        print(f"  sampling-window test: pure {f0} Hz sine at logged t -> LS top", [(round(fine[k], 2), round(p[k], 3)) for k in idx])
    frb = np.arange(3, 12, 0.01)
    pk_p = [round(frb[ls(t0, pos[:, j], frb).argmax()], 2) for j in range(12)]
    pk_a = [round(frb[ls(t0, act[:, j], frb).argmax()], 2) for j in range(12)]
    print("per-joint LS peak 3-12 Hz  pos:", pk_p, "\n                           act:", pk_a)
    for dt in (0.02, 0.0543, 0.1):
        tu = np.arange(t0[0], t0[-1], dt); y = np.interp(tu, t0, pos[:, 0])
        fw, pw = welch(y, 1 / dt, nperseg=256, detrend="linear"); print(f"uniform resample dt={dt*1e3:.0f} ms: Welch peak >2 Hz {fw[(fw>2)][pw[fw>2].argmax()]:.2f} Hz")
    print("10 s windows: LS peak 3-9 Hz (norm. power) of R hip_pitch act, act std")
    frw = np.arange(3, 9, 0.01); row = []
    for w in range(int(t0[-1] // 10)):
        m = (t0 >= 10 * w) & (t0 < 10 * w + 10); p = ls(t0[m], act[m, 0], frw); row.append(f"{10*w}s:{frw[p.argmax()]:.2f}Hz/{p.max():.2f}/sd{act[m,0].std():.3f}")
    print("  " + " ".join(row))
    for a, b in ((150, 152), (60, 62)):
        m = (t0 >= a) & (t0 < b); y = act[m, 0]; pk, _ = find_peaks(y - y.mean(), prominence=0.1)
        print(f"  {a}-{b} s: {m.sum()} rows, {len(pk)} cycles, step gaps {np.diff(pk).tolist()}, time gaps ms {np.round(np.diff(t0[m][pk])*1e3).astype(int).tolist()}")
    for nm, f in (("1020", STAND["1020"]), ("vib", STAND["vib"]), ("Kd", STAND["Kd"])):
        d = load(f); tt = d.t.to_numpy(); mdt = np.diff(tt).mean(); a = cols(d, "act"); ps = cols(d, "pos"); tg = cols(d, "tgt")
        f_, coh = coherence(a[:, 0], ps[:, 0], fs=1.0, nperseg=256, detrend="linear")
        f_, Pxy = csd(tg[:, 0], ps[:, 0], fs=1.0, nperseg=256, detrend="linear"); f_, Pxx = welch(tg[:, 0], fs=1.0, nperseg=256, detrend="linear")
        b = (f_ > 0.15) & (f_ < 0.4); k = np.where(b)[0][Pxx[b].argmax()]
        print(f"  {nm}: index-domain Welch R hip_pitch peak {f_[k]:.3f} cyc/step = period {1/f_[k]:.2f} steps ({f_[k]/mdt:.2f} Hz at mean dt {mdt*1e3:.1f} ms); act-pos coherence {coh[k]:.2f}; |tgt->pos| {abs(Pxy[k]/Pxx[k]):.2f}")


def c4_c5():
    print("== C4 mean(tgt-pos) crank [Ra, Rb, La, Lb] deg ; C5 hip_yaw mirror offset -(R+L) (axes both +z => symmetric pose has R=-L)")
    for nm, f in {**OLD, **STAND, **MOVE}.items():
        df = load(f); pos = cols(df, "pos"); tgt = cols(df, "tgt")
        print(f"{nm:10s} crank err {np.round((tgt-pos)[:,[4,5,10,11]].mean(0),2)} | yaw offset pos {-(pos[:,2].mean()+pos[:,8].mean()):+.2f} tgt {-(tgt[:,2].mean()+tgt[:,8].mean()):+.2f} | crank pos-DEF {np.round((pos-DEF)[:,[4,5,10,11]].mean(0),2)} tgt-DEF {np.round((tgt-DEF)[:,[4,5,10,11]].mean(0),2)}")


def runs_len(m):
    best = cur = st = bs = 0
    for i, v in enumerate(m):
        if v:
            if cur == 0: st = i
            cur += 1
            if cur > best: best, bs = cur, st
        else: cur = 0
    return best, bs


def c6():
    print("== C6 IMU freeze (exact repeat of previous row)")
    for nm, f in {**OLD, **STAND, **MOVE}.items():
        df = load(f); t0 = (df.t - df.t.iloc[0]).to_numpy()
        g = df[["obs/gyro/x", "obs/gyro/y", "obs/gyro/z"]].to_numpy(); gr = df[["obs/grav/x", "obs/grav/y", "obs/grav/z"]].to_numpy(); q = cols(df, "obs/q")
        fg = np.r_[False, (g[1:] == g[:-1]).all(1)]; fgr = np.r_[False, (gr[1:] == gr[:-1]).all(1)]
        L, s = runs_len(fg); dq = np.r_[0, np.degrees(np.abs(np.diff(q, axis=0))).mean(1)]
        zero = ((g == 0).all(1) & fg).sum() / max(fg.sum(), 1)
        print(f"{nm:10s} gyro repeat {fg.mean()*100:5.1f}% longest {L} rows ({t0[s+L-1]-t0[s]:.2f}s from {t0[s]:.1f}s) value==0 {zero*100:.0f}% grav repeat {fgr.mean()*100:.1f}% | joint move/row frozen {dq[fg].mean() if fg.any() else 0:.3f} live {dq[~fg].mean():.3f} deg")


def c7():
    print("== C7 heading (trapz of gyro z), deg/s; decimation spread shows sampling noise")
    for nm, f in {**STAND, **MOVE}.items():
        df = load(f); t = df.t.to_numpy(); gz = np.degrees(df["obs/gyro/z"].to_numpy()); T = t[-1] - t[0]
        full = np.trapz(gz, t); dec = [np.trapz(gz[o::2], t[o::2]) for o in range(2)]
        print(f"{nm:7s} T={T:6.1f}s mean {full/T:+6.2f} deg/s total {full:+7.1f} deg | every-2nd-row estimates {np.round(dec,1)} | iid scale {gz.std()*np.diff(t).mean()*np.sqrt(len(t)):.0f} deg")


def load_policy(ckpt):
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    mean, std = ck["obs_normalizer._mean"].double(), ck["obs_normalizer._std"].double()
    W = [(ck[f"mlp.{i}.weight"].double(), ck[f"mlp.{i}.bias"].double()) for i in (0, 2, 4, 6)]

    def pol(x, norm=True):
        x = torch.as_tensor(x, dtype=torch.float64)
        if norm: x = (x - mean) / (std + 0.01)
        for i, (w, b) in enumerate(W):
            x = x @ w.T + b
            if i < 3: x = torch.nn.functional.elu(x)
        return x.numpy()
    return pol


def c8_c9(ckpt):
    pol = load_policy(ckpt)
    print("== C8 replay max|act-replay|")
    for nm, f in {**STAND, **MOVE}.items():
        df = load(f); X = obs_of(df); A = cols(df, "act"); P = pol(X)
        print(f"{nm:7s} {np.abs(P-A).max():.2e} (no-norm {np.abs(pol(X,False)-A).max():.2f}, row-shift {np.abs(pol(X[1:])-A[:-1]).max():.2f})")
    print("== C9 finite difference, 12-target 2-norm [deg] per 1 deg input (mean over rows); q-only / alternating(+q,-q_prev)")
    for nm, f in {"vib": STAND["vib"], "wz": MOVE["wz"], "vx0.5": MOVE["vx0.5"]}.items():
        X = obs_of(load(f)); out = []
        for jn, j in (("R hip_roll", 1), ("L hip_roll", 7)):
            d = np.radians(1.0); Y = X.copy(); Y[:, 18 + j] += d
            n1 = np.linalg.norm(np.degrees(0.25 * (pol(Y) - pol(X))), axis=1)
            Y[:, 6 + j] -= d; n2 = np.linalg.norm(np.degrees(0.25 * (pol(Y) - pol(X))), axis=1)
            out.append(f"{jn} {n1.mean():.2f} / {n2.mean():.2f}")
        print(f"{nm:6s}", " | ".join(out))
    print("== N1 which input group carries the 5 Hz act energy (vib-check 50-110 s, open-loop counterfactual, band 4.4-5.8 Hz)")
    df = load(STAND["vib"]); X = obs_of(df); t0 = (df.t - df.t.iloc[0]).to_numpy(); m = (t0 > 50) & (t0 < 110); Xb = X[m]; tb = t0[m]; mu = Xb.mean(0)

    def band(A):
        tu = np.arange(tb[0], tb[-1], 0.02); P = 0
        for j in range(12):
            fw, pw = welch(np.interp(tu, tb, A[:, j]), 50.0, nperseg=512, detrend="linear"); b = (fw > 4.4) & (fw < 5.8); P += pw[b].sum() * (fw[1] - fw[0])
        return P
    P0 = band(pol(Xb)); G = {"gyro": slice(0, 3), "grav": slice(3, 6), "q_prev": slice(6, 18), "q": slice(18, 30), "last_act": slice(30, 42)}

    def run(keep):
        Y = np.tile(mu, (len(Xb), 1))
        for k in keep: Y[:, G[k]] = Xb[:, G[k]]
        return band(pol(Y)) / P0
    for k in G: print(f"  only {k:8s} varies: {run([k]):.3f}   all but {k:8s}: {run([x for x in G if x != k]):.3f}")
    print("  only q and q_prev vary:", round(run(["q", "q_prev"]), 3))


def ma_detrend(t, y, w=1.0):
    c = np.r_[0, np.cumsum(y)]; lo = np.searchsorted(t, t - w / 2, "left"); hi = np.searchsorted(t, t + w / 2, "right")
    return y - (c[hi] - c[lo]) / (hi - lo)


def sinfit(t, y, f):
    X = np.c_[np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t), np.ones_like(t)]; b = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.hypot(b[0], b[1]), np.arctan2(-b[1], b[0])


def c10_c12():
    print("== C10 per-joint LS peak 1-9 Hz of tgt after 1 s moving-average removal (joints with peak in 4.7-6.1 Hz)")
    runs = {"vib g1": STAND["vib"], "1020 g1": STAND["1020"], "Kd1.5": STAND["Kd"], "A g0.6": STAND["A"], "B g0.6": STAND["B"], "wz0.6": MOVE["wz0.6"]}
    fr = np.arange(1, 9.0001, 0.01)
    for nm, f in runs.items():
        df = load(f); t0 = (df.t - df.t.iloc[0]).to_numpy(); tgt = cols(df, "tgt")
        pk = [fr[ls(t0, ma_detrend(t0, tgt[:, j]), fr).argmax()] for j in range(12)]
        print(f"{nm:7s} peaks {np.round(pk,2).tolist()}  -> {sum(4.7 <= x <= 6.1 for x in pk)}/12")
    print("  10 s windows, mean LS power of R/L hip_pitch+knee act, 3-9 Hz (freq/power/mean gyro deg/s)")
    frb = np.arange(3, 9, 0.01)
    for nm in ("vib g1", "1020 g1", "Kd1.5", "B g0.6"):
        df = load(runs[nm]); t0 = (df.t - df.t.iloc[0]).to_numpy(); act = cols(df, "act")
        g = np.degrees(np.linalg.norm(df[["obs/gyro/x", "obs/gyro/y", "obs/gyro/z"]].to_numpy(), axis=1)); out = []
        for w in range(int(t0[-1] // 10)):
            m = (t0 >= 10 * w) & (t0 < 10 * w + 10)
            if m.sum() < 60: continue
            P = np.mean([ls(t0[m], act[m, j], frb) for j in (0, 3, 6, 9)], axis=0); k = P.argmax()
            out.append(f"{10*w}:{frb[k]:.1f}/{P[k]:.2f}/{g[m].mean():.0f}")
        print(f"  {nm}: " + " ".join(out))
    print("== C11 amplitude / lag of 5 Hz line (R hip_pitch, vib-check): coherent line over whole run vs 10 s windows in burst")
    df = load(STAND["vib"]); t0 = (df.t - df.t.iloc[0]).to_numpy(); tgt = cols(df, "tgt"); pos = cols(df, "pos")
    for jn, j in (("R hip_pitch", 0), ("R knee", 3), ("L knee", 9)):
        yt, yp = ma_detrend(t0, tgt[:, j]), ma_detrend(t0, pos[:, j]); At, _ = sinfit(t0, yt, 5.10); Ap, _ = sinfit(t0, yp, 5.10)
        ph = []; Ab = []; Bb = []
        for a in np.arange(50, 100.1, 10):
            m = (t0 >= a) & (t0 < a + 10); a1, p1 = sinfit(t0[m], yt[m], 5.10); a2, p2 = sinfit(t0[m], yp[m], 5.10); ph.append(np.exp(1j * (p2 - p1))); Ab.append(a1); Bb.append(a2)
        lag = (-np.degrees(np.angle(np.mean(ph))) / 360 / 5.10 * 1e3) % (1e3 / 5.10)
        print(f"  {jn}: whole-run line amp tgt {At:.2f} pos {Ap:.2f} (ratio {Ap/At:.2f}); in-burst 10 s fit tgt {np.mean(Ab):.2f} pos {np.mean(Bb):.2f}; lag of pos behind tgt {lag:.0f} ms (mod {1e3/5.10:.0f})")
    print("== C12 alias leakage: unit sine at any f0 (1-100 Hz, |f0-f_obs|>1) sampled at logged t, LS power at f_obs +-0.3 Hz")
    for nm, fobs in (("vib g1", 5.10), ("1020 g1", 5.70)):
        df = load(runs[nm]); t0 = (df.t - df.t.iloc[0]).to_numpy(); tt = t0[t0 < 80]; fine = np.arange(fobs - 0.3, fobs + 0.3, 0.01)
        w = max((lombscargle(tt, np.sin(2 * np.pi * f0 * tt + .3) - np.sin(2 * np.pi * f0 * tt + .3).mean(), 2 * np.pi * fine, normalize=True).max(), f0)
                for f0 in np.arange(1, 100, 0.1) if abs(f0 - fobs) > 1)
        print(f"  {nm}: max leakage {w[0]:.3f} at f0={w[1]:.1f} Hz (in-burst observed LS power 0.15-0.65; 1.0 would be a pure alias)")


def c14():
    print("== C14 gyro zeros")
    for nm, f in STAND.items():
        g = load(f)[["obs/gyro/x", "obs/gyro/y", "obs/gyro/z"]].to_numpy(); fg = np.r_[False, (g[1:] == g[:-1]).all(1)]; nz = np.abs(g[g != 0])
        live = np.degrees(np.abs(g[~fg])).ravel()
        print(f"{nm:5s} frozen values all exactly 0: {not (np.abs(g[fg]) > 0).any()}; min nonzero |gyro| {nz.min():.6f} rad/s = {np.degrees(nz.min()):.4f} deg/s; live-row |gyro| counts 0.01-0.05: {((live>0)&(live<=0.05)).sum()}, 0.05-0.1: {((live>0.05)&(live<=0.1)).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="/tmp/ckpt/terrain10_2048_model_37400.pt")
    ap.add_argument("--data", default="Expr_Data"); ap.add_argument("--only", default="")
    a = ap.parse_args(); D = a.data; sel = set(a.only.split(",")) if a.only else None
    for k, fn in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4_c5), ("C6", c6), ("C7", c7), ("C8", lambda: c8_c9(a.ckpt)), ("C10", c10_c12), ("C14", c14)):
        if sel is None or k in sel: fn()
