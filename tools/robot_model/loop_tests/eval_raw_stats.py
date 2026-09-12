"""Impact / gait statistics over the EVALUATOR's 32 episodes, not one rollout.

Every judgement in this project that rested on a single deterministic rollout has been wrong
at least once (seven times in the 2026-08-26 session): duty, stride/s and tracking from
`apex_recheck` split 0.57 vs 0.77 for the *same* configuration at different batch sizes, and
`impact_probe`'s 30 touchdowns all come from one trajectory, so a rollout stuck in a bad mode
poisons all of them. The built-in evaluator holds one command for a full 20 s episode across
32 independent envs; this reads its raw.npz and computes the same quantities per episode, so
a per-episode spread is available instead of a single number.

  .venv/bin/python3 eval_raw_stats.py <raw.npz> [BW]
"""
import sys
import numpy as np

f = sys.argv[1]
BW = float(sys.argv[2]) if len(sys.argv) > 2 else 346.8
d = np.load(f, allow_pickle=True)
warm = int(d['warmup_steps'])
jn = [str(x) for x in d['joint_names']]
knee = [i for i, n in enumerate(jn) if 'knee' in n]
ff = d['foot_force']          # [T, E, F] or [T, E, F, 3]
if ff.ndim == 4:
    ff = np.linalg.norm(ff, axis=-1)
fc = d['foot_contact'].astype(bool)
jp = d['joint_pos']
v = d['root_lin_vel_b']
cmd = d['command'][:2]
T, E = ff.shape[0], ff.shape[1]
dt = float(d['time'][1] - d['time'][0])

per = []
for e in range(E):
    on = fc[warm:, e, :]                      # [T', F]
    F = ff[warm:, e, :] / BW
    duty = float(on.mean())
    # touchdowns per foot
    peaks, rates, strides = [], [], 0
    for k in range(on.shape[1]):
        edge = np.diff(on[:, k].astype(int))
        td = np.where(edge == 1)[0] + 1
        strides += len(td)
        for t in td:
            w = F[t:t + 12, k]                # 240 ms window
            if len(w) < 3:
                continue
            peaks.append(float(w.max()))
            rates.append(float(np.max(np.diff(w)) / dt))
    kn = np.degrees(np.abs(jp[warm:, e, knee]))
    st = on.any(axis=1)
    per.append(dict(
        duty=duty,
        stride_per_s=strides / (on.shape[0] * dt) / max(on.shape[1], 1) * 2,
        peak_BW=float(np.median(peaks)) if peaks else np.nan,
        rate_BWs=float(np.median(rates)) if rates else np.nan,
        knee_stance=float(kn[st].mean()) if st.any() else np.nan,
        err=float(np.linalg.norm(v[warm:, e, :2] - cmd, axis=1).mean()),
    ))

keys = ['duty', 'stride_per_s', 'peak_BW', 'rate_BWs', 'knee_stance', 'err']
print(f'{f.split("/")[-4]}  ({E} episodes)')
print(f"  {'metric':14s}{'median':>10s}{'p10':>9s}{'p90':>9s}{'spread':>9s}")
for k in keys:
    a = np.array([p[k] for p in per], float)
    a = a[~np.isnan(a)]
    if not len(a):
        continue
    print(f'  {k:14s}{np.median(a):10.3f}{np.percentile(a,10):9.3f}{np.percentile(a,90):9.3f}'
          f'{np.percentile(a,90)-np.percentile(a,10):9.3f}')
