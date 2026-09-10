"""RP-mode ankle: position-dependent torque envelope and reflected motor parameters from
the closed-loop (2-RSU) model, by IK/FK on the loop closure.

For every (pitch, roll) node of the ankle ROM grid, per leg:
  IK        crank angles theta_c = (A, B) that close both rods:  |pin_i(theta_c_i) - ball_i(p, r)| = L_i
  Jacobian  Jc = d theta_c / d (p, r)  from the closure equations (implicit differentiation,
            partials by central differences on the MuJoCo FK)
  torques   tau_ankle = Jc^T tau_crank, |tau_crank_i| <= 60 N*m (RS03 peak)
            -> axis extents (pitch alone / roll alone), inscribed box, and the matrices the
            RP actuator uses online: M = Jc^-T (ankle torque -> crank torque) and Jc^T (back)
  reflected J_ankle = Jc^T diag(J_m) Jc, b_ankle likewise, tau_f,ankle = tc * sum_i |Jc_i,axis|

Output: pygmalion_locomotion/assets/pygmalion_v2/ankle_rp_envelope.json, docs/img/ankle_rp_envelope.png
Usage: ankle_rp_envelope.py [--tag=pygmalion_v3_printed_loop]   (mjlab .venv python)
"""
import json, os, sys
import numpy as np
import mujoco

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
XMLS = f'{REPO}/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls'
OUT = f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2/ankle_rp_envelope.json'
IMG = f'{REPO}/docs/img/ankle_rp_envelope.png'
# RS03 = both ankle motors. J/b/tc are read from the sim's single source of truth
# (pygmalion_constants.MOTOR_MEAS) so this file cannot drift away from it again.
#
# ★ 2026-09-11 correction. Until today this line hard-coded J=0.015265, b=0.022342,
# tc=0.285370 - the 2026-08-23 numbers - while MOTOR_MEAS had already been updated to the
# re-measured 0.0638 / 0.0252 / 0.251 on 2026-09-10. The ankles were therefore training on
# a rotor inertia 4.2x too low while hip_yaw, the same motor family, used the measured one.
# Inconsistent inside one model, and silent, because nothing checks the two against
# each other. Reading MOTOR_MEAS here removes the second copy.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'pyg_const',
    f'{REPO}/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py')
try:
    _c = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_c)
    _m = _c.MOTOR_MEAS['RS03']
    _J, _B, _TC = _m['armature'], _m['damping'], _m['frictionloss']
except Exception as _e:                                     # noqa: BLE001
    raise SystemExit(f'could not read MOTOR_MEAS from pygmalion_constants: {_e}')
MOTOR = dict(name='RS03', peak=60.0, rated=20.0, J=_J, b=_B, tc=_TC)
print(f'motor params from MOTOR_MEAS: J={_J} b={_B} tc={_TC}')
PITCH = np.radians(np.arange(-50, 30 + 1e-9, 5.0))       # design ROM, docs/88
ROLL = np.radians(np.arange(-20, 20 + 1e-9, 5.0))
H = 1e-4


def main():
    tag = next((a.split('=')[1] for a in sys.argv if a.startswith('--tag=')), 'pygmalion_v3_printed_loop')
    m = mujoco.MjModel.from_xml_path(f'{XMLS}/{tag}.xml')
    d = mujoco.MjData(m)
    jq = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    out = dict(tag=tag, motor=MOTOR, grid=dict(pitch_deg=np.degrees(PITCH).round(3).tolist(), roll_deg=np.degrees(ROLL).round(3).tolist()),
               legs={}, note='tau_ankle = JcT @ tau_crank; tau_crank = M @ tau_ankle; arrays indexed [pitch][roll]')
    for s in 'LR':
        qp, qr = jq(f'{s}_ankle_pitch_joint'), jq(f'{s}_ankle_roll_joint')
        qc = {t: jq(f'{s}_crank_{t}_joint') for t in 'AB'}
        rod = {t: bid(f'{s}_rod_{t}') for t in 'AB'}
        ball = {t: sid(f'{s}_ball_{t}') for t in 'AB'}
        shin = bid(f'{s}_shin_link')

        def fk(p, r, cA, cB):
            d.qpos[:] = 0; d.qpos[3] = 1
            d.qpos[qp], d.qpos[qr], d.qpos[qc['A']], d.qpos[qc['B']] = p, r, cA, cB
            mujoco.mj_kinematics(m, d)
            Rs = d.xmat[shin].reshape(3, 3)             # express everything in the shin frame
            pin = {t: Rs.T @ (d.xpos[rod[t]] - d.xpos[shin]) for t in 'AB'}
            bl = {t: Rs.T @ (d.site_xpos[ball[t]] - d.xpos[shin]) for t in 'AB'}
            return pin, bl
        pin0, bl0 = fk(0, 0, 0, 0)
        L = {t: float(np.linalg.norm(pin0[t] - bl0[t])) for t in 'AB'}

        def f(t, p, r, c):          # closure residual of rod t
            pin, bl = fk(p, r, c if t == 'A' else 0.0, c if t == 'B' else 0.0)
            return float(np.sum((pin[t] - bl[t]) ** 2) - L[t] ** 2)

        def ik(t, p, r, near):
            cs = np.linspace(-1.3, 1.3, 261)
            vals = np.array([f(t, p, r, c) for c in cs])
            roots = []
            for i in range(len(cs) - 1):
                if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
                    a, b = cs[i], cs[i + 1]
                    fa = vals[i]
                    for _ in range(50):          # bisection
                        c = 0.5 * (a + b); fc = f(t, p, r, c)
                        if fa * fc <= 0: b = c
                        else: a, fa = c, fc
                    roots.append(0.5 * (a + b))
            if not roots:
                return None
            return min(roots, key=lambda c: abs(c - near))

        nP, nR = len(PITCH), len(ROLL)
        crank = np.full((nP, nR, 2), np.nan); Jc = np.full((nP, nR, 2, 2), np.nan)
        order = sorted([(i, j) for i in range(nP) for j in range(nR)], key=lambda ij: PITCH[ij[0]] ** 2 + ROLL[ij[1]] ** 2)
        i0, j0 = order[0]
        for (i, j) in order:
            p, r = PITCH[i], ROLL[j]
            # continuation: start from the nearest already-solved node
            done = [(a, b) for (a, b) in order[:order.index((i, j))] if not np.isnan(crank[a, b, 0])]
            near = crank[min(done, key=lambda ab: (PITCH[ab[0]] - p) ** 2 + (ROLL[ab[1]] - r) ** 2)] if done else np.zeros(2)
            ok = True
            for k, t in enumerate('AB'):
                c = ik(t, p, r, near[k])
                if c is None:
                    ok = False; break
                crank[i, j, k] = c
            if not ok:
                continue
            for k, t in enumerate('AB'):
                c = crank[i, j, k]
                dfc = (f(t, p, r, c + H) - f(t, p, r, c - H)) / (2 * H)
                dfp = (f(t, p + H, r, c) - f(t, p - H, r, c)) / (2 * H)
                dfr = (f(t, p, r + H, c) - f(t, p, r - H, c)) / (2 * H)
                Jc[i, j, k, 0] = -dfp / dfc
                Jc[i, j, k, 1] = -dfr / dfc
        JcT = np.transpose(Jc, (0, 1, 3, 2))
        M = np.full_like(Jc, np.nan); ext = np.full((nP, nR, 2), np.nan); box = np.full((nP, nR, 2), np.nan)
        for i in range(nP):
            for j in range(nR):
                if np.isnan(Jc[i, j]).any():
                    continue
                Mi = np.linalg.inv(JcT[i, j])            # crank torque from ankle torque
                M[i, j] = Mi
                ext[i, j] = [MOTOR['peak'] / np.abs(Mi[:, 0]).max(), MOTOR['peak'] / np.abs(Mi[:, 1]).max()]
                sc = min(MOTOR['peak'] / (abs(Mi[k, 0]) * ext[i, j, 0] + abs(Mi[k, 1]) * ext[i, j, 1]) for k in range(2))
                box[i, j] = ext[i, j] * sc
        c0 = Jc[i0, j0]
        Jref = c0.T @ np.diag([MOTOR['J']] * 2) @ c0
        bref = c0.T @ np.diag([MOTOR['b']] * 2) @ c0
        tcref = MOTOR['tc'] * np.abs(c0).sum(0)
        nan = int(np.isnan(crank[..., 0]).sum())
        out['legs'][s] = dict(rod_length_m={t: round(L[t], 6) for t in 'AB'}, crank_rad=np.round(crank, 6).tolist(), Jc=np.round(Jc, 6).tolist(),
                              JcT=np.round(JcT, 6).tolist(), M=np.round(M, 6).tolist(), tau_extent=np.round(ext, 4).tolist(),
                              tau_box=np.round(box, 4).tolist(), unreachable_nodes=nan,
                              reflected_center=dict(armature=np.round(np.diag(Jref), 6).tolist(), armature_full=np.round(Jref, 6).tolist(),
                                                    damping=np.round(np.diag(bref), 6).tolist(), frictionloss=np.round(tcref, 5).tolist(),
                                                    ratio_crank_per_ankle=np.round(c0, 4).tolist()))
        print(f'{s}: rod L {L}  unreachable {nan}/{nP*nR}  centre Jc (crank per ankle) =\n{np.round(c0, 4)}')
        print(f'   reflected at centre: armature {np.diag(Jref).round(5)} damping {np.diag(bref).round(5)} frictionloss {tcref.round(4)}')
        print(f'   torque extents at centre: pitch {ext[i0, j0, 0]:.1f} roll {ext[i0, j0, 1]:.1f} N*m; inscribed box {box[i0, j0].round(1)}')
        print(f'   extents over the ROM: pitch {np.nanmin(ext[..., 0]):.1f}..{np.nanmax(ext[..., 0]):.1f}  roll {np.nanmin(ext[..., 1]):.1f}..{np.nanmax(ext[..., 1]):.1f}')
    json.dump(out, open(OUT, 'w'))
    print('->', OUT)
    # figure (left leg)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Lg = out['legs']['L']; ext = np.array(Lg['tau_extent']); box = np.array(Lg['tau_box']); crank = np.array(Lg['crank_rad'])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), dpi=110)
    P, Rr = np.degrees(PITCH), np.degrees(ROLL)
    for ax, Z, ttl in zip(axes[:3], (ext[..., 0], ext[..., 1], box[..., 0]),
                          ('pitch torque limit [N*m] (roll torque 0)', 'roll torque limit [N*m] (pitch torque 0)', 'inscribed box: pitch limit [N*m] (simultaneous)')):
        im = ax.imshow(Z.T, origin='lower', extent=[P[0], P[-1], Rr[0], Rr[-1]], aspect='auto', cmap='viridis')
        ax.set_xlabel('ankle pitch [deg]'); ax.set_ylabel('ankle roll [deg]'); ax.set_title(ttl, fontsize=9); fig.colorbar(im, ax=ax)
    ax = axes[3]
    for (i, j, col) in ((len(P) // 2 + 0, len(Rr) // 2, 'k'), (0, len(Rr) // 2, 'tab:red'), (len(P) - 1, len(Rr) // 2, 'tab:blue'), (len(P) // 2, 0, 'tab:green')):
        JT = np.array(Lg['JcT'][i][j])
        if np.isnan(JT).any():
            continue
        corners = np.array([[sa * 60, sb * 60] for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))])
        poly = corners @ JT.T
        ax.plot(poly[:, 0], poly[:, 1], color=col, label=f'pitch {P[i]:+.0f} roll {Rr[j]:+.0f}')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('ankle pitch torque [N*m]'); ax.set_ylabel('ankle roll torque [N*m]'); ax.set_title('feasible ankle torque (crank |tau| <= 60)', fontsize=9); ax.legend(fontsize=7); ax.set_aspect('equal')
    fig.suptitle(f'{tag} L ankle: RP-mode torque envelope from the 2-RSU loop (RS03 peak 60 N*m per crank)', fontsize=10)
    fig.tight_layout(); fig.savefig(IMG); print('->', IMG)


if __name__ == '__main__':
    main()
