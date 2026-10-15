#!/usr/bin/env python3
"""Policy-independent quasi-static joint torques of the Pygmalion legs on slopes and stairs (docs/169 s8-4).

No policy, no dynamics: the robot is put in a statically balanced pose with MuJoCo kinematics, and the
torque each actuated joint needs to hold it is solved from the inverse dynamics (qfrc_bias with qacc = 0),
with the closed ankle loop (crank A/B -> push rods -> ankle) in the model, so the crank torques are the real
motor torques, not an equivalent ankle torque. Same machinery as compare_robot_gains.py (B1): model
analysis/out/stand10_v9caps_m8998_s1_model.mjb (LegOnly 23.63 kg, AB ankle), default pose from env.yaml.

Method (CPU only):
  * Slope: the slope is replaced by a flat floor with TILTED GRAVITY, g = 9.81 * (-sin a, 0, -cos a) for an
    uphill-facing robot (a > 0 uphill, a < 0 downhill). Statics are identical, the soles stay flat on the
    floor, and "body upright" means the pelvis z axis is anti-parallel to gravity (pelvis pitched by a).
  * Pose: least-squares over the stance leg (hip pitch, knee, ankle pitch [, hip roll, ankle roll]) and the pelvis
    position so that (1) the sole is flat on the floor / step top, (2) the pelvis is at the standing height
    above the stance foot (minus the riser for a step), (3) the gravity projection of the whole-body centre
    of mass falls on the stance-foot centre (+ com_dx along the foot).
  * Torque: qfrc_bias = S^T tau + J_foot^T w + J_loop^T lambda, solved in the least-squares sense (w = ground
    wrench on the stance foot, lambda = loop constraint forces). Centre of pressure and joint-range violations
    are reported.
  * Step-down landing: SAME pose as the step-up with the same riser, gravity multiplied by 1.5 / 2.0 (a
    decelerating body in its own frame feels (1 + a/g) g), plus a flat-ground reference at 2.0 body weights.

Run (any python with mujoco + scipy; the mjlab venv has both):
  CUDA_VISIBLE_DEVICES="" nice -n 10 ../../mujoco-sim/mjlab/.venv/bin/python3 quasi_static_terrain_torque.py
Output: results/quasi_static_terrain.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_robot_gains as cg  # noqa: E402

G = 9.81
SOLE_X = (-0.1747, 0.0853)  # sole extent in x of the foot box geoms (world x at the default pose), m


def _pitch_roll(R, R0):
    Rr = R @ R0.T
    return math.atan2(Rr[0, 2], Rr[2, 2]), math.atan2(-Rr[1, 2], Rr[2, 2])


class Scenario:
    def __init__(self, case, stance, *, slope_deg=0.0, foot_h=0.0, single=False, com_dx=0.0, h_drop=0.0,
                 lift_swing=False, lean_deg=0.0, foot_y=None, relax=()):
        """stance: 'L' or 'R' (the loaded leg; for double stance the other leg mirrors it)."""
        self.case, self.m = case, case.model
        m = self.m
        self.s, self.o = stance, ("R" if stance == "L" else "L")
        self.alpha = math.radians(slope_deg)
        self.foot_h, self.single, self.com_dx, self.lift = foot_h, single, com_dx, lift_swing
        self.g_hat = np.array([-math.sin(self.alpha), 0.0, -math.cos(self.alpha)])
        self.lean = math.radians(lean_deg)  # trunk (pelvis) lean forward of vertical
        pit = self.alpha + self.lean
        self.base_quat = np.array([math.cos(pit / 2), 0.0, math.sin(pit / 2), 0.0])
        d0 = mujoco.MjData(m)
        d0.qpos[:] = case.qpos_placed
        mujoco.mj_forward(m, d0)
        self.fid = {s: m.body(case.foot_body[s]).id for s in "LR"}
        self.R0 = {s: d0.xmat[self.fid[s]].reshape(3, 3).copy() for s in "LR"}
        self.q_def = case.qpos_placed.copy()
        self.foot_xy0 = cg._foot_center(case, d0, stance)[:2].copy()
        if foot_y is not None:  # stance foot placed closer to the midline (stairs: the other foot is on another step)
            self.foot_xy0[1] = foot_y
        self.hb0 = float(d0.qpos[2] - cg._foot_center(case, d0, stance)[2])  # pelvis height above foot centre
        self.hb = self.hb0 - h_drop
        self.floor = case.floor_geom
        self.geoms = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.fid[s]
                          and (m.geom_contype[g] or m.geom_conaffinity[g])] for s in "LR"}
        n = lambda t: f"robot/{self.s}_{t}"  # noqa: E731
        self.jn = dict(hp=n("hip_pitch_joint"), hr=n("hip_roll_joint"), kn=n("knee_joint"),
                       ap=n("ankle_pitch_joint"), ar=n("ankle_roll_joint"))
        self.adr = {k: cg.jqadr(m, v) for k, v in self.jn.items()}
        self.lim = {k: tuple(m.jnt_range[m.joint(v).id]) for k, v in self.jn.items()}
        # mirror factors L<->R for the double-stance copy: other = q0_o + mf * (q_s - q0_s)
        o = self.o
        self.mirror = {}
        for k, t in (("hp", "hip_pitch_joint"), ("hr", "hip_roll_joint"), ("kn", "knee_joint"),
                     ("ap", "ankle_pitch_joint"), ("ar", "ankle_roll_joint")):
            a_s = self.q_def[cg.jqadr(m, f"robot/{self.s}_{t}")]
            a_o = self.q_def[cg.jqadr(m, f"robot/{o}_{t}")]
            self.mirror[k] = (a_o / a_s) if abs(a_s) > 1e-6 else (-1.0 if k in ("hr", "ar") else 1.0)
        for k in relax:  # explore beyond the hard joint range to see how far the pose would have to go
            self.lim[k] = (-math.radians(60.0), math.radians(60.0))
        self.lateral = single
        self.keys = ["hp", "kn", "ap"] + (["hr", "ar"] if self.lateral else [])

    # --- pose ---------------------------------------------------------------------------
    def build(self, u):
        m, case = self.m, self.case
        nl = len(self.keys)
        d = mujoco.MjData(m)
        d.qpos[:] = self.q_def
        d.qpos[0:3] = [u[nl], u[nl + 1] if self.lateral else 0.0, u[nl + 2 if self.lateral else nl + 1]]
        d.qpos[3:7] = self.base_quat
        vals = dict(zip(self.keys, u[:nl]))
        for k, v in vals.items():
            d.qpos[self.adr[k]] = v
        o = self.o
        if self.single:
            # swing leg: flexed off the ground, same recipe as compare_robot_gains.single_leg_pose
            hj, kj = case.jt(case.hip_joint[o]), case.jt(case.knee_joint[o])
            d.qpos[cg.jqadr(m, hj.name)] = hj.q0 + math.copysign(0.35, hj.q0)
            d.qpos[cg.jqadr(m, kj.name)] = kj.q0 + math.copysign(0.70, kj.q0)
        elif not self.lift:
            for k, t in (("hp", "hip_pitch_joint"), ("hr", "hip_roll_joint"), ("kn", "knee_joint"),
                         ("ap", "ankle_pitch_joint"), ("ar", "ankle_roll_joint")):
                qs0 = self.q_def[cg.jqadr(m, f"robot/{self.s}_{t}")]
                qv = vals.get(k, qs0)
                d.qpos[cg.jqadr(m, f"robot/{o}_{t}")] = self.q_def[cg.jqadr(m, f"robot/{o}_{t}")] + self.mirror[k] * (qv - qs0)
        else:  # trailing leg lifted (stairs): same flexion recipe as the swing leg
            hj, kj = case.jt(case.hip_joint[o]), case.jt(case.knee_joint[o])
            d.qpos[cg.jqadr(m, hj.name)] = hj.q0 + math.copysign(0.35, hj.q0)
            d.qpos[cg.jqadr(m, kj.name)] = kj.q0 + math.copysign(0.70, kj.q0)
        self._close_loops(d)
        return d

    def _close_loops(self, d, tol=1e-10, iters=40):
        """Gauss-Newton on the crank / rod hinges of BOTH legs at once (the equality residual of one leg is not
        zero while the other leg's ankle has just been moved, so compare_robot_gains.close_loop per leg fails)."""
        m, case = self.m, self.case
        names = [n for sd in "LR" for n in cg._loop_dofs(case, sd)]
        cols = [cg.jdadr(m, n) for n in names]
        qadr = [cg.jqadr(m, n) for n in names]
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            _, J, r = cg.eq_rows(m, d)
            if np.abs(r).max() < tol:
                return
            dq, *_ = np.linalg.lstsq(J[:, cols], -r, rcond=None)
            for qa, v in zip(qadr, dq):
                d.qpos[qa] += v
        mujoco.mj_forward(m, d)
        _, J, r = cg.eq_rows(m, d)
        assert np.abs(r).max() < 1e-6, f"loop closure failed {np.abs(r).max()}"

    def residual(self, u):
        try:
            d = self.build(u)
        except AssertionError:  # loop does not close at this trial point (ankle outside its reachable set)
            return np.full(5 + (3 if self.lateral else 0), 10.0)
        m = self.m
        fc = cg._foot_center(self.case, d, self.s)
        sole = cg.min_floor_distance(m, d, self.geoms[self.s], self.floor)
        pit, rol = _pitch_roll(d.xmat[self.fid[self.s]].reshape(3, 3), self.R0[self.s])
        base = d.qpos[0:3]
        hb = -float((base - fc) @ self.g_hat)
        com = d.subtree_com[int(m.jnt_bodyid[0])]
        t = (self.foot_h - com[2]) / self.g_hat[2]
        P = com + t * self.g_hat  # gravity projection of the CoM onto the support plane
        r = [fc[0] - self.foot_xy0[0], sole - (self.foot_h + 0.0005), pit, hb - self.hb,
             P[0] - (fc[0] + self.com_dx)]
        if self.lateral:
            r += [fc[1] - self.foot_xy0[1], rol, P[1] - fc[1]]
        return np.array(r)

    def solve(self):
        nl = len(self.keys)
        base0 = self.q_def[0:3]
        lo = [self.lim[k][0] for k in self.keys] + [-2.0] * (3 if self.lateral else 2)
        hi = [self.lim[k][1] for k in self.keys] + [2.0] * (3 if self.lateral else 2)

        def jac(u):  # absolute-step forward differences (relative steps are useless near zero)
            f0 = self.residual(u)
            J = np.zeros((len(f0), len(u)))
            for i in range(len(u)):
                up = np.array(u, dtype=float)
                up[i] += 1e-5
                J[:, i] = (self.residual(up) - f0) / 1e-5
            return J

        best = None
        # multi-start: progressively deeper leg flexion (stance 'L': hip +, knee -, ankle +, sum 0 for a flat foot)
        for kb in (0.0, 0.4, 0.8, 1.2, 1.6):
            u0 = []
            for k in self.keys:
                q = self.q_def[self.adr[k]]
                u0.append(q + {"hp": 0.6 * kb, "kn": -kb, "ap": 0.4 * kb}.get(k, 0.0))
            u0 = [min(max(v, l + 1e-6), h - 1e-6) for v, l, h in zip(u0, lo, hi)]
            u0 = np.array(u0 + ([base0[0], base0[1], base0[2]] if self.lateral else [base0[0], base0[2]]))
            sol = least_squares(self.residual, u0, jac=jac, bounds=(lo, hi), xtol=1e-14, ftol=1e-14,
                                gtol=1e-14, max_nfev=200)
            if best is None or np.abs(sol.fun).max() < np.abs(best.fun).max():
                best = sol
            if np.abs(best.fun).max() < 1e-8:
                break
        self.u = best.x
        self.res = float(np.abs(best.fun).max())
        self.d = self.build(best.x)
        return self

    # --- statics ------------------------------------------------------------------------
    def torques(self, g_scale=1.0):
        m, d, case = self.m, self.d, self.case
        saved = m.opt.gravity.copy()
        m.opt.gravity[:] = G * g_scale * self.g_hat
        try:
            mujoco.mj_forward(m, d)
            ai = cg.act_index(case)
            nv = m.nv
            cols, names = [], []
            for n in ai:
                c = np.zeros(nv)
                c[cg.jdadr(m, n)] = 1.0
                cols.append(c)
                names.append(n)
            _, Jeq, _ = cg.eq_rows(m, d)
            Jf = {}
            for sd in ("L", "R"):
                jp = np.zeros((3, nv))
                jr = np.zeros((3, nv))
                mujoco.mj_jacBody(m, d, jp, jr, self.fid[sd])
                Jf[sd] = np.vstack([jp, jr])
            if self.single:
                W = Jf[self.s].T  # 6 wrench components on the stance foot
                nw = 6
            else:  # double stance: a free wrench on each foot, min-norm split (symmetric pose -> equal split)
                W = np.hstack([Jf[self.s].T, Jf[self.o].T])
                nw = 12
            A = np.column_stack(cols + [W] + ([Jeq.T] if Jeq.shape[0] else []))
            b = d.qfrc_bias.copy()
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            resid = float(np.abs(A @ x - b).max())
            tau = {n.split("/")[1]: float(x[i]) for i, n in enumerate(names)}
            w = x[len(names):len(names) + nw]
        finally:
            m.opt.gravity[:] = saved
        # centre of pressure of the stance foot along the foot (x), in the world frame
        F, M = w[:3], w[3:6]  # wrench on the stance foot (the first foot in the double-stance split)
        origin = d.xpos[self.fid[self.s]]
        rz = self.foot_h - origin[2]
        cop_x = origin[0] + (rz * F[0] - M[1]) / F[2] if abs(F[2]) > 1e-6 else float("nan")
        # in the tilted frame the foot lies flat on the floor plane, so x_s is the sole coordinate
        return dict(tau=tau, resid=resid, force_z_N=float(F[2]), force_x_N=float(F[0]), cop_x=float(cop_x),
                    cop_in_sole=bool(SOLE_X[0] - 1e-3 <= cop_x <= SOLE_X[1] + 1e-3))

    def summary(self, g_scale=1.0, load_share=1.0):
        t = self.torques(g_scale)
        s = self.s
        tau = {k: v for k, v in t["tau"].items()}
        legs = {"hip_pitch": tau[f"{s}_hip_pitch_joint"], "hip_roll": tau[f"{s}_hip_roll_joint"],
                "hip_yaw": tau[f"{s}_hip_yaw_joint"], "knee": tau[f"{s}_knee_joint"],
                "crank_A": tau[f"{s}_crank_A_joint"], "crank_B": tau[f"{s}_crank_B_joint"]}
        q = {k: math.degrees(self.d.qpos[self.adr[k]] - self.q_def[self.adr[k]]) for k in self.keys}
        viol = []
        for jid in range(self.m.njnt):
            if self.m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE and self.m.jnt_limited[jid]:
                v = float(self.d.qpos[self.m.jnt_qposadr[jid]])
                lo, hi = self.m.jnt_range[jid]
                if v < lo - 1e-3 or v > hi + 1e-3:
                    viol.append([self.m.joint(jid).name, round(v, 3), float(lo), float(hi)])
        pelvis_up = math.degrees(math.acos(float(-(self.d.xmat[int(self.m.jnt_bodyid[0])].reshape(3, 3)[:, 2] @ self.g_hat))))
        return dict(tau=legs, tau_abs_max=float(max(abs(v) for v in legs.values())),
                    joint_delta_from_default_deg=q, pose_resid=self.res, statics_resid=t["resid"],
                    stance_force_N=t["force_z_N"], cop_x_m=t["cop_x"], cop_in_sole=t["cop_in_sole"],
                    pelvis_tilt_from_vertical_deg=pelvis_up, range_violations=viol,
                    abs_joint_deg={k: math.degrees(float(self.d.qpos[self.adr[k]])) for k in self.keys},
                    knee_deg=math.degrees(abs(self.d.qpos[self.adr["kn"]])),
                    hip_pitch_deg=math.degrees(abs(self.d.qpos[self.adr["hp"]])))


def main():
    case = cg.build_pyg(55.0, 2.5, "qs", "quasi-static")
    cg.prepare(case)
    m = case.model
    mass = cg.total_mass(m)
    out = dict(mass_kg=mass, weight_N=mass * G, model="analysis/out/stand10_v9caps_m8998_s1_model.mjb",
               effort_limit_Nm={"hip_pitch": 120, "hip_roll": 120, "knee": 120, "hip_yaw": 60, "crank": 60},
               scenarios={})

    def add(name, **kw):
        g_scales = kw.pop("g_scales", [1.0])
        h0 = kw.pop("h_drop", 0.0)
        extra, lean, relaxed = 0.0, 0.0, False
        # 1) strict joint ranges, pelvis lowered by up to 3 cm (numerical slack of the standing height)
        for extra in (0.0, 0.01, 0.02, 0.03):
            sc = Scenario(case, "L", h_drop=h0 + extra, **kw).solve()
            if sc.res < 1e-5:
                break
        # 2) not reachable inside the hard ranges (hip roll adduction cap 25 deg, ankle dorsiflexion 30 deg, knee):
        #    solve again with hip roll and ankle pitch limits opened to +-60 deg and flag it
        if sc.res >= 1e-5:
            relaxed = True
            for extra in [round(0.01 * i, 2) for i in range(0, 11)]:
                sc = Scenario(case, "L", h_drop=h0 + extra, relax=("hr", "ap"), **kw).solve()
                if sc.res < 1e-5:
                    break
        kw["h_drop"] = h0
        kw["extra_pelvis_lowering_m"] = round(extra, 3)
        kw["joint_limits_relaxed"] = relaxed
        if sc.res > 1e-5:
            print(f"  WARNING {name}: pose residual {sc.res:.2e} even with relaxed limits")
        for gs in g_scales:
            key = name if gs == 1.0 else f"{name}__x{gs:g}"
            r = sc.summary(gs)
            r["params"] = dict(kw, g_scale=gs)
            out["scenarios"][key] = r
            tt = r["tau"]
            print(f"{key:<38} low {extra:4.2f} relax {int(relaxed)} hipP {tt['hip_pitch']:7.1f} knee {tt['knee']:7.1f} crkA {tt['crank_A']:7.1f} "
                  f"crkB {tt['crank_B']:7.1f}  CoP x {r['cop_x_m']:+.3f} in-sole {r['cop_in_sole']}  "
                  f"viol {len(r['range_violations'])} res {sc.res:.1e}/{r['statics_resid']:.1e}", flush=True)
        return sc

    # (0)/(1) standing on both feet, flat and slopes (each leg carries half the weight)
    add("stand_double_flat")
    for a in (10, 20, 30):
        add(f"stand_double_up{a}", slope_deg=a)
        add(f"stand_double_down{a}", slope_deg=-a)
    # (2) single-leg stance
    add("stance_single_flat", single=True)
    add("stance_single_up20", single=True, slope_deg=20)
    add("stance_single_down20", single=True, slope_deg=-20)
    # (3)/(4) stair step-up (whole weight on the front leg on the step, trailing leg lifted) and step-down landing
    for h in (0.10, 0.15, 0.20, 0.25):
        cm = int(round(h * 100))
        add(f"stepup_{cm}cm_com0", single=True, foot_h=h, h_drop=h, com_dx=0.0)
        add(f"stepup_{cm}cm_com-10", single=True, foot_h=h, h_drop=h, com_dx=-0.10)
        # landing: same pose as the step-up, 1.5 and 2.0 body weights
        add(f"landing_{cm}cm", single=True, foot_h=h, h_drop=h, com_dx=0.0, g_scales=[1.5, 2.0])
    # flat-ground drop landing in the default single-leg stance, 2 body weights (reference)
    add("landing_flat_default", single=True, g_scales=[1.5, 2.0])
    p = Path(__file__).resolve().parent / "results" / "quasi_static_terrain.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
