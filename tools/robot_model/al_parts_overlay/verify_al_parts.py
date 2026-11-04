#!/usr/bin/env python3
"""Verify `..._reinf_alparts_loop` (URDF + both MJCF copies) against its base `..._reinf_loop` and the Fusion crank snapshot.
Exit code 1 on any FAIL. Writes <OUT>_verify.json next to the URDF."""
import json, sys
from pathlib import Path
import mujoco, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reinforce_overlay"))
import verify_reinforce as V  # noqa: E402  (loaders, inertial(), in-base helpers)
import build_al_parts as C  # noqa: E402

BASE, OUT = C.BASE, C.OUT
rows = V.rows
check = V.check


def main():
    cad = json.load(open(C.B.REPO / "tools/robot_model/fusion_snapshots/alparts_20261104/cranks_cad.json"))["parts"]
    rep = json.load(open(V.ASSETS / f"{OUT}_report.json"))
    models = {"urdf": (V.load_urdf(BASE), V.load_urdf(OUT)),
              "mjcf_repo": (V.load_mjcf(V.ASSETS / f"{BASE}.xml"), V.load_mjcf(V.ASSETS / f"{OUT}.xml")),
              "mjcf_mjlab": (V.load_mjcf(V.XMLS / f"{BASE}.xml"), V.load_mjcf(V.XMLS / f"{OUT}.xml"))}
    touched = {f"{s}_crank_{t}" for s in "RL" for t in "AB"} | {f"{s}_hip_roll_link" for s in "RL"}
    mount = json.load(open(C.B.REPO / "tools/robot_model/fusion_snapshots/alparts_20261104/mount_cad.json"))["part"]
    d_mount = mount["v"] * C.RHO_AL / 1000 - C.MOUNT_OLD_KG
    for tag, ((mb, db), (mn, dn)) in models.items():
        tot = 0.0
        for t in "AB":
            want = (cad[f"Crank_{t}"]["v"] * C.RHO_AL + cad[f"Crank_{t}_Cover"]["v"] * C.RHO_PRINTED) / 1000
            for s in "RL":
                n = f"{s}_crank_{t}"
                got = mn.body_mass[mn.body(n).id]
                tot += got - mb.body_mass[mb.body(n).id]
                check(f"{tag}: {n} mass = V_crank x 2.70 + V_cover x 0.8874", abs(got - want) < 5e-6, f"{got*1000:.3f} g (want {want*1000:.3f} g)")
        for s in "RL":
            n = f"{s}_hip_roll_link"
            got = mn.body_mass[mn.body(n).id] - mb.body_mass[mb.body(n).id]
            tot += got
            check(f"{tag}: {n} gains (V_mount x 2.70 - 67.0 g)", abs(got - d_mount) < 5e-6, f"+{got*1000:.3f} g (want +{d_mount*1000:.3f} g)")
        check(f"{tag}: total mass change", abs((mn.body_mass.sum() - mb.body_mass.sum()) - tot) < 1e-9 and abs(tot - rep["total_mass_change_g"] / 1000) < 2e-5,
              f"{tot*1000:.3f} g; {mb.body_mass.sum():.4f} -> {mn.body_mass.sum():.4f} kg")
        worst = 0.0
        for i in range(mb.nbody):
            nm = mb.body(i).name
            if nm in touched:
                continue
            j = mn.body(nm).id
            worst = max(worst, abs(mb.body_mass[i] - mn.body_mass[j]), np.abs(mb.body_ipos[i] - mn.body_ipos[j]).max(), np.abs(mb.body_inertia[i] - mn.body_inertia[j]).max(), np.abs(mb.body_pos[i] - mn.body_pos[j]).max())
        check(f"{tag}: other {mb.nbody - 6} bodies unchanged", worst < 1e-12, f"worst {worst:.2e}")
        check(f"{tag}: joints, equality constraints, actuators unchanged", (mn.njnt, mn.neq, mn.nu, mn.nbody) == (mb.njnt, mb.neq, mb.nu, mb.nbody), f"njnt {mn.njnt} neq {mn.neq}")
        # COM of each crank link in the base_link frame vs the CAD COM of crank (Al) + cover (printed)
        for t in "AB":
            c, cov = cad[f"Crank_{t}"], cad[f"Crank_{t}_Cover"]
            ma, mc = c["v"] * C.RHO_AL, cov["v"] * C.RHO_PRINTED
            com_cad = (ma * np.array(c["c"]) + mc * np.array(cov["c"])) / (ma + mc)             # cm
            want = V.R_SIM @ (com_cad * 10 - V.BASE_CAD) / 1000
            for s in "RL":
                w = want * np.array([1, -1, 1]) if s == "L" else want
                n = f"{s}_crank_{t}"; b = mn.body(n).id; base = mn.body("base_link").id
                got = dn.xmat[base].reshape(3, 3).T @ (dn.xipos[b] - dn.xpos[base])
                check(f"{tag}: {n} COM matches CAD (base_link frame, zero pose)", np.abs(got - w).max() < 2e-6, f"{np.round(got*1000,2)} vs {np.round(w*1000,2)} mm")
        for s in "RL":                                   # hip_roll_link COM = mass-weighted mean of the old COM and the CAD COM of the mount's extra mass
            n = f"{s}_hip_roll_link"; bb_, bn_ = mb.body(n).id, mn.body(n).id
            cm = V.R_SIM @ (np.array(mount["c"]) * 10 - C.HIP_ORIGIN) / 1000
            cm = cm * np.array([1, -1, 1]) if s == "L" else cm
            m0 = mb.body_mass[bb_]
            want = (m0 * mb.body_ipos[bb_] + d_mount * cm) / (m0 + d_mount)
            check(f"{tag}: {n} COM = old COM + mount extra mass at its CAD COM", np.abs(mn.body_ipos[bn_] - want).max() < 1e-8,
                  f"{np.round(mn.body_ipos[bn_]*1000,3)} mm (moved {np.round((mn.body_ipos[bn_]-mb.body_ipos[bb_])*1000,3)} mm)")
        ok = all((lambda w: w.min() > 0 and w[0] + w[1] >= w[2] * (1 - 1e-9))(np.linalg.eigvalsh(V.inertial(mn, n)[2])) for n in touched)
        check(f"{tag}: crank inertias are physical", ok, "positive, triangle inequality")
        # closed loop: constraint violation at the zero pose and after 200 steps must not have got worse
        def closure(m, d):
            return float(np.abs(d.efc_pos[: m.neq * 3]).max()) if m.neq else 0.0
        before = closure(mb, db)
        d2 = mujoco.MjData(mn)
        for _ in range(200):
            mujoco.mj_step(mn, d2)
        check(f"{tag}: 200 steps, no NaN, loop closure still closed", np.isfinite(d2.qpos).all() and np.isfinite(d2.qacc).all() and closure(mn, d2) < max(1e-3, 3 * before),
              f"closure {closure(mn, d2)*1000:.4f} mm (base at rest {before*1000:.4f} mm), max|qacc| {np.abs(d2.qacc).max():.3g}")
    # URDF vs MJCF: delta of the changed links
    def about(m, n):
        mass, c, I = V.inertial(m, n)
        return mass, mass * c, I + mass * (c @ c * np.eye(3) - np.outer(c, c))
    (mbu, _), (mnu, _) = models["urdf"]
    for tag in ("mjcf_repo", "mjcf_mjlab"):
        (mbm, _), (mnm, _) = models[tag]
        worst = 0.0
        for n in sorted(touched):
            du = [x - y for x, y in zip(about(mnu, n), about(mbu, n))]
            dm = [x - y for x, y in zip(about(mnm, n), about(mbm, n))]
            worst = max(worst, abs(du[0] - dm[0]), np.abs(du[1] - dm[1]).max(), np.abs(du[2] - dm[2]).max())
        check(f"urdf vs {tag}: overlay delta of the 6 changed links (mass, first moment, inertia)", worst < 2e-7, f"worst {worst:.2e}")
    nfail = sum(not r["ok"] for r in rows)
    json.dump(dict(rows=rows), open(V.ASSETS / f"{OUT}_verify.json", "w"), indent=1)
    print(f"{len(rows) - nfail}/{len(rows)} checks passed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
