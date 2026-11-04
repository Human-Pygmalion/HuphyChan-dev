#!/usr/bin/env python3
"""Verify `..._reinf_alparts2_loop` (URDF + both MJCF copies) against round 1 (`..._alparts_loop`) with a calculation that does NOT use the
massprops aggregation the builder used: per part, mass, first moment and inertia about the link origin from the part's own CAD numbers (new: Fusion
now; old: the dump the shipped model was built from), summed per link. Exit code 1 on any FAIL; writes <OUT>_verify.json."""
import json, sys
from pathlib import Path
import mujoco, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reinforce_overlay"))
import verify_reinforce as V  # noqa: E402
import build_al_round2 as R2  # noqa: E402
B = R2.B
check, rows = V.check, V.rows
LINK_OF = {"HipRoll2YawA": "hip_roll_link", "HipRoll2YawB": "hip_roll_link", "Yaw2KneePlateA": "thigh", "Knee2Ankle_IdleSide_1": "shin",
           "Knee2Ankle_RotorSide_1": "shin", "AnkleUniversalJointCore": "ankle_pitch_link",
           "AnkleFeetPillowA": "foot", "AnkleFeetPillowB": "foot"}


def part_about_origin(e, O):
    """(mass, first moment, inertia about O) in sim axes, link frame; e = part dict (m kg, c cm, I kg cm2 about the CAD root origin)."""
    m, c = e["m"], np.array(e["c"]) * 10.0
    xx, yy, zz, xy, yz, xz = np.array(e["I"]) * 100.0                       # kg mm2 about the root origin
    Io = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    Ic = Io - m * (c @ c * np.eye(3) - np.outer(c, c))
    d = c - O
    IO = Ic + m * (d @ d * np.eye(3) - np.outer(d, d))
    Rm = B.R_SIM
    return m, m * (Rm @ d) / 1000.0, Rm @ IO @ Rm.T * 1e-6


def main():
    OUT, BASE = R2.OUT, R2.BASE
    d0 = json.load(open(R2.DUMP)); r2 = R2.load_round2()["parts"]
    exp = {}
    for name, lk in LINK_OF.items():
        zc = r2[name]["c"][2] * 10.0                      # CAD COM z in mm: the model's own rule for the ankle region (massprops_fusion ANKLE_SPLIT)
        if name.startswith(("AnkleFeet", "AnkleUniversal")):
            want = "ankle_pitch_link" if abs(zc + 800.0) < 0.6 else ("foot" if zc < -800.0 else "shin")
            assert want == lk, (name, zc, want, lk)
        O = R2.LINKS[lk][1]
        new = part_about_origin(r2[name], O)
        old = part_about_origin(d0[R2.find(d0, name)], O)
        t = exp.setdefault(lk, [0.0, np.zeros(3), np.zeros((3, 3))])
        t[0] += new[0] - old[0]; t[1] += new[1] - old[1]; t[2] += new[2] - old[2]
    models = {"urdf": (V.load_urdf(BASE), V.load_urdf(OUT)),
              "mjcf_repo": (V.load_mjcf(V.ASSETS / f"{BASE}.xml"), V.load_mjcf(V.ASSETS / f"{OUT}.xml")),
              "mjcf_mjlab": (V.load_mjcf(V.XMLS / f"{BASE}.xml"), V.load_mjcf(V.XMLS / f"{OUT}.xml"))}
    touched = {f"{s}_{R2.LINKS[lk][0]}" for s in "RL" for lk in R2.LINKS}
    for tag, ((mb, db), (mn, dn)) in models.items():
        tot = 0.0
        for s in "RL":
            for lk, (mname, O) in R2.LINKS.items():
                n = f"{s}_{mname}"
                m0, c0, I0 = V.inertial(mb, n); m1, c1, I1 = V.inertial(mn, n)
                em, ef, eI = exp[lk]
                if s == "L":
                    ef, eI = B.MIRROR @ ef, B.MIRROR @ eI @ B.MIRROR
                tot += m1 - m0
                check(f"{tag}: {n} mass change = sum over its parts (new V x 2.70 - old)", abs((m1 - m0) - em) < 3e-6, f"+{(m1-m0)*1000:.3f} g (want +{em*1000:.3f} g)")
                f_new = m1 * c1 - m0 * c0
                check(f"{tag}: {n} first moment change matches the parts' CAD COMs", np.abs(f_new - ef).max() < 2e-8, f"max diff {np.abs(f_new-ef).max():.2e} kg m")
                Io1 = I1 + m1 * (c1 @ c1 * np.eye(3) - np.outer(c1, c1)); Io0 = I0 + m0 * (c0 @ c0 * np.eye(3) - np.outer(c0, c0))
                check(f"{tag}: {n} inertia (about link origin) change matches the parts' CAD tensors", np.abs((Io1 - Io0) - eI).max() < 2e-7, f"max diff {np.abs((Io1-Io0)-eI).max():.2e} kg m2 of {np.abs(eI).max():.2e}")
        check(f"{tag}: total mass change", abs((mn.body_mass.sum() - mb.body_mass.sum()) - tot) < 1e-9 and abs(tot * 1000 - sum(v[0] for v in exp.values()) * 2000) < 0.01,
              f"+{tot*1000:.3f} g; {mb.body_mass.sum():.4f} -> {mn.body_mass.sum():.4f} kg")
        worst = 0.0
        for i in range(mb.nbody):
            nm = mb.body(i).name
            if nm in touched:
                continue
            j = mn.body(nm).id
            worst = max(worst, abs(mb.body_mass[i] - mn.body_mass[j]), np.abs(mb.body_ipos[i] - mn.body_ipos[j]).max(), np.abs(mb.body_inertia[i] - mn.body_inertia[j]).max(), np.abs(mb.body_pos[i] - mn.body_pos[j]).max())
        check(f"{tag}: other {mb.nbody - len(touched)} bodies unchanged (cranks, plates, camera, feet ...)", worst < 1e-12, f"worst {worst:.2e}")
        check(f"{tag}: joints / constraints / actuators unchanged", (mn.njnt, mn.neq, mn.nu, mn.nbody) == (mb.njnt, mb.neq, mb.nu, mb.nbody), f"njnt {mn.njnt} neq {mn.neq}")
        ok = all((lambda w: w.min() > 0 and w[0] + w[1] >= w[2] * (1 - 1e-9))(np.linalg.eigvalsh(V.inertial(mn, n)[2])) for n in touched)
        check(f"{tag}: changed inertias are physical", ok, "positive, triangle inequality")
        d2 = mujoco.MjData(mn)
        for _ in range(200):
            mujoco.mj_step(mn, d2)
        cl = float(np.abs(d2.efc_pos[: mn.neq * 3]).max()) if mn.neq else 0.0
        check(f"{tag}: 200 steps, no NaN, ankle loops closed", np.isfinite(d2.qpos).all() and np.isfinite(d2.qacc).all() and cl < 1e-3, f"closure {cl*1000:.4f} mm, max|qacc| {np.abs(d2.qacc).max():.3g}")
    # the revised core is drawn at the CAD position (both legs): bbox of the mesh in the ankle_pitch_link frame vs the CAD part bbox
    import trimesh
    (mb, db), (mn, dn) = models["urdf"]
    for s in "RL":
        t = trimesh.load(B.ASSETS / B.MESHDIR / f"{s}_ankle_core_revised.stl", process=False)
        b = mn.body(f"{s}_ankle_pitch_link").id; base = mn.body("base_link").id
        Vv = (dn.xmat[b].reshape(3, 3) @ np.asarray(t.vertices).T).T + dn.xpos[b] - dn.xpos[base]
        bb = np.array(R2.load_round2()["parts"]["AnkleUniversalJointCore"]["bb"])
        lo, hi = bb[:3], bb[3:]
        cs = (B.R_SIM @ (np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - B.BASE_CAD).T).T / 1000
        if s == "L":
            cs = cs * np.array([1, -1, 1])
        err = max(np.abs(Vv.min(0) - cs.min(0)).max(), np.abs(Vv.max(0) - cs.max(0)).max())
        check(f"{s} revised core mesh bbox matches the CAD bbox (base_link frame)", err < 1.5e-3, f"{err*1000:.2f} mm")
    nfail = sum(not r["ok"] for r in rows)
    json.dump(dict(rows=rows), open(B.ASSETS / f"{OUT}_verify.json", "w"), indent=1)
    print(f"{len(rows) - nfail}/{len(rows)} checks passed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
