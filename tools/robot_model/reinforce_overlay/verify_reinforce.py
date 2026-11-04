#!/usr/bin/env python3
"""Verify `<name>_footv5_d455_reinf_loop` (URDF + both MJCF copies) against its base `..._footv5_d455_loop` and the Fusion snapshot.
usage (mjlab venv): verify_reinforce.py [--snapshot DIR]; exit code 1 on any FAIL. Writes <OUT>_verify.json next to the URDF."""
import argparse, json, sys
from pathlib import Path
import mujoco, numpy as np, trimesh

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "pygmalion_locomotion/assets/pygmalion_v2"
XMLS = REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
BASE = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_loop"
OUT = BASE.replace("_loop", "_reinf_loop")
R_SIM = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
BASE_CAD = np.array([0.0, 70.0, 60.0])
rows = []


def check(name, ok, detail):
    rows.append(dict(check=name, ok=bool(ok), detail=detail))
    print(("PASS " if ok else "FAIL ") + name + "  " + detail)


def load_urdf(name):
    t = (ASSETS / f"{name}.urdf").read_text()
    i = t.index(">", t.index("<robot")) + 1
    t = t[:i] + f'\n  <mujoco><compiler fusestatic="false" discardvisual="false" meshdir="{ASSETS}"/></mujoco>' + t[i:]
    m = mujoco.MjModel.from_xml_string(t)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def load_mjcf(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def inertial(m, body):
    b = m.body(body).id
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.body_iquat[b]); R = R.reshape(3, 3)
    return float(m.body_mass[b]), m.body_ipos[b].copy(), R @ np.diag(m.body_inertia[b]) @ R.T


def in_base(m, d, body):
    b, base = m.body(body).id, m.body("base_link").id
    return d.xmat[base].reshape(3, 3).T @ (d.xpos[b] - d.xpos[base])


def robot_com(m, d):
    return (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum() - d.xpos[m.body("base_link").id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(REPO / "tools/robot_model/fusion_snapshots/reinforce_20261104"))
    a = ap.parse_args()
    rep = json.load(open(ASSETS / f"{OUT}_report.json"))
    cad = json.load(open(Path(a.snapshot) / "reinforce_cad.json"))["parts"]
    models = {"urdf": (load_urdf(BASE), load_urdf(OUT)),
              "mjcf_repo": (load_mjcf(ASSETS / f"{BASE}.xml"), load_mjcf(ASSETS / f"{OUT}.xml")),
              "mjcf_mjlab": (load_mjcf(XMLS / f"{BASE}.xml"), load_mjcf(XMLS / f"{OUT}.xml"))}
    P = rep["plates"]
    hip, knee, camt = rep["mass_g"]["hip"] / 1000, rep["mass_g"]["knee"] / 1000, rep["mass_g"]["camera_total"] / 1000
    old_mount = 0.0619659
    for tag, ((mb, db), (mn, dn)) in models.items():
        # 1. link masses
        for s in "RL":
            for link, dm in (("hip_roll_link", hip), ("shin_link", knee)):
                n = f"{s}_{link}"
                got = mn.body_mass[mn.body(n).id] - mb.body_mass[mb.body(n).id]
                check(f"{tag}: {n} mass +{dm*1000:.1f} g", abs(got - dm) < 1e-7, f"{got*1000:.4f} g")
        mount = mn.body_mass[mn.body("pelvis_camera_mount").id]
        d455 = mn.body_mass[mn.body("d455_link").id]
        check(f"{tag}: Pelvis_Camera = 166 g (D455 + bracket)", abs(mount + d455 - camt) < 1e-7, f"{d455*1000:.3f} + {mount*1000:.3f} = {(mount+d455)*1000:.3f} g")
        # 2. total
        want = 2 * (hip + knee) + (mount - old_mount)
        got = mn.body_mass.sum() - mb.body_mass.sum()
        check(f"{tag}: total mass change", abs(got - want) < 1e-6, f"{got:.6f} kg (want {want:.6f}); {mb.body_mass.sum():.4f} -> {mn.body_mass.sum():.4f} kg")
        # 3. every other body untouched
        touched = {f"{s}_{l}" for s in "RL" for l in ("hip_roll_link", "shin_link")} | {"pelvis_camera_mount"}
        worst = 0.0
        for i in range(mb.nbody):
            nm = mb.body(i).name
            if nm in touched:
                continue
            j = mn.body(nm).id
            worst = max(worst, abs(mb.body_mass[i] - mn.body_mass[j]), np.abs(mb.body_ipos[i] - mn.body_ipos[j]).max(), np.abs(mb.body_inertia[i] - mn.body_inertia[j]).max(), np.abs(mb.body_pos[i] - mn.body_pos[j]).max())
        check(f"{tag}: other {mb.nbody - len(touched)} bodies unchanged (mass, COM, inertia, pose)", worst < 1e-12, f"worst {worst:.2e}")
        check(f"{tag}: body count and joints", mn.nbody == mb.nbody and mn.njnt == mb.njnt and mn.nu == mb.nu, f"nbody {mn.nbody} njnt {mn.njnt} nu {mn.nu}")
        # 4. whole-robot COM shift (zero pose, base_link frame)
        c0, c1 = robot_com(mb, db), robot_com(mn, dn)
        rep.setdefault("com_shift_mm", {})[tag] = dict(before=(c0 * 1000).round(2).tolist(), after=(c1 * 1000).round(2).tolist())
        print(f"     robot COM in base_link {tag}: {np.round(c0*1000,2)} -> {np.round(c1*1000,2)} mm")
        # 5. plate COM in the base_link frame equals the CAD COM (R side direct, L side y-mirrored)
        for s in "RL":
            for key, link, ent in (("hip", "hip_roll_link", cad["hip_reinforcement"]), ("knee", "shin_link", cad["knee_reinforcement"])):
                want = R_SIM @ (np.array(ent["c"]) * 10 - BASE_CAD) / 1000
                if s == "L":
                    want = want * np.array([1, -1, 1])
                b = mn.body(f"{s}_{link}").id
                Rb = dn.xmat[b].reshape(3, 3); base = mn.body("base_link").id
                com_base = dn.xmat[base].reshape(3, 3).T @ (dn.xpos[b] + Rb @ np.array(P[f"{s}_{key}"]["com_link_m"]) - dn.xpos[base])
                check(f"{tag}: {s}_{key} plate COM matches CAD (base_link frame)", np.abs(com_base - want).max() < 2e-6, f"{np.round(com_base*1000,2)} vs {np.round(want*1000,2)} mm")
        # 6. combined link tensor = base + plate (parallel axis, independent of the builder)
        for s in "RL":
            for link, key in (("hip_roll_link", "hip"), ("shin_link", "knee")):
                n = f"{s}_{link}"
                m0, c0_, I0 = inertial(mb, n); m1, c1_, I1 = inertial(mn, n)
                p = P[f"{s}_{key}"]; mp, cp, Ip = p["mass_kg"], np.array(p["com_link_m"]), np.array(p["inertia_com_kg_m2"])
                Ish = I0 + Ip + m0 * ((c0_ - c1_) @ (c0_ - c1_) * np.eye(3) - np.outer(c0_ - c1_, c0_ - c1_)) + mp * ((cp - c1_) @ (cp - c1_) * np.eye(3) - np.outer(cp - c1_, cp - c1_))
                check(f"{tag}: {n} inertia = base + plate", np.abs(Ish - I1).max() < 1e-7 and abs((m0 * c0_ + mp * cp) / (m0 + mp) - c1_).max() < 1e-8, f"max |dI| {np.abs(Ish - I1).max():.2e} kg m2 (MuJoCo's own eigen-decomposition of the stored tensor is good to ~1e-6 relative)")
        w = np.linalg.eigvalsh(inertial(mn, "R_hip_roll_link")[2])
        # 7. physical sanity of every changed inertia
        ok = all((lambda w: w.min() > 0 and w[0] + w[1] >= w[2] * (1 - 1e-9))(np.linalg.eigvalsh(inertial(mn, n)[2])) for n in touched)
        check(f"{tag}: changed inertias are physical", ok, "positive, triangle inequality")
        # 8. no NaN after stepping
        d2 = mujoco.MjData(mn)
        for _ in range(200):
            mujoco.mj_step(mn, d2)
        check(f"{tag}: 200 steps without NaN", np.isfinite(d2.qpos).all() and np.isfinite(d2.qacc).all(), f"max|qacc| {np.abs(d2.qacc).max():.3g}")
    # 9. URDF vs MJCF: the DELTA each overlay made to the changed bodies (mass, first moment, inertia about the link origin). The two base
    #    models differ slightly in their own link values (docs/185), so only the overlay's delta is compared.
    def about_origin(m, n):
        mass, c, I = inertial(m, n)
        return mass, mass * c, I + mass * (c @ c * np.eye(3) - np.outer(c, c))
    (mbu, _), (mnu, _) = models["urdf"]
    for tag in ("mjcf_repo", "mjcf_mjlab"):
        (mbm, _), (mnm, _) = models[tag]
        worst = 0.0
        for n in [f"{s}_{l}" for s in "RL" for l in ("hip_roll_link", "shin_link")] + ["pelvis_camera_mount"]:
            du_ = [x - y if not isinstance(x, float) else x - y for x, y in zip(about_origin(mnu, n), about_origin(mbu, n))]
            dm_ = [x - y for x, y in zip(about_origin(mnm, n), about_origin(mbm, n))]
            if n == "pelvis_camera_mount":                      # the old bracket differs between the two bases only through rounding
                pass
            worst = max(worst, abs(du_[0] - dm_[0]), np.abs(du_[1] - dm_[1]).max(), np.abs(du_[2] - dm_[2]).max())
        check(f"urdf vs {tag}: overlay delta of the 5 changed bodies (mass, first moment, inertia)", worst < 2e-7, f"worst {worst:.2e}")
    # 10. mesh placement: plate meshes in the link frame, moved to base_link, fill the CAD bounding box (R) / its y-mirror (L)
    for s in "RL":
        for key, link, ent in (("hip", "hip_roll_link", cad["hip_reinforcement"]), ("knee", "shin_link", cad["knee_reinforcement"])):
            t = trimesh.load(ASSETS / "meshes_reinforce" / f"{s}_{key}_reinforcement.stl", process=False)
            (mb, db), (mn, dn) = models["urdf"]
            b = mn.body(f"{s}_{link}").id; base = mn.body("base_link").id
            V = (dn.xmat[b].reshape(3, 3) @ np.asarray(t.vertices).T).T + dn.xpos[b] - dn.xpos[base]
            lo, hi = np.array(ent["bb"][:3]), np.array(ent["bb"][3:])
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            cs = (R_SIM @ (corners - BASE_CAD).T).T / 1000
            if s == "L":
                cs = cs * np.array([1, -1, 1])
            err = max(np.abs(V.min(0) - cs.min(0)).max(), np.abs(V.max(0) - cs.max(0)).max())
            check(f"{s}_{key} plate mesh bbox matches CAD bbox (base_link frame)", err < 1.5e-3, f"{err*1000:.2f} mm (CAD bbox rounding 0.01 mm; tessellation < 1 mm)")
    # 11. how much of each plate lies inside the existing link hull / outside it
    (mb, db), (mn, dn) = models["urdf"]
    for s in "R":
        for key, link, hull in (("hip", "hip_roll_link", "hip_roll_link_hull"), ("knee", "shin_link", "shin_hull")):
            t = trimesh.load(ASSETS / "meshes_reinforce" / f"{s}_{key}_reinforcement.stl", process=False)
            h = trimesh.load(Path(REPO / "tools/robot_model/fusion_snapshots/v30_inspection/meshes_armfix") / f"{hull}.stl", process=False)
            inside = h.convex_hull.contains(np.asarray(t.vertices))
            dist = trimesh.proximity.signed_distance(h.convex_hull, np.asarray(t.vertices))
            print(f"     {s}_{key} plate vs existing {hull}: {inside.mean()*100:.0f} % of vertices inside the hull, max outside {max(0, -dist.min())*1000:.1f} mm")
            rep.setdefault("plate_vs_hull", {})[f"{s}_{key}"] = dict(inside_pct=round(float(inside.mean()) * 100, 1), max_outside_mm=round(float(max(0, -dist.min())) * 1000, 2))
    nfail = sum(not r["ok"] for r in rows)
    json.dump(dict(rows=rows, extra={k: rep[k] for k in ("com_shift_mm", "plate_vs_hull") if k in rep}), open(ASSETS / f"{OUT}_verify.json", "w"), indent=1)
    print(f"{len(rows) - nfail}/{len(rows)} checks passed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
