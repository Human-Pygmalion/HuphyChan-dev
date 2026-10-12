#!/usr/bin/env python3
"""Compare the shared joints of two MJCF models in their compiled zero pose: world axis, anchor
(relative to the root body), range, parent/child body, armature/friction/damping, loop equalities.

  .venv/bin/python3 tools/robot_model/compare_joint_axes.py A.xml B.xml
"""
def load(p):
    m = mujoco.MjModel.from_xml_path(p); d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0; mujoco.mj_kinematics(m, d)
    J = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if m.jnt_type[j] not in (2, 3): continue  # slide/hinge only
        b = m.jnt_bodyid[j]
        J[n] = dict(axis=d.xaxis[j].copy(), anchor=d.xanchor[j].copy() - d.xpos[1], rng=m.jnt_range[j].copy(),
                    lim=bool(m.jnt_limited[j]), body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                    parent=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]),
                    arm=m.dof_armature[m.jnt_dofadr[j]], fr=m.dof_frictionloss[m.jnt_dofadr[j]], dmp=m.dof_damping[m.jnt_dofadr[j]])
    return m, J
a, b = sys.argv[1], sys.argv[2]
ma, A = load(a); mb, B = load(b)
common = [n for n in A if n in B]
print(f"A joints {len(A)}, B joints {len(B)}, common {len(common)}")
print("only in A:", [n for n in A if n not in B]); print("only in B:", [n for n in B if n not in A])
bad = 0
print(f"{'joint':22s} {'axis A (world)':>24s} {'dot':>6s} {'anchor diff mm':>14s} {'range A deg':>16s} {'range B deg':>16s}  body/parent same")
for n in common:
    x, y = A[n], B[n]
    dot = float(np.dot(x['axis'], y['axis'])); da = 1000*np.linalg.norm(x['anchor']-y['anchor'])
    ra, rb = np.degrees(x['rng']), np.degrees(y['rng'])
    same_bp = (x['body'], x['parent']) == (y['body'], y['parent'])
    flag = "" if (dot > 0.9999 and da < 0.5 and np.allclose(ra, rb, atol=0.01) and same_bp and x['lim'] == y['lim']) else "  <-- DIFF"
    bad += bool(flag)
    print(f"{n:22s} {np.array2string(x['axis'], precision=3, suppress_small=True):>24s} {dot:6.3f} {da:14.3f} {str(np.round(ra,2)):>16s} {str(np.round(rb,2)):>16s}  {same_bp}{flag}")
    if (x['arm'], x['fr'], x['dmp']) != (y['arm'], y['fr'], y['dmp']):
        print(f"   dynamics differ: A arm/fric/damp {x['arm']:.4f}/{x['fr']:.3f}/{x['dmp']:.4f}  B {y['arm']:.4f}/{y['fr']:.3f}/{y['dmp']:.4f}")
print("joints with differences:", bad)
# equality constraints (ankle loop)
def eqs(m): return sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.eq_obj1id[i]), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.eq_obj2id[i]), tuple(np.round(m.eq_data[i,:6],5))) for i in range(m.neq))
ea, eb = eqs(ma), eqs(mb)
print("loop equalities identical:", ea == eb, len(ea), len(eb))
