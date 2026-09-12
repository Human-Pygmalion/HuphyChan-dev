"""Turn measured rotor-face normals into a per-joint PASS/FAIL against the user's sign convention.

Convention (user, reconfirmed 2026-09-02): look straight at the actuator ROTOR face - the one
carrying 6 M4 threaded holes and 3 alignment pins - from outside the motor. CLOCKWISE is POSITIVE.

The algebra, spelled out because getting it backwards inverts every joint. The viewpoint was
re-audited after the old render put the camera on the back side of the flange:
  n            = OUTWARD normal of the rotor face (points away from the motor body)
  viewer       = is outside, so n points TOWARD the viewer / OUT OF the screen
  view ray     = -n (from the viewer into the scene)
  right-hand   = a rotation vector along -n appears CLOCKWISE to that viewer
  therefore    = CLOCKWISE (the user's motor +) is a rotation vector along -n

  raw_rotor_vs_stator_axis = -n

That is still not the URDF joint sign.  URDF q is CHILD relative to PARENT, while the
motor convention describes ROTOR relative to STATOR.  An explicit attachment parity is
therefore mandatory:

  attachment_parity = +1  when child is fixed to rotor and parent to stator
                    = -1  when child is fixed to stator and parent to rotor
  required_child_vs_parent_axis = attachment_parity * (-n)

Never default this value.  A fused actuator body/face normal cannot prove it.

The superseded 2026-09-01 grid used +n as both the camera direction and required axis.  That
placed the camera behind the rotor face and inverted every displayed CW/CCW verdict.

A joint MATCHES when the model's own axis, expressed in the same frame, is parallel to -n
(dot > 0) and is INVERTED when antiparallel (dot < 0).

Input : rotor_faces.json  {motor_key: {joint, n_root: [x,y,z], ...}}  (from the CAD measurement)
Output: motor_sign_convention.json  {joint_name: {matches, axis, n_base, dot, note}}
        consumed by rom_sweep_video.py to put the verdict on screen.

  .venv/bin/python3 tools/robot_model/motor_sign_check.py [--xml=robot.xml]
"""
import json
import os
import sys

import numpy as np
import mujoco

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
XML = (f'{REPO}/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/'
       'prototype-tempmass-motormeasured-armfix_v30_loop.xml')
OUT = f'{REPO}/tools/robot_model/motor_sign_convention.json'

# CAD root -> URDF/base_link is a +90 deg rotation about Z followed by a translation. A normal
# is a direction, so only the rotation applies: (x, y, z)_cad -> (-y, x, z)_base.
def cad_dir_to_base(v):
    x, y, z = v
    return np.array([-y, x, z], dtype=float)


def main():
    src = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--faces=')),
               f'{REPO}/tools/robot_model/rotor_faces.json')
    xml = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--xml=')), XML)
    if not os.path.exists(src):
        print(f'!! {src} missing - run the CAD rotor-face measurement first')
        return 2
    rotor = {k: v for k, v in json.load(open(src)).items() if not k.startswith('_')}
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    if m.nq >= 7 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    mujoco.mj_forward(m, d)

    out = {}
    print(f'{"joint":26s} {"n (base)":>20s} {"required +":>20s} {"model axis":>20s}  verdict')
    for key, r in rotor.items():
        joints = r['joint'] if isinstance(r.get('joint'), list) else [r.get('joint')]
        n_cad = np.array(r['n_root'], dtype=float)
        n_cad /= np.linalg.norm(n_cad)
        n_base = cad_dir_to_base(n_cad)
        expanded = []
        for jn in [j for j in joints if j]:
            expanded.append((jn, n_base, key, r.get('evidence', ''),
                             r.get('attachment_parity'), r.get('side_mapping_verified')))
            # Do not synthesize the opposite side by reflecting (or copying) a motor normal.
            # A purchased motor is not a mirrored solid and installation orientation must be
            # proved by that side's CAD occurrence/photo or by a low-current hardware test.
        for jn, n_joint, motor_key, evidence, attachment_parity, side_mapping_verified in expanded:
            try:
                jid = m.joint(jn).id
            except Exception:
                print(f'{jn:26s} -- not in the model, skipped')
                continue
            # xaxis is the neutral-pose world/base direction.  m.jnt_axis is body-local and
            # only happens to be comparable with CAD when every ancestor is identity.
            axis = np.array(d.xaxis[jid], dtype=float)
            axis /= np.linalg.norm(axis)
            rotor_vs_stator = -n_joint
            if attachment_parity not in (-1, 1) or side_mapping_verified is not True:
                need = None
                dot = None
                missing = []
                if attachment_parity not in (-1, 1):
                    missing.append('attachment')
                if side_mapping_verified is not True:
                    missing.append('physical-side mapping')
                verdict, matches = 'UNKNOWN - ' + ' + '.join(missing) + ' not proved', None
            else:
                need = attachment_parity * rotor_vs_stator
                dot = float(np.dot(axis, need))
            if dot is not None and abs(dot) < 0.9:
                verdict, matches = 'AXIS NOT PARALLEL - check frames', None
            elif dot is not None:
                matches = dot > 0
                verdict = 'match' if matches else 'INVERTED'
            out[jn] = dict(matches=matches,
                           dot=None if dot is None else round(dot, 4),
                           axis=[round(v, 3) for v in axis.tolist()],
                           n_base=[round(v, 3) for v in n_joint.tolist()],
                           n_root=[round(v, 3) for v in n_cad.tolist()],
                           motor=motor_key, note=evidence,
                           evidence_level=('CAD rotor face + attachment mapping'
                                           if attachment_parity in (-1, 1)
                                           else 'CAD rotor face only; attachment UNKNOWN'),
                           attachment_parity=attachment_parity,
                           side_mapping_verified=side_mapping_verified,
                           raw_rotor_vs_stator_axis=[round(v, 3) for v in rotor_vs_stator.tolist()],
                           required_child_vs_parent_axis=(None if need is None else
                               [round(v, 3) for v in need.tolist()]),
                           convention=('face-on from outside: outward normal points toward '
                                       'viewer; raw rotor-vs-stator CW axis = -normal; '
                                       'URDF sign additionally requires attachment parity'))
            fmt = lambda v: '[' + ' '.join(f'{x:+.2f}' for x in v) + ']'
            need_text = '[UNKNOWN]' if need is None else fmt(need)
            print(f'{jn:26s} {fmt(n_joint):>20s} {need_text:>20s} {fmt(axis):>20s}  {verdict}')
    out['_meta'] = dict(model=os.path.abspath(xml), faces=os.path.abspath(src),
                        right_side=('not synthesized; requires right-side CAD/photo or '
                                    'low-current hardware evidence'),
                        supersedes='2026-09-01 grid that viewed the flange from its back side',
                        convention=('raw motor positive = rotor CW face-on from outside, so '
                                    'rotor-vs-stator axis=-normal; model sign requires explicit '
                                    'rotor/stator-to-child/parent attachment parity'),
                        attachment_gate=('missing attachment_parity must produce UNKNOWN; '
                                         'never assume child=rotor'),
                        side_gate=('CAD physical side, simulator L/R name, and CAN side must '
                                   'agree; missing verification must produce UNKNOWN'))
    json.dump(out, open(OUT, 'w'), indent=1)
    vals = [v for k, v in out.items() if not k.startswith('_')]
    n_bad = sum(1 for v in vals if v['matches'] is False)
    n_ok = sum(1 for v in vals if v['matches'] is True)
    print(f'\n{n_ok} match, {n_bad} inverted, {len(vals) - n_ok - n_bad} undetermined -> {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
