"""Domain-randomization ranges for link mass, COM and inertia, from the measured uncertainty.

Instead of guessing "+-10 % mass, +-2 cm COM" for every link, this propagates what we
actually know about each BODY through the same aggregation that builds the robot
(massprops_fusion.collect / aggregate). Every body gets a relative mass uncertainty from
its provenance, thousands of robots are sampled, and the spread of each LINK's mass, COM
and inertia is what the randomizer should cover.

Per-body uncertainty model (1 sigma, relative mass; the tensor scales with the mass):

  class            sigma   basis
  printed, weighed 0.03    its own v5 ratio is in the CAD; reading error + A/B pair ambiguity
  printed, mean    0.10    carries the mean ratio 0.329 - the part-to-part sd of the survey
                           is 0.033, i.e. 10 % of the mean (docs/89 s1)
  printed, common  0.03    one draw SHARED by every printed part: the mean itself is known to
                           sd/sqrt(n) = 0.010 (3 %), and the other leg came off the same
                           printer - whatever bias the batch has, every part has it
  motor            0.02    RobStride catalogue +-20 g on 1420/880 g, +-3 g on 380/310
  bearing          0.02    catalogue figure; shields and grease
  fastener/steel   0.02    length and count as drawn, lubricant, stray washers
  aluminium        0.02    machining tolerance on a 6061 part (rods, upper body)

Output: mass_dr.json with, per link, the 2.5/97.5 percentiles of mass scale, COM shift per
axis (body frame, m) and inertia scale, plus the mjlab `dr.pseudo_inertia` parameters that
cover them (alpha = ln(mass_scale)/2, t = COM shift), and a figure.

Usage: mass_dr.py [--bodies=...] [--samples=4000] [--out=...] [--out-png=...]
       (mjlab .venv python)
"""
import json
import os
import sys

import numpy as np

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
sys.path.insert(0, f'{REPO}/tools/robot_model')
import massprops_fusion as MP                       # noqa: E402
from build_robot import ORIGIN_CAD, R, BNAME        # noqa: E402

MEAS = f'{REPO}/tools/robot_model/alu_parts_measured.json'
OUT_JSON = f'{REPO}/tools/robot_model/mass_dr.json'
OUT_PNG = f'{REPO}/docs/img/mass_dr_ranges.png'
SIGMA = dict(printed_measured=0.03, printed_mean=0.10, printed_common=0.03,
             motor=0.02, bearing=0.02, fastener=0.02, aluminium=0.02)
FASTENER_COMPLETENESS_MIN = float(os.environ.get('PYG_FASTENER_COMPLETENESS_MIN', '1.0'))
LINKS = ['pelvis', 'hip_pitch_link', 'hip_roll_link', 'thigh', 'shin', 'ankle_pitch_link',
         'foot', 'torso', 'shoulder_pitch_link', 'arm']
# What the measurement does NOT cover, and the randomizer still has to: cable harnesses and
# connectors (not in the CAD at all), the left leg coming off a different print batch,
# thermal paste / grease / zip ties, the serial-ankle stand-in for the 2-RSU loop, and a
# battery or electronics that move around on the trunk. These are judgment, written down so
# they can be argued with; the measured interval is the floor they are combined with.
STRUCT = dict(
    mass_floor=0.05,          # every link: at least +-5 % mass
    com_floor_m=0.005,        # every link: at least +-5 mm COM, every axis
    trunk_mass_extra=0.10,    # pelvis/torso: up to +10 % for harness, electronics, battery
    trunk_com_floor_m=0.020,  # pelvis/torso: +-20 mm - where that extra mass sits is unknown
    d_cap=0.02,               # residual inertia shape: never more than e^(2*0.02) = +-4 %
)
TRUNK = ('pelvis', 'torso')


def body_class(item, measured_ok):
    p, mat = item['path'], item.get('mat', '')
    body = p.split('::')[-1].split(' (')[0].split(' /')[0]
    if mat.startswith('PLA '):
        return 'printed_measured' if body in measured_ok else 'printed_mean'
    if 'Robstride' in p:
        return 'motor'
    if any(k in p for k in ('ZZ', 'CRBS', 'JS06')):
        return 'bearing'
    if 'Alumin' in mat:
        return 'aluminium'
    return 'fastener'


def is_fastener_item(item):
    p = item['path'].lower()
    return any(token in p for token in ('screw', 'bolt', 'nut', 'washer'))


def main():
    if not 0.0 < FASTENER_COMPLETENESS_MIN <= 1.0:
        raise ValueError('PYG_FASTENER_COMPLETENESS_MIN must be in (0, 1]')
    bodies_file = next((a.split('=')[1] for a in sys.argv if a.startswith('--bodies=')),
                       '/home/syaro/pyg_fea/fusion/bodies_printed.json')
    n = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--samples=')), 4000))
    out_json = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--out=')),
                    OUT_JSON)
    out_png = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--out-png=')),
                   OUT_PNG)
    rng = np.random.default_rng(7)
    B = json.load(open(bodies_file))
    links, _, _ = MP.collect(B)
    measured_ok = {e['body'] for e in json.load(open(MEAS))['entries']
                   if e['g'] is not None and e['conf'] in ('high', 'med')}

    # one shared draw for the printed batch, independent draws for everything else
    common = rng.normal(0.0, SIGMA['printed_common'], n)
    # One shared manufacturing-completeness event preserves the fact that a partially
    # populated CAD BOM misses screws robot-wide, rather than independently inventing a
    # different count for every bolt.  If only 50% were modeled, actual fastener mass may
    # be 1/c in [1, 2] times the included screw bodies.  This is opt-in sensitivity DR;
    # the default c=1 keeps the measured nominal unchanged.
    completion = rng.uniform(FASTENER_COMPLETENESS_MIN, 1.0, n)
    fastener_common = 1.0 / completion
    report, counts = {}, {}
    for link in LINKS:
        items = links[link]
        cls = [body_class(it, measured_ok) for it in items]
        counts[link] = {c: cls.count(c) for c in set(cls)}
        m0, c0, I0 = MP.aggregate(items)
        w0 = np.linalg.eigvalsh(I0)
        ms, cs, ws = np.empty(n), np.empty((n, 3)), np.empty((n, 3))
        for k in range(n):
            scale = {}
            for it, c in zip(items, cls):
                f = 1.0 + rng.normal(0.0, SIGMA[c])
                if c.startswith('printed'):
                    f *= 1.0 + common[k]
                if is_fastener_item(it):
                    f *= fastener_common[k]
                scale[it['path']] = max(f, 0.05)
            m, c, I = MP.aggregate(items, scale)
            ms[k] = m / m0
            cs[k] = (R @ (c - c0)) / 1000.0           # COM shift, link frame, metres
            ws[k] = np.linalg.eigvalsh(I) / w0
        q = lambda a: np.percentile(a, [2.5, 50, 97.5], axis=0)
        mq, cq, wq = q(ms), q(cs), q(ws)
        # inertia change NOT explained by the mass scale: shape factor
        shape = ws / ms[:, None]
        sq = q(shape)
        report[link] = dict(
            mass_kg=float(m0), n_bodies=len(items), classes=counts[link],
            mass_scale=dict(p2_5=float(mq[0]), p50=float(mq[1]), p97_5=float(mq[2]),
                            sd=float(ms.std())),
            com_shift_m=dict(p2_5=cq[0].tolist(), p97_5=cq[2].tolist(),
                             sd=cs.std(0).tolist()),
            inertia_scale=dict(p2_5=wq[0].tolist(), p97_5=wq[2].tolist()),
            inertia_shape_residual=dict(p2_5=sq[0].tolist(), p97_5=sq[2].tolist()),
        )
        # mjlab pseudo_inertia parameters that cover the 95 % interval
        a_lo, a_hi = np.log(mq[0]) / 2, np.log(mq[2]) / 2
        t = [[float(cq[0][i]), float(cq[2][i])] for i in range(3)]
        d_lo = float(np.log(sq[0].min()) / 2)
        d_hi = float(np.log(sq[2].max()) / 2)
        report[link]['mjlab_pseudo_inertia_measured'] = dict(
            alpha_range=[round(float(a_lo), 4), round(float(a_hi), 4)],
            t1_range=[round(v, 4) for v in t[0]], t2_range=[round(v, 4) for v in t[1]],
            t3_range=[round(v, 4) for v in t[2]],
            d_range=[round(d_lo, 4), round(d_hi, 4)])
        # recommended = measured interval widened to the structural floor
        trunk = link in TRUNK
        lo_m = min(mq[0], 1.0 - STRUCT['mass_floor'])
        hi_m = max(mq[2], 1.0 + STRUCT['mass_floor'] + (STRUCT['trunk_mass_extra'] if trunk else 0.0))
        cf = STRUCT['trunk_com_floor_m'] if trunk else STRUCT['com_floor_m']
        tr = [[round(min(cq[0][i], -cf), 4), round(max(cq[2][i], cf), 4)] for i in range(3)]
        # d stretches the mass distribution about the BODY ORIGIN, so it moves the COM by
        # |c_i| (e^d - 1) on each axis - on the thigh (COM 0.33 m below the joint) a d of 0.02
        # is 6.7 mm, more than the whole COM floor. Cap d per link so that excursion stays
        # within the floor, and report the resulting effective COM envelope honestly.
        c_body = (R @ (c0 - np.array(ORIGIN_CAD[link]))) / 1000.0      # nominal COM, body frame
        big = np.abs(c_body) > 1e-3
        d_by_com = float(np.log(1.0 + cf / np.abs(c_body[big])).min()) if big.any() else STRUCT['d_cap']
        dd = min(max(abs(d_lo), abs(d_hi)), STRUCT['d_cap'], d_by_com)
        eff = [[round(float(tr[i][0] - abs(c_body[i]) * (np.exp(dd) - 1)), 4),
                round(float(tr[i][1] + abs(c_body[i]) * (np.exp(dd) - 1)), 4)] for i in range(3)]
        report[link]['mjlab_pseudo_inertia_recommended'] = dict(
            alpha_range=[round(float(np.log(lo_m) / 2), 4), round(float(np.log(hi_m) / 2), 4)],
            t1_range=tr[0], t2_range=tr[1], t3_range=tr[2],
            d_range=[round(-dd, 4), round(dd, 4)],
            mass_scale=[round(float(lo_m), 3), round(float(hi_m), 3)],
            com_effective_m=dict(x=eff[0], y=eff[1], z=eff[2],
                                 note='t plus the COM excursion d causes about the body origin'),
            note='alpha: mass & inertia x e^(2a); t: COM shift [m] in the body frame; d is '
                 'the residual inertia shape factor and ALSO stretches the COM about the '
                 'body origin, so it is capped small')

    out = dict(bodies=bodies_file, samples=n, sigma=SIGMA, struct=STRUCT,
               fastener_completeness=dict(
                   modeled_fraction_range=[FASTENER_COMPLETENESS_MIN, 1.0],
                   included_fastener_mass_scale_range=[1.0, 1.0 / FASTENER_COMPLETENESS_MIN],
                   correlation='one shared robot-wide draw; per-fastener 2% residual remains',
                   status=('nominal measurement DR' if FASTENER_COMPLETENESS_MIN == 1.0
                           else 'sensitivity envelope, not a new nominal')),
               links=report,
               mjlab_body_names={l: BNAME.get(l, l if l != 'pelvis' else 'base_link')
                                 for l in LINKS})
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    json.dump(out, open(out_json, 'w'), indent=1)

    print(f"{'link':18s} {'kg':>6s} {'mass 95%':>14s} {'COM 95% x / y / z  [mm]':>34s} {'inertia 95%':>14s}")
    for l, r in report.items():
        ms_, cs_, ws_ = r['mass_scale'], r['com_shift_m'], r['inertia_scale']
        com = '  '.join(f"{cs_['p2_5'][i]*1000:+.1f}/{cs_['p97_5'][i]*1000:+.1f}" for i in range(3))
        print(f"{l:18s} {r['mass_kg']:6.3f} {ms_['p2_5']:6.3f}..{ms_['p97_5']:5.3f} {com:>34s} "
              f"{min(ws_['p2_5']):6.3f}..{max(ws_['p97_5']):5.3f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.2))
    y = np.arange(len(LINKS))
    for i, l in enumerate(LINKS):
        r = report[l]
        ms_ = r['mass_scale']
        ax[0].plot([ms_['p2_5'] * 100 - 100, ms_['p97_5'] * 100 - 100], [i, i], lw=6, color='#3d7ea6')
        cs_ = r['com_shift_m']
        for j, col in enumerate(('#ef476f', '#06d6a0', '#ffd166')):
            ax[1].plot([cs_['p2_5'][j] * 1000, cs_['p97_5'][j] * 1000], [i + (j - 1) * 0.22] * 2, lw=4,
                       color=col, label=('x', 'y', 'z')[j] if i == 0 else None)
        ws_ = r['inertia_scale']
        ax[2].plot([min(ws_['p2_5']) * 100 - 100, max(ws_['p97_5']) * 100 - 100], [i, i], lw=6, color='#9d4edd')
    for a, t, xl in zip(ax, ('link mass, 95 % interval', 'COM shift, 95 % interval (body frame)',
                             'principal inertia, 95 % interval'),
                        ('% of nominal', 'mm', '% of nominal')):
        a.set_yticks(y)
        a.set_yticklabels(LINKS if a is ax[0] else [''] * len(LINKS))
        a.invert_yaxis()
        a.axvline(0, color='k', lw=0.8)
        a.grid(axis='x', alpha=0.3)
        a.set_title(t, fontsize=10)
        a.set_xlabel(xl)
    ax[1].legend(fontsize=8, loc='lower right')
    fig.suptitle('Mass-property uncertainty propagated body-by-body through the aggregation '
                 f'({n} samples) - the ranges domain randomization has to cover', fontsize=10.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"\n{'link':18s} {'RECOMMENDED mass':>18s} {'t +- [mm] x/y/z':>17s} {'d':>7s} {'effective COM +- [mm]':>22s} {'alpha':>16s}")
    for l, r in report.items():
        rc = r['mjlab_pseudo_inertia_recommended']
        com = '/'.join(f"{max(abs(rc[k][0]), abs(rc[k][1]))*1000:.0f}" for k in ('t1_range', 't2_range', 't3_range'))
        ce = rc['com_effective_m']
        eff = '/'.join(f"{max(abs(ce[k][0]), abs(ce[k][1]))*1000:.1f}" for k in ('x', 'y', 'z'))
        print(f"{l:18s} {rc['mass_scale'][0]:8.3f}..{rc['mass_scale'][1]:5.3f} {com:>17s} {rc['d_range'][1]:7.4f} {eff:>22s} {rc['alpha_range'][0]:8.4f}..{rc['alpha_range'][1]:7.4f}")
    print(f'-> {out_json}\n-> {out_png}')


if __name__ == '__main__':
    main()
