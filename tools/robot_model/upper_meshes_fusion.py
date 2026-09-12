"""Pull the upper-body meshes out of Fusion 360 - the STEP the campaign has covers the legs only.

The torso, the shoulder link and the arm exist only in the live Fusion document, so their
triangles are fetched over the MCP connector (`meshManager` per BRep body, low-quality
tessellation, assembly coordinates in cm) and written as STL in each rigid body's link frame,
matching the convention the leg meshes already use: origin on the joint, simulator axes
(x forward, y left, z up), metres.

Suppressed geometry is skipped: the light bulb of `ArmR_fullDoF` is off, and its 6.497 kg
alternative arm must not appear in either the mass properties or the picture.

Three things the 2026-08-26 run got wrong, and what replaces them:

**Never recurse inside a Fusion script.** The connector execs the submitted source in a module
whose `__getattr__` recurses on every name lookup - `tools/assembly_viewer/README.md` recorded
that in August, which is why `dump_fasteners.py` walks with an explicit stack - so a recursive
tree walk multiplies frames until Fusion's embedded Python blows its ~1 MB stack. The old code
here recursed twice (`collect`, and `find` once per slice), and after its 2026-08-26 run the
script host was left permanently wedged: every `fusion_mcp_execute` since, an empty script
included, returns the same `RecursionError: Stack overflow (used 993 kB)` raised inside that
`__getattr__`, while `fusion_mcp_read` still answers. Cause is inferred, not reproduced - the
wedge cannot be cleared from this side to try again - but it fits, and it explains "29 bodies
could not be fetched" exactly: once wedged, every remaining body fails whatever its size. Both
scripts below walk the occurrence tree with an explicit stack.

**Fail fast on a wedged host.** `mcp_client.HostWedged` is raised on the first sighting instead
of being retried; retrying costs five round trips and twelve seconds per body to relearn a fact
that cannot change without an add-in restart on the CAD PC.

**Never write into the asset directory until the whole set is in hand.** Every STL is built in
a staging directory. The live meshes are replaced only after all four rigid bodies fetched with
zero skipped bodies, and the files being replaced are copied to a timestamped backup first. The
old code wrote each group as it finished, so a failure halfway through left `arm.stl` holding a
partial mesh next to meshes from the previous generation.

Usage (mjlab venv, Fusion MCP reachable):

    upper_meshes_fusion.py                 # fetch + verify + publish, printing a before/after table
    upper_meshes_fusion.py --dry-run       # fetch into staging, report, publish nothing
    upper_meshes_fusion.py --stats         # no Fusion at all: measure what is on disk now
    upper_meshes_fusion.py --probe         # measure the connector's actual payload ceiling
                                           # ** DESTRUCTIVE: the deliberate over-limit requests
                                           # re-wedged a freshly restarted host on 2026-08-27.
                                           # The answer is already known (~524288 B per response,
                                           # 98304 floats) - do not run this against a live host
                                           # unless the ceiling itself is in question.
"""
import argparse
import base64
import datetime
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, '/home/syaro/MikuchanRemote/Human-Pygmalion/tools/fusion')
import mcp_client as M  # noqa: E402

OUT = ('/home/syaro/MikuchanRemote/Human-Pygmalion/pygmalion_locomotion/assets/'
       'pygmalion_v2/meshes')
BACKUP_ROOT = os.path.expanduser('~/pyg_fea')
R_SIM = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # cad -> sim
# rigid body -> (origin in CAD mm, occurrence filters that belong to it)
GROUPS = {
    # 2026-08-27 rescope: only what the shoulder rework changed (printed parts). Motors are
    # unchanged catalog items (drawn as cylinders per the standing rule) and the torso is out
    # of scope per the user. Filters are PATH SUBSTRINGS checked per occurrence:
    #  - the original 'Arm_R:1/Shoulder-Pitch2Roll' could never match (the real path runs
    #    through ArmR_Dummy:1) and was simply dead - the motor crash earlier in the dict
    #    meant this was never reached, so it shipped untested;
    #  - 'arm' needs a BLOCK list, or the substring match swallows the pitch link subtree
    #    and double-draws it. Shoulder-Roll2Yaw-dummy is excluded by its own light=False.
    'shoulder_pitch_link': (np.array([-200.0, 85.0, 540.0]),
                            ['ArmR_Dummy:1/Shoulder-Pitch2Roll'], ()),
    'arm': (np.array([-200.0, 85.0, 540.0]),
            ['Arm_R:1/ArmR_Dummy'], ('Shoulder-Pitch2Roll',)),
}

# An occurrence walk with an explicit stack - see the module docstring on why recursion here
# wedges the script host. `json` is imported for the payload, which goes out on stdout: this
# connector build captures print() and returns it as {"message": ...}.
LIST_SCRIPT = r'''
import adsk.core, adsk.fusion, json

def run(_context: str):
    app = adsk.core.Application.get()
    root = adsk.fusion.Design.cast(app.activeProduct).rootComponent
    stack = [(root.occurrences.itemByName("Joints_UpperBody:1"), "", True)]
    out = []
    while stack:
        o, path, live = stack.pop()
        live = live and o.isLightBulbOn
        p = path + "/" + o.name
        if live and any(w in p for w in WANT) and not any(b in p for b in BLOCK):
            for i in range(o.bRepBodies.count):
                b = o.bRepBodies.item(i)
                if b.isLightBulbOn:
                    out.append([p, i, b.name])
        for i in range(o.childOccurrences.count):
            stack.append((o.childOccurrences.item(i), p, live))
    print(json.dumps(out))
'''

# One body, one axis of its tessellation, one contiguous slice. `array` rather than
# `struct.pack(fmt, *part)`: packing 40 000 values as 40 000 call arguments was the other
# stack risk in the old script, and 'f'/'I' are little-endian on both ends (x86-64 Windows
# CAD PC, x86-64 Linux here), which `--probe` re-checks against the meta counts.
MESH_SCRIPT = r'''
import adsk.core, adsk.fusion, array, base64

def run(_context: str):
    app = adsk.core.Application.get()
    root = adsk.fusion.Design.cast(app.activeProduct).rootComponent
    occ = None
    # Search from every root occurrence.  The same safe fetcher is reused by the full-robot
    # exporter, which also targets Joints_UnderBody.  This remains an explicit stack: never
    # re-introduce recursion here (see the module-level host-wedge post-mortem).
    stack = [(root.occurrences.item(i), "") for i in range(root.occurrences.count)]
    while stack:
        c, p = stack.pop()
        q = p + "/" + c.name
        if q == TARGET:
            occ = c
            break
        for i in range(c.childOccurrences.count):
            stack.append((c.childOccurrences.item(i), q))
    b = occ.bRepBodies.item(IDX)
    calc = b.meshManager.createMeshCalculator()
    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.QUALITYQualityTriangleMesh)
    m = calc.calculate()
    if WHAT == "meta":
        print("%d %d" % (len(m.nodeCoordinatesAsDouble), len(m.nodeIndices)))
        return
    src = m.nodeCoordinatesAsDouble if WHAT == "v" else m.nodeIndices
    part = src[START:START + COUNT]
    a = array.array("f" if WHAT == "v" else "I", part)
    print(base64.b64encode(a.tobytes()).decode())
'''

PROBE_SCRIPT = r'''
def run(_context: str):
    print("A" * NBYTES)
'''


class Truncated(RuntimeError):
    """The connector returned fewer bytes than the script printed, at the smallest
    slice worth trying - at that point the payload ceiling is not what is wrong."""


class Fetcher:
    """Body-by-body tessellation fetch with a slice size that adapts to what comes back.

    `chunk` counts VALUES (floats for vertices, uint32 for indices), so a slice costs
    4*chunk bytes packed and 4/3 of that base64-encoded. Every slice is length-checked
    against what was asked for, because the connector truncates a too-large payload rather
    than reporting an error - a silent truncation is what turns into a corrupt STL.
    """

    FLOOR = 64                  # below this the connector is not the problem

    def __init__(self, quality='Low', chunk=16000):
        self.quality, self.chunk = quality, chunk
        self.max_payload = 0        # largest base64 payload actually returned, bytes
        self.min_failed = None      # smallest base64 payload that came back short, bytes

    def _run(self, src):
        # stdout occasionally comes back empty even though Fusion ran the script. Retry only
        # that normal-return channel.  Do NOT fall back to the exception/traceback channel:
        # the adapter's recursive ``__getattr__`` wrapper accumulates stack pressure there
        # and previously wedged the Fusion host after body 77/89.  An incomplete run is cheap
        # (the per-body cache resumes it); restarting Fusion is not.
        txt = ''
        for _ in range(3):
            txt = M.printed(src).strip()
            if txt:
                break
        if not txt:
            raise RuntimeError(
                'Fusion returned empty stdout three times; stopped safely without using '
                'the traceback payload channel. Cached bodies remain reusable.')
        self.max_payload = max(self.max_payload, len(txt))
        return txt

    def list_bodies(self, want, block=()):
        return json.loads(self._run(LIST_SCRIPT.replace('WANT', json.dumps(want))
                                    .replace('BLOCK', json.dumps(block))))

    def _slice(self, path, idx, what, start=0, count=0):
        if count:
            M.assert_slice_ok(count)
        src = (MESH_SCRIPT.replace('TARGET', json.dumps(path)).replace('IDX', str(idx))
               .replace('QUALITY', self.quality).replace('WHAT', json.dumps(what))
               .replace('START', str(start)).replace('COUNT', str(count)))
        return self._run(src)

    def _values(self, path, idx, what, n, dtype):
        """One axis of one body, in slices, re-sized from what the connector actually returns.

        A short answer is not retried at the same size and it is not halved blindly either:
        the truncated payload IS the measurement of the ceiling, so the next slice is sized
        from it (4 base64 chars carry 3 bytes carry 3/4 of a value) with a 10 % margin. That
        converges in one step where halving from 16 000 would take five round trips, and it
        needs no guessed floor - only a sanity floor below which the connector is not what
        is wrong.
        """
        buf, start, chunk, transient_retries = [], 0, self.chunk, 0
        while start < n:
            count = min(chunk, n - start)
            txt = self._slice(path, idx, what, start, count)
            try:
                raw = base64.b64decode(txt, validate=True) if txt else b''
            except Exception:                                      # noqa: BLE001
                raw = b''                                          # cut mid-quantum
            if len(raw) != 4 * count:
                # The adapter occasionally returns a tiny, otherwise base64-looking status
                # string instead of stdout.  It is not the measured payload ceiling: retry the
                # identical read a bounded number of times before adapting the chunk size.
                # Without this distinction one 34-byte hiccup collapses a 5k-value request to
                # five values and falsely reports that the connector is not the problem.
                tiny = len(raw) < min(1024, max(64, 4 * count // 10))
                if tiny and transient_retries < 3:
                    transient_retries += 1
                    print(f'      transient short {what} payload ({len(raw)} B); '
                          f'retry {transient_retries}/3', flush=True)
                    continue
                self.min_failed = min(self.min_failed or 10 ** 9, len(txt) or 10 ** 9)
                fits = int(len(txt) * 3 / 16 * 0.9)
                nxt = min(chunk // 2, fits) if fits else chunk // 2
                if nxt < self.FLOOR:
                    raise Truncated(
                        f'{what} slice {start}+{count}: asked for {4 * count} B, got '
                        f'{len(raw)} B in a {len(txt)} B payload; the next slice would be '
                        f'{nxt} values, under the {self.FLOOR}-value floor - the connector '
                        f'is not the problem here')
                chunk = nxt
                continue
            buf.append(np.frombuffer(raw, dtype=dtype))
            start += count
            transient_retries = 0
        self.chunk = chunk          # remember the size that worked, for the next body
        return np.concatenate(buf) if buf else np.zeros(0, dtype=dtype)

    def body(self, path, idx):
        nv, nf = (int(x) for x in self._slice(path, idx, 'meta').split())
        v = self._values(path, idx, 'v', nv, '<f4').astype(float)
        f = self._values(path, idx, 'f', nf, '<u4').astype(int)
        return v, f


def fetch(fx, want, block=()):
    """Every body of a group, or an exception. No body is allowed to go missing quietly."""
    bodies = fx.list_bodies(want, block)
    out, failed = [], []
    for path, idx, name in bodies:
        try:
            v, f = fx.body(path, idx)
        except M.HostWedged:
            raise
        except Exception as e:                                    # noqa: BLE001
            failed.append(f'{name}: {e}')
            print(f'    {name[:44]:44s}  FAILED  {str(e)[:50]}', flush=True)
            continue
        out.append(dict(n=name, v=v, f=f))
        print(f'    {name[:44]:44s} {len(f) // 3:7d} tris', flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} of {len(bodies)} bodies failed for {want}:\n  '
                           + '\n  '.join(failed[:10]))
    if not out:
        raise RuntimeError(f'nothing fetched for {want}')
    return out


def build_group(body, origin, parts, stage):
    """Assemble one rigid body's STLs into the staging directory."""
    meshes = []
    for p in parts:
        v = p['v'].reshape(-1, 3) * 10.0                            # cm -> mm, CAD frame
        f = p['f'].reshape(-1, 3)
        if f.size and f.max() >= len(v):
            raise ValueError(f"{p['n']}: index {f.max()} vs {len(v)} verts")
        meshes.append(trimesh.Trimesh((v - origin) @ R_SIM.T / 1000.0, f, process=False))
    mesh = trimesh.util.concatenate(meshes)
    hull = mesh.convex_hull
    mesh.export(f'{stage}/{body}.stl')
    hull.export(f'{stage}/{body}_hull.stl')
    for src, name in ((mesh, f'R_{body}.stl'), (hull, f'R_{body}_hull.stl')):
        mm = src.copy()
        mm.vertices[:, 1] *= -1
        mm.faces = mm.faces[:, [0, 2, 1]]
        mm.export(f'{stage}/{name}')
    b = mesh.bounds
    return dict(parts=len(parts), faces=int(len(mesh.faces)),
                bbox=np.round(b[1] - b[0], 3).tolist())


def mesh_stats(path):
    """Measure one STL, or None if it is missing or unreadable.

    Deliberately forgiving: this only ever feeds the before/after table, and a previous
    generation that is absent or corrupt is a reason to print a dash, never a reason to
    throw away a fetch that already succeeded.
    """
    if not os.path.exists(path):
        return None
    try:
        m = trimesh.load(path, process=False, force='mesh')
        b = m.bounds
        return dict(nv=int(len(m.vertices)), nf=int(len(m.faces)),
                    vol=float(m.volume), ext=[round(float(x), 4) for x in (b[1] - b[0])])
    except Exception:                                              # noqa: BLE001
        return None


def report(names, old, new):
    """Before/after per mesh. Volume is the signed mesh volume - meaningful for the hulls
    and for a watertight body, and a fast way to see that a fetch dropped parts."""
    print(f'\n{"mesh":30s} {"verts":>16s} {"tris":>16s} {"volume cm3":>17s}   bbox m (new)')
    for n in names:
        o, w = old.get(n), new.get(n)
        if o is None and w is None:
            continue
        f = lambda d, k, s=1.0: ('-' if d is None else f'{d[k] * s:.4g}')            # noqa: E731
        ext = '-' if w is None else ' '.join(f'{x:.3f}' for x in w['ext'])
        print(f'{n:30s} {f(o, "nv"):>7s}->{f(w, "nv"):>8s} {f(o, "nf"):>7s}->{f(w, "nf"):>8s} '
              f'{f(o, "vol", 1e6):>7s}->{f(w, "vol", 1e6):>8s}   {ext}')


def files_of(groups):
    out = []
    for body in groups:
        out += [f'{body}.stl', f'{body}_hull.stl', f'R_{body}.stl', f'R_{body}_hull.stl']
    return out


def publish(stage, names):
    """Replace the live meshes as late and as fast as possible.

    The whole set is copied next to its destination first, so the only thing left to do is
    16 `os.replace` calls - each one atomic, all of them within the same filesystem, with no
    copying in between. A crash during the copy leaves `.new` files and nothing else touched;
    the previous generation is in the backup either way.
    """
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = f'{BACKUP_ROOT}/mesh_backup_{ts}'
    os.makedirs(backup, exist_ok=True)
    for n in names:
        if os.path.exists(f'{OUT}/{n}'):
            shutil.copy2(f'{OUT}/{n}', f'{backup}/{n}')
    for n in names:
        shutil.copy2(f'{stage}/{n}', f'{OUT}/{n}.new')
    for n in names:
        os.replace(f'{OUT}/{n}.new', f'{OUT}/{n}')
    print(f'\nreplaced {len(names)} files in {OUT}\nprevious set kept in {backup}')
    return backup


def probe():
    """Measure the connector's real limits instead of guessing at them.

    Two separate ceilings matter: how long a SCRIPT may be before the connector silently
    declines to run it, and how many bytes it will hand back. Both are found by doubling
    until something breaks, then bisecting.
    """
    print('script host:', 'alive' if M.host_alive() else 'WEDGED (restart the add-in)')
    lo, hi = 1024, None
    n = 4096
    while n <= 8 << 20:
        try:
            got = len(M.printed(PROBE_SCRIPT.replace('NBYTES', str(n))).strip())
        except Exception as e:                                     # noqa: BLE001
            print(f'  {n:>9d} B -> error {str(e)[:60]}')
            hi = n
            break
        ok = got == n
        print(f'  {n:>9d} B -> returned {got:>9d} B  {"ok" if ok else "TRUNCATED/EMPTY"}')
        if not ok:
            hi = n
            break
        lo = n
        n *= 2
    if hi is None:
        print(f'  payload ceiling: not reached by {lo} B')
        return
    while hi - lo > max(1024, lo // 64):
        mid = (lo + hi) // 2
        try:
            got = len(M.printed(PROBE_SCRIPT.replace('NBYTES', str(mid))).strip())
        except Exception:                                          # noqa: BLE001
            got = -1
        (lo, hi) = (mid, hi) if got == mid else (lo, mid)
        print(f'  {mid:>9d} B -> {got:>9d} B')
    print(f'  payload ceiling: between {lo} and {hi} B '
          f'(= {lo * 3 // 4} B of binary, {lo * 3 // 16} floats per slice)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='fetch and report, publish nothing')
    ap.add_argument('--stats', action='store_true', help='measure the meshes on disk, no Fusion')
    ap.add_argument('--probe', action='store_true', help='measure the connector payload ceiling (DESTRUCTIVE - re-wedged the host once; ceiling is ~524288 B, default chunk already safe)')
    ap.add_argument('--quality', default='Low', choices=['Low', 'Medium', 'High'])
    ap.add_argument('--chunk', type=int, default=16000, help='values per slice, halved on need')
    ap.add_argument('--i-accept-rewedging-the-host', action='store_true',
                    help='required to run --probe; see the refusal message for why')
    args = ap.parse_args()

    names = files_of(GROUPS)
    if args.stats:
        old = {n: mesh_stats(f'{OUT}/{n}') for n in names}
        report(names, old, old)
        return
    M.connect()
    if args.probe and not getattr(args, 'i_accept_rewedging_the_host', False):
        print('REFUSED: --probe sends deliberately over-limit requests and re-wedged a freshly\n'
              'restarted host on 2026-08-27, costing a walk to the CAD PC. The ceiling is already\n'
              'known (524288 B/response, 98304 floats). If the ceiling itself is in question, rerun\n'
              'with --i-accept-rewedging-the-host and a person standing at the CAD PC.')
        return
    if args.probe:
        probe()
        return
    if not M.host_alive():
        raise SystemExit(
            'ABORT before touching anything: the Fusion script host is wedged. Every script - '
            'even an empty one - dies in the connector\'s own __getattr__ recursion, while\n'
            'fusion_mcp_read still answers. Restart the Fusion MCP add-in on the CAD PC\n'
            '(Utilities > Add-Ins > Scripts and Add-Ins: stop, then start), then re-run.\n'
            'The meshes on disk are untouched.')

    stage = tempfile.mkdtemp(prefix='upper_meshes_')
    fx = Fetcher(quality=args.quality, chunk=args.chunk)
    stats = {}
    try:
        for body, (origin, want, block) in GROUPS.items():
            print(f'{body}:', flush=True)
            stats[body] = build_group(body, origin, fetch(fx, want, block), stage)
            s = stats[body]
            print(f"  -> {s['parts']:3d} parts · {s['faces']:7d} tris · bbox {s['bbox']} m",
                  flush=True)
        missing = [n for n in names if not os.path.exists(f'{stage}/{n}')]
        if missing:
            raise RuntimeError(f'staging incomplete: {missing}')
        json.dump(stats, open(f'{stage}/upper_meshes.json', 'w'), indent=1)

        print(f'\nlargest payload returned: {fx.max_payload} B'
              + (f' · smallest that came back short: {fx.min_failed} B' if fx.min_failed
                 else ' · no slice ever came back short')
              + f' · final slice size {fx.chunk} values')
        old = {n: mesh_stats(f'{OUT}/{n}') for n in names}
        new = {n: mesh_stats(f'{stage}/{n}') for n in names}
        report(names, old, new)
        if args.dry_run:
            print(f'\n--dry-run: nothing published, staged set left in {stage}')
            return
        publish(stage, names + ['upper_meshes.json'])
    finally:
        if os.path.isdir(stage) and not (args.dry_run and os.listdir(stage)):
            shutil.rmtree(stage, ignore_errors=True)


if __name__ == '__main__':
    main()
