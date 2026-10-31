#!/usr/bin/env python3
"""Fetch the triangle meshes of the visible bodies of a sole component (default `Sole_v3_byClaude(Mirror):1`) from the live Fusion document.

Read-only. Results come back through stdout (mcp_client.printed), never through deliberate exceptions, in slices of at most
40,000 values (SLICE_VALUES_MAX 65,536 is 2/3 of the measured host ceiling). Hidden bodies (light bulb off) are skipped.
Output: <out>/<body>.npz (v [n,3] mm in the Fusion document frame, f [m,3]) and <out>/index.json.

Usage: FUSION_MCP=http://192.168.20.161:27182/mcp FUSION_MCP_HOST=127.0.0.1:27182 python3 tools/foot_sole/fusion_sole_fetch.py <out> [quality] [occurrence]
       quality: low | medium | high (default high)
       occurrence: a root occurrence name, or "component:<name>" for the bodies of a (possibly nested) component, e.g.
       "component:Feet_Up" (its occurrence transform must be identity - true for the v5 foot parts on 2026-10-31)
"""
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fusion"))
import mcp_client as M  # noqa: E402

OUT = Path(sys.argv[1])
QUAL = {"low": "LowQualityTriangleMesh", "medium": "NormalQualityTriangleMesh", "high": "HighQualityTriangleMesh"}[
  sys.argv[2] if len(sys.argv) > 2 else "high"]
TARGET = sys.argv[3] if len(sys.argv) > 3 else "Sole_v3_byClaude(Mirror):1"  # root occurrence name, exact
CHUNK = int(__import__("os").environ.get("FETCH_CHUNK", "20000"))  # 40,000-value slices came back truncated on the stdout channel once (2026-10-31)

SRC = r'''
import adsk.core, adsk.fusion, json, struct, base64
def run(_c):
    app = adsk.core.Application.get()
    root = adsk.fusion.Design.cast(app.activeProduct).rootComponent
    bodies = []
    st = []
    if __TARGET__.startswith("component:"):
        c = adsk.fusion.Design.cast(app.activeProduct).allComponents.itemByName(__TARGET__[10:])
        bodies = [(c.name + "|" + c.bRepBodies.item(i).name, c.bRepBodies.item(i)) for i in range(c.bRepBodies.count)]
    else:
        for i in range(root.occurrences.count):
            if root.occurrences.item(i).name == __TARGET__:
                st = [(root.occurrences.item(i), root.occurrences.item(i).isLightBulbOn)]
    while st:
        o, live = st.pop()
        for i in range(o.bRepBodies.count):
            b = o.bRepBodies.item(i)
            if live and b.isLightBulbOn:
                bodies.append((o.name + "|" + b.name, b))
        for i in range(o.childOccurrences.count):
            c = o.childOccurrences.item(i)
            st.append((c, live and c.isLightBulbOn))
    bodies.sort(key=lambda x: x[0])
    if __WHAT__ == "list":
        print(json.dumps([n for n, _ in bodies]))
        return
    name, b = bodies[__IDX__]
    calc = b.meshManager.createMeshCalculator()
    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.__QUAL__)
    m = calc.calculate()
    v = m.nodeCoordinatesAsDouble
    f = m.nodeIndices
    if __WHAT__ == "meta":
        print(json.dumps({"name": name, "nv": len(v), "nf": len(f)}))
        return
    src = v if __WHAT__ == "v" else f
    part = src[__START__:__START__ + __COUNT__]
    fmt = "<%df" % len(part) if __WHAT__ == "v" else "<%dI" % len(part)
    print(base64.b64encode(struct.pack(fmt, *part)).decode())
'''


def call(what, idx=0, start=0, count=0):
  src = (SRC.replace("__TARGET__", json.dumps(TARGET)).replace("__WHAT__", json.dumps(what)).replace("__IDX__", str(idx)).replace("__QUAL__", QUAL)
         .replace("__START__", str(start)).replace("__COUNT__", str(count)))
  txt = M.printed(src).strip()
  if not txt:
    raise RuntimeError("empty stdout from Fusion (stdout capture not available on this build?)")
  return txt


def stream(idx, what, n, fmt):
  buf = []
  for st in range(0, n, CHUNK):
    cnt = min(CHUNK, n - st)
    M.assert_slice_ok(cnt)
    txt = call(what, idx, st, cnt)
    try:
      raw = base64.b64decode(txt)
    except Exception:
      raise SystemExit(f"slice {what}[{st}:{st + cnt}] is not base64 ({len(txt)} chars): {txt[:200]!r}")
    buf.append(np.array(struct.unpack(fmt % cnt, raw)))
  return np.concatenate(buf)


def main():
  M.connect()
  OUT.mkdir(parents=True, exist_ok=True)
  names = json.loads(call("list"))
  index = []
  for i, nm in enumerate(names):
    meta = json.loads(call("meta", i))
    print(nm, meta, flush=True)
    if meta["nv"] > 600000 or meta["nf"] > 1200000:
      raise SystemExit(f"{nm}: mesh too large for this quality ({meta}); rerun with medium")
    v = stream(i, "v", meta["nv"], "<%df").astype(float).reshape(-1, 3) * 10.0  # cm -> mm
    f = stream(i, "f", meta["nf"], "<%dI").astype(int).reshape(-1, 3)
    safe = nm.replace("|", "__").replace(":", "_").replace(" ", "_")
    np.savez_compressed(OUT / f"{safe}.npz", v=v, f=f)
    index.append(dict(name=nm, file=f"{safe}.npz", nv=len(v), nf=len(f), lo=v.min(0).round(2).tolist(), hi=v.max(0).round(2).tolist()))
  (OUT / "index.json").write_text(json.dumps(dict(quality=QUAL, bodies=index), indent=1))
  print("saved", OUT)


if __name__ == "__main__":
  main()
