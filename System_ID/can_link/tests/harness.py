"""Shared test rig: loads the vendored HUPHY motors package twice, as it is (`orig`) and with
the two huphy_patch files laid over a copy (`patched`) -- exactly what a developer does when
copying the two files in -- and builds buses on a FakeWire.

No python-can is needed: `fake_can.install_fake_can_module` provides the `can` stand-in.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import types
from typing import Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # System_ID/can_link
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fake_can as fc  # noqa: E402

VENDOR = os.path.join(ROOT, "vendor")
ORIG_NAME = "huphy_motors_biped_a1b67b1"
PATCHED_NAME = "huphy_motors_patched"
_cache: Dict[str, types.SimpleNamespace] = {}
_tmp = None


def _load(name: str, path_root: str) -> types.SimpleNamespace:
    if path_root not in sys.path:
        sys.path.insert(0, path_root)
    base = importlib.import_module(name)
    canbus = importlib.import_module(name + ".canbus")
    bus = importlib.import_module(name + ".robstride.bus")
    tables = importlib.import_module(name + ".robstride.tables")
    basemod = importlib.import_module(name + ".base")
    return types.SimpleNamespace(name=name, pkg=base, canbus=canbus, bus=bus, tables=tables, base=basemod)


def orig() -> types.SimpleNamespace:
    if "orig" not in _cache:
        _cache["orig"] = _load(ORIG_NAME, VENDOR)
    return _cache["orig"]


def patched() -> types.SimpleNamespace:
    """Copy the vendored tree, overlay the two patched files, import under another name."""
    global _tmp
    if "patched" not in _cache:
        _tmp = tempfile.mkdtemp(prefix="canlink_patched_")
        dst = os.path.join(_tmp, PATCHED_NAME)
        shutil.copytree(os.path.join(VENDOR, ORIG_NAME), dst)
        shutil.copy(os.path.join(ROOT, "huphy_patch", "canbus.py"), os.path.join(dst, "canbus.py"))
        shutil.copy(os.path.join(ROOT, "huphy_patch", "robstride", "bus.py"), os.path.join(dst, "robstride", "bus.py"))
        _cache["patched"] = _load(PATCHED_NAME, _tmp)
    return _cache["patched"]


class Rig:
    """6 motors (ids 7..12 like the right leg) on a FakeWire, one RobStrideBus on top."""

    def __init__(self, pkg, *, virtual: bool = True, reader: Optional[bool] = None, n: int = 6,
                 first_id: int = 7, drain_s: float = 0.004, proc_s: float = 0.0004,
                 scripts: Optional[Dict[int, Callable]] = None, host_ids: Optional[Dict[int, int]] = None,
                 bus_kwargs: Optional[dict] = None, canbus_kwargs: Optional[dict] = None,
                 collide: bool = False, model: str = "RS04", **wire_kw):
        self.pkg = pkg
        self.clock = fc.VClock() if virtual else fc.RealClock()
        self.ids = list(range(first_id, first_id + n))
        motors = {}
        for i in self.ids:
            motors[i] = fc.FakeMotor(i, proc_s=proc_s,
                                     host_id=(host_ids or {}).get(i, fc.HOST_ID_DEFAULT),
                                     script=(scripts or {}).get(i))
        self.wire = fc.FakeWire(motors, clock=self.clock, collide_same_id=collide, **wire_kw)
        self.motors = motors
        self.canmod = fc.install_fake_can_module(self.wire)
        if virtual:
            # the modules read time.monotonic()/time.time()/time.sleep() -> give them the virtual clock
            pkg.canbus.time = self.clock
            pkg.bus.time = self.clock
        else:
            import time as _real_time
            pkg.canbus.time = _real_time
            pkg.bus.time = _real_time
        kw = dict(canbus_kwargs or {})
        if reader is not None and "reader" in self._canbus_params(pkg):
            kw["reader"] = reader
        elif reader is not None:
            pass
        if virtual and "reader" in self._canbus_params(pkg):
            kw.setdefault("reader", False)       # a reader thread cannot use a virtual clock
        self.can = pkg.canbus.CanBus("fake0", **kw)
        spec = {i: pkg.base.Motor(id=i, name=f"m{i}", model=model) for i in self.ids} \
            if "name" in pkg.base.Motor.__dataclass_fields__ else None
        self.rs = pkg.bus.RobStrideBus(self.can, spec or self._motors(pkg, model), drain_s=drain_s, **(bus_kwargs or {}))
        self.rs.connect()

    @staticmethod
    def _canbus_params(pkg):
        import inspect
        return inspect.signature(pkg.canbus.CanBus.__init__).parameters

    def _motors(self, pkg, model):
        fields = pkg.base.Motor.__dataclass_fields__
        out = {}
        for i in self.ids:
            kw = {}
            for k in fields:
                if k == "id":
                    kw[k] = i
                elif k == "model":
                    kw[k] = model
                elif k == "name":
                    kw[k] = f"m{i}"
            out[i] = pkg.base.Motor(**kw)
        return out

    def close(self) -> None:
        try:
            self.can.disconnect()
        finally:
            fc.uninstall_fake_can_module()


def run_cycles(rig: Rig, n: int, period: float, *, commanded: Optional[List[int]] = None) -> List[dict]:
    """Send command k (position = k deg) to every motor, collect, record each state's lag.

    lag = k - decoded position: 0 means the state is from this cycle, 1 = one cycle old.
    Needs a virtual clock (it idles by advancing it)."""
    mit_cmd = rig.pkg.bus.MitCommand
    ids = commanded or rig.ids
    t0 = rig.clock.monotonic() + period
    out = []
    for k in range(n):
        rig.clock.advance_to(t0 + k * period)
        rig.rs.send_mit({m: mit_cmd(position_deg=float(k)) for m in ids})
        missing = rig.rs.collect(expect=len(ids))
        lag = {}
        for m in ids:
            st = rig.rs.state(m)
            lag[m] = (k - round(st.position_deg)) if st.stamp > 0 else None
        out.append(dict(k=k, missing=list(missing), lag=lag,
                        stats=getattr(rig.rs, "last_collect", None)))
    return out
