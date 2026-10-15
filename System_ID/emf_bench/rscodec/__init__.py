"""Vendored read-only copy of the HUPHY RobStride codec (see headers of mit.py / tables.py).

Layout matches what System_ID/rs_direct_socket.py and bus_load_test.py expect:
    from rscodec import mit, tables
"""

from . import mit, tables

__all__ = ["mit", "tables"]
