"""Independent, PyTorch-free numpy reimplementation of the LegOnly AB policy's
inference pipeline (checkpoint legonly_ab_v3_keyfix / model_27999.pt).

This package intentionally does not import anything from tools/pygviewer/pygviewer --
see docs/136_legonly_ab_v3_keyfix_sim2real_gaps.md for why (independent audit trail).
The only module that touches torch is weights.py, and only at extraction time.
"""
