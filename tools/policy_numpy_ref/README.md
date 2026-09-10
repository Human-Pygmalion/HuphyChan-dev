# policy_numpy_ref

Independent, PyTorch-free numpy reimplementation of the `legonly_ab_v3_keyfix`
(`model_27999.pt`) actor's inference pipeline, plus a verification suite that
cross-checks it against the real checkpoint and the existing ONNX export.

See `docs/134_legonly_ab_v3_keyfix_policy_eli5.md` (concepts, worked examples),
`docs/135_legonly_ab_v3_keyfix_schema_reference.md` (exact formulas/tables), and
`docs/136_legonly_ab_v3_keyfix_sim2real_gaps.md` (sim2real correspondence +
these verification results) for the full writeup. This README only covers how
to regenerate the artifacts.

## Regenerating everything from scratch

```bash
cd /home/syaro/MikuchanRemote/Human-Pygmalion

# 1. Extract the checkpoint's actor weights to a plain numpy .npz (needs torch, once).
CUDA_VISIBLE_DEVICES= mujoco-sim/mjlab/.venv/bin/python3 \
  tools/policy_numpy_ref/scripts/extract_weights.py \
  --pt mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2/model_27999.pt \
  --out tools/policy_numpy_ref/artifacts/legonly_ab_v3_keyfix_weights.npz

# 2. Bake the pygviewer oracle artifacts for this checkpoint (policy_contract.json,
#    .onnx, parity.npz, obs_parity.npz) -- only needed once per checkpoint, skip if
#    /home/syaro/pyg_fea/pygviewer/cache/ already has a bake for this run+iter.
CUDA_VISIBLE_DEVICES= mujoco-sim/mjlab/.venv/bin/python3 tools/pygviewer/run.py bake policy \
  --pt mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2/model_27999.pt \
  --variant LegOnly-AB \
  --cache /home/syaro/pyg_fea/pygviewer/cache

# 3. Roll a real episode with forward hooks to capture every pipeline stage's
#    actual numbers (feeds the ELI5 doc's worked examples and one of the
#    action-pipeline tests).
CUDA_VISIBLE_DEVICES= mujoco-sim/mjlab/.venv/bin/python3 \
  tools/policy_numpy_ref/scripts/hook_live_trace.py --steps 40

# 4. Run the verification suite.
CUDA_VISIBLE_DEVICES= mujoco-sim/mjlab/.venv/bin/python3 -m pytest tools/policy_numpy_ref/tests/ -v
```

## Independence note

`policy_numpy_ref/` does not import anything from `tools/pygviewer/pygviewer/`
(its `policy.py`, `_bake_policy.py`, etc.) -- it is a from-scratch reimplementation,
used as an audit trail. The *tests* are allowed to load pygviewer's baked
artifacts (`policy_contract.json`, `.onnx`, `.parity.npz`) as independent oracle
data, and `scripts/hook_live_trace.py` reuses `pygviewer.bake`'s `variant_env`/
`TASK` env-var plumbing and `pygviewer.contract.load_contract` (configuration
helpers, not inference logic) to build an identical mjlab environment -- neither
of these counts as reusing the numpy inference implementation itself.
