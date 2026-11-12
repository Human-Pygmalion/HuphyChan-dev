# HUPHY 00399 reference motion

This package contains the exact 50 Hz HUPHY reference used by the 00399
motion-tracking runs and a standalone MuJoCo player/recorder. It does not load a
policy and does not require MJLab task registration.

## Contents

- `00399_interactionmesh_50hz.npz`: 1,511 frames, 50 Hz, 24 scalar joints,
  `qpos[31]`, and `qvel[30]`.
- `play_reference.py`: native MuJoCo playback and headless MP4 recording.
- `manifest.sha256`: hashes for both files.

The robot is the repository's capsule7plus5 HUPHY MJCF:

```text
pygmalion_locomotion/assets/pygmalion_v2/
LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml
```

## Interactive MuJoCo playback

From the repository root, in an environment with `mujoco` and `numpy`:

```bash
python mjlab_policy/reference_motions/huphy_00399/play_reference.py --loop
```

The camera follows the reference root. Use `--start-frame` and `--end-frame` to
play a subrange.

## Headless MP4 recording

`ffmpeg` must be installed:

```bash
MUJOCO_GL=egl python \
  mjlab_policy/reference_motions/huphy_00399/play_reference.py \
  --output /tmp/huphy_00399_reference.mp4
```

The output is 960x720 at the motion's native 50 Hz unless overridden with
`--width` and `--height`.

## Integrity check

```bash
cd mjlab_policy/reference_motions/huphy_00399
sha256sum -c manifest.sha256
```
