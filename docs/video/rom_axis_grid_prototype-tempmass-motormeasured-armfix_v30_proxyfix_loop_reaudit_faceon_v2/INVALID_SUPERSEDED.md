# INVALID — do not use for motor-direction review

This grid/video is superseded. It converted a rotor-face normal directly into an URDF
joint sign and therefore silently assumed that every URDF child was attached to the rotor.
That assumption was not established in Fusion. The generator also coerced an UNKNOWN
`matches: null` value to `False`, which could display an unproved joint as CCW.

Required before regeneration:

1. identify the rotor/output flange face;
2. prove which mechanism link is fixed to rotor and which to stator;
3. map those links to URDF child and parent;
4. only then calculate and display CW/CCW.

The ROM motion itself may still be inspected as a q-positive kinematic sweep, but every
motor-direction caption in this directory is invalid.
