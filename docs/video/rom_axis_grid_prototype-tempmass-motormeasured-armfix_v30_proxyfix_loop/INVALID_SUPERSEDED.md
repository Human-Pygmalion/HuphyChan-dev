# INVALID — do not use for motor direction or deployment

This 2026-09-01 grid placed the camera behind the rotor flange by treating the outward
surface normal as pointing into the screen.  For the user's face-on convention, the outward
normal points toward the viewer and the CW motor-positive angular axis is `-normal`.

It also encoded the now-retracted A/B crossing and synthesized right-side normals without
direct evidence.  Use the sibling `..._reaudit_faceon_v2/` artifact instead.
