# Leg parts still at printed density in the current models (candidates for Al 6061)

Mass as Al = CAD volume x 2.70 g/cm3. Delta is per leg (the other leg is the mirror, so the robot gains twice the sum).
Source dump is one CAD revision behind the live Fusion document; volumes are re-read from Fusion before any build.

| CAD group | part | volume cm3 | now g | now g/cm3 | as Al g | +g per leg | link |
|---|---|---|---|---|---|---|---|
| Ankle2Feet | KneeB2AnkleB | 63.64 | 56.5 | 0.8874 | 171.8 | +115.4 | (ankle group, see below) |
| Ankle2Feet | KneeB2AnkleA | 62.41 | 57.2 | 0.9170 | 168.5 | +111.3 | (ankle group, see below) |
| Ankle2Feet | AnkleFeetPillowAB | 38.95 | 34.6 | 0.8874 | 105.2 | +70.6 | (ankle group, see below) |
| Ankle2Feet | AnkleUniversalJointCore | 24.81 | 22.0 | 0.8874 | 67.0 | +45.0 | (ankle group, see below) |
| Ankle2Feet | AnkleFeetPillowA | 15.14 | 13.4 | 0.8874 | 40.9 | +27.4 | (ankle group, see below) |
| Ankle2Feet | AnkleFeetPillowB | 15.14 | 13.4 | 0.8874 | 40.9 | +27.4 | (ankle group, see below) |
| Ankle2Feet | AnkleBraceB | 15.14 | 13.4 | 0.8874 | 40.9 | +27.4 | (ankle group, see below) |
| Ankle2Feet | AnkleBraceA | 15.02 | 13.3 | 0.8874 | 40.5 | +27.2 | (ankle group, see below) |
| Ankle2Feet | Crank_A_Cover | 11.71 | 10.4 | 0.8874 | 31.6 | +21.2 | (ankle group, see below) |
| Ankle2Feet | Crank_B_Cover | 11.45 | 10.2 | 0.8874 | 30.9 | +20.8 | (ankle group, see below) |
| Ankle2Feet | Ankle_Stopper_Roll | 9.83 | 8.7 | 0.8874 | 26.5 | +17.8 | (ankle group, see below) |
| Ankle2Feet | Ankle_Stopper_Pitch | 7.61 | 6.8 | 0.8874 | 20.5 | +13.8 | (ankle group, see below) |
| Ankle2Feet | Crank_A (done in M18) | 27.36 | 24.3 | 0.8874 | 73.9 | - | (ankle group, see below) |
| Ankle2Feet | Crank_B (done in M18) | 26.73 | 23.7 | 0.8874 | 72.2 | - | (ankle group, see below) |
| HipPitch2Roll | HipPitchRotor | 73.78 | 65.5 | 0.8874 | 199.2 | +133.7 | hip_pitch_link |
| HipPitch2Roll | HipPitch2Roll_B | 54.81 | 48.6 | 0.8874 | 148.0 | +99.3 | hip_pitch_link |
| HipPitch2Roll | HipPitch2Roll_A | 47.47 | 42.1 | 0.8874 | 128.2 | +86.0 | hip_pitch_link |
| HipPitch2Roll | HipPitchFlange | 39.62 | 35.2 | 0.8874 | 107.0 | +71.8 | hip_pitch_link |
| HipRoll2Yaw | HipRoll2YawA | 61.28 | 48.6 | 0.7927 | 165.5 | +116.9 | hip_roll_link |
| HipRoll2Yaw | HipRoll2YawB | 50.19 | 44.5 | 0.8874 | 135.5 | +91.0 | hip_roll_link |
| HipRoll2Yaw | Body4 | 14.54 | 13.4 | 0.9213 | 39.3 | +25.9 | hip_roll_link |
| HipRoll2Yaw | HipRoll2Yaw_Mount (done in M18) | 60.97 | 67.0 | 1.0989 | 164.6 | - | hip_roll_link |
| HipYaw2Knee | Yaw2KneePlateA | 119.24 | 98.1 | 0.8227 | 322.0 | +223.9 | thigh_link |
| HipYaw2Knee | HipYaw2Knee_Rotor | 117.48 | 104.2 | 0.8874 | 317.2 | +212.9 | thigh_link |
| HipYaw2Knee | Yaw2KneePlateB | 109.90 | 97.5 | 0.8874 | 296.7 | +199.2 | thigh_link |
| HipYaw2Knee | HipYaw2Knee_outer | 51.81 | 46.7 | 0.9022 | 139.9 | +93.1 | thigh_link |
| HipYaw2Knee | SupportA | 39.39 | 35.0 | 0.8874 | 106.4 | +71.4 | thigh_link |
| HipYaw2Knee | SupportB | 32.25 | 28.6 | 0.8874 | 87.1 | +58.5 | thigh_link |
| Knee2Ankle | Knee2Ankle_RotorSide_1 | 59.01 | 55.2 | 0.9362 | 159.3 | +104.1 | shin_link |
| Knee2Ankle | Knee2Ankle_RotorSide_2 | 47.69 | 39.3 | 0.8231 | 128.8 | +89.5 | shin_link |
| Knee2Ankle | Knee2Ankle_IdleSide_1 | 48.34 | 42.9 | 0.8874 | 130.5 | +87.6 | shin_link |
| Knee2Ankle | Knee2Ankle_IdleSide_2 | 45.79 | 40.6 | 0.8874 | 123.6 | +83.0 | shin_link |
| Knee2Ankle | Knee2Ankle_Connector_A | 16.53 | 14.7 | 0.8874 | 44.6 | +30.0 | shin_link |
| Knee2Ankle | Knee2Ankle_Connector_B | 16.53 | 14.7 | 0.8874 | 44.6 | +30.0 | shin_link |
| Knee2Ankle | AnkleA2B_A_R | 14.58 | 31.8 | 2.1835 | 39.4 | +7.5 | shin_link |
| Knee2Ankle | AnkleA2B_A_L | 13.05 | 32.0 | 2.4552 | 35.2 | +3.2 | shin_link |

The 33 parts not yet changed, as Al: +2444 g per leg, +4.888 kg for the robot.
Rows marked 'done in M18' are already aluminium in the newest model. AnkleA2B_A_L / _A_R show a density above 0.89 because the `tempmass` step (ledger M06) already gave them aluminium mass from an older volume.
Ankle2Feet parts are split over shin / ankle_pitch / foot / crank / rod by position (massprops_fusion.py); the foot link of the v5 foot models is replaced by the v5 foot, so foot-link parts need a separate decision.
