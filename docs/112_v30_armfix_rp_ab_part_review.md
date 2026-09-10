# Pygmalion v30 arm fix, RP/AB contract, and material-body review

Date: 2026-09-01  
Model tag: `prototype-tempmass-rs03measured-armfix_v30`  
Decision: **CONDITIONAL GO for prototype progress only / NO-GO for final Sim2Real**

## ELI5

로봇 그림을 여러 종이 조각으로 나눴다고 생각하면, 어깨 껍데기가 팔 종이가 아니라 몸통 종이에 잘못 붙어 있었습니다. 그 껍데기를 팔 종이로 옮겨 이제 팔과 함께 15도 돌아갑니다. AB는 발목 모터 A/B를 직접 움직이고, RP는 발목 pitch/roll을 가상으로 움직이는 비교 실험입니다. 두 방식 모두 Policy 버튼은 정확히 12개이며 상체 무게는 남지만 상체 관절은 잠깁니다. 다만 알루미늄 실물 10개가 CAD의 어느 10개인지는 아래 그림을 보고 마지막 확인이 필요합니다.

## Decision table

| Criterion | Evidence state | Evidence | Consequence | Next action |
|---|---|---|---|---|
| Circled shoulder housing follows arm roll | PASS | before/after render; two `Shoulder-Roll2Yaw-dummy` bodies moved to `arm` | Semi pose is visually coherent | reviewer inspects image |
| Three geometry variants | PASS | Full 17 movable; Semi 13 movable + 4 fixed; LegOnly 12 movable, no upper mesh/mass | variant contract can be reviewed | keep validation JSON with artifacts |
| AB 12-action motor contract | PASS | exact action order; 4 closure equalities; 35.0491 kg; actor 45 | model/runtime contract is internally consistent | training launcher must use the recorded selectors |
| RP comparison contract | PASS | exact 12 serial actions; pose-dependent `AnkleRpTnActuator`; crank-space clamp 35.69 < 60 N·m | valid comparison arm, not final hardware topology | use only as RP baseline |
| Upper body welded but retained | PASS | no waist/shoulder joints or actions at runtime; retained upper mass 12.24226 kg | matches final 12-action walking contract | keep `PYG_UPPER_DOF` unset |
| RS03 mass provenance | PASS | 0.9195 kg user scale measurement; source in code, massprops JSON, MJCF/URDF XML comment | all 10 installed RS03 representations use measured mass | later attach scale photo/date/serial if available |
| Printed/aluminium inventory | PASS with provenance note | 55 rows and 55 solo images; 51 current v30 cache, four tiny JS06 flange renders from exact-name legacy cache | every material-affected body is reviewable | refresh four flange meshes on next Fusion export |
| Physical aluminium ten-part one-to-one map | FAIL / unresolved | current wildcard produces 9 CAD bodies per leg-equivalent, while physical statement requires 5 per leg | final mass/COM/inertia cannot be certified | visually confirm one of the two mappings or name corrected bodies |

## 1. Shoulder housing fix

Root cause: `massprops_fusion.classify()` assigned the visible
`Shoulder-Roll2Yaw-dummy::{Body1,Body2}` housing to `shoulder_pitch_link`.  SemiFull welds
shoulder roll at 15 degrees; therefore the rod rotated but this housing remained on the
upstream link.  Both bodies are now classified as `arm`, downstream of shoulder roll.

- Review image: `tools/robot_model/fusion_snapshots/v30_inspection/renders_armfix/arm_housing_before_after.png`
- Full three-variant sheet: `tools/robot_model/fusion_snapshots/v30_inspection/renders_armfix/Pygmalion_v30_variants_all_views.png`
- Full/Semi upper masses after split: shoulder-pitch link 1.501 kg per side; arm 0.667 kg per side.
- Total mass is unchanged by the ownership correction: 35.049 kg.

The ownership correction changes the link COM/inertia split, not just appearance.  That is
why moving only the STL would have been wrong; mass aggregation and mesh grouping were both
regenerated.

## 2. RP/AB branching and authority

| Mode | Authoritative physics | Per-leg policy actions | Passive coordinates | Closure | Runtime selector |
|---|---|---|---|---|---|
| AB | closed-loop MJCF | hip pitch/roll/yaw, knee, crank A, crank B | ankle pitch/roll and rod universal hinges | 4 `equality/connect`, two per leg | `PYG_ANKLE_MODE=AB`, `PYG_MODEL_TAG=prototype-tempmass-rs03measured-armfix_v30` |
| RP | serial MJCF/URDF comparator | hip pitch/roll/yaw, knee, ankle pitch, ankle roll | none in action chain | none; torque transformed/clamped through saved 2-RSU Jacobian | `PYG_ANKLE_MODE=RP`, `PYG_V2_XML=prototype-tempmass-rs03measured-armfix_v30.xml` |

Standard URDF is a tree format and cannot be the authoritative representation of a closed
2-RSU loop.  The AB URDF is useful for names/visualization, but the four closure constraints
live in MJCF and the AB training/deployment contract must be verified against that MJCF.

Exact AB action order:

1. `L_hip_pitch_joint`
2. `L_hip_roll_joint`
3. `L_hip_yaw_joint`
4. `L_knee_joint`
5. `L_crank_A_joint`
6. `L_crank_B_joint`
7. `R_hip_pitch_joint`
8. `R_hip_roll_joint`
9. `R_hip_yaw_joint`
10. `R_knee_joint`
11. `R_crank_A_joint`
12. `R_crank_B_joint`

AB CPU smoke evidence on the new model:

- Action dimension 12 and exact order above.
- Actor observation dimension 45: angular velocity 3 + projected gravity 3 + two frames of 12 motor positions 24 + previous action 12 + command 3.
- Critic dimension 68; passive foot pose/velocity and privileged signals remain critic-side.
- Four equality constraints are present: `L_loop_A/B`, `R_loop_A/B`.
- Dynamic closure error was 0.055–0.061 mm in the two-second/action smoke run.
- Total model mass 35.0491 kg; upper joints absent after weld; upper geometry/mass retained.

RP evidence:

- Action dimension 12 with ankle pitch/roll replacing crank A/B.
- Actor is the same deployable 45-dimensional contract; critic dimension is 60 because no passive loop hinges are present.
- External-torque clamp test observed worst transformed crank torque 35.69 N·m against a 60 N·m per-crank limit: PASS.

Machine-readable evidence:

- `tools/robot_model/fusion_snapshots/v30_inspection/runtime_contract/AB_runtime_contract.json`
- `tools/robot_model/fusion_snapshots/v30_inspection/runtime_contract/RP_runtime_contract.json`

## 3. RS03 measured-mass correction

Nominal RS03 mass is now **0.9195 kg**, from the user's physical scale measurement.  The
0.880 kg catalogue value remains only in the source comment for traceability.  The complete
Full/Semi model contains ten RS03 units: two hip-yaw, four ankle, and four shoulder motors.
Therefore the corrected total increased by exactly
`10 × (0.9195 - 0.880) = 0.395 kg`, from 34.654 kg to 35.049 kg.

The provenance is written in:

- `tools/robot_model/massprops_fusion.py` (`MOTOR_MASS_SOURCE`)
- generated massprops `motor_mass_sources.RS03`
- generated URDF/MJCF root XML comment

## 4. Material-body inventory

The union is 53 printed-density bodies plus two existing-aluminium push rods not present in
that printed list, for **55 unique CAD body rows**.  Each row contains: stable CAD path,
owner link, standalone frame-filling render, volume, current density, current body-level
mass input, Al6061 equivalent mass, inferred physical multiplicity, mapping membership,
notes, and rendering provenance.

- Browser table: `tools/robot_model/fusion_snapshots/v30_inspection/part_inventory/index.html`
- Spreadsheet with embedded images: `tools/robot_model/fusion_snapshots/v30_inspection/part_inventory/part_inventory.xlsx`
- Canonical JSON: `tools/robot_model/fusion_snapshots/v30_inspection/part_inventory/part_inventory.json`

Density colour runs from print-density yellow to Al6061 blue.  “URDF 집계 입력” is explicitly
named that way because these CAD bodies are aggregated into rigid URDF links; it is not a
claim that every row is an independently named URDF link.

## 5. Why the physical ten do not match the current CAD manifest

Current wildcard interpretation, one modeled leg:

- `AnkleA2B_A_L`, `AnkleA2B_A_L (1)`
- `AnkleA2B_A_R`, `AnkleA2B_A_R (1)`
- `AnkleA2B_A_Support`, `AnkleA2B_A_Support (1)`
- `Arm_A`, `Arm_B`
- `FEET`

That is **nine CAD bodies**, mirrored to **18 physical placements**.  It cannot literally be
the user's “four connector parts per leg plus one foot per leg = ten total.”  In addition,
`Arm_A/B` are the two push rods and the workbook already labels them existing aluminium;
they should not automatically be counted as newly replaced connector bodies.

Current-map visual:
`tools/robot_model/fusion_snapshots/v30_inspection/part_inventory/current_tempmass_mapping.png`

The current leading physical-function candidate is, per leg:

- `KneeB2AnkleA`
- `KneeB2AnkleB`
- `AnkleBraceA`
- `AnkleBraceB`
- `FEET`

This is five bodies per leg and exactly ten over both legs.  It is a **candidate**, not yet a
promoted fact, because only inspection of the ten parts or explicit user confirmation can
establish one-to-one identity.

Candidate-map visual:
`tools/robot_model/fusion_snapshots/v30_inspection/part_inventory/leading_10part_mapping.png`

## 6. 80-point red-team completion

What was initially easy to miss and is now covered:

- The circled shoulder part needed both mesh and inertia ownership correction.
- `PYG_UPPER_DOF=0` used to be truthy; boolean parsing now treats `0/false/no/off` as false.
- Candidate XMLs previously could silently compile against the production `assets_v2`
  symlink; staged builds now have a selectable mesh namespace (`PYG_MJCF_MESHDIR`).
- Loop mass reporting omitted separately split crank/rod bodies even though the MJCF held
  them; serial and loop totals now both report 35.049 kg.
- AB and RP must be tested in fresh processes because selection flags resolve at import.
- A closed-loop AB URDF must not be mistaken for the authoritative AB physics model.
- Four small JS06 flange rows use transparent legacy-render provenance, not falsely labeled
  v30 meshes.

Remaining hard blockers for final Sim2Real:

1. Visual/physical confirmation of the five-per-leg aluminium map.
2. Replace temporary worksheet-derived aluminium masses with a signed physical manifest if
   individual scale readings exist; retain body path, side, serial/photo, date, and scale.
3. Hardware-level AB sign/zero/gear mapping test: command one motor at low current while
   logging both crank encoders and reconstructed ankle pitch/roll.
4. Latency/dropout/timestamp measurement for EBIMU and motor buses, followed by matching DR.
5. Locked replay digest proving the deployment code uses the exact 12-name order recorded
   above.

Until item 1 is confirmed, a `prototype-tempmass` run may be used to inspect learning
progress, but it must not be selected as the final Sim2Real checkpoint.
