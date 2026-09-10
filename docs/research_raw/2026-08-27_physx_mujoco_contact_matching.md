# Raw research — PhysX (IsaacLab 5.x) contact matching to MuJoCo for biped sim2sim (2026-08-27, Sonnet subagent)

Context: MuJoCo-trained biped (200 Hz physics / 50 Hz control, solver=Newton, foot=box, friction=1.0, restitution=0)
walks ~18% fast in IsaacSim (vel-tracking error 0.29 m/s vs 0.14 native) after floor friction/restitution, armature,
and torque-speed clamp were already ported. Residual believed to be contact/solver-level. This note is WebSearch/WebFetch
research only — no code was touched.

## 0. Verdict — prioritised recommendation list

| # | Parameter | Value to try | Why | Source |
|---|---|---|---|---|
| 1 | `RigidBodyPropertiesCfg.max_depenetration_velocity` (all robot links, esp. feet/shins) | **1.0 m/s** (not the 100 m/s engine default) | This is the single biggest divergence we found between "default PhysX" and "what NVIDIA's own legged configs actually ship." Every official IsaacLab biped/quadruped asset (G1, H1, ANYmal-C/D, Go2) explicitly overrides the 100 m/s default down to 1.0. At 100 m/s, any residual foot-ground penetration (near-inevitable with a box collider + finite solver iterations) gets corrected by injecting up to 100 m/s of separation velocity in a single substep — a huge, non-physical energy/momentum source that MuJoCo's compliant contact never has (it resolves penetration via a spring-damper over ~solref's time constant, not an instantaneous velocity kick). If your ported robot/USD import did not set this explicitly, it is silently running at the 100x-looser default. This is our top suspect for a systematic forward-speed bias. | `isaaclab_assets/robots/unitree.py` (G1_CFG, H1_CFG, GO2_CFG) and `isaaclab_assets/robots/anymal.py` (ANYMAL_C_CFG, ANYMAL_D_CFG), github.com/isaac-sim/IsaacLab, fetched 2026-08-27; engine default confirmed in `isaaclab.sim.schemas` API docs and IsaacGymEnvs migration doc |
| 2 | `ArticulationRootPropertiesCfg.solver_position_iteration_count` / `solver_velocity_iteration_count` | Try **8 / 4** (G1's values) instead of the lower-iteration quadruped defaults (4/0) if you copied a quadruped config | PhysX's TGS solver "solves friction constraints every iteration," and position-iteration count = number of internal temporal substeps within one `dt`. Too few iterations under-converges the friction/contact LCP each physics step, which manifests as **foot slip during stance** — a very plausible direct cause of "distance traveled per commanded velocity is too high" (foot creeps forward under the robot each step without the policy/controller "seeing" it as slip). NVIDIA's own G1 biped config uses meaningfully higher iterations (8/4) than the ANYmal/Go2 quadruped configs (4/0) — worth matching for a biped. | PhysX 5.4.1 `RigidBodyDynamics` docs (nvidia-omniverse.github.io) — TGS substep explanation; `isaaclab_assets/robots/unitree.py`, `anymal.py` for concrete numbers |
| 3 | `CollisionPropertiesCfg.contact_offset` / `rest_offset` on the foot collider | Set **explicitly** to something matched to the foot box's real thickness (e.g. `contact_offset≈0.005–0.01 m`, `rest_offset≈0.0–0.002 m`) — do not leave unset | NVIDIA's official ANYmal/quadruped configs actually leave these commented out (rely on PhysX auto-computing `contact_offset ≈ 0.02 × min(extents)` of the collider). A separate NVIDIA forum thread flags that an *explicitly authored* leftover value (people have shipped assets with contact_offset as high as ~0.1 m = 10 cm) silently persists and is **not** auto-corrected for geometry scale. MuJoCo's box geom has no such margin by default (`margin=0`, contact generated at literal touch). If your foot's contact_offset ends up large relative to the box's own dimensions, you get an effectively "floating" contact that starts pushing early/softly, changing effective leg length and stance timing — this changes stride length/frequency for a given command, i.e. exactly the overspeed symptom. | `isaaclab.sim.schemas` docs; `isaaclab_assets/robots/anymal.py` (collision_props line explicitly commented out); NVIDIA forum thread "Default contact offset is 10cm?" (forums.developer.nvidia.com/t/default-contact-offset-is-10cm/306623), Sept 2024 |
| 4 | `RigidBodyMaterialCfg.friction_combine_mode` / `restitution_combine_mode` on foot + ground materials | Set to **`"max"`** on both | **Correction to the question's premise**: MuJoCo does *not* use geometric mean for friction combination. Per MuJoCo's own docs, when both geoms have equal `priority` (the default, priority=0), friction is combined via **element-wise maximum** of the two friction vectors (`solref`/`solimp` use a `solmix`-weighted average instead, which is a *different* rule from friction itself). PhysX/IsaacLab's engine default is `"average"`; IsaacLab's own example locomotion configs override the terrain material to `"multiply"` — neither matches MuJoCo's actual `"max"` rule. With foot friction = ground friction = 1.0 exactly, this makes **no numeric difference right now** (max/avg/multiply of 1.0 and 1.0 are all 1.0), so it is *not* the cause of your 18% today — but it's a real, verified mismatch worth fixing before you introduce any friction randomization or per-terrain-patch friction, since it will silently diverge then. | MuJoCo docs `modeling.html#contact-parameters` (fetched 2026-08-27, verbatim quote below); `isaaclab.sim.spawners.materials.physics_materials_cfg` source (default `"average"`); `isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py` (terrain override `"multiply"`) |
| 5 | `SimulationCfg.dt` / `decimation` | Confirm **`sim.dt = 0.005`** (200 Hz) and **`decimation = 4`** (→ 50 Hz control) are actually active in the *deployed* env config, not left at IsaacLab's raw default (`dt = 1/60`) | This is the exact number NVIDIA's own base locomotion env config (`velocity_env_cfg.py`) ships, and it exactly matches your stated MuJoCo setup (200 Hz physics / 50 Hz control). A silent fallback to the un-overridden `SimulationCfg` default of 60 Hz physics with the same decimation would desync control-loop timing from what the policy was trained against, and would also coarsen contact temporal resolution — a classic, boringly-simple cause of a fixed-percentage speed bias. Cheap to verify, so check it before anything more exotic. | `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` (github.com/isaac-sim/IsaacLab, fetched 2026-08-27); `isaaclab.sim.simulation_cfg` default `dt=1/60` |
| 6 | PhysX Scene `bounce_threshold_velocity` | Explicitly set **0.2 m/s** (IsaacGym's original documented default; IsaacLab's `PhysxCfg` default reads `0.5 m/s` in current source) | Below this relative contact velocity, PhysX suppresses restitution regardless of the material's restitution coefficient — a foot strike well above this at push-off/heel-strike (bipeds regularly exceed 0.5 m/s vertically on firm contact) could re-enable a spurious bounce path even with restitution material = 0 in some PhysX code paths. Since your restitution is already matched at 0.0, this is a secondary/defensive check, not a primary suspect. | `isaaclab.sim.simulation_cfg` (`bounce_threshold_velocity: float = 0.5`); IsaacGym `Simulation Tuning` doc via search snippet (0.2 m/s cited as engine default in the Gym-era docs) |
| 7 | `PhysxCfg.friction_correlation_distance` / `friction_offset_threshold` | Check foot box footprint > `2 × friction_correlation_distance` (default 0.025 m → footprint > 5 cm) | PhysX merges contact points closer than `friction_correlation_distance` into one friction anchor. A very small/narrow foot box could have all corner contacts collapsed into a single anchor, changing effective yaw/tangential friction torque distribution vs. MuJoCo's per-corner contact resolution — a secondary contributor to foot slip. Only relevant if your foot box footprint is small; most biped feet (15–25 cm) are safely above this. | `isaaclab.sim.simulation_cfg` defaults (`friction_offset_threshold=0.04`, `friction_correlation_distance=0.025`) |
| 8 (structural, not a single knob) | Treat PhysX-vs-MuJoCo peak GRF comparison as **not apples-to-apples** without accounting for contact model family | MuJoCo's default contact is a compliant spring-damper (`solref = "0.02 1"` → mass-normalized `ke = 1/(timeconst²·dampratio²) = 2500 [1/s²]`, `kd = 2/timeconst = 100 [1/s]` at defaults) that spreads impact momentum transfer over roughly a 20 ms time constant. PhysX's TGS solver is an iterative rigid/impulse-based LCP-family solver (like Bullet/ODE) that resolves penetration much more abruptly unless you explicitly enable IsaacLab's `compliant_contact_stiffness`/`compliant_contact_damping` (implicit-spring contact, off by default at stiffness=0). If you are directly comparing per-strike peak forces, either (a) filter both traces with the same window before comparing peaks, or (b) turn on compliant contact on the foot material and start from the MuJoCo-equivalent mass-normalized stiffness/damping above as a first guess. | MuJoCo docs (`solref`/`solimp` cheat sheet, GitHub newton-physics/newton issue #2009 for the timeconst→stiffness conversion formula); IsaacLab `RigidBodyMaterialCfg.compliant_contact_stiffness/damping` docstrings; general LCP-vs-compliant-contact literature (arXiv:2405.17020, arXiv:2506.14186) |

**Read order if you can only try 3 things**: #1 (`max_depenetration_velocity=1.0`) → #5 (verify dt/decimation actually applied) → #2 (bump solver iterations for a biped) → then #3 (contact/rest offset). #4, #6, #7 are correctness fixes/defensive checks with low expected impact *right now* given your matched friction/restitution values, but fix them anyway since they're cheap and will matter once you add domain randomization.

---

## 1. PhysX solver settings — official defaults and what NVIDIA's own legged configs actually use

### 1.1 Global/scene-level `PhysxCfg` defaults (IsaacLab `isaaclab/sim/simulation_cfg.py`, fetched 2026-08-27)

| Field | Default | Note |
|---|---|---|
| `dt` (SimulationCfg) | `1/60 s` (~0.0167s) | overridden to `0.005` (200 Hz) by locomotion example env configs |
| `solver_type` | `1` = **TGS** (0 = PGS) | TGS is also what all official locomotion examples implicitly use (no override away from default seen) |
| `min_position_iteration_count` / `max_position_iteration_count` | `1` / `255` | actual per-actor values come from `ArticulationRootPropertiesCfg.solver_position_iteration_count` (see §1.2) |
| `min_velocity_iteration_count` / `max_velocity_iteration_count` | `0` / `255` | ditto, per-actor |
| `enable_ccd` | `False` | not enabled in official locomotion examples |
| `bounce_threshold_velocity` | `0.5 m/s` (current IsaacLab source) vs `0.2 m/s` (older IsaacGym-era docs) | version drift between Gym-era and Lab-era defaults — verify which your version ships |
| `friction_offset_threshold` | `0.04 m` | |
| `friction_correlation_distance` | `0.025 m` | |
| GPU buffers (`gpu_max_rigid_contact_count`, etc.) | large powers of 2 | irrelevant to accuracy, only to not crashing at scale |

### 1.2 Per-actor overrides used by NVIDIA's *shipped* legged robot configs (github.com/isaac-sim/IsaacLab `isaaclab_assets/robots/{unitree,anymal}.py`, fetched 2026-08-27)

| Robot | `solver_position_iteration_count` | `solver_velocity_iteration_count` | `max_depenetration_velocity` | `max_linear/angular_velocity` |
|---|---|---|---|---|
| G1 (biped) | **8** | **4** | **1.0** | 1000 / 1000 |
| H1 (biped) | 4 | 4 | **1.0** | 1000 / 1000 |
| Go2 (quadruped) | 4 | 0 | **1.0** | 1000 / 1000 |
| ANYmal-C (quadruped) | 4 | 0 | **1.0** | — |
| ANYmal-D (quadruped) | 4 | 0 | **1.0** | — |

Note the engine/schema-documented default for `max_depenetration_velocity` is **100.0 m/s** (`isaaclab.sim.schemas` docs, IsaacGymEnvs migration doc) — every single official example above explicitly clamps it two orders of magnitude tighter, to 1.0 m/s. This is the strongest, most concrete, most verifiable single lever we found in this research pass.

Collision properties (`contact_offset`/`rest_offset`) are **commented out** in both `anymal.py` configs:
```python
# collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0)
```
i.e. NVIDIA relies on PhysX's geometry-based auto-computation rather than a manual global override, contradicting the commonly-cited "IsaacGym default contact_offset=0.02 / rest_offset=0.01" numbers that come from the *engine* default docs (`isaaclab.sim.schemas`, migration doc) rather than from what shipped robot assets actually use.

### 1.3 TGS solver mechanics (PhysX 5.4.1 docs, nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html, fetched 2026-08-27)

> "TGS subdivides the time step into multiple equally-sized substeps. The number of substeps equals the number of position iterations."
> "Position iterations solve for both geometric and velocity error... Velocity iterations only solve for velocity error."
> Default settings are 4 position and 1 velocity iteration, though these should be increased for "objects with lots of joints and a small tolerance for joint error."
> TGS "solves friction constraints every iteration, whereas PGS solves friction constraints only in the final 3 position iterations by default."

Practical implication: iteration count is not just a stability knob, it changes how well friction (hence foot slip) converges within a single 5 ms physics step. A biped standing/walking with tight joint-error tolerance (arguably the G1/H1 case) benefits from more iterations than a stiffer quadruped chain.

### 1.4 Depenetration-velocity energy-injection concern (community corroboration)

From `github.com/NVIDIA-Omniverse/PhysX` discussion #115 ("Feature request: adjust max depenetration velocity in contact modification callback"), fetched 2026-08-27 — a user reports needing near-zero depenetration velocity precisely to prevent bodies that start interpenetrated from "exploding," i.e. from being kicked with unwanted velocity as PhysX resolves the overlap. This is the same mechanism as our #1 recommendation, just observed from the opposite direction (avoiding *unwanted* velocity injection vs. tolerating it).

---

## 2. MuJoCo's contact model vs PhysX's — verified mechanics

### 2.1 Contact formulation (arXiv:2506.14186, arXiv:2405.17020, general search synthesis)

- PhysX (like Bullet, ODE) formulates rigid contact as an **LCP** (linear complementarity problem) — "hard contact," complementarity strictly enforced, resolved via iterative impulse-based solvers (PGS/TGS family).
- MuJoCo formulates contact as a **convex optimization problem minimizing post-collision kinetic energy**; the complementarity condition *can* be violated, and by default contacts are "soft" — pushing harder always yields proportionally larger (but bounded, non-infinite) acceleration, per a spring-damper analogy parameterized by `solref`/`solimp`.

### 2.2 MuJoCo `solref` defaults, quantified (mujoco.readthedocs.io + github.com/newton-physics/newton issue #2009, both fetched/searched 2026-08-27)

> Default `solref = "0.02 1"` → timeconst = 0.02 s, dampratio = 1 (critically damped).
> Conversion to mass-normalized stiffness/damping: `ke = 1/(timeconst² · dampratio²)` [1/s²], `kd = 2/timeconst` [1/s].
> At defaults: `ke = 1/(0.02² · 1²) = 2500 [1/s²]`, `kd = 2/0.02 = 100 [1/s]`.
> These are **mass-normalized** units, not force-based N/m / N·s/m — true stiffness depends on the contact-frame effective inertia.

This means every MuJoCo contact resolves penetration over roughly a 20 ms window by construction (its default time constant), rather than the PhysX/TGS approach of trying to fully resolve within the current substep bounded only by `max_depenetration_velocity`. This is a structural difference that will show up as **shorter, sharper impulse peaks in PhysX vs. broader, lower peaks in MuJoCo** for the same nominal impact — directly relevant to the "we compare per-strike peak forces" caveat in the question. IsaacLab exposes a matching lever for this: `RigidBodyMaterialCfg.compliant_contact_stiffness`/`compliant_contact_damping` — "spring stiffness/damping for a compliant contact model using implicit springs... only enabled if stiffness > 0" (isaaclab.sim.spawners.materials.physics_materials_cfg docs). This is *off* by default in IsaacLab (stiffness=0.0), meaning **IsaacLab's default is the stiff/rigid PhysX contact, not a MuJoCo-like compliant one**, unless explicitly turned on.

### 2.3 Friction combination rule — MuJoCo (verbatim, mujoco.readthedocs.io/en/stable/modeling.html, fetched 2026-08-27)

> "if one of the two geoms has higher priority, its friction coefficients are used. Otherwise the **element-wise maximum** of each friction coefficient over the two geoms is used."
>
> "If one of the two geoms has higher priority, its solref and solimp parameters are used. If both geoms have the same priority, the weighted average is used. The weights are proportional to the solmix attributes... If solref for either geom is non-positive... the element-wise minimum is used regardless of solmix."
>
> "If one of the two geoms has higher priority, its condim is used. If both geoms have the same priority, the maximum of the two condims is used."

**This directly falsifies the question's premise that MuJoCo uses a geometric mean for friction.** Friction specifically uses element-wise **max** (at equal priority, the common case); `solref`/`solimp` (contact *softness*, not friction) use a `solmix`-weighted average — a completely separate parameter family. PhysX/IsaacLab's `RigidBodyMaterialCfg.friction_combine_mode` engine default is `"average"` (isaaclab source, fetched 2026-08-27); IsaacLab's own example terrain config overrides this to `"multiply"` — neither replicates MuJoCo's actual `"max"` rule. As noted in the verdict table, this makes no numeric difference while both your surfaces are pinned at friction=1.0, but is worth fixing for correctness before any friction randomization.

### 2.4 Friction cone shape and slip suppression (mujoco.readthedocs.io/en/stable/computation, github.com/google-deepmind/mujoco discussion #656, fetched 2026-08-27)

> MuJoCo supports both pyramidal and elliptic friction cones ("elliptic cones correspond more closely to physical reality"). "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large `impratio`, and the Newton algorithm with very small tolerance. If that is not sufficient, enable the Noslip solver."

Your stated MuJoCo config uses the default Newton solver — worth checking whether your model also uses non-default `impratio`/elliptic cones/Noslip, since if it does, your MuJoCo baseline is already slip-suppressed in a way PhysX has no direct equivalent for (PhysX's friction-cone handling is a fixed part of its LCP solve, not separately tunable the same way). This is a possible source of asymmetric slip behavior — worth checking your MuJoCo XML's `<option>` block for these flags before assuming PhysX is "the only side that's wrong."

---

## 3. Known sim2sim reports — what we could and couldn't confirm

- We could **not** find a public, quantified report of "MuJoCo→Isaac systematically runs N% fast" with a documented fix — this appears to be a case where the community's failure mode is usually the opposite direction (Isaac Gym/Lab → MuJoCo, checking whether an Isaac-trained policy survives MuJoCo's stricter dynamics), not what you're doing.
- `github.com/isaac-sim/IsaacLab discussions/2894` ("Sim2Sim Transfer (Isaaclab/Isaacsim to Mujoco)"): thread is a dead end — OP reported "tried it vanilla policy in mujoco but results are not same," a maintainer asked for details, OP later marked it "Completed" with **no technical detail ever posted**. Not usable as an evidence source beyond confirming the problem class is recognized as real by the community.
- `github.com/isaac-sim/IsaacLab discussions/2637` ("Deploying trained humanoid policy in Isaac Sim"): concrete failure mode found here was **joint indexing/ordering mismatch** (13 trained leg joints vs. 37-DOF USD articulation, causing shape mismatches and conflicting joint-index derivations) — not a contact/solver issue, but worth ruling out in your own setup: confirm your ported joint order and `joint_indices` exactly match training, since an indexing bug can also masquerade as "everything is slightly wrong" including apparent speed.
- `github.com/isaac-sim/IsaacLab discussions/4121` ("PD Torque Behavior... Actions Too High"): root cause was a **logging/interpretation** issue (policy output and desired position were the same tensor, τ≈Kp(q_des−q) looked like a bug but wasn't) — not directly applicable, but the maintainer's side recommendations (moderate action scale, check `DCMotorCfg` vs `IdealPDActuatorCfg` if you don't need torque-speed saturation) are a sanity check worth applying given you've already ported a torque-speed clamp: confirm you're using `DCMotorCfg` (not `IdealPDActuatorCfg`) so the clamp is actually active in the actuator model, not bypassed.
- General community sentiment (multiple aggregator/blog sources, lower confidence, not verified against primary docs): "PhysX demonstrates inferior physical fidelity compared to MuJoCo's solver, potentially exacerbating the Sim2Real gap" — directionally consistent with what we found in §2, but these are secondary blog sources, not authoritative.
- No quantitative GRF/peak-force comparison paper specific to PhysX-vs-MuJoCo legged contact was found; the closest primary academic source remains Erez/Tassa et al. "Simulation tools for model-based robotics: Comparison of Bullet, Havok, MuJoCo, ODE and PhysX" (2015) — old (pre-TGS PhysX, pre-IsaacLab), cited by search summaries as finding "MuJoCo... fastest and most accurate on constrained systems relevant to robotics," but we did not pull primary numeric tables from it in this pass.

---

## 4. Python API — where to set these in Isaac Sim 5.x headless

Two viable layers, in order of what we'd actually recommend:

### 4.1 Recommended: IsaacLab config-object layer (works headless, what all the above defaults/examples use)

```python
from isaaclab.sim import schemas as sim_utils  # isaaclab.sim.schemas module

# On the robot's ArticulationCfg.spawn:
spawn=sim_utils.UsdFileCfg(
    ...,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        max_depenetration_velocity=1.0,           # <- #1 in verdict table
        ...
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        solver_position_iteration_count=8,          # <- #2
        solver_velocity_iteration_count=4,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=0.01,                        # <- #3
        rest_offset=0.0,
    ),
)

# Material (foot + ground), via RigidBodyMaterialCfg:
physics_material=sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
    friction_combine_mode="max",        # <- #4, matches MuJoCo's actual rule
    restitution_combine_mode="max",
    compliant_contact_stiffness=0.0,    # set >0.0 to enable MuJoCo-like soft contact, see §2.2
    compliant_contact_damping=0.0,
)

# Scene-level, on SimulationCfg.physx (PhysxCfg):
sim.physx.bounce_threshold_velocity = 0.2            # <- #6
sim.physx.friction_correlation_distance = 0.025      # <- #7, check vs. foot footprint
sim.dt = 0.005                                       # <- #5
sim.render_interval = decimation  # decimation = 4 for 50 Hz control
```
Class/field names verified against `isaaclab.sim.schemas` API docs and `isaaclab.sim.simulation_cfg` source (both fetched 2026-08-27) and the concrete `unitree.py`/`anymal.py` robot configs.

### 4.2 Lower-level: raw USD/PhysX schema attributes (if you're bypassing IsaacLab's config wrapper for a lean headless playback script)

These are the underlying USD attribute names IsaacLab's config classes write to (standard Omniverse PhysX schema, moderate confidence — not independently re-verified against the 5.0 schema header in this pass, but consistent across PhysX/Omniverse versions 2022–2025):

- `PhysxSchema.PhysxRigidBodyAPI` → `physxRigidBody:solverPositionIterationCount`, `physxRigidBody:solverVelocityIterationCount`, `physxRigidBody:maxDepenetrationVelocity`, `physxRigidBody:enableCCD`
- `PhysxSchema.PhysxCollisionAPI` → `physxCollision:contactOffset`, `physxCollision:restOffset`, `physxCollision:torsionalPatchRadius`
- `PhysxSchema.PhysxSceneAPI` (on the `/PhysicsScene` prim) → `physxScene:timeStepsPerSecond`, `physxScene:bounceThreshold`, `physxScene:frictionOffsetThreshold`, `physxScene:frictionCorrelationDistance`, `physxScene:solverType` (token, `"TGS"`/`"PGS"`)
- `PhysxSchema.PhysxMaterialAPI` → `physxMaterial:frictionCombineMode`, `physxMaterial:restitutionCombineMode` (token enum: `average`/`min`/`multiply`/`max`), `physxMaterial:compliantContactStiffness`, `physxMaterial:compliantContactDamping`
- `UsdPhysics.MaterialAPI` (base, non-PhysX-specific) → `physics:staticFriction`, `physics:dynamicFriction`, `physics:restitution`

Search example confirming the combine-mode call pattern (search snippet, not independently re-verified against source):
```python
physxMaterialAPI = PhysxSchema.PhysxMaterialAPI.Apply(materialPrim)
physxMaterialAPI.CreateFrictionCombineModeAttr().Set(PhysxSchema.Tokens.min)  # or "max"
physxMaterialAPI.CreateRestitutionCombineModeAttr().Set(PhysxSchema.Tokens.multiply)
```
The `isaacsim.core.prims.Articulation`/`isaacsim.core.api.PhysicsContext` Python wrapper classes (5.x) also expose getter/setter pairs following the pattern `get_solver_position_iteration_counts()/set_solver_position_iteration_counts()`, `get_max_depenetration_velocities()/set_max_depenetration_velocities()`, `get_contact_offsets()/set_contact_offsets()`, `get_rest_offsets()/set_rest_offsets()`, `get_friction_coefficients()/set_friction_coefficients()` on `Articulation`/`RigidPrimView`, and `PhysicsContext.enable_ccd()` at the scene level — **this pattern is consistent with what these API pages show, but the two docs pages we fetched (`isaacsim.core.prims`, `isaacsim.core.api` 5.0/5.1) both returned truncated content, so treat the exact method names in this paragraph as moderate-confidence, worth confirming with `dir()`/`help()` against your actual installed 5.x build rather than copy-pasting blind.**

---

## Sources

- [isaaclab.sim.schemas — Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html)
- [From IsaacGymEnvs — Isaac Lab Documentation (migration guide)](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
- [Comparing Simulations Between Isaac Gym and Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/migration/comparing_simulation_isaacgym.html)
- [isaaclab.sim.spawners.materials.physics_materials_cfg source](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sim/spawners/materials/physics_materials_cfg.html)
- [isaaclab_assets/robots/unitree.py (G1_CFG, H1_CFG, GO2_CFG)](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_assets/isaaclab_assets/robots/unitree.py)
- [isaaclab_assets/robots/anymal.py (ANYMAL_C_CFG, ANYMAL_D_CFG)](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_assets/isaaclab_assets/robots/anymal.py)
- [isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- [isaaclab.sim.simulation_cfg source (via mirror)](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/sim/simulation_cfg.py)
- [NVIDIA PhysX 5.4.1 Rigid Body Dynamics docs](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html)
- [Feature request: adjust max depenetration velocity in contact modification callback — NVIDIA-Omniverse/PhysX Discussion #115](https://github.com/NVIDIA-Omniverse/PhysX/discussions/115)
- [Default contact offset is 10cm? — Isaac Sim NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/default-contact-offset-is-10cm/306623)
- [MuJoCo Modeling docs — contact parameter combination rules](https://mujoco.readthedocs.io/en/stable/modeling.html)
- [MuJoCo Computation docs — friction cones](https://mujoco.readthedocs.io/en/stable/computation/index.html)
- [Mujoco impratio — google-deepmind/mujoco Discussion #656](https://github.com/google-deepmind/mujoco/discussions/656)
- [shape_material_ke/kd units mismatch — newton-physics/newton Issue #2009 (solref→stiffness conversion)](https://github.com/newton-physics/newton/issues/2009)
- [solref/solimp Parameter Cheat Sheet — ROBOLAWEB Docs](https://robolaweb.gitbook.io/robolaweb-docs/basic-concept/solref-solimp-parameter-cheat-sheet)
- [Hard Contacts with Soft Gradients (arXiv:2506.14186)](https://arxiv.org/html/2506.14186v1)
- [From Compliant to Rigid Contact Simulation (arXiv:2405.17020)](https://arxiv.org/pdf/2405.17020)
- [Sim2Sim Transfer (Isaaclab/Isaacsim to Mujoco) — isaac-sim/IsaacLab Discussion #2894](https://github.com/isaac-sim/IsaacLab/discussions/2894)
- [Deploying trained humanoid policy in Isaac Sim — isaac-sim/IsaacLab Discussion #2637](https://github.com/isaac-sim/IsaacLab/discussions/2637)
- [PD Torque Behavior in Isaac Lab — isaac-sim/IsaacLab Discussion #4121](https://github.com/isaac-sim/IsaacLab/discussions/4121)
- [isaacsim.core.prims API docs (5.0.0, truncated)](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/py/source/extensions/isaacsim.core.prims/docs/index.html)
- [isaacsim.core.api API docs (5.1.0, truncated)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html)
- [Simulation tools for model-based robotics: Comparison of Bullet, Havok, MuJoCo, ODE and PhysX (Erez/Tassa et al., 2015) — cited via search summary only, not directly pulled](https://www.researchgate.net/publication/280944465_Simulation_tools_for_model-based_robotics_Comparison_of_Bullet_Havok_MuJoCo_ODE_and_PhysX)

## Gaps / not found in this pass
- No primary source gave a directly-quantified "X% overspeed, fixed by Y" MuJoCo→Isaac case study; §0/§1's recommendations are inferred from (a) verified parameter-default deltas between engine defaults and NVIDIA's own shipped configs, and (b) documented solver mechanics — not from a report of someone hitting this exact symptom and confirming the same fix. Treat as a prioritised hypothesis list, verify empirically by toggling one at a time (start with #1).
- Could not confirm exact 5.x `isaacsim.core.prims`/`isaacsim.core.api` method signatures from primary docs (both fetches truncated) — confirm against the installed build before relying on §4.2's method names verbatim.
- Did not find a primary-source numeric GRF/peak-force comparison table between PhysX and MuJoCo for legged contact specifically.
