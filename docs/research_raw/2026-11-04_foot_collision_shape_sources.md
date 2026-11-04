# Raw excerpts: foot collision shape (mesh pieces vs sphere/capsule primitives) — fetched 2026-11-04 (task #191)
Question (user): is the 29-convex-piece foot collision the best choice? Penetration / boundary bugs may destabilise the policy. Does a sphere/capsule foot work on rough terrain?
Evidence grades: P = primary (engine docs/source read in full this session), L = local measurement in this repo, S = search-result summary only (page not read in full; do not cite as verified).

## P1. MuJoCo docs, "Computation > Collision detection" (raw .rst of google-deepmind/mujoco main, doc/computation/index.rst, lines 1816-1930)
https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection
- "MuJoCo supports several primitive geometric shapes: plane, sphere, capsule, cylinder, ellipsoid, and box. It also supports triangulated meshes and height-fields."
- "collision detection is limited to *convex* geoms ... Meshes specified by the user can be non-convex, and are rendered as such. For collision purposes however they are replaced with their convex hulls ... computed by the qhull library."
- "All collisions involving pairs of geoms that do not have an analytic collider (e.g., meshes), are handled by one of two general-purpose convex collision detection (CCD) pipelines" (native GJK/EPA, default; legacy libccd MPR).
- "Standard general-purpose convex collision algorithms like MPR and GJK/EPA always return a single contact point, which is problematic for surface contact scenarios (e.g., box-stacking)." "the capsule-plane and box-plane colliders can return up to two or four contacts".
- "In order to model a non-convex object other than a height field, the user must decompose it into a union of convex geoms (which can be primitive shapes or meshes) and attach them to the same body." Tools named: CoACD.
=> The 29 strips are exactly this "convex decomposition" route; mesh-vs-box terrain goes through the single-contact general CCD path, primitives (sphere/capsule/box vs plane/box) through analytic colliders.

## P2. mujoco_warp source in this repo's mjlab venv: mujoco_warp/_src/collision_primitive.py (read 2026-11-04)
_PRIMITIVE_COLLISIONS keys: (PLANE,SPHERE) (PLANE,CAPSULE) (PLANE,ELLIPSOID) (PLANE,CYLINDER) (PLANE,BOX) (PLANE,MESH)=plane_convex_wrapper (SPHERE,SPHERE) (SPHERE,CAPSULE) (SPHERE,CYLINDER) (SPHERE,BOX) (CAPSULE,CAPSULE) (CAPSULE,BOX) (BOX,BOX).
No (MESH,BOX) entry => mesh-vs-box terrain uses the generic convex (GJK/EPA) path. Sphere-box and capsule-box are analytic.
mujoco_warp README (github.com/google-deepmind/mujoco_warp) lists unsupported features (PGS solver, implicitfast midpoint ...); no statement about mesh-box robustness.

## L1. This repo
- mjlab terrain used for training/eval ("BOX_TERRAINS_CFG", mujoco-sim/mjlab/src/mjlab/terrains/config.py:330-352): flat, pyramid stairs, box_random_grid (0-0.12 m), tilted_grid (15 deg) - all boxes; comment records that MuJoCo-Warp heightfield contact misbehaves for the Pygmalion foot (flat heightfield tile: 77-100 % collapse vs 0 % on flat box tile; docs/experiments/2026-10-11_flat_vs_rough_policy_compare.md §2). Foot-shape study stones/pebbles are box-built.
- Feet_design/.../notes/발바닥 v3 충돌 모델과 sim 비교.md §3 (flat ground, policy 49999, 8 envs, CPU): base box foot / v3full (15 mesh pieces) / v3sph (12 spheres): falls 0/8 all; sensor penetration P99 28.6 / 12.1 / 5.9 mm; slip P95 184 / 96 / 42 mm/s; GRF P99 3.27 / 2.97 / 2.57 BW; contact points per stance foot 2.0 / 4.4 / 5.9; CoP y inner/outer P99 50/-50, 57/-57, 46/-18 mm (v3sph covers the outer edge poorly).
- No sphere/capsule foot variant has been evaluated on stones/pebbles/static bump (eval tags: base, v3full, v5full, v5fullm, v5_*, orig only; static_bump.csv variants base, v5fullm, v5_* only).
- Current v5full/v5fullm foot: 3 upper boxes + 29 convex meshes; piece thickness down to 1.2-3.6 mm, up to 956 hull vertices; flat-walk penetration P99 9.7 mm (v5full) exceeds the thinnest pieces.

## P3. Public humanoid models (docs/research_raw/2026-10-31_menagerie_foot_collision_survey.md, fetched from raw main files)
H1: 3 capsules r=14 mm; G1: 4 spheres r=5 mm (menagerie), mjlab G1 7 capsules; T1: 1 box; Berkeley Humanoid: 1 box + 1 cylinder; Cassie: 1 capsule; none uses a many-piece mesh sole.

## S. Search-result summaries (NOT read in full; verify before citing)
- MuJoCo jitter write-up (q-refler-github-io.pages.dev, 2024-03-06): single contact between a mesh hull with large flat faces and a box must "run around over time steps"; suggested plane instead of box, or decompose into sub-meshes.
- NVIDIA SimReady MuJoCo Mesh Collision API (nvidia.github.io/simready-foundation/...mujoco-mesh-collision-api.html): MuJoCo mesh collision is convex-hull based.
- MoveIt Pro "optimizing robot model meshes" (docs.picknik.ai): MuJoCo hull fills voids; decompose with CoACD, start ~5-10 hulls.
- PhysX 5 "Advanced Collision Detection" (nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/AdvancedCollisionDetection.html): contactOffset/restOffset; larger offsets raise contact count; tunnelling mitigations.
- Bullet SDK manual (Coumans): primitives (sphere, capsule) have zero margin and are precise; hulls only when no good primitive approximation exists.
- NimbRo-OP2X, arXiv 2010.09308: capsule checks are faster, boxes sometimes needed for feet. LadderMan arXiv 2606.05873: feet/shins/hands as simple primitives are adequate on flat ground, cause sim-to-real gaps in contact-rich tasks. Gong et al. arXiv 2508.11885: many studies model the foot as one box or capsule.
- No paper found that directly compares sphere/capsule vs convex-decomposition foot collision for humanoid RL on rough terrain.
