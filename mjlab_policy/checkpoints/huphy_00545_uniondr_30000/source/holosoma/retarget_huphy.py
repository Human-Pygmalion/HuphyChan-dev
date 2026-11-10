#!/usr/bin/env python3
"""Retarget an SMPL-X motion to the closed-loop HUPHY leg-only robot.

Two independent solvers are produced:
  * InteractionMesh with nine lower-body keypoints (no arm/torso keypoints).
  * GMR/Mink with the same nine targets and MuJoCo connect equalities enforced.

The source HUPHY XML is never edited.  A derived XML adds two massless toe
keypoint bodies and makes mesh paths absolute so it can live in the output dir.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mink
import mujoco
import numpy as np
import smplx
import torch
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting import params as gmr_params
from holosoma_retargeting.src.interaction_mesh_retargeter import (
    InteractionMeshRetargeter,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/jooyoungbae/mocap_dataset/data/smplx/00399.npz")
DEFAULT_BODY_MODELS = Path("/home/jooyoungbae/mocap_dataset/models")
DEFAULT_HUPHY_XML = Path(
    "/home/jooyoungbae/HuphyChan-dev/pygmalion_locomotion/assets/"
    "pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_"
    "proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "00399_huphy_retarget"

# SMPL-X indices used by both methods.  This intentionally replaces the usual
# 15 whole-body targets with lower-body-only targets.
LOWER_BODY = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "L_Knee": 4,
    "R_Knee": 5,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "L_Foot": 10,
    "R_Foot": 11,
}
SMPL_TO_GMR = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip",
    "R_Hip": "right_hip",
    "L_Knee": "left_knee",
    "R_Knee": "right_knee",
    "L_Ankle": "left_ankle",
    "R_Ankle": "right_ankle",
    "L_Foot": "left_foot",
    "R_Foot": "right_foot",
}
LOWER_BODY_LINKS = {
    "Pelvis": "base_link",
    "L_Hip": "L_hip_pitch_link",
    "R_Hip": "R_hip_pitch_link",
    "L_Knee": "L_shin_link",
    "R_Knee": "R_shin_link",
    "L_Ankle": "L_ankle_pitch_link",
    "R_Ankle": "R_ankle_pitch_link",
    "L_Foot": "L_toe_keypoint",
    "R_Foot": "R_toe_keypoint",
}

# Closed-loop nominal pose copied from the HUPHY MJLab robot definition.  The
# passive rod and ankle values are essential; a zero pose violates the 2-RSU
# connect equalities.
HUPHY_INIT_JOINTS = {
    "L_hip_pitch_joint": 0.175,
    "R_hip_pitch_joint": -0.175,
    "L_knee_joint": -0.35,
    "R_knee_joint": 0.35,
    "L_crank_A_joint": -0.1438849965517705,
    "L_rod_A_u1": 0.1479036260054202,
    "L_rod_A_u2": -1.6276083554340244e-07,
    "L_crank_B_joint": 0.14687020555452607,
    "L_rod_B_u1": -0.1487249952991747,
    "L_rod_B_u2": 2.412252153998891e-07,
    "L_ankle_pitch_joint": 0.1750072557823706,
    "L_ankle_roll_joint": 4.703560734910088e-06,
    "R_crank_A_joint": 0.14395487369574003,
    "R_rod_A_u1": -0.14797270816558003,
    "R_rod_A_u2": 2.6819833875474064e-06,
    "R_crank_B_joint": -0.14706307530170212,
    "R_rod_B_u1": 0.14892320455004768,
    "R_rod_B_u2": -3.976385353225781e-06,
    "R_ankle_pitch_joint": 0.17516470240179116,
    "R_ankle_roll_joint": -7.751934743886312e-05,
}
MOTOR_JOINTS = (
    "R_hip_pitch_joint", "R_hip_roll_joint", "R_hip_yaw_joint",
    "R_knee_joint", "R_crank_A_joint", "R_crank_B_joint",
    "L_hip_pitch_joint", "L_hip_roll_joint", "L_hip_yaw_joint",
    "L_knee_joint", "L_crank_A_joint", "L_crank_B_joint",
)

# Native SMPL coordinates -> HUPHY coordinates.  The dataset's stored joints
# are native (-x, z, y), then HUPHY uses +x forward and +y left, yielding
# (native z, native x, native y).
NATIVE_TO_HUPHY = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
STORED_TO_HUPHY = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--body-models", type=Path, default=DEFAULT_BODY_MODELS)
    parser.add_argument("--huphy-xml", type=Path, default=DEFAULT_HUPHY_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("both", "interactionmesh", "gmr"), default="both")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--interaction-initial-iters", type=int, default=30)
    parser.add_argument("--interaction-iters", type=int, default=5)
    parser.add_argument("--foot-sticking", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def build_derived_xml(source: Path, destination: Path, add_floor: bool = False) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        # Do not resolve this path: meshes_v30_armfix is a symlink and several
        # mesh filenames intentionally use ../meshes_footv5 relative to the
        # symlink's parent, not relative to its resolved target.
        compiler.set("meshdir", str((source.parent / meshdir).absolute()))
    for side in ("L", "R"):
        foot = root.find(f".//body[@name='{side}_foot_link']")
        if foot is None:
            raise ValueError(f"Missing {side}_foot_link in {source}")
        marker = ET.SubElement(
            foot,
            "body",
            name=f"{side}_toe_keypoint",
            pos="0.14 0 -0.043",
        )
        ET.SubElement(marker, "site", name=f"{side}_toe_target", size="0.004")
    if add_floor:
        worldbody = root.find("worldbody")
        assert worldbody is not None
        ET.SubElement(
            worldbody,
            "geom",
            name="retarget_floor",
            type="plane",
            size="20 20 0.1",
            rgba="0.82 0.84 0.88 1",
            contype="1",
            conaffinity="1",
        )
        ET.SubElement(
            worldbody,
            "light",
            name="retarget_key",
            pos="0 -2 4",
            dir="0 0 -1",
            diffuse="0.8 0.8 0.8",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")


def initial_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64)
    q[3] = 1.0
    for name, value in HUPHY_INIT_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Missing HUPHY joint: {name}")
        q[int(model.jnt_qposadr[jid])] = value
    return q


def body_positions(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return {
        name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
        for name in LOWER_BODY_LINKS.values()
    }


def estimate_scale(stored_joints: np.ndarray, model: mujoco.MjModel, q0: np.ndarray) -> float:
    robot = body_positions(model, q0)
    robot_lengths = []
    human_lengths = []
    for side in ("L", "R"):
        r_names = [
            LOWER_BODY_LINKS[f"{side}_Hip"], LOWER_BODY_LINKS[f"{side}_Knee"],
            LOWER_BODY_LINKS[f"{side}_Ankle"], LOWER_BODY_LINKS[f"{side}_Foot"],
        ]
        h_ids = [LOWER_BODY[f"{side}_Hip"], LOWER_BODY[f"{side}_Knee"],
                 LOWER_BODY[f"{side}_Ankle"], LOWER_BODY[f"{side}_Foot"]]
        robot_lengths.append(sum(np.linalg.norm(robot[b] - robot[a]) for a, b in zip(r_names[:-1], r_names[1:])))
        human_lengths.append(np.median(sum(
            np.linalg.norm(stored_joints[:, b] - stored_joints[:, a], axis=1)
            for a, b in zip(h_ids[:-1], h_ids[1:])
        )))
    return float(np.mean(robot_lengths) / np.mean(human_lengths))


def prepare_keypoints(stored_joints: np.ndarray, scale: float) -> tuple[np.ndarray, float, np.ndarray]:
    indices = [LOWER_BODY[name] for name in LOWER_BODY]
    points = stored_joints[:, indices].astype(np.float64)
    points = points @ STORED_TO_HUPHY.T
    origin_xy = points[0, 0, :2].copy()
    floor_z = float(np.percentile(points[:, [7, 8], 2], 2.0))
    points[:, :, :2] -= origin_xy
    points[:, :, 2] -= floor_z
    points *= scale
    return points, floor_z, origin_xy


def reconstruct_smpl_frames(
    source: dict[str, np.ndarray], body_models: Path, scale: float,
    floor_z: float, origin_xy: np.ndarray,
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    frames_count = len(source["body_pose"])
    model = smplx.create(body_models, "smplx", gender="neutral", use_pca=False, num_betas=10)
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(source["betas"]).float().view(1, -1).repeat(frames_count, 1),
            global_orient=torch.as_tensor(source["global_orient"]).float(),
            body_pose=torch.as_tensor(source["body_pose"]).float(),
            transl=torch.as_tensor(source["transl"]).float(),
            left_hand_pose=torch.as_tensor(source["left_hand_pose"]).float(),
            right_hand_pose=torch.as_tensor(source["right_hand_pose"]).float(),
            jaw_pose=torch.as_tensor(source["jaw_pose"]).float(),
            leye_pose=torch.as_tensor(source["leye_pose"]).float(),
            reye_pose=torch.as_tensor(source["reye_pose"]).float(),
            expression=torch.zeros(frames_count, model.num_expression_coeffs),
            return_full_pose=True,
        )
    full_pose = output.full_pose.cpu().numpy().reshape(frames_count, -1, 3)
    parents = np.asarray(model.parents, dtype=np.int64)
    stored = source["joints"]
    result = []
    ordered_names = list(LOWER_BODY)
    for frame_index in range(frames_count):
        global_rotations: list[Rotation] = []
        for joint_index in range(len(parents)):
            local = Rotation.from_rotvec(full_pose[frame_index, joint_index])
            if joint_index == 0:
                global_rotations.append(local)
            else:
                global_rotations.append(global_rotations[int(parents[joint_index])] * local)
        frame = {}
        for name in ordered_names:
            joint_index = LOWER_BODY[name]
            pos = stored[frame_index, joint_index] @ STORED_TO_HUPHY.T
            pos[:2] -= origin_xy
            pos[2] -= floor_z
            pos *= scale
            rot = NATIVE_TO_HUPHY @ global_rotations[joint_index].as_matrix()
            frame[SMPL_TO_GMR[name]] = (
                pos.astype(np.float64),
                Rotation.from_matrix(rot).as_quat(scalar_first=True),
            )
        result.append(frame)
    return result


def equality_residual(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    mask = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    return float(np.max(np.abs(data.efc_pos[mask]))) if np.any(mask) else 0.0


def save_motion(path: Path, qpos: np.ndarray, model: mujoco.MjModel, method: str,
                fps: float, keypoints: np.ndarray, scale: float) -> None:
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    motor_qpos_indices = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in MOTOR_JOINTS
    ]
    np.savez_compressed(
        path,
        qpos=qpos,
        root_pos=qpos[:, :3],
        root_quat_wxyz=qpos[:, 3:7],
        joint_pos=qpos[:, 7:],
        motor_joint_pos=qpos[:, motor_qpos_indices],
        joint_names=np.asarray(joint_names),
        motor_joint_names=np.asarray(MOTOR_JOINTS),
        fps=np.asarray(fps),
        source=np.asarray(str(DEFAULT_SOURCE)),
        method=np.asarray(method),
        lower_body_keypoint_names=np.asarray(list(LOWER_BODY)),
        lower_body_targets=keypoints,
        source_to_robot_scale=np.asarray(scale),
    )


def run_interactionmesh(
    xml_path: Path, keypoints: np.ndarray, output: Path, fps: float,
    initial_iters: int, iters: int, foot_sticking: bool = False,
) -> np.ndarray:
    # Compute contacts before constructing the optimizer. Some optimizer setup
    # paths normalize their input arrays in place; stance detection must always
    # use the untouched world-space source trajectory.
    contact_sequences = (
        detect_foot_sticking(keypoints.copy(), fps)
        if foot_sticking
        else [{"L_foot_link": False, "R_foot_link": False}] * len(keypoints)
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    q0 = initial_qpos(model)
    q0[:3] = keypoints[0, 0]
    constants = SimpleNamespace(
        ROBOT_URDF_FILE=str(xml_path), ROBOT_DOF=model.nq - 7,
        OBJECT_NAME="ground",
        FOOT_STICKING_LINKS=(
            ["L_foot_link", "R_foot_link"] if foot_sticking else []
        ),
        DEMO_JOINTS=list(LOWER_BODY), JOINTS_MAPPING=LOWER_BODY_LINKS,
        MANUAL_LB={}, MANUAL_UB={}, MANUAL_COST={},
        NOMINAL_TRACKING_INDICES=np.array([], dtype=int),
    )
    retargeter = InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path="",
        q_a_init_idx=-7,
        activate_foot_sticking=foot_sticking,
        activate_obj_non_penetration=False,
        activate_joint_limits=True,
        activate_joint_equalities=True,
        step_size=0.22,
        visualize=False,
        debug=False,
        w_nominal_tracking_init=0.0,
        initial_iterations=initial_iters,
        iterations_per_frame=iters,
        checkpoint_interval=100,
        mujoco_model=model,
    )
    # Keep temporal motion while allowing fast lower-body tracking.
    retargeter.smooth_weight = 0.12
    mins = keypoints.reshape(-1, 3).min(axis=0) - np.array([0.5, 0.5, 0.0])
    maxs = keypoints.reshape(-1, 3).max(axis=0) + np.array([0.5, 0.5, 0.0])
    gx = np.linspace(mins[0], maxs[0], 7)
    gy = np.linspace(mins[1], maxs[1], 7)
    xx, yy = np.meshgrid(gx, gy)
    ground = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    count = len(keypoints)
    object_pose = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (count, 1))
    qpos, _, _, _ = retargeter.retarget_motion(
        human_joint_motions=keypoints,
        object_poses=object_pose,
        object_poses_augmented=object_pose,
        object_points_local_demo=ground,
        object_points_local=ground,
        foot_sticking_sequences=contact_sequences,
        q_a_init=q0,
        original=True,
        dest_res_path=str(output),
    )
    return qpos


def detect_foot_sticking(
    keypoints: np.ndarray, fps: float,
    height_threshold: float = 0.08,
    speed_threshold: float = 0.35,
) -> list[dict[str, bool]]:
    """Detect stable stance feet from toe height and horizontal speed."""
    masks = []
    for point_index in (7, 8):
        toe = keypoints[:, point_index]
        velocity_xy = np.gradient(toe[:, :2], 1.0 / fps, axis=0)
        speed_xy = np.linalg.norm(velocity_xy, axis=1)
        mask = (toe[:, 2] <= height_threshold) & (speed_xy <= speed_threshold)
        # Close one-frame holes, then discard isolated detections shorter than
        # three frames so a swing-foot apex is never accidentally pinned.
        for i in range(1, len(mask) - 1):
            if not mask[i] and mask[i - 1] and mask[i + 1]:
                mask[i] = True
        start = 0
        while start < len(mask):
            if not mask[start]:
                start += 1
                continue
            end = start + 1
            while end < len(mask) and mask[end]:
                end += 1
            if end - start < 3:
                mask[start:end] = False
            start = end
        masks.append(mask)
    print(
        "Foot sticking stance fractions: "
        f"L={masks[0].mean():.3f}, R={masks[1].mean():.3f}",
        flush=True,
    )
    return [
        {"L_foot_link": bool(masks[0][i]), "R_foot_link": bool(masks[1][i])}
        for i in range(len(keypoints))
    ]


def make_gmr_config(
    model: mujoco.MjModel, q0: np.ndarray,
    first_frame: dict[str, tuple[np.ndarray, np.ndarray]], path: Path,
) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    position_weights = {
        "Pelvis": 100.0, "L_Hip": 30.0, "R_Hip": 30.0,
        "L_Knee": 60.0, "R_Knee": 60.0,
        "L_Ankle": 80.0, "R_Ankle": 80.0,
        "L_Foot": 100.0, "R_Foot": 100.0,
    }
    rotation_weights = {
        "Pelvis": 10.0, "L_Knee": 2.0, "R_Knee": 2.0,
        "L_Foot": 8.0, "R_Foot": 8.0,
    }
    table = {}
    for source_name, robot_body in LOWER_BODY_LINKS.items():
        human_name = SMPL_TO_GMR[source_name]
        human_rot = Rotation.from_quat(first_frame[human_name][1], scalar_first=True)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
        robot_rot = Rotation.from_matrix(data.xmat[body_id].reshape(3, 3))
        offset = (human_rot.inv() * robot_rot).as_quat(scalar_first=True)
        table[robot_body] = [
            human_name, position_weights[source_name], rotation_weights.get(source_name, 0.0),
            [0.0, 0.0, 0.0], offset.tolist(),
        ]
    config = {
        "robot_root_name": "base_link", "human_root_name": "pelvis",
        "ground_height": 0.0, "human_height_assumption": 1.0,
        "use_ik_match_table1": True, "use_ik_match_table2": False,
        "human_scale_table": {name: 1.0 for name in SMPL_TO_GMR.values()},
        "ik_match_table1": table, "ik_match_table2": {},
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def run_gmr(
    xml_path: Path, frames: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    config_path: Path,
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    q0 = initial_qpos(model)
    q0[:3] = frames[0]["pelvis"][0]
    make_gmr_config(model, q0, frames[0], config_path)
    gmr_params.ROBOT_XML_DICT["huphy_legonly"] = xml_path
    gmr_params.IK_CONFIG_DICT["smplx"]["huphy_legonly"] = config_path
    solver = GeneralMotionRetargeting(
        src_human="smplx", tgt_robot="huphy_legonly",
        actual_human_height=None, solver="daqp", damping=0.2,
        verbose=False, use_velocity_limit=False,
    )
    solver.configuration.update(q0)
    equality = mink.EqualityConstraintTask(solver.model, cost=1.0)
    output = []
    for frame_index, frame in enumerate(frames):
        solver.update_targets({key: [value[0].copy(), value[1].copy()] for key, value in frame.items()})
        last_error = np.inf
        for _ in range(18 if frame_index == 0 else 7):
            error = solver.error1()
            velocity = mink.solve_ik(
                solver.configuration, solver.tasks1,
                solver.configuration.model.opt.timestep,
                solver.solver, solver.damping, solver.ik_limits,
                constraints=[equality],
            )
            solver.configuration.integrate_inplace(velocity, solver.configuration.model.opt.timestep)
            if abs(last_error - error) < 1e-6:
                break
            last_error = error
        output.append(solver.configuration.data.qpos.copy())
        if frame_index % 100 == 0 or frame_index + 1 == len(frames):
            print(f"GMR frame {frame_index + 1}/{len(frames)}", flush=True)
    return np.asarray(output)


def mapped_mpjpe(model: mujoco.MjModel, qpos: np.ndarray, targets: np.ndarray) -> float:
    data = mujoco.MjData(model)
    errors = []
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) for body in LOWER_BODY_LINKS.values()]
    for frame, target in zip(qpos, targets):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        predicted = data.xpos[body_ids]
        errors.append(np.linalg.norm(predicted - target, axis=1))
    return float(np.mean(errors))


def validate(model: mujoco.MjModel, qpos: np.ndarray, targets: np.ndarray) -> dict[str, float | int | bool]:
    residuals = [equality_residual(model, q) for q in qpos]
    limited_violations = 0
    for joint_id in range(1, model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        adr = int(model.jnt_qposadr[joint_id])
        lo, hi = model.jnt_range[joint_id]
        limited_violations += int(np.count_nonzero((qpos[:, adr] < lo - 1e-5) | (qpos[:, adr] > hi + 1e-5)))
    return {
        "frames": len(qpos), "finite": bool(np.isfinite(qpos).all()),
        "max_loop_residual_m": float(max(residuals)),
        "mean_loop_residual_m": float(np.mean(residuals)),
        "joint_limit_violations": limited_violations,
        "lower_body_mpjpe_m": mapped_mpjpe(model, qpos, targets),
    }


def render_comparison(
    xml_path: Path, interaction: np.ndarray, gmr: np.ndarray,
    output: Path, fps: float,
    left_label: str = "InteractionMesh - lower 9 points",
    right_label: str = "GMR - lower 9 targets",
) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    # The source HUPHY XML uses MuJoCo's default 480 px offscreen
    # framebuffer height. Keep the per-panel renderer within that limit.
    renderer = mujoco.Renderer(model, height=480, width=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    camera.distance = 2.35
    camera.azimuth = 90.0
    camera.elevation = -8.0
    writer = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "960x480", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame_index, (q_im, q_gmr) in enumerate(zip(interaction, gmr)):
            panels = []
            for q, label in ((q_im, left_label), (q_gmr, right_label)):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                panel = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 28), fill=(20, 20, 20))
                draw.text((10, 7), label, fill=(255, 255, 255))
                panels.append(np.asarray(panel))
            combined = np.concatenate(panels, axis=1)
            assert writer.stdin is not None
            writer.stdin.write(combined.tobytes())
            if frame_index % 150 == 0:
                print(f"Render frame {frame_index + 1}/{len(interaction)}", flush=True)
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
        renderer.close()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def main() -> None:
    args = parse_args()
    motion_id = args.source.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    derived_xml = args.output_dir / "huphy_capsule7plus5_loop_retarget.xml"
    render_xml = args.output_dir / "huphy_capsule7plus5_loop_retarget_floor.xml"
    build_derived_xml(args.huphy_xml, derived_xml)
    build_derived_xml(args.huphy_xml, render_xml, add_floor=True)
    model = mujoco.MjModel.from_xml_path(str(derived_xml))
    source_npz = np.load(args.source, allow_pickle=True)
    source = {key: source_npz[key] for key in source_npz.files}
    fps = float(json.loads(str(source["metadata"])).get("fps", 30.0))
    source_frame_count = len(source["body_pose"])
    start_frame = max(0, int(round(args.start_time * fps)))
    end_frame = (
        source_frame_count
        if args.end_time is None
        else min(source_frame_count, int(round(args.end_time * fps)))
    )
    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid time range {args.start_time:g}..{args.end_time}; "
            f"source duration is {source_frame_count / fps:.3f}s"
        )
    for key, value in list(source.items()):
        if isinstance(value, np.ndarray) and value.ndim and len(value) == source_frame_count:
            source[key] = value[start_frame:end_frame]
    if args.max_frames is not None:
        sliced_frame_count = len(source["body_pose"])
        for key, value in list(source.items()):
            if isinstance(value, np.ndarray) and value.ndim and len(value) == sliced_frame_count:
                source[key] = value[: args.max_frames]
    q0 = initial_qpos(model)
    scale = estimate_scale(source["joints"], model, q0)
    keypoints, floor_z, origin_xy = prepare_keypoints(source["joints"], scale)
    print(f"frames={len(keypoints)} fps={fps:g} lower_body_points=9 scale={scale:.6f}")
    print(f"initial_loop_residual={equality_residual(model, q0):.3e} m")

    results: dict[str, np.ndarray] = {}
    if args.method in ("both", "interactionmesh"):
        im_path = args.output_dir / f"{motion_id}_huphy_interactionmesh_lower9.npz"
        results["interactionmesh"] = run_interactionmesh(
            derived_xml, keypoints, im_path, fps,
            args.interaction_initial_iters, args.interaction_iters,
            foot_sticking=args.foot_sticking,
        )
        method_name = (
            "interactionmesh_lower9_foot_sticking"
            if args.foot_sticking else "interactionmesh_lower9"
        )
        save_motion(im_path, results["interactionmesh"], model, method_name, fps, keypoints, scale)
    if args.method in ("both", "gmr"):
        frames = reconstruct_smpl_frames(source, args.body_models, scale, floor_z, origin_xy)
        config_path = args.output_dir / "smplx_to_huphy_lower9_gmr.json"
        results["gmr"] = run_gmr(derived_xml, frames, config_path)
        gmr_path = args.output_dir / f"{motion_id}_huphy_gmr_lower9.npz"
        save_motion(gmr_path, results["gmr"], model, "gmr_lower9_closed_loop", fps, keypoints, scale)

    metrics = {name: validate(model, motion, keypoints) for name, motion in results.items()}
    metrics["source"] = {
        "path": str(args.source), "fps": fps, "frames": len(keypoints),
        "source_to_robot_scale": scale, "keypoints": list(LOWER_BODY),
        "robot_nq": model.nq, "robot_nv": model.nv, "scalar_joints": model.nq - 7,
        "motor_joints": len(MOTOR_JOINTS), "connect_equalities": model.neq,
    }
    (args.output_dir / "validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not args.skip_render and set(results) == {"interactionmesh", "gmr"}:
        render_comparison(
            render_xml, results["interactionmesh"], results["gmr"],
            args.output_dir / f"{motion_id}_huphy_interactionmesh_vs_gmr.mp4", fps,
        )


if __name__ == "__main__":
    main()
