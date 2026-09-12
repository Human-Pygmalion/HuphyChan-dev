"""Interactive browser review of the current Fusion-derived Pygmalion model.

The viewer loads the canonical FullDoF AB MJCF sidecar generated alongside the
URDF.  MuJoCo is used only for exact forward kinematics; viser serves the actual
meshes and controls in a browser.  The twelve policy motor joints are exposed.

This is deliberately a q-sign inspection mode.  Moving one crank joint directly
shows its child-on-joint transform but does not solve the two-RSU equality loop;
use the mechanism/invariant tests for closure validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser


REPO = Path(__file__).resolve().parents[2]
DEFAULT_XML = (
    REPO
    / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
    / "FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml"
)
DEFAULT_REVIEW = (
    REPO
    / "tools/robot_model/fusion_snapshots/v30_inspection"
    / "joint_direction_user_review.json"
)
MOTOR_JOINTS = tuple(
    f"{side}_{suffix}_joint"
    for side in ("L", "R")
    for suffix in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "crank_A",
        "crank_B",
    )
)
UPPER_FIXED = (
    "waist_yaw_joint",
    "L_shoulder_pitch_joint",
    "R_shoulder_pitch_joint",
)
SHOULDER_ROLL = ("L_shoulder_roll_joint", "R_shoulder_roll_joint")
STAND_Z = 0.903


def _primitive(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    geom_type = model.geom_type[geom_id]
    radius = float(model.geom_size[geom_id][0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = float(model.geom_size[geom_id][1])
        return trimesh.creation.capsule(height=2.0 * half, radius=radius, count=[12, 12])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(
            radius=radius, height=2.0 * float(model.geom_size[geom_id][1]), sections=24
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * np.asarray(model.geom_size[geom_id][:3]))
    return None


def _quat(matrix: np.ndarray) -> tuple[float, float, float, float]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, matrix)
    return tuple(float(x) for x in quat)


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or f"{obj.name}_{idx}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8893)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = args.xml.resolve()
    xml_sha = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    model_name = xml_path.stem
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    missing = [
        joint
        for joint in MOTOR_JOINTS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) < 0
    ]
    if missing:
        raise RuntimeError(f"12-joint contract failed; missing: {missing}")
    if len(MOTOR_JOINTS) != 12:
        raise RuntimeError("viewer motor-joint contract is not 12")

    if args.selftest:
        print(
            json.dumps(
                {
                    "model": model_name,
                    "xml": str(xml_path),
                    "sha256": xml_sha,
                    "policy_joint_count": len(MOTOR_JOINTS),
                    "policy_joints": MOTOR_JOINTS,
                    "total_mass_kg": float(model.body_subtreemass[1]),
                    "equality_constraints": int(model.neq),
                },
                indent=2,
            )
        )
        return

    server = viser.ViserServer(host=args.host, port=args.port, label="Pygmalion joint review")
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground",
        width=4.0,
        height=4.0,
        cell_size=0.1,
        section_size=0.5,
        plane="xy",
        position=(0.0, 0.0, 0.0),
    )

    palette = (
        (157, 177, 218),
        (235, 159, 125),
        (153, 201, 160),
        (210, 149, 194),
        (166, 199, 215),
        (224, 191, 124),
    )
    handles: list[tuple[int, object, str]] = []
    groups: dict[str, list[object]] = {
        "visual": [],
        "motor": [],
        "collision": [],
        "hull": [],
    }
    for geom_id in range(model.ngeom):
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        group_id = int(model.geom_group[geom_id])
        if group_id == 4 or geom_name.endswith("_hull"):
            kind, color, opacity, visible = "hull", (230, 150, 45), 0.25, False
        elif group_id == 3 or model.geom_contype[geom_id]:
            kind, color, opacity, visible = "collision", (55, 185, 100), 0.28, False
        elif "motor" in geom_name:
            kind, color, opacity, visible = "motor", (55, 58, 66), 1.0, True
        else:
            kind, color, opacity, visible = "visual", palette[body_id % len(palette)], 1.0, True

        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            vert_adr, vert_num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            face_adr, face_num = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
            faces = model.mesh_face[face_adr : face_adr + face_num]
        else:
            primitive = _primitive(model, geom_id)
            if primitive is None or body_id == 0:
                continue
            vertices, faces = primitive.vertices, primitive.faces

        handle = server.scene.add_mesh_simple(
            f"/robot/{kind}/{geom_id:03d}_{geom_name}",
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32),
            color=color,
            opacity=opacity,
            side="double",
            visible=visible,
        )
        handles.append((geom_id, handle, kind))
        groups[kind].append(handle)

    joint_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in MOTOR_JOINTS
    }
    qpos_ids = {name: int(model.jnt_qposadr[jid]) for name, jid in joint_ids.items()}
    review = {
        name: {"status": "미검토", "note": ""}
        for name in MOTOR_JOINTS
    }
    if args.review.exists():
        try:
            old = json.loads(args.review.read_text())
            if old.get("model_sha256") == xml_sha:
                review.update(old.get("joints", {}))
        except (json.JSONDecodeError, OSError):
            pass

    selected_name = MOTOR_JOINTS[0]
    guard = False
    sliders: dict[str, viser.GuiSliderHandle] = {}

    def initialize_pose() -> None:
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = (0.0, 0.0, STAND_Z)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name in UPPER_FIXED:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = 0.0
        for name in SHOULDER_ROLL:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(-15.0)

    def refresh() -> None:
        mujoco.mj_forward(model, data)
        for geom_id, handle, _kind in handles:
            handle.position = tuple(float(x) for x in data.geom_xpos[geom_id])
            handle.wxyz = _quat(data.geom_xmat[geom_id])
        jid = joint_ids[selected_name]
        anchor = np.asarray(data.xanchor[jid], dtype=float)
        axis = np.asarray(data.xaxis[jid], dtype=float)
        axis /= np.linalg.norm(axis)
        points = np.asarray(
            [[anchor - 0.07 * axis, anchor + 0.13 * axis]], dtype=np.float32
        )
        server.scene.add_arrows(
            "/review/selected_q_plus_axis",
            points,
            np.asarray([[40, 210, 255]], dtype=np.uint8),
            shaft_radius=0.008,
            head_radius=0.018,
            head_length=0.035,
        )
        server.scene.add_icosphere(
            "/review/selected_joint_anchor",
            radius=0.018,
            color=(255, 190, 40),
            position=tuple(anchor),
        )
        child = int(model.jnt_bodyid[jid])
        parent = int(model.body_parentid[child])
        q_deg = np.degrees(data.qpos[qpos_ids[selected_name]])
        lo_deg, hi_deg = np.degrees(model.jnt_range[jid])
        info.content = (
            f"**선택:** `{selected_name}`  \n"
            f"현재 **{q_deg:+.1f}°** · ROM **[{lo_deg:+.1f}°, {hi_deg:+.1f}°]**  \n"
            f"parent `{_name(model, mujoco.mjtObj.mjOBJ_BODY, parent)}` → "
            f"child `{_name(model, mujoco.mjtObj.mjOBJ_BODY, child)}`  \n"
            f"청록 화살표 = **MJCF/URDF q+ 축** · 황색 구 = joint anchor  \n"
            f"world axis `[{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]`"
        )

    initialize_pose()

    with server.gui.add_folder("12-DoF 방향 검토"):
        selector = server.gui.add_dropdown(
            "관절 선택", options=MOTOR_JOINTS, initial_value=selected_name
        )
        info = server.gui.add_markdown("")
        isolate = server.gui.add_checkbox("선택 관절만 움직이기", initial_value=True)
        with server.gui.add_folder("빠른 위치"):
            zero_button = server.gui.add_button("중립 0°")
            lower_button = server.gui.add_button("ROM 하한")
            upper_button = server.gui.add_button("ROM 상한")
            reset_button = server.gui.add_button("12관절 모두 중립")

    for side, side_label in (("L", "L 관절"), ("R", "R 관절")):
        with server.gui.add_folder(side_label):
            for name in [n for n in MOTOR_JOINTS if n.startswith(side + "_")]:
                jid = joint_ids[name]
                lo, hi = np.degrees(model.jnt_range[jid])
                slider = server.gui.add_slider(
                    name,
                    min=float(np.round(lo, 1)),
                    max=float(np.round(hi, 1)),
                    step=0.5,
                    initial_value=0.0,
                )
                sliders[name] = slider

                @slider.on_update
                def _slider_update(_event, name=name, slider=slider):
                    nonlocal selected_name, guard
                    if guard:
                        return
                    selected_name = name
                    selector.value = name
                    if isolate.value:
                        guard = True
                        for other, other_slider in sliders.items():
                            if other != name and other_slider.value != 0.0:
                                other_slider.value = 0.0
                                data.qpos[qpos_ids[other]] = 0.0
                        guard = False
                    data.qpos[qpos_ids[name]] = np.radians(slider.value)
                    refresh()

    with server.gui.add_folder("표시"):
        show_visual = server.gui.add_checkbox("Fusion visual mesh", initial_value=True)
        show_motor = server.gui.add_checkbox("motor proxy", initial_value=True)
        show_collision = server.gui.add_checkbox("collision", initial_value=False)
        show_hull = server.gui.add_checkbox("URDF hull", initial_value=False)

    with server.gui.add_folder("사용자 판정"):
        status = server.gui.add_dropdown(
            "선택 관절 판정",
            options=("미검토", "정상", "반대", "불명"),
            initial_value="미검토",
        )
        note = server.gui.add_text("메모", initial_value="")
        save = server.gui.add_button("판정 JSON 저장")
        save_message = server.gui.add_markdown("")

    server.gui.add_markdown(
        f"모델 `{model_name}`  \n"
        f"총질량 **{model.body_subtreemass[1]:.4f} kg** · equality **{model.neq}** · "
        f"SHA256 `{xml_sha[:12]}…`  \n"
        "팔은 학습 자세와 같이 양쪽 shoulder-roll q=-15°로 고정 표시합니다."
    )

    @selector.on_update
    def _select(_event):
        nonlocal selected_name, guard
        if guard:
            return
        selected_name = selector.value
        guard = True
        status.value = review[selected_name].get("status", "미검토")
        note.value = review[selected_name].get("note", "")
        guard = False
        refresh()

    def _set_selected(value_rad: float) -> None:
        nonlocal guard
        guard = True
        if isolate.value:
            for other, other_slider in sliders.items():
                if other != selected_name:
                    other_slider.value = 0.0
                    data.qpos[qpos_ids[other]] = 0.0
        value_deg = float(np.degrees(value_rad))
        sliders[selected_name].value = value_deg
        data.qpos[qpos_ids[selected_name]] = value_rad
        guard = False
        refresh()

    @zero_button.on_click
    def _zero(_event):
        _set_selected(0.0)

    @lower_button.on_click
    def _lower(_event):
        _set_selected(float(model.jnt_range[joint_ids[selected_name], 0]))

    @upper_button.on_click
    def _upper(_event):
        _set_selected(float(model.jnt_range[joint_ids[selected_name], 1]))

    @reset_button.on_click
    def _reset(_event):
        nonlocal guard
        guard = True
        initialize_pose()
        for slider in sliders.values():
            slider.value = 0.0
        guard = False
        refresh()

    def _toggle(kind: str, visible: bool) -> None:
        for handle in groups[kind]:
            handle.visible = visible

    show_visual.on_update(lambda _event: _toggle("visual", show_visual.value))
    show_motor.on_update(lambda _event: _toggle("motor", show_motor.value))
    show_collision.on_update(lambda _event: _toggle("collision", show_collision.value))
    show_hull.on_update(lambda _event: _toggle("hull", show_hull.value))

    @status.on_update
    def _status(_event):
        if not guard:
            review[selected_name]["status"] = status.value

    @note.on_update
    def _note(_event):
        if not guard:
            review[selected_name]["note"] = note.value

    @save.on_click
    def _save(_event):
        args.review.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "pygmalion-joint-direction-user-review-v1",
            "model": model_name,
            "model_path": str(xml_path),
            "model_sha256": xml_sha,
            "saved_unix": time.time(),
            "definition": "cyan arrow is MJCF/URDF q+; physical motor+ is not inferred here",
            "joints": review,
        }
        args.review.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        done = sum(v.get("status") != "미검토" for v in review.values())
        save_message.content = f"저장 완료: **{done}/12** 관절 · `{args.review}`"

    @server.on_client_connect
    def _client(client: viser.ClientHandle) -> None:
        client.camera.position = (2.2, -2.2, 1.4)
        client.camera.look_at = (0.0, 0.0, 0.55)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    refresh()
    print(f"viewer http://{args.host}:{args.port}", flush=True)
    print(f"model  {xml_path}", flush=True)
    print(f"sha256 {xml_sha}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
