#!/usr/bin/env python3
"""One axis-aligned grid video for all physical Pygmalion motor joints.

Evidence-backed cells view the rotor flange face-on from outside: the outward normal points
OUT of the screen and the camera/view ray is -normal.  Cells without physical flange evidence
are explicitly marked UNKNOWN and must not be used to infer CW/CCW.  The 17 joints run
0 -> +upper -> -lower -> 0 in parallel.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pygmalion-matplotlib")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rom_sweep_video import PHYSICAL_MOTOR_JOINTS, ease, joint_kind, load_signs, circ_arrow
from rom_sweep_video_cpu import SoftwareProjector

ROOT = Path(__file__).resolve().parents[2]
XMLS = ROOT / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
OUTROOT = ROOT / "docs/video"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

COLS, ROWS = 4, 5
CELL_W, CELL_H, HEAD_H = 400, 264, 80
VIEW_H = 178
W, H = COLS * CELL_W, HEAD_H + ROWS * CELL_H
FPS, SAMPLE_HZ = 20, 5
TRANSITION_S, HOLD_S = 4.0, 0.25

# User review of the previous fixed-camera videos (2026-09-02).  It is evidence, but
# disagreements with the CAD normal remain a physical flange-photo/low-current gate.
REVIEWER_REVERSED = {
    "L_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_crank_A_joint",
    "R_knee_joint", "R_hip_roll_joint", "L_crank_B_joint", "L_hip_pitch_joint",
}

# HUPHY v1.0 IDs with the user-confirmed physical convention:
# upper/closer-to-knee motor = A, lower/closer-to-ankle motor = B.
MOTOR_ID = {
    "L_hip_pitch_joint": 1, "L_hip_roll_joint": 2, "L_hip_yaw_joint": 3,
    "L_knee_joint": 4, "L_crank_A_joint": 5, "L_crank_B_joint": 6,
    "R_hip_pitch_joint": 7, "R_hip_roll_joint": 8, "R_hip_yaw_joint": 9,
    "R_knee_joint": 10, "R_crank_A_joint": 11, "R_crank_B_joint": 12,
}


def aligned_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face-on basis: outward `normal` points toward viewer; view ray is `-normal`."""
    forward = -normal / np.linalg.norm(normal)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up_hint)) > 0.9:
        up_hint = np.array([1.0, 0.0, 0.0])
    up = up_hint - forward * float(forward @ up_hint)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return right, up, forward


def motor_geom_name(joint: str) -> str:
    base = joint.removesuffix("_joint")
    # The generated XML still carries superseded internal proxy labels.  Its upper physical A
    # proxy is named ankle_b_motor and lower physical B proxy ankle_a_motor.  This lookup is
    # visual-only; runtime IDs follow the corrected physical A/B contract above.
    if "crank_A" in base:
        return base.replace("crank_A", "ankle_b") + "_motor"
    if "crank_B" in base:
        return base.replace("crank_B", "ankle_a") + "_motor"
    return base + "_motor"


def zoom_for(name: str) -> float:
    kind = joint_kind(name)
    if "crank" in kind:
        return 0.19
    if "shoulder" in kind:
        return 0.25
    if "waist" in kind:
        return 0.31
    if "knee" in kind:
        return 0.29
    return 0.25


def phase_frames(lo: float, hi: float):
    n = int(TRANSITION_S * SAMPLE_HZ)
    phases = (("0 -> +UPPER", 0.0, hi),
              ("+UPPER -> -LOWER", hi, lo),
              ("-LOWER -> 0", lo, 0.0))
    for index, (phase, a, b) in enumerate(phases):
        for k in range(n):
            t = k / max(n - 1, 1)
            yield phase, float(a + (b - a) * ease(t)), FPS // SAMPLE_HZ
        if index < 2:
            yield ("HOLD +UPPER" if index == 0 else "HOLD -LOWER"), float(b), int(HOLD_S * FPS)


class Sink:
    def __init__(self, path: Path):
        self.frames = 0
        self.proc = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "20", str(path),
        ], stdin=subprocess.PIPE)

    def add(self, image: Image.Image, repeat: int):
        raw = image.convert("RGB").tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(raw)
            self.frames += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError("ffmpeg failed")


def main() -> None:
    args = sys.argv[1:]
    tag = next((x.split("=", 1)[1] for x in args if x.startswith("--tag=")),
               "FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix")
    xml = XMLS / f"{tag}_loop.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    signs = load_signs() or {}
    crosscheck_path = ROOT / "tools/robot_model/huphy_mjcf_crosscheck.json"
    crosscheck = json.loads(crosscheck_path.read_text()) if crosscheck_path.exists() else {"joints": {}}
    names = {model.joint(i).name for i in range(model.njnt)}
    joints = [j for j in PHYSICAL_MOTOR_JOINTS if j in names]
    if len(joints) != 17:
        raise RuntimeError(f"expected 17 motor joints, got {len(joints)}")

    # A rotor face proves the face-on camera, but not the motor-to-URDF sign.  That also
    # requires the rotor/stator-to-child/parent attachment parity.  Never coerce None to
    # False (the old bool(None) path falsely labelled UNKNOWN as CCW), and never promote an
    # independent MJCF gear sign into a physical motor sign.
    view_info = {}
    for j in joints:
        jid = model.joint(j).id
        axis = np.asarray(model.jnt_axis[jid], float)
        axis /= np.linalg.norm(axis)
        if j in signs and isinstance(signs[j].get("matches"), bool):
            view_info[j] = {"normal": np.asarray(signs[j]["n_base"], float),
                            "matches": signs[j]["matches"],
                            "evidence": "FUSION FLANGE + ATTACHMENT PARITY"}
        elif j in signs:
            view_info[j] = {"normal": np.asarray(signs[j]["n_base"], float),
                            "matches": None,
                            "evidence": "FUSION FLANGE; ATTACHMENT UNKNOWN"}
        elif j in crosscheck.get("joints", {}):
            view_info[j] = {"normal": -axis, "matches": None,
                            "evidence": "INDEPENDENT MJCF; NOT PHYSICAL SIGN EVIDENCE"}
        else:
            view_info[j] = {"normal": -axis, "matches": None,
                            "evidence": "UNKNOWN FLANGE; q+ VIEW ONLY"}

    normals = {tuple(v["normal"]) for v in view_info.values()}
    projectors = {}
    for normal in normals:
        p = SoftwareProjector(model, CELL_W, VIEW_H)
        p.right, p.up, p.forward = aligned_basis(np.asarray(normal, float))
        projectors[normal] = p

    data = {j: mujoco.MjData(model) for j in joints}
    limits = {j: tuple(np.degrees(model.jnt_range[model.joint(j).id])) for j in joints}
    sequences = {j: list(phase_frames(*limits[j])) for j in joints}
    count = len(sequences[joints[0]])
    if any(len(v) != count for v in sequences.values()):
        raise RuntimeError("sequence length mismatch")

    outdir = OUTROOT / f"rom_axis_grid_{tag}_loop_reaudit_faceon_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    video = outdir / "all_17_motor_axes.mp4"
    sink = Sink(video)
    poster = None
    font_b = ImageFont.truetype(FONT_B, 16)
    font_r = ImageFont.truetype(FONT_R, 13)

    for frame_index in range(count):
        canvas = Image.new("RGB", (W, H), (18, 22, 29))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 12), "Pygmalion motor-axis re-audit | face-on outside | outward normal = OUT OF SCREEN",
                  font=ImageFont.truetype(FONT_B, 23), fill=(242, 245, 250))
        draw.text((18, 46), "CW motor+ axis = view ray = -normal | A=upper, B=lower/ankle-side | old grid INVALID",
                  font=ImageFont.truetype(FONT_R, 15), fill=(156, 170, 188))
        repeats = None
        for i, name in enumerate(joints):
            phase, angle, repeat = sequences[name][frame_index]
            repeats = repeat if repeats is None else repeats
            if repeat != repeats:
                raise RuntimeError("repeat mismatch")
            jid = model.joint(name).id
            d = data[name]
            mujoco.mj_resetData(model, d)
            d.qpos[:7] = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
            d.qpos[model.jnt_qposadr[jid]] = np.radians(angle)
            mujoco.mj_forward(model, d)
            info = view_info[name]
            normal = tuple(info["normal"])
            child = int(model.jnt_bodyid[jid])
            geom_name = motor_geom_name(name)
            try:
                motor_geoms = {model.geom(geom_name).id}
            except KeyError:
                motor_geoms = set()
            rgb = projectors[normal].render(
                d, child, d.xanchor[jid].copy(), zoom_half=zoom_for(name),
                highlighted_geoms=motor_geoms)
            cell = Image.new("RGB", (CELL_W, CELL_H), (29, 34, 44))
            cell.paste(Image.fromarray(rgb), (0, 0))
            cd = ImageDraw.Draw(cell)
            cd.rectangle((0, VIEW_H, CELL_W, CELL_H), fill=(26, 31, 40))
            lo, hi = limits[name]
            matches = info["matches"]
            sense = "UNKNOWN" if matches is None else ("CW" if matches else "CCW")
            mid = f"m{MOTOR_ID[name]}" if name in MOTOR_ID else "upper / not policy"
            cd.text((8, VIEW_H + 5), f"{i+1:02d} {name}  [{mid}]", font=font_b, fill=(245, 247, 250))
            cd.text((8, VIEW_H + 27), f"ROM {lo:+.0f}..{hi:+.0f} deg | now {angle:+.1f} | {phase}",
                    font=font_r, fill=(170, 181, 196))
            cd.text((8, VIEW_H + 47), f"model q+ on this view: {sense}",
                    font=font_b, fill=(244, 196, 92) if matches is None else (82, 205, 134))
            cd.text((8, VIEW_H + 68), info["evidence"], font=font_r,
                    fill=(244, 196, 92) if matches is None else (111, 166, 232))
            if matches is not None:
                circ_arrow(cd, CELL_W - 34, VIEW_H + 54, 19, bool(matches),
                           (82, 205, 134), width=4)
            x, y = (i % COLS) * CELL_W, HEAD_H + (i // COLS) * CELL_H
            canvas.paste(cell, (x, y))
        if poster is None and frame_index == max(1, int(TRANSITION_S * SAMPLE_HZ * 0.45)):
            poster = canvas.copy()
        sink.add(canvas, repeats or 1)
        print(f"frame {frame_index+1}/{count}", flush=True)

    sink.close()
    if poster is None:
        poster = canvas.copy()
    poster.save(outdir / "all_17_motor_axes_grid.png")
    manifest = {
        "model": str(xml.resolve()), "fps": FPS, "duration_s": sink.frames / FPS,
        "sequence": "0 -> +upper -> -lower -> 0; 4 s each, 0.25 s holds",
        "view": "face-on outside: outward normal points out of screen; camera ray and CW motor-positive axis are -normal",
        "supersedes": "2026-09-01 grid that put outward normal into the screen",
        "right_normal": "never mirrored/copied; missing direct evidence is UNKNOWN",
        "sign_gate": "CW/CCW requires rotor face normal AND explicit attachment parity",
        "huphy_ab_mapping": {"crank_A_upper": "ankle_a", "crank_B_lower": "ankle_b"},
        "reviewer_reversed": sorted(REVIEWER_REVERSED),
        "joints": {j: {"lower_deg": limits[j][0], "upper_deg": limits[j][1],
                        "model_qplus": ("UNKNOWN" if view_info[j]["matches"] is None else
                                        "CW" if view_info[j]["matches"] else "CCW"),
                        "normal": [float(x) for x in view_info[j]["normal"]],
                        "evidence": view_info[j]["evidence"],
                        "huphy_motor_id": MOTOR_ID.get(j),
                        "prior_reviewed_as_reversed": j in REVIEWER_REVERSED}
                   for j in joints},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(video)


if __name__ == "__main__":
    main()
