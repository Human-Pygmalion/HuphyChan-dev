#!/usr/bin/env python3
"""CPU-only orthographic ROM/sign renderer for Pygmalion physical motor joints.

This is the deterministic fallback when the Codex session has no working EGL/NVIDIA
context. It reads the same compiled MJCF used by training. Motion is sampled at 5 Hz and
each rendered frame is repeated four times in a 20 fps video, preserving the requested
4.00 s transitions and 0.25 s endpoint holds exactly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pygmalion-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import trimesh  # noqa: E402

from rom_sweep_video import (  # noqa: E402
    W, H, FPS, FONT, FONT_R, PHYSICAL_MOTOR_JOINTS, HUMAN, ZOOM_BODY,
    draw_frame, ease, joint_kind, load_signs,
)

ROOT = Path(__file__).resolve().parents[2]
XMLS = ROOT / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
OUTROOT = ROOT / "docs/video"
TRANSITION_S = 4.0
HOLD_S = 0.25
RENDER_FPS = 5
MAX_FACES_PER_GEOM = 450


def basis(azimuth: float, elevation: float) -> tuple[np.ndarray, ...]:
    az, el = np.radians((azimuth, elevation))
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


class SoftwareProjector:
    def __init__(self, model: mujoco.MjModel, width: int, height: int,
                 azimuth: float = 140.0, elevation: float = -8.0):
        self.model, self.width, self.height = model, width, height
        self.right, self.up, self.forward = basis(azimuth, elevation)
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.axis("off")
        self.fig.patch.set_facecolor((0.93, 0.94, 0.96))
        self.ax.set_facecolor((0.93, 0.94, 0.96))
        self.parts = []
        for geom in range(model.ngeom):
            if model.geom_group[geom] != 2:
                continue
            kind = model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                vertices = np.asarray(model.mesh_vert[va:va + vn])
                faces = np.asarray(model.mesh_face[fa:fa + fn])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                primitive = trimesh.creation.cylinder(
                    radius=float(model.geom_size[geom, 0]),
                    height=float(2 * model.geom_size[geom, 1]), sections=18)
                vertices, faces = np.asarray(primitive.vertices), np.asarray(primitive.faces)
            else:
                continue
            if len(faces) > MAX_FACES_PER_GEOM:
                take = np.linspace(0, len(faces) - 1, MAX_FACES_PER_GEOM, dtype=int)
                faces = faces[take]
            self.parts.append((geom, vertices, faces))

    def _triangles(self, data: mujoco.MjData, highlighted_body: int,
                   highlighted_geoms: set[int] | None = None):
        polys, cols, depths = [], [], []
        light = self.right * 0.35 + self.up * 0.55 - self.forward * 0.75
        light /= np.linalg.norm(light)
        for geom, vertices, faces in self.parts:
            rotation = data.geom_xmat[geom].reshape(3, 3)
            world = vertices @ rotation.T + data.geom_xpos[geom]
            tri = world[faces]
            projected = np.stack((tri @ self.right, tri @ self.up), axis=-1)
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            shade = np.clip(0.30 + 0.70 * np.abs(normal @ light), 0.0, 1.0)
            if highlighted_geoms and geom in highlighted_geoms:
                base = np.array([0.10, 0.43, 0.95])
            elif self.model.geom_bodyid[geom] == highlighted_body:
                base = np.array([0.98, 0.16, 0.06])
            else:
                base = np.array([0.64, 0.68, 0.74])
            polys.append(projected)
            cols.append(np.clip(base[None, :] * shade[:, None], 0.0, 1.0))
            depths.append((tri @ self.forward).mean(axis=1))
        poly = np.concatenate(polys)
        colour = np.concatenate(cols)
        depth = np.concatenate(depths)
        order = np.argsort(depth)[::-1]
        return poly[order], colour[order]

    def render(self, data: mujoco.MjData, highlighted_body: int, anchor: np.ndarray,
               zoom_half: float | None = None,
               highlighted_geoms: set[int] | None = None) -> np.ndarray:
        poly, colour = self._triangles(data, highlighted_body, highlighted_geoms)
        self.ax.clear()
        self.ax.axis("off")
        self.ax.add_collection(PolyCollection(poly, facecolors=colour, edgecolors="none",
                                              rasterized=True))
        anchor2 = np.array([anchor @ self.right, anchor @ self.up])
        if zoom_half is None:
            points = poly.reshape(-1, 2)
            low, high = points.min(0), points.max(0)
            centre = (low + high) / 2
            half_y = max((high[1] - low[1]) * 0.56, 0.8)
            half_x = max((high[0] - low[0]) * 0.56, half_y * self.width / self.height)
        else:
            centre = anchor2
            half_y = zoom_half
            half_x = zoom_half * self.width / self.height
        self.ax.set_xlim(centre[0] - half_x, centre[0] + half_x)
        self.ax.set_ylim(centre[1] - half_y, centre[1] + half_y)
        self.ax.set_aspect("equal")
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        image = Image.fromarray(rgba[:, :, :3].copy())
        # Explicit overview/close-up joint marker, independent of the MuJoCo joint-visual flag.
        px = int((anchor2[0] - (centre[0] - half_x)) / (2 * half_x) * self.width)
        py = int((1 - (anchor2[1] - (centre[1] - half_y)) / (2 * half_y)) * self.height)
        draw = ImageDraw.Draw(image)
        draw.ellipse((px - 13, py - 13, px + 13, py + 13), outline=(255, 30, 20), width=5)
        draw.line((px - 18, py, px + 18, py), fill=(255, 30, 20), width=3)
        draw.line((px, py - 18, px, py + 18), fill=(255, 30, 20), width=3)
        return np.asarray(image)


class VideoSink:
    def __init__(self, path: Path):
        self.path = path
        self.frames = 0
        self.proc = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "20", str(path),
        ], stdin=subprocess.PIPE)

    def add(self, image: Image.Image, repeat: int = 1):
        raw = image.convert("RGB").tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(raw)
            self.frames += 1

    def close(self):
        self.proc.stdin.close()
        code = self.proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with exit code {code}: {self.path}")


def main() -> None:
    args = sys.argv[1:]
    tag = next((a.split("=", 1)[1] for a in args if a.startswith("--tag=")),
               "prototype-tempmass-motormeasured-armfix_v30")
    mode = next((a.split("=", 1)[1] for a in args if a.startswith("--model=")), "loop")
    only = next((a.split("=", 1)[1].split(",") for a in args if a.startswith("--joints=")), None)
    xml = XMLS / f"{tag}{'_loop' if mode == 'loop' else ''}.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    names = {model.joint(j).name for j in range(model.njnt)}
    joints = [name for name in PHYSICAL_MOTOR_JOINTS if name in names]
    if only:
        joints = [name for name in joints if any(token in name for token in only)]
    outdir = OUTROOT / f"rom_sign_{tag}_{mode}_cpu"
    outdir.mkdir(parents=True, exist_ok=True)
    signs = load_signs() or {}
    main_renderer = SoftwareProjector(model, 1024, 768)
    zoom_renderer = SoftwareProjector(model, 430, 430)
    manifest = {"_meta": {
        "model": str(xml.resolve()), "renderer": "CPU software orthographic",
        "fps": FPS, "unique_motion_sample_hz": RENDER_FPS,
        "transition_s": TRANSITION_S, "endpoint_hold_s": HOLD_S,
        "sequence": "neutral(0) -> positive upper -> negative lower -> neutral(0)",
        "convention": "positive = CW looking along outward rotor-flange normal",
        "scope": "17 physical motors; passive ankle and rod universal joints excluded",
        "closed_loop_note": "kinematic rotor/crank direction render; loop constraints are not dynamically settled",
    }}
    posters = []
    repeat = FPS // RENDER_FPS
    assert FPS % RENDER_FPS == 0
    transition_samples = int(TRANSITION_S * RENDER_FPS)
    hold_frames = int(HOLD_S * FPS)
    font = ImageFont.truetype(FONT, 18)
    for order, name in enumerate(joints, 1):
        jid = model.joint(name).id
        lo, hi = np.degrees(model.jnt_range[jid])
        kind = joint_kind(name)
        side = "L" if name.startswith("L_") else ("R" if name.startswith("R_") else "")
        child = int(model.jnt_bodyid[jid])
        sg = signs.get(name)
        state = "unknown" if not sg else ("match" if sg.get("matches") else "inverted")
        detail = "" if not sg else f"rotor n {sg.get('n_base')}  axis {sg.get('axis')}"
        path = outdir / f"{order:02d}_{name}.mp4"
        sink = VideoSink(path)
        records = []
        poster = None
        prev = 0.0
        phases = ((0.0, hi, "0 -> + UPPER"), (hi, lo, "+ UPPER -> - LOWER"),
                  (lo, 0.0, "- LOWER -> 0"))
        for phase_index, (start, end, phase) in enumerate(phases):
            for k in range(transition_samples):
                t = k / max(transition_samples - 1, 1)
                angle = float(start + (end - start) * ease(t))
                mujoco.mj_resetData(model, data)
                data.qpos[:7] = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
                data.qpos[model.jnt_qposadr[jid]] = np.radians(angle)
                mujoco.mj_forward(model, data)
                anchor = data.xanchor[jid].copy()
                main_rgb = main_renderer.render(data, child, anchor)
                zoom_half = 0.28 if "crank" in name else (0.42 if "shoulder" in name else 0.34)
                zoom_rgb = zoom_renderer.render(data, child, anchor, zoom_half=zoom_half)
                frame = draw_frame(main_rgb, zoom_rgb, dict(
                    title=name, human=HUMAN.get(kind, ""), ang=angle, lo=lo, hi=hi,
                    phase=phase, model_tag=xml.name + " | CPU software render",
                    dir=angle - prev, sign_state=state, sign_detail=detail,
                    location=(f"joint anchor [base] ({anchor[0]:+.3f}, {anchor[1]:+.3f}, "
                              f"{anchor[2]:+.3f}) m; child={model.body(child).name}")))
                sink.add(frame, repeat=repeat)
                if poster is None:
                    poster = frame.copy().resize((320, 180))
                    draw = ImageDraw.Draw(poster)
                    draw.rectangle((0, 0, 320, 30), fill=(16, 24, 34))
                    draw.text((8, 6), f"{order:02d} {name}", font=font, fill="white")
                records.append(angle)
                prev = angle
            if phase_index < 2:
                sink.add(frame, repeat=hold_frames)
        sink.close()
        posters.append(poster)
        manifest[name] = {
            "order": order, "video": path.name, "lower_deg": float(lo), "upper_deg": float(hi),
            "duration_s": sink.frames / FPS, "frames": sink.frames,
            "reached_lower_deg": min(records), "reached_upper_deg": max(records),
            "returned_deg": records[-1], "sign_state": state,
            "sign_dot": None if not sg else sg.get("dot"),
            "highlighted_body": model.body(child).name,
        }
        print(f"[{order:02d}/{len(joints):02d}] {path.name} {sink.frames/FPS:.2f}s", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if posters:
        cols, rows = 4, (len(posters) + 3) // 4
        sheet = Image.new("RGB", (cols * 320, rows * 180), (20, 24, 30))
        for i, poster in enumerate(posters):
            sheet.paste(poster, ((i % cols) * 320, (i // cols) * 180))
        sheet.save(outdir / "joint_index.png")
    print(outdir / "manifest.json")


if __name__ == "__main__":
    main()
