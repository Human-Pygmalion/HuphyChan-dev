#!/usr/bin/env python3
"""CPU-render the staged Pygmalion MJCF variants for visual inspection."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import trimesh


VIEWS = (("front", 0.0, 7.0), ("iso", 35.0, 12.0), ("side", 90.0, 7.0))
WIDTH, HEIGHT, DPI = 640, 480, 100


def basis(azimuth: float, elevation: float) -> tuple[np.ndarray, ...]:
    az, el = np.radians((azimuth, elevation))
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


def geom_triangles(model: mujoco.MjModel, data: mujoco.MjData, geom: int):
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = model.mesh_vert[va:va + vn]
        faces = model.mesh_face[fa:fa + fn]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        primitive = trimesh.creation.cylinder(
            radius=float(model.geom_size[geom, 0]),
            height=float(2 * model.geom_size[geom, 1]), sections=24)
        vertices, faces = np.asarray(primitive.vertices), np.asarray(primitive.faces)
    else:
        return None
    rotation = data.geom_xmat[geom].reshape(3, 3)
    return vertices @ rotation.T + data.geom_xpos[geom], faces


def render(xml: Path, out: Path, title: str) -> list[Path]:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    data.qpos[:7] = (0.0, 0.0, 0.91, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    paths = []
    for label, azimuth, elevation in VIEWS:
        right, up, forward = basis(azimuth, elevation)
        polygons, colors, depth = [], [], []
        for geom in range(model.ngeom):
            if model.geom_group[geom] != 2:
                continue
            item = geom_triangles(model, data, geom)
            if item is None:
                continue
            vertices, faces = item
            projected = np.column_stack((vertices @ right, vertices @ up))
            polygons.append(projected[faces])
            depth.append((vertices @ forward)[faces].mean(1))
            normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                               vertices[faces[:, 2]] - vertices[faces[:, 0]])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            light = right * 0.35 + up * 0.55 - forward * 0.75
            light /= np.linalg.norm(light)
            shade = np.clip(0.28 + 0.72 * np.abs(normals @ light), 0, 1)
            base = np.array([0.70, 0.73, 0.78])
            colors.append(np.clip(base * shade[:, None], 0, 1))
        poly = np.vstack(polygons)
        color = np.vstack(colors)
        order = np.argsort(np.concatenate(depth))[::-1]
        points = poly.reshape(-1, 2)
        low, high = points.min(0), points.max(0)
        centre = (low + high) / 2
        half_y = (high[1] - low[1]) * 0.56 + 0.03
        half_x = max((high[0] - low[0]) * 0.56 + 0.03, half_y * WIDTH / HEIGHT)
        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.add_collection(PolyCollection(poly[order], facecolors=color[order],
                                         edgecolors="none", rasterized=True))
        ax.set_xlim(centre[0] - half_x, centre[0] + half_x)
        ax.set_ylim(centre[1] - half_y, centre[1] + half_y)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor((0.96, 0.96, 0.96))
        path = out / f"{title}_{label}.png"
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, WIDTH, 38), fill=(18, 18, 18))
        draw.text((14, 12), f"{title} | {label}", fill=(255, 255, 255))
        image.save(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--semi", type=Path, required=True)
    parser.add_argument("--leg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [render(xml, args.out, title) for title, xml in (
        ("FullDoF", args.full), ("SemiFullDoF", args.semi), ("LegOnly", args.leg))]
    sheet = Image.new("RGB", (WIDTH * 3, HEIGHT * 3), (245, 245, 245))
    for row, paths in enumerate(rows):
        for column, path in enumerate(paths):
            sheet.paste(Image.open(path), (WIDTH * column, HEIGHT * row))
    sheet_path = args.out / "Pygmalion_v30_variants_all_views.png"
    sheet.save(sheet_path)
    print(sheet_path)


if __name__ == "__main__":
    main()
