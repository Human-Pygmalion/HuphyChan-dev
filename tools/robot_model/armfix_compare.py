#!/usr/bin/env python3
"""Make a reviewer-facing before/after crop for the shoulder-roll mesh ownership fix."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tools/robot_model/fusion_snapshots/v30_inspection"
OLD = BASE / "renders/SemiFullDoF_front.png"
NEW = BASE / "renders_armfix/SemiFullDoF_front.png"
OUT = BASE / "renders_armfix/arm_housing_before_after.png"


def panel(path: Path, label: str, subtitle: str) -> Image.Image:
    source = Image.open(path).convert("RGB").crop((0, 38, 640, 330))
    canvas = Image.new("RGB", (640, 350), "#f3f5f6")
    canvas.paste(source, (0, 58))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 640, 58), fill="#132334")
    draw.text((15, 10), label, fill="white")
    draw.text((15, 32), subtitle, fill="#d9e5ed")
    # Mark both roll housings.  The red circles cover the same world-image regions, making
    # it obvious whether the outer housing stays level or follows the abducted arm.
    draw.ellipse((205, 69, 313, 170), outline="#ef3e42", width=4)
    draw.ellipse((327, 69, 435, 170), outline="#ef3e42", width=4)
    return canvas


left = panel(OLD, "BEFORE", "Shoulder-Roll2Yaw housing owned by fixed shoulder-pitch link")
right = panel(NEW, "AFTER", "housing moved downstream; housing + rod rotate together at 15 deg")
sheet = Image.new("RGB", (1280, 350), "white")
sheet.paste(left, (0, 0)); sheet.paste(right, (640, 0))
ImageDraw.Draw(sheet).line((640, 0, 640, 350), fill="#687787", width=3)
sheet.save(OUT)
print(OUT)
