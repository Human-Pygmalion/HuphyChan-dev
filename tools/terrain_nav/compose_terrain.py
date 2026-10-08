#!/usr/bin/env python3
"""Article picture, mountain trail: real RViz close-up behind the robot + top-down route inset + one line that says
how it moves (autonomy pipeline, not remote control).  usage: compose_terrain.py <hero.png> <top.png> <out.png> [--stats "182 m, 9 m descent"]"""
import argparse

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("hero")
ap.add_argument("top")
ap.add_argument("out")
ap.add_argument("--stats", default="")
a = ap.parse_args()

CROP = (470, 70, 2540, 1400)  # RViz 3D viewport inside the 2560x1440 capture (panels removed)
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

img = Image.open(a.hero).convert("RGB").crop(CROP)
W, H = img.size
d = ImageDraw.Draw(img, "RGBA")

# inset: whole route seen from above
top = Image.open(a.top).convert("RGB").crop(CROP)
iw = int(W * 0.34)
top = top.resize((iw, int(top.height * iw / top.width)), Image.LANCZOS)
M = 30
x0, y0 = W - iw - M, M
img.paste(top, (x0, y0))
d.rectangle([x0 - 2, y0 - 2, x0 + iw + 1, y0 + top.height + 1], outline=(235, 238, 242, 255), width=2)
fc = ImageFont.truetype(FR, 30)
cap = "Planned route" + (f"  {a.stats}" if a.stats else "")
l, t, r, b = d.textbbox((0, 0), cap, font=fc)
d.rectangle([x0, y0 + top.height, x0 + iw, y0 + top.height + (b - t) + 20], fill=(255, 255, 255, 200))
d.text((x0 + 12, y0 + top.height + 10 - t), cap, font=fc, fill=(25, 28, 34, 255))

# bottom line: how the robot moves
fb, fr = ImageFont.truetype(FB, 40), ImageFont.truetype(FR, 36)
steps = ["LiDAR map", "terrain classes", "A* path", "velocity command", "RL walking policy"]
bar_h = 92
d.rectangle([0, H - bar_h, W, H], fill=(10, 12, 16, 215))
x = 34
head = "AUTONOMOUS  "
d.text((x, H - bar_h / 2), head, font=fb, fill=(60, 235, 130, 255), anchor="lm")
x += d.textlength(head, font=fb) + 10
for i, s in enumerate(steps):
    d.text((x, H - bar_h / 2), s, font=fr, fill=(236, 239, 244, 255), anchor="lm")
    x += d.textlength(s, font=fr)
    if i < len(steps) - 1:
        d.text((x + 14, H - bar_h / 2), "→", font=fr, fill=(150, 156, 166, 255), anchor="lm")
        x += 14 + d.textlength("→", font=fr) + 14
img.save(a.out)
print("saved", a.out, img.size)
