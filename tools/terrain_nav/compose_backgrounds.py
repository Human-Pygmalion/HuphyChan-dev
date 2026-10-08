#!/usr/bin/env python3
"""Background variants of the article picture (layout_trail_autonomy): calm solid colours and HDRI skies.

The RViz shots are taken twice, on a black and on a white background (capture_terrain.sh with *_bg0 / *_bg255 configs).
From the pair, every pixel's coverage is exact:  black = a*F,  white = a*F + (1-a)*255  ->  1-a = (white-black)/255,
so any background G goes in without halos:  out = black + (1-a)*G.  The scene itself (map, path, robot) is unchanged.

Solid colours replace the whole background (light and calm).  The HDRI sky would show through the see-through ground,
so for the HDRI versions it is a backdrop behind the upper part of the view (the canopy), fading to a dark ground tone
over the middle of the picture.  The top-down inset has no sky: a solid colour.

HDRI backdrop: a window of the equirectangular sky, azimuth chosen facing away from the sun (calm, no glare),
elevation -10..+40 deg top-to-bottom reversed onto the picture, Reinhard tone mapping, slight blur (out of focus).
usage: compose_backgrounds.py <dir with hero_bg0/255.png, top_bg0/255.png> <out dir> [hdri ...]
"""
import os
import subprocess
import sys

import cv2
import numpy as np

src, out = sys.argv[1], sys.argv[2]
hdris = sys.argv[3:]
os.makedirs(out, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
SOLID = {"mist": (228, 234, 240), "stone": (236, 232, 225)}  # light blue-grey, warm light grey (RGB)


def load(name):
    return cv2.imread(os.path.join(src, name), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)  # RGB


def matte(stem):
    B, W = load(f"{stem}_bg0.png"), load(f"{stem}_bg255.png")
    t = np.clip((W - B).mean(-1, keepdims=True) / 255.0, 0, 1)  # 1 - coverage
    solid = t[..., 0] < 0.02
    print(f"{stem}: background {np.mean(t > 0.98) * 100:.0f} % of pixels, partly covered {np.mean((t > 0.02) & (t < 0.98)) * 100:.1f} %, "
          f"black/white mismatch on solid pixels {np.abs(W - B)[solid].mean():.2f} (0 = the two shots line up)")
    return B, t


def sky(path, h, w, el_top=40.0, el_bot=-10.0):
    E = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    EH, EW = E.shape[:2]
    lum = E.mean(-1)
    sun_az = (np.unravel_index(np.argmax(cv2.GaussianBlur(lum, (0, 0), 5)), lum.shape)[1] / EW) * 360.0
    az0 = (sun_az + 180.0) % 360.0  # look away from the sun
    span = (el_top - el_bot) * w / h
    az = (az0 + np.linspace(-span / 2, span / 2, w)) % 360.0
    el = np.linspace(el_top, el_bot, h)
    mx = (az / 360.0 * EW).astype(np.float32)[None, :].repeat(h, 0)
    my = ((90.0 - el) / 180.0 * EH).astype(np.float32)[:, None].repeat(w, 1)
    win = cv2.remap(E, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    L = win.mean(-1)
    win *= 0.6 / max(float(np.percentile(L, 70)), 1e-6)  # exposure: most of the sky at mid-light
    win = win / (1.0 + win)  # Reinhard
    win = np.clip(win, 0, 1) ** (1 / 2.2) * 255.0
    win = cv2.GaussianBlur(win, (0, 0), 2.0)
    return win, sun_az


def compose(hero, top, name, border):
    hp, tp = os.path.join(out, f"hero_{name}.png"), os.path.join(out, f"top_{name}.png")
    cv2.imwrite(hp, np.clip(hero, 0, 255).astype(np.uint8)[..., ::-1])
    cv2.imwrite(tp, np.clip(top, 0, 255).astype(np.uint8)[..., ::-1])
    lay = os.path.join(out, f"layout_trail_autonomy_{name}.png")
    subprocess.run([sys.executable, os.path.join(HERE, "compose_terrain.py"), hp, tp, lay, "--stats", "182 m", "--border", border], check=True)
    return lay


Bh, th = matte("hero")
# HDRI versions: the sky is a backdrop behind the upper part of the 3D view (behind the canopy), fading over the middle
# into a dark ground tone, so the sky never shows through the ground.  Solid versions replace the whole background.
VIEW_TOP, VIEW_BOT = 70, 1400  # RViz 3D viewport rows in the 2560 x 1440 capture (compose_terrain.py CROP)
rows = (np.arange(Bh.shape[0], dtype=np.float32) - VIEW_TOP) / (VIEW_BOT - VIEW_TOP)
wsky = np.clip((0.58 - rows) / 0.33, 0, 1)[:, None, None]  # 1 down to 25 % of the height, 0 from 58 %
DARK = np.array([22, 24, 24], np.float32)


Bt, tt = matte("top")
H, W = Bh.shape[:2]
made = []
for name, c in SOLID.items():
    G = np.array(c, np.float32)
    made.append(compose(Bh + th * G, Bt + tt * G, name, "60,66,74"))
G_top = np.array(SOLID["mist"], np.float32)  # top-down inset: a sky makes no sense looking straight down
for p in hdris:
    nm = os.path.basename(p).replace("_4k.hdr", "").replace(".hdr", "")
    g, sun = sky(p, H, W)
    print(f"{nm}: sun at azimuth {sun:.0f} deg, window centred at {(sun + 180) % 360:.0f} deg")
    made.append(compose(Bh + th * (wsky * g + (1 - wsky) * DARK), Bt + tt * G_top, f"hdri_{nm}", "235,238,242"))
print("\n".join(made))
