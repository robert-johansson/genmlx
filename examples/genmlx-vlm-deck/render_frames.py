#!/usr/bin/env python3
"""Render ARC-AGI-3 frame PNGs for the VLM deck from dev/arc_frames/*.json.

The VLM needs an image per scene. This recovers the exact 16-colour palette from
the existing sk48_frame0.png (a grid+PNG pair), then renders the most colourful
frame of each game to <game>_deck.png at 8x. The deck draws the terminal grid
from the same frame JSON with the same palette, so what you see is what the VLM saw.

Run from the repo root:   python3 examples/genmlx-vlm-deck/render_frames.py
Requires Pillow (PIL).
"""
import json
from PIL import Image

FR = "dev/arc_frames"
# fallback for colour indices not present in sk48 (kept distinct from the bg/white)
DEFAULT = {0:(0,0,0),1:(0,116,217),2:(255,65,54),3:(46,204,64),4:(255,220,0),
           5:(170,170,170),6:(240,18,190),7:(255,133,27),8:(127,219,255),9:(135,12,37),
           10:(146,18,49),11:(255,220,0),12:(177,13,201),13:(57,204,204),14:(90,30,160),15:(127,219,255)}

# recover the real palette from the sk48 grid + its rendered PNG
grid = json.load(open(f"{FR}/sk48.json"))["steps"][0]["frame"][0]
img = Image.open(f"{FR}/sk48_frame0.png").convert("RGB")
W, H = img.size; sx, sy = W // 64, H // 64
pal = dict(DEFAULT)
for r in range(64):
    for c in range(64):
        pal[grid[r][c]] = img.getpixel((c * sx + sx // 2, r * sy + sy // 2))

CELL = 8
for g in ["sk48", "g50t", "re86", "bp35"]:
    steps = json.load(open(f"{FR}/{g}.json"))["steps"]
    best = max(range(len(steps)),
               key=lambda i: len({v for row in steps[i]["frame"][0] for v in row}))
    grid = steps[best]["frame"][0]
    h, w = len(grid), len(grid[0])
    im = Image.new("RGB", (w * CELL, h * CELL)); px = im.load()
    for r in range(h):
        for c in range(w):
            col = pal.get(grid[r][c], (255, 0, 255))
            for dy in range(CELL):
                for dx in range(CELL):
                    px[c * CELL + dx, r * CELL + dy] = col
    im.save(f"{FR}/{g}_deck.png")
    print(f"{g}: rendered frame {best}/{len(steps)-1} -> {FR}/{g}_deck.png")
print("PALETTE " + " ".join(f"{i}:#{pal[i][0]:02X}{pal[i][1]:02X}{pal[i][2]:02X}" for i in range(16)))
