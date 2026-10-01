"""Overlays the active velocity command on a rendered video (render.py output).

Usage: python overlay_cmds.py --video in.mp4 --out out.mp4 --cmd-schedule "3 1 0 0" "3 0 0.5 0" ...
"""

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cmd-schedule", nargs="+", required=True)
ap.add_argument("--size", type=int, default=20)
a = ap.parse_args()
r = imageio.get_reader(a.video)
fps = r.get_meta_data()["fps"]
segs = []
for seg in a.cmd_schedule:
    sec, vx, vy, wz = (float(x) for x in seg.split())
    segs += [f"command: vx {vx:+.1f} m/s, vy {vy:+.1f} m/s, yaw {wz:+.1f} rad/s"] * round(sec * fps)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", a.size)
except OSError:
    font = ImageFont.load_default()
with imageio.get_writer(a.out, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=8,
                        ffmpeg_params=["-movflags", "+faststart"]) as w:
    for i, frame in enumerate(r):
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im, "RGBA")
        text = segs[min(i, len(segs) - 1)]
        x0, y0, x1, y1 = d.textbbox((12, 10), text, font=font)
        d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0, 110))
        d.text((12, 10), text, font=font, fill=(255, 255, 255, 255))
        w.append_data(np.asarray(im))
print(f"wrote {a.out}")
