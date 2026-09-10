"""Render saved simulated trajectories beside their kinematic reference."""

import os

os.environ["MUJOCO_GL"] = "egl"
import argparse
from pathlib import Path
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import newton.viewer
from newton.examples.robot.example_robot_g1_wbc import Example
from newton.examples.robot.wbc_controller import MotionReference

p = argparse.ArgumentParser()
p.add_argument("trajectory")
p.add_argument("output")
a = p.parse_args()
ex = Example(
    newton.viewer.ViewerNull(), Example.create_parser().parse_args(["--viewer", "null"])
)
m = ex.mj
m.vis.global_.offwidth = 640
m.vis.global_.offheight = 480
m.vis.headlight.ambient[:] = 0.5
m.vis.headlight.diffuse[:] = 0.8
m.vis.rgba.fog[:] = [0.96, 0.97, 0.98, 1]
d = mujoco.MjData(m)
render = mujoco.Renderer(m, height=480, width=640)
raw = np.load(a.trajectory)
ref = MotionReference(m, raw["reference"])
rows = raw["rows"]
poses = raw["qpos"]
cam = mujoco.MjvCamera()
cam.azimuth = 100
cam.elevation = -12
cam.distance = 3.1
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
writer = imageio_ffmpeg.write_frames(
    a.output,
    (1280, 480),
    fps=25,
    codec="libx264",
    pix_fmt_out="yuv420p",
    output_params=["-crf", "22", "-movflags", "+faststart"],
)
writer.send(None)
poster = None
for index in range(0, len(rows), max(1, round(0.04 / np.median(np.diff(rows[:, 0]))))):
    t = rows[index, 0]
    qr = ref.sample(t)[0]
    qa = poses[index]
    # A shared tracking camera preserves the visible spatial error.
    cam.lookat[:] = (qr[:3] + qa[:3]) / 2
    cam.lookat[2] = 0.7
    cam.distance = max(3.1, float(np.linalg.norm(qr[:2] - qa[:2])) * 1.15 + 1.5)
    views = []
    for q in (qr, qa):
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        render.update_scene(d, camera=cam)
        views.append(render.render())
    frame = Image.fromarray(np.concatenate(views, axis=1))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 1280, 37), fill="#ffffff")
    draw.text((14, 7), "Kimodo reference (kinematic)", font=font, fill="#192531")
    draw.text((654, 7), "SolverMuJoCo (actuated simulation)", font=font, fill="#192531")
    draw.text((1150, 7), f"{t:.2f} s", font=font, fill="#192531")
    writer.send(np.asarray(frame))
    if poster is None or abs(t - 1) < 0.015:
        poster = frame.copy()
writer.close()
render.close()
poster.save(Path(a.output).with_suffix(".jpg"), quality=90)
