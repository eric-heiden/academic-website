"""Renders recorded qpos trajectories (from evaluate.py --record) to MP4.

The trajectory was simulated by MJWarp; here native MuJoCo only computes forward
kinematics for each recorded configuration and rasterises it with a tracking camera.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True)
p.add_argument("--episode", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--width", type=int, default=960)
p.add_argument("--height", type=int, default=540)
p.add_argument("--distance", type=float, default=3.0)
p.add_argument("--elevation", type=float, default=-15.0)
p.add_argument("--azimuth", type=float, default=135.0)
p.add_argument("--seconds", type=float, default=None)
p.add_argument("--poster-time", type=float, default=2.0)
args = p.parse_args()

data = np.load(args.npz, allow_pickle=True)
qpos = data["qpos"][args.episode]
control_dt = float(data["control_dt"])
xml = str(data["xml"])
m = mujoco.MjModel.from_xml_path(xml)
m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.width)
m.vis.global_.offheight = max(m.vis.global_.offheight, args.height)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, args.height, args.width)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = args.distance, args.elevation, args.azimuth
fps = round(1.0 / control_dt)
n = len(qpos) if args.seconds is None else min(len(qpos), round(args.seconds * fps))
target = qpos[0, :3].copy()
with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p",
                        macro_block_size=8, ffmpeg_params=["-movflags", "+faststart"]) as w:
    for t in range(n):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        # Smoothly follow the root in the horizontal plane at a fixed height.
        target[:2] = qpos[t, :2]
        target[2] = 0.6 * float(np.clip(qpos[0, 2], 0.3, 1.2))
        cam.lookat[:] = target
        renderer.update_scene(d, camera=cam)
        frame = renderer.render()
        w.append_data(frame)
        if t == min(n - 1, round(fps * args.poster_time)):
            imageio.imwrite(os.path.splitext(args.out)[0] + "_poster.jpg", frame, quality=88)
print(f"wrote {args.out} ({n} frames at {fps} fps)")
