"""Replay measured qpos through Newton visual shapes, beside the reference."""

import argparse
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

import newton
import newton.utils
import newton.viewer
from newton.examples.robot.wbc_controller import MotionReference

p = argparse.ArgumentParser()
p.add_argument("trajectory")
p.add_argument("output")
p.add_argument("--poster-time", type=float, default=1.0)
p.add_argument("--camera-distance", type=float, default=2.55)
p.add_argument("--poster-only", action="store_true")
p.add_argument("--thumbnail", help="Optional 320 by 320 JPEG of the simulated pose")
a = p.parse_args()
# Rendering replays saved states on CPU; this does not re-simulate the trajectory.
with wp.ScopedDevice("cpu"):
    builder = newton.ModelBuilder()
    asset = newton.utils.download_asset("unitree_g1")
    xml = str(asset / "mjcf/g1_29dof_rev_1_0.xml")
    builder.add_mjcf(xml, collapse_fixed_joints=True, enable_self_collisions=False)
    for i, shape_type in enumerate(builder.shape_type):
        if shape_type == newton.GeoType.PLANE:
            builder.shape_flags[i] |= newton.ShapeFlags.VISIBLE
            builder.shape_color[i] = (0.125, 0.125, 0.15)
    model = builder.finalize()
    state = model.state()
    viewer = newton.viewer.ViewerGL(
        width=640, height=640, headless=True, enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE
    )
    viewer.set_model(model)
    raw = np.load(a.trajectory)
    reference = MotionReference(mujoco.MjModel.from_xml_path(xml), raw["reference"])
    rows, poses = raw["rows"], raw["qpos"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    writer = imageio_ffmpeg.write_frames(
        a.output,
        (1280, 640),
        fps=25,
        codec="libx264",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    poster_distance = float("inf")
    times = [a.poster_time] if a.poster_only else np.arange(rows[0, 0], rows[-1, 0], 0.04)
    for t in times:
        index = min(int(np.searchsorted(rows[:, 0], t)), len(rows) - 1)
        t = rows[index, 0]
        qr, qa = reference.sample(t)[0], poses[index]
        center = (qr[:3] + qa[:3]) / 2
        center[2] = 0.8
        distance = max(a.camera_distance, float(np.linalg.norm(qr[:2] - qa[:2])) + 1.8)
        yaw, pitch = 135.0, -12.0
        direction = np.array(
            [
                np.cos(np.deg2rad(yaw)) * np.cos(np.deg2rad(pitch)),
                np.sin(np.deg2rad(yaw)) * np.cos(np.deg2rad(pitch)),
                np.sin(np.deg2rad(pitch)),
            ]
        )
        viewer.set_camera(wp.vec3(*(center - distance * direction)), pitch=pitch, yaw=yaw)
        views = []
        for q in (qr, qa):
            nq = q.copy()
            nq[3:7] = q[[4, 5, 6, 3]]
            state.joint_q.assign(nq.astype(np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            viewer.begin_frame(float(t))
            viewer.log_state(state)
            viewer.end_frame()
            views.append(viewer.get_frame().numpy())
        frame = Image.fromarray(np.concatenate(views, axis=1))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 1280, 38), fill="#ffffff")
        draw.text((14, 9), "Kimodo reference (kinematic)", font=font, fill="#192531")
        draw.text((654, 9), "SolverMuJoCo (actuated simulation)", font=font, fill="#192531")
        draw.text((1170, 9), f"{t:.2f} s", font=font, fill="#192531")
        draw.line((640, 38, 640, 640), fill="#ffffff", width=2)
        writer.send(np.asarray(frame))
        if abs(t - a.poster_time) < poster_distance:
            poster_distance = abs(t - a.poster_time)
            poster = frame.copy()
            thumbnail = Image.fromarray(views[1]).resize((320, 320), Image.Resampling.LANCZOS)
    writer.close()
    viewer.close()
    poster.save(Path(a.output).with_suffix(".jpg"), quality=94)
    if a.thumbnail:
        thumbnail.save(a.thumbnail, quality=94)
