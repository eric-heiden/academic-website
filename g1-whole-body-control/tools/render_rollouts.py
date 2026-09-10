"""Render actual recorded MPC predictions, anchored at their planning state.

Run from the Newton WBC branch with the wbc extra and imageio-ffmpeg/Pillow.
The input is a local --show-rollouts --output archive; no physics is re-run.
"""

import argparse
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

import newton
import newton.utils
import newton.viewer
from newton.examples.robot.wbc_rollouts import TRACE_COLORS, draw_rollouts

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("trajectory")
p.add_argument("output")
p.add_argument("--poster-time", type=float, default=1.0)
p.add_argument("--camera-distance", type=float, default=2.8)
p.add_argument("--selected-only", action="store_true")
p.add_argument("--poster-only", action="store_true")
a = p.parse_args()
raw = np.load(a.trajectory)
times, poses, paths = raw["trace_time"], raw["trace_qpos"], raw["trace_positions"]
offsets = raw["trace_offsets"]
assert np.isfinite(paths[:, 0]).all(), "No selected prediction in part of this recording"
assert np.all(np.diff(times) >= 0), "Planning times must not run backwards"
with wp.ScopedDevice("cpu"):
    builder = newton.ModelBuilder()
    asset = newton.utils.download_asset("unitree_g1")
    builder.add_mjcf(str(asset / "mjcf/g1_29dof_rev_1_0.xml"), collapse_fixed_joints=True, enable_self_collisions=False)
    for i, shape_type in enumerate(builder.shape_type):
        if shape_type == newton.GeoType.PLANE:
            builder.shape_flags[i] |= newton.ShapeFlags.VISIBLE
            builder.shape_color[i] = (0.125, 0.125, 0.15)
    model = builder.finalize()
    state = model.state()
    viewer = newton.viewer.ViewerGL(
        width=960, height=720, headless=True, enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE
    )
    viewer.set_model(model)
    viewer.renderer.line_width = 2.2
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default(size=18)
    small = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default(size=16)
    writer = None
    if not a.poster_only:
        writer = imageio_ffmpeg.write_frames(
            a.output, (960, 720), fps=25, codec="libx264", pix_fmt_out="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        writer.send(None)
    best = float("inf")
    for requested in ([a.poster_time] if a.poster_only else np.arange(times[0], times[-1] + 1e-5, 0.04)):
        index = int(np.abs(times - requested).argmin())
        t, q = times[index], poses[index]
        nq = q.copy()
        nq[3:7] = q[[4, 5, 6, 3]]
        state.joint_q.assign(nq.astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        center = q[:3].copy()
        center[2] = 0.82
        yaw, pitch = 135.0, -12.0
        direction = np.array([
            np.cos(np.deg2rad(yaw)) * np.cos(np.deg2rad(pitch)),
            np.sin(np.deg2rad(yaw)) * np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch)),
        ])
        viewer.set_camera(wp.vec3(*(center - a.camera_distance * direction)), pitch=pitch, yaw=yaw)
        viewer.begin_frame(float(t))
        viewer.log_state(state)
        visible = paths[index, :1] if a.selected_only else paths[index]
        draw_rollouts(viewer, visible, offsets)
        viewer.end_frame()
        frame = Image.fromarray(viewer.get_frame().numpy())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 960, 78), fill="white")
        draw.text((18, 10), f"Predicted futures: next {offsets[-1]:.2f} s", font=font, fill="#192531")
        draw.text((804, 10), f"t = {t:.2f} s", font=font, fill="#192531")
        for body, label in enumerate(["Left foot", "Right foot", "Left hand", "Right hand", "Torso"][:visible.shape[2]]):
            x = 18 + body * 174
            color = tuple((TRACE_COLORS[body] * 210).astype(int))
            draw.line((x, 57, x + 23, 57), fill=color, width=4)
            draw.text((x + 31, 47), label, font=small, fill="#192531")
        draw.rectangle((0, 679, 960, 720), fill="white")
        description = "Solid: selected plan"
        if len(visible) > 1:
            description += f"   |   Muted dashes: {len(visible) - 1} candidate alternatives"
        draw.text((18, 692), description, font=small, fill="#192531")
        if writer:
            writer.send(np.asarray(frame))
        if abs(t - a.poster_time) < best:
            best, poster = abs(t - a.poster_time), frame.copy()
    if writer:
        writer.close()
    viewer.close()
    poster.save(Path(a.output).with_suffix(".jpg"), quality=95)
