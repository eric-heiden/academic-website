# SPDX-License-Identifier: Apache-2.0
"""Record shake-test videos headless with ViewerGL under Xvfb.

Usage:
    xvfb-run -a uv run python -m newtontests.record --variant baseline \
        --out-dir /path/to/media --name baseline
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import warp as wp

import newton

from newtontests.franka_cube_shake import Phase, create_parser
from newtontests.experiments import Variant, VARIANTS
from newtontests.measure import rel_pose


def record(variant, name, out_dir, num_frames=900, amplitude=0.03, frequency=1.0,
           stride=2, fps=30, camera=None, overlay=True):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    argv_backup = sys.argv
    sys.argv = ["record", "--viewer", "gl", "--headless", "--num-frames", str(num_frames)]
    viewer, args = newton.examples.init(create_parser())
    sys.argv = argv_backup
    args.shake_amplitude = amplitude
    args.shake_frequency = frequency

    ex = Variant(viewer, args, VARIANTS[variant])
    if camera is not None:
        viewer.set_camera(pos=wp.vec3(*camera["pos"]), pitch=camera["pitch"], yaw=camera["yaw"])

    video_path = os.path.join(out_dir, f"{name}.mp4")
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=7, macro_block_size=None)

    ref = None
    poster_saved = False
    for frame in range(num_frames):
        ex.step()
        bq = ex.state_0.body_q.numpy()
        rp, _ = rel_pose(bq[ex.ee_index], bq[ex.cube_index])
        phase = int(ex.phase_index.numpy()[0])
        if phase == Phase.SHAKE.value and ref is None:
            ref = rp.copy()
        slip_mm = float(np.linalg.norm(rp - ref)) * 1000.0 if ref is not None else 0.0

        ex.render()
        if frame % stride == 0:
            img = viewer.get_frame().numpy()
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            if overlay:
                pil = Image.fromarray(img)
                dr = ImageDraw.Draw(pil)
                dr.rectangle([0, 0, 300, 56], fill=(12, 18, 28))
                dr.text((10, 8), f"{name}   t={frame / 60.0:5.2f}s   {Phase(phase).name}", fill=(235, 240, 250))
                dr.text((10, 30), f"slip in gripper frame: {slip_mm:6.2f} mm", fill=(120, 230, 200))
                img = np.asarray(pil)
            writer.append_data(img)
            if not poster_saved and frame > 0.6 * num_frames:
                Image.fromarray(img).save(os.path.join(out_dir, f"{name}.jpg"))
                poster_saved = True

    writer.close()
    if hasattr(viewer, "close"):
        viewer.close()
    print(f"wrote {video_path}  final slip {slip_mm:.2f} mm", flush=True)
    return video_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--amplitude", type=float, default=0.03)
    ap.add_argument("--frequency", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    cam = {"pos": (0.20, -0.72, 0.34), "pitch": -12.0, "yaw": 118.0}
    record(a.variant, a.name or a.variant, a.out_dir, a.frames, a.amplitude, a.frequency,
           stride=a.stride, camera=cam)
