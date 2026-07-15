"""Run the MAC fluid examples headless, recording metrics JSON and videos.

Run from the newton worktree under Xvfb:
    xvfb-run -a uv run --extra sim --extra examples --with "imageio[ffmpeg]" \
        python <this script> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np


def run_case(module_name, case_name, out_dir, num_frames, extra_args, video=True, video_stride=2, camera=None):
    import imageio.v2 as imageio
    from PIL import Image

    import newton.examples

    module = importlib.import_module(module_name)

    metrics_path = os.path.join(out_dir, "data", f"{case_name}.json")
    sys.argv = [
        "example",
        "--viewer",
        "gl" if video else "null",
        "--headless",
        "--num-frames",
        str(num_frames),
        "--metrics-output",
        metrics_path,
        *extra_args,
    ]
    parser = module.Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = module.Example(viewer, args)
    if camera is not None and hasattr(viewer, "set_camera"):
        import warp as wp

        viewer.set_camera(pos=wp.vec3(*camera["pos"]), pitch=camera["pitch"], yaw=camera["yaw"])

    writer = None
    screenshot_saved = False
    if video:
        video_path = os.path.join(out_dir, "media", f"{case_name}.mp4")
        writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=7, macro_block_size=None)

    for frame in range(num_frames):
        example.step()
        if writer is not None:
            example.render()
            if frame % video_stride == 0:
                img = viewer.get_frame().numpy()
                writer.append_data(img)
            if not screenshot_saved and frame == int(0.4 * num_frames):
                img = viewer.get_frame().numpy()
                h, w = img.shape[:2]
                side = min(h, w)
                crop = img[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
                Image.fromarray(crop).resize((320, 320)).save(os.path.join(out_dir, "media", f"{case_name}.jpg"))
                screenshot_saved = True

    if writer is not None:
        writer.close()
    example.metrics.save()
    if hasattr(viewer, "close"):
        viewer.close()
    print(f"done: {case_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--case", type=str, default=None, help="Run a single named case.")
    args = parser.parse_args()
    out = args.output_dir
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    os.makedirs(os.path.join(out, "media"), exist_ok=True)

    sphere = "newton.examples.multiphysics.example_macfluid_settling_sphere"
    paddle = "newton.examples.multiphysics.example_macfluid_paddle"
    swimmer = "newton.examples.multiphysics.example_macfluid_swimmer"

    sphere_cam = {"pos": (1.35, -1.35, 0.95), "pitch": -16.0, "yaw": 135.0}
    paddle_cam = {"pos": (1.15, -1.15, 1.05), "pitch": -35.0, "yaw": 135.0}
    swimmer_cam = {"pos": (1.5, -1.9, 1.35), "pitch": -30.0, "yaw": 128.0}

    cases = {
        "sphere_settling": (sphere, 300, [], True, sphere_cam),
        "sphere_rising": (sphere, 300, ["--sphere-density", "500"], True, sphere_cam),
        "sphere_dry": (sphere, 300, ["--dry"], False, None),
        "paddle_wet": (paddle, 420, [], True, paddle_cam),
        "paddle_dry": (paddle, 420, ["--dry"], False, None),
        "swimmer_forward": (swimmer, 480, [], True, swimmer_cam),
        "swimmer_reverse": (swimmer, 480, ["--reverse"], True, swimmer_cam),
        "swimmer_dry": (swimmer, 480, ["--dry"], False, None),
    }

    if args.case is None:
        # one subprocess per case: fresh CUDA/GL context each time
        import subprocess

        for name in cases:
            print(f"=== {name} ===", flush=True)
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--output-dir", out, "--case", name],
                check=True,
            )
        return

    module, frames, extra, video, cam = cases[args.case]
    run_case(module, args.case, out, frames, extra, video=video, camera=cam)


if __name__ == "__main__":
    main()
