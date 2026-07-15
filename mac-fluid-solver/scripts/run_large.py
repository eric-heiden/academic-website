"""Run the large-scale long-horizon swimmer rollouts: videos, metrics, perf.

Run from the newton worktree under Xvfb:
    xvfb-run -a uv run --extra sim --extra examples --with "imageio[ffmpeg]" \
        python <this script> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

CASES = {
    # 45+ second out-and-back marathon in an 8 m tank
    "swimmer_marathon": {
        "argv": [
            "--tank-length", "8", "--tank-width", "1.6", "--fluid-res", "384",
            "--pressure-iterations", "160", "--reverse-at", "13",
        ],
        "frames": 1830,  # 30.5 s (returns past its start without leaving the tank)
        "camera": {"pos": (0.0, -5.4, 4.0), "pitch": -33.0, "yaw": 90.0},
    },
    # three swimmers with different gait frequencies race out and back (wide lanes)
    "swimmer_race": {
        "argv": [
            "--num-swimmers", "3", "--tank-length", "8", "--tank-width", "3.2",
            "--fluid-res", "384", "--swimmer-frequencies", "0.8,1.2,1.6",
            "--amplitude", "0.4",
            "--pressure-iterations", "160", "--reverse-at", "14",
        ],
        "frames": 1680,  # 28 s (wide lanes keep fast tails clear of the side walls)
        "camera": {"pos": (0.0, -6.8, 5.2), "pitch": -33.0, "yaw": 90.0},
    },
    # nine-link eel cruising one way down a 14 m tank
    "swimmer_eel": {
        "argv": [
            "--num-links", "9", "--tank-length", "14", "--tank-width", "1.6",
            "--tank-depth", "0.8", "--fluid-res", "672", "--frequency", "0.7",
            "--amplitude", "0.45", "--start-x", "-6.0", "--pressure-iterations", "160",
        ],
        "frames": 1980,  # 33 s one-way cruise
        "camera": {"pos": (0.0, -8.8, 6.6), "pitch": -33.0, "yaw": 90.0},
    },
}


def run_case(case_name, out_dir, spec):
    import imageio.v2 as imageio
    import warp as wp
    from PIL import Image

    import newton.examples
    import newton.examples.multiphysics.example_macfluid_swimmer as M

    metrics_path = os.path.join(out_dir, "data", f"{case_name}.json")
    sys.argv = [
        "example",
        "--viewer",
        "gl",
        "--headless",
        "--num-frames",
        str(spec["frames"]),
        "--metrics-output",
        metrics_path,
        *spec["argv"],
    ]
    parser = M.Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = M.Example(viewer, args)
    cam = spec.get("camera")
    if cam is not None:
        viewer.set_camera(pos=wp.vec3(*cam["pos"]), pitch=cam["pitch"], yaw=cam["yaw"])

    video_path = os.path.join(out_dir, "media", f"{case_name}.mp4")
    writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=6, macro_block_size=None)

    frames = spec["frames"]
    t_start = time.perf_counter()
    t_step = 0.0
    screenshot_saved = False
    for frame in range(frames):
        t0 = time.perf_counter()
        example.step()
        t_step += time.perf_counter() - t0
        example.render()
        if frame % 2 == 0:
            writer.append_data(viewer.get_frame().numpy())
        if not screenshot_saved and frame == int(0.35 * frames):
            img = viewer.get_frame().numpy()
            h, w = img.shape[:2]
            side = min(h, w)
            crop = img[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
            Image.fromarray(crop).resize((320, 320)).save(os.path.join(out_dir, "media", f"{case_name}.jpg"))
            screenshot_saved = True
    wall = time.perf_counter() - t_start
    writer.close()
    example.metrics.save()

    # pure simulation timing: no per-frame diagnostics readback, no rendering
    example._record_metrics = lambda: None
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(120):
        example.step()
    wp.synchronize_device()
    sim_only_ms = 1000.0 * (time.perf_counter() - t0) / 120

    perf = {
        "case": case_name,
        "frames": frames,
        "sim_seconds": frames / 60.0,
        "fluid_cells": example.metrics.meta["fluid_cells"],
        "bodies": int(example.model.body_count),
        "joints": int(example.model.joint_count),
        "step_ms_with_diagnostics": 1000.0 * t_step / frames,
        "sim_only_ms": sim_only_ms,
        "realtime_factor_sim_only": (1000.0 / 60.0) / sim_only_ms,
        "wall_seconds_total_with_video": wall,
    }
    with open(os.path.join(out_dir, "data", f"{case_name}_perf.json"), "w") as f:
        json.dump(perf, f, indent=1)
    print(json.dumps(perf, indent=1))
    if hasattr(viewer, "close"):
        viewer.close()
    print(f"done: {case_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--case", type=str, default=None)
    args = parser.parse_args()
    out = args.output_dir
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    os.makedirs(os.path.join(out, "media"), exist_ok=True)

    if args.case is None:
        import subprocess

        for name in CASES:
            print(f"=== {name} ===", flush=True)
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--output-dir", out, "--case", name],
                check=True,
            )
        return

    run_case(args.case, out, CASES[args.case])


if __name__ == "__main__":
    main()
