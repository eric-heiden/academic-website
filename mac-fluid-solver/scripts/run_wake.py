"""Realistic-scale 50 cm swimmer scenarios: wake fidelity and scale realism.

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

# 50 cm swimmer, water viscosity, near-neutral density, 7.8 mm grid
BASE_50CM = [
    "--swimmer-scale", "0.538",  # 0.50 m body length
    "--frequency", "1.6",
    "--viscosity", "1e-6",  # water
    "--link-density", "1200",
    "--proxy-relaxation", "0.5", "--proxy-relaxation-mode", "aitken",
    "--joint-ke", "8", "--joint-kd", "0.3",
    "--pressure-iterations", "160",
    "--tank-width", "0.8", "--tank-depth", "0.4",
    "--slice-field", "vorticity", "--slice-scale", "8.0",
]

CASES = {
    # out-and-back rollout of the realistic-scale robot (MacCormack advection)
    "swimmer_50cm": {
        "argv": BASE_50CM + ["--tank-length", "4", "--fluid-res", "512",
                             "--advection", "maccormack", "--reverse-at", "11"],
        "frames": 1560,  # 26 s
        "camera": {"pos": (0.0, -2.9, 2.1), "pitch": -33.0, "yaw": 90.0},
    },
    # wake-persistence comparison: identical one-way cruises, two advection schemes
    "wake_maccormack": {
        "argv": BASE_50CM + ["--tank-length", "4", "--fluid-res", "512",
                             "--advection", "maccormack", "--start-x", "-1.4"],
        "frames": 900,  # 15 s
        "camera": {"pos": (0.0, -2.9, 2.1), "pitch": -33.0, "yaw": 90.0},
    },
    "wake_semi_lagrangian": {
        "argv": BASE_50CM + ["--tank-length", "4", "--fluid-res", "512",
                             "--advection", "semi_lagrangian", "--start-x", "-1.4"],
        "frames": 900,
        "camera": {"pos": (0.0, -2.9, 2.1), "pitch": -33.0, "yaw": 90.0},
    },
}


def fluid_kinetic_energy(fluid):
    """Total fluid kinetic energy [J] over pure fluid faces (approximate)."""
    import numpy as np

    u = fluid.velocity_u.numpy()
    v = fluid.velocity_v.numpy()
    w = fluid.velocity_w.numpy()
    cell_vol = fluid.dx**3
    return float(0.5 * fluid.density * cell_vol * ((u**2).sum() + (v**2).sum() + (w**2).sum()))


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
    wake_energy = []  # (t, KE) samples
    t_step = 0.0
    screenshot_saved = False
    for frame in range(frames):
        t0 = time.perf_counter()
        example.step()
        t_step += time.perf_counter() - t0
        example.render()
        if frame % 2 == 0:
            writer.append_data(viewer.get_frame().numpy())
        if frame % 10 == 9:
            wake_energy.append([example.sim_time, fluid_kinetic_energy(example.fluid)])
        if not screenshot_saved and frame == int(0.5 * frames):
            img = viewer.get_frame().numpy()
            h, w = img.shape[:2]
            side = min(h, w)
            crop = img[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
            Image.fromarray(crop).resize((320, 320)).save(os.path.join(out_dir, "media", f"{case_name}.jpg"))
            screenshot_saved = True
    writer.close()
    example.metrics.save()

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
        "advection": example.metrics.meta["advection"],
        "step_ms_with_diagnostics": 1000.0 * t_step / frames,
        "sim_only_ms": sim_only_ms,
        "realtime_factor_sim_only": (1000.0 / 60.0) / sim_only_ms,
        "wake_energy": wake_energy,
    }
    with open(os.path.join(out_dir, "data", f"{case_name}_perf.json"), "w") as f:
        json.dump(perf, f, indent=1)
    print(json.dumps({k: v for k, v in perf.items() if k != "wake_energy"}, indent=1))
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
