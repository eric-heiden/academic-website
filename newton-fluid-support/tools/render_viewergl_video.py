"""Render the fluid report videos directly from Newton's headless ViewerGL."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import newton


PRESETS = {
    "dam-break": (
        "newton.examples.fluid.example_fluid_sph_dam_break",
        [
            "--dim-x", "28", "--dim-y", "18", "--dim-z", "14",
            "--bounds-lower", "-0.8", "-0.55", "0.0",
            "--bounds-upper", "2.2", "0.55", "1.2",
            "--emit-lower", "-0.70", "-0.45", "0.06",
            "--initial-velocity", "1.8", "0.0", "0.0",
            "--camera-pos", "2.7", "-3.5", "1.7",
            "--camera-pitch", "-18", "--camera-yaw", "132",
            "--fluid-shadow-size", "1024",
        ],
    ),
    "interactive-tank": (
        "newton.examples.fluid.example_fluid_sph_interactive_tank",
        [],
    ),
    "wave-pool": (
        "newton.examples.fluid.example_fluid_sph_wave_pool",
        [],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    module_name, preset_args = PRESETS[args.preset]
    module = importlib.import_module(module_name)
    example_args = module.Example.create_parser().parse_args(
        ["--viewer", "gl", "--headless", "--num-frames", "100000", *preset_args]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    poster_path = args.output.with_name(f"{args.output.stem}_poster.jpg")
    metadata_path = args.output.with_suffix(".json")
    viewer = newton.viewer.ViewerGL(
        headless=True,
        width=args.width,
        height=args.height,
        num_frames=100000,
    )
    example = module.Example(viewer, example_args)
    if args.preset == "interactive-tank":
        # Exercise the merged ViewerGL mixed-opacity batching and weighted OIT
        # path while the same bodies continue to interact with the fluid.
        opacities = example.model.shape_opacity.numpy()
        opacities[2:] = (0.45, 0.65, 0.35, 0.60)
        example.model.shape_opacity.assign(opacities)

    output_frames = int(round(args.duration * args.video_fps))
    simulation_steps_per_frame = max(int(round(example.fps / args.video_fps)), 1)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{args.width}x{args.height}",
            "-r", str(args.video_fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )

    started = time.perf_counter()
    poster_frame: np.ndarray | None = None
    poster_index = output_frames // 3
    try:
        assert ffmpeg.stdin is not None
        for frame_index in range(output_frames):
            for _ in range(simulation_steps_per_frame):
                example.step()
            example.render()
            frame = np.ascontiguousarray(viewer.get_frame().numpy())
            if frame_index == poster_index:
                poster_frame = frame.copy()
            ffmpeg.stdin.write(frame.tobytes())
            if frame_index % args.video_fps == 0:
                print(f"{args.preset}: {frame_index / args.video_fps:.0f}/{args.duration:.0f} s", flush=True)
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        viewer.close()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    if poster_frame is None:
        raise RuntimeError("No frames were rendered")

    Image.fromarray(poster_frame).save(poster_path, quality=92, optimize=True)
    metadata_path.write_text(
        json.dumps(
            {
                "preset": args.preset,
                "duration_seconds": args.duration,
                "video_fps": args.video_fps,
                "resolution": [args.width, args.height],
                "simulation_fps": example.fps,
                "simulation_steps_per_video_frame": simulation_steps_per_frame,
                "render_wall_seconds": round(time.perf_counter() - started, 3),
                "particle_count": example.model.particle_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} and {poster_path}")


if __name__ == "__main__":
    main()
