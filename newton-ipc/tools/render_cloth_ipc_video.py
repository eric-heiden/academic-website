"""Render the SolverIPC cloth example with Newton's headless ViewerGL."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path

import newton
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readme-image", type=Path)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    module = importlib.import_module("newton.examples.cloth.example_cloth_ipc")
    example_args = module.Example.create_parser().parse_args(
        [
            "--viewer",
            "gl",
            "--headless",
            "--num-frames",
            "100000",
            "--resolution",
            str(args.resolution),
        ]
    )
    viewer = newton.viewer.ViewerGL(
        headless=True,
        width=args.width,
        height=args.height,
        num_frames=100000,
    )
    example = module.Example(viewer, example_args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    poster_path = args.output.with_name(f"{args.output.stem}_poster.jpg")
    metadata_path = args.output.with_suffix(".json")
    output_frames = round(args.duration * args.video_fps)
    simulation_frames_per_video_frame = max(round(example.fps / args.video_fps), 1)
    poster_index = min(round(0.85 * args.video_fps), output_frames - 1)
    ffmpeg = subprocess.Popen(
        [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{args.width}x{args.height}",
            "-r",
            str(args.video_fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )

    started = time.perf_counter()
    poster: np.ndarray | None = None
    try:
        assert ffmpeg.stdin is not None
        for frame_index in range(output_frames):
            for _ in range(simulation_frames_per_video_frame):
                example.step()
            example.render()
            frame = np.ascontiguousarray(viewer.get_frame().numpy())
            if frame_index == poster_index:
                poster = frame.copy()
            ffmpeg.stdin.write(frame.tobytes())
            if frame_index % args.video_fps == 0:
                print(
                    f"{frame_index / args.video_fps:.0f}/{args.duration:.0f} s",
                    flush=True,
                )
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        viewer.close()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    if poster is None:
        raise RuntimeError("no poster frame was rendered")

    poster_image = Image.fromarray(poster)
    poster_image.save(poster_path, quality=92, optimize=True)
    if args.readme_image is not None:
        args.readme_image.parent.mkdir(parents=True, exist_ok=True)
        side = min(poster_image.size)
        left = (poster_image.width - side) // 2
        top = (poster_image.height - side) // 2
        poster_image.crop((left, top, left + side, top + side)).resize((320, 320)).save(
            args.readme_image,
            quality=92,
            optimize=True,
        )

    elapsed = time.perf_counter() - started
    metadata_path.write_text(
        json.dumps(
            {
                "duration_seconds": args.duration,
                "video_fps": args.video_fps,
                "resolution": [args.width, args.height],
                "cloth_resolution": args.resolution,
                "simulation_fps": example.fps,
                "simulation_frames_per_video_frame": simulation_frames_per_video_frame,
                "particle_count": example.model.particle_count,
                "triangle_count": example.model.tri_count,
                "newton_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(newton.__file__).resolve().parents[1],
                    text=True,
                ).strip(),
                "solver_status": example.solver.Status(
                    int(example.solver.diagnostics.status.numpy()[0])
                ).name,
                "failed_steps": int(example.solver.diagnostics.failed_steps.numpy()[0]),
                "render_wall_seconds": round(elapsed, 3),
                "end_to_end_fps": round(output_frames / elapsed, 3),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}, {poster_path}, and {metadata_path}")


if __name__ == "__main__":
    main()
