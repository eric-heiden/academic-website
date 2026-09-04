#!/usr/bin/env python3
"""Render and measure the Newton robot workloads used by the public report.

Run from a current Newton checkout with its project environment, for example:

    xvfb-run -a uv run --with imageio --with imageio-ffmpeg \
      ../academic-website-reports/superdex-vs-newton/scripts/render_newton_robot_workloads.py all

The script intentionally reuses Newton's first-party examples. It writes one MP4,
one poster image, and one JSON trace per workload.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import warp as wp

import newton
import newton.examples
from newton.viewer import ViewerGL


REPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPORT_ROOT / "media"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def newton_revision() -> str:
    checkout = Path(newton.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_viewer(num_frames: int) -> ViewerGL:
    viewer = ViewerGL(width=960, height=540, headless=True, num_frames=num_frames, vsync=False)
    viewer.renderer.sky_upper = (0.91, 0.94, 0.98)
    viewer.renderer.sky_lower = (0.72, 0.80, 0.88)
    viewer.renderer._light_color = (1.0, 0.98, 0.94)
    viewer.renderer.draw_shadows = True
    return viewer


def frame_rgb(viewer: ViewerGL) -> np.ndarray:
    frame = viewer.get_frame(render_ui=False).numpy()
    return np.ascontiguousarray(frame[:, :, :3])


def writer_for(path: Path, fps: int):
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    )


def common_metadata(workload: str, duration_s: float, video_fps: int) -> dict[str, Any]:
    return {
        "schema": 1,
        "workload": workload,
        "captured_at_utc": utc_now(),
        "newton_revision": newton_revision(),
        "newton_version": getattr(newton, "__version__", "unknown"),
        "warp_version": getattr(wp, "__version__", "unknown"),
        "device": str(wp.get_device()),
        "duration_s": duration_s,
        "video_fps": video_fps,
    }


class CommandViewer(ViewerGL):
    """Viewer with a deterministic keyboard command schedule for robot_policy."""

    active_keys: set[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_keys = set()
        self.renderer.sky_upper = (0.91, 0.94, 0.98)
        self.renderer.sky_lower = (0.72, 0.80, 0.88)
        self.renderer._light_color = (1.0, 0.98, 0.94)
        self.renderer.draw_shadows = True

    def is_key_down(self, key: str) -> bool:
        return key in self.active_keys


def render_g1(output: Path) -> dict[str, Any]:
    from newton.examples.robot.example_robot_policy import Example

    control_hz = 50
    duration_s = 8.0
    steps = int(duration_s * control_hz)
    capture_every = 2
    video_fps = control_hz // capture_every
    viewer = CommandViewer(width=960, height=540, headless=True, num_frames=steps + 4, vsync=False)
    args = newton.examples.default_args(Example.create_parser())
    args.robot = "g1_29dof"
    args.physx = False
    args.test = True
    example = Example(viewer, args)
    viewer.camera.fov = 38.0
    trace: list[dict[str, Any]] = []
    poster: np.ndarray | None = None
    video_path = output / "newton-g1-locomotion.mp4"

    try:
        with writer_for(video_path, video_fps) as writer:
            for frame_idx in range(steps):
                t = frame_idx / control_hz
                if 0.75 <= t < 5.5:
                    viewer.active_keys = {"i"}
                    command = [1.0, 0.0, 0.0]
                elif 5.5 <= t < 7.25:
                    viewer.active_keys = {"i", "u"}
                    command = [1.0, 0.0, 1.0]
                else:
                    viewer.active_keys = set()
                    command = [0.0, 0.0, 0.0]

                example.step()
                q = example.state_0.joint_q.numpy()
                qd = example.state_0.joint_qd.numpy()
                root = q[:3].astype(float)
                viewer.set_camera(wp.vec3(root[0] + 2.20, root[1] + 1.95, 1.45), -12.0, -138.0)
                example.render()

                horizontal_speed = float(np.linalg.norm(qd[:2]))
                trace.append(
                    {
                        "t": round(t, 4),
                        "root_x": round(float(root[0]), 6),
                        "root_y": round(float(root[1]), 6),
                        "root_z": round(float(root[2]), 6),
                        "speed_xy": round(horizontal_speed, 6),
                        "command_forward": command[0],
                        "command_turn": command[2],
                    }
                )

                if frame_idx % capture_every == 0:
                    image = frame_rgb(viewer)
                    writer.append_data(image)
                    if poster is None and t >= 4.0:
                        poster = image.copy()

        if poster is None:
            poster = frame_rgb(viewer)
        imageio.imwrite(output / "newton-g1-locomotion.jpg", poster, quality=90)
        example.test_final()
    finally:
        viewer.close()

    displacement = math.hypot(trace[-1]["root_x"] - trace[0]["root_x"], trace[-1]["root_y"] - trace[0]["root_y"])
    result = common_metadata("Unitree G1 29-DoF policy locomotion", duration_s, video_fps)
    result.update(
        {
            "source_example": "newton.examples.robot.example_robot_policy",
            "robot": "Unitree G1 29-DoF",
            "solver": "SolverMuJoCo",
            "policy_runtime": "Warp-NN ONNX",
            "samples": trace,
            "summary": {
                "horizontal_displacement_m": round(displacement, 6),
                "peak_horizontal_speed_m_s": round(max(v["speed_xy"] for v in trace), 6),
                "minimum_root_height_m": round(min(v["root_z"] for v in trace), 6),
            },
        }
    )
    write_json(output.parent / "data" / "g1-locomotion.json", result)
    return result


def render_panda(output: Path) -> dict[str, Any]:
    from newton.examples.robot.example_robot_panda_hydro import Example

    sim_hz = 60
    duration_s = 10.0
    steps = int(duration_s * sim_hz)
    capture_every = 2
    video_fps = sim_hz // capture_every
    viewer = make_viewer(steps + 4)
    args = newton.examples.default_args(Example.create_parser())
    args.scene = "pen"
    args.world_count = 1
    args.test = True
    args.deterministic = True
    args.deterministic_solver = False
    example = Example(viewer, args)
    trace: list[dict[str, Any]] = []
    poster: np.ndarray | None = None
    video_path = output / "newton-panda-pick-place.mp4"

    try:
        with writer_for(video_path, video_fps) as writer:
            for frame_idx in range(steps):
                example.step()
                example.render()
                body_q = example.state_0.body_q.numpy()
                obj = body_q[example.object_body_local][:3].astype(float)
                ee = body_q[example.ee_index][:3].astype(float)
                trace.append(
                    {
                        "t": round((frame_idx + 1) / sim_hz, 4),
                        "object_x": round(float(obj[0]), 6),
                        "object_y": round(float(obj[1]), 6),
                        "object_z": round(float(obj[2]), 6),
                        "end_effector_z": round(float(ee[2]), 6),
                        "waypoint": int(example.current_waypoint),
                    }
                )
                if frame_idx % capture_every == 0:
                    image = frame_rgb(viewer)
                    writer.append_data(image)
                    if poster is None and frame_idx / sim_hz >= 5.1:
                        poster = image.copy()

        if poster is None:
            poster = frame_rgb(viewer)
        imageio.imwrite(output / "newton-panda-pick-place.jpg", poster, quality=90)
        example.test_final()
    finally:
        viewer.close()

    initial_z = trace[0]["object_z"]
    max_z = max(v["object_z"] for v in trace)
    result = common_metadata("Franka Panda parallel-gripper pick and place", duration_s, video_fps)
    result.update(
        {
            "source_example": "newton.examples.robot.example_robot_panda_hydro",
            "robot": "Franka Research 3 with parallel gripper",
            "solver": "SolverMuJoCo with SDF hydroelastic contact",
            "object": "pen",
            "samples": trace,
            "summary": {
                "object_lift_m": round(max_z - initial_z, 6),
                "peak_object_height_m": round(max_z, 6),
                "final_object_position_m": [trace[-1]["object_x"], trace[-1]["object_y"], trace[-1]["object_z"]],
                "final_waypoint": trace[-1]["waypoint"],
            },
        }
    )
    write_json(output.parent / "data" / "panda-pick-place.json", result)
    return result


def render_cloth(output: Path) -> dict[str, Any]:
    from newton.examples.cloth.example_cloth_h1 import Example

    sim_hz = 60
    steps = 601
    duration_s = (steps - 1) / sim_hz
    capture_every = 2
    video_fps = sim_hz // capture_every
    viewer = make_viewer(steps + 4)
    args = newton.examples.default_args()
    args.test = True
    example = Example(viewer, args)
    trace: list[dict[str, Any]] = []
    poster: np.ndarray | None = None
    video_path = output / "newton-h1-jacket.mp4"

    try:
        with writer_for(video_path, video_fps) as writer:
            for frame_idx in range(steps):
                example.step()
                example.render()
                particles = example.state.particle_q.numpy()
                bodies = example.state.body_q.numpy()
                z = particles[:, 2].astype(float)
                trace.append(
                    {
                        "t": round(frame_idx / sim_hz, 4),
                        "cloth_z_min": round(float(np.min(z)), 6),
                        "cloth_z_median": round(float(np.median(z)), 6),
                        "cloth_z_max": round(float(np.max(z)), 6),
                        "left_hand_z": round(float(bodies[16][2]), 6),
                        "right_hand_z": round(float(bodies[33][2]), 6),
                    }
                )
                if frame_idx % capture_every == 0:
                    image = frame_rgb(viewer)
                    writer.append_data(image)
                    if poster is None and frame_idx / sim_hz >= 4.0:
                        poster = image.copy()

        if poster is None:
            poster = frame_rgb(viewer)
        imageio.imwrite(output / "newton-h1-jacket.jpg", poster, quality=90)
        example.test_final()
    finally:
        viewer.close()

    span = [v["cloth_z_max"] - v["cloth_z_min"] for v in trace]
    result = common_metadata("Unitree H1 jacket interaction", duration_s, video_fps)
    result.update(
        {
            "source_example": "newton.examples.cloth.example_cloth_h1",
            "robot": "Unitree H1",
            "solver": "SolverStyle3D",
            "samples": trace,
            "summary": {
                "particle_count": int(example.model.particle_count),
                "cloth_vertical_span_min_m": round(min(span), 6),
                "cloth_vertical_span_max_m": round(max(span), 6),
                "peak_left_hand_height_m": round(max(v["left_hand_z"] for v in trace), 6),
            },
        }
    )
    write_json(output.parent / "data" / "h1-jacket.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=["g1", "panda", "cloth", "all"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output.parent / "data").mkdir(parents=True, exist_ok=True)

    selected = [args.workload] if args.workload != "all" else ["g1", "panda", "cloth"]
    runners = {"g1": render_g1, "panda": render_panda, "cloth": render_cloth}
    summary = {}
    for name in selected:
        print(f"[report] rendering {name}", flush=True)
        summary[name] = runners[name](output)["summary"]
        print(f"[report] completed {name}", flush=True)
    write_json(output.parent / "data" / "newton-workload-summary.json", summary)


if __name__ == "__main__":
    main()
