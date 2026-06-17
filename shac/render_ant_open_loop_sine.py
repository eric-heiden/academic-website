#!/usr/bin/env python3
"""Render the Ant open-loop sine probe with the shared report camera."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import torch
import warp as wp

import newton
from follow_camera import SmoothedFollowCamera
from run_newton_shac import (
    ANT_ISAACLAB_START_JOINT_Q,
    AntRewardWeights,
    NewtonMuJoCoTorchEnv,
    write_json,
)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def render(args: argparse.Namespace) -> dict:
    wp.init()
    result_path = args.result_json.resolve()
    result = load_json(result_path)
    out_dir = result_path.parent

    env = NewtonMuJoCoTorchEnv(
        env_name="ant",
        num_envs=1,
        device=args.device,
        dt=float(result.get("dt", 1.0 / 60.0)),
        sim_substeps=int(result.get("sim_substeps") or 16),
        force_scale=float(result.get("force_scale") or args.force_scale),
        contact_backend=result.get("contact_backend") or "mujoco",
        ant_contact_margin=float(result.get("ant_contact_margin") or 0.0),
        ant_start_height=result.get("ant_start_height", 0.5),
        ant_start_joint_q=result.get("ant_start_joint_q") or list(ANT_ISAACLAB_START_JOINT_Q),
        ant_termination_height=float(result.get("ant_termination_height") or 0.31),
        ant_observation_style=result.get("ant_observation_style") or "isaac",
        ant_reward_style=result.get("ant_reward_style") or "isaac",
        ant_action_order=result.get("ant_action_order") or "joint",
        ant_reward=AntRewardWeights(),
    )

    freq = float(result["sine_freq"])
    amp = torch.tensor(result["sine_amp"], dtype=torch.float32, device=env.torch_device).view(1, env.num_actions)
    bias = torch.tensor(result["sine_bias"], dtype=torch.float32, device=env.torch_device).view(1, env.num_actions)
    phase = torch.tensor(result["sine_phase"], dtype=torch.float32, device=env.torch_device).view(1, env.num_actions)

    horizon = int(args.horizon or result.get("horizon") or 480)
    video_path = out_dir / "ant_open_loop_sine.mp4"
    poster_path = out_dir / "ant_open_loop_sine_poster.png"
    metadata_path = out_dir / "ant_open_loop_sine_render_metadata.json"

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    viewer.show_static = True
    viewer.show_collision = False
    viewer.set_model(env.model)
    follow_camera = SmoothedFollowCamera("ant", env.dt)

    q, qd = env.reset(noise=0.0)
    frames = []
    with imageio.get_writer(video_path, fps=max(1, int(round(1.0 / env.dt))), codec="libx264", quality=8) as writer:
        with torch.no_grad():
            for frame_idx in range(horizon):
                state = env.make_viewer_state(q, qd)
                follow_camera.update(viewer, q, state=state, model=env.model)
                viewer.begin_frame(frame_idx * env.dt)
                viewer.log_state(state)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()
                frames.append(frame)
                writer.append_data(frame)

                t = frame_idx * env.dt
                action = torch.clamp(bias + amp * torch.sin(2.0 * math.pi * freq * t + phase), -1.0, 1.0)
                q, qd = env.step(q, qd, env.action_to_joint_f(action))
                invalid = env.invalid_state(q, qd)
                q, qd, _ = env.sanitize_state(q, qd, action, invalid, stochastic_init=False)
                q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
                qd = torch.nan_to_num(qd, nan=0.0, posinf=0.0, neginf=0.0)
    viewer.close()
    imageio.imwrite(poster_path, frames[len(frames) // 2])

    metadata = {
        "source_result": str(result_path),
        "horizon": horizon,
        "force_scale_used": env.force_scale,
        "video": video_path.name,
        "poster": poster_path.name,
        "camera": "SmoothedFollowCamera",
        "source": "ViewerGL.get_frame()",
        "overlays": False,
    }
    write_json(metadata_path, metadata)
    print(f"wrote {video_path}")
    print(f"wrote {poster_path}")
    print(f"wrote {metadata_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--force-scale", type=float, default=150.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
