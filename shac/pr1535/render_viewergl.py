#!/usr/bin/env python3
"""Render a checkpoint rollout with Newton ViewerGL.

The policy and simulation semantics come from ``train_shac.py``.  A single,
non-noisy lane is advanced through ``MJWarpTorchBridge._forward_raw`` and is
frozen at its first terminal state, matching the evaluation harness.  The
recorded MJWarp qpos values are converted to body poses with native MuJoCo
forward kinematics, then displayed by a CPU Newton model in headless ViewerGL.
Native MuJoCo is never used to advance dynamics.

The command writes three sibling files: an H.264/yuv420p fast-start MP4, a
JPEG poster, and a JSON provenance/behavior manifest.

Example:

  PYGLET_HEADLESS=1 uv run --with imageio-ffmpeg==0.6.0 \
    python /path/to/render_viewergl.py \
    --checkpoint /path/to/ant_seed17.pt --policy best \
    --output /tmp/ant_seed17_best.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Pyglet selects its display backend on first import.  This must precede the
# Newton/ViewerGL import on machines without X11 or Wayland.
os.environ.setdefault("PYGLET_HEADLESS", "1")

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import torch
import warp as wp
from PIL import Image

import train_shac as harness

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PR_ROOT = Path("/home/horde/repos/mujoco_warp-pr1535")
DEFAULT_NEWTON_ROOT = Path("/home/horde/repos/newton-shac-pr1535")
SUPPORTED_CHECKPOINT_FORMATS = {
    "mjwarp-pr1535-shac-style-v1",
    "mjwarp-pr1535-shac-style-v2",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver() -> str | None:
    try:
        return (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.splitlines()[0]
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _as_state_dict(value: object) -> Mapping[str, torch.Tensor] | None:
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, Mapping) or not value:
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        return None
    return value


def _saved_policy_state(
    checkpoint: Mapping[str, Any], policy: str
) -> tuple[Mapping[str, torch.Tensor] | None, str | None]:
    aliases = {
        "initial": ("initial_actor", "actor_initial", "initial"),
        "best": ("best_actor", "actor_best", "best"),
        "final": ("final_actor", "actor_final", "final", "actor"),
    }[policy]

    for key in aliases:
        state = _as_state_dict(checkpoint.get(key))
        if state is not None:
            return state, f"checkpoint.{key}"

    for container_name in ("actors", "policies"):
        container = checkpoint.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key in aliases:
            state = _as_state_dict(container.get(key))
            if state is not None:
                return state, f"checkpoint.{container_name}.{key}"
    return None, None


def _saved_normalizer_state(
    checkpoint: Mapping[str, Any], policy: str
) -> tuple[Mapping[str, torch.Tensor] | None, str | None]:
    aliases = {
        "initial": ("initial_normalizer", "normalizer_initial"),
        "best": ("best_normalizer", "normalizer_best"),
        "final": ("final_normalizer", "normalizer_final", "normalizer"),
    }[policy]
    for key in aliases:
        state = _as_state_dict(checkpoint.get(key))
        if state is not None:
            return state, f"checkpoint.{key}"
    return None, None


def _policy_epoch(checkpoint: Mapping[str, Any], policy: str) -> int:
    if policy == "initial":
        return 0
    if policy == "best":
        return int(checkpoint.get("best_actor_epoch", 0))
    return int(checkpoint.get("epoch", 0))


def _checkpoint_seed(checkpoint: Mapping[str, Any]) -> int:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping) or "seed" not in config:
        raise ValueError("Checkpoint does not contain config.seed")
    return int(config["seed"])


def _checkpoint_eval_steps(checkpoint: Mapping[str, Any]) -> int:
    config = checkpoint.get("config")
    if isinstance(config, Mapping) and config.get("eval_steps") is not None:
        return int(config["eval_steps"])
    return 500


def _checkpoint_config(checkpoint: Mapping[str, Any], key: str, default: Any) -> Any:
    config = checkpoint.get("config")
    if isinstance(config, Mapping) and config.get(key) is not None:
        return config[key]
    return default


def _v2_harness():
    try:
        import train_shac_v2
    except ImportError as exc:
        raise RuntimeError(
            "A v2 checkpoint requires train_shac_v2.py beside this renderer"
        ) from exc
    return train_shac_v2


def _resolve_xml(task: str, explicit: Path | None, pr_root: Path) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
    elif task == "ant":
        path = harness.ANT_XML.resolve()
    else:
        path = (pr_root / harness.HUMANOID_XML_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint root must be a dictionary")
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint_format!r}; expected one of "
            f"{sorted(SUPPORTED_CHECKPOINT_FORMATS)}"
        )
    return checkpoint, str(checkpoint_format)


def _select_policy(
    checkpoint: Mapping[str, Any],
    checkpoint_format: str,
    policy: str,
    loaded: harness.LoadedModel,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module | None, dict[str, Any]]:
    if checkpoint_format.endswith("-v2"):
        v2 = _v2_harness()
        actor, normalizer = v2._checkpoint_networks(checkpoint, loaded, device)
    else:
        hidden = int(checkpoint["hidden"])
        actor, _, _ = harness.make_networks(loaded, hidden, device)
        normalizer = None
    reconstructed_initial = {
        name: tensor.detach().clone() for name, tensor in actor.state_dict().items()
    }
    state, source = _saved_policy_state(checkpoint, policy)
    reconstructed = False
    if state is None:
        if policy != "initial":
            raise KeyError(f"Checkpoint has no saved {policy!r} actor")
        state = reconstructed_initial
        source = "deterministic reconstruction from config.seed and current harness"
        reconstructed = True

    actor.load_state_dict(state, strict=True)
    actor.eval()

    normalizer_source: str | None = None
    normalizer_sha256: str | None = None
    if normalizer is not None:
        normalizer_state, normalizer_source = _saved_normalizer_state(
            checkpoint, policy
        )
        if normalizer_state is None:
            if policy != "initial":
                raise KeyError(f"Checkpoint has no saved {policy!r} normalizer")
            normalizer_source = "deterministic initial normalizer reconstruction"
        else:
            normalizer.load_state_dict(normalizer_state, strict=True)
        normalizer.eval()
        normalizer_sha256 = _sha256_state_dict(normalizer.state_dict())

    epoch_zero_validation: bool | None = None
    if reconstructed and int(checkpoint.get("best_actor_epoch", -1)) == 0:
        best_state, _ = _saved_policy_state(checkpoint, "best")
        if best_state is not None:
            epoch_zero_validation = all(
                torch.equal(
                    actor.state_dict()[name].detach().cpu(),
                    best_state[name].detach().cpu(),
                )
                for name in actor.state_dict()
            )
            if not epoch_zero_validation:
                raise RuntimeError(
                    "Reconstructed initial actor does not match saved epoch-zero best actor"
                )

    return (
        actor,
        normalizer,
        {
            "requested": policy,
            "checkpoint_source": source,
            "reconstructed": reconstructed,
            "action_selection": (
                "deterministic mean action"
                if checkpoint_format.endswith("-v2")
                else "deterministic actor"
            ),
            "epoch": _policy_epoch(checkpoint, policy),
            "state_dict_sha256": _sha256_state_dict(actor.state_dict()),
            "normalizer_source": normalizer_source,
            "normalizer_state_dict_sha256": normalizer_sha256,
            "epoch_zero_saved_best_exact_match": epoch_zero_validation,
        },
    )


def _terminal_reasons(
    loaded: harness.LoadedModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    *,
    checkpoint_format: str,
    reward_profile: str,
) -> list[str]:
    reasons = []
    if not bool(torch.isfinite(qpos).all() and torch.isfinite(qvel).all()):
        reasons.append("non_finite_state")
        return reasons
    height = float(qpos[0, 2].item())
    if checkpoint_format.endswith("-v2") and reward_profile != "legacy":
        minimum_height = 0.27 if loaded.spec.name == "ant" else 0.74
        if height < minimum_height:
            reasons.append("root_height_below_v2_minimum")
    else:
        if not loaded.spec.healthy_z[0] < height < loaded.spec.healthy_z[1]:
            reasons.append("root_height_outside_healthy_range")
        if float(harness.root_up(qpos)[0].item()) <= loaded.spec.upright_min:
            reasons.append("root_orientation_below_upright_threshold")
    return reasons or ["unknown_health_predicate"]


def _policy_action(
    checkpoint_format: str,
    actor: torch.nn.Module,
    normalizer: torch.nn.Module | None,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
) -> torch.Tensor:
    if checkpoint_format.endswith("-v2"):
        if normalizer is None:
            raise RuntimeError("v2 policy is missing its observation normalizer")
        v2 = _v2_harness()
        return actor(normalizer(v2.raw_observation(qpos, qvel)))
    return actor(harness.observation(qpos, qvel))


def _healthy(
    checkpoint_format: str,
    loaded: harness.LoadedModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    reward_profile: str,
) -> torch.Tensor:
    if checkpoint_format.endswith("-v2"):
        return _v2_harness().healthy(loaded, qpos, qvel, reward_profile)
    return harness.healthy(loaded.spec, qpos, qvel)


def _step_reward(
    checkpoint_format: str,
    loaded: harness.LoadedModel,
    reward_profile: str,
    control_dt: float,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    qpos_next: torch.Tensor,
    qvel_next: torch.Tensor,
) -> torch.Tensor:
    if checkpoint_format.endswith("-v2"):
        v2 = _v2_harness()
        components = v2.reward_components(
            loaded,
            reward_profile,
            control_dt,
            qpos,
            qvel,
            action,
            qpos_next,
            qvel_next,
        )
        return v2._reward_total(components)
    return harness.reward(
        loaded.spec,
        control_dt,
        qpos,
        qvel,
        action,
        qpos_next,
        qvel_next,
    )


@torch.no_grad()
def _simulate(
    loaded: harness.LoadedModel,
    bridge: Any,
    actor: torch.nn.Module,
    normalizer: torch.nn.Module | None,
    *,
    checkpoint_format: str,
    steps: int,
    capture_stride: int,
    action_repeat: int,
    reward_profile: str,
    gamma: float,
) -> tuple[list[np.ndarray], list[int], dict[str, Any]]:
    # Both harnesses reduce to the same fixed qpos and zero qvel when noise is
    # disabled.  Use v1's helper so this path is valid for both formats.
    qpos, qvel = harness.sample_initial_states(
        loaded,
        1,
        np.random.default_rng(0),
        bridge.torch_device,
        noisy=False,
    )
    if not bool(
        _healthy(checkpoint_format, loaded, qpos, qvel, reward_profile)[0].item()
    ):
        raise RuntimeError("The task's fixed non-noisy reset is not healthy")

    initial_qpos = qpos.detach().cpu().numpy()[0].copy()
    frame_qpos = [initial_qpos.copy()]
    capture_steps = [0]
    reward_sum = 0.0
    discounted_reward_sum = 0.0
    discount = 1.0
    alive = True
    alive_after_sum = 0
    terminal_control_step: int | None = None
    terminal_raw_step: int | None = None
    terminal_reasons: list[str] = []
    action_abs_sum = 0.0
    action_square_mean_sum = 0.0
    action_element_count = 0
    action_abs_max = 0.0
    active_contact_counts: list[int] = []
    observed_heights = [float(qpos[0, 2].item())]
    raw_step = 0
    total_raw_steps = steps * action_repeat
    timestep = float(loaded.model.opt.timestep)
    control_dt = timestep * action_repeat

    for control_step in range(1, steps + 1):
        if alive:
            action = _policy_action(checkpoint_format, actor, normalizer, qpos, qvel)
            qpos_before, qvel_before = qpos, qvel
            qpos_candidate, qvel_candidate = qpos, qvel
            for _ in range(action_repeat):
                qpos_candidate, qvel_candidate = bridge._forward_raw(
                    qpos_candidate, qvel_candidate, action
                )
                raw_step += 1
                raw_finite = bool(
                    torch.isfinite(qpos_candidate).all()
                    and torch.isfinite(qvel_candidate).all()
                )
                if raw_finite:
                    observed_heights.append(float(qpos_candidate[0, 2].item()))
                    capture_qpos = qpos_candidate
                else:
                    capture_qpos = qpos_before
                active_contact_counts.append(
                    int(bridge.data_out.nacon.numpy().reshape(-1)[0])
                )
                if raw_step % capture_stride == 0 or raw_step == total_raw_steps:
                    frame_qpos.append(capture_qpos.detach().cpu().numpy()[0].copy())
                    capture_steps.append(raw_step)

            finite = bool(
                torch.isfinite(qpos_candidate).all()
                and torch.isfinite(qvel_candidate).all()
            )
            step_reward = _step_reward(
                checkpoint_format,
                loaded,
                reward_profile,
                control_dt,
                qpos_before,
                qvel_before,
                action,
                qpos_candidate,
                qvel_candidate,
            )
            step_reward = torch.nan_to_num(step_reward, nan=0.0, posinf=0.0, neginf=0.0)
            reward_value = float(step_reward[0].item())
            reward_sum += reward_value
            discounted_reward_sum += discount * reward_value

            action_abs = action.abs()
            action_abs_sum += float(action_abs.sum().item())
            action_square_mean_sum += float(action.square().mean().item())
            action_element_count += action.numel()
            action_abs_max = max(action_abs_max, float(action_abs.max().item()))

            if finite:
                qpos, qvel = qpos_candidate, qvel_candidate
                next_alive = bool(
                    _healthy(checkpoint_format, loaded, qpos, qvel, reward_profile)[
                        0
                    ].item()
                )
            else:
                next_alive = False

            if not next_alive:
                terminal_control_step = control_step
                terminal_raw_step = raw_step
                terminal_reasons = _terminal_reasons(
                    loaded,
                    qpos_candidate,
                    qvel_candidate,
                    checkpoint_format=checkpoint_format,
                    reward_profile=reward_profile,
                )
            alive = next_alive
        else:
            for _ in range(action_repeat):
                raw_step += 1
                if raw_step % capture_stride == 0 or raw_step == total_raw_steps:
                    frame_qpos.append(qpos.detach().cpu().numpy()[0].copy())
                    capture_steps.append(raw_step)

        alive_after_sum += int(alive)
        discount *= gamma

    final_qpos = qpos.detach().cpu().numpy()[0].copy()
    active_control_steps = terminal_control_step or steps
    active_raw_steps = terminal_raw_step or total_raw_steps
    contact_array = np.asarray(active_contact_counts, dtype=np.float64)
    metrics = {
        "return": reward_sum,
        "discounted_return": discounted_reward_sum,
        "displacement_x": float(final_qpos[0] - initial_qpos[0]),
        "mean_forward_velocity_over_active_time": float(
            (final_qpos[0] - initial_qpos[0]) / (active_raw_steps * timestep)
        ),
        "final_alive": alive,
        "mean_alive_fraction": alive_after_sum / steps,
        "terminal_step": terminal_control_step,
        "terminal_control_step": terminal_control_step,
        "terminal_raw_physics_step": terminal_raw_step,
        "terminal_time_seconds": (
            terminal_raw_step * timestep if terminal_raw_step is not None else None
        ),
        "terminal_reasons": terminal_reasons,
        "final_root_height": float(final_qpos[2]),
        "minimum_root_height": min(observed_heights),
        "maximum_root_height": max(observed_heights),
        "mean_absolute_action": (
            action_abs_sum / action_element_count if action_element_count else 0.0
        ),
        "mean_action_rms_over_requested_steps": float(
            np.sqrt(action_square_mean_sum / steps)
        ),
        "maximum_absolute_action": action_abs_max,
        "active_contact_count_mean": (
            float(contact_array.mean()) if contact_array.size else None
        ),
        "active_contact_count_min": (
            int(contact_array.min()) if contact_array.size else None
        ),
        "active_contact_count_max": (
            int(contact_array.max()) if contact_array.size else None
        ),
        "control_steps_requested": steps,
        "control_steps_actively_simulated": active_control_steps,
        "control_steps_frozen_after_terminal": steps - active_control_steps,
        "action_repeat": action_repeat,
        "raw_physics_steps_requested": total_raw_steps,
        "raw_physics_steps_actively_simulated": active_raw_steps,
        "raw_physics_steps_frozen_after_terminal": (total_raw_steps - active_raw_steps),
        "initial_qpos_sha256": _sha256_array(initial_qpos),
        "final_qpos_sha256": _sha256_array(final_qpos),
        "rendered_qpos_trajectory_sha256": _sha256_array(np.stack(frame_qpos)),
    }
    return frame_qpos, capture_steps, metrics


def _assert_body_alignment(
    render_model: newton.Model, mj_model: mujoco.MjModel
) -> None:
    if render_model.body_count != mj_model.nbody - 1:
        raise RuntimeError(
            "Newton/MuJoCo body count mismatch: "
            f"{render_model.body_count} != {mj_model.nbody - 1}"
        )
    for newton_index, label in enumerate(render_model.body_label):
        mj_name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, newton_index + 1
        )
        if mj_name is not None and label.rsplit("/", 1)[-1] != mj_name:
            raise RuntimeError(
                f"Body order mismatch at {newton_index}: {label!r} vs {mj_name!r}"
            )


def _body_xforms_from_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    xforms = np.empty((model.nbody - 1, 7), dtype=np.float32)
    xforms[:, :3] = data.xpos[1:]
    # MuJoCo is wxyz; a Warp transform stores xyzw.
    xforms[:, 3:6] = data.xquat[1:, 1:]
    xforms[:, 6] = data.xquat[1:, 0]
    return xforms


def _default_camera_offset(task: str) -> tuple[float, float, float]:
    if task == "ant":
        return (-2.8, -4.2, 1.8)
    return (-2.8, -4.2, 1.6)


def _default_camera_target_height(task: str, target_height: float) -> float:
    # Humanoid's root is near its upper torso; aim lower to center the full body.
    return 0.9 if task == "humanoid" else target_height


def _default_camera_fov(task: str) -> float:
    return 38.0 if task == "humanoid" else 42.0


def _render(
    loaded: harness.LoadedModel,
    frame_qpos: list[np.ndarray],
    capture_steps: list[int],
    *,
    output_path: Path,
    poster_path: Path,
    width: int,
    height: int,
    fps: float,
    camera_mode: str,
    camera_offset: tuple[float, float, float],
    camera_target_height: float,
    camera_fov: float,
    draw_edges: bool,
    draw_shadows: bool,
    terminal_step: int | None,
) -> dict[str, Any]:
    try:
        import imageio_ffmpeg as ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required; run with "
            "`uv run --with imageio-ffmpeg==0.6.0 ...`"
        ) from exc

    builder = newton.ModelBuilder()
    builder.add_mjcf(str(loaded.xml_path), ctrl_direct=True)
    render_model = builder.finalize(device="cpu")
    _assert_body_alignment(render_model, loaded.model)
    render_state = render_model.state()
    fk_data = mujoco.MjData(loaded.model)

    viewer = newton.viewer.ViewerGL(
        width=width,
        height=height,
        headless=True,
        enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE,
    )
    writer = None
    poster_rgb: np.ndarray | None = None
    initial_root_xy: np.ndarray | None = None
    if terminal_step is None:
        active_last_index = len(frame_qpos) - 1
    else:
        active_last_index = next(
            (
                index
                for index, captured_step in enumerate(capture_steps)
                if captured_step >= terminal_step
            ),
            len(frame_qpos) - 1,
        )
    poster_index = active_last_index // 2

    try:
        viewer.set_model(render_model)
        viewer.renderer.draw_fps = False
        viewer.renderer.draw_edges = draw_edges
        viewer.renderer.draw_shadows = draw_shadows
        viewer.camera.fov = camera_fov

        writer = ffmpeg.write_frames(
            str(output_path),
            size=(width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=fps,
            quality=8,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="warning",
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        frame_buffer = None

        for frame_index, (qpos, physics_step) in enumerate(
            zip(frame_qpos, capture_steps, strict=True)
        ):
            xforms = _body_xforms_from_qpos(loaded.model, fk_data, qpos)
            render_state.body_q.assign(xforms)
            root_xy = xforms[0, :2].astype(np.float64)
            if initial_root_xy is None:
                initial_root_xy = root_xy.copy()
            focus_xy = root_xy if camera_mode == "track" else initial_root_xy
            target = np.array(
                [focus_xy[0], focus_xy[1], camera_target_height],
                dtype=np.float64,
            )
            position = target + np.asarray(camera_offset)
            viewer.set_camera(wp.vec3(*position), pitch=0.0, yaw=0.0)
            viewer.camera.look_at(tuple(target))

            sim_time = physics_step * float(loaded.model.opt.timestep)
            viewer.begin_frame(sim_time)
            viewer.log_state(render_state)
            viewer.end_frame()
            frame_buffer = viewer.get_frame(target_image=frame_buffer)
            frame_rgb = np.ascontiguousarray(frame_buffer.numpy())
            writer.send(frame_rgb)
            if frame_index == poster_index:
                poster_rgb = frame_rgb.copy()
    finally:
        if writer is not None:
            writer.close()
        viewer.close()

    if poster_rgb is None:
        raise RuntimeError("No frame was selected for the poster")
    Image.fromarray(poster_rgb).save(
        poster_path, format="JPEG", quality=91, optimize=True, progressive=True
    )
    return {
        "viewer": "newton.viewer.ViewerGL",
        "headless": True,
        "render_device": "cpu",
        "cuda_interop": "NONE",
        "width": width,
        "height": height,
        "fps": fps,
        "encoded_frame_count": len(frame_qpos),
        "poster_frame_index": poster_index,
        "poster_physics_step": capture_steps[poster_index],
        "camera": {
            "mode": camera_mode,
            "offset_xyz": list(camera_offset),
            "target_height": camera_target_height,
            "vertical_fov_degrees": camera_fov,
        },
        "draw_edges": draw_edges,
        "draw_shadows": draw_shadows,
        "newton_body_count": render_model.body_count,
        "newton_shape_count": render_model.shape_count,
        "body_pose_source": (
            "native MuJoCo mj_forward FK from recorded MJWarp qpos; "
            "native MuJoCo dynamics are not used"
        ),
        "encoder": {
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "faststart": True,
            "imageio_ffmpeg": getattr(ffmpeg, "__version__", "unknown"),
            "ffmpeg": ffmpeg.get_ffmpeg_version(),
        },
    }


def _output_paths(args: argparse.Namespace, task: str) -> tuple[Path, Path, Path]:
    if args.output is None:
        output = (
            SCRIPT_DIR
            / "results"
            / "videos"
            / f"{args.checkpoint.stem}_{task}_{args.policy}.mp4"
        )
    else:
        output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    return output, output.with_suffix(".jpg"), output.with_suffix(".json")


def _check_outputs(paths: tuple[Path, ...], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists (pass --overwrite): {joined}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="v1 or v2 training checkpoint"
    )
    parser.add_argument(
        "--policy",
        choices=("initial", "best", "final"),
        default="best",
        help="saved policy to render; a missing v1 initial policy is reconstructed",
    )
    parser.add_argument(
        "--output", type=Path, help="MP4 path; sibling .jpg and .json are also written"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--pr-root", type=Path, default=DEFAULT_PR_ROOT)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="policy/control transitions; defaults to checkpoint config.eval_steps",
    )
    parser.add_argument(
        "--capture-stride",
        type=int,
        help="raw MJWarp physics steps per encoded frame; inferred from --fps by default",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--camera-mode", choices=("fixed", "track"), default="fixed")
    parser.add_argument("--camera-offset", type=float, nargs=3)
    parser.add_argument("--camera-target-height", type=float)
    parser.add_argument("--camera-fov", type=float)
    parser.add_argument(
        "--draw-edges", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--draw-shadows", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--allow-other-mjw", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.capture_stride is not None and args.capture_stride < 1:
        parser.error("--capture-stride must be positive")
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even integers")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.camera_fov is not None and not 1.0 <= args.camera_fov < 179.0:
        parser.error("--camera-fov must be in [1, 179)")
    return args


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.pr_root = args.pr_root.expanduser().resolve()
    args.newton_root = args.newton_root.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    imported_mjw_root = Path(mjw.__file__).resolve().parent.parent
    if not args.allow_other_mjw and imported_mjw_root != args.pr_root:
        raise RuntimeError(
            f"Imported mujoco_warp from {imported_mjw_root}, expected {args.pr_root}"
        )

    device = torch.device(args.device)
    checkpoint, checkpoint_format = _load_checkpoint(args.checkpoint, device)
    task = str(checkpoint.get("task"))
    if task not in harness.TASKS:
        raise ValueError(f"Unsupported checkpoint task {task!r}")
    xml_path = _resolve_xml(task, args.xml, args.pr_root)
    output_path, poster_path, manifest_path = _output_paths(args, task)
    _check_outputs((output_path, poster_path, manifest_path), args.overwrite)

    seed = _checkpoint_seed(checkpoint)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    mjw.enable_grad()

    loaded = harness.load_model(harness.TASKS[task], xml_path)
    bridge_args = SimpleNamespace(
        worlds=1,
        device=args.device,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    bridge = harness.make_bridge(loaded, bridge_args)
    actor, normalizer, policy_info = _select_policy(
        checkpoint, checkpoint_format, args.policy, loaded, device
    )

    steps = args.steps or _checkpoint_eval_steps(checkpoint)
    timestep = float(loaded.model.opt.timestep)
    is_v2 = checkpoint_format.endswith("-v2")
    action_repeat = int(
        _checkpoint_config(
            checkpoint,
            "action_repeat",
            (2 if task == "ant" else 3) if is_v2 else 1,
        )
    )
    reward_profile = str(
        _checkpoint_config(
            checkpoint, "reward_profile", "diffrl" if is_v2 else "legacy"
        )
    )
    gamma = float(_checkpoint_config(checkpoint, "gamma", 0.99))
    capture_stride = args.capture_stride or max(1, round(1.0 / (args.fps * timestep)))
    frame_qpos, capture_steps, behavior = _simulate(
        loaded,
        bridge,
        actor,
        normalizer,
        checkpoint_format=checkpoint_format,
        steps=steps,
        capture_stride=capture_stride,
        action_repeat=action_repeat,
        reward_profile=reward_profile,
        gamma=gamma,
    )

    camera_offset = (
        tuple(args.camera_offset)
        if args.camera_offset is not None
        else _default_camera_offset(task)
    )
    camera_target_height = (
        float(args.camera_target_height)
        if args.camera_target_height is not None
        else _default_camera_target_height(task, loaded.spec.target_height)
    )
    camera_fov = (
        float(args.camera_fov)
        if args.camera_fov is not None
        else _default_camera_fov(task)
    )
    render_info = _render(
        loaded,
        frame_qpos,
        capture_steps,
        output_path=output_path,
        poster_path=poster_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_mode=args.camera_mode,
        camera_offset=camera_offset,
        camera_target_height=camera_target_height,
        camera_fov=camera_fov,
        draw_edges=args.draw_edges,
        draw_shadows=args.draw_shadows,
        terminal_step=behavior["terminal_raw_physics_step"],
    )

    script_path = Path(__file__).resolve()
    base_harness_path = Path(harness.__file__).resolve()
    harness_path = (
        Path(_v2_harness().__file__).resolve() if is_v2 else base_harness_path
    )
    bridge_path = base_harness_path.with_name("mjwarp_torch_bridge.py")
    actual_capture_fps = 1.0 / (capture_stride * timestep)
    manifest = {
        "schema": "mjwarp-pr1535-viewergl-render-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "file": str(args.checkpoint),
            "sha256": _sha256_file(args.checkpoint),
            "format": checkpoint_format,
            "training_seed": seed,
            "pr_head_recorded": checkpoint.get("pr_head"),
        },
        "policy": policy_info,
        "task": task,
        "simulation": {
            "engine": "MJWarp PR #1535 through MJWarpTorchBridge._forward_raw",
            "worlds": 1,
            "reset": "fixed task initial qpos and zero qvel; noise disabled",
            "freeze_at_first_terminal": True,
            "control_steps_requested": steps,
            "action_repeat": action_repeat,
            "reward_profile": reward_profile,
            "gamma": gamma,
            "timestep_seconds": timestep,
            "control_timestep_seconds": timestep * action_repeat,
            "capture_stride_physics_steps": capture_stride,
            "requested_video_fps": args.fps,
            "physics_time_capture_fps": actual_capture_fps,
            "fps_matches_physics_time": bool(
                np.isclose(actual_capture_fps, args.fps, rtol=0.0, atol=1.0e-12)
            ),
            "capture_steps": capture_steps,
        },
        "behavior": behavior,
        "render": render_info,
        "outputs": {
            "video": {
                "file": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": _sha256_file(output_path),
                "mime_type": "video/mp4",
            },
            "poster": {
                "file": poster_path.name,
                "bytes": poster_path.stat().st_size,
                "sha256": _sha256_file(poster_path),
                "mime_type": "image/jpeg",
            },
        },
        "provenance": {
            "pr": {
                "url": "https://github.com/google-deepmind/mujoco_warp/pull/1535",
                "worktree": str(args.pr_root),
                "head": _git_head(args.pr_root),
                "import_path": str(Path(mjw.__file__).resolve()),
                "exact_worktree_import": imported_mjw_root == args.pr_root,
            },
            "newton": {
                "worktree": str(args.newton_root),
                "head": _git_head(args.newton_root),
                "import_path": str(Path(newton.__file__).resolve()),
            },
            "model_xml": {
                "file": str(xml_path),
                "sha256": _sha256_file(xml_path),
            },
            "renderer_script": {
                "file": str(script_path),
                "sha256": _sha256_file(script_path),
            },
            "training_harness": {
                "file": str(harness_path),
                "sha256": _sha256_file(harness_path),
            },
            "base_v1_harness": {
                "file": str(base_harness_path),
                "sha256": _sha256_file(base_harness_path),
            },
            "torch_bridge": {
                "file": str(bridge_path),
                "sha256": _sha256_file(bridge_path),
            },
            "versions": {
                "python": platform.python_version(),
                "newton": getattr(newton, "__version__", "unknown"),
                "mujoco_warp": getattr(mjw, "__version__", "unknown"),
                "mujoco": mujoco.__version__,
                "warp": wp.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pillow": _distribution_version("pillow"),
                "pyglet": _distribution_version("pyglet"),
                "imgui_bundle": _distribution_version("imgui-bundle"),
            },
            "device": {
                "simulation": args.device,
                "render": "cpu",
                "name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "nvidia_driver": _nvidia_driver(),
            },
            "argv": sys.argv,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote {output_path}")
    print(f"Wrote {poster_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
