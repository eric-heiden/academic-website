#!/usr/bin/env python3
"""Render a v3 gait checkpoint through Newton ViewerGL.

MJWarp PR #1535 advances the policy trajectory.  Native MuJoCo is used only
for forward kinematics on the recorded qpos samples, and Newton ViewerGL is
the sole rasterizer.  The MP4, JPEG poster, and JSON manifest are written as
sibling files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("PYGLET_HEADLESS", "1")

import mujoco_warp as mjw
import numpy as np
import render_viewergl as viewer
import torch
from conditioned_policy_v3 import conditioned_actor
import train_gaits_v3 as gait
import warp as wp

SUPPORTED_FORMATS = {
    "mjwarp-pr1535-full-gait-v3",
    "mjwarp-pr1535-shac-gait-v3",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def policy_state(
    checkpoint: dict[str, Any], policy: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    actor_key = f"{policy}_actor"
    normalizer_key = f"{policy}_normalizer"
    if actor_key not in checkpoint or normalizer_key not in checkpoint:
        raise KeyError(f"checkpoint has no complete {policy!r} policy")
    return checkpoint[actor_key], checkpoint[normalizer_key]


@torch.no_grad()
def record_rollout(
    loaded: gait.v1.LoadedModel,
    bridge: gait.v1.MJWarpTorchBridge,
    actor: gait.PPOActor,
    normalizer: gait.RunningMeanStd,
    config: SimpleNamespace,
    *,
    steps: int,
    capture_stride: int,
) -> tuple[list[np.ndarray], list[int], dict[str, Any], dict[str, Any]]:
    device = bridge.torch_device
    qpos = torch.as_tensor(
        loaded.initial_qpos, dtype=torch.float32, device=device
    ).repeat(1, 1)
    qvel = torch.zeros((1, loaded.model.nv), dtype=torch.float32, device=device)
    previous_action = torch.zeros(
        (1, loaded.model.nu), dtype=torch.float32, device=device
    )
    progress = torch.zeros(1, dtype=torch.long, device=device)
    initial_qpos = qpos.detach().cpu().numpy()[0].copy()
    frames = [initial_qpos.copy()]
    capture_steps = [0]
    action_square_sum = 0.0
    action_rate_square_sum = 0.0
    action_count = 0
    contact_counts: list[int] = []
    control_qpos = [qpos.detach().cpu().numpy()[0].copy()]
    control_qvel = [qvel.detach().cpu().numpy()[0].copy()]
    control_actions: list[np.ndarray] = []
    control_active: list[bool] = []
    alive = True
    terminal_control_step: int | None = None
    raw_step = 0

    for control_step in range(1, steps + 1):
        control_active.append(alive)
        if alive:
            observation = normalizer(
                gait.raw_observation(
                    qpos,
                    qvel,
                    previous_action,
                    progress,
                    int(config.phase_period),
                    int(getattr(config, "phase_warmup_steps", 0)),
                )
            )
            action = actor.mean_action(observation)
            action_square_sum += float(action.square().mean().item())
            action_rate_square_sum += float(
                (action - previous_action).square().mean().item()
            )
            action_count += 1
            last_finite_qpos = qpos
            for _ in range(int(config.action_repeat)):
                candidate_qpos, candidate_qvel = bridge._forward_raw(qpos, qvel, action)
                raw_step += 1
                finite = bool(
                    torch.isfinite(candidate_qpos).all()
                    and torch.isfinite(candidate_qvel).all()
                )
                if finite:
                    qpos, qvel = candidate_qpos, candidate_qvel
                    last_finite_qpos = qpos
                try:
                    contact_counts.append(
                        int(bridge.data_out.nacon.numpy().reshape(-1)[0])
                    )
                except AttributeError:
                    pass
                if raw_step % capture_stride == 0:
                    frames.append(last_finite_qpos.detach().cpu().numpy()[0].copy())
                    capture_steps.append(raw_step)
            alive = finite and bool(
                gait.healthy(
                    loaded.spec.name,
                    qpos,
                    qvel,
                    minimum_height=float(config.minimum_height),
                    maximum_height=float(config.maximum_height),
                    minimum_up=float(config.minimum_up),
                )[0].item()
            )
            if not alive:
                terminal_control_step = control_step
            previous_action = action
            progress += 1
        else:
            action = previous_action
            for _ in range(int(config.action_repeat)):
                raw_step += 1
                if raw_step % capture_stride == 0:
                    frames.append(qpos.detach().cpu().numpy()[0].copy())
                    capture_steps.append(raw_step)
        control_actions.append(action.detach().cpu().numpy()[0].copy())
        control_qpos.append(qpos.detach().cpu().numpy()[0].copy())
        control_qvel.append(qvel.detach().cpu().numpy()[0].copy())

    total_raw_steps = steps * int(config.action_repeat)
    if capture_steps[-1] != total_raw_steps:
        frames.append(qpos.detach().cpu().numpy()[0].copy())
        capture_steps.append(total_raw_steps)
    final_qpos = qpos.detach().cpu().numpy()[0].copy()
    simulated_seconds = (
        steps * float(loaded.model.opt.timestep) * int(config.action_repeat)
    )
    contacts = np.asarray(contact_counts, dtype=np.float64)
    return (
        frames,
        capture_steps,
        {
            "final_alive": alive,
            "terminal_control_step": terminal_control_step,
            "terminal_time_seconds": (
                terminal_control_step
                * float(loaded.model.opt.timestep)
                * int(config.action_repeat)
                if terminal_control_step is not None
                else None
            ),
            "control_steps": steps,
            "raw_physics_steps": total_raw_steps,
            "simulated_seconds": simulated_seconds,
            "displacement_x": float(final_qpos[0] - initial_qpos[0]),
            "displacement_y": float(final_qpos[1] - initial_qpos[1]),
            "forward_speed_over_horizon": float(
                (final_qpos[0] - initial_qpos[0]) / simulated_seconds
            ),
            "action_rms": float(np.sqrt(action_square_sum / max(action_count, 1))),
            "action_rate_rms": float(
                np.sqrt(action_rate_square_sum / max(action_count, 1))
            ),
            "active_contact_count_mean": (
                float(contacts.mean()) if contacts.size else None
            ),
            "active_contact_count_min": int(contacts.min()) if contacts.size else None,
            "active_contact_count_max": int(contacts.max()) if contacts.size else None,
            "initial_qpos_sha256": viewer._sha256_array(initial_qpos),
            "final_qpos_sha256": viewer._sha256_array(final_qpos),
            "recorded_qpos_trajectory_sha256": viewer._sha256_array(np.stack(frames)),
        },
        {
            "qpos": np.stack(control_qpos),
            "qvel": np.stack(control_qvel),
            "actions": np.stack(control_actions),
            "active": np.asarray(control_active, dtype=np.bool_),
        },
    )


@torch.no_grad()
def audit_recorded_rollout(
    loaded: gait.v1.LoadedModel,
    bridge: gait.v1.MJWarpTorchBridge,
    trace: dict[str, Any],
    config: SimpleNamespace,
) -> dict[str, Any]:
    """Measure the exact qpos trajectory sent to ViewerGL.

    A separate nominal rollout can enter a different long-horizon contact
    branch even from the same reset.  Auditing the recorded states makes the
    physical-plausibility claim about the actual video rather than a sibling
    trajectory.
    """
    qpos = torch.as_tensor(trace["qpos"], device=bridge.torch_device)
    qvel = torch.as_tensor(trace["qvel"], device=bridge.torch_device)
    actions = torch.as_tensor(trace["actions"], device=bridge.torch_device)
    active = torch.as_tensor(trace["active"], device=bridge.torch_device)
    steps = actions.shape[0]
    active_float = active.to(qpos.dtype)
    denominator = active_float.sum().clamp_min(1.0)
    control_dt = float(loaded.model.opt.timestep) * int(config.action_repeat)
    simulated_seconds = steps * control_dt

    body_positions: list[torch.Tensor] = []
    geom_positions: list[torch.Tensor] = []
    for index in range(steps + 1):
        body, geom = gait.scene_positions(
            bridge, qpos[index : index + 1], qvel[index : index + 1]
        )
        body_positions.append(body[0])
        geom_positions.append(geom[0])
    body_xpos = torch.stack(body_positions)
    geom_xpos = torch.stack(geom_positions)

    if loaded.spec.name == "humanoid":
        feet = body_xpos[:, (7, 10)]
        support_height = 0.060
    else:
        feet = 2.0 * geom_xpos[:, (4, 7, 10, 13)] - body_xpos[:, (4, 7, 10, 13)]
        support_height = 0.100
    foot_velocity = (feet[1:] - feet[:-1]) / control_dt
    support = feet[1:, :, 2] <= support_height
    support_count = support.sum(dim=-1)
    horizontal_speed_squared = foot_velocity[:, :, :2].square().sum(dim=-1)
    support_samples = (active[:, None] & support).sum().clamp_min(1)
    slip_rms = torch.sqrt(
        (
            horizontal_speed_squared
            * (active[:, None] & support).to(horizontal_speed_squared.dtype)
        ).sum()
        / support_samples
    )

    previous_actions = torch.cat((torch.zeros_like(actions[:1]), actions[:-1]))
    action_rate_rms = torch.sqrt(
        (
            (actions - previous_actions).square().mean(dim=-1) * active_float
        ).sum()
        / denominator
    )
    next_qpos = qpos[1:]
    final_alive = gait.healthy(
        loaded.spec.name,
        qpos[-1:],
        qvel[-1:],
        minimum_height=float(config.minimum_height),
        maximum_height=float(config.maximum_height),
        minimum_up=float(config.minimum_up),
    )[0]
    metrics: dict[str, Any] = {
        "final_alive_fraction": float(final_alive.item()),
        "mean_survival_fraction": float((active_float.sum() / steps).item()),
        "mean_forward_speed_over_horizon": float(
            ((qpos[-1, 0] - qpos[0, 0]) / simulated_seconds).item()
        ),
        "mean_abs_lateral_displacement": float(
            (qpos[-1, 1] - qpos[0, 1]).abs().item()
        ),
        "mean_action_rate_rms": float(action_rate_rms.item()),
        "mean_support_foot_slip_rms": float(slip_rms.item()),
        "mean_up_while_alive": float(
            (
                gait.root_up(next_qpos) * active_float
            ).sum().div(denominator).item()
        ),
        "mean_heading_while_alive": float(
            (
                gait.root_heading(next_qpos) * active_float
            ).sum().div(denominator).item()
        ),
        "mean_flight_fraction": float(
            (((support_count == 0).to(qpos.dtype) * active_float).sum() / denominator).item()
        ),
        "control_steps": int(steps),
        "simulated_seconds": simulated_seconds,
        "support_height_m": support_height,
        "source": "exact control-rate states from the trajectory rendered by ViewerGL",
    }

    if loaded.spec.name == "humanoid":
        single = support_count == 1
        last_support = 0
        switches = 0
        for is_active, is_single, pair in zip(
            active.tolist(), single.tolist(), support.tolist(), strict=True
        ):
            if not is_active or not is_single:
                continue
            dominant = 1 if pair[0] else -1
            if last_support and dominant != last_support:
                switches += 1
            last_support = dominant
        metrics.update(
            {
                "mean_single_support_fraction": float(
                    ((single.to(qpos.dtype) * active_float).sum() / denominator).item()
                ),
                "mean_alternating_support_switches_per_second": switches
                / simulated_seconds,
            }
        )
    else:
        two_or_more = support_count >= 2
        pair_a = support[:, 0] & support[:, 2]
        pair_b = support[:, 1] & support[:, 3]
        diagonal = (support_count == 2) & (pair_a | pair_b)
        last_diagonal = 0
        switches = 0
        for is_active, is_diagonal, is_a, is_b in zip(
            active.tolist(),
            diagonal.tolist(),
            pair_a.tolist(),
            pair_b.tolist(),
            strict=True,
        ):
            if not is_active or not is_diagonal:
                continue
            dominant = 1 if is_a and not is_b else -1
            if last_diagonal and dominant != last_diagonal:
                switches += 1
            last_diagonal = dominant
        metrics.update(
            {
                "mean_two_or_more_support_fraction": float(
                    (
                        (two_or_more.to(qpos.dtype) * active_float).sum()
                        / denominator
                    ).item()
                ),
                "mean_diagonal_support_fraction": float(
                    ((diagonal.to(qpos.dtype) * active_float).sum() / denominator).item()
                ),
                "mean_alternating_diagonal_support_switches_per_second": switches
                / simulated_seconds,
            }
        )
    metrics["gate"] = gait.gait_gate(loaded.spec.name, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("initial", "best", "final"), default="best"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--capture-stride", type=int)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--camera-mode", choices=("fixed", "track"), default="track")
    parser.add_argument("--camera-offset", type=float, nargs=3)
    parser.add_argument("--camera-target-height", type=float)
    parser.add_argument("--camera-fov", type=float)
    parser.add_argument(
        "--draw-edges", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--draw-shadows", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.capture_stride is not None and args.capture_stride <= 0:
        parser.error("--capture-stride must be positive")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even integers")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    poster_path = output_path.with_suffix(".jpg")
    manifest_path = output_path.with_suffix(".json")
    viewer._check_outputs(
        (output_path, poster_path, manifest_path), overwrite=args.overwrite
    )

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported checkpoint format: {checkpoint_format!r}")
    config_dict = dict(checkpoint["config"])
    config = SimpleNamespace(**config_dict)
    task = str(checkpoint["task"])
    if task not in gait.v1.TASKS:
        raise ValueError(f"unsupported checkpoint task: {task!r}")

    imported_root = Path(mjw.__file__).resolve().parent.parent
    pr_root = Path(config.pr_root).resolve()
    if imported_root != pr_root:
        raise RuntimeError(
            f"imported mujoco_warp from {imported_root}, expected {pr_root}"
        )
    xml_path = (
        Path(config.xml).expanduser().resolve()
        if config_dict.get("xml")
        else (
            pr_root / gait.v1.TASKS[task].default_xml
            if task == "humanoid"
            else gait.v1.TASKS[task].default_xml
        ).resolve()
    )

    seed = int(config.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    mjw.enable_grad()
    loaded = gait.v1.load_model(gait.v1.TASKS[task], xml_path)
    bridge_args = SimpleNamespace(
        worlds=1,
        device=args.device,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    evaluation_bridge = gait.v1.make_bridge(
        loaded,
        bridge_args,
    )
    actor, critic, normalizer = gait.make_networks(loaded, config, device)
    del critic
    actor_state, normalizer_state = policy_state(checkpoint, args.policy)
    actor.load_state_dict(actor_state)
    normalizer.load_state_dict(normalizer_state)
    actor.eval()
    normalizer.eval()
    evaluation_actor = conditioned_actor(actor, normalizer, checkpoint, loaded)

    steps = args.steps or int(config.eval_steps)
    capture_stride = args.capture_stride or max(
        1, round(1.0 / (args.fps * float(loaded.model.opt.timestep)))
    )
    evaluation = gait.evaluate_policy(
        evaluation_bridge,
        loaded,
        evaluation_actor,
        normalizer,
        config,
        seed=seed,
        steps=steps,
        noise=False,
    )
    # A render is also a reproducibility check.  Give recording an independent
    # MJWarp Data pair so no non-qpos/qvel step state from the audit rollout can
    # influence the captured trajectory.
    record_bridge = gait.v1.make_bridge(loaded, bridge_args)
    frames, capture_steps, behavior, recorded_trace = record_rollout(
        loaded,
        record_bridge,
        evaluation_actor,
        normalizer,
        config,
        steps=steps,
        capture_stride=capture_stride,
    )
    recorded_metric_bridge = gait.v1.make_bridge(loaded, bridge_args)
    recorded_evaluation = audit_recorded_rollout(
        loaded, recorded_metric_bridge, recorded_trace, config
    )
    if behavior["final_alive"] != bool(evaluation["final_alive_fraction"] == 1.0):
        raise RuntimeError(
            "recorded rollout and independent nominal evaluation disagree: "
            f"recorded_final_alive={behavior['final_alive']}, "
            f"recorded_terminal_step={behavior['terminal_control_step']}, "
            f"evaluation_final_alive_fraction="
            f"{evaluation['final_alive_fraction']}, "
            f"evaluation_survival_fraction="
            f"{evaluation['mean_survival_fraction']}"
        )
    if not recorded_evaluation["gate"]["pass"]:
        failed_checks = [
            name
            for name, passed in recorded_evaluation["gate"]["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "the exact trajectory requested for ViewerGL fails its gait gate: "
            + ", ".join(failed_checks)
            + "; metrics="
            + json.dumps(
                {
                    name: value
                    for name, value in recorded_evaluation.items()
                    if name not in {"gate", "source"}
                },
                sort_keys=True,
            )
        )

    camera_offset = (
        tuple(args.camera_offset)
        if args.camera_offset is not None
        else viewer._default_camera_offset(task)
    )
    camera_target_height = (
        float(args.camera_target_height)
        if args.camera_target_height is not None
        else viewer._default_camera_target_height(task, loaded.spec.target_height)
    )
    camera_fov = (
        float(args.camera_fov)
        if args.camera_fov is not None
        else viewer._default_camera_fov(task)
    )
    render = viewer._render(
        loaded,
        frames,
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
        terminal_step=(
            behavior["terminal_control_step"] * int(config.action_repeat)
            if behavior["terminal_control_step"] is not None
            else None
        ),
    )
    manifest = {
        "schema": "mjwarp-pr1535-viewergl-gait-v3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "checkpoint_format": checkpoint_format,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": viewer._sha256_file(checkpoint_path),
        "policy": args.policy,
        "policy_update": (
            int(checkpoint.get("best_update", 0))
            if args.policy == "best"
            else int(config_dict.get("updates", 0))
        ),
        "control_conditioning": checkpoint.get("control_conditioning"),
        "actor_state_dict_sha256": viewer._sha256_state_dict(actor.state_dict()),
        "normalizer_state_dict_sha256": viewer._sha256_state_dict(
            normalizer.state_dict()
        ),
        "physics": {
            "engine": "MuJoCo Warp",
            "dynamics": "MJWarpTorchBridge._forward_raw",
            "pr_head": checkpoint.get("pr_head"),
            "newton_head": checkpoint.get("newton_head"),
            "xml": str(xml_path),
            "xml_sha256": viewer._sha256_file(xml_path),
            "timestep": float(loaded.model.opt.timestep),
            "action_repeat": int(config.action_repeat),
        },
        "behavior": behavior,
        "recorded_gait_evaluation": recorded_evaluation,
        "independent_nominal_evaluation": evaluation,
        "render": render,
        "artifacts": {
            "video": output_path.name,
            "video_sha256": viewer._sha256_file(output_path),
            "poster": poster_path.name,
            "poster_sha256": viewer._sha256_file(poster_path),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "mujoco_warp": getattr(mjw, "__version__", None),
            "newton": viewer._distribution_version("newton"),
            "nvidia_driver": viewer._nvidia_driver(),
            "renderer_script_sha256": viewer._sha256_file(Path(__file__).resolve()),
            "conditioning_script_sha256": viewer._sha256_file(
                Path(__file__).with_name("conditioned_policy_v3.py")
            ),
            "viewer_backend_script_sha256": viewer._sha256_file(
                Path(viewer.__file__).resolve()
            ),
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote {output_path}, {poster_path}, and {manifest_path}")


if __name__ == "__main__":
    main()
