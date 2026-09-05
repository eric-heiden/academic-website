#!/usr/bin/env python3
"""Guarded analytic-adjoint fine-tuning for v3 locomotion policies.

This is the actor half of SHAC: a frozen PPO critic supplies the terminal
value after a short differentiable rollout, while PR #1535 supplies every
physics VJP.  Each analytic direction is subjected to a signed, full-horizon
line search.  A candidate is accepted only when the uninterrupted gait audit
improves without regressing survival or physical-plausibility metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
from conditioned_policy_v3 import conditioned_actor
import train_gaits_v3 as gait
import warp as wp


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def clone_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.named_parameters()
        if value.requires_grad
    }


def restore_parameters(
    module: torch.nn.Module, values: dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name in values:
                parameter.copy_(values[name])


def apply_direction(
    module: torch.nn.Module,
    base: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    step: float,
    norm: float,
) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name in gradients:
                # gradients are for loss=-objective, hence the minus sign.
                parameter.copy_(base[name] - step * gradients[name] / norm)


@torch.no_grad()
def warm_states(
    loaded: gait.v1.LoadedModel,
    bridge: gait.v1.MJWarpTorchBridge,
    actor: gait.PPOActor,
    normalizer: gait.RunningMeanStd,
    config: SimpleNamespace,
    *,
    seed: int,
    maximum_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device=bridge.torch_device).manual_seed(seed)
    qpos, qvel = gait.sample_initial_states(
        loaded,
        bridge.nworld,
        bridge.torch_device,
        generator,
        **gait.task_noise(config, evaluation=False),
    )
    initial_qpos = qpos.clone()
    initial_qvel = qvel.clone()
    previous_action = torch.zeros(
        (bridge.nworld, loaded.model.nu),
        dtype=torch.float32,
        device=bridge.torch_device,
    )
    progress = torch.zeros(bridge.nworld, dtype=torch.long, device=bridge.torch_device)
    target = torch.randint(
        0,
        maximum_steps + 1,
        (bridge.nworld,),
        generator=generator,
        device=bridge.torch_device,
    )
    valid = torch.ones(bridge.nworld, dtype=torch.bool, device=bridge.torch_device)
    for step in range(maximum_steps):
        active = target > step
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
        qpos_next, qvel_next = gait.physics_transition(
            bridge, qpos, qvel, action, int(config.action_repeat)
        )
        finite = torch.isfinite(qpos_next).all(dim=-1) & torch.isfinite(qvel_next).all(
            dim=-1
        )
        healthy = gait.healthy(
            loaded.spec.name,
            qpos_next,
            qvel_next,
            minimum_height=float(config.minimum_height),
            maximum_height=float(config.maximum_height),
            minimum_up=float(config.minimum_up),
        )
        valid &= (~active) | (finite & healthy)
        qpos = torch.where(active[:, None] & finite[:, None], qpos_next, qpos)
        qvel = torch.where(active[:, None] & finite[:, None], qvel_next, qvel)
        previous_action = torch.where(active[:, None], action, previous_action)
        progress += active.long()

    # Failed warm-up lanes remain useful nominal starting states instead of
    # feeding a contact-topology failure into the analytic rollout.
    qpos = torch.where(valid[:, None], qpos, initial_qpos)
    qvel = torch.where(valid[:, None], qvel, initial_qvel)
    previous_action = torch.where(
        valid[:, None], previous_action, torch.zeros_like(previous_action)
    )
    progress = torch.where(valid, progress, torch.zeros_like(progress))
    return (
        qpos,
        qvel,
        previous_action,
        progress,
        {
            "maximum_steps": maximum_steps,
            "mean_requested_steps": float(target.float().mean().item()),
            "valid_fraction": float(valid.float().mean().item()),
            "phase_histogram": torch.bincount(
                progress % max(int(config.phase_period), 1),
                minlength=max(int(config.phase_period), 1),
            )
            .cpu()
            .tolist(),
        },
    )


def ant_toe_positions(qpos: torch.Tensor) -> torch.Tensor:
    """Exact differentiable FK for the four distal Ant capsule endpoints.

    The vendored Ant has a two-hinge chain per leg.  Expressing this tiny FK
    directly in Torch keeps the physical gait terms on the analytic SHAC
    path; using detached MJWarp ``xpos`` values had silently removed every
    stance-slip and contact-schedule gradient from earlier fine-tuning.
    """
    base = qpos.new_tensor(
        ((0.2, 0.2, 0.0), (-0.2, 0.2, 0.0), (-0.2, -0.2, 0.0), (0.2, -0.2, 0.0))
    )
    ankle_axis = qpos.new_tensor(
        ((-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0))
    ) / math.sqrt(2.0)
    hip = qpos[:, 7:15:2]
    ankle = qpos[:, 8:15:2]

    def rotate_z(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        cosine = angle.cos()
        sine = angle.sin()
        return torch.stack(
            (
                vector[..., 0] * cosine - vector[..., 1] * sine,
                vector[..., 0] * sine + vector[..., 1] * cosine,
                vector[..., 2],
            ),
            dim=-1,
        )

    batch_base = base[None].expand(qpos.shape[0], -1, -1)
    axis = ankle_axis[None].expand_as(batch_base)
    distal = 2.0 * batch_base
    ankle_cosine = ankle.cos()[..., None]
    ankle_sine = ankle.sin()[..., None]
    ankle_rotated = (
        distal * ankle_cosine
        + torch.cross(axis, distal, dim=-1) * ankle_sine
        + axis
        * (axis * distal).sum(dim=-1, keepdim=True)
        * (1.0 - ankle_cosine)
    )
    local_toe = batch_base + rotate_z(batch_base, hip) + rotate_z(ankle_rotated, hip)

    root_quaternion = qpos[:, None, 3:7].expand(-1, 4, -1)
    root_vector = root_quaternion[..., 1:]
    root_scalar = root_quaternion[..., :1]
    rotated_toe = (
        local_toe
        + 2.0 * root_scalar * torch.cross(root_vector, local_toe, dim=-1)
        + 2.0
        * torch.cross(
            root_vector, torch.cross(root_vector, local_toe, dim=-1), dim=-1
        )
    )
    return qpos[:, None, :3] + rotated_toe


def differentiable_ant_gait_components(
    qpos: torch.Tensor,
    qpos_next: torch.Tensor,
    progress: torch.Tensor,
    config: SimpleNamespace,
    *,
    control_dt: float,
) -> dict[str, torch.Tensor]:
    feet = ant_toe_positions(qpos)
    feet_next = ant_toe_positions(qpos_next)
    foot_velocity = (feet_next - feet) / control_dt
    support = torch.exp(-((feet_next[:, :, 2] - 0.08) / 0.055).square())
    horizontal_speed_squared = foot_velocity[:, :, :2].square().sum(dim=-1)
    foot_slip = -float(config.foot_slip_weight) * (
        support * horizontal_speed_squared
    ).mean(dim=-1)
    swing_clearance = (
        (1.0 - support)
        * ((feet_next[:, :, 2] - 0.10) / 0.18).clamp(0.0, 1.0)
    ).mean(dim=-1)
    second_support = support.topk(2, dim=-1).values[:, 1]
    gait_weight = float(config.gait_shaping_weight)
    foot_clearance = 0.20 * gait_weight * swing_clearance
    foot_support = 0.20 * gait_weight * second_support
    flight_avoidance = -float(config.flight_avoidance_weight) * (
        1.0 - second_support
    ).square()
    step_span = (
        0.15
        * gait_weight
        * (
            (
                feet_next[:, (0, 3), 0].mean(dim=-1)
                - feet_next[:, (1, 2), 0].mean(dim=-1)
            ).abs()
            / 0.45
        ).clamp(0.0, 1.0)
    )
    phase_progress = (
        progress + 1 - int(getattr(config, "phase_warmup_steps", 0))
    ).clamp_min(0)
    phase = (
        2.0
        * math.pi
        * phase_progress.to(qpos.dtype)
        / max(int(config.phase_period), 1)
    )
    phase_sine = phase.sin()
    swing = torch.stack(
        (
            phase_sine.clamp_min(0.0),
            (-phase_sine).clamp_min(0.0),
            phase_sine.clamp_min(0.0),
            (-phase_sine).clamp_min(0.0),
        ),
        dim=-1,
    )
    desired_height = 0.08 + 0.16 * swing
    desired_support = 1.0 - swing
    height_error = ((feet_next[:, :, 2] - desired_height) / 0.14).square()
    support_error = (support - desired_support).square()
    gait_phase = -float(config.gait_phase_weight) * (
        height_error + 0.5 * support_error
    ).mean(dim=-1)
    return {
        "foot_slip": foot_slip,
        "foot_clearance": foot_clearance,
        "foot_support": foot_support,
        "flight_avoidance": flight_avoidance,
        "step_span": step_span,
        "gait_phase": gait_phase,
    }


def short_horizon_objective(
    loaded: gait.v1.LoadedModel,
    bridge: gait.v1.MJWarpTorchBridge,
    actor: nn.Module,
    anchor_actor: nn.Module,
    critic: gait.PPOCritic,
    normalizer: gait.RunningMeanStd,
    config: SimpleNamespace,
    qpos_start: torch.Tensor,
    qvel_start: torch.Tensor,
    previous_action_start: torch.Tensor,
    progress_start: torch.Tensor,
    *,
    horizon: int,
    action_repeat: int,
    terminal_value_weight: float,
    anchor_action_weight: float,
    differentiable: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    qpos = qpos_start
    qvel = qvel_start
    previous_action = previous_action_start
    # The objective is replayed for the finite-difference check and every
    # line-search candidate.  Keep its phase state referentially transparent.
    progress = progress_start.clone()
    objective = torch.zeros(bridge.nworld, device=bridge.torch_device)
    reward_sum = torch.zeros_like(objective)
    anchor_sum = torch.zeros_like(objective)
    discount = 1.0
    control_dt = float(loaded.model.opt.timestep) * action_repeat
    body_placeholder = torch.zeros(
        (bridge.nworld, loaded.model.nbody, 3),
        dtype=torch.float32,
        device=bridge.torch_device,
    )
    transition = bridge.step if differentiable else bridge._forward_raw

    for _ in range(horizon):
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
        # ``actor`` is the complete checkpoint-declared controller, not just
        # the PPO network.  This is important for the Ant CPG residual and for
        # the causal previous-action filters used by both final gaits: the
        # analytic direction must differentiate through exactly the same
        # action map that the full-horizon evaluator and ViewerGL use.
        action = actor.mean_action(observation)
        with torch.no_grad():
            anchor_action = anchor_actor.mean_action(observation.detach())
        qpos_next, qvel_next = qpos, qvel
        for _ in range(action_repeat):
            qpos_next, qvel_next = transition(qpos_next, qvel_next, action)
        if not bool(
            torch.isfinite(qpos_next).all() and torch.isfinite(qvel_next).all()
        ):
            raise FloatingPointError("non-finite state in analytic rollout")
        components = gait.reward_components(
            loaded.spec.name,
            qpos,
            qvel,
            action,
            previous_action,
            qpos_next,
            qvel_next,
            body_placeholder,
            body_placeholder,
            torch.ones(bridge.nworld, dtype=torch.bool, device=bridge.torch_device),
            progress,
            control_dt=control_dt,
            locomotion_scale=1.0,
            minimum_height=float(config.minimum_height),
            minimum_up=float(config.minimum_up),
            termination_penalty=0.0,
            survival_weight=float(
                getattr(
                    config,
                    "survival_weight",
                    2.0 if loaded.spec.name == "humanoid" else 1.0,
                )
            ),
            angular_weight=float(
                getattr(
                    config,
                    "angular_weight",
                    0.02 if loaded.spec.name == "humanoid" else 0.01,
                )
            ),
            forward_weight_override=float(config.forward_weight),
            smoothness_weight_override=float(config.smoothness_weight),
            lateral_weight_override=float(config.lateral_weight),
            lateral_position_weight=float(config.lateral_position_weight),
            foot_slip_weight=0.0,
            flight_avoidance_weight=0.0,
            gait_shaping_weight=0.0,
            gait_phase_weight=0.0,
            joint_reference_weight=float(config.joint_reference_weight),
            phase_period=int(config.phase_period),
            phase_warmup_steps=int(getattr(config, "phase_warmup_steps", 0)),
            target_speed_override=float(config.target_speed),
        )
        if loaded.spec.name == "ant":
            components.update(
                differentiable_ant_gait_components(
                    qpos, qpos_next, progress, config, control_dt=control_dt
                )
            )
        reward = gait.total_reward(components)
        anchor_penalty = anchor_action_weight * (action - anchor_action).square().mean(
            dim=-1
        )
        objective += discount * (reward - anchor_penalty)
        reward_sum += reward
        anchor_sum += anchor_penalty
        qpos, qvel = qpos_next, qvel_next
        previous_action = action
        progress += 1
        discount *= float(config.gamma)

    terminal_observation = normalizer(
        gait.raw_observation(
            qpos,
            qvel,
            previous_action,
            progress,
            int(config.phase_period),
            int(getattr(config, "phase_warmup_steps", 0)),
        )
    )
    terminal_value = critic(terminal_observation)
    objective += terminal_value_weight * discount * terminal_value
    result = objective.mean() / horizon
    diagnostics = {
        "objective": float(result.detach().item()),
        "direct_reward_per_step": float((reward_sum.mean() / horizon).detach().item()),
        "anchor_penalty_per_step": float((anchor_sum.mean() / horizon).detach().item()),
        "terminal_value_mean": float(terminal_value.mean().detach().item()),
    }
    return result, diagnostics


def candidate_safe(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    lower_bounds = {
        "final_alive_fraction": 0.005,
        "mean_survival_fraction": 0.003,
        "mean_up_while_alive": 0.005,
        "mean_heading_while_alive": 0.005,
    }
    upper_bounds = {
        "mean_abs_lateral_displacement": 0.03,
        "mean_action_rate_rms": 0.015,
    }
    if "mean_support_foot_slip_rms" in baseline:
        upper_bounds.update(
            {
                "mean_support_foot_slip_rms": 0.015,
                "mean_flight_fraction": 0.015,
            }
        )
    return all(
        candidate[name] >= baseline[name] - tolerance
        for name, tolerance in lower_bounds.items()
    ) and all(
        candidate[name] <= baseline[name] + tolerance
        for name, tolerance in upper_bounds.items()
    )


def robust_selection_score(
    noisy: dict[str, Any], nominal: dict[str, Any]
) -> float:
    """Use the weaker of noisy and nominal full-horizon performance.

    Optimizing only a randomized batch can admit a policy whose deterministic
    rollout crossed into a different contact regime.  The minimum makes that
    failure visible to the line search instead of averaging it away.
    """
    return min(float(noisy["selection_score"]), float(nominal["selection_score"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-policy", choices=("best", "final"), default="best"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--selection-worlds", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--gradient-action-repeat", type=int)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--selection-steps", type=int)
    parser.add_argument("--holdout-steps", type=int)
    parser.add_argument("--holdout-repeats", type=int, default=3)
    parser.add_argument("--terminal-value-weight", type=float, default=0.05)
    parser.add_argument("--anchor-action-weight", type=float, default=0.5)
    parser.add_argument(
        "--line-search-steps",
        type=float,
        nargs="+",
        default=[0.0, 1.0e-4, -1.0e-4, 3.0e-4, -3.0e-4, 1.0e-3, -1.0e-3],
    )
    parser.add_argument("--direction-check-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--minimum-score-gain", type=float, default=0.01)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    args = parser.parse_args()
    for name in (
        "worlds",
        "selection_worlds",
        "horizon",
        "epochs",
        "selection_steps",
        "holdout_steps",
        "holdout_repeats",
        "nconmax",
        "njmax",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be nonnegative")
    if args.direction_check_epsilon <= 0.0:
        parser.error("--direction-check-epsilon must be positive")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    if checkpoint.get("format") not in {
        "mjwarp-pr1535-full-gait-v3",
        "mjwarp-pr1535-shac-gait-v3",
    }:
        raise ValueError("input must be a v3 PPO or SHAC gait checkpoint")
    config_dict = dict(checkpoint["config"])
    config = SimpleNamespace(**config_dict)
    task = str(checkpoint["task"])
    pr_root = Path(config.pr_root).resolve()
    if Path(mjw.__file__).resolve().parent.parent != pr_root:
        raise RuntimeError("the imported mujoco_warp is not the checkpoint PR tree")
    if gait.git_head(pr_root) != checkpoint.get("pr_head"):
        raise RuntimeError("checkpoint and checked-out MuJoCo Warp revisions differ")
    xml_path = (
        Path(config.xml).resolve()
        if config_dict.get("xml")
        else (
            pr_root / gait.v1.TASKS[task].default_xml
            if task == "humanoid"
            else gait.v1.TASKS[task].default_xml
        ).resolve()
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    mjw.enable_grad()
    loaded = gait.v1.load_model(gait.v1.TASKS[task], xml_path)
    train_bridge = gait.v1.make_bridge(loaded, args)
    selection_bridge = gait.v1.make_bridge(
        loaded,
        SimpleNamespace(
            worlds=args.selection_worlds,
            device=args.device,
            nconmax=args.nconmax,
            njmax=args.njmax,
        ),
    )
    ant_fk_max_abs_error = None
    if task == "ant":
        fk_qpos = torch.as_tensor(
            loaded.initial_qpos,
            dtype=torch.float32,
            device=train_bridge.torch_device,
        ).repeat(train_bridge.nworld, 1)
        fk_qvel = torch.zeros(
            (train_bridge.nworld, loaded.model.nv),
            dtype=torch.float32,
            device=train_bridge.torch_device,
        )
        fk_body, fk_geom = gait.scene_positions(train_bridge, fk_qpos, fk_qvel)
        exact_toes = (
            2.0 * fk_geom[:, (4, 7, 10, 13)]
            - fk_body[:, (4, 7, 10, 13)]
        )
        ant_fk_max_abs_error = float(
            (ant_toe_positions(fk_qpos) - exact_toes).abs().amax().item()
        )
        if ant_fk_max_abs_error > 2.0e-6:
            raise RuntimeError(
                "differentiable Ant toe FK disagrees with MJWarp: "
                f"max_abs_error={ant_fk_max_abs_error:.3e}"
            )
    actor, critic, normalizer = gait.make_networks(
        loaded, config, train_bridge.torch_device
    )
    policy = args.checkpoint_policy
    actor.load_state_dict(checkpoint[f"{policy}_actor"])
    critic.load_state_dict(checkpoint[f"{policy}_critic"])
    normalizer.load_state_dict(checkpoint[f"{policy}_normalizer"])
    actor.eval()
    critic.eval()
    normalizer.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    anchor_actor = copy.deepcopy(actor).eval()
    for parameter in anchor_actor.parameters():
        parameter.requires_grad_(False)

    gradient_action_repeat = args.gradient_action_repeat or int(config.action_repeat)
    evaluation_actor = conditioned_actor(actor, normalizer, checkpoint, loaded)
    anchor_evaluation_actor = conditioned_actor(
        anchor_actor, normalizer, checkpoint, loaded
    )
    selection_steps = args.selection_steps or int(config.eval_steps)
    holdout_steps = args.holdout_steps or selection_steps
    initial_actor = gait.clone_state(actor)
    initial_metrics = gait.evaluate_policy(
        selection_bridge,
        loaded,
        evaluation_actor,
        normalizer,
        config,
        seed=args.seed + 10_000,
        steps=selection_steps,
        noise=True,
    )
    initial_nominal_metrics = gait.evaluate_policy(
        selection_bridge,
        loaded,
        evaluation_actor,
        normalizer,
        config,
        seed=0,
        steps=selection_steps,
        noise=False,
    )
    current_metrics = initial_metrics
    current_nominal_metrics = initial_nominal_metrics
    current_robust_score = robust_selection_score(
        current_metrics, current_nominal_metrics
    )
    best_actor = gait.clone_state(actor)
    best_metrics = copy.deepcopy(initial_metrics)
    best_nominal_metrics = copy.deepcopy(initial_nominal_metrics)
    best_robust_score = current_robust_score
    best_update = 0
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        qpos, qvel, previous_action, progress, warmup = warm_states(
            loaded,
            train_bridge,
            evaluation_actor,
            normalizer,
            config,
            seed=args.seed + epoch,
            maximum_steps=args.warmup_steps,
        )
        actor.zero_grad(set_to_none=True)
        objective, objective_diagnostics = short_horizon_objective(
            loaded,
            train_bridge,
            evaluation_actor,
            anchor_evaluation_actor,
            critic,
            normalizer,
            config,
            qpos,
            qvel,
            previous_action,
            progress,
            horizon=args.horizon,
            action_repeat=gradient_action_repeat,
            terminal_value_weight=args.terminal_value_weight,
            anchor_action_weight=args.anchor_action_weight,
            differentiable=True,
        )
        (-objective).backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in actor.named_parameters()
            if parameter.grad is not None
        }
        norm_tensor = torch.sqrt(
            sum(value.double().square().sum() for value in gradients.values())
        )
        gradient_norm = float(norm_tensor.item())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise FloatingPointError(f"invalid actor gradient norm: {gradient_norm}")
        base = clone_parameters(actor)

        epsilon = args.direction_check_epsilon
        direction_values = []
        for signed_epsilon in (-epsilon, epsilon):
            apply_direction(actor, base, gradients, signed_epsilon, gradient_norm)
            with torch.no_grad():
                value, _ = short_horizon_objective(
                    loaded,
                    train_bridge,
                    evaluation_actor,
                    anchor_evaluation_actor,
                    critic,
                    normalizer,
                    config,
                    qpos,
                    qvel,
                    previous_action,
                    progress,
                    horizon=args.horizon,
                    action_repeat=gradient_action_repeat,
                    terminal_value_weight=args.terminal_value_weight,
                    anchor_action_weight=args.anchor_action_weight,
                    differentiable=False,
                )
            direction_values.append(float(value.item()))
        finite_difference_slope = (direction_values[1] - direction_values[0]) / (
            2.0 * epsilon
        )
        direction_check = {
            "epsilon": epsilon,
            "analytic_slope": gradient_norm,
            "finite_difference_slope": finite_difference_slope,
            "relative_error": abs(finite_difference_slope - gradient_norm)
            / max(abs(finite_difference_slope), abs(gradient_norm), 1.0e-12),
            "same_sign": finite_difference_slope > 0.0,
        }

        candidates = []
        accepted_step = 0.0
        accepted_metrics = current_metrics
        accepted_nominal_metrics = current_nominal_metrics
        accepted_robust_score = current_robust_score
        for step in args.line_search_steps:
            apply_direction(actor, base, gradients, step, gradient_norm)
            metrics = gait.evaluate_policy(
                selection_bridge,
                loaded,
                evaluation_actor,
                normalizer,
                config,
                seed=args.seed + 10_000,
                steps=selection_steps,
                noise=True,
            )
            nominal_metrics = gait.evaluate_policy(
                selection_bridge,
                loaded,
                evaluation_actor,
                normalizer,
                config,
                seed=0,
                steps=selection_steps,
                noise=False,
            )
            safe = candidate_safe(metrics, current_metrics) and candidate_safe(
                nominal_metrics, current_nominal_metrics
            )
            gates_pass = bool(
                metrics["gate"]["pass"] and nominal_metrics["gate"]["pass"]
            )
            candidate_robust_score = robust_selection_score(
                metrics, nominal_metrics
            )
            candidate = {
                "step": step,
                "safe": safe,
                "noisy_gate_pass": metrics["gate"]["pass"],
                "nominal_gate_pass": nominal_metrics["gate"]["pass"],
                "robust_selection_score": candidate_robust_score,
                "selection_score": metrics["selection_score"],
                "nominal_selection_score": nominal_metrics["selection_score"],
                "final_alive_fraction": metrics["final_alive_fraction"],
                "nominal_final_alive_fraction": nominal_metrics[
                    "final_alive_fraction"
                ],
                "mean_survival_fraction": metrics["mean_survival_fraction"],
                "mean_forward_speed_over_horizon": metrics[
                    "mean_forward_speed_over_horizon"
                ],
                "mean_abs_lateral_displacement": metrics[
                    "mean_abs_lateral_displacement"
                ],
                "nominal_mean_abs_lateral_displacement": nominal_metrics[
                    "mean_abs_lateral_displacement"
                ],
                "mean_action_rate_rms": metrics["mean_action_rate_rms"],
            }
            for name in ("mean_support_foot_slip_rms", "mean_flight_fraction"):
                if name in metrics:
                    candidate[name] = metrics[name]
                    candidate[f"nominal_{name}"] = nominal_metrics[name]
            candidates.append(candidate)
            if (
                safe
                and gates_pass
                and candidate_robust_score
                > accepted_robust_score + args.minimum_score_gain
            ):
                accepted_step = step
                accepted_metrics = metrics
                accepted_nominal_metrics = nominal_metrics
                accepted_robust_score = candidate_robust_score

        apply_direction(actor, base, gradients, accepted_step, gradient_norm)
        accepted = accepted_step != 0.0
        current_metrics = accepted_metrics
        current_nominal_metrics = accepted_nominal_metrics
        current_robust_score = accepted_robust_score
        if current_robust_score > best_robust_score:
            best_metrics = copy.deepcopy(current_metrics)
            best_nominal_metrics = copy.deepcopy(current_nominal_metrics)
            best_robust_score = current_robust_score
            best_actor = gait.clone_state(actor)
            best_update = epoch
        row = {
            "epoch": epoch,
            "warmup": warmup,
            "objective": objective_diagnostics,
            "gradient_norm": gradient_norm,
            "direction_check": direction_check,
            "candidates": candidates,
            "accepted": accepted,
            "accepted_step": accepted_step,
            "selection": current_metrics,
            "nominal_selection": current_nominal_metrics,
            "robust_selection_score": current_robust_score,
        }
        history.append(row)
        write_json(
            output_path.with_suffix(".live.json"),
            {
                "schema": "mjwarp-pr1535-shac-gait-v3-live",
                "task": task,
                "history": history,
            },
        )
        print(
            f"{task} SHAC epoch {epoch:03d}: grad={gradient_norm:.4g} "
            f"fd_err={direction_check['relative_error']:.3%} "
            f"step={accepted_step:+.1e} "
            f"robust_score={current_robust_score:.1f} "
            f"gates={current_metrics['gate']['pass']}/"
            f"{current_nominal_metrics['gate']['pass']}",
            flush=True,
        )

    actor.load_state_dict(best_actor)
    holdouts = [
        gait.evaluate_policy(
            selection_bridge,
            loaded,
            evaluation_actor,
            normalizer,
            config,
            seed=args.seed + 100_000 + repeat,
            steps=holdout_steps,
            noise=True,
        )
        for repeat in range(args.holdout_repeats)
    ]
    nominal_holdout = gait.evaluate_policy(
        selection_bridge,
        loaded,
        evaluation_actor,
        normalizer,
        config,
        seed=0,
        steps=holdout_steps,
        noise=False,
    )
    checkpoint_out = {
        "format": "mjwarp-pr1535-shac-gait-v3",
        "algorithm": "guarded SHAC actor fine-tune from PPO gait anchor",
        "task": task,
        "pr_head": checkpoint.get("pr_head"),
        "newton_head": checkpoint.get("newton_head"),
        "observation_dim": checkpoint["observation_dim"],
        "action_dim": checkpoint["action_dim"],
        "actor_hidden": checkpoint["actor_hidden"],
        "critic_hidden": checkpoint["critic_hidden"],
        "config": config_dict,
        "shac_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "initial_actor": initial_actor,
        "initial_critic": checkpoint[f"{policy}_critic"],
        "initial_normalizer": checkpoint[f"{policy}_normalizer"],
        "best_actor": best_actor,
        "best_critic": checkpoint[f"{policy}_critic"],
        "best_normalizer": checkpoint[f"{policy}_normalizer"],
        "final_actor": gait.clone_state(actor),
        "final_critic": checkpoint[f"{policy}_critic"],
        "final_normalizer": checkpoint[f"{policy}_normalizer"],
        "best_update": best_update,
        "source_checkpoint_sha256": gait.sha256_file(checkpoint_path),
    }
    for metadata_key in (
        "control_conditioning",
        "calibration",
        "gait_gate_profile",
        "cpg_search",
    ):
        if metadata_key in checkpoint:
            checkpoint_out[metadata_key] = copy.deepcopy(checkpoint[metadata_key])
    checkpoint_out_path = output_path.with_suffix(".pt")
    checkpoint_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_out, checkpoint_out_path)
    result = {
        "schema": "mjwarp-pr1535-shac-gait-v3",
        "status": "completed",
        "task": task,
        "algorithm": "guarded short-horizon analytic-adjoint actor fine-tune",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": gait.sha256_file(checkpoint_path),
        "checkpoint": str(checkpoint_out_path),
        "checkpoint_sha256": gait.sha256_file(checkpoint_out_path),
        "initial_selection": initial_metrics,
        "initial_nominal_selection": initial_nominal_metrics,
        "best_selection": best_metrics,
        "best_nominal_selection": best_nominal_metrics,
        "best_robust_selection_score": best_robust_score,
        "best_update": best_update,
        "history": history,
        "holdouts": holdouts,
        "nominal_holdout": nominal_holdout,
        "holdout_all_gates_pass": bool(
            all(item["gate"]["pass"] for item in holdouts)
            and nominal_holdout["gate"]["pass"]
        ),
        "ant_fk_max_abs_error": ant_fk_max_abs_error,
        "timing_seconds": time.perf_counter() - start_time,
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": gait.sha256_file(Path(__file__).resolve()),
            "gait_harness_sha256": gait.sha256_file(Path(gait.__file__).resolve()),
            "bridge_sha256": gait.sha256_file(
                gait.SCRIPT_DIR / "mjwarp_torch_bridge.py"
            ),
            "xml": str(xml_path),
            "xml_sha256": gait.sha256_file(xml_path),
            "mjwarp_pr_head": gait.git_head(pr_root),
            "newton_head": gait.git_head(Path(config.newton_root)),
            "versions": {
                "python": platform.python_version(),
                "mujoco": mujoco.__version__,
                "mujoco_warp": getattr(mjw, "__version__", None),
                "warp": wp.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "method_notes": [
                "The actor direction differentiates through the PR #1535 out-of-place analytic adjoint.",
                "The frozen PPO critic supplies a short-horizon terminal value, as in SHAC.",
                "Signed line-search candidates are selected only by exact uninterrupted full-horizon MJWarp evaluation.",
                "Both randomized and nominal full-horizon gates must pass; selection maximizes their weaker score.",
                "Foot-contact plausibility remains a non-differentiable selection guard; it is not claimed as an analytic gradient path.",
                "Checkpoint-declared causal control conditioning is differentiated through and included in selection.",
                "For Ant, exact closed-form toe FK restores stance-slip and gait-schedule terms to the analytic actor objective.",
            ],
        },
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote {output_path} and {checkpoint_out_path}")


if __name__ == "__main__":
    main()
