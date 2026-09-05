#!/usr/bin/env python3
"""Fit a compact Ant trot residual with exact MJWarp rollouts.

The PPO locomotion policy can discover a fast airborne bound, while a heavily
weighted contact reward can collapse to standing still.  This script searches
the small, interpretable middle ground: a state-feedback PPO actor plus a
phase-locked diagonal-leg residual.  Candidates share identical noisy initial
states and are evaluated in parallel MJWarp worlds.  The saved residual is
consumed by ``conditioned_policy_v3.py`` in audits, SHAC and ViewerGL renders.
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
import warp as wp

import train_gaits_v3 as gait


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def decode(parameters: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map twelve bounded search parameters to actuator-order conditioning."""
    population = parameters.shape[0]
    sine = parameters.new_zeros((population, 8))
    cosine = parameters.new_zeros((population, 8))
    bias = parameters.new_zeros((population, 8))

    # XML actuator order: hip4, ankle4, hip1, ankle1, hip2, ankle2,
    # hip3, ankle3.  The signs below were derived by finite-differencing the
    # exact XML toe endpoints.  Positive clock sine advances and lifts the
    # hip1/hip3 diagonal while retracting and loading hip2/hip4.
    hip = parameters.new_tensor((-1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0))
    ankle = parameters.new_tensor((0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0))
    diagonal = parameters.new_tensor((1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0))
    lift = parameters.new_tensor((0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0))
    side = parameters.new_tensor((-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0))

    sine += parameters[:, 2:3] * hip
    sine += parameters[:, 3:4] * hip * diagonal
    sine += parameters[:, 4:5] * ankle
    sine += parameters[:, 5:6] * ankle * diagonal
    cosine += parameters[:, 6:7] * hip
    cosine += parameters[:, 7:8] * ankle
    bias += parameters[:, 8:9] * lift
    bias += parameters[:, 9:10] * hip
    bias += parameters[:, 10:11] * side * (hip != 0.0)
    bias += parameters[:, 11:12] * side * (ankle != 0.0)
    return {
        "base_action_scale": parameters[:, 0],
        "alpha": parameters[:, 1],
        "sine": sine,
        "cosine": cosine,
        "bias": bias,
    }


def decode_pd(parameters: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "hip_amplitude": parameters[:, 0],
        "ankle_base": parameters[:, 1],
        "ankle_lift": parameters[:, 2],
        "hip_stiffness": parameters[:, 3],
        "ankle_stiffness": parameters[:, 4],
        "hip_damping": parameters[:, 5],
        "ankle_damping": parameters[:, 6],
        "alpha": parameters[:, 7],
        "hip_forward_bias": parameters[:, 8],
        "hip_side_bias": parameters[:, 9],
        "ankle_bias": parameters[:, 10],
        "base_action_scale": parameters[:, 11],
        "pd_action_scale": parameters[:, 12],
        "hip_phase_offset": parameters[:, 13],
        "lift_phase_offset": parameters[:, 14],
    }


@torch.no_grad()
def evaluate_candidates(
    loaded: gait.v1.LoadedModel,
    bridge: gait.v1.MJWarpTorchBridge,
    actor: gait.PPOActor,
    normalizer: gait.RunningMeanStd,
    config: SimpleNamespace,
    parameters: torch.Tensor,
    *,
    controller: str,
    replicas: int,
    steps: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    candidates = parameters.shape[0]
    if bridge.nworld != candidates * replicas:
        raise ValueError("bridge world count does not match candidates x replicas")
    decoded_once = decode_pd(parameters) if controller == "pd" else decode(parameters)
    decoded = {
        name: value.repeat_interleave(replicas, dim=0)
        for name, value in decoded_once.items()
    }
    generator = torch.Generator(device=bridge.torch_device).manual_seed(seed)
    initial_qpos, initial_qvel = gait.sample_initial_states(
        loaded,
        replicas,
        bridge.torch_device,
        generator,
        **gait.task_noise(config, evaluation=True),
    )
    qpos = initial_qpos.repeat(candidates, 1)
    qvel = initial_qvel.repeat(candidates, 1)
    rollout_initial_qpos = qpos.clone()
    previous_action = torch.zeros(
        (bridge.nworld, loaded.model.nu),
        dtype=torch.float32,
        device=bridge.torch_device,
    )
    progress = torch.zeros(
        bridge.nworld, dtype=torch.long, device=bridge.torch_device
    )
    alive = gait.healthy(
        "ant",
        qpos,
        qvel,
        minimum_height=float(config.minimum_height),
        maximum_height=float(config.maximum_height),
        minimum_up=float(config.minimum_up),
    )
    alive_steps = torch.zeros(bridge.nworld, device=bridge.torch_device)
    up_sum = torch.zeros_like(alive_steps)
    heading_sum = torch.zeros_like(alive_steps)
    action_rate_square_sum = torch.zeros_like(alive_steps)
    slip_square_sum = torch.zeros_like(alive_steps)
    support_samples = torch.zeros_like(alive_steps)
    two_support_sum = torch.zeros_like(alive_steps)
    diagonal_sum = torch.zeros_like(alive_steps)
    flight_sum = torch.zeros_like(alive_steps)
    switches = torch.zeros_like(alive_steps)
    last_diagonal = torch.zeros(
        bridge.nworld, dtype=torch.int8, device=bridge.torch_device
    )
    control_dt = float(loaded.model.opt.timestep) * int(config.action_repeat)
    ankle_body_ids = (4, 7, 10, 13)
    ankle_geom_ids = (4, 7, 10, 13)

    for _ in range(steps):
        active = alive
        body, geom = gait.scene_positions(bridge, qpos, qvel)
        raw_observation = gait.raw_observation(
            qpos,
            qvel,
            previous_action,
            progress,
            int(config.phase_period),
            int(getattr(config, "phase_warmup_steps", 0)),
        )
        observation = normalizer(raw_observation)
        phase_sine = raw_observation[:, -2]
        phase_cosine = raw_observation[:, -1]
        if controller == "pd":
            joint_qpos = qpos[:, 7:15]
            joint_qvel = qvel[:, 6:14]
            hip_phase = (
                phase_cosine * decoded["hip_phase_offset"].cos()
                - phase_sine * decoded["hip_phase_offset"].sin()
            )
            lift_phase = (
                phase_sine * decoded["lift_phase_offset"].cos()
                + phase_cosine * decoded["lift_phase_offset"].sin()
            )
            pair_a_swing = lift_phase.clamp_min(0.0)
            pair_b_swing = (-lift_phase).clamp_min(0.0)
            swing = torch.stack(
                (pair_a_swing, pair_b_swing, pair_a_swing, pair_b_swing),
                dim=-1,
            )
            hip_sign = qpos.new_tensor((-1.0, 1.0, 1.0, -1.0))
            hip_side = qpos.new_tensor((1.0, 1.0, -1.0, -1.0))
            ankle_sign = qpos.new_tensor((1.0, -1.0, -1.0, 1.0))
            hip_target = (
                -decoded["hip_amplitude"][:, None]
                * hip_sign
                * hip_phase[:, None]
                + decoded["hip_forward_bias"][:, None] * hip_sign
                + decoded["hip_side_bias"][:, None] * hip_side
            )
            ankle_magnitude = (
                decoded["ankle_base"][:, None]
                - decoded["ankle_lift"][:, None] * swing
                + decoded["ankle_bias"][:, None]
            )
            target = torch.empty_like(joint_qpos)
            target[:, (0, 2, 4, 6)] = hip_target
            target[:, (1, 3, 5, 7)] = ankle_sign * ankle_magnitude
            stiffness = torch.stack(
                (
                    decoded["hip_stiffness"],
                    decoded["ankle_stiffness"],
                    decoded["hip_stiffness"],
                    decoded["ankle_stiffness"],
                    decoded["hip_stiffness"],
                    decoded["ankle_stiffness"],
                    decoded["hip_stiffness"],
                    decoded["ankle_stiffness"],
                ),
                dim=-1,
            )
            damping = torch.stack(
                (
                    decoded["hip_damping"],
                    decoded["ankle_damping"],
                    decoded["hip_damping"],
                    decoded["ankle_damping"],
                    decoded["hip_damping"],
                    decoded["ankle_damping"],
                    decoded["hip_damping"],
                    decoded["ankle_damping"],
                ),
                dim=-1,
            )
            joint_action = (
                stiffness * (target - joint_qpos) - damping * joint_qvel
            ) / 150.0
            pd_action = joint_action[:, (6, 7, 0, 1, 2, 3, 4, 5)]
            desired_action = (
                decoded["base_action_scale"][:, None]
                * actor.mean_action(observation)
                + decoded["pd_action_scale"][:, None] * pd_action
            ).clamp(-1.0, 1.0)
        else:
            base_action = actor.mean_action(observation)
            phase_cosine_minus_one = raw_observation[:, -1:] - 1.0
            desired_action = (
                decoded["base_action_scale"][:, None] * base_action
                + decoded["bias"]
                + phase_sine[:, None] * decoded["sine"]
                + phase_cosine_minus_one * decoded["cosine"]
            ).clamp(-1.0, 1.0)
        action = previous_action + decoded["alpha"][:, None] * (
            desired_action - previous_action
        )
        candidate_qpos, candidate_qvel = gait.physics_transition(
            bridge, qpos, qvel, action, int(config.action_repeat)
        )
        candidate_body, candidate_geom = gait.scene_positions(
            bridge, candidate_qpos, candidate_qvel
        )
        finite = torch.isfinite(candidate_qpos).all(dim=-1) & torch.isfinite(
            candidate_qvel
        ).all(dim=-1)
        safe_qpos = torch.where(finite[:, None], candidate_qpos, qpos)
        safe_qvel = torch.where(finite[:, None], candidate_qvel, qvel)
        safe_body = torch.where(finite[:, None, None], candidate_body, body)
        safe_geom = torch.where(finite[:, None, None], candidate_geom, geom)
        next_alive = active & finite & gait.healthy(
            "ant",
            safe_qpos,
            safe_qvel,
            minimum_height=float(config.minimum_height),
            maximum_height=float(config.maximum_height),
            minimum_up=float(config.minimum_up),
        )
        active_float = active.to(torch.float32)
        alive_steps += active_float
        up_sum += active_float * gait.root_up(safe_qpos)
        heading_sum += active_float * gait.root_heading(safe_qpos)
        action_rate_square_sum += active_float * (
            action - previous_action
        ).square().mean(dim=-1)

        feet = 2.0 * geom[:, ankle_geom_ids] - body[:, ankle_body_ids]
        feet_next = (
            2.0 * safe_geom[:, ankle_geom_ids] - safe_body[:, ankle_body_ids]
        )
        foot_velocity = (feet_next - feet) / control_dt
        # Capsule radius (0.08 m) plus the 0.02 m combined contact margin.
        # A former 0.12 m proxy counted fast near-ground swing as stance.
        support = feet_next[:, :, 2] <= 0.100
        support_float = support.to(torch.float32)
        slip_square_sum += active_float * (
            support_float * foot_velocity[:, :, :2].square().sum(dim=-1)
        ).sum(dim=-1)
        support_samples += active_float * support_float.sum(dim=-1)
        support_count = support.sum(dim=-1)
        two_support_sum += active_float * (support_count >= 2).to(torch.float32)
        flight_sum += active_float * (support_count == 0).to(torch.float32)
        pair_a = support[:, 0] & support[:, 2]
        pair_b = support[:, 1] & support[:, 3]
        diagonal = (support_count == 2) & (pair_a | pair_b)
        diagonal_sum += active_float * diagonal.to(torch.float32)
        dominant = torch.where(
            pair_a & ~pair_b,
            torch.ones_like(last_diagonal),
            torch.where(
                pair_b & ~pair_a,
                -torch.ones_like(last_diagonal),
                torch.zeros_like(last_diagonal),
            ),
        )
        switched = (
            active
            & diagonal
            & (last_diagonal != 0)
            & (dominant != last_diagonal)
        )
        switches += switched.to(torch.float32)
        last_diagonal = torch.where(active & diagonal, dominant, last_diagonal)

        qpos = torch.where(active[:, None], safe_qpos, qpos)
        qvel = torch.where(active[:, None], safe_qvel, qvel)
        previous_action = torch.where(
            active[:, None], action, previous_action
        )
        progress += active.to(torch.long)
        alive = next_alive

    denominator = alive_steps.clamp_min(1.0)
    seconds = steps * control_dt
    lane_metrics = {
        "final_alive_fraction": alive.to(torch.float32),
        "mean_survival_fraction": alive_steps / steps,
        "mean_forward_speed_over_horizon": (
            qpos[:, 0] - rollout_initial_qpos[:, 0]
        )
        / seconds,
        "mean_up_while_alive": up_sum / denominator,
        "mean_heading_while_alive": heading_sum / denominator,
        "mean_abs_lateral_displacement": (
            qpos[:, 1] - rollout_initial_qpos[:, 1]
        ).abs(),
        "mean_action_rate_rms": (action_rate_square_sum / denominator).sqrt(),
        "mean_support_foot_slip_rms": (
            slip_square_sum / support_samples.clamp_min(1.0)
        ).sqrt(),
        "mean_two_or_more_support_fraction": two_support_sum / denominator,
        "mean_diagonal_support_fraction": diagonal_sum / denominator,
        "mean_alternating_diagonal_support_switches_per_second": switches / seconds,
        "mean_flight_fraction": flight_sum / denominator,
    }
    return {
        name: value.reshape(candidates, replicas).mean(dim=1)
        for name, value in lane_metrics.items()
    }


def fitness(
    metrics: dict[str, torch.Tensor], *, objective_profile: str
) -> torch.Tensor:
    """Continuous full-horizon objective for the requested gait regime.

    ``strict_grounded`` preserves the intentionally conservative diagnostic
    objective used during the first root-cause pass.  It turned out to
    over-reward slow, almost-static contact: demanding 60% multi-foot support
    and under 10% flight is a walking constraint, while simultaneously asking
    this Ant to cover 1.5 m/s.  ``fast_trot`` instead permits the short flight
    interval of a running trot while keeping survival, diagonal alternation,
    smooth control and stance-slip explicit.
    """
    relu = torch.relu
    if objective_profile == "dynamic_trot":
        score = (
            160.0 * metrics["mean_forward_speed_over_horizon"].clamp(-1.0, 2.0)
            + 12.0 * metrics["mean_diagonal_support_fraction"]
            + 4.0
            * metrics[
                "mean_alternating_diagonal_support_switches_per_second"
            ].clamp(0.0, 4.0)
            - 24.0 * metrics["mean_support_foot_slip_rms"]
            - 8.0 * metrics["mean_action_rate_rms"]
            - 18.0 * metrics["mean_flight_fraction"]
            - 8.0 * metrics["mean_abs_lateral_displacement"]
        )
        # Optimize to margins beyond the publication gate.  A 16-replica CEM
        # can otherwise select a controller that looks perfect in-search but
        # loses roughly 1--2% of lanes in a 1024-world audit.
        score -= 5_000.0 * relu(0.995 - metrics["mean_survival_fraction"])
        score -= 2_000.0 * relu(0.990 - metrics["final_alive_fraction"])
        score -= 50.0 * relu(0.90 - metrics["mean_up_while_alive"])
        score -= 50.0 * relu(0.75 - metrics["mean_heading_while_alive"])
        score -= 80.0 * relu(metrics["mean_abs_lateral_displacement"] - 1.40)
        score -= 50.0 * relu(metrics["mean_action_rate_rms"] - 0.35)
        score -= 45.0 * relu(metrics["mean_support_foot_slip_rms"] - 1.10)
        score -= 30.0 * relu(0.05 - metrics["mean_diagonal_support_fraction"])
        score -= 200.0 * relu(
            0.65
            - metrics["mean_alternating_diagonal_support_switches_per_second"]
        )
        score -= 80.0 * relu(metrics["mean_flight_fraction"] - 0.58)
        score -= 150.0 * relu(0.75 - metrics["mean_forward_speed_over_horizon"])
        return score

    if objective_profile == "fast_trot":
        score = (
            85.0 * metrics["mean_forward_speed_over_horizon"].clamp(-1.0, 2.0)
            + 8.0 * metrics["mean_two_or_more_support_fraction"]
            + 18.0 * metrics["mean_diagonal_support_fraction"]
            + 7.0
            * metrics[
                "mean_alternating_diagonal_support_switches_per_second"
            ].clamp(0.0, 3.0)
            - 25.0 * metrics["mean_support_foot_slip_rms"]
            - 7.0 * metrics["mean_action_rate_rms"]
            - 35.0 * metrics["mean_flight_fraction"]
            - 4.0 * metrics["mean_abs_lateral_displacement"]
        )
        score -= 500.0 * relu(0.990 - metrics["mean_survival_fraction"])
        score -= 200.0 * relu(0.980 - metrics["final_alive_fraction"])
        score -= 50.0 * relu(0.85 - metrics["mean_up_while_alive"])
        score -= 40.0 * relu(0.75 - metrics["mean_heading_while_alive"])
        score -= 25.0 * relu(metrics["mean_abs_lateral_displacement"] - 1.0)
        score -= 45.0 * relu(metrics["mean_action_rate_rms"] - 0.55)
        score -= 120.0 * relu(metrics["mean_support_foot_slip_rms"] - 0.95)
        score -= 80.0 * relu(0.45 - metrics["mean_two_or_more_support_fraction"])
        score -= 45.0 * relu(0.15 - metrics["mean_diagonal_support_fraction"])
        score -= 20.0 * relu(
            0.80
            - metrics["mean_alternating_diagonal_support_switches_per_second"]
        )
        score -= 120.0 * relu(metrics["mean_flight_fraction"] - 0.20)
        score -= 90.0 * relu(1.00 - metrics["mean_forward_speed_over_horizon"])
        return score

    score = (
        40.0 * metrics["mean_forward_speed_over_horizon"].clamp(-1.0, 1.5)
        + 10.0 * metrics["mean_two_or_more_support_fraction"]
        + 20.0 * metrics["mean_diagonal_support_fraction"]
        + 5.0
        * metrics["mean_alternating_diagonal_support_switches_per_second"].clamp(
            0.0, 3.0
        )
        - 15.0 * metrics["mean_support_foot_slip_rms"]
        - 5.0 * metrics["mean_action_rate_rms"]
        - 20.0 * metrics["mean_flight_fraction"]
        - 3.0 * metrics["mean_abs_lateral_displacement"]
    )
    score -= 250.0 * relu(0.995 - metrics["mean_survival_fraction"])
    score -= 100.0 * relu(0.98 - metrics["final_alive_fraction"])
    score -= 40.0 * relu(0.85 - metrics["mean_up_while_alive"])
    score -= 30.0 * relu(0.75 - metrics["mean_heading_while_alive"])
    score -= 12.0 * relu(metrics["mean_abs_lateral_displacement"] - 1.0)
    score -= 30.0 * relu(metrics["mean_action_rate_rms"] - 0.45)
    score -= 80.0 * relu(metrics["mean_support_foot_slip_rms"] - 0.50)
    score -= 50.0 * relu(0.60 - metrics["mean_two_or_more_support_fraction"])
    score -= 50.0 * relu(0.15 - metrics["mean_diagonal_support_fraction"])
    score -= 10.0 * relu(
        0.60 - metrics["mean_alternating_diagonal_support_switches_per_second"]
    )
    score -= 60.0 * relu(metrics["mean_flight_fraction"] - 0.10)
    score -= 45.0 * relu(1.50 - metrics["mean_forward_speed_over_horizon"])
    return score


def metric_row(metrics: dict[str, torch.Tensor], index: int) -> dict[str, float]:
    return {name: float(value[index].item()) for name, value in metrics.items()}


def gate_pass(row: dict[str, float], *, objective_profile: str) -> bool:
    if objective_profile == "dynamic_trot":
        return bool(
            row["final_alive_fraction"] >= 0.98
            and row["mean_survival_fraction"] >= 0.980
            and row["mean_forward_speed_over_horizon"] >= 0.75
            and row["mean_up_while_alive"] >= 0.90
            and row["mean_heading_while_alive"] >= 0.75
            and row["mean_abs_lateral_displacement"] <= 1.50
            and row["mean_action_rate_rms"] <= 0.35
            and row["mean_support_foot_slip_rms"] <= 1.10
            and row["mean_two_or_more_support_fraction"] >= 0.05
            and row["mean_diagonal_support_fraction"] >= 0.05
            and row[
                "mean_alternating_diagonal_support_switches_per_second"
            ]
            >= 0.60
            and row["mean_flight_fraction"] <= 0.60
        )
    if objective_profile == "fast_trot":
        return bool(
            row["final_alive_fraction"] >= 0.98
            and row["mean_survival_fraction"] >= 0.990
            and row["mean_forward_speed_over_horizon"] >= 1.0
            and row["mean_up_while_alive"] >= 0.85
            and row["mean_heading_while_alive"] >= 0.75
            and row["mean_abs_lateral_displacement"] <= 1.0
            and row["mean_action_rate_rms"] <= 0.55
            and row["mean_support_foot_slip_rms"] <= 0.95
            and row["mean_two_or_more_support_fraction"] >= 0.45
            and row["mean_diagonal_support_fraction"] >= 0.15
            and row[
                "mean_alternating_diagonal_support_switches_per_second"
            ]
            >= 0.80
            and row["mean_flight_fraction"] <= 0.20
        )
    return bool(
        row["final_alive_fraction"] >= 0.98
        and row["mean_survival_fraction"] >= 0.995
        and row["mean_forward_speed_over_horizon"] >= 1.5
        and row["mean_up_while_alive"] >= 0.85
        and row["mean_heading_while_alive"] >= 0.75
        and row["mean_abs_lateral_displacement"] <= 1.0
        and row["mean_action_rate_rms"] <= 0.45
        and row["mean_support_foot_slip_rms"] <= 0.50
        and row["mean_two_or_more_support_fraction"] >= 0.60
        and row["mean_diagonal_support_fraction"] >= 0.15
        and row["mean_alternating_diagonal_support_switches_per_second"] >= 0.60
        and row["mean_flight_fraction"] <= 0.10
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", choices=("best", "final"), default="best")
    parser.add_argument("--controller", choices=("residual", "pd"), default="residual")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--elite", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--phase-period", type=int)
    parser.add_argument(
        "--objective-profile",
        choices=("strict_grounded", "fast_trot", "dynamic_trot"),
        default="fast_trot",
    )
    parser.add_argument(
        "--initial-json",
        type=Path,
        help="optional earlier search whose best parameters seed this run",
    )
    parser.add_argument(
        "--continue-distribution",
        action="store_true",
        help="also reuse the last CEM standard deviation from --initial-json",
    )
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    if min(args.population, args.replicas, args.generations, args.steps) <= 0:
        parser.error("population, replicas, generations and steps must be positive")
    if not 1 <= args.elite < args.population:
        parser.error("--elite must be in [1, population)")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "mjwarp-pr1535-full-gait-v3":
        raise ValueError("input must be a full-gait v3 PPO checkpoint")
    if checkpoint.get("task") != "ant":
        raise ValueError("this structured residual is specific to Ant")
    config_dict = dict(checkpoint["config"])
    if args.phase_period is not None:
        config_dict["phase_period"] = args.phase_period
    config = SimpleNamespace(**config_dict)
    pr_root = Path(config.pr_root).resolve()
    if Path(mjw.__file__).resolve().parent.parent != pr_root:
        raise RuntimeError("the imported MuJoCo Warp is not the checkpoint PR tree")
    if gait.git_head(pr_root) != checkpoint.get("pr_head"):
        raise RuntimeError("checkpoint and checked-out MuJoCo Warp revisions differ")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    mjw.enable_grad()
    loaded = gait.v1.load_model(gait.v1.TASKS["ant"], gait.v1.TASKS["ant"].default_xml)
    bridge = gait.v1.make_bridge(
        loaded,
        SimpleNamespace(
            worlds=args.population * args.replicas,
            device=args.device,
            nconmax=args.nconmax,
            njmax=args.njmax,
        ),
    )
    actor, critic, normalizer = gait.make_networks(loaded, config, bridge.torch_device)
    del critic
    policy = args.checkpoint_policy
    actor.load_state_dict(checkpoint[f"{policy}_actor"])
    normalizer.load_state_dict(checkpoint[f"{policy}_normalizer"])
    actor.eval()
    normalizer.eval()

    if args.controller == "pd":
        lower = torch.tensor(
            [-0.52, 0.60, 0.05, 30.0, 30.0, 0.5, 0.5, 0.50, -0.25, -0.20, -0.20, 0.0, 0.50, -math.pi, -math.pi],
            device=bridge.torch_device,
        )
        upper = torch.tensor(
            [0.52, 1.18, 0.60, 500.0, 500.0, 40.0, 40.0, 1.00, 0.25, 0.20, 0.20, 0.35, 1.50, math.pi, math.pi],
            device=bridge.torch_device,
        )
        mean = torch.tensor(
            [0.30, 1.00, 0.42, 220.0, 260.0, 8.0, 10.0, 0.85, 0.0, 0.0, 0.0, 0.10, 1.00, -0.5 * math.pi, 0.0],
            device=bridge.torch_device,
        )
        std = torch.tensor(
            [0.18, 0.15, 0.18, 110.0, 110.0, 6.0, 7.0, 0.12, 0.10, 0.08, 0.08, 0.12, 0.30, 1.0, 0.8],
            device=bridge.torch_device,
        )
    else:
        lower = torch.tensor(
            [0.20, 0.50, -0.80, -0.40, -0.80, -0.40, -0.60, -0.60, -0.30, -0.20, -0.20, -0.20],
            device=bridge.torch_device,
        )
        upper = torch.tensor(
            [1.30, 1.00, 0.80, 0.40, 0.80, 0.40, 0.60, 0.60, 0.30, 0.20, 0.20, 0.20],
            device=bridge.torch_device,
        )
        mean = torch.tensor(
            [0.85, 0.82, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            device=bridge.torch_device,
        )
        std = torch.tensor(
            [0.25, 0.12, 0.30, 0.15, 0.30, 0.15, 0.25, 0.25, 0.12, 0.08, 0.08, 0.08],
            device=bridge.torch_device,
        )
    if args.initial_json is not None:
        with args.initial_json.expanduser().open() as stream:
            initial_search = json.load(stream)
        seeded = torch.as_tensor(
            initial_search["best_parameters"],
            dtype=torch.float32,
            device=bridge.torch_device,
        )
        if seeded.shape != mean.shape:
            raise ValueError("--initial-json uses a different parameterization")
        mean = seeded
        if args.continue_distribution:
            continued_std = torch.as_tensor(
                initial_search["history"][-1]["std"],
                dtype=torch.float32,
                device=bridge.torch_device,
            )
            if continued_std.shape != std.shape:
                raise ValueError(
                    "--initial-json has an incompatible CEM distribution"
                )
            std = continued_std
    generator = torch.Generator(device=bridge.torch_device).manual_seed(args.seed)
    history: list[dict[str, Any]] = []
    best_fitness = -math.inf
    best_parameters = mean.clone()
    best_metrics: dict[str, float] = {}
    start = time.perf_counter()

    for generation in range(1, args.generations + 1):
        samples = mean + std * torch.randn(
            (args.population, mean.numel()),
            generator=generator,
            device=bridge.torch_device,
        )
        samples = torch.maximum(torch.minimum(samples, upper), lower)
        samples[0] = mean
        if generation == 1 and args.controller == "residual":
            samples[1] = torch.tensor(
                [1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                device=bridge.torch_device,
            )
        metrics = evaluate_candidates(
            loaded,
            bridge,
            actor,
            normalizer,
            config,
            samples,
            controller=args.controller,
            replicas=args.replicas,
            steps=args.steps,
            seed=args.seed + 10_000 + generation,
        )
        scores = fitness(metrics, objective_profile=args.objective_profile)
        ranking = torch.argsort(scores, descending=True)
        elite = samples[ranking[: args.elite]]
        elite_mean = elite.mean(dim=0)
        elite_std = elite.std(dim=0, unbiased=False)
        parameter_floor = 0.015 * (upper - lower)
        mean = 0.25 * mean + 0.75 * elite_mean
        std = torch.maximum(0.35 * std + 0.65 * elite_std, parameter_floor)
        top_index = int(ranking[0].item())
        top_fitness = float(scores[top_index].item())
        top_metrics = metric_row(metrics, top_index)
        if top_fitness > best_fitness:
            best_fitness = top_fitness
            best_parameters = samples[top_index].clone()
            best_metrics = top_metrics
        row = {
            "generation": generation,
            "top_fitness": top_fitness,
            "top_parameters": samples[top_index].detach().cpu().tolist(),
            "top_metrics": top_metrics,
            "top_gate_pass": gate_pass(
                top_metrics, objective_profile=args.objective_profile
            ),
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "mean": mean.detach().cpu().tolist(),
            "std": std.detach().cpu().tolist(),
        }
        history.append(row)
        write_json(
            output_path.with_suffix(".live.json"),
            {
                "schema": "mjwarp-pr1535-ant-cpg-search-v3-live",
                "history": history,
            },
        )
        print(
            f"generation {generation:02d}: fitness={top_fitness:.3f} "
            f"speed={top_metrics['mean_forward_speed_over_horizon']:.3f} "
            f"survival={top_metrics['mean_survival_fraction']:.4f} "
            f"slip={top_metrics['mean_support_foot_slip_rms']:.3f} "
            f"two={top_metrics['mean_two_or_more_support_fraction']:.3f} "
            f"diag={top_metrics['mean_diagonal_support_fraction']:.3f} "
            f"switch={top_metrics['mean_alternating_diagonal_support_switches_per_second']:.3f} "
            f"flight={top_metrics['mean_flight_fraction']:.3f} "
            f"gate={gate_pass(top_metrics, objective_profile=args.objective_profile)}",
            flush=True,
        )

    decoded_best = (
        decode_pd(best_parameters[None])
        if args.controller == "pd"
        else decode(best_parameters[None])
    )
    if args.controller == "pd":
        pd_names = (
            "hip_amplitude",
            "ankle_base",
            "ankle_lift",
            "hip_stiffness",
            "ankle_stiffness",
            "hip_damping",
            "ankle_damping",
            "hip_forward_bias",
            "hip_side_bias",
            "ankle_bias",
            "base_action_scale",
            "pd_action_scale",
            "hip_phase_offset",
            "lift_phase_offset",
        )
        conditioning = {
            "previous_action_low_pass_alpha": float(
                decoded_best["alpha"][0].item()
            ),
            "ant_phase_pd": {
                **{
                    name: float(decoded_best[name][0].item()) for name in pd_names
                },
                "method": "phase-locked joint-space PD teacher fitted by parallel CEM rollouts",
            },
        }
    else:
        conditioning = {
            "previous_action_low_pass_alpha": float(
                decoded_best["alpha"][0].item()
            ),
            "periodic_action_residual": {
                "base_action_scale": float(
                    decoded_best["base_action_scale"][0].item()
                ),
                "sine_coefficients": decoded_best["sine"][0].cpu().tolist(),
                "cosine_minus_one_coefficients": decoded_best["cosine"][0]
                .cpu()
                .tolist(),
                "action_bias": decoded_best["bias"][0].cpu().tolist(),
                "method": "phase-locked diagonal CPG residual fitted by parallel CEM rollouts",
            },
        }
    result_checkpoint = copy.deepcopy(checkpoint)
    result_checkpoint["config"] = config_dict
    for prefix in ("initial", "best", "final"):
        result_checkpoint[f"{prefix}_actor"] = copy.deepcopy(
            checkpoint[f"{policy}_actor"]
        )
        result_checkpoint[f"{prefix}_normalizer"] = copy.deepcopy(
            checkpoint[f"{policy}_normalizer"]
        )
        critic_key = f"{policy}_critic"
        if critic_key in checkpoint:
            result_checkpoint[f"{prefix}_critic"] = copy.deepcopy(
                checkpoint[critic_key]
            )
    result_checkpoint["best_update"] = 0
    result_checkpoint["control_conditioning"] = conditioning
    result_checkpoint["gait_gate_profile"] = args.objective_profile
    result_checkpoint["cpg_search"] = {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": gait.sha256_file(checkpoint_path),
        "source_policy": policy,
        "best_parameters": best_parameters.cpu().tolist(),
        "best_fitness": best_fitness,
        "best_metrics_during_search": best_metrics,
        "search_script_sha256": gait.sha256_file(Path(__file__).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_out = output_path.with_suffix(".pt")
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result_checkpoint, checkpoint_out)
    result = {
        "schema": "mjwarp-pr1535-ant-cpg-search-v3",
        "status": "completed",
        "task": "ant",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": gait.sha256_file(checkpoint_path),
        "checkpoint": str(checkpoint_out),
        "checkpoint_sha256": gait.sha256_file(checkpoint_out),
        "conditioning": conditioning,
        "gait_gate_profile": args.objective_profile,
        "best_parameters": best_parameters.cpu().tolist(),
        "best_fitness": best_fitness,
        "best_metrics_during_search": best_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - start,
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
        },
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": gait.sha256_file(Path(__file__).resolve()),
            "gait_harness_sha256": gait.sha256_file(Path(gait.__file__).resolve()),
            "conditioning_script_sha256": gait.sha256_file(
                Path(__file__).with_name("conditioned_policy_v3.py")
            ),
            "xml_sha256": gait.sha256_file(loaded.xml_path),
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
                "Every candidate in a generation receives identical noisy initial states.",
                "Terminal lanes freeze; displacement is divided by the complete requested horizon.",
                "Search fitness contains every published Ant physicality gate.",
                "The fitted controller is phase-locked, symmetric by construction, and uses explicit state feedback.",
            ],
        },
    }
    write_json(output_path, result)
    print(f"Wrote {output_path} and {checkpoint_out}")


if __name__ == "__main__":
    main()
