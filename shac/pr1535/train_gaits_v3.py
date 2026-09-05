#!/usr/bin/env python3
"""Full-episode gait training for the MJWarp PR #1535 report.

The earlier compact SHAC-style experiment was useful for diagnosing the
adjoint, but it was not a competitive locomotion learner.  This harness uses
PPO to discover robust full-episode policies directly in the exact MJWarp PR
physics.  Its checkpoints are also designed to be usable as anchors for a
subsequent differentiable-physics fine-tuning pass.

The task reward deliberately gates forward progress by torso posture.  This
prevents the short-lived forward dive that dominated the earlier Humanoid
experiment.  Evaluation is uninterrupted: a fallen lane is frozen, never
reset, so survival and displacement cannot be inflated by repeated episodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import torch.nn.functional as F
import train_shac as v1
import warp as wp
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PR_ROOT = Path("/home/horde/repos/mujoco_warp-pr1535")
DEFAULT_NEWTON_ROOT = Path("/home/horde/repos/newton-shac-pr1535")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sync(device: torch.device) -> None:
    wp.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class RunningMeanStd(nn.Module):
    def __init__(self, size: int, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size, device=device))
        self.register_buffer("variance", torch.ones(size, device=device))
        self.register_buffer("count", torch.tensor(1.0e-4, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1, values.shape[-1])
        if values.shape[0] == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(values.shape[0]), device=values.device)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        old_m2 = self.variance * self.count
        batch_m2 = batch_variance * batch_count
        new_m2 = old_m2 + batch_m2 + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.variance.copy_(new_m2 / total)
        self.count.copy_(total)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.mean) / torch.sqrt(self.variance + 1.0e-6)
        return normalized.clamp(-10.0, 10.0)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        *,
        output_gain: float,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        self.hidden = nn.ModuleList()
        self.norms = nn.ModuleList()
        for input_size, output_size in pairwise(dims):
            layer = nn.Linear(input_size, output_size)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            self.norms.append(nn.LayerNorm(output_size))
        self.output = nn.Linear(dims[-1], output_dim)
        nn.init.orthogonal_(self.output.weight, gain=output_gain)
        nn.init.zeros_(self.output.bias)

    def features(self, values: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.hidden, self.norms, strict=True):
            values = norm(F.elu(layer(values)))
        return values

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(values))


class PPOActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: list[int],
        *,
        initial_log_std: float,
    ) -> None:
        super().__init__()
        action_dim = len(action_low)
        self.policy = MLP(observation_dim, hidden_dims, action_dim, output_gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32)
        )

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.policy(observation)
        log_std = self.log_std.clamp(-5.0, 1.0)
        return torch.distributions.Normal(mean, log_std.exp().expand_as(mean))

    def scale_action(self, unit_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + 0.5 * (unit_action + 1.0) * (
            self.action_high - self.action_low
        )

    def mean_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.scale_action(torch.tanh(self.policy(observation)))

    def sample(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw_action = distribution.sample()
        unit_action = torch.tanh(raw_action)
        action = self.scale_action(unit_action)
        log_probability = tanh_log_probability(distribution, raw_action, unit_action)
        return raw_action, action, log_probability

    def evaluate_raw(
        self, observation: torch.Tensor, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        unit_action = torch.tanh(raw_action)
        log_probability = tanh_log_probability(distribution, raw_action, unit_action)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy


class PPOCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.value = MLP(observation_dim, hidden_dims, 1, output_gain=1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(observation).squeeze(-1)


def tanh_log_probability(
    distribution: torch.distributions.Normal,
    raw_action: torch.Tensor,
    unit_action: torch.Tensor,
) -> torch.Tensor:
    log_probability = distribution.log_prob(raw_action).sum(dim=-1)
    correction = torch.log(torch.clamp(1.0 - unit_action.square(), min=1.0e-6)).sum(
        dim=-1
    )
    return log_probability - correction


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def root_up(qpos: torch.Tensor) -> torch.Tensor:
    return 1.0 - 2.0 * (qpos[:, 4].square() + qpos[:, 5].square())


def root_heading(qpos: torch.Tensor) -> torch.Tensor:
    return 1.0 - 2.0 * (qpos[:, 5].square() + qpos[:, 6].square())


def raw_observation(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    previous_action: torch.Tensor,
    progress: torch.Tensor,
    phase_period: int,
    phase_warmup_steps: int = 0,
) -> torch.Tensor:
    values = [qpos[:, 2:], qvel, previous_action]
    if phase_period > 0:
        phase_progress = (progress - phase_warmup_steps).clamp_min(0)
        phase = 2.0 * math.pi * phase_progress.to(qpos.dtype) / phase_period
        values.extend((phase.sin()[:, None], phase.cos()[:, None]))
    return torch.cat(values, dim=-1).clamp(-100.0, 100.0)


def sample_initial_states(
    loaded: v1.LoadedModel,
    count: int,
    device: torch.device,
    generator: torch.Generator,
    *,
    position_noise: float,
    angle_noise: float,
    joint_noise: float,
    velocity_noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = torch.as_tensor(
        loaded.initial_qpos, dtype=torch.float32, device=device
    ).repeat(count, 1)
    qvel = torch.zeros((count, loaded.model.nv), dtype=torch.float32, device=device)
    if position_noise > 0.0:
        qpos[:, :3] += position_noise * (
            2.0 * torch.rand((count, 3), device=device, generator=generator) - 1.0
        )
    if angle_noise > 0.0:
        angles = angle_noise * (
            2.0 * torch.rand(count, device=device, generator=generator) - 1.0
        )
        axes = torch.randn((count, 3), device=device, generator=generator)
        axes = F.normalize(axes, dim=-1)
        perturbation = torch.cat(
            (
                torch.cos(0.5 * angles)[:, None],
                axes * torch.sin(0.5 * angles)[:, None],
            ),
            dim=-1,
        )
        qpos[:, 3:7] = quaternion_multiply(perturbation, qpos[:, 3:7])
        qpos[:, 3:7] = F.normalize(qpos[:, 3:7], dim=-1)
    if joint_noise > 0.0 and qpos.shape[1] > 7:
        qpos[:, 7:] += joint_noise * (
            2.0
            * torch.rand((count, qpos.shape[1] - 7), device=device, generator=generator)
            - 1.0
        )
    if velocity_noise > 0.0:
        qvel += velocity_noise * (
            2.0 * torch.rand(qvel.shape, device=device, generator=generator) - 1.0
        )
    return qpos, qvel


def healthy(
    task: str,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    *,
    minimum_height: float,
    maximum_height: float,
    minimum_up: float,
) -> torch.Tensor:
    finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
    bounded = (qpos.abs().amax(dim=-1) < 1.0e3) & (qvel.abs().amax(dim=-1) < 1.0e3)
    posture = (
        (qpos[:, 2] >= minimum_height)
        & (qpos[:, 2] <= maximum_height)
        & (root_up(qpos) >= minimum_up)
    )
    del task
    return finite & bounded & posture


def posture_gate(
    qpos: torch.Tensor, *, minimum_height: float, minimum_up: float
) -> torch.Tensor:
    height_gate = ((qpos[:, 2] - minimum_height) / 0.25).clamp(0.0, 1.0)
    up_gate = ((root_up(qpos) - minimum_up) / (1.0 - minimum_up)).clamp(0.0, 1.0)
    return height_gate * up_gate


def reward_components(
    task: str,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    qpos_next: torch.Tensor,
    qvel_next: torch.Tensor,
    body_xpos: torch.Tensor,
    body_xpos_next: torch.Tensor,
    alive_next: torch.Tensor,
    progress: torch.Tensor,
    *,
    geom_xpos: torch.Tensor | None = None,
    geom_xpos_next: torch.Tensor | None = None,
    control_dt: float,
    locomotion_scale: float,
    minimum_height: float,
    minimum_up: float,
    termination_penalty: float,
    survival_weight: float,
    angular_weight: float,
    forward_weight_override: float | None,
    smoothness_weight_override: float | None,
    lateral_weight_override: float | None,
    lateral_position_weight: float,
    foot_slip_weight: float,
    flight_avoidance_weight: float,
    gait_shaping_weight: float,
    gait_phase_weight: float,
    joint_reference_weight: float,
    phase_period: int,
    phase_warmup_steps: int,
    target_speed_override: float | None,
) -> dict[str, torch.Tensor]:
    forward_velocity = (qpos_next[:, 0] - qpos[:, 0]) / control_dt
    forward_velocity = forward_velocity.clamp(-5.0, 8.0)
    up = root_up(qpos_next)
    heading = root_heading(qpos_next)
    gate = posture_gate(qpos_next, minimum_height=minimum_height, minimum_up=minimum_up)
    action_cost = action.square().mean(dim=-1)
    action_rate = (action - previous_action).square().mean(dim=-1)
    joint_velocity = qvel_next[:, 6:].square().mean(dim=-1)
    lateral_velocity = qvel_next[:, 1].square()
    angular_velocity = qvel_next[:, 3:6].square().mean(dim=-1)

    if task == "humanoid":
        target_height = 1.282
        forward_weight = forward_weight_override or 3.0
        target_speed = target_speed_override or 1.5
        survival = torch.full_like(forward_velocity, survival_weight)
        upright = 1.5 * up.clamp(0.0, 1.0).square()
        heading_reward = 0.5 * heading.clamp(-1.0, 1.0)
        height = 1.5 * torch.exp(-((qpos_next[:, 2] - target_height) / 0.16).square())
        control = -0.035 * action_cost
        smoothness = -(smoothness_weight_override or 0.08) * action_rate
        joint_speed = -0.0015 * joint_velocity
        lateral = -(lateral_weight_override or 0.08) * lateral_velocity
        lateral_position = -lateral_position_weight * qpos_next[:, 1].square()
        angular = -angular_weight * angular_velocity
        posture = -0.004 * qpos_next[:, 7:].square().mean(dim=-1)

        # A root-velocity reward alone admits an unphysical skating solution:
        # the torso advances while both feet slide along the floor.  Foot
        # positions come from MJWarp's own post-step kinematics.  Weighting
        # horizontal foot speed by a smooth support estimate penalizes stance
        # slip but leaves a lifted swing foot free to advance.
        feet = body_xpos[:, (7, 10)]
        feet_next = body_xpos_next[:, (7, 10)]
        foot_velocity = (feet_next - feet) / control_dt
        support = torch.exp(-((feet_next[:, :, 2] - 0.027) / 0.035).square())
        foot_horizontal_speed_squared = foot_velocity[:, :, :2].square().sum(dim=-1)
        foot_slip = -foot_slip_weight * (support * foot_horizontal_speed_squared).mean(
            dim=-1
        )
        swing_clearance = (
            (1.0 - support) * ((feet_next[:, :, 2] - 0.04) / 0.12).clamp(0.0, 1.0)
        ).mean(dim=-1)
        foot_clearance = 0.25 * gait_shaping_weight * swing_clearance
        foot_support = 0.15 * gait_shaping_weight * support.amax(dim=-1)
        flight_avoidance = (
            -flight_avoidance_weight * (1.0 - support.amax(dim=-1)).square()
        )
        step_span = (
            0.25
            * gait_shaping_weight
            * ((feet_next[:, 0, 0] - feet_next[:, 1, 0]).abs() / 0.35).clamp(0.0, 1.0)
        )
        step_width = (
            -0.20
            * gait_shaping_weight
            * (((feet_next[:, 0, 1] - feet_next[:, 1, 1]).abs() - 0.18) / 0.12).square()
        )

        # A clock does not dictate the learned action.  It supplies a smooth,
        # symmetric contact schedule and a deliberately modest leg reference,
        # which removes the equally rewarding solutions where both feet skate
        # or chatter in phase.  Phase zero is the neutral standing pose, so a
        # warm-started balance policy enters the curriculum without a jump.
        if phase_period > 0:
            phase_progress = (progress + 1 - phase_warmup_steps).clamp_min(0)
            phase = 2.0 * math.pi * phase_progress.to(qpos.dtype) / phase_period
            phase_sine = phase.sin()
            right_swing = phase_sine.clamp_min(0.0)
            left_swing = (-phase_sine).clamp_min(0.0)
            desired_foot_height = 0.027 + 0.12 * torch.stack(
                (right_swing, left_swing), dim=-1
            )
            desired_support = 1.0 - torch.stack((right_swing, left_swing), dim=-1)
            foot_height_error = (
                (feet_next[:, :, 2] - desired_foot_height) / 0.10
            ).square()
            support_error = (support - desired_support).square()
            gait_phase = -gait_phase_weight * (
                foot_height_error + 0.5 * support_error
            ).mean(dim=-1)

            joint_target = torch.zeros_like(qpos_next[:, 7:])
            # Right/left hip pitch, knee, and ankle pitch.  The modest target
            # remains well inside the XML joint limits and leaves PPO free to
            # discover dynamically necessary deviations.
            joint_target[:, 5] = -0.30 * phase_sine
            joint_target[:, 6] = -0.65 * right_swing
            joint_target[:, 7] = 0.18 * right_swing
            joint_target[:, 11] = 0.30 * phase_sine
            joint_target[:, 12] = -0.65 * left_swing
            joint_target[:, 13] = 0.18 * left_swing
            reference_error = torch.stack(
                (
                    ((qpos_next[:, 12] - joint_target[:, 5]) / 0.35).square(),
                    ((qpos_next[:, 13] - joint_target[:, 6]) / 0.70).square(),
                    ((qpos_next[:, 14] - joint_target[:, 7]) / 0.35).square(),
                    ((qpos_next[:, 18] - joint_target[:, 11]) / 0.35).square(),
                    ((qpos_next[:, 19] - joint_target[:, 12]) / 0.70).square(),
                    ((qpos_next[:, 20] - joint_target[:, 13]) / 0.35).square(),
                ),
                dim=-1,
            )
            joint_reference = -joint_reference_weight * reference_error.mean(dim=-1)
        else:
            gait_phase = torch.zeros_like(forward_velocity)
            joint_reference = torch.zeros_like(forward_velocity)
    else:
        target_height = 0.55
        forward_weight = forward_weight_override or 2.5
        target_speed = target_speed_override or 2.5
        survival = torch.full_like(forward_velocity, survival_weight)
        upright = 0.75 * up.clamp(0.0, 1.0).square()
        heading_reward = 0.25 * heading.clamp(-1.0, 1.0)
        height = 0.75 * torch.exp(-((qpos_next[:, 2] - target_height) / 0.12).square())
        control = -0.025 * action_cost
        smoothness = -(smoothness_weight_override or 0.04) * action_rate
        joint_speed = -0.001 * joint_velocity
        lateral = -(lateral_weight_override or 0.05) * lateral_velocity
        lateral_position = -lateral_position_weight * qpos_next[:, 1].square()
        angular = -angular_weight * angular_velocity
        posture = torch.zeros_like(forward_velocity)

        # The Ant's distal capsules have no named toe bodies.  Their body
        # origin is the inner capsule endpoint and geom_xpos is the centre, so
        # reflection about the centre gives the outer endpoint exactly.  A
        # toe centre at z~=0.08 touches the ground because the capsule radius
        # is 0.08 m.  This makes stance-slip and support constraints about the
        # physical contact point rather than a convenient limb origin.
        if geom_xpos is not None and geom_xpos_next is not None:
            ankle_body_ids = (4, 7, 10, 13)
            ankle_geom_ids = (4, 7, 10, 13)
            feet = (
                2.0 * geom_xpos[:, ankle_geom_ids] - body_xpos[:, ankle_body_ids]
            )
            feet_next = (
                2.0 * geom_xpos_next[:, ankle_geom_ids]
                - body_xpos_next[:, ankle_body_ids]
            )
            foot_velocity = (feet_next - feet) / control_dt
            support = torch.exp(-((feet_next[:, :, 2] - 0.08) / 0.055).square())
            foot_horizontal_speed_squared = foot_velocity[:, :, :2].square().sum(
                dim=-1
            )
            foot_slip = -foot_slip_weight * (
                support * foot_horizontal_speed_squared
            ).mean(dim=-1)
            swing_clearance = (
                (1.0 - support)
                * ((feet_next[:, :, 2] - 0.10) / 0.18).clamp(0.0, 1.0)
            ).mean(dim=-1)
            foot_clearance = 0.20 * gait_shaping_weight * swing_clearance
            second_support = support.topk(2, dim=-1).values[:, 1]
            foot_support = 0.20 * gait_shaping_weight * second_support
            flight_avoidance = -flight_avoidance_weight * (
                1.0 - second_support
            ).square()
            step_span = (
                0.15
                * gait_shaping_weight
                * (
                    (
                        feet_next[:, (0, 3), 0].mean(dim=-1)
                        - feet_next[:, (1, 2), 0].mean(dim=-1)
                    ).abs()
                    / 0.45
                ).clamp(0.0, 1.0)
            )
            step_width = torch.zeros_like(forward_velocity)
        else:
            feet_next = None
            support = None
            foot_slip = torch.zeros_like(forward_velocity)
            foot_clearance = torch.zeros_like(forward_velocity)
            foot_support = torch.zeros_like(forward_velocity)
            flight_avoidance = torch.zeros_like(forward_velocity)
            step_span = torch.zeros_like(forward_velocity)
            step_width = torch.zeros_like(forward_velocity)

        if phase_period > 0:
            phase_progress = (progress + 1 - phase_warmup_steps).clamp_min(0)
            phase = 2.0 * math.pi * phase_progress.to(qpos.dtype) / phase_period
            phase_sine = phase.sin()
            phase_cosine = phase.cos()
            pair_a_swing = phase_sine.clamp_min(0.0)
            pair_b_swing = (-phase_sine).clamp_min(0.0)
            swing = torch.stack(
                (pair_a_swing, pair_b_swing, pair_a_swing, pair_b_swing),
                dim=-1,
            )
            if feet_next is not None and support is not None:
                desired_foot_height = 0.08 + 0.16 * swing
                desired_support = 1.0 - swing
                foot_height_error = (
                    (feet_next[:, :, 2] - desired_foot_height) / 0.14
                ).square()
                support_error = (support - desired_support).square()
                gait_phase = -gait_phase_weight * (
                    foot_height_error + 0.5 * support_error
                ).mean(dim=-1)
            else:
                gait_phase = torch.zeros_like(forward_velocity)

            neutral = qpos_next.new_tensor(
                (0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0)
            )
            joint_target = neutral.expand(qpos_next.shape[0], -1).clone()
            # Finite differences of the exact XML toe endpoints give the hip
            # signs below.  The former ankle-only reference could lift a
            # diagonal but never advance it; under strong contact penalties
            # the resulting local optimum was simply to stand still.  This
            # modest fore-aft sweep advances the lifted hip1/hip3 diagonal
            # while retracting the supporting hip2/hip4 diagonal.  Hip
            # position follows cosine, not sine: its derivative is therefore
            # monotonic through each stance half-cycle.  A sine hip target
            # reverses direction at mid-stance and makes the toe scrub.
            hip_sign = qpos_next.new_tensor((-1.0, 1.0, 1.0, -1.0))
            phase_activation = (
                phase_progress.to(qpos.dtype) / max(phase_period, 1)
            ).clamp(0.0, 1.0)
            joint_target[:, (0, 2, 4, 6)] = (
                -0.30
                * hip_sign
                * phase_cosine[:, None]
                * phase_activation[:, None]
            )
            ankle_sign = qpos_next.new_tensor((1.0, -1.0, -1.0, 1.0))
            joint_target[:, (1, 3, 5, 7)] = ankle_sign * (1.0 - 0.45 * swing)
            reference_error = torch.cat(
                (
                    (
                        qpos_next[:, (7, 9, 11, 13)]
                        - joint_target[:, (0, 2, 4, 6)]
                    )
                    / 0.35,
                    (
                        qpos_next[:, (8, 10, 12, 14)]
                        - joint_target[:, (1, 3, 5, 7)]
                    )
                    / 0.45,
                ),
                dim=-1,
            ).square()
            joint_reference = -joint_reference_weight * reference_error.mean(dim=-1)
        else:
            gait_phase = torch.zeros_like(forward_velocity)
            joint_reference = torch.zeros_like(forward_velocity)

    # Saturating at the declared task speed keeps the objective about useful
    # locomotion rather than rewarding arbitrarily fast contact exploits.
    normalized_forward = (forward_velocity / target_speed).clamp(-1.0, 1.0)
    if phase_warmup_steps > 0:
        startup_scale = ((progress + 1).to(qpos.dtype) / phase_warmup_steps).clamp(
            0.0, 1.0
        )
    else:
        startup_scale = torch.ones_like(forward_velocity)
    forward = (
        locomotion_scale * startup_scale * forward_weight * normalized_forward * gate
    )
    termination = (~alive_next).to(qpos.dtype) * -termination_penalty
    return {
        "forward": forward,
        "survival": survival,
        "upright": upright,
        "heading": heading_reward,
        "height": height,
        "control": control,
        "smoothness": smoothness,
        "joint_speed": joint_speed,
        "lateral": lateral,
        "lateral_position": lateral_position,
        "angular": angular,
        "posture": posture,
        "foot_slip": foot_slip,
        "foot_clearance": foot_clearance,
        "foot_support": foot_support,
        "flight_avoidance": flight_avoidance,
        "step_span": step_span,
        "step_width": step_width,
        "gait_phase": gait_phase,
        "joint_reference": joint_reference,
        "termination": termination,
    }


def total_reward(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple(components.values()), dim=0).sum(dim=0)


def physics_transition(
    bridge: v1.MJWarpTorchBridge,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    action_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for _ in range(action_repeat):
        qpos, qvel = bridge._forward_raw(qpos, qvel, action)
    return qpos, qvel


def scene_positions(
    bridge: v1.MJWarpTorchBridge,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MJWarp world-space body origins and geom centres."""
    zero_control = torch.zeros(
        (bridge.nworld, bridge.nu),
        dtype=torch.float32,
        device=bridge.torch_device,
    )
    with wp.ScopedDevice(bridge.wp_device):
        bridge._load_inputs(qpos, qvel, zero_control)
        mjw.kinematics(bridge.model, bridge.data_in)
        sync(bridge.torch_device)
        return (
            wp.to_torch(bridge.data_in.xpos).detach().clone(),
            wp.to_torch(bridge.data_in.geom_xpos).detach().clone(),
        )


def body_positions(
    bridge: v1.MJWarpTorchBridge,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
) -> torch.Tensor:
    """Compatibility helper returning only world-space body origins."""
    return scene_positions(bridge, qpos, qvel)[0]


def observation_dim(loaded: v1.LoadedModel, phase_period: int) -> int:
    return (
        loaded.model.nq
        - 2
        + loaded.model.nv
        + loaded.model.nu
        + (2 if phase_period > 0 else 0)
    )


def task_noise(args: argparse.Namespace, *, evaluation: bool) -> dict[str, float]:
    prefix = "eval_" if evaluation else "reset_"
    return {
        "position_noise": float(getattr(args, prefix + "position_noise")),
        "angle_noise": float(getattr(args, prefix + "angle_noise")),
        "joint_noise": float(getattr(args, prefix + "joint_noise")),
        "velocity_noise": float(getattr(args, prefix + "velocity_noise")),
    }


@torch.no_grad()
def evaluate_policy(
    bridge: v1.MJWarpTorchBridge,
    loaded: v1.LoadedModel,
    actor: PPOActor,
    normalizer: RunningMeanStd,
    args: argparse.Namespace,
    *,
    seed: int,
    steps: int,
    noise: bool,
) -> dict[str, Any]:
    generator = torch.Generator(device=bridge.torch_device).manual_seed(seed)
    noise_values = (
        task_noise(args, evaluation=True)
        if noise
        else {
            "position_noise": 0.0,
            "angle_noise": 0.0,
            "joint_noise": 0.0,
            "velocity_noise": 0.0,
        }
    )
    qpos, qvel = sample_initial_states(
        loaded, bridge.nworld, bridge.torch_device, generator, **noise_values
    )
    initial_qpos = qpos.clone()
    previous_action = torch.zeros(
        (bridge.nworld, loaded.model.nu),
        dtype=torch.float32,
        device=bridge.torch_device,
    )
    progress = torch.zeros(bridge.nworld, dtype=torch.long, device=bridge.torch_device)
    alive = healthy(
        loaded.spec.name,
        qpos,
        qvel,
        minimum_height=args.minimum_height,
        maximum_height=args.maximum_height,
        minimum_up=args.minimum_up,
    )
    alive_steps = torch.zeros(
        bridge.nworld, dtype=torch.float32, device=bridge.torch_device
    )
    returns = torch.zeros_like(alive_steps)
    height_sum = torch.zeros_like(alive_steps)
    up_sum = torch.zeros_like(alive_steps)
    heading_sum = torch.zeros_like(alive_steps)
    lateral_velocity_square_sum = torch.zeros_like(alive_steps)
    action_square_sum = torch.zeros_like(alive_steps)
    action_rate_square_sum = torch.zeros_like(alive_steps)
    joint_velocity_square_sum = torch.zeros_like(alive_steps)
    support_foot_slip_square_sum = torch.zeros_like(alive_steps)
    support_foot_samples = torch.zeros_like(alive_steps)
    single_support_sum = torch.zeros_like(alive_steps)
    double_support_sum = torch.zeros_like(alive_steps)
    flight_sum = torch.zeros_like(alive_steps)
    swing_clearance_sum = torch.zeros_like(alive_steps)
    swing_foot_samples = torch.zeros_like(alive_steps)
    step_span_sum = torch.zeros_like(alive_steps)
    alternating_support_switches = torch.zeros_like(alive_steps)
    two_or_more_support_sum = torch.zeros_like(alive_steps)
    diagonal_support_sum = torch.zeros_like(alive_steps)
    alternating_diagonal_support_switches = torch.zeros_like(alive_steps)
    last_single_support = torch.zeros(
        bridge.nworld, dtype=torch.int8, device=bridge.torch_device
    )
    last_diagonal_support = torch.zeros(
        bridge.nworld, dtype=torch.int8, device=bridge.torch_device
    )
    minimum_observed_height = qpos[:, 2].clone()
    minimum_observed_up = root_up(qpos).clone()
    maximum_abs_joint = qpos[:, 7:].abs().amax(dim=-1)
    component_sums: dict[str, torch.Tensor] = {}
    control_dt = float(loaded.model.opt.timestep) * args.action_repeat

    for _ in range(steps):
        active = alive
        current_body_xpos, current_geom_xpos = scene_positions(bridge, qpos, qvel)
        observation = normalizer(
            raw_observation(
                qpos,
                qvel,
                previous_action,
                progress,
                args.phase_period,
                args.phase_warmup_steps,
            )
        )
        action = actor.mean_action(observation)
        qpos_candidate, qvel_candidate = physics_transition(
            bridge, qpos, qvel, action, args.action_repeat
        )
        # Like native MuJoCo, the post-step derived fields can trail the
        # integrated qpos.  Re-run MJWarp kinematics at the returned state so
        # foot displacement spans the complete control interval.
        candidate_body_xpos, candidate_geom_xpos = scene_positions(
            bridge, qpos_candidate, qvel_candidate
        )
        finite = torch.isfinite(qpos_candidate).all(dim=-1) & torch.isfinite(
            qvel_candidate
        ).all(dim=-1)
        safe_qpos = torch.where(finite[:, None], qpos_candidate, qpos)
        safe_qvel = torch.where(finite[:, None], qvel_candidate, qvel)
        safe_body_xpos = torch.where(
            finite[:, None, None], candidate_body_xpos, current_body_xpos
        )
        safe_geom_xpos = torch.where(
            finite[:, None, None], candidate_geom_xpos, current_geom_xpos
        )
        next_alive = active & healthy(
            loaded.spec.name,
            safe_qpos,
            safe_qvel,
            minimum_height=args.minimum_height,
            maximum_height=args.maximum_height,
            minimum_up=args.minimum_up,
        )
        components = reward_components(
            loaded.spec.name,
            qpos,
            qvel,
            action,
            previous_action,
            safe_qpos,
            safe_qvel,
            current_body_xpos,
            safe_body_xpos,
            next_alive,
            progress,
            geom_xpos=current_geom_xpos,
            geom_xpos_next=safe_geom_xpos,
            control_dt=control_dt,
            locomotion_scale=1.0,
            minimum_height=args.minimum_height,
            minimum_up=args.minimum_up,
            termination_penalty=args.termination_penalty,
            survival_weight=args.survival_weight,
            angular_weight=args.angular_weight,
            forward_weight_override=args.forward_weight,
            smoothness_weight_override=args.smoothness_weight,
            lateral_weight_override=args.lateral_weight,
            lateral_position_weight=args.lateral_position_weight,
            foot_slip_weight=args.foot_slip_weight,
            flight_avoidance_weight=args.flight_avoidance_weight,
            gait_shaping_weight=args.gait_shaping_weight,
            gait_phase_weight=args.gait_phase_weight,
            joint_reference_weight=args.joint_reference_weight,
            phase_period=args.phase_period,
            phase_warmup_steps=args.phase_warmup_steps,
            target_speed_override=args.target_speed,
        )
        reward = torch.nan_to_num(total_reward(components), nan=-100.0)
        active_float = active.to(reward.dtype)
        returns += active_float * reward
        alive_steps += active_float
        height_sum += active_float * safe_qpos[:, 2]
        up_sum += active_float * root_up(safe_qpos)
        heading_sum += active_float * root_heading(safe_qpos)
        lateral_velocity_square_sum += active_float * safe_qvel[:, 1].square()
        action_square_sum += active_float * action.square().mean(dim=-1)
        action_rate_square_sum += active_float * (
            action - previous_action
        ).square().mean(dim=-1)
        joint_velocity_square_sum += active_float * safe_qvel[:, 6:].square().mean(
            dim=-1
        )
        if loaded.spec.name in {"humanoid", "ant"}:
            if loaded.spec.name == "humanoid":
                feet = current_body_xpos[:, (7, 10)]
                feet_next = safe_body_xpos[:, (7, 10)]
                support_height = 0.060
                contact_centre_height = 0.027
            else:
                ankle_body_ids = (4, 7, 10, 13)
                ankle_geom_ids = (4, 7, 10, 13)
                feet = (
                    2.0 * current_geom_xpos[:, ankle_geom_ids]
                    - current_body_xpos[:, ankle_body_ids]
                )
                feet_next = (
                    2.0 * safe_geom_xpos[:, ankle_geom_ids]
                    - safe_body_xpos[:, ankle_body_ids]
                )
                # Distal capsule radius is 0.08 m and the colliding pair has
                # 0.02 m combined margin.  The earlier 0.12 m cutoff labeled
                # toes a further 2 cm above the contact envelope as planted,
                # inflating both support and "stance" slip with swing motion.
                support_height = 0.100
                contact_centre_height = 0.080
            foot_velocity = (feet_next - feet) / control_dt
            foot_horizontal_speed_squared = foot_velocity[:, :, :2].square().sum(dim=-1)
            support = feet_next[:, :, 2] <= support_height
            support_float = support.to(reward.dtype)
            support_foot_slip_square_sum += active_float * (
                support_float * foot_horizontal_speed_squared
            ).sum(dim=-1)
            support_foot_samples += active_float * support_float.sum(dim=-1)
            support_count = support.sum(dim=-1)
            single_support = support_count == 1
            double_support = support_count == 2
            flight = support_count == 0
            single_support_sum += active_float * single_support.to(reward.dtype)
            double_support_sum += active_float * double_support.to(reward.dtype)
            flight_sum += active_float * flight.to(reward.dtype)
            two_or_more_support_sum += active_float * (support_count >= 2).to(
                reward.dtype
            )
            swing = ~support
            swing_float = swing.to(reward.dtype)
            swing_clearance_sum += active_float * (
                swing_float
                * (feet_next[:, :, 2] - contact_centre_height).clamp_min(0.0)
            ).sum(dim=-1)
            swing_foot_samples += active_float * swing_float.sum(dim=-1)
            if loaded.spec.name == "humanoid":
                step_span_sum += (
                    active_float * (feet_next[:, 0, 0] - feet_next[:, 1, 0]).abs()
                )
                dominant_support = torch.where(
                    support[:, 0] & ~support[:, 1],
                    torch.ones_like(last_single_support),
                    torch.where(
                        support[:, 1] & ~support[:, 0],
                        -torch.ones_like(last_single_support),
                        torch.zeros_like(last_single_support),
                    ),
                )
                switched = (
                    active
                    & single_support
                    & (last_single_support != 0)
                    & (dominant_support != last_single_support)
                )
                alternating_support_switches += switched.to(reward.dtype)
                last_single_support = torch.where(
                    active & single_support, dominant_support, last_single_support
                )
            else:
                step_span_sum += active_float * (
                    feet_next[:, :, 0].amax(dim=-1)
                    - feet_next[:, :, 0].amin(dim=-1)
                )
                pair_a_support = support[:, 0] & support[:, 2]
                pair_b_support = support[:, 1] & support[:, 3]
                diagonal_support = (support_count == 2) & (
                    pair_a_support | pair_b_support
                )
                diagonal_support_sum += active_float * diagonal_support.to(
                    reward.dtype
                )
                dominant_diagonal = torch.where(
                    pair_a_support & ~pair_b_support,
                    torch.ones_like(last_diagonal_support),
                    torch.where(
                        pair_b_support & ~pair_a_support,
                        -torch.ones_like(last_diagonal_support),
                        torch.zeros_like(last_diagonal_support),
                    ),
                )
                switched = (
                    active
                    & diagonal_support
                    & (last_diagonal_support != 0)
                    & (dominant_diagonal != last_diagonal_support)
                )
                alternating_diagonal_support_switches += switched.to(reward.dtype)
                last_diagonal_support = torch.where(
                    active & diagonal_support,
                    dominant_diagonal,
                    last_diagonal_support,
                )
        for name, value in components.items():
            component_sums.setdefault(name, torch.zeros_like(returns))
            component_sums[name] += active_float * value
        minimum_observed_height = torch.minimum(
            minimum_observed_height,
            torch.where(active, safe_qpos[:, 2], minimum_observed_height),
        )
        minimum_observed_up = torch.minimum(
            minimum_observed_up,
            torch.where(active, root_up(safe_qpos), minimum_observed_up),
        )
        maximum_abs_joint = torch.maximum(
            maximum_abs_joint,
            torch.where(
                active,
                safe_qpos[:, 7:].abs().amax(dim=-1),
                maximum_abs_joint,
            ),
        )
        qpos = torch.where(active[:, None], safe_qpos, qpos)
        qvel = torch.where(active[:, None], safe_qvel, qvel)
        previous_action = torch.where(active[:, None], action, previous_action)
        progress += active.long()
        alive = next_alive

    denominator = alive_steps.clamp_min(1.0)
    displacement = qpos[:, 0] - initial_qpos[:, 0]
    lateral_displacement = qpos[:, 1] - initial_qpos[:, 1]
    survival_fraction = alive_steps / float(steps)
    simulated_seconds = steps * control_dt
    metrics = {
        "seed": seed,
        "worlds": bridge.nworld,
        "steps": steps,
        "control_dt": control_dt,
        "simulated_seconds": simulated_seconds,
        "noise": noise,
        "mean_return": float(returns.mean().item()),
        "return_std": float(returns.std(unbiased=False).item()),
        "mean_displacement": float(displacement.mean().item()),
        "displacement_std": float(displacement.std(unbiased=False).item()),
        "mean_forward_speed_over_horizon": float(
            (displacement / simulated_seconds).mean().item()
        ),
        "mean_abs_lateral_displacement": float(
            lateral_displacement.abs().mean().item()
        ),
        "final_alive_fraction": float(alive.float().mean().item()),
        "mean_survival_fraction": float(survival_fraction.mean().item()),
        "mean_survival_seconds": float(
            (survival_fraction * simulated_seconds).mean().item()
        ),
        "mean_height_while_alive": float((height_sum / denominator).mean().item()),
        "mean_up_while_alive": float((up_sum / denominator).mean().item()),
        "mean_heading_while_alive": float((heading_sum / denominator).mean().item()),
        "mean_minimum_height": float(minimum_observed_height.mean().item()),
        "mean_minimum_up": float(minimum_observed_up.mean().item()),
        "mean_maximum_abs_joint": float(maximum_abs_joint.mean().item()),
        "mean_action_rms": float(
            (action_square_sum / denominator).mean().sqrt().item()
        ),
        "mean_action_rate_rms": float(
            (action_rate_square_sum / denominator).mean().sqrt().item()
        ),
        "mean_lateral_velocity_rms": float(
            (lateral_velocity_square_sum / denominator).mean().sqrt().item()
        ),
        "mean_joint_velocity_rms": float(
            (joint_velocity_square_sum / denominator).mean().sqrt().item()
        ),
        "component_means": {
            name: float((value / denominator).mean().item())
            for name, value in component_sums.items()
        },
    }
    if loaded.spec.name == "humanoid":
        metrics.update(
            {
                "mean_support_foot_slip_rms": float(
                    (support_foot_slip_square_sum / support_foot_samples.clamp_min(1.0))
                    .sqrt()
                    .mean()
                    .item()
                ),
                "mean_single_support_fraction": float(
                    (single_support_sum / denominator).mean().item()
                ),
                "mean_double_support_fraction": float(
                    (double_support_sum / denominator).mean().item()
                ),
                "mean_flight_fraction": float((flight_sum / denominator).mean().item()),
                "mean_swing_foot_clearance": float(
                    (swing_clearance_sum / swing_foot_samples.clamp_min(1.0))
                    .mean()
                    .item()
                ),
                "mean_step_span": float((step_span_sum / denominator).mean().item()),
                "mean_alternating_support_switches": float(
                    alternating_support_switches.mean().item()
                ),
                "mean_alternating_support_switches_per_second": float(
                    (alternating_support_switches / simulated_seconds).mean().item()
                ),
            }
        )
    elif loaded.spec.name == "ant":
        metrics.update(
            {
                "mean_support_foot_slip_rms": float(
                    (support_foot_slip_square_sum / support_foot_samples.clamp_min(1.0))
                    .sqrt()
                    .mean()
                    .item()
                ),
                "mean_two_or_more_support_fraction": float(
                    (two_or_more_support_sum / denominator).mean().item()
                ),
                "mean_diagonal_support_fraction": float(
                    (diagonal_support_sum / denominator).mean().item()
                ),
                "mean_double_support_fraction": float(
                    (double_support_sum / denominator).mean().item()
                ),
                "mean_flight_fraction": float((flight_sum / denominator).mean().item()),
                "mean_swing_foot_clearance": float(
                    (swing_clearance_sum / swing_foot_samples.clamp_min(1.0))
                    .mean()
                    .item()
                ),
                "mean_foot_span": float((step_span_sum / denominator).mean().item()),
                "mean_alternating_diagonal_support_switches": float(
                    alternating_diagonal_support_switches.mean().item()
                ),
                "mean_alternating_diagonal_support_switches_per_second": float(
                    (
                        alternating_diagonal_support_switches / simulated_seconds
                    ).mean().item()
                ),
            }
        )
    metrics["gate"] = gait_gate(loaded.spec.name, metrics)
    metrics["selection_score"] = selection_score(metrics)
    return metrics


def gait_gate(task: str, metrics: dict[str, Any]) -> dict[str, Any]:
    thresholds = (
        {
            "final_alive_fraction": 0.98,
            # Match the independently enforced final-alive fraction.  The
            # former 0.99 mean-survival floor redundantly rejected controllers
            # with 98--99% full-horizon success even when every locomotion and
            # contact check passed.
            "mean_survival_fraction": 0.980,
            "mean_forward_speed_over_horizon": 0.75,
            "mean_up_while_alive": 0.90,
            "mean_heading_while_alive": 0.75,
            "maximum_mean_abs_lateral_displacement": 1.50,
            "maximum_mean_action_rate_rms": 0.35,
            "maximum_mean_support_foot_slip_rms": 1.10,
            "minimum_mean_two_or_more_support_fraction": 0.05,
            "minimum_mean_diagonal_support_fraction": 0.05,
            "minimum_alternating_diagonal_support_switches_per_second": 0.60,
            "maximum_mean_flight_fraction": 0.60,
        }
        if task == "ant"
        else {
            "final_alive_fraction": 0.95,
            "mean_survival_fraction": 0.98,
            "mean_forward_speed_over_horizon": 1.0,
            "mean_up_while_alive": 0.85,
            "mean_heading_while_alive": 0.75,
            "maximum_mean_abs_lateral_displacement": 1.0,
            "maximum_mean_action_rate_rms": 0.35,
            "maximum_mean_support_foot_slip_rms": 0.30,
            "minimum_mean_single_support_fraction": 0.10,
            "minimum_alternating_support_switches_per_second": 0.80,
            "maximum_mean_flight_fraction": 0.25,
        }
    )
    checks = {
        "final_alive_fraction": metrics["final_alive_fraction"]
        >= thresholds["final_alive_fraction"],
        "mean_survival_fraction": metrics["mean_survival_fraction"]
        >= thresholds["mean_survival_fraction"],
        "mean_forward_speed_over_horizon": metrics["mean_forward_speed_over_horizon"]
        >= thresholds["mean_forward_speed_over_horizon"],
        "mean_up_while_alive": metrics["mean_up_while_alive"]
        >= thresholds["mean_up_while_alive"],
        "mean_heading_while_alive": metrics["mean_heading_while_alive"]
        >= thresholds["mean_heading_while_alive"],
        "mean_abs_lateral_displacement": metrics["mean_abs_lateral_displacement"]
        <= thresholds["maximum_mean_abs_lateral_displacement"],
        "mean_action_rate_rms": metrics["mean_action_rate_rms"]
        <= thresholds["maximum_mean_action_rate_rms"],
    }
    if task == "humanoid":
        checks.update(
            {
                "mean_support_foot_slip_rms": metrics["mean_support_foot_slip_rms"]
                <= thresholds["maximum_mean_support_foot_slip_rms"],
                "mean_single_support_fraction": metrics["mean_single_support_fraction"]
                >= thresholds["minimum_mean_single_support_fraction"],
                "mean_alternating_support_switches_per_second": metrics[
                    "mean_alternating_support_switches_per_second"
                ]
                >= thresholds["minimum_alternating_support_switches_per_second"],
                "mean_flight_fraction": metrics["mean_flight_fraction"]
                <= thresholds["maximum_mean_flight_fraction"],
            }
        )
    else:
        checks.update(
            {
                "mean_support_foot_slip_rms": metrics["mean_support_foot_slip_rms"]
                <= thresholds["maximum_mean_support_foot_slip_rms"],
                "mean_two_or_more_support_fraction": metrics[
                    "mean_two_or_more_support_fraction"
                ]
                >= thresholds["minimum_mean_two_or_more_support_fraction"],
                "mean_diagonal_support_fraction": metrics[
                    "mean_diagonal_support_fraction"
                ]
                >= thresholds["minimum_mean_diagonal_support_fraction"],
                "mean_alternating_diagonal_support_switches_per_second": metrics[
                    "mean_alternating_diagonal_support_switches_per_second"
                ]
                >= thresholds[
                    "minimum_alternating_diagonal_support_switches_per_second"
                ],
                "mean_flight_fraction": metrics["mean_flight_fraction"]
                <= thresholds["maximum_mean_flight_fraction"],
            }
        )
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
    }


def selection_score(metrics: dict[str, Any]) -> float:
    score = float(
        10_000.0 * metrics["final_alive_fraction"]
        + 8_000.0 * metrics["mean_survival_fraction"]
        + 500.0 * metrics["mean_forward_speed_over_horizon"]
        + 300.0 * metrics["mean_up_while_alive"]
        + 150.0 * metrics["mean_heading_while_alive"]
        - 500.0 * metrics["mean_abs_lateral_displacement"]
        - 100.0 * metrics["mean_action_rate_rms"]
    )
    if "mean_alternating_support_switches_per_second" in metrics:
        score += (
            500.0 * min(metrics["mean_alternating_support_switches_per_second"], 3.0)
            + 250.0 * metrics["mean_single_support_fraction"]
            - 1_000.0 * metrics["mean_support_foot_slip_rms"]
            - 250.0 * metrics["mean_flight_fraction"]
        )
    elif "mean_diagonal_support_fraction" in metrics:
        score += (
            600.0
            * min(
                metrics["mean_alternating_diagonal_support_switches_per_second"],
                3.0,
            )
            + 500.0 * metrics["mean_diagonal_support_fraction"]
            + 250.0 * metrics["mean_two_or_more_support_fraction"]
            - 750.0 * metrics["mean_support_foot_slip_rms"]
            - 500.0 * metrics["mean_flight_fraction"]
        )
    return score


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def restore_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(state)


def make_networks(
    loaded: v1.LoadedModel,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PPOActor, PPOCritic, RunningMeanStd]:
    obs_dim = observation_dim(loaded, args.phase_period)
    action_low = loaded.model.actuator_ctrlrange[:, 0].astype(np.float32)
    action_high = loaded.model.actuator_ctrlrange[:, 1].astype(np.float32)
    actor = PPOActor(
        obs_dim,
        action_low,
        action_high,
        args.actor_hidden,
        initial_log_std=args.initial_log_std,
    ).to(device)
    critic = PPOCritic(obs_dim, args.critic_hidden).to(device)
    normalizer = RunningMeanStd(obs_dim, device).to(device)
    return actor, critic, normalizer


def compact_evaluation(metrics: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(metrics)


def save_interim_best(
    path: Path,
    loaded: v1.LoadedModel,
    args: argparse.Namespace,
    actor: dict[str, torch.Tensor],
    critic: dict[str, torch.Tensor],
    normalizer: dict[str, torch.Tensor],
    *,
    update: int,
    score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mjwarp-pr1535-full-gait-v3",
            "task": loaded.spec.name,
            "pr_head": git_head(args.pr_root),
            "newton_head": git_head(args.newton_root),
            "observation_dim": observation_dim(loaded, args.phase_period),
            "action_dim": loaded.model.nu,
            "actor_hidden": list(args.actor_hidden),
            "critic_hidden": list(args.critic_hidden),
            "config": jsonable_args(args),
            "best_actor": actor,
            "best_critic": critic,
            "best_normalizer": normalizer,
            "best_update": update,
            "best_score": score,
        },
        path,
    )


def make_checkpoint(
    loaded: v1.LoadedModel,
    args: argparse.Namespace,
    actor: PPOActor,
    critic: PPOCritic,
    normalizer: RunningMeanStd,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    initial_actor: dict[str, torch.Tensor],
    initial_normalizer: dict[str, torch.Tensor],
    best_actor: dict[str, torch.Tensor],
    best_critic: dict[str, torch.Tensor],
    best_normalizer: dict[str, torch.Tensor],
    best_update: int,
    best_score: float,
) -> dict[str, Any]:
    return {
        "format": "mjwarp-pr1535-full-gait-v3",
        "algorithm": "PPO anchor for differentiable-physics gait training",
        "task": loaded.spec.name,
        "xml": str(loaded.xml_path),
        "pr_head": git_head(args.pr_root),
        "newton_head": git_head(args.newton_root),
        "observation_dim": observation_dim(loaded, args.phase_period),
        "action_dim": loaded.model.nu,
        "actor_hidden": list(args.actor_hidden),
        "critic_hidden": list(args.critic_hidden),
        "config": jsonable_args(args),
        "initial_actor": initial_actor,
        "initial_normalizer": initial_normalizer,
        "best_actor": best_actor,
        "best_critic": best_critic,
        "best_normalizer": best_normalizer,
        "best_update": best_update,
        "best_score": best_score,
        "final_actor": clone_state(actor),
        "final_critic": clone_state(critic),
        "final_normalizer": clone_state(normalizer),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
    }


def load_with_appended_observations(
    module: nn.Module, state: dict[str, torch.Tensor]
) -> bool:
    """Load a checkpoint, permitting only appended observation features.

    Phase sine/cosine are appended to the raw observation.  A phase-free
    checkpoint can therefore be a behavior-preserving warm start: new input
    columns are exactly zero for the networks, while normalizer entries start
    at their ordinary zero-mean/unit-variance defaults.  Any other shape
    mismatch remains an error.
    """
    target = module.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    expanded = False
    for name, target_value in target.items():
        if name not in state:
            raise ValueError(f"checkpoint is missing {name}")
        source_value = state[name]
        if source_value.shape == target_value.shape:
            adapted[name] = source_value
            continue
        can_append_matrix_columns = (
            source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] < target_value.shape[1]
        )
        can_append_vector_values = (
            source_value.ndim == 1
            and target_value.ndim == 1
            and source_value.shape[0] < target_value.shape[0]
            and name in {"mean", "variance"}
        )
        if can_append_matrix_columns:
            value = target_value.detach().clone()
            value[:, : source_value.shape[1]] = source_value
        elif can_append_vector_values:
            value = target_value.detach().clone()
            value[: source_value.shape[0]] = source_value
        else:
            raise ValueError(
                f"unsupported checkpoint shape for {name}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
        adapted[name] = value
        expanded = True
    module.load_state_dict(adapted)
    return expanded


def load_resume(
    path: Path,
    actor: PPOActor,
    critic: PPOCritic,
    normalizer: RunningMeanStd,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    policy: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=next(actor.parameters()).device)
    if checkpoint.get("format") != "mjwarp-pr1535-full-gait-v3":
        raise ValueError(f"unsupported resume checkpoint: {path}")
    observation_expanded = load_with_appended_observations(
        actor, checkpoint[f"{policy}_actor"]
    )
    observation_expanded |= load_with_appended_observations(
        normalizer, checkpoint[f"{policy}_normalizer"]
    )
    critic_key = f"{policy}_critic"
    if critic_key in checkpoint:
        observation_expanded |= load_with_appended_observations(
            critic, checkpoint[critic_key]
        )
    elif "best_critic" in checkpoint:
        observation_expanded |= load_with_appended_observations(
            critic, checkpoint["best_critic"]
        )
    if policy == "final":
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    checkpoint["observation_expanded_on_resume"] = observation_expanded
    return checkpoint


def provenance(args: argparse.Namespace, loaded: v1.LoadedModel) -> dict[str, Any]:
    script = Path(__file__).resolve()
    bridge = SCRIPT_DIR / "mjwarp_torch_bridge.py"
    base = Path(v1.__file__).resolve()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(script),
        "script_sha256": sha256_file(script),
        "bridge_sha256": sha256_file(bridge),
        "base_harness_sha256": sha256_file(base),
        "xml_sha256": sha256_file(loaded.xml_path),
        "mjwarp_pr_head": git_head(args.pr_root),
        "newton_head": git_head(args.newton_root),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "mujoco_warp": getattr(mjw, "__version__", None),
            "warp": wp.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "device": str(args.device),
        "method_notes": [
            "PPO transitions are evaluated by the exact PR #1535 MJWarp out-of-place step.",
            "Only qpos and qvel are carried between raw steps; solver warm-start state is reset by the bridge contract.",
            "Selection and holdout evaluations are deterministic and uninterrupted; dead lanes freeze and are never reset.",
            "Forward reward is posture-gated to reject falling or diving as locomotion.",
            "Humanoid posture uses the model's 1.282 m standing reference, not the erroneous 0.84 m v2 target.",
            "Humanoid stance-foot slip and gait metrics use world-space body positions computed by MJWarp kinematics.",
            "A capped target-speed reward removes the incentive for arbitrarily fast stance-foot sliding.",
            "Phase-conditioned Humanoid training uses symmetric foot-contact and leg-angle references; the actor remains a state-feedback policy.",
            "A smooth at-least-one-foot support penalty closes the airborne loophole created by stance-slip regularization.",
            "A configurable startup balance window holds the gait clock at double support before locomotion begins.",
            "Ant toe endpoints are derived exactly from distal capsule centres and inner-endpoint body origins.",
            "Ant physicality gates measure stance slip, flight, two-foot support, and alternating diagonal support.",
            "Ant phase references include the XML-derived fore-aft hip sweep; an ankle-only reference was diagnosed as a stationary local optimum.",
            "Training-only episode ages are independently staggered after every reset so timeouts do not create synchronized nonstationary batches.",
            "PPO bootstraps the pre-reset value at time-limit truncations; physical falls and invalid states remain true terminals.",
            "Optional resumed-policy action anchoring and frozen observation statistics preserve a learned contact cycle during speed fine-tuning.",
        ],
    }


def train(
    loaded: v1.LoadedModel,
    bridge: v1.MJWarpTorchBridge,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = bridge.torch_device
    actor, critic, normalizer = make_networks(loaded, args, device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=1.0e-5)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr, eps=1.0e-5
    )
    resumed_from = None
    if args.resume is not None:
        resumed = load_resume(
            args.resume,
            actor,
            critic,
            normalizer,
            actor_optimizer,
            critic_optimizer,
            policy=args.resume_policy,
        )
        if resumed.get("task") != loaded.spec.name:
            raise ValueError("resume checkpoint task mismatch")
        resumed_from = {
            "path": str(args.resume),
            "sha256": sha256_file(args.resume),
            "policy": args.resume_policy,
            "observation_expanded": resumed["observation_expanded_on_resume"],
        }

    anchor_actor = copy.deepcopy(actor).eval()
    for parameter in anchor_actor.parameters():
        parameter.requires_grad_(False)

    initial_actor = clone_state(actor)
    initial_normalizer = clone_state(normalizer)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    qpos, qvel = sample_initial_states(
        loaded,
        bridge.nworld,
        device,
        generator,
        **task_noise(args, evaluation=False),
    )
    previous_action = torch.zeros(
        (bridge.nworld, loaded.model.nu), dtype=torch.float32, device=device
    )
    progress = torch.zeros(bridge.nworld, dtype=torch.long, device=device)
    episode_age = torch.randint(
        0,
        args.episode_steps,
        (bridge.nworld,),
        generator=generator,
        device=device,
    )
    best_actor = clone_state(actor)
    best_critic = clone_state(critic)
    best_normalizer = clone_state(normalizer)
    initial_evaluation = evaluate_policy(
        bridge,
        loaded,
        actor,
        normalizer,
        args,
        seed=args.seed + 10_000,
        steps=args.eval_steps,
        noise=True,
    )
    best_score = float(initial_evaluation["selection_score"])
    best_update = 0
    evaluations = [{"update": 0, "metrics": initial_evaluation}]
    history: list[dict[str, Any]] = []
    live_path = args.output.with_suffix(".live.json")
    interim_best_path = args.output.with_suffix(".best.pt")
    save_interim_best(
        interim_best_path,
        loaded,
        args,
        best_actor,
        best_critic,
        best_normalizer,
        update=best_update,
        score=best_score,
    )
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for update in range(1, args.updates + 1):
        update_start = time.perf_counter()
        fraction = (update - 1) / max(args.updates - 1, 1)
        learning_rate_scale = max(0.05, 1.0 - fraction)
        actor_optimizer.param_groups[0]["lr"] = args.actor_lr * learning_rate_scale
        critic_optimizer.param_groups[0]["lr"] = args.critic_lr * learning_rate_scale
        if args.curriculum_fraction <= 0.0:
            locomotion_scale = 1.0
        else:
            curriculum_updates = max(1, round(args.updates * args.curriculum_fraction))
            locomotion_scale = min(
                1.0,
                args.initial_locomotion_scale
                + (1.0 - args.initial_locomotion_scale) * update / curriculum_updates,
            )

        observations: list[torch.Tensor] = []
        raw_actions: list[torch.Tensor] = []
        old_log_probabilities: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        environment_rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        component_accumulator: dict[str, torch.Tensor] = {}
        fall_count = 0
        timeout_count = 0
        invalid_count = 0

        for _ in range(args.rollout_steps):
            with torch.no_grad():
                raw_obs = raw_observation(
                    qpos,
                    qvel,
                    previous_action,
                    progress,
                    args.phase_period,
                    args.phase_warmup_steps,
                )
                if not args.freeze_normalizer:
                    normalizer.update(raw_obs)
                observation = normalizer(raw_obs)
                raw_action, action, log_probability = actor.sample(observation)
                value = critic(observation)
                current_body_xpos, current_geom_xpos = scene_positions(
                    bridge, qpos, qvel
                )
                qpos_next, qvel_next = physics_transition(
                    bridge, qpos, qvel, action, args.action_repeat
                )
                next_body_xpos, next_geom_xpos = scene_positions(
                    bridge, qpos_next, qvel_next
                )
                finite = torch.isfinite(qpos_next).all(dim=-1) & torch.isfinite(
                    qvel_next
                ).all(dim=-1)
                safe_qpos = torch.where(finite[:, None], qpos_next, qpos)
                safe_qvel = torch.where(finite[:, None], qvel_next, qvel)
                safe_body_xpos = torch.where(
                    finite[:, None, None], next_body_xpos, current_body_xpos
                )
                safe_geom_xpos = torch.where(
                    finite[:, None, None], next_geom_xpos, current_geom_xpos
                )
                next_healthy = healthy(
                    loaded.spec.name,
                    safe_qpos,
                    safe_qvel,
                    minimum_height=args.minimum_height,
                    maximum_height=args.maximum_height,
                    minimum_up=args.minimum_up,
                )
                fell = finite & ~next_healthy
                invalid = ~finite
                components = reward_components(
                    loaded.spec.name,
                    qpos,
                    qvel,
                    action,
                    previous_action,
                    safe_qpos,
                    safe_qvel,
                    current_body_xpos,
                    safe_body_xpos,
                    next_healthy,
                    progress,
                    geom_xpos=current_geom_xpos,
                    geom_xpos_next=safe_geom_xpos,
                    control_dt=float(loaded.model.opt.timestep) * args.action_repeat,
                    locomotion_scale=locomotion_scale,
                    minimum_height=args.minimum_height,
                    minimum_up=args.minimum_up,
                    termination_penalty=args.termination_penalty,
                    survival_weight=args.survival_weight,
                    angular_weight=args.angular_weight,
                    forward_weight_override=args.forward_weight,
                    smoothness_weight_override=args.smoothness_weight,
                    lateral_weight_override=args.lateral_weight,
                    lateral_position_weight=args.lateral_position_weight,
                    foot_slip_weight=args.foot_slip_weight,
                    flight_avoidance_weight=args.flight_avoidance_weight,
                    gait_shaping_weight=args.gait_shaping_weight,
                    gait_phase_weight=args.gait_phase_weight,
                    joint_reference_weight=args.joint_reference_weight,
                    phase_period=args.phase_period,
                    phase_warmup_steps=args.phase_warmup_steps,
                    target_speed_override=args.target_speed,
                )
                reward = torch.nan_to_num(
                    total_reward(components), nan=-100.0, posinf=-100.0, neginf=-100.0
                )
                reward = torch.where(invalid, torch.full_like(reward, -100.0), reward)
                next_progress = progress + 1
                next_episode_age = episode_age + 1
                timeout = next_episode_age >= args.episode_steps
                done = invalid | fell | timeout
                truncated = timeout & ~invalid & ~fell

                # A time limit is a training segmentation boundary, not a
                # physical terminal state.  Bootstrap its final pre-reset
                # state, then cut the GAE trace at the reset.  Falls and
                # invalid states remain unbootstrapped terminals.
                learning_reward = reward
                if truncated.any():
                    timeout_observation = normalizer(
                        raw_observation(
                            safe_qpos,
                            safe_qvel,
                            action,
                            next_progress,
                            args.phase_period,
                            args.phase_warmup_steps,
                        )
                    )
                    timeout_value = critic(timeout_observation)
                    learning_reward = learning_reward + (
                        args.gamma
                        * truncated.to(learning_reward.dtype)
                        * timeout_value
                    )

                observations.append(observation)
                raw_actions.append(raw_action)
                old_log_probabilities.append(log_probability)
                rewards.append(learning_reward)
                environment_rewards.append(reward)
                dones.append(done.to(torch.float32))
                values.append(value)
                for name, component in components.items():
                    component_accumulator.setdefault(
                        name, torch.zeros((), dtype=torch.float32, device=device)
                    )
                    component_accumulator[name] += component.mean()
                fall_count += int(fell.sum().item())
                timeout_count += int(timeout.sum().item())
                invalid_count += int(invalid.sum().item())

                if done.any():
                    reset_qpos, reset_qvel = sample_initial_states(
                        loaded,
                        bridge.nworld,
                        device,
                        generator,
                        **task_noise(args, evaluation=False),
                    )
                    reset_episode_age = torch.randint(
                        0,
                        args.episode_steps,
                        (bridge.nworld,),
                        generator=generator,
                        device=device,
                    )
                    safe_qpos = torch.where(done[:, None], reset_qpos, safe_qpos)
                    safe_qvel = torch.where(done[:, None], reset_qvel, safe_qvel)
                    action = torch.where(
                        done[:, None], torch.zeros_like(action), action
                    )
                    next_progress = torch.where(
                        done, torch.zeros_like(next_progress), next_progress
                    )
                    next_episode_age = torch.where(
                        done, reset_episode_age, next_episode_age
                    )
                qpos, qvel = safe_qpos, safe_qvel
                previous_action = action
                progress = next_progress
                episode_age = next_episode_age

        with torch.no_grad():
            last_observation = normalizer(
                raw_observation(
                    qpos,
                    qvel,
                    previous_action,
                    progress,
                    args.phase_period,
                    args.phase_warmup_steps,
                )
            )
            last_value = critic(last_observation)
            reward_tensor = torch.stack(rewards)
            environment_reward_tensor = torch.stack(environment_rewards)
            done_tensor = torch.stack(dones)
            value_tensor = torch.stack(values)
            advantages = torch.zeros_like(reward_tensor)
            gae = torch.zeros(bridge.nworld, device=device)
            for step in reversed(range(args.rollout_steps)):
                next_value = (
                    last_value
                    if step == args.rollout_steps - 1
                    else value_tensor[step + 1]
                )
                continuation = 1.0 - done_tensor[step]
                delta = (
                    reward_tensor[step]
                    + args.gamma * continuation * next_value
                    - value_tensor[step]
                )
                gae = delta + args.gamma * args.gae_lambda * continuation * gae
                advantages[step] = gae
            returns = advantages + value_tensor

        obs_flat = torch.stack(observations).reshape(-1, observations[0].shape[-1])
        raw_action_flat = torch.stack(raw_actions).reshape(-1, loaded.model.nu)
        old_log_probability_flat = torch.stack(old_log_probabilities).reshape(-1)
        old_value_flat = value_tensor.reshape(-1)
        return_flat = returns.reshape(-1)
        advantage_flat = advantages.reshape(-1)
        advantage_flat = (advantage_flat - advantage_flat.mean()) / (
            advantage_flat.std(unbiased=False) + 1.0e-6
        )
        sample_count = obs_flat.shape[0]
        batch_size = min(args.minibatch_size, sample_count)
        losses: list[dict[str, float]] = []
        stop_early = False
        for _ in range(args.ppo_epochs):
            order = torch.randperm(sample_count, device=device, generator=generator)
            for start in range(0, sample_count, batch_size):
                indexes = order[start : start + batch_size]
                new_log_probability, entropy = actor.evaluate_raw(
                    obs_flat[indexes], raw_action_flat[indexes]
                )
                log_ratio = new_log_probability - old_log_probability_flat[indexes]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * advantage_flat[indexes],
                    ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    * advantage_flat[indexes],
                ).mean()
                if args.anchor_action_weight > 0.0:
                    with torch.no_grad():
                        anchor_action = anchor_actor.mean_action(obs_flat[indexes])
                    anchor_loss = (
                        actor.mean_action(obs_flat[indexes]) - anchor_action
                    ).square().mean()
                else:
                    anchor_loss = torch.zeros((), device=device)
                actor_loss = (
                    policy_loss
                    - args.entropy_coef * entropy.mean()
                    + args.anchor_action_weight * anchor_loss
                )
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_gradient = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), args.max_grad_norm
                )
                actor_optimizer.step()
                with torch.no_grad():
                    actor.log_std.clamp_(-5.0, 1.0)

                new_value = critic(obs_flat[indexes])
                clipped_value = old_value_flat[indexes] + (
                    new_value - old_value_flat[indexes]
                ).clamp(-args.value_clip_coef, args.value_clip_coef)
                value_loss = (
                    0.5
                    * torch.maximum(
                        (new_value - return_flat[indexes]).square(),
                        (clipped_value - return_flat[indexes]).square(),
                    ).mean()
                )
                critic_optimizer.zero_grad(set_to_none=True)
                (args.value_coef * value_loss).backward()
                critic_gradient = torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), args.max_grad_norm
                )
                critic_optimizer.step()
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                losses.append(
                    {
                        "policy": float(policy_loss.detach().item()),
                        "value": float(value_loss.detach().item()),
                        "entropy": float(entropy.mean().detach().item()),
                        "anchor": float(anchor_loss.detach().item()),
                        "approximate_kl": float(approximate_kl.detach().item()),
                        "actor_gradient": float(actor_gradient.detach().item()),
                        "critic_gradient": float(critic_gradient.detach().item()),
                    }
                )
                if approximate_kl > args.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        sync(device)
        update_seconds = time.perf_counter() - update_start
        loss_mean = {
            key: sum(item[key] for item in losses) / len(losses) for key in losses[0]
        }
        row = {
            "update": update,
            "mean_training_reward": float(environment_reward_tensor.mean().item()),
            "locomotion_scale": locomotion_scale,
            "fall_resets": fall_count,
            "timeout_resets": timeout_count,
            "invalid_resets": invalid_count,
            "actor_lr": actor_optimizer.param_groups[0]["lr"],
            "critic_lr": critic_optimizer.param_groups[0]["lr"],
            "log_std_mean": float(actor.log_std.detach().mean().item()),
            "loss": loss_mean,
            "ppo_early_stop": stop_early,
            "reward_components": {
                name: float((value / args.rollout_steps).item())
                for name, value in component_accumulator.items()
            },
            "seconds": update_seconds,
            "physics_control_samples_per_second": bridge.nworld
            * args.rollout_steps
            / update_seconds,
        }

        should_evaluate = (
            update == 1
            or update == args.updates
            or (args.eval_every > 0 and update % args.eval_every == 0)
        )
        if should_evaluate:
            metrics = evaluate_policy(
                bridge,
                loaded,
                actor,
                normalizer,
                args,
                seed=args.seed + 10_000,
                steps=args.eval_steps,
                noise=True,
            )
            evaluations.append({"update": update, "metrics": metrics})
            row["evaluation"] = compact_evaluation(metrics)
            if args.snapshot_every > 0 and update % args.snapshot_every == 0:
                snapshot_path = args.output.with_name(
                    f"{args.output.stem}.u{update:04d}.pt"
                )
                save_interim_best(
                    snapshot_path,
                    loaded,
                    args,
                    clone_state(actor),
                    clone_state(critic),
                    clone_state(normalizer),
                    update=update,
                    score=float(metrics["selection_score"]),
                )
            if metrics["selection_score"] > best_score:
                best_score = float(metrics["selection_score"])
                best_update = update
                best_actor = clone_state(actor)
                best_critic = clone_state(critic)
                best_normalizer = clone_state(normalizer)
                save_interim_best(
                    interim_best_path,
                    loaded,
                    args,
                    best_actor,
                    best_critic,
                    best_normalizer,
                    update=best_update,
                    score=best_score,
                )
            print(
                f"{loaded.spec.name} update {update:04d}: "
                f"reward={row['mean_training_reward']:.3f} "
                f"alive={metrics['final_alive_fraction']:.3f} "
                f"survival={metrics['mean_survival_fraction']:.3f} "
                f"speed={metrics['mean_forward_speed_over_horizon']:.3f} "
                f"up={metrics['mean_up_while_alive']:.3f} "
                f"score={metrics['selection_score']:.1f} "
                f"gate={metrics['gate']['pass']}",
                flush=True,
            )
        else:
            print(
                f"{loaded.spec.name} update {update:04d}: "
                f"reward={row['mean_training_reward']:.3f} "
                f"falls={fall_count} kl={loss_mean['approximate_kl']:.5f} "
                f"fps={row['physics_control_samples_per_second']:.0f}",
                flush=True,
            )
        history.append(row)
        write_json(
            live_path,
            {
                "schema": "mjwarp-pr1535-full-gait-v3-live",
                "task": loaded.spec.name,
                "seed": args.seed,
                "best_update": best_update,
                "best_score": best_score,
                "history": history,
            },
        )

    final_actor = clone_state(actor)
    final_critic = clone_state(critic)
    final_normalizer = clone_state(normalizer)
    restore_state(actor, best_actor)
    restore_state(critic, best_critic)
    restore_state(normalizer, best_normalizer)
    holdouts = []
    for repeat in range(args.holdout_repeats):
        holdouts.append(
            evaluate_policy(
                bridge,
                loaded,
                actor,
                normalizer,
                args,
                seed=args.seed + 100_000 + repeat,
                steps=args.eval_steps,
                noise=True,
            )
        )
    checkpoint = make_checkpoint(
        loaded,
        args,
        actor,
        critic,
        normalizer,
        actor_optimizer,
        critic_optimizer,
        initial_actor=initial_actor,
        initial_normalizer=initial_normalizer,
        best_actor=best_actor,
        best_critic=best_critic,
        best_normalizer=best_normalizer,
        best_update=best_update,
        best_score=best_score,
    )
    checkpoint["final_actor"] = final_actor
    checkpoint["final_critic"] = final_critic
    checkpoint["final_normalizer"] = final_normalizer
    result = {
        "status": "completed",
        "algorithm": "PPO in exact MJWarp PR #1535 physics",
        "task": loaded.spec.name,
        "seed": args.seed,
        "resumed_from": resumed_from,
        "best_update": best_update,
        "best_selection_score": best_score,
        "initial_evaluation": initial_evaluation,
        "evaluations": evaluations,
        "holdouts": holdouts,
        "holdout_all_gates_pass": all(item["gate"]["pass"] for item in holdouts),
        "training_history": history,
        "timing": {
            "total_seconds": time.perf_counter() - start_time,
            "mean_update_seconds": sum(item["seconds"] for item in history)
            / len(history),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
        },
    }
    return result, checkpoint


def evaluate_checkpoint(
    loaded: v1.LoadedModel,
    bridge: v1.MJWarpTorchBridge,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in evaluate mode")
    checkpoint = torch.load(
        args.checkpoint, map_location=bridge.torch_device, weights_only=False
    )
    if checkpoint.get("format") != "mjwarp-pr1535-full-gait-v3":
        raise ValueError("evaluate mode requires a v3 gait checkpoint")
    if checkpoint.get("task") != loaded.spec.name:
        raise ValueError("checkpoint task mismatch")
    actor, critic, normalizer = make_networks(loaded, args, bridge.torch_device)
    del critic
    actor.load_state_dict(checkpoint[f"{args.checkpoint_policy}_actor"])
    normalizer.load_state_dict(checkpoint[f"{args.checkpoint_policy}_normalizer"])
    metrics = []
    for repeat in range(args.holdout_repeats):
        metrics.append(
            evaluate_policy(
                bridge,
                loaded,
                actor,
                normalizer,
                args,
                seed=args.seed + 100_000 + repeat,
                steps=args.eval_steps,
                noise=not args.nominal_evaluation,
            )
        )
    return {
        "status": "completed",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "policy": args.checkpoint_policy,
        "evaluations": metrics,
        "all_gates_pass": all(item["gate"]["pass"] for item in metrics),
    }


def apply_task_defaults(args: argparse.Namespace) -> None:
    humanoid = args.task == "humanoid"
    defaults = {
        "worlds": 2048,
        "rollout_steps": 64 if humanoid else 32,
        "updates": 800 if humanoid else 400,
        "action_repeat": 5,
        "episode_steps": 600 if humanoid else 400,
        "eval_steps": 400 if humanoid else 200,
        "phase_period": 32 if humanoid else 12,
        "phase_warmup_steps": 16 if humanoid else 8,
        "minimum_height": 0.82 if humanoid else 0.25,
        "maximum_height": 1.75 if humanoid else 1.0,
        "minimum_up": 0.0 if humanoid else 0.1,
        "reset_position_noise": 0.025 if humanoid else 0.05,
        "reset_angle_noise": math.radians(3.0) if humanoid else math.radians(5.0),
        "reset_joint_noise": 0.06 if humanoid else 0.10,
        "reset_velocity_noise": 0.08 if humanoid else 0.10,
        "eval_position_noise": 0.04 if humanoid else 0.08,
        "eval_angle_noise": math.radians(5.0) if humanoid else math.radians(7.5),
        "eval_joint_noise": 0.10 if humanoid else 0.15,
        "eval_velocity_noise": 0.12 if humanoid else 0.15,
        "termination_penalty": 25.0 if humanoid else 15.0,
        "survival_weight": 2.0 if humanoid else 1.0,
        "angular_weight": 0.02 if humanoid else 0.01,
        "forward_weight": 3.0 if humanoid else 2.5,
        "smoothness_weight": 2.0 if humanoid else 0.08,
        "lateral_weight": 1.0 if humanoid else 0.10,
        "lateral_position_weight": 2.0 if humanoid else 0.05,
        "foot_slip_weight": 4.0,
        "flight_avoidance_weight": 4.0,
        "gait_shaping_weight": 1.0,
        "gait_phase_weight": 1.0 if humanoid else 2.0,
        "joint_reference_weight": 0.5,
        "target_speed": 1.5 if humanoid else 2.5,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--task", choices=tuple(v1.TASKS), required=True)
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--pr-root", type=Path, default=DEFAULT_PR_ROOT)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--action-repeat", type=int)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--holdout-repeats", type=int, default=3)
    parser.add_argument("--actor-hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=1.0e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=16384)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--anchor-action-weight", type=float, default=0.0)
    parser.add_argument("--freeze-normalizer", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--initial-log-std", type=float, default=-0.5)
    parser.add_argument("--curriculum-fraction", type=float, default=0.4)
    parser.add_argument("--initial-locomotion-scale", type=float, default=0.1)
    parser.add_argument("--phase-period", type=int)
    parser.add_argument("--phase-warmup-steps", type=int)
    parser.add_argument("--minimum-height", type=float)
    parser.add_argument("--maximum-height", type=float)
    parser.add_argument("--minimum-up", type=float)
    parser.add_argument("--reset-position-noise", type=float)
    parser.add_argument("--reset-angle-noise", type=float)
    parser.add_argument("--reset-joint-noise", type=float)
    parser.add_argument("--reset-velocity-noise", type=float)
    parser.add_argument("--eval-position-noise", type=float)
    parser.add_argument("--eval-angle-noise", type=float)
    parser.add_argument("--eval-joint-noise", type=float)
    parser.add_argument("--eval-velocity-noise", type=float)
    parser.add_argument("--termination-penalty", type=float)
    parser.add_argument("--survival-weight", type=float)
    parser.add_argument("--angular-weight", type=float)
    parser.add_argument("--forward-weight", type=float)
    parser.add_argument("--smoothness-weight", type=float)
    parser.add_argument("--lateral-weight", type=float)
    parser.add_argument("--lateral-position-weight", type=float)
    parser.add_argument("--foot-slip-weight", type=float)
    parser.add_argument("--flight-avoidance-weight", type=float)
    parser.add_argument("--gait-shaping-weight", type=float)
    parser.add_argument("--gait-phase-weight", type=float)
    parser.add_argument("--joint-reference-weight", type=float)
    parser.add_argument("--target-speed", type=float)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-policy", choices=("best", "final"), default="best")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-policy", choices=("initial", "best", "final"), default="best"
    )
    parser.add_argument("--nominal-evaluation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-other-mjw", action="store_true")
    args = parser.parse_args()
    apply_task_defaults(args)
    positive = (
        "worlds",
        "rollout_steps",
        "action_repeat",
        "episode_steps",
        "eval_steps",
        "holdout_repeats",
        "ppo_epochs",
        "minibatch_size",
        "nconmax",
        "njmax",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.mode == "train" and args.updates <= 0:
        parser.error("--updates must be positive in train mode")
    if args.snapshot_every < 0:
        parser.error("--snapshot-every must be nonnegative")
    if args.phase_period < 0 or args.phase_warmup_steps < 0:
        parser.error("--phase-period and --phase-warmup-steps must be nonnegative")
    if args.target_speed <= 0.0:
        parser.error("--target-speed must be positive")
    for name in (
        "smoothness_weight",
        "lateral_weight",
        "lateral_position_weight",
        "foot_slip_weight",
        "flight_avoidance_weight",
        "gait_shaping_weight",
        "gait_phase_weight",
        "joint_reference_weight",
        "survival_weight",
        "angular_weight",
        "anchor_action_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0.0 <= args.curriculum_fraction <= 1.0:
        parser.error("--curriculum-fraction must be in [0, 1]")
    args.pr_root = args.pr_root.expanduser().resolve()
    args.newton_root = args.newton_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    spec = v1.TASKS[args.task]
    if args.xml is not None:
        xml_path = args.xml.expanduser().resolve()
    elif args.task == "humanoid":
        xml_path = (args.pr_root / spec.default_xml).resolve()
    else:
        xml_path = spec.default_xml.resolve()
    imported_root = Path(mjw.__file__).resolve().parent.parent
    if not args.allow_other_mjw and imported_root != args.pr_root:
        raise RuntimeError(
            f"imported mujoco_warp from {imported_root}, expected {args.pr_root}"
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    mjw.enable_grad()
    loaded = v1.load_model(spec, xml_path)
    bridge = v1.make_bridge(loaded, args)
    sync(bridge.torch_device)

    if args.mode == "train":
        result, checkpoint = train(loaded, bridge, args)
        checkpoint_path = args.output.with_suffix(".pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        result["checkpoint"] = str(checkpoint_path)
        result["checkpoint_sha256"] = sha256_file(checkpoint_path)
    else:
        result = evaluate_checkpoint(loaded, bridge, args)
        checkpoint_path = None
    payload = {
        "schema": "mjwarp-pr1535-full-gait-v3",
        "mode": args.mode,
        "config": jsonable_args(args),
        "provenance": provenance(args, loaded),
        "result": result,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote {args.output}")
    if checkpoint_path is not None:
        print(f"Checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
