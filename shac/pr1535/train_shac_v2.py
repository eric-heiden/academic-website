#!/usr/bin/env python3
"""Conditioned SHAC-style training follow-up for MJWarp PR #1535.

This is the corrective experiment paired with ``train_shac.py``.  The v1
harness is intentionally left unchanged so its published artifacts remain
reproducible.  V2 fixes the missing horizon normalization, uses a conditioned
ELU/LayerNorm critic with running observation statistics, restores meaningful
control-step duration and locomotion reward shaping, diversifies rollouts with
reparameterized Gaussian actions, uses minibatched critic updates, and applies
task-specific target-critic retention and learning-rate decay.

It is still a compact SHAC-style experiment rather than a claim of canonical
DiffRL reproduction.  Physics is stepped directly through the narrow
qpos/qvel PyTorch bridge in ``mjwarp_torch_bridge.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
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
import warp as wp
from torch import nn

import train_shac as v1

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PR_ROOT = Path("/home/horde/repos/mujoco_warp-pr1535")
DEFAULT_NEWTON_ROOT = Path("/home/horde/repos/newton-shac-pr1535")


class RunningMeanStd(nn.Module):
    """Numerically stable, detached running observation statistics."""

    def __init__(self, size: int, device: torch.device, *, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.zeros(size, device=device))
        self.register_buffer("variance", torch.ones(size, device=device))
        self.register_buffer("count", torch.tensor(1.0e-4, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if not self.enabled or values.numel() == 0:
            return
        flat = values.detach().reshape(-1, values.shape[-1])
        batch_mean = flat.mean(dim=0)
        batch_variance = flat.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(flat.shape[0]), device=flat.device)
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
        if not self.enabled:
            return values
        return ((values - self.mean) / torch.sqrt(self.variance + 1.0e-6)).clamp(
            -10.0, 10.0
        )


class ConditionedMLP(nn.Module):
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
        for in_dim, out_dim in pairwise(dims):
            layer = nn.Linear(in_dim, out_dim)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            self.norms.append(nn.LayerNorm(out_dim))
        self.output = nn.Linear(dims[-1], output_dim)
        nn.init.orthogonal_(self.output.weight, gain=output_gain)
        nn.init.zeros_(self.output.bias)

    def features(self, values: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.hidden, self.norms, strict=True):
            # Match DiffRL's published order: Linear -> ELU -> LayerNorm.
            values = norm(F.elu(layer(values)))
        return values

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(values))


class StochasticActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: list[int],
        *,
        initial_std: float,
    ) -> None:
        super().__init__()
        self.policy = ConditionedMLP(
            observation_dim, hidden_dims, len(action_low), output_gain=0.01
        )
        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32)
        )
        initial_logstd = math.log(max(initial_std, 1.0e-8))
        self.logstd = nn.Parameter(
            torch.full((len(action_low),), initial_logstd, dtype=torch.float32),
            requires_grad=initial_std > 0.0,
        )
        self.stochastic = initial_std > 0.0

    def _scale(self, unit_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + 0.5 * (unit_action + 1.0) * (
            self.action_high - self.action_low
        )

    def mean_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self._scale(torch.tanh(self.policy(observation)))

    def sample_action(self, observation: torch.Tensor) -> torch.Tensor:
        latent = self.policy(observation)
        if self.stochastic:
            std = self.logstd.clamp(-5.0, 1.0).exp()
            latent = latent + torch.randn_like(latent) * std
        return self._scale(torch.tanh(latent))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mean_action(observation)


class ConditionedCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.value = ConditionedMLP(
            observation_dim, hidden_dims, 1, output_gain=math.sqrt(2.0)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(observation).squeeze(-1)


def raw_observation(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    return torch.cat((qpos[:, 2:], 0.1 * qvel), dim=-1).clamp(-100.0, 100.0)


def root_up(qpos: torch.Tensor) -> torch.Tensor:
    return 1.0 - 2.0 * (qpos[:, 4].square() + qpos[:, 5].square())


def root_heading(qpos: torch.Tensor) -> torch.Tensor:
    # Body x-axis dotted with world x for MuJoCo's w,x,y,z quaternion layout.
    return 1.0 - 2.0 * (qpos[:, 5].square() + qpos[:, 6].square())


def healthy(
    loaded: v1.LoadedModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    reward_profile: str,
) -> torch.Tensor:
    finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
    if reward_profile == "legacy":
        return finite & v1.healthy(loaded.spec, qpos, qvel)
    minimum_height = 0.27 if loaded.spec.name == "ant" else 0.74
    return finite & (qpos[:, 2] >= minimum_height)


def reward_components(
    loaded: v1.LoadedModel,
    reward_profile: str,
    control_dt: float,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    qpos_next: torch.Tensor,
    qvel_next: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if reward_profile == "legacy":
        forward = (
            loaded.spec.forward_weight * (qpos_next[:, 0] - qpos[:, 0]) / control_dt
        )
        upright = loaded.spec.upright_weight * root_up(qpos_next)
        height = (
            -loaded.spec.height_weight
            * (qpos_next[:, 2] - loaded.spec.target_height).square()
        )
        control = -loaded.spec.control_weight * action.square().mean(dim=-1)
        velocity = -loaded.spec.velocity_weight * qvel_next.square().mean(dim=-1)
        alive = torch.full_like(forward, loaded.spec.alive_bonus)
        return {
            "forward": forward,
            "heading": torch.zeros_like(forward),
            "upright": upright,
            "height": height,
            "control": control,
            "velocity": velocity,
            "alive": alive,
        }

    forward = (qpos_next[:, 0] - qpos[:, 0]) / control_dt
    upright = 0.1 * root_up(qpos_next)
    heading = root_heading(qpos_next)
    if loaded.spec.name == "ant":
        height = qpos_next[:, 2] - 0.27
        control = torch.zeros_like(forward)
    else:
        height_delta = (qpos_next[:, 2] - 0.84).clamp(-1.0, 0.1)
        height = torch.where(
            height_delta < 0.0,
            -200.0 * height_delta.square(),
            10.0 * height_delta,
        )
        control = -0.002 * action.square().sum(dim=-1)
    return {
        "forward": forward,
        "heading": heading,
        "upright": upright,
        "height": height,
        "control": control,
        "velocity": torch.zeros_like(forward),
        "alive": torch.zeros_like(forward),
    }


def _reward_total(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(components.values()), dim=0).sum(dim=0)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def sample_initial_states(
    loaded: v1.LoadedModel,
    worlds: int,
    rng: np.random.Generator,
    device: torch.device,
    *,
    noise_profile: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = np.tile(loaded.initial_qpos.astype(np.float32), (worlds, 1))
    qvel = np.zeros((worlds, loaded.model.nv), dtype=np.float32)
    if noise_profile == "narrow":
        qpos[:, 2] += rng.normal(0.0, loaded.spec.height_noise, worlds).astype(
            np.float32
        )
        qpos[:, 7:] += rng.normal(
            0.0, loaded.spec.joint_noise, (worlds, loaded.model.nq - 7)
        ).astype(np.float32)
        qvel += rng.normal(0.0, loaded.spec.velocity_noise, qvel.shape).astype(
            np.float32
        )
    elif noise_profile == "canonical":
        qpos[:, :3] += rng.uniform(-0.1, 0.1, (worlds, 3)).astype(np.float32)
        angles = rng.uniform(-math.radians(7.5), math.radians(7.5), worlds)
        axes = rng.normal(size=(worlds, 3))
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True).clip(1.0e-12)
        perturb = np.empty((worlds, 4), dtype=np.float32)
        perturb[:, 0] = np.cos(0.5 * angles)
        perturb[:, 1:] = axes * np.sin(0.5 * angles)[:, None]
        qpos[:, 3:7] = _quat_multiply(perturb, qpos[:, 3:7])
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
        qpos[:, 7:] += rng.uniform(-0.2, 0.2, (worlds, loaded.model.nq - 7)).astype(
            np.float32
        )
        qvel += rng.uniform(-0.25, 0.25, qvel.shape).astype(np.float32)
    elif noise_profile != "none":
        raise ValueError(f"unknown noise profile {noise_profile!r}")
    return (
        torch.as_tensor(qpos, dtype=torch.float32, device=device),
        torch.as_tensor(qvel, dtype=torch.float32, device=device),
    )


def make_networks(
    loaded: v1.LoadedModel,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[StochasticActor, ConditionedCritic, ConditionedCritic, RunningMeanStd]:
    observation_dim = loaded.model.nq - 2 + loaded.model.nv
    actor_hidden = args.actor_hidden or (
        [128, 64, 32] if loaded.spec.name == "ant" else [256, 128]
    )
    critic_hidden = args.critic_hidden or (
        [64, 64] if loaded.spec.name == "ant" else [128, 128]
    )
    low = loaded.model.actuator_ctrlrange[:, 0].astype(np.float32)
    high = loaded.model.actuator_ctrlrange[:, 1].astype(np.float32)
    actor = StochasticActor(
        observation_dim,
        low,
        high,
        actor_hidden,
        initial_std=args.stochastic_std,
    ).to(device)
    critic = ConditionedCritic(observation_dim, critic_hidden).to(device)
    target = copy.deepcopy(critic).to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    normalizer = RunningMeanStd(
        observation_dim, device, enabled=args.normalize_observations
    ).to(device)
    return actor, critic, target, normalizer


def physics_transition(
    bridge: v1.MJWarpTorchBridge,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    *,
    action_repeat: int,
    differentiable: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    step = bridge.step if differentiable else bridge._forward_raw
    for _ in range(action_repeat):
        qpos, qvel = step(qpos, qvel, action)
    return qpos, qvel


def rollout_actor(
    bridge: v1.MJWarpTorchBridge,
    loaded: v1.LoadedModel,
    actor: StochasticActor,
    target_critic: ConditionedCritic,
    normalizer: RunningMeanStd,
    qpos_start: torch.Tensor,
    qvel_start: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    qpos, qvel = qpos_start, qvel_start
    alive = healthy(loaded, qpos, qvel, args.reward_profile)
    objective = torch.zeros(bridge.nworld, device=bridge.torch_device)
    direct = torch.zeros_like(objective)
    discount = 1.0
    control_dt = float(loaded.model.opt.timestep) * args.action_repeat

    qpos_states = [qpos]
    qvel_states = [qvel]
    rewards: list[torch.Tensor] = []
    alive_before: list[torch.Tensor] = []
    alive_after: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    component_values: dict[str, list[torch.Tensor]] = {}

    for _ in range(args.horizon):
        active = alive
        obs = normalizer(raw_observation(qpos, qvel))
        action = actor.sample_action(obs)
        qpos_next, qvel_next = physics_transition(
            bridge,
            qpos,
            qvel,
            action,
            action_repeat=args.action_repeat,
            differentiable=True,
        )
        if not torch.isfinite(qpos_next).all() or not torch.isfinite(qvel_next).all():
            raise FloatingPointError("non-finite state inside differentiable rollout")
        components = reward_components(
            loaded,
            args.reward_profile,
            control_dt,
            qpos,
            qvel,
            action,
            qpos_next,
            qvel_next,
        )
        step_reward = _reward_total(components)
        next_alive = active & healthy(loaded, qpos_next, qvel_next, args.reward_profile)
        weight = active.to(step_reward.dtype)
        objective = objective + discount * weight * step_reward
        direct = direct + weight * step_reward
        for name, value in components.items():
            component_values.setdefault(name, []).append(weight * value)
        rewards.append(step_reward)
        alive_before.append(active)
        alive_after.append(next_alive)
        actions.append(action)
        qpos_states.append(qpos_next)
        qvel_states.append(qvel_next)
        qpos, qvel, alive = qpos_next, qvel_next, next_alive
        discount *= args.gamma

    terminal_value = target_critic(normalizer(raw_observation(qpos, qvel)))
    terminal_contribution = (
        args.terminal_value_weight
        * discount
        * alive.to(terminal_value.dtype)
        * terminal_value
    )
    objective = objective + terminal_contribution
    return {
        "objective": objective.mean(),
        "direct": direct.mean(),
        "terminal_value": terminal_value.mean(),
        "terminal_contribution": terminal_contribution.mean(),
        "qpos": qpos_states,
        "qvel": qvel_states,
        "rewards": rewards,
        "alive_before": alive_before,
        "alive_after": alive_after,
        "end_alive": alive,
        "actions": actions,
        "components": {
            name: torch.stack(values).sum(dim=0).mean()
            for name, values in component_values.items()
        },
    }


def critic_training_data(
    critic: ConditionedCritic,
    target_critic: ConditionedCritic,
    normalizer: RunningMeanStd,
    rollout: dict[str, Any],
    gamma: float,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del critic
    qpos_states = [state.detach() for state in rollout["qpos"]]
    qvel_states = [state.detach() for state in rollout["qvel"]]
    rewards = [value.detach() for value in rollout["rewards"]]
    alive_before = [value.detach() for value in rollout["alive_before"]]
    alive_after = [value.detach() for value in rollout["alive_after"]]
    with torch.no_grad():
        observations = [
            normalizer(raw_observation(qpos, qvel))
            for qpos, qvel in zip(qpos_states, qvel_states, strict=True)
        ]
        values = [target_critic(obs) for obs in observations]
        running = values[-1] * alive_after[-1].to(values[-1].dtype)
        reversed_targets = []
        for index in reversed(range(len(rewards))):
            continuation = alive_after[index].to(running.dtype)
            bootstrap = (1.0 - lambda_) * values[index + 1] + lambda_ * running
            running = rewards[index] + gamma * continuation * bootstrap
            reversed_targets.append(running)
        targets = torch.stack(list(reversed(reversed_targets)))
        valid = torch.stack(alive_before)
        obs = torch.stack(observations[:-1])
    return obs[valid], targets[valid]


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, retention: float) -> None:
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.mul_(retention)
        target_parameter.add_((1.0 - retention) * source_parameter)


@torch.no_grad()
def evaluate_policy(
    bridge: v1.MJWarpTorchBridge,
    loaded: v1.LoadedModel,
    actor: StochasticActor,
    normalizer: RunningMeanStd,
    args: argparse.Namespace,
    *,
    steps: int,
    seed: int,
    noise_profile: str | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    qpos, qvel = sample_initial_states(
        loaded,
        bridge.nworld,
        rng,
        bridge.torch_device,
        noise_profile=noise_profile or args.eval_noise_profile,
    )
    initial_x = qpos[:, 0].clone()
    returns = torch.zeros(bridge.nworld, device=bridge.torch_device)
    discounted_returns = torch.zeros_like(returns)
    alive = healthy(loaded, qpos, qvel, args.reward_profile)
    alive_fraction_sum = 0.0
    minimum_height = qpos[:, 2].clone()
    action_square_sum = torch.zeros_like(returns)
    action_abs_max = torch.zeros_like(returns)
    discount = 1.0
    control_dt = float(loaded.model.opt.timestep) * args.action_repeat

    for _ in range(steps):
        action = actor(normalizer(raw_observation(qpos, qvel)))
        qpos_candidate, qvel_candidate = physics_transition(
            bridge,
            qpos,
            qvel,
            action,
            action_repeat=args.action_repeat,
            differentiable=False,
        )
        finite = torch.isfinite(qpos_candidate).all(dim=-1) & torch.isfinite(
            qvel_candidate
        ).all(dim=-1)
        components = reward_components(
            loaded,
            args.reward_profile,
            control_dt,
            qpos,
            qvel,
            action,
            qpos_candidate,
            qvel_candidate,
        )
        step_reward = torch.nan_to_num(_reward_total(components))
        active_float = alive.to(step_reward.dtype)
        returns += active_float * step_reward
        discounted_returns += discount * active_float * step_reward
        action_square_sum += active_float * action.square().mean(dim=-1)
        action_abs_max = torch.maximum(action_abs_max, action.abs().amax(dim=-1))
        next_alive = (
            alive
            & finite
            & healthy(loaded, qpos_candidate, qvel_candidate, args.reward_profile)
        )
        safe_qpos = torch.where(finite[:, None], qpos_candidate, qpos)
        safe_qvel = torch.where(finite[:, None], qvel_candidate, qvel)
        qpos = torch.where(alive[:, None], safe_qpos, qpos)
        qvel = torch.where(alive[:, None], safe_qvel, qvel)
        minimum_height = torch.minimum(minimum_height, qpos[:, 2])
        alive = next_alive
        alive_fraction_sum += float(alive.float().mean().item())
        discount *= args.gamma

    displacement = qpos[:, 0] - initial_x
    return {
        "mean_return": float(returns.mean().item()),
        "return_std": float(returns.std(unbiased=False).item()),
        "mean_discounted_return": float(discounted_returns.mean().item()),
        "mean_displacement": float(displacement.mean().item()),
        "displacement_std": float(displacement.std(unbiased=False).item()),
        "final_alive_fraction": float(alive.float().mean().item()),
        "mean_alive_fraction": alive_fraction_sum / steps,
        "mean_minimum_height": float(minimum_height.mean().item()),
        "mean_action_rms": float((action_square_sum / steps).mean().sqrt().item()),
        "max_abs_action": float(action_abs_max.max().item()),
        "worlds": bridge.nworld,
        "steps": steps,
        "control_dt": control_dt,
        "simulated_seconds": steps * control_dt,
        "seed": seed,
        "noise_profile": noise_profile or args.eval_noise_profile,
    }


def _state_gradient_diagnostic(
    target_critic: ConditionedCritic,
    normalizer: RunningMeanStd,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
) -> dict[str, float]:
    obs = raw_observation(qpos.detach(), qvel.detach()).requires_grad_(True)
    value = target_critic(normalizer(obs))
    gradient = torch.autograd.grad(value.mean(), obs)[0]
    hidden = target_critic.value.features(normalizer(obs)).detach()
    return {
        "target_value_mean": float(value.detach().mean().item()),
        "target_value_std": float(value.detach().std(unbiased=False).item()),
        "mean_state_gradient_l2": float(gradient.detach().norm(dim=-1).mean().item()),
        "hidden_rms": float(hidden.square().mean().sqrt().item()),
        "hidden_abs_max": float(hidden.abs().max().item()),
    }


def _metrics_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    initial, final = evaluations[0], evaluations[-1]
    best = max(evaluations, key=lambda item: item["metrics"]["mean_return"])
    return {
        "initial": {"epoch": initial["epoch"], **initial["metrics"]},
        "best": {"epoch": best["epoch"], **best["metrics"]},
        "final": {"epoch": final["epoch"], **final["metrics"]},
    }


def _restore(module: nn.Module, state: dict[str, Any]) -> None:
    module.load_state_dict(state)


def train(
    args: argparse.Namespace,
    loaded: v1.LoadedModel,
    bridge: v1.MJWarpTorchBridge,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actor, critic, target, normalizer = make_networks(loaded, args, bridge.torch_device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    rng = np.random.default_rng(args.seed + 101)
    qpos, qvel = sample_initial_states(
        loaded,
        bridge.nworld,
        rng,
        bridge.torch_device,
        noise_profile=args.train_noise_profile,
    )
    progress = torch.zeros(bridge.nworld, dtype=torch.int64, device=bridge.torch_device)

    initial_actor = copy.deepcopy(actor.state_dict())
    initial_normalizer = copy.deepcopy(normalizer.state_dict())
    evaluations = [
        {
            "epoch": 0,
            "metrics": evaluate_policy(
                bridge,
                loaded,
                actor,
                normalizer,
                args,
                steps=args.eval_steps,
                seed=args.seed + 10_000,
            ),
        }
    ]
    best_actor = copy.deepcopy(initial_actor)
    best_normalizer = copy.deepcopy(initial_normalizer)
    best_epoch = 0
    best_return = evaluations[0]["metrics"]["mean_return"]
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if args.lr_schedule == "linear":
            fraction = (epoch - 1) / max(args.epochs - 1, 1)
            actor_lr = args.actor_lr + fraction * (args.min_lr - args.actor_lr)
            critic_lr = args.critic_lr + fraction * (args.min_lr - args.critic_lr)
            actor_optimizer.param_groups[0]["lr"] = actor_lr
            critic_optimizer.param_groups[0]["lr"] = critic_lr
        else:
            actor_lr, critic_lr = args.actor_lr, args.critic_lr

        normalizer.update(raw_observation(qpos, qvel))
        start_x = qpos[:, 0].clone()
        actor_optimizer.zero_grad(set_to_none=True)
        rollout = rollout_actor(
            bridge, loaded, actor, target, normalizer, qpos, qvel, args
        )
        divisor = args.horizon if args.normalize_actor_loss else 1
        actor_loss = -rollout["objective"] / divisor
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), args.max_grad_norm
        )
        if not torch.isfinite(actor_grad_norm):
            raise FloatingPointError(f"non-finite actor gradient at epoch {epoch}")
        actor_optimizer.step()

        critic_obs, critic_targets = critic_training_data(
            critic, target, normalizer, rollout, args.gamma, args.td_lambda
        )
        sample_count = critic_obs.shape[0]
        batch_size = max(1, math.ceil(sample_count / args.critic_batches))
        critic_losses: list[torch.Tensor] = []
        critic_grad_norm = torch.zeros((), device=bridge.torch_device)
        for _ in range(args.critic_iterations):
            order = torch.randperm(sample_count, device=bridge.torch_device)
            for start in range(0, sample_count, batch_size):
                indexes = order[start : start + batch_size]
                prediction = critic(critic_obs[indexes])
                loss = (prediction - critic_targets[indexes]).square().mean()
                critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), args.max_grad_norm
                )
                if not torch.isfinite(critic_grad_norm):
                    raise FloatingPointError(
                        f"non-finite critic gradient at epoch {epoch}"
                    )
                critic_optimizer.step()
                critic_losses.append(loss.detach())
        polyak_update(target, critic, args.target_polyak)

        qpos_end = rollout["qpos"][-1].detach()
        qvel_end = rollout["qvel"][-1].detach()
        progress = progress + args.horizon
        reset_mask = ~healthy(loaded, qpos_end, qvel_end, args.reward_profile)
        if args.episode_steps > 0:
            reset_mask |= progress >= args.episode_steps
        if args.reset_interval > 0 and epoch % args.reset_interval == 0:
            reset_mask[:] = True
        reset_count = int(reset_mask.sum().item())
        reset_qpos, reset_qvel = sample_initial_states(
            loaded,
            bridge.nworld,
            rng,
            bridge.torch_device,
            noise_profile=args.train_noise_profile,
        )
        qpos = torch.where(reset_mask[:, None], reset_qpos, qpos_end).detach()
        qvel = torch.where(reset_mask[:, None], reset_qvel, qvel_end).detach()
        progress = torch.where(reset_mask, torch.zeros_like(progress), progress)

        state_diag = _state_gradient_diagnostic(target, normalizer, qpos_end, qvel_end)
        actions = torch.stack(rollout["actions"]).detach()
        v1._sync(bridge.torch_device)
        history.append(
            {
                "epoch": epoch,
                "actor_objective": float(rollout["objective"].detach().item()),
                "actor_loss_divisor": divisor,
                "direct_horizon_return": float(rollout["direct"].detach().item()),
                "terminal_value_contribution": float(
                    rollout["terminal_contribution"].detach().item()
                ),
                "reward_components": {
                    name: float(value.detach().item())
                    for name, value in rollout["components"].items()
                },
                "action_rms": float(actions.square().mean().sqrt().item()),
                "action_abs_max": float(actions.abs().max().item()),
                "critic_loss_mean": float(torch.stack(critic_losses).mean().item()),
                "critic_loss_last": float(critic_losses[-1].item()),
                "mean_td_lambda_target": float(critic_targets.mean().item()),
                "actor_grad_norm_before_clip": float(actor_grad_norm.item()),
                "critic_grad_norm_before_clip": float(critic_grad_norm.item()),
                "rollout_end_alive_fraction": float(
                    rollout["end_alive"].float().mean().item()
                ),
                "mean_horizon_displacement": float(
                    (qpos_end[:, 0] - start_x).mean().item()
                ),
                "mean_end_height": float(qpos_end[:, 2].mean().item()),
                "mean_end_up": float(root_up(qpos_end).mean().item()),
                "reset_worlds": reset_count,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "normalizer_count": float(normalizer.count.item()),
                "critic_diagnostic": state_diag,
                "seconds": time.perf_counter() - epoch_start,
            }
        )

        if epoch == args.epochs or (
            args.eval_every > 0 and epoch % args.eval_every == 0
        ):
            evaluation = {
                "epoch": epoch,
                "metrics": evaluate_policy(
                    bridge,
                    loaded,
                    actor,
                    normalizer,
                    args,
                    steps=args.eval_steps,
                    seed=args.seed + 10_000,
                ),
            }
            evaluations.append(evaluation)
            if evaluation["metrics"]["mean_return"] > best_return:
                best_return = evaluation["metrics"]["mean_return"]
                best_epoch = epoch
                best_actor = copy.deepcopy(actor.state_dict())
                best_normalizer = copy.deepcopy(normalizer.state_dict())

    training_seconds = time.perf_counter() - start_time
    final_actor = copy.deepcopy(actor.state_dict())
    final_normalizer = copy.deepcopy(normalizer.state_dict())
    holdout_seed = args.seed + 20_000
    _restore(actor, initial_actor)
    _restore(normalizer, initial_normalizer)
    holdout_initial = evaluate_policy(
        bridge,
        loaded,
        actor,
        normalizer,
        args,
        steps=args.eval_steps,
        seed=holdout_seed,
    )
    _restore(actor, best_actor)
    _restore(normalizer, best_normalizer)
    holdout_best = evaluate_policy(
        bridge,
        loaded,
        actor,
        normalizer,
        args,
        steps=args.eval_steps,
        seed=holdout_seed,
    )
    _restore(actor, final_actor)
    _restore(normalizer, final_normalizer)

    checkpoint = {
        "format": "mjwarp-pr1535-shac-style-v2",
        "task": loaded.spec.name,
        "xml": str(loaded.xml_path),
        "pr_head": v1._git_head(args.pr_root),
        "observation_dim": loaded.model.nq - 2 + loaded.model.nv,
        "action_dim": loaded.model.nu,
        "actor_hidden": args.actor_hidden
        or ([128, 64, 32] if loaded.spec.name == "ant" else [256, 128]),
        "critic_hidden": args.critic_hidden
        or ([64, 64] if loaded.spec.name == "ant" else [128, 128]),
        "stochastic_std": args.stochastic_std,
        "normalize_observations": args.normalize_observations,
        "epoch": args.epochs,
        "initial_actor": initial_actor,
        "best_actor": best_actor,
        "final_actor": final_actor,
        "actor": final_actor,
        "initial_normalizer": initial_normalizer,
        "best_normalizer": best_normalizer,
        "final_normalizer": final_normalizer,
        "normalizer": final_normalizer,
        "best_actor_epoch": best_epoch,
        "best_actor_return": best_return,
        "critic": critic.state_dict(),
        "target_critic": target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "config": v1._jsonable_args(args),
    }
    result = {
        "status": "completed",
        "algorithm": "conditioned SHAC-style qpos/qvel short-horizon actor-critic",
        "canonical_shac": False,
        "cold_start": True,
        "epochs_completed": args.epochs,
        "metrics": _metrics_summary(evaluations),
        "holdout": {
            "selection_independent": True,
            "selected_epoch": best_epoch,
            "initial": holdout_initial,
            "selected_best": holdout_best,
        },
        "evaluations": evaluations,
        "training_history": history,
        "timing_seconds": {
            "training_including_periodic_evaluation": training_seconds,
            "mean_epoch": training_seconds / args.epochs,
        },
    }
    return result, checkpoint


def _checkpoint_networks(
    checkpoint: dict[str, Any],
    loaded: v1.LoadedModel,
    device: torch.device,
) -> tuple[StochasticActor, RunningMeanStd]:
    low = loaded.model.actuator_ctrlrange[:, 0].astype(np.float32)
    high = loaded.model.actuator_ctrlrange[:, 1].astype(np.float32)
    actor = StochasticActor(
        int(checkpoint["observation_dim"]),
        low,
        high,
        list(checkpoint["actor_hidden"]),
        initial_std=float(checkpoint.get("stochastic_std", 0.0)),
    ).to(device)
    normalizer = RunningMeanStd(
        int(checkpoint["observation_dim"]),
        device,
        enabled=bool(checkpoint.get("normalize_observations", True)),
    ).to(device)
    return actor, normalizer


def evaluate_checkpoint(
    args: argparse.Namespace,
    loaded: v1.LoadedModel,
    bridge: v1.MJWarpTorchBridge,
) -> dict[str, Any]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in evaluate mode")
    checkpoint = torch.load(
        args.checkpoint, map_location=bridge.torch_device, weights_only=False
    )
    if checkpoint.get("format") != "mjwarp-pr1535-shac-style-v2":
        raise ValueError("evaluate mode requires a v2 checkpoint")
    if checkpoint.get("task") != loaded.spec.name:
        raise ValueError("checkpoint task does not match --task")
    actor, normalizer = _checkpoint_networks(checkpoint, loaded, bridge.torch_device)
    policy = args.checkpoint_policy
    actor.load_state_dict(checkpoint[f"{policy}_actor"])
    normalizer.load_state_dict(checkpoint[f"{policy}_normalizer"])
    actor.eval()
    metrics = evaluate_policy(
        bridge,
        loaded,
        actor,
        normalizer,
        args,
        steps=args.eval_steps,
        seed=args.seed + 10_000,
    )
    return {
        "status": "completed",
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluated_policy": policy,
        "evaluated_policy_epoch": checkpoint.get("best_actor_epoch")
        if policy == "best"
        else (0 if policy == "initial" else checkpoint.get("epoch")),
        "metrics": metrics,
    }


def provenance(args: argparse.Namespace, loaded: v1.LoadedModel) -> dict[str, Any]:
    result = v1.provenance(args, loaded)
    v1_script = Path(v1.__file__).resolve()
    script = Path(__file__).resolve()
    result["base_harness"] = {
        "path": str(v1_script),
        "sha256": v1._sha256_bytes(v1_script.read_bytes()),
    }
    result["script"] = str(script)
    result["script_sha256"] = v1._sha256_bytes(script.read_bytes())
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    result["method_notes"] = [
        "Cold-start SHAC-style corrective experiment; no pretrained policy is loaded.",
        "The actor loss is divided by control horizon unless explicitly disabled.",
        "A control transition repeats one action for action_repeat raw MJWarp steps and computes reward once over control_dt.",
        "The target retention, reward profile, exploration, observation normalization, and LR schedule are recorded in config.",
        "Only qpos/qvel are carried through the bridge; resets and health masks remain non-differentiable boundaries.",
        "ViewerGL rendering is a separate state-visualization pass and is not part of training.",
    ]
    result["versions"].update(
        {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "warp": wp.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--task", choices=tuple(v1.TASKS), default="ant")
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--pr-root", type=Path, default=DEFAULT_PR_ROOT)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--actor-hidden", type=int, nargs="+")
    parser.add_argument("--critic-hidden", type=int, nargs="+")
    parser.add_argument("--actor-lr", type=float, default=2.0e-3)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--lr-schedule", choices=("linear", "constant"), default="linear"
    )
    parser.add_argument("--adam-beta1", type=float, default=0.7)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--critic-iterations", type=int, default=16)
    parser.add_argument("--critic-batches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--target-polyak", type=float)
    parser.add_argument("--terminal-value-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--normalize-actor-loss", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--normalize-observations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stochastic-std", type=float, default=math.exp(-1.0))
    parser.add_argument(
        "--reward-profile", choices=("diffrl", "legacy"), default="diffrl"
    )
    parser.add_argument(
        "--train-noise-profile",
        choices=("canonical", "narrow", "none"),
        default="canonical",
    )
    parser.add_argument(
        "--eval-noise-profile",
        choices=("canonical", "narrow", "none"),
        default="canonical",
    )
    parser.add_argument("--action-repeat", type=int)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--reset-interval", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-policy", choices=("initial", "best", "final"), default="best"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-other-mjw", action="store_true")
    args = parser.parse_args()
    if args.action_repeat is None:
        args.action_repeat = 2 if args.task == "ant" else 3
    if args.target_polyak is None:
        args.target_polyak = 0.2 if args.task == "ant" else 0.995
    if args.critic_lr is None:
        args.critic_lr = 2.0e-3 if args.task == "ant" else 5.0e-4
    positive = {
        "worlds": args.worlds,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "action_repeat": args.action_repeat,
        "critic_iterations": args.critic_iterations,
        "critic_batches": args.critic_batches,
        "eval_steps": args.eval_steps,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.target_polyak < 1.0:
        parser.error("--target-polyak must be in [0, 1)")
    if args.stochastic_std < 0.0:
        parser.error("--stochastic-std must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    args.pr_root = args.pr_root.expanduser().resolve()
    args.newton_root = args.newton_root.expanduser().resolve()
    spec = v1.TASKS[args.task]
    if args.xml is not None:
        xml_path = args.xml.expanduser().resolve()
    elif args.task == "humanoid":
        xml_path = (args.pr_root / spec.default_xml).resolve()
    else:
        xml_path = spec.default_xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

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
    setup_start = time.perf_counter()
    bridge = v1.make_bridge(loaded, args)
    v1._sync(bridge.torch_device)
    setup_seconds = time.perf_counter() - setup_start
    if args.mode == "train":
        run, checkpoint = train(args, loaded, bridge)
    else:
        run, checkpoint = evaluate_checkpoint(args, loaded, bridge), None

    output = args.output or Path(f"pr1535_shac_v2_{args.task}_{args.mode}.json")
    output = output.expanduser().resolve()
    checkpoint_path = args.checkpoint
    if checkpoint is not None:
        checkpoint_path = checkpoint_path or output.with_suffix(".pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        run["checkpoint"] = str(checkpoint_path.resolve())
    payload = {
        "schema": "mjwarp-pr1535-shac-style-v2",
        "mode": args.mode,
        "config": v1._jsonable_args(args),
        "provenance": provenance(args, loaded),
        "setup_seconds": setup_seconds,
        "run": run,
    }
    v1._write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote {output}")
    if checkpoint_path is not None:
        print(f"Checkpoint {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
