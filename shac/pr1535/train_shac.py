#!/usr/bin/env python3
"""A compact SHAC-style trainer for MJWarp PR #1535.

This is deliberately called *SHAC-style*, not a canonical SHAC reproduction.
It has SHAC's central ingredients: a deterministic actor, a TD(lambda) critic,
a slowly updated target critic, and short-horizon actor gradients propagated
through differentiable physics.  The terminal target-critic value remains
differentiable with respect to the final simulated state.

The important simplification is explicit: the Torch/MJWarp bridge carries only
``qpos`` and ``qvel`` between steps.  Solver ``qacc_warmstart`` is detached and
reset to its template value on every step; both supported models use stateless
motor actuators (``na == 0``).  Resets and healthy-state masks are also treated
as non-differentiable boundaries.

Examples (invoke by path; bundled assets resolve relative to this file):

  python /path/to/pr1535/train_shac.py --mode gradcheck --task ant
  python /path/to/pr1535/train_shac.py --mode train --task ant --epochs 100
  python /path/to/pr1535/train_shac.py --mode evaluate --task ant \
      --checkpoint pr1535_shac_ant_train.pt
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mjwarp_torch_bridge import MJWarpTorchBridge
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PR_ROOT = Path("/home/horde/repos/mujoco_warp-pr1535")
DEFAULT_NEWTON_ROOT = Path("/home/horde/repos/newton-shac-pr1535")
ANT_XML = SCRIPT_DIR / "models/ant.xml"
HUMANOID_XML_RELATIVE = Path("benchmarks/humanoid/humanoid.xml")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    default_xml: Path
    healthy_z: tuple[float, float]
    upright_min: float
    target_height: float
    forward_weight: float
    alive_bonus: float
    upright_weight: float
    height_weight: float
    control_weight: float
    velocity_weight: float
    joint_noise: float
    height_noise: float
    velocity_noise: float


TASKS = {
    "ant": TaskSpec(
        name="ant",
        default_xml=ANT_XML,
        healthy_z=(0.20, 1.00),
        upright_min=-1.01,
        target_height=0.55,
        forward_weight=1.0,
        alive_bonus=1.0,
        upright_weight=0.25,
        height_weight=1.0,
        control_weight=0.05,
        velocity_weight=0.001,
        joint_noise=0.04,
        height_noise=0.015,
        velocity_noise=0.03,
    ),
    "humanoid": TaskSpec(
        name="humanoid",
        default_xml=HUMANOID_XML_RELATIVE,
        healthy_z=(0.75, 1.80),
        upright_min=0.20,
        target_height=1.282,
        forward_weight=1.25,
        alive_bonus=5.0,
        upright_weight=1.0,
        height_weight=2.0,
        control_weight=0.025,
        velocity_weight=0.001,
        joint_noise=0.025,
        height_noise=0.01,
        velocity_noise=0.02,
    ),
}


@dataclass
class LoadedModel:
    spec: TaskSpec
    xml_path: Path
    model: mujoco.MjModel
    data: mujoco.MjData
    initial_qpos: np.ndarray
    initial_contacts: int
    solimp_sha256: str


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden: int, *, output_gain: float
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )
        linear_layers = [module for module in self.net if isinstance(module, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DeterministicActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: int,
    ) -> None:
        super().__init__()
        self.policy = MLP(observation_dim, len(action_low), hidden, output_gain=0.01)
        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        unit_action = torch.tanh(self.policy(observation))
        return self.action_low + 0.5 * (unit_action + 1.0) * (
            self.action_high - self.action_low
        )


class Critic(nn.Module):
    def __init__(self, observation_dim: int, hidden: int) -> None:
        super().__init__()
        self.value = MLP(observation_dim, 1, hidden, output_gain=0.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(observation).squeeze(-1)


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


def _nvidia_driver() -> str | None:
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.splitlines()[0]
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _sync(device: torch.device) -> None:
    wp.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _initial_qpos(model: mujoco.MjModel, task: str) -> np.ndarray:
    if task == "ant":
        numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "init_qpos")
        if numeric_id >= 0 and model.numeric_size[numeric_id] >= model.nq:
            start = model.numeric_adr[numeric_id]
            return model.numeric_data[start : start + model.nq].copy()
    return model.qpos0.copy()


def load_model(spec: TaskSpec, xml_path: Path) -> LoadedModel:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.na != 0:
        raise ValueError(
            f"{spec.name} has na={model.na}; this qpos/qvel-only bridge supports stateless actuators only"
        )

    solimp_before = model.geom_solimp.copy()
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    if not np.array_equal(solimp_before, model.geom_solimp):
        raise AssertionError("Model setup unexpectedly changed geom_solimp")

    initial_qpos = _initial_qpos(model, spec.name)
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    return LoadedModel(
        spec=spec,
        xml_path=xml_path,
        model=model,
        data=data,
        initial_qpos=initial_qpos,
        initial_contacts=int(data.ncon),
        solimp_sha256=_sha256_bytes(model.geom_solimp.tobytes()),
    )


def observation(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    """Translation-invariant state with a fixed velocity scale."""
    return torch.cat((qpos[:, 2:], 0.1 * qvel), dim=-1).clamp(-10.0, 10.0)


def root_up(qpos: torch.Tensor) -> torch.Tensor:
    # MuJoCo free-joint quaternion layout is w, x, y, z.  This is the body-z /
    # world-z dot product, equal to one for the nominal upright orientation.
    return 1.0 - 2.0 * (qpos[:, 4].square() + qpos[:, 5].square())


def healthy(spec: TaskSpec, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
    height_ok = (qpos[:, 2] > spec.healthy_z[0]) & (qpos[:, 2] < spec.healthy_z[1])
    return finite & height_ok & (root_up(qpos) > spec.upright_min)


def reward(
    spec: TaskSpec,
    timestep: float,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    qpos_next: torch.Tensor,
    qvel_next: torch.Tensor,
) -> torch.Tensor:
    del qvel
    forward_velocity = (qpos_next[:, 0] - qpos[:, 0]) / timestep
    height_error = qpos_next[:, 2] - spec.target_height
    return (
        spec.forward_weight * forward_velocity
        + spec.alive_bonus
        + spec.upright_weight * root_up(qpos_next)
        - spec.height_weight * height_error.square()
        - spec.control_weight * action.square().mean(dim=-1)
        - spec.velocity_weight * qvel_next.square().mean(dim=-1)
    )


def sample_initial_states(
    loaded: LoadedModel,
    worlds: int,
    rng: np.random.Generator,
    device: torch.device,
    *,
    noisy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = np.tile(loaded.initial_qpos.astype(np.float32), (worlds, 1))
    qvel = np.zeros((worlds, loaded.model.nv), dtype=np.float32)
    if noisy:
        qpos[:, 2] += rng.normal(0.0, loaded.spec.height_noise, worlds).astype(
            np.float32
        )
        if loaded.model.nq > 7:
            qpos[:, 7:] += rng.normal(
                0.0,
                loaded.spec.joint_noise,
                (worlds, loaded.model.nq - 7),
            ).astype(np.float32)
        qvel += rng.normal(
            0.0,
            loaded.spec.velocity_noise,
            qvel.shape,
        ).astype(np.float32)
    return (
        torch.as_tensor(qpos, dtype=torch.float32, device=device),
        torch.as_tensor(qvel, dtype=torch.float32, device=device),
    )


def make_bridge(loaded: LoadedModel, args: argparse.Namespace) -> MJWarpTorchBridge:
    return MJWarpTorchBridge(
        loaded.model,
        loaded.data,
        nworld=args.worlds,
        device=args.device,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )


def make_networks(
    loaded: LoadedModel,
    hidden: int,
    device: torch.device,
) -> tuple[DeterministicActor, Critic, Critic]:
    observation_dim = loaded.model.nq - 2 + loaded.model.nv
    low = loaded.model.actuator_ctrlrange[:, 0].astype(np.float32)
    high = loaded.model.actuator_ctrlrange[:, 1].astype(np.float32)
    actor = DeterministicActor(observation_dim, low, high, hidden).to(device)
    critic = Critic(observation_dim, hidden).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    for parameter in target_critic.parameters():
        parameter.requires_grad_(False)
    target_critic.eval()
    return actor, critic, target_critic


def rollout_actor(
    bridge: MJWarpTorchBridge,
    loaded: LoadedModel,
    actor: DeterministicActor,
    target_critic: Critic,
    qpos_start: torch.Tensor,
    qvel_start: torch.Tensor,
    horizon: int,
    gamma: float,
) -> dict[str, Any]:
    qpos = qpos_start
    qvel = qvel_start
    alive = healthy(loaded.spec, qpos, qvel)
    objective = torch.zeros(
        bridge.nworld, dtype=torch.float32, device=bridge.torch_device
    )
    undiscounted = torch.zeros_like(objective)
    discount = 1.0

    qpos_states = [qpos]
    qvel_states = [qvel]
    rewards = []
    alive_before = []
    alive_after = []

    for _ in range(horizon):
        active = alive
        action = actor(observation(qpos, qvel))
        qpos_next, qvel_next = bridge.step(qpos, qvel, action)
        if not torch.isfinite(qpos_next).all() or not torch.isfinite(qvel_next).all():
            raise FloatingPointError("Non-finite state inside differentiable rollout")
        step_reward = reward(
            loaded.spec,
            float(loaded.model.opt.timestep),
            qpos,
            qvel,
            action,
            qpos_next,
            qvel_next,
        )
        next_alive = active & healthy(loaded.spec, qpos_next, qvel_next)
        weight = active.to(step_reward.dtype)
        objective = objective + discount * weight * step_reward
        undiscounted = undiscounted + weight * step_reward

        rewards.append(step_reward)
        alive_before.append(active)
        alive_after.append(next_alive)
        qpos_states.append(qpos_next)
        qvel_states.append(qvel_next)
        qpos, qvel, alive = qpos_next, qvel_next, next_alive
        discount *= gamma

    terminal_value = target_critic(observation(qpos, qvel))
    objective = objective + discount * alive.to(terminal_value.dtype) * terminal_value
    return {
        "objective": objective.mean(),
        "undiscounted": undiscounted.mean(),
        "qpos": qpos_states,
        "qvel": qvel_states,
        "rewards": rewards,
        "alive_before": alive_before,
        "alive_after": alive_after,
        "end_alive": alive,
    }


def critic_td_lambda_loss(
    critic: Critic,
    target_critic: Critic,
    rollout: dict[str, Any],
    gamma: float,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qpos_states = [state.detach() for state in rollout["qpos"]]
    qvel_states = [state.detach() for state in rollout["qvel"]]
    rewards = [value.detach() for value in rollout["rewards"]]
    alive_before = [value.detach() for value in rollout["alive_before"]]
    alive_after = [value.detach() for value in rollout["alive_after"]]

    with torch.no_grad():
        target_values = [
            target_critic(observation(qpos, qvel))
            for qpos, qvel in zip(qpos_states, qvel_states, strict=True)
        ]
        running = target_values[-1] * alive_after[-1].to(target_values[-1].dtype)
        targets_reversed = []
        for index in reversed(range(len(rewards))):
            continuation = alive_after[index].to(running.dtype)
            mixed_bootstrap = (1.0 - lambda_) * target_values[
                index + 1
            ] + lambda_ * running
            running = rewards[index] + gamma * continuation * mixed_bootstrap
            targets_reversed.append(running)
        targets = torch.stack(list(reversed(targets_reversed)))

    predictions = torch.stack(
        [
            critic(observation(qpos, qvel))
            for qpos, qvel in zip(qpos_states[:-1], qvel_states[:-1], strict=True)
        ]
    )
    valid = torch.stack(alive_before).to(predictions.dtype)
    squared_error = (predictions - targets).square()
    loss = (valid * squared_error).sum() / valid.sum().clamp_min(1.0)
    return loss, targets.mean()


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, coefficient: float) -> None:
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.lerp_(source_parameter, 1.0 - coefficient)


@torch.no_grad()
def evaluate_policy(
    bridge: MJWarpTorchBridge,
    loaded: LoadedModel,
    actor: DeterministicActor,
    *,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    qpos, qvel = sample_initial_states(
        loaded,
        bridge.nworld,
        rng,
        bridge.torch_device,
        noisy=True,
    )
    initial_x = qpos[:, 0].clone()
    returns = torch.zeros(bridge.nworld, device=bridge.torch_device)
    alive = healthy(loaded.spec, qpos, qvel)
    alive_fraction_sum = 0.0

    for _ in range(steps):
        action = actor(observation(qpos, qvel))
        qpos_candidate, qvel_candidate = bridge._forward_raw(qpos, qvel, action)
        candidate_finite = torch.isfinite(qpos_candidate).all(dim=-1) & torch.isfinite(
            qvel_candidate
        ).all(dim=-1)
        step_reward = reward(
            loaded.spec,
            float(loaded.model.opt.timestep),
            qpos,
            qvel,
            action,
            qpos_candidate,
            qvel_candidate,
        )
        step_reward = torch.nan_to_num(step_reward, nan=0.0, posinf=0.0, neginf=0.0)
        returns += alive.to(step_reward.dtype) * step_reward
        next_alive = (
            alive
            & candidate_finite
            & healthy(loaded.spec, qpos_candidate, qvel_candidate)
        )

        # Freeze a lane after its first terminal state.  A newly dead lane keeps
        # that terminal state if finite, so displacement remains meaningful.
        safe_qpos = torch.where(candidate_finite[:, None], qpos_candidate, qpos)
        safe_qvel = torch.where(candidate_finite[:, None], qvel_candidate, qvel)
        qpos = torch.where(alive[:, None], safe_qpos, qpos)
        qvel = torch.where(alive[:, None], safe_qvel, qvel)
        alive = next_alive
        alive_fraction_sum += float(alive.float().mean().item())

    displacement = qpos[:, 0] - initial_x
    return {
        "mean_return": float(returns.mean().item()),
        "return_std": float(returns.std(unbiased=False).item()),
        "mean_displacement": float(displacement.mean().item()),
        "displacement_std": float(displacement.std(unbiased=False).item()),
        "final_alive_fraction": float(alive.float().mean().item()),
        "mean_alive_fraction": alive_fraction_sum / steps,
        "worlds": bridge.nworld,
        "steps": steps,
        "seed": seed,
    }


def _reset_unhealthy(
    loaded: LoadedModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    rng: np.random.Generator,
    *,
    force_all: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    reset_qpos, reset_qvel = sample_initial_states(
        loaded,
        qpos.shape[0],
        rng,
        qpos.device,
        noisy=True,
    )
    reset_mask = ~healthy(loaded.spec, qpos, qvel)
    if force_all:
        reset_mask = torch.ones_like(reset_mask)
    count = int(reset_mask.sum().item())
    qpos = torch.where(reset_mask[:, None], reset_qpos, qpos).detach()
    qvel = torch.where(reset_mask[:, None], reset_qvel, qvel).detach()
    return qpos, qvel, count


def _metrics_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    initial = evaluations[0]
    final = evaluations[-1]
    best = max(evaluations, key=lambda entry: entry["metrics"]["mean_return"])
    best_return = max(evaluations, key=lambda entry: entry["metrics"]["mean_return"])
    best_displacement = max(
        evaluations, key=lambda entry: entry["metrics"]["mean_displacement"]
    )
    best_alive = max(
        evaluations, key=lambda entry: entry["metrics"]["final_alive_fraction"]
    )
    return {
        "initial": {"epoch": initial["epoch"], **initial["metrics"]},
        "final": {"epoch": final["epoch"], **final["metrics"]},
        "best": {"epoch": best["epoch"], **best["metrics"]},
        "headline": {
            "initial_return": initial["metrics"]["mean_return"],
            "final_return": final["metrics"]["mean_return"],
            "best_return": best_return["metrics"]["mean_return"],
            "best_return_epoch": best_return["epoch"],
            "initial_displacement": initial["metrics"]["mean_displacement"],
            "final_displacement": final["metrics"]["mean_displacement"],
            "best_displacement": best_displacement["metrics"]["mean_displacement"],
            "best_displacement_epoch": best_displacement["epoch"],
            "initial_alive_fraction": initial["metrics"]["final_alive_fraction"],
            "final_alive_fraction": final["metrics"]["final_alive_fraction"],
            "best_alive_fraction": best_alive["metrics"]["final_alive_fraction"],
            "best_alive_fraction_epoch": best_alive["epoch"],
        },
    }


def train(
    args: argparse.Namespace,
    loaded: LoadedModel,
    bridge: MJWarpTorchBridge,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actor, critic, target_critic = make_networks(
        loaded, args.hidden, bridge.torch_device
    )
    adam_betas = (args.adam_beta1, args.adam_beta2)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=args.actor_lr, betas=adam_betas
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr, betas=adam_betas
    )
    train_rng = np.random.default_rng(args.seed + 101)
    qpos, qvel = sample_initial_states(
        loaded,
        bridge.nworld,
        train_rng,
        bridge.torch_device,
        noisy=True,
    )

    evaluations = [
        {
            "epoch": 0,
            "metrics": evaluate_policy(
                bridge,
                loaded,
                actor,
                steps=args.eval_steps,
                seed=args.seed + 10_000,
            ),
        }
    ]
    initial_actor_state = copy.deepcopy(actor.state_dict())
    best_actor_state = copy.deepcopy(actor.state_dict())
    best_actor_epoch = 0
    best_actor_return = evaluations[0]["metrics"]["mean_return"]
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        start_x = qpos[:, 0].clone()
        actor_optimizer.zero_grad(set_to_none=True)
        rollout = rollout_actor(
            bridge,
            loaded,
            actor,
            target_critic,
            qpos,
            qvel,
            args.horizon,
            args.gamma,
        )
        actor_loss = -rollout["objective"]
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), args.max_grad_norm
        )
        if not torch.isfinite(actor_grad_norm):
            raise FloatingPointError(f"Non-finite actor gradient at epoch {epoch}")
        actor_optimizer.step()

        critic_losses = []
        critic_grad_norm = torch.zeros((), device=bridge.torch_device)
        mean_td_target = torch.zeros((), device=bridge.torch_device)
        for _ in range(args.critic_iterations):
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss, mean_td_target = critic_td_lambda_loss(
                critic,
                target_critic,
                rollout,
                args.gamma,
                args.td_lambda,
            )
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(critic_grad_norm):
                raise FloatingPointError(f"Non-finite critic gradient at epoch {epoch}")
            critic_optimizer.step()
            critic_losses.append(critic_loss.detach())
        polyak_update(target_critic, critic, args.target_polyak)

        qpos_end = rollout["qpos"][-1].detach()
        qvel_end = rollout["qvel"][-1].detach()
        force_all = args.reset_interval > 0 and epoch % args.reset_interval == 0
        qpos, qvel, reset_count = _reset_unhealthy(
            loaded,
            qpos_end,
            qvel_end,
            train_rng,
            force_all=force_all,
        )
        _sync(bridge.torch_device)
        history.append(
            {
                "epoch": epoch,
                "actor_objective": float(rollout["objective"].detach().item()),
                "short_horizon_return": float(rollout["undiscounted"].detach().item()),
                "critic_loss": float(torch.stack(critic_losses).mean().item()),
                "critic_loss_last": float(critic_losses[-1].item()),
                "mean_td_lambda_target": float(mean_td_target.detach().item()),
                "actor_grad_norm_before_clip": float(actor_grad_norm.detach().item()),
                "critic_grad_norm_before_clip": float(critic_grad_norm.detach().item()),
                "rollout_end_alive_fraction": float(
                    rollout["end_alive"].float().mean().detach().item()
                ),
                "mean_horizon_displacement": float(
                    (qpos_end[:, 0] - start_x).mean().item()
                ),
                "reset_worlds": reset_count,
                "seconds": time.perf_counter() - epoch_start,
            }
        )

        should_evaluate = epoch == args.epochs or (
            args.eval_every > 0 and epoch % args.eval_every == 0
        )
        if should_evaluate:
            evaluation = {
                "epoch": epoch,
                "metrics": evaluate_policy(
                    bridge,
                    loaded,
                    actor,
                    steps=args.eval_steps,
                    seed=args.seed + 10_000,
                ),
            }
            evaluations.append(evaluation)
            if evaluation["metrics"]["mean_return"] > best_actor_return:
                best_actor_return = evaluation["metrics"]["mean_return"]
                best_actor_epoch = epoch
                best_actor_state = copy.deepcopy(actor.state_dict())

    training_seconds = time.perf_counter() - training_start
    final_actor_state = copy.deepcopy(actor.state_dict())
    holdout_seed = args.seed + 20_000
    actor.load_state_dict(initial_actor_state)
    holdout_initial = evaluate_policy(
        bridge,
        loaded,
        actor,
        steps=args.eval_steps,
        seed=holdout_seed,
    )
    actor.load_state_dict(best_actor_state)
    holdout_best = evaluate_policy(
        bridge,
        loaded,
        actor,
        steps=args.eval_steps,
        seed=holdout_seed,
    )
    actor.load_state_dict(final_actor_state)
    checkpoint = {
        "format": "mjwarp-pr1535-shac-style-v1",
        "task": loaded.spec.name,
        "xml": str(loaded.xml_path),
        "pr_head": _git_head(args.pr_root),
        "hidden": args.hidden,
        "observation_dim": loaded.model.nq - 2 + loaded.model.nv,
        "action_dim": loaded.model.nu,
        "epoch": args.epochs,
        "actor": actor.state_dict(),
        "best_actor": best_actor_state,
        "best_actor_epoch": best_actor_epoch,
        "best_actor_return": best_actor_return,
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "config": _jsonable_args(args),
    }
    result = {
        "status": "completed",
        "algorithm": "SHAC-style qpos/qvel-state short-horizon actor-critic",
        "canonical_shac": False,
        "cold_start": True,
        "epochs_completed": args.epochs,
        "timing_seconds": {
            "training_including_periodic_evaluation": training_seconds,
            "mean_epoch": training_seconds / max(args.epochs, 1),
        },
        "metrics": _metrics_summary(evaluations),
        "holdout": {
            "selection_independent": True,
            "selected_epoch": best_actor_epoch,
            "initial": holdout_initial,
            "selected_best": holdout_best,
        },
        "evaluations": evaluations,
        "training_history": history,
    }
    return result, checkpoint


def _schedule_objective(
    bridge: MJWarpTorchBridge,
    loaded: LoadedModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    actions: torch.Tensor,
    *,
    differentiable: bool,
    gamma: float,
    contact_trace: list[list[int]] | None = None,
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32, device=bridge.torch_device)
    discount = 1.0
    for index in range(actions.shape[0]):
        action = actions[index]
        if differentiable:
            qpos_next, qvel_next = bridge.step(qpos, qvel, action)
        else:
            qpos_next, qvel_next = bridge._forward_raw(qpos, qvel, action)
        if contact_trace is not None:
            contact_trace.append(
                bridge.data_out.nacon.numpy().astype(np.int64).tolist()
            )
        step_score = reward(
            loaded.spec,
            float(loaded.model.opt.timestep),
            qpos,
            qvel,
            action,
            qpos_next,
            qvel_next,
        )
        # Removing constants leaves the derivative unchanged while avoiding
        # float32 cancellation in the central-difference subtraction.
        step_score = step_score - loaded.spec.alive_bonus - loaded.spec.upright_weight
        total = total + discount * step_score.mean()
        qpos, qvel = qpos_next, qvel_next
        discount *= gamma
    return total


def _directional_gradcheck(
    bridge: MJWarpTorchBridge,
    loaded: LoadedModel,
    args: argparse.Namespace,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    qpos, qvel = sample_initial_states(
        loaded,
        bridge.nworld,
        rng,
        bridge.torch_device,
        noisy=False,
    )
    low = loaded.model.actuator_ctrlrange[:, 0].astype(np.float32)
    high = loaded.model.actuator_ctrlrange[:, 1].astype(np.float32)
    center = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    actions_np = (
        center
        + rng.uniform(
            -0.2,
            0.2,
            (horizon, bridge.nworld, loaded.model.nu),
        ).astype(np.float32)
        * half_range
    )
    actions = torch.tensor(
        actions_np,
        dtype=torch.float32,
        device=bridge.torch_device,
        requires_grad=True,
    )

    base_contact_trace: list[list[int]] = []
    objective = _schedule_objective(
        bridge,
        loaded,
        qpos,
        qvel,
        actions,
        differentiable=True,
        gamma=args.gamma,
        contact_trace=base_contact_trace,
    )
    objective.backward()
    _sync(bridge.torch_device)
    gradient = actions.grad.detach()
    gradient_norm = float(torch.linalg.vector_norm(gradient).item())
    comparisons = []

    for direction_index in range(args.gradcheck_directions):
        direction_np = rng.normal(size=actions_np.shape).astype(np.float32)
        direction_np /= max(float(np.linalg.norm(direction_np)), 1.0e-12)
        direction = torch.as_tensor(
            direction_np, dtype=torch.float32, device=bridge.torch_device
        )
        analytic_directional = float((gradient * direction).sum().item())
        plus_contact_trace: list[list[int]] = []
        minus_contact_trace: list[list[int]] = []
        with torch.no_grad():
            plus = actions.detach() + args.gradcheck_eps * direction
            minus = actions.detach() - args.gradcheck_eps * direction
            plus_value = _schedule_objective(
                bridge,
                loaded,
                qpos,
                qvel,
                plus,
                differentiable=False,
                gamma=args.gamma,
                contact_trace=plus_contact_trace,
            )
            minus_value = _schedule_objective(
                bridge,
                loaded,
                qpos,
                qvel,
                minus,
                differentiable=False,
                gamma=args.gamma,
                contact_trace=minus_contact_trace,
            )
            finite_difference = float(
                ((plus_value - minus_value) / (2.0 * args.gradcheck_eps)).item()
            )
        absolute_error = abs(analytic_directional - finite_difference)
        relative_error = absolute_error / max(
            abs(analytic_directional), abs(finite_difference), 1.0e-7
        )
        contact_count_mismatch_steps = sum(
            plus != minus or plus != base
            for base, plus, minus in zip(
                base_contact_trace,
                plus_contact_trace,
                minus_contact_trace,
                strict=True,
            )
        )
        comparisons.append(
            {
                "direction": direction_index,
                "analytic": analytic_directional,
                "finite_difference": finite_difference,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "same_sign": analytic_directional * finite_difference >= 0.0,
                "contact_count_mismatch_steps": contact_count_mismatch_steps,
                "plus_contact_count_trace": plus_contact_trace,
                "minus_contact_count_trace": minus_contact_trace,
            }
        )

    max_relative = max(item["relative_error"] for item in comparisons)
    return {
        "horizon": horizon,
        "objective": float(objective.detach().item()),
        "gradient_l2_norm": gradient_norm,
        "epsilon": args.gradcheck_eps,
        "base_contact_count_trace": base_contact_trace,
        "comparisons": comparisons,
        "max_relative_error": max_relative,
        "all_same_sign": all(item["same_sign"] for item in comparisons),
        "pass": bool(
            math.isfinite(gradient_norm)
            and gradient_norm > 0.0
            and max_relative <= args.gradcheck_max_relative
            and all(item["same_sign"] for item in comparisons)
        ),
    }


def gradcheck(
    args: argparse.Namespace,
    loaded: LoadedModel,
    bridge: MJWarpTorchBridge,
) -> dict[str, Any]:
    start = time.perf_counter()
    horizons = args.gradcheck_horizons
    if horizons is None:
        horizons = [1, args.horizon if args.horizon > 1 else 4]
    horizons = list(dict.fromkeys(horizons))
    checks = [
        _directional_gradcheck(bridge, loaded, args, horizon, args.seed + horizon)
        for horizon in horizons
    ]
    return {
        "status": "pass" if all(check["pass"] for check in checks) else "fail",
        "algorithm": "MJWarp action-schedule derivative check",
        "checks": checks,
        "timing_seconds": {"total": time.perf_counter() - start},
    }


def evaluate_checkpoint(
    args: argparse.Namespace,
    loaded: LoadedModel,
    bridge: MJWarpTorchBridge,
) -> dict[str, Any]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --mode evaluate")
    checkpoint = torch.load(
        args.checkpoint, map_location=bridge.torch_device, weights_only=False
    )
    if checkpoint.get("task") != loaded.spec.name:
        raise ValueError(
            f"Checkpoint task {checkpoint.get('task')!r} does not match --task {loaded.spec.name!r}"
        )
    hidden = int(checkpoint["hidden"])
    actor, _, _ = make_networks(loaded, hidden, bridge.torch_device)
    actor_key = (
        "best_actor"
        if args.checkpoint_policy == "best" and "best_actor" in checkpoint
        else "actor"
    )
    actor.load_state_dict(checkpoint[actor_key])
    actor.eval()
    start = time.perf_counter()
    metrics = evaluate_policy(
        bridge,
        loaded,
        actor,
        steps=args.eval_steps,
        seed=args.seed + 10_000,
    )
    return {
        "status": "completed",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluated_policy": actor_key,
        "evaluated_policy_epoch": checkpoint.get("best_actor_epoch")
        if actor_key == "best_actor"
        else checkpoint.get("epoch"),
        "metrics": metrics,
        "timing_seconds": {"evaluation": time.perf_counter() - start},
    }


def provenance(args: argparse.Namespace, loaded: LoadedModel) -> dict[str, Any]:
    imported_root = Path(mjw.__file__).resolve().parent.parent
    script_path = Path(__file__).resolve()
    bridge_path = Path(__file__).with_name("mjwarp_torch_bridge.py").resolve()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pr": {
            "url": "https://github.com/google-deepmind/mujoco_warp/pull/1535",
            "head": _git_head(args.pr_root),
            "import_path": str(Path(mjw.__file__).resolve()),
            "expected_worktree": str(args.pr_root),
            "exact_worktree_import": imported_root == args.pr_root.resolve(),
        },
        "newton_head": _git_head(args.newton_root),
        "script": str(script_path),
        "script_sha256": _sha256_bytes(script_path.read_bytes()),
        "bridge": str(bridge_path),
        "bridge_sha256": _sha256_bytes(bridge_path.read_bytes()),
        "versions": {
            "python": platform.python_version(),
            "mujoco_warp": getattr(mjw, "__version__", "unknown"),
            "mujoco": mujoco.__version__,
            "warp": wp.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "device": {
            "warp": args.device,
            "torch": args.device,
            "name": torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else "CPU",
            "nvidia_driver": _nvidia_driver(),
        },
        "model": {
            "task": loaded.spec.name,
            "xml": str(loaded.xml_path),
            "xml_sha256": _sha256_bytes(loaded.xml_path.read_bytes()),
            "nq": loaded.model.nq,
            "nv": loaded.model.nv,
            "nu": loaded.model.nu,
            "na": loaded.model.na,
            "initial_contacts": loaded.initial_contacts,
            "timestep": float(loaded.model.opt.timestep),
            "integrator": mujoco.mjtIntegrator(loaded.model.opt.integrator).name,
            "solver": mujoco.mjtSolver(loaded.model.opt.solver).name,
            "cone": mujoco.mjtCone(loaded.model.opt.cone).name,
            "eulerdamp_disabled": bool(
                loaded.model.opt.disableflags
                & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
            ),
            "geom_solimp_sha256": loaded.solimp_sha256,
            "geom_solimp_modified_by_harness": False,
        },
        "method_notes": [
            "Cold start: no pretrained actor or critic is loaded in train mode.",
            "Actor objective is a discounted short differentiable rollout plus a differentiable terminal target-critic value.",
            "Critic targets use backward-view TD(lambda) recursion with target-critic bootstraps.",
            "Only qpos and qvel are carried through Torch; qacc_warmstart is detached at each physics step.",
            "Unhealthy-world resets and alive masks are non-differentiable boundaries between short rollouts.",
            "The policy is deterministic; seeds are fixed, but GPU contact atomics are not claimed bitwise deterministic.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("gradcheck", "train", "evaluate"), default="train"
    )
    parser.add_argument("--task", choices=tuple(TASKS), default="ant")
    parser.add_argument("--xml", type=Path)
    parser.add_argument(
        "--pr-root",
        type=Path,
        default=DEFAULT_PR_ROOT,
        help="MJWarp PR #1535 worktree; supplies Humanoid and exact-head provenance.",
    )
    parser.add_argument(
        "--newton-root",
        type=Path,
        default=DEFAULT_NEWTON_ROOT,
        help="Newton worktree used only for provenance.",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=1.0e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.7)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--critic-iterations", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--target-polyak", type=float, default=0.995)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--gradcheck-eps", type=float, default=1.0e-2)
    parser.add_argument("--gradcheck-directions", type=int, default=3)
    parser.add_argument("--gradcheck-horizons", type=int, nargs="+")
    parser.add_argument("--gradcheck-max-relative", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-policy", choices=("best", "final"), default="best"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-other-mjw",
        action="store_true",
        help="Allow a mujoco_warp import outside the exact checked-out PR worktree.",
    )
    args = parser.parse_args()

    if args.worlds < 1 or args.horizon < 1:
        parser.error("--worlds and --horizon must be positive")
    if args.epochs < 1 and args.mode == "train":
        parser.error("--epochs must be positive in train mode")
    if args.critic_iterations < 1:
        parser.error("--critic-iterations must be positive")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        parser.error("Adam betas must be in [0, 1)")
    if args.eval_steps < 1:
        parser.error("--eval-steps must be positive")
    if not 0.0 <= args.td_lambda <= 1.0:
        parser.error("--td-lambda must be in [0, 1]")
    if not 0.0 <= args.target_polyak < 1.0:
        parser.error("--target-polyak must be in [0, 1)")
    if args.gradcheck_directions < 1 or args.gradcheck_eps <= 0.0:
        parser.error("gradient-check directions and epsilon must be positive")
    if args.gradcheck_horizons is not None and any(
        horizon < 1 for horizon in args.gradcheck_horizons
    ):
        parser.error("gradient-check horizons must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.pr_root = args.pr_root.expanduser().resolve()
    args.newton_root = args.newton_root.expanduser().resolve()
    spec = TASKS[args.task]
    if args.xml is not None:
        xml_path = args.xml.expanduser().resolve()
    elif args.task == "humanoid":
        xml_path = (args.pr_root / spec.default_xml).resolve()
    else:
        xml_path = spec.default_xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    imported_root = Path(mjw.__file__).resolve().parent.parent
    if not args.allow_other_mjw and imported_root != args.pr_root.resolve():
        raise RuntimeError(
            f"Imported mujoco_warp from {imported_root}, expected {args.pr_root}. "
            "Use --allow-other-mjw only when intentional."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    # PR #1535's gradient gate must be enabled before put_model / differentiated launches.
    mjw.enable_grad()

    loaded = load_model(spec, xml_path)
    setup_start = time.perf_counter()
    bridge = make_bridge(loaded, args)
    _sync(bridge.torch_device)
    setup_seconds = time.perf_counter() - setup_start

    checkpoint_payload = None
    if args.mode == "train":
        run, checkpoint_payload = train(args, loaded, bridge)
    elif args.mode == "gradcheck":
        run = gradcheck(args, loaded, bridge)
    else:
        run = evaluate_checkpoint(args, loaded, bridge)

    output_path = args.output or Path(f"pr1535_shac_{args.task}_{args.mode}.json")
    checkpoint_path = args.checkpoint
    if args.mode == "train":
        checkpoint_path = checkpoint_path or output_path.with_suffix(".pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, checkpoint_path)
        run["checkpoint"] = str(checkpoint_path.resolve())

    payload = {
        "schema": "mjwarp-pr1535-shac-style-v1",
        "mode": args.mode,
        "config": _jsonable_args(args),
        "provenance": provenance(args, loaded),
        "setup_seconds": setup_seconds,
        "run": run,
    }
    _write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"Wrote {output_path.resolve()}")
    if checkpoint_path is not None:
        print(f"Checkpoint {checkpoint_path.resolve()}")
    if args.mode == "gradcheck" and run["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
