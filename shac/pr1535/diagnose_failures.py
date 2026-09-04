#!/usr/bin/env python3
"""Reproducible diagnostics for the PR #1535 SHAC-style v1 checkpoints.

This script is intentionally separate from ``train_shac.py`` so that the v1
training evidence remains byte-for-byte auditable.  It performs three checks:

1. Measure actor outputs and critic hidden-unit/output saturation on states
   visited by each saved v1 checkpoint.
2. Freeze a one-lane, on-policy action tape and replay it in direct MJWarp and
   CPU MuJoCo, reporting state error through the first terminal transition.
   Both ordinary CPU stepping and a qpos/qvel-only CPU state contract are
   reported because the v1 bridge deliberately does not carry warm-start state.
3. Compare an analytic directional derivative of the complete v1 actor
   objective (including its terminal target critic) with central differences
   in actor-parameter space, at the nominal state and the last state before an
   on-policy fall when one is found.  Contact-count and alive-mask traces are
   retained for every perturbation.

Example, from the report checkout, using the frozen integration environment::

    /home/horde/repos/newton-shac-pr1535/.venv/bin/python \
      shac/pr1535/diagnose_failures.py --output /tmp/pr1535_diagnostics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import train_shac as v1
import warp as wp

EXPECTED_PR_HEAD = "02d09b139fdf091e1e859d7f41c47a8f71d30574"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results"
DEFAULT_PR_ROOT = Path("/home/horde/repos/mujoco_warp-pr1535")
DEFAULT_NEWTON_ROOT = Path("/home/horde/repos/newton-shac-pr1535")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray | Iterable[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise FloatingPointError("Non-finite value encountered while summarizing")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean_absolute": float(np.abs(array).mean()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "max_absolute": float(np.abs(array).max()),
    }


def _sync(device: torch.device) -> None:
    wp.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_xml(task: str, pr_root: Path) -> Path:
    spec = v1.TASKS[task]
    if task == "humanoid":
        return (pr_root / spec.default_xml).resolve()
    return spec.default_xml.resolve()


@dataclass
class Context:
    task: str
    loaded: v1.LoadedModel
    bridge: v1.MJWarpTorchBridge


def _make_context(
    task: str,
    *,
    worlds: int,
    device: str,
    pr_root: Path,
    nconmax: int,
    njmax: int,
) -> Context:
    loaded = v1.load_model(v1.TASKS[task], _resolve_xml(task, pr_root))
    bridge_args = argparse.Namespace(
        worlds=worlds,
        device=device,
        nconmax=nconmax,
        njmax=njmax,
    )
    return Context(task, loaded, v1.make_bridge(loaded, bridge_args))


def _load_checkpoint(
    path: Path,
    *,
    task: str,
    expected_head: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "mjwarp-pr1535-shac-style-v1":
        raise ValueError(f"Unexpected checkpoint format in {path}")
    if checkpoint.get("task") != task:
        raise ValueError(f"Checkpoint {path} is not for {task}")
    if checkpoint.get("pr_head") != expected_head:
        raise ValueError(
            f"Checkpoint {path} pins {checkpoint.get('pr_head')}, expected {expected_head}"
        )
    return checkpoint


def _net_parameter_statistics(module: torch.nn.Module) -> dict[str, Any]:
    flat = torch.cat(
        [parameter.detach().reshape(-1).cpu() for parameter in module.parameters()]
    )
    result = _summary(flat.numpy())
    result["l2_norm"] = float(torch.linalg.vector_norm(flat).item())
    return result


def _critic_activation_statistics(
    critic: v1.Critic, observations: torch.Tensor
) -> dict[str, Any]:
    # This function is called from a no-grad rollout.  Re-enable autograd only
    # for a detached copy of the sampled observations: dV/dobs is the signal
    # the terminal critic actually contributes to the actor through physics.
    with torch.enable_grad():
        differentiable_observations = observations.detach().clone().requires_grad_(True)
        value = differentiable_observations
        tanh_layers: list[dict[str, Any]] = []
        for module in critic.value.net:
            value = module(value)
            if isinstance(module, torch.nn.Tanh):
                absolute = value.abs()
                tanh_layers.append(
                    {
                        **_summary(value.detach().cpu().numpy()),
                        "fraction_abs_gt_0_95": float(
                            (absolute > 0.95).float().mean().item()
                        ),
                        "fraction_abs_gt_0_99": float(
                            (absolute > 0.99).float().mean().item()
                        ),
                        "fraction_abs_gt_0_999": float(
                            (absolute > 0.999).float().mean().item()
                        ),
                    }
                )
        output = critic(differentiable_observations)
        input_gradient = torch.autograd.grad(
            output.sum(), differentiable_observations, only_inputs=True
        )[0]
        input_gradient_l2 = torch.linalg.vector_norm(input_gradient, dim=-1)
    return {
        "output": _summary(output.detach().cpu().numpy()),
        "input_gradient_l2_per_state": _summary(
            input_gradient_l2.detach().cpu().numpy()
        ),
        "input_gradient_elements": _summary(input_gradient.detach().cpu().numpy()),
        "hidden_tanh_layers": tanh_layers,
        "parameters": _net_parameter_statistics(critic),
    }


@torch.no_grad()
def _raw_objective_breakdown(
    context: Context,
    actor: v1.DeterministicActor,
    target_critic: v1.Critic,
    qpos_start: torch.Tensor,
    qvel_start: torch.Tensor,
    *,
    horizon: int,
    gamma: float,
) -> dict[str, Any]:
    qpos = qpos_start.clone()
    qvel = qvel_start.clone()
    alive = v1.healthy(context.loaded.spec, qpos, qvel)
    reward_component = torch.zeros_like(alive, dtype=torch.float32)
    discount = 1.0
    alive_after_trace: list[float] = []
    contact_trace: list[list[int]] = []
    for _ in range(horizon):
        active = alive
        action = actor(v1.observation(qpos, qvel))
        qpos_next, qvel_next = context.bridge._forward_raw(qpos, qvel, action)
        step_reward = v1.reward(
            context.loaded.spec,
            float(context.loaded.model.opt.timestep),
            qpos,
            qvel,
            action,
            qpos_next,
            qvel_next,
        )
        alive = active & v1.healthy(context.loaded.spec, qpos_next, qvel_next)
        reward_component += discount * active.float() * step_reward
        discount *= gamma
        qpos, qvel = qpos_next, qvel_next
        alive_after_trace.append(float(alive.float().mean().item()))
        contact_trace.append(
            context.bridge.data_out.nacon.numpy().astype(np.int64).tolist()
        )
    terminal_values = target_critic(v1.observation(qpos, qvel))
    terminal_component = discount * alive.float() * terminal_values
    reward_mean = float(reward_component.mean().item())
    terminal_mean = float(terminal_component.mean().item())
    total = reward_mean + terminal_mean
    return {
        "horizon": horizon,
        "physical_horizon_seconds": horizon * float(context.loaded.model.opt.timestep),
        "discounted_reward_component_mean": reward_mean,
        "terminal_target_component_mean": terminal_mean,
        "objective_mean": total,
        "terminal_fraction_of_objective_absolute": abs(terminal_mean)
        / max(abs(reward_mean) + abs(terminal_mean), 1.0e-12),
        "terminal_target_value": _summary(terminal_values.cpu().numpy()),
        "alive_fraction_after_each_step": alive_after_trace,
        "active_contact_count_trace": contact_trace,
    }


@torch.no_grad()
def _checkpoint_policy_statistics(
    context: Context,
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    policy_key: str,
    steps: int,
    worlds: int,
) -> dict[str, Any]:
    device = context.bridge.torch_device
    actor, critic, target_critic = v1.make_networks(
        context.loaded, int(checkpoint["hidden"]), device
    )
    actor.load_state_dict(checkpoint[policy_key])
    critic.load_state_dict(checkpoint["critic"])
    target_critic.load_state_dict(checkpoint["target_critic"])
    actor.eval()
    critic.eval()
    target_critic.eval()

    seed = int(checkpoint["config"]["seed"])
    rng = np.random.default_rng(seed + 10_000)
    qpos, qvel = v1.sample_initial_states(
        context.loaded, worlds, rng, device, noisy=True
    )
    objective_qpos = qpos.clone()
    objective_qvel = qvel.clone()
    initial_x = qpos[:, 0].clone()
    alive = v1.healthy(context.loaded.spec, qpos, qvel)
    action_rows: list[np.ndarray] = []
    observation_rows: list[np.ndarray] = []
    raw_observation_rows: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    active_lane_steps = 0
    alive_trace: list[float] = []

    low = actor.action_low
    high = actor.action_high
    center = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    normalized_action_rows: list[np.ndarray] = []

    for _ in range(steps):
        raw_obs = torch.cat((qpos[:, 2:], 0.1 * qvel), dim=-1)
        obs = v1.observation(qpos, qvel)
        action = actor(obs)
        active = alive
        if bool(active.any().item()):
            observation_rows.append(obs[active].cpu().numpy())
            raw_observation_rows.append(raw_obs[active].cpu().numpy())
            action_rows.append(action[active].cpu().numpy())
            normalized = (action[active] - center) / half_range.clamp_min(1.0e-12)
            normalized_action_rows.append(normalized.cpu().numpy())
            active_lane_steps += int(active.sum().item())

        qpos_candidate, qvel_candidate = context.bridge._forward_raw(qpos, qvel, action)
        step_reward = v1.reward(
            context.loaded.spec,
            float(context.loaded.model.opt.timestep),
            qpos,
            qvel,
            action,
            qpos_candidate,
            qvel_candidate,
        )
        if bool(active.any().item()):
            rewards.append(step_reward[active].cpu().numpy())
        candidate_finite = torch.isfinite(qpos_candidate).all(dim=-1) & torch.isfinite(
            qvel_candidate
        ).all(dim=-1)
        next_alive = (
            active
            & candidate_finite
            & v1.healthy(context.loaded.spec, qpos_candidate, qvel_candidate)
        )
        safe_qpos = torch.where(candidate_finite[:, None], qpos_candidate, qpos)
        safe_qvel = torch.where(candidate_finite[:, None], qvel_candidate, qvel)
        qpos = torch.where(active[:, None], safe_qpos, qpos)
        qvel = torch.where(active[:, None], safe_qvel, qvel)
        alive = next_alive
        alive_trace.append(float(alive.float().mean().item()))
        if not bool(alive.any().item()):
            break

    observations = torch.as_tensor(
        np.concatenate(observation_rows), dtype=torch.float32, device=device
    )
    raw_observations = np.concatenate(raw_observation_rows)
    actions = np.concatenate(action_rows)
    normalized_actions = np.concatenate(normalized_action_rows)
    action_abs = np.abs(normalized_actions)
    config = checkpoint["config"]
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_seed": seed,
        "policy_key": policy_key,
        "policy_epoch": int(checkpoint["best_actor_epoch"])
        if policy_key == "best_actor"
        else int(checkpoint["epoch"]),
        "requested_steps": steps,
        "executed_steps": len(alive_trace),
        "worlds": worlds,
        "active_lane_steps": active_lane_steps,
        "final_alive_fraction": float(alive.float().mean().item()),
        "alive_fraction_after_each_step": alive_trace,
        "mean_displacement": float((qpos[:, 0] - initial_x).mean().item()),
        "reward_on_active_lane_steps": _summary(np.concatenate(rewards)),
        "action_raw": _summary(actions),
        "action_normalized_to_ctrl_range": {
            **_summary(normalized_actions),
            "fraction_abs_gt_0_90": float((action_abs > 0.90).mean()),
            "fraction_abs_gt_0_95": float((action_abs > 0.95).mean()),
            "fraction_abs_gt_0_99": float((action_abs > 0.99).mean()),
        },
        "raw_observation": {
            **_summary(raw_observations),
            "fraction_clipped_by_v1_observation": float(
                (np.abs(raw_observations) > 10.0).mean()
            ),
        },
        "critic": _critic_activation_statistics(critic, observations),
        "target_critic": _critic_activation_statistics(target_critic, observations),
        "short_horizon_objective": _raw_objective_breakdown(
            context,
            actor,
            target_critic,
            objective_qpos,
            objective_qvel,
            horizon=int(config["horizon"]),
            gamma=float(config["gamma"]),
        ),
    }


def checkpoint_statistics(args: argparse.Namespace, task: str) -> dict[str, Any]:
    context = _make_context(
        task,
        worlds=args.stats_worlds,
        device=args.device,
        pr_root=args.pr_root,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        path = args.results_dir / f"{task}_seed{seed}.pt"
        checkpoint = _load_checkpoint(
            path,
            task=task,
            expected_head=args.expected_pr_head,
            device=context.bridge.torch_device,
        )
        for policy_key in ("best_actor", "actor"):
            entries.append(
                _checkpoint_policy_statistics(
                    context,
                    checkpoint,
                    checkpoint_path=path,
                    policy_key=policy_key,
                    steps=args.stats_steps,
                    worlds=args.stats_worlds,
                )
            )
    return {
        "definition": {
            "critic_saturation": "fraction of hidden tanh outputs with absolute value above 0.95/0.99/0.999",
            "action_saturation": "fraction of actions exceeding 90/95/99% of the actuator half-range",
            "state_distribution": "active lanes of a fixed-seed noisy rollout; terminal lanes are excluded after first termination",
        },
        "entries": entries,
    }


def _healthy_numpy(spec: v1.TaskSpec, qpos: np.ndarray, qvel: np.ndarray) -> bool:
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        return False
    root_up = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
    return bool(
        spec.healthy_z[0] < qpos[2] < spec.healthy_z[1] and root_up > spec.upright_min
    )


@torch.no_grad()
def _make_open_loop_tape(
    context: Context,
    actor: v1.DeterministicActor,
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    qpos, qvel = v1.sample_initial_states(
        context.loaded, 1, rng, context.bridge.torch_device, noisy=True
    )
    initial_qpos = qpos[0].cpu().numpy().astype(np.float64)
    initial_qvel = qvel[0].cpu().numpy().astype(np.float64)
    actions: list[np.ndarray] = []
    states: list[tuple[np.ndarray, np.ndarray]] = [
        (initial_qpos.copy(), initial_qvel.copy())
    ]
    contacts: list[int] = []
    terminal_step: int | None = None
    for step in range(1, steps + 1):
        action = actor(v1.observation(qpos, qvel))
        actions.append(action[0].cpu().numpy().astype(np.float64))
        qpos, qvel = context.bridge._forward_raw(qpos, qvel, action)
        qpos_np = qpos[0].cpu().numpy().astype(np.float64)
        qvel_np = qvel[0].cpu().numpy().astype(np.float64)
        states.append((qpos_np, qvel_np))
        contacts.append(int(context.bridge.data_out.nacon.numpy()[0]))
        if not _healthy_numpy(context.loaded.spec, qpos_np, qvel_np):
            terminal_step = step
            break
    return {
        "initial_qpos": initial_qpos,
        "initial_qvel": initial_qvel,
        "actions": np.asarray(actions),
        "states": states,
        "contacts": contacts,
        "terminal_step": terminal_step,
    }


def _cpu_replay(
    loaded: v1.LoadedModel,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    actions: np.ndarray,
    *,
    reset_warmstart_each_step: bool,
) -> dict[str, Any]:
    data = mujoco.MjData(loaded.model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0.0
    mujoco.mj_forward(loaded.model, data)
    template_warmstart = loaded.data.qacc_warmstart.copy()
    states: list[tuple[np.ndarray, np.ndarray]] = [(data.qpos.copy(), data.qvel.copy())]
    contacts: list[int] = []
    terminal_step: int | None = None
    for step, action in enumerate(actions, start=1):
        data.ctrl[:] = action
        if reset_warmstart_each_step:
            data.qacc_warmstart[:] = template_warmstart
        mujoco.mj_step(loaded.model, data)
        states.append((data.qpos.copy(), data.qvel.copy()))
        contacts.append(int(data.ncon))
        if terminal_step is None and not _healthy_numpy(
            loaded.spec, data.qpos, data.qvel
        ):
            terminal_step = step
    return {
        "states": states,
        "contacts": contacts,
        "terminal_step": terminal_step,
    }


def _parity_comparison(mjw_tape: dict[str, Any], cpu: dict[str, Any]) -> dict[str, Any]:
    terminal_candidates = [
        step
        for step in (mjw_tape["terminal_step"], cpu["terminal_step"])
        if step is not None
    ]
    end_step = (
        min(terminal_candidates) if terminal_candidates else len(mjw_tape["actions"])
    )
    per_step: list[dict[str, Any]] = []
    all_qpos: list[np.ndarray] = []
    all_qvel: list[np.ndarray] = []
    for step in range(1, end_step + 1):
        mjw_qpos, mjw_qvel = mjw_tape["states"][step]
        cpu_qpos, cpu_qvel = cpu["states"][step]
        qpos_error = mjw_qpos - cpu_qpos
        qvel_error = mjw_qvel - cpu_qvel
        all_qpos.append(qpos_error)
        all_qvel.append(qvel_error)
        per_step.append(
            {
                "step": step,
                "time_seconds": step * 0.0,  # Filled by the caller.
                "qpos_max_absolute_error": float(np.abs(qpos_error).max()),
                "qpos_rms_error": float(np.sqrt(np.mean(np.square(qpos_error)))),
                "qvel_max_absolute_error": float(np.abs(qvel_error).max()),
                "qvel_rms_error": float(np.sqrt(np.mean(np.square(qvel_error)))),
                "mjwarp_contact_count": int(mjw_tape["contacts"][step - 1]),
                "cpu_contact_count": int(cpu["contacts"][step - 1]),
            }
        )
    qpos_errors = np.concatenate(all_qpos) if all_qpos else np.empty(0)
    qvel_errors = np.concatenate(all_qvel) if all_qvel else np.empty(0)
    return {
        "comparison_end_step": end_step,
        "ended_at_first_terminal": bool(terminal_candidates),
        "cpu_terminal_step": cpu["terminal_step"],
        "qpos_error_all_steps": _summary(qpos_errors),
        "qvel_error_all_steps": _summary(qvel_errors),
        "contact_count_mismatch_steps": sum(
            row["mjwarp_contact_count"] != row["cpu_contact_count"] for row in per_step
        ),
        "per_step": per_step,
    }


def open_loop_parity(args: argparse.Namespace, task: str) -> dict[str, Any]:
    context = _make_context(
        task,
        worlds=1,
        device=args.device,
        pr_root=args.pr_root,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    path = args.results_dir / f"{task}_seed{args.primary_seed}.pt"
    checkpoint = _load_checkpoint(
        path,
        task=task,
        expected_head=args.expected_pr_head,
        device=context.bridge.torch_device,
    )
    actor, _, _ = v1.make_networks(
        context.loaded, int(checkpoint["hidden"]), context.bridge.torch_device
    )
    actor.load_state_dict(checkpoint[args.parity_policy])
    actor.eval()
    tape = _make_open_loop_tape(
        context,
        actor,
        seed=int(checkpoint["config"]["seed"]) + 10_000,
        steps=args.parity_steps,
    )
    cpu_carry = _cpu_replay(
        context.loaded,
        tape["initial_qpos"],
        tape["initial_qvel"],
        tape["actions"],
        reset_warmstart_each_step=False,
    )
    cpu_reset = _cpu_replay(
        context.loaded,
        tape["initial_qpos"],
        tape["initial_qvel"],
        tape["actions"],
        reset_warmstart_each_step=True,
    )
    timestep = float(context.loaded.model.opt.timestep)
    carry_comparison = _parity_comparison(tape, cpu_carry)
    reset_comparison = _parity_comparison(tape, cpu_reset)
    for comparison in (carry_comparison, reset_comparison):
        for row in comparison["per_step"]:
            row["time_seconds"] = row["step"] * timestep
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha256(path),
        "policy_key": args.parity_policy,
        "action_tape_source": "actions emitted by the MJWarp on-policy trajectory, then frozen for all CPU replays",
        "initial_state_seed": int(checkpoint["config"]["seed"]) + 10_000,
        "timestep": timestep,
        "requested_steps": args.parity_steps,
        "action_tape_steps": len(tape["actions"]),
        "mjwarp_terminal_step": tape["terminal_step"],
        "cpu_carry_all_state": carry_comparison,
        "cpu_reset_warmstart_each_step": reset_comparison,
        "interpretation_note": (
            "The reset-warmstart replay matches the bridge's documented qpos/qvel-only "
            "state contract more closely; ordinary CPU stepping carries qacc_warmstart."
        ),
    }


def _objective(
    context: Context,
    actor: v1.DeterministicActor,
    target_critic: v1.Critic,
    qpos_start: torch.Tensor,
    qvel_start: torch.Tensor,
    *,
    horizon: int,
    gamma: float,
    differentiable: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    qpos = qpos_start
    qvel = qvel_start
    alive = v1.healthy(context.loaded.spec, qpos, qvel)
    objective = torch.zeros(1, dtype=torch.float32, device=context.bridge.torch_device)
    reward_component = torch.zeros_like(objective)
    discount = 1.0
    contacts: list[int] = []
    alive_before: list[bool] = []
    alive_after: list[bool] = []
    for _ in range(horizon):
        active = alive
        action = actor(v1.observation(qpos, qvel))
        if differentiable:
            qpos_next, qvel_next = context.bridge.step(qpos, qvel, action)
        else:
            qpos_next, qvel_next = context.bridge._forward_raw(qpos, qvel, action)
        step_reward = v1.reward(
            context.loaded.spec,
            float(context.loaded.model.opt.timestep),
            qpos,
            qvel,
            action,
            qpos_next,
            qvel_next,
        )
        next_alive = active & v1.healthy(context.loaded.spec, qpos_next, qvel_next)
        weighted = discount * active.float() * step_reward
        objective = objective + weighted
        reward_component = reward_component + weighted
        contacts.append(int(context.bridge.data_out.nacon.numpy()[0]))
        alive_before.append(bool(active.item()))
        alive_after.append(bool(next_alive.item()))
        qpos, qvel, alive = qpos_next, qvel_next, next_alive
        discount *= gamma
    terminal_value = target_critic(v1.observation(qpos, qvel))
    terminal_component = discount * alive.float() * terminal_value
    objective = objective + terminal_component
    trace = {
        "contact_count": contacts,
        "alive_before": alive_before,
        "alive_after": alive_after,
        "discounted_reward_component": float(reward_component.detach().item()),
        "terminal_target_component": float(terminal_component.detach().item()),
        "terminal_target_value": float(terminal_value.detach().item()),
    }
    return objective.mean(), trace


@torch.no_grad()
def _last_preterminal_state(
    context: Context,
    actor: v1.DeterministicActor,
    qpos_start: torch.Tensor,
    qvel_start: torch.Tensor,
    *,
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
    qpos = qpos_start.clone()
    qvel = qvel_start.clone()
    for step in range(1, max_steps + 1):
        before_qpos = qpos.clone()
        before_qvel = qvel.clone()
        action = actor(v1.observation(qpos, qvel))
        qpos, qvel = context.bridge._forward_raw(qpos, qvel, action)
        if not bool(v1.healthy(context.loaded.spec, qpos, qvel).item()):
            return (
                before_qpos,
                before_qvel,
                {
                    "terminal_transition_step": step,
                    "preterminal_height": float(before_qpos[0, 2].item()),
                    "preterminal_root_up": float(v1.root_up(before_qpos).item()),
                    "terminal_height": float(qpos[0, 2].item()),
                    "terminal_root_up": float(v1.root_up(qpos).item()),
                },
            )
    return None


def _parameter_directions(
    actor: torch.nn.Module,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[list[torch.Tensor]]:
    directions: list[list[torch.Tensor]] = []
    parameters = list(actor.parameters())
    for _ in range(count):
        arrays = [
            rng.normal(size=tuple(p.shape)).astype(np.float32) for p in parameters
        ]
        norm = math.sqrt(sum(float(np.square(array).sum()) for array in arrays))
        directions.append(
            [
                torch.as_tensor(array / norm, dtype=p.dtype, device=p.device)
                for array, p in zip(arrays, parameters, strict=True)
            ]
        )
    return directions


def _directional_objective_check(
    context: Context,
    actor: v1.DeterministicActor,
    target_critic: v1.Critic,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    *,
    horizon: int,
    gamma: float,
    epsilon: float,
    direction_count: int,
    seed: int,
) -> dict[str, Any]:
    actor.zero_grad(set_to_none=True)
    objective, base_trace = _objective(
        context,
        actor,
        target_critic,
        qpos,
        qvel,
        horizon=horizon,
        gamma=gamma,
        differentiable=True,
    )
    objective.backward()
    _sync(context.bridge.torch_device)
    parameters = list(actor.parameters())
    gradients = [
        torch.zeros_like(parameter)
        if parameter.grad is None
        else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    gradient_l2 = math.sqrt(
        sum(float(torch.sum(gradient.square()).item()) for gradient in gradients)
    )
    rng = np.random.default_rng(seed)
    directions = _parameter_directions(actor, count=direction_count, rng=rng)
    comparisons: list[dict[str, Any]] = []
    for index, direction in enumerate(directions):
        analytic = sum(
            float(torch.sum(gradient * vector).item())
            for gradient, vector in zip(gradients, direction, strict=True)
        )
        try:
            with torch.no_grad():
                for parameter, vector in zip(parameters, direction, strict=True):
                    parameter.add_(vector, alpha=epsilon)
                plus, plus_trace = _objective(
                    context,
                    actor,
                    target_critic,
                    qpos,
                    qvel,
                    horizon=horizon,
                    gamma=gamma,
                    differentiable=False,
                )
                for parameter, vector in zip(parameters, direction, strict=True):
                    parameter.add_(vector, alpha=-2.0 * epsilon)
                minus, minus_trace = _objective(
                    context,
                    actor,
                    target_critic,
                    qpos,
                    qvel,
                    horizon=horizon,
                    gamma=gamma,
                    differentiable=False,
                )
        finally:
            with torch.no_grad():
                for parameter, vector in zip(parameters, direction, strict=True):
                    parameter.add_(vector, alpha=epsilon)
        finite_difference = float(((plus - minus) / (2.0 * epsilon)).item())
        absolute_error = abs(analytic - finite_difference)
        relative_error = absolute_error / max(
            abs(analytic), abs(finite_difference), 1.0e-7
        )
        contact_mismatches = sum(
            p != m or p != b
            for b, p, m in zip(
                base_trace["contact_count"],
                plus_trace["contact_count"],
                minus_trace["contact_count"],
                strict=True,
            )
        )
        alive_mismatches = sum(
            p != m or p != b
            for b, p, m in zip(
                base_trace["alive_after"],
                plus_trace["alive_after"],
                minus_trace["alive_after"],
                strict=True,
            )
        )
        comparisons.append(
            {
                "direction": index,
                "analytic": analytic,
                "finite_difference": finite_difference,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "same_sign": analytic * finite_difference >= 0.0,
                "contact_count_mismatch_steps": contact_mismatches,
                "alive_mask_mismatch_steps": alive_mismatches,
                "plus_objective": float(plus.item()),
                "minus_objective": float(minus.item()),
                "plus_trace": plus_trace,
                "minus_trace": minus_trace,
            }
        )
    return {
        "horizon": horizon,
        "physical_horizon_seconds": horizon * float(context.loaded.model.opt.timestep),
        "epsilon_parameter_l2": epsilon,
        "objective": float(objective.detach().item()),
        "actor_parameter_gradient_l2": gradient_l2,
        "base_trace": base_trace,
        "comparisons": comparisons,
        "max_relative_error": max(item["relative_error"] for item in comparisons),
        "all_same_sign": all(item["same_sign"] for item in comparisons),
    }


def actor_objective_gradients(args: argparse.Namespace, task: str) -> dict[str, Any]:
    context = _make_context(
        task,
        worlds=1,
        device=args.device,
        pr_root=args.pr_root,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    path = args.results_dir / f"{task}_seed{args.primary_seed}.pt"
    checkpoint = _load_checkpoint(
        path,
        task=task,
        expected_head=args.expected_pr_head,
        device=context.bridge.torch_device,
    )
    actor, _, target_critic = v1.make_networks(
        context.loaded, int(checkpoint["hidden"]), context.bridge.torch_device
    )
    actor.load_state_dict(checkpoint[args.gradient_policy])
    target_critic.load_state_dict(checkpoint["target_critic"])
    actor.eval()
    target_critic.eval()
    config = checkpoint["config"]
    horizon = int(config["horizon"])
    gamma = float(config["gamma"])
    rng = np.random.default_rng(int(config["seed"]) + 10_000)
    nominal_qpos, nominal_qvel = v1.sample_initial_states(
        context.loaded, 1, rng, context.bridge.torch_device, noisy=False
    )
    noisy_qpos, noisy_qvel = v1.sample_initial_states(
        context.loaded,
        1,
        np.random.default_rng(int(config["seed"]) + 10_000),
        context.bridge.torch_device,
        noisy=True,
    )
    states: dict[str, Any] = {
        "nominal": {
            "state_metadata": {
                "height": float(nominal_qpos[0, 2].item()),
                "root_up": float(v1.root_up(nominal_qpos).item()),
            },
            "check": _directional_objective_check(
                context,
                actor,
                target_critic,
                nominal_qpos,
                nominal_qvel,
                horizon=horizon,
                gamma=gamma,
                epsilon=args.gradient_epsilon,
                direction_count=args.gradient_directions,
                seed=args.gradient_seed,
            ),
        }
    }
    preterminal = _last_preterminal_state(
        context,
        actor,
        noisy_qpos,
        noisy_qvel,
        max_steps=args.prefall_search_steps,
    )
    if preterminal is None:
        states["pre_fall"] = {
            "available": False,
            "reason": f"No terminal transition found in {args.prefall_search_steps} on-policy steps",
        }
    else:
        pre_qpos, pre_qvel, metadata = preterminal
        states["pre_fall"] = {
            "available": True,
            "state_metadata": metadata,
            "check": _directional_objective_check(
                context,
                actor,
                target_critic,
                pre_qpos,
                pre_qvel,
                horizon=horizon,
                gamma=gamma,
                epsilon=args.gradient_epsilon,
                direction_count=args.gradient_directions,
                seed=args.gradient_seed + 1,
            ),
        }
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha256(path),
        "policy_key": args.gradient_policy,
        "derivative": "complete v1 closed-loop actor objective with terminal target critic, directional derivative in actor-parameter space",
        "states": states,
    }


def _aggregate_findings(payload: dict[str, Any]) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    if "checkpoint_statistics" in payload["checks"]:
        task_findings = {}
        for task, result in payload["checks"]["checkpoint_statistics"].items():
            entries = result["entries"]
            by_policy = {}
            for policy_key in ("best_actor", "actor"):
                policy_entries = [
                    entry for entry in entries if entry["policy_key"] == policy_key
                ]
                by_policy[policy_key] = {
                    "target_input_gradient_l2_mean_range": [
                        min(
                            entry["target_critic"]["input_gradient_l2_per_state"][
                                "mean"
                            ]
                            for entry in policy_entries
                        ),
                        max(
                            entry["target_critic"]["input_gradient_l2_per_state"][
                                "mean"
                            ]
                            for entry in policy_entries
                        ),
                    ],
                    "second_hidden_abs_gt_0_99_fraction_range": [
                        min(
                            entry["target_critic"]["hidden_tanh_layers"][1][
                                "fraction_abs_gt_0_99"
                            ]
                            for entry in policy_entries
                        ),
                        max(
                            entry["target_critic"]["hidden_tanh_layers"][1][
                                "fraction_abs_gt_0_99"
                            ]
                            for entry in policy_entries
                        ),
                    ],
                    "action_abs_gt_0_95_fraction_range": [
                        min(
                            entry["action_normalized_to_ctrl_range"][
                                "fraction_abs_gt_0_95"
                            ]
                            for entry in policy_entries
                        ),
                        max(
                            entry["action_normalized_to_ctrl_range"][
                                "fraction_abs_gt_0_95"
                            ]
                            for entry in policy_entries
                        ),
                    ],
                    "action_normalized_rms_range": [
                        min(
                            entry["action_normalized_to_ctrl_range"]["rms"]
                            for entry in policy_entries
                        ),
                        max(
                            entry["action_normalized_to_ctrl_range"]["rms"]
                            for entry in policy_entries
                        ),
                    ],
                }
            task_findings[task] = {
                "by_policy": by_policy,
                "max_critic_hidden_abs_gt_0_99_fraction": max(
                    layer["fraction_abs_gt_0_99"]
                    for entry in entries
                    for layer in entry["critic"]["hidden_tanh_layers"]
                ),
                "max_target_hidden_abs_gt_0_99_fraction": max(
                    layer["fraction_abs_gt_0_99"]
                    for entry in entries
                    for layer in entry["target_critic"]["hidden_tanh_layers"]
                ),
                "critic_input_gradient_l2_mean_range": [
                    min(
                        entry["critic"]["input_gradient_l2_per_state"]["mean"]
                        for entry in entries
                    ),
                    max(
                        entry["critic"]["input_gradient_l2_per_state"]["mean"]
                        for entry in entries
                    ),
                ],
                "target_input_gradient_l2_mean_range": [
                    min(
                        entry["target_critic"]["input_gradient_l2_per_state"]["mean"]
                        for entry in entries
                    ),
                    max(
                        entry["target_critic"]["input_gradient_l2_per_state"]["mean"]
                        for entry in entries
                    ),
                ],
                "max_action_abs_gt_0_95_fraction": max(
                    entry["action_normalized_to_ctrl_range"]["fraction_abs_gt_0_95"]
                    for entry in entries
                ),
                "max_terminal_fraction_of_objective_absolute": max(
                    entry["short_horizon_objective"][
                        "terminal_fraction_of_objective_absolute"
                    ]
                    for entry in entries
                ),
            }
        findings["checkpoint_statistics"] = task_findings
    if "open_loop_parity" in payload["checks"]:
        findings["open_loop_parity"] = {
            task: {
                "mjwarp_terminal_step": result["mjwarp_terminal_step"],
                "cpu_carry_terminal_step": result["cpu_carry_all_state"][
                    "cpu_terminal_step"
                ],
                "cpu_reset_terminal_step": result["cpu_reset_warmstart_each_step"][
                    "cpu_terminal_step"
                ],
                "cpu_carry_qpos_max_absolute_error": result["cpu_carry_all_state"][
                    "qpos_error_all_steps"
                ].get("max_absolute"),
                "cpu_reset_qpos_max_absolute_error": result[
                    "cpu_reset_warmstart_each_step"
                ]["qpos_error_all_steps"].get("max_absolute"),
            }
            for task, result in payload["checks"]["open_loop_parity"].items()
        }
    if "actor_objective_gradients" in payload["checks"]:
        findings["actor_objective_gradients"] = {}
        for task, result in payload["checks"]["actor_objective_gradients"].items():
            states = {}
            for state_name, state in result["states"].items():
                if state.get("available") is False:
                    states[state_name] = {"available": False}
                else:
                    check = state["check"]
                    states[state_name] = {
                        "max_relative_error": check["max_relative_error"],
                        "all_same_sign": check["all_same_sign"],
                        "contact_mismatch_directions": sum(
                            item["contact_count_mismatch_steps"] > 0
                            for item in check["comparisons"]
                        ),
                        "alive_mismatch_directions": sum(
                            item["alive_mask_mismatch_steps"] > 0
                            for item in check["comparisons"]
                        ),
                    }
            findings["actor_objective_gradients"][task] = states
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(v1.TASKS), default=list(v1.TASKS)
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=("stats", "parity", "gradient"),
        default=["stats", "parity", "gradient"],
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pr-root", type=Path, default=DEFAULT_PR_ROOT)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT)
    parser.add_argument("--expected-pr-head", default=EXPECTED_PR_HEAD)
    parser.add_argument("--allow-head-mismatch", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--primary-seed", type=int, default=17)
    parser.add_argument("--stats-worlds", type=int, default=8)
    parser.add_argument("--stats-steps", type=int, default=128)
    parser.add_argument("--parity-steps", type=int, default=500)
    parser.add_argument(
        "--parity-policy", choices=("best_actor", "actor"), default="actor"
    )
    parser.add_argument(
        "--gradient-policy", choices=("best_actor", "actor"), default="actor"
    )
    parser.add_argument("--gradient-directions", type=int, default=3)
    parser.add_argument("--gradient-epsilon", type=float, default=2.0e-2)
    parser.add_argument("--gradient-seed", type=int, default=1535)
    parser.add_argument("--prefall-search-steps", type=int, default=500)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/pr1535_failure_diagnostics.json")
    )
    args = parser.parse_args()
    for name in (
        "stats_worlds",
        "stats_steps",
        "parity_steps",
        "gradient_directions",
        "prefall_search_steps",
        "nconmax",
        "njmax",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gradient_epsilon <= 0.0:
        parser.error("--gradient-epsilon must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.pr_root = args.pr_root.expanduser().resolve()
    args.newton_root = args.newton_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    actual_head = _git_head(args.pr_root)
    imported_root = Path(mjw.__file__).resolve().parent.parent
    if not args.allow_head_mismatch:
        if actual_head != args.expected_pr_head:
            raise RuntimeError(
                f"MJWarp worktree head is {actual_head}, expected {args.expected_pr_head}"
            )
        if imported_root != args.pr_root:
            raise RuntimeError(
                f"Imported mujoco_warp from {imported_root}, expected {args.pr_root}"
            )

    torch.manual_seed(args.gradient_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.gradient_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    wp.config.max_unroll = max(wp.config.max_unroll, 64)
    wp.set_device(args.device)
    mjw.enable_grad()
    started = time.perf_counter()

    payload: dict[str, Any] = {
        "schema": "mjwarp-pr1535-failure-diagnostics-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "pull_request": "https://github.com/google-deepmind/mujoco_warp/pull/1535",
            "expected_pr_head": args.expected_pr_head,
            "actual_pr_head": actual_head,
            "exact_head": actual_head == args.expected_pr_head,
            "mjwarp_import": str(Path(mjw.__file__).resolve()),
            "exact_worktree_import": imported_root == args.pr_root,
            "newton_head": _git_head(args.newton_root),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "v1_trainer_sha256": _sha256(SCRIPT_DIR / "train_shac.py"),
            "bridge_sha256": _sha256(SCRIPT_DIR / "mjwarp_torch_bridge.py"),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "warp": wp.__version__,
                "mujoco": mujoco.__version__,
                "mujoco_warp": getattr(mjw, "__version__", "unknown"),
                "numpy": np.__version__,
            },
            "device": args.device,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "checks": {},
    }
    if "stats" in args.checks:
        payload["checks"]["checkpoint_statistics"] = {
            task: checkpoint_statistics(args, task) for task in args.tasks
        }
    if "parity" in args.checks:
        payload["checks"]["open_loop_parity"] = {
            task: open_loop_parity(args, task) for task in args.tasks
        }
    if "gradient" in args.checks:
        payload["checks"]["actor_objective_gradients"] = {
            task: actor_objective_gradients(args, task) for task in args.tasks
        }
    payload["findings"] = _aggregate_findings(payload)
    payload["timing_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(payload["findings"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
