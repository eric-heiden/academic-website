#!/usr/bin/env python3
"""Focused Ant state-Jacobian checks for the SHAC report."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import warp as wp

from run_newton_shac import (
    DEFAULT_GRAD_CHECK_EPS,
    NewtonMuJoCoTorchEnv,
    central_difference_rows,
    finite_float,
    normalize_vec,
    pacific_now_iso,
    write_json,
)


def _masked_directions(width: int, mask: torch.Tensor, count: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    directions = torch.randn((count, width), generator=generator, device=device)
    directions = directions * mask.view(1, -1)
    return directions / directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)


def _best(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: row["mean_relative_error"])


def _apply_state_values(values: torch.Tensor, q_count: int, qd_count: int, q_shape, qd_shape) -> tuple[torch.Tensor, torch.Tensor]:
    q = values[:q_count].view(q_shape).clone()
    qd = values[q_count : q_count + qd_count].view(qd_shape).clone()
    if q.shape[-1] >= 7:
        q[:, 3:7] = normalize_vec(q[:, 3:7])
    return q, qd


def _normalize_state_q(q: torch.Tensor) -> torch.Tensor:
    if q.shape[-1] < 7:
        return q
    return torch.cat([q[:, :3], normalize_vec(q[:, 3:7]), q[:, 7:]], dim=-1)


def _transition_losses(
    env: NewtonMuJoCoTorchEnv,
    q0: torch.Tensor,
    qd0: torch.Tensor,
    action0: torch.Tensor,
    action1: torch.Tensor,
    q_weight: torch.Tensor,
    qd_weight: torch.Tensor,
    obs_weight: torch.Tensor,
    simple_policy_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    q1, qd1 = env.step(q0, qd0, env.action_to_joint_f(action0))
    obs1 = env.observe(q1, qd1, action0)
    reward1 = env.reward(q1, qd1, action0, obs=obs1).mean()

    q2_fixed, qd2_fixed = env.step(q1, qd1, env.action_to_joint_f(action1))
    obs2_fixed = env.observe(q2_fixed, qd2_fixed, action1)
    reward2_fixed = env.reward(q2_fixed, qd2_fixed, action1, obs=obs2_fixed).mean()

    simple_action1 = torch.tanh(obs1 @ simple_policy_w.T)
    q2_policy, qd2_policy = env.step(q1, qd1, env.action_to_joint_f(simple_action1))
    obs2_policy = env.observe(q2_policy, qd2_policy, simple_action1)
    reward2_policy = env.reward(q2_policy, qd2_policy, simple_action1, obs=obs2_policy).mean()

    return {
        "q1_weighted": (q1 * q_weight).mean(),
        "qd1_weighted": (qd1 * qd_weight).mean(),
        "obs1_weighted": (obs1 * obs_weight).mean(),
        "reward1": reward1,
        "reward2_fixed_action": reward2_fixed,
        "reward2_simple_policy": reward2_policy,
    }


def _check_action0(
    *,
    env: NewtonMuJoCoTorchEnv,
    q0: torch.Tensor,
    qd0: torch.Tensor,
    action0: torch.Tensor,
    action1: torch.Tensor,
    q_weight: torch.Tensor,
    qd_weight: torch.Tensor,
    obs_weight: torch.Tensor,
    simple_policy_w: torch.Tensor,
    epsilons: list[float],
    directions: int,
    seed: int,
) -> dict:
    base = action0.detach().reshape(-1)
    mutable = base.clone()

    def assign(values: torch.Tensor) -> None:
        mutable.copy_(values)

    rows_by_loss = {}
    for loss_name in [
        "q1_weighted",
        "qd1_weighted",
        "obs1_weighted",
        "reward1",
        "reward2_fixed_action",
        "reward2_simple_policy",
    ]:
        action_req = action0.detach().clone().requires_grad_(True)
        losses = _transition_losses(env, q0, qd0, action_req, action1, q_weight, qd_weight, obs_weight, simple_policy_w)
        loss = losses[loss_name]
        loss.backward()
        analytic = torch.nan_to_num(action_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

        generator = torch.Generator(device=env.torch_device)
        generator.manual_seed(seed + hash(loss_name) % 10000)
        dirs = torch.randn((directions, base.numel()), generator=generator, device=env.torch_device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1.0e-12)

        def evaluate() -> float:
            with torch.no_grad():
                losses_eval = _transition_losses(
                    env,
                    q0,
                    qd0,
                    mutable.view_as(action0),
                    action1,
                    q_weight,
                    qd_weight,
                    obs_weight,
                    simple_policy_w,
                )
            return float(losses_eval[loss_name].detach().cpu())

        rows = central_difference_rows(
            base_values=base,
            analytic_grad=analytic,
            directions=dirs,
            epsilons=epsilons,
            evaluate=evaluate,
            assign=assign,
        )
        rows_by_loss[loss_name] = {
            "analytic_grad_norm": finite_float(float(analytic.to(torch.float64).norm().detach().cpu())),
            "best": _best(rows),
            "epsilon_sweep": rows,
        }
    return rows_by_loss


def _check_state0(
    *,
    env: NewtonMuJoCoTorchEnv,
    q0: torch.Tensor,
    qd0: torch.Tensor,
    action0: torch.Tensor,
    action1: torch.Tensor,
    q_weight: torch.Tensor,
    qd_weight: torch.Tensor,
    obs_weight: torch.Tensor,
    simple_policy_w: torch.Tensor,
    epsilons: list[float],
    directions: int,
    seed: int,
) -> dict:
    q_req = q0.detach().clone().requires_grad_(True)
    qd_req = qd0.detach().clone().requires_grad_(True)
    losses = _transition_losses(
        env, _normalize_state_q(q_req), qd_req, action0, action1, q_weight, qd_weight, obs_weight, simple_policy_w
    )
    loss = losses["reward2_simple_policy"]
    loss.backward()
    analytic = torch.cat(
        [
            torch.nan_to_num(q_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(qd_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
        ]
    )
    base = torch.cat([q0.detach().reshape(-1), qd0.detach().reshape(-1)])
    mutable = base.clone()
    q_count = q0.numel()
    qd_count = qd0.numel()

    def assign(values: torch.Tensor) -> None:
        mutable.copy_(values)

    def evaluate() -> float:
        q_eval, qd_eval = _apply_state_values(mutable, q_count, qd_count, q0.shape, qd0.shape)
        with torch.no_grad():
            losses_eval = _transition_losses(
                env,
                q_eval,
                qd_eval,
                action0,
                action1,
                q_weight,
                qd_weight,
                obs_weight,
                simple_policy_w,
            )
        return float(losses_eval["reward2_simple_policy"].detach().cpu())

    masks = {
        "root_pos_q": torch.zeros_like(base),
        "root_quat_q_normalized": torch.zeros_like(base),
        "joint_q": torch.zeros_like(base),
        "root_qd": torch.zeros_like(base),
        "joint_qd": torch.zeros_like(base),
    }
    for env_id in range(env.num_envs):
        q_offset = env_id * env.q_dim
        qd_offset = q_count + env_id * env.qd_dim
        masks["root_pos_q"][q_offset : q_offset + 3] = 1.0
        masks["root_quat_q_normalized"][q_offset + 3 : q_offset + 7] = 1.0
        masks["joint_q"][q_offset + 7 : q_offset + env.q_dim] = 1.0
        masks["root_qd"][qd_offset : qd_offset + 6] = 1.0
        masks["joint_qd"][qd_offset + 6 : qd_offset + env.qd_dim] = 1.0

    rows_by_component = {}
    for i, (name, mask) in enumerate(masks.items()):
        dirs = _masked_directions(base.numel(), mask, directions, seed + 1000 + i, env.torch_device)
        rows = central_difference_rows(
            base_values=base,
            analytic_grad=analytic,
            directions=dirs,
            epsilons=epsilons,
            evaluate=evaluate,
            assign=assign,
        )
        rows_by_component[name] = {
            "analytic_grad_norm": finite_float(float((analytic * mask).to(torch.float64).norm().detach().cpu())),
            "best": _best(rows),
            "epsilon_sweep": rows,
        }
    return {
        "loss": float(loss.detach().cpu()),
        "analytic_grad_norm": finite_float(float(analytic.to(torch.float64).norm().detach().cpu())),
        "components": rows_by_component,
    }


def _check_one_step_state(
    *,
    env: NewtonMuJoCoTorchEnv,
    q_base: torch.Tensor,
    qd_base: torch.Tensor,
    action: torch.Tensor,
    epsilons: list[float],
    directions: int,
    seed: int,
) -> dict:
    q_req = q_base.detach().clone().requires_grad_(True)
    qd_req = qd_base.detach().clone().requires_grad_(True)
    q_next, qd_next = env.step(_normalize_state_q(q_req), qd_req, env.action_to_joint_f(action))
    obs_next = env.observe(q_next, qd_next, action)
    loss = env.reward(q_next, qd_next, action, obs=obs_next).mean()
    loss.backward()
    analytic = torch.cat(
        [
            torch.nan_to_num(q_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(qd_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
        ]
    )
    base = torch.cat([q_base.detach().reshape(-1), qd_base.detach().reshape(-1)])
    mutable = base.clone()
    q_count = q_base.numel()
    qd_count = qd_base.numel()

    def assign(values: torch.Tensor) -> None:
        mutable.copy_(values)

    def evaluate() -> float:
        q_eval, qd_eval = _apply_state_values(mutable, q_count, qd_count, q_base.shape, qd_base.shape)
        with torch.no_grad():
            q_next_eval, qd_next_eval = env.step(q_eval, qd_eval, env.action_to_joint_f(action))
            obs_next_eval = env.observe(q_next_eval, qd_next_eval, action)
            loss_eval = env.reward(q_next_eval, qd_next_eval, action, obs=obs_next_eval).mean()
        return float(loss_eval.detach().cpu())

    masks = {
        "root_pos_q": torch.zeros_like(base),
        "root_quat_q_normalized": torch.zeros_like(base),
        "joint_q": torch.zeros_like(base),
        "root_qd": torch.zeros_like(base),
        "joint_qd": torch.zeros_like(base),
    }
    for env_id in range(env.num_envs):
        q_offset = env_id * env.q_dim
        qd_offset = q_count + env_id * env.qd_dim
        masks["root_pos_q"][q_offset : q_offset + 3] = 1.0
        masks["root_quat_q_normalized"][q_offset + 3 : q_offset + 7] = 1.0
        masks["joint_q"][q_offset + 7 : q_offset + env.q_dim] = 1.0
        masks["root_qd"][qd_offset : qd_offset + 6] = 1.0
        masks["joint_qd"][qd_offset + 6 : qd_offset + env.qd_dim] = 1.0

    components = {}
    for i, (name, mask) in enumerate(masks.items()):
        dirs = _masked_directions(base.numel(), mask, directions, seed + i, env.torch_device)
        rows = central_difference_rows(
            base_values=base,
            analytic_grad=analytic,
            directions=dirs,
            epsilons=epsilons,
            evaluate=evaluate,
            assign=assign,
        )
        components[name] = {
            "analytic_grad_norm": finite_float(float((analytic * mask).to(torch.float64).norm().detach().cpu())),
            "best": _best(rows),
            "epsilon_sweep": rows,
        }
    return {
        "loss": float(loss.detach().cpu()),
        "analytic_grad_norm": finite_float(float(analytic.to(torch.float64).norm().detach().cpu())),
        "components": components,
    }


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wp.init()

    env = NewtonMuJoCoTorchEnv(
        env_name="ant",
        num_envs=args.num_envs,
        device=args.device,
        dt=args.dt,
        force_scale=args.force_scale,
        contact_backend=args.contact_backend,
    )
    q0, qd0 = env.reset(noise=0.0, stochastic_init=False)

    generator = torch.Generator(device=env.torch_device)
    generator.manual_seed(args.seed + 10)
    action0 = 0.35 * torch.tanh(torch.randn((env.num_envs, env.num_actions), generator=generator, device=env.torch_device))
    action1 = 0.25 * torch.tanh(torch.randn((env.num_envs, env.num_actions), generator=generator, device=env.torch_device))
    q_weight = torch.randn_like(q0, generator=generator)
    qd_weight = torch.randn_like(qd0, generator=generator)
    obs_weight = torch.randn((env.num_envs, env.num_obs), generator=generator, device=env.torch_device)
    simple_policy_w = 0.1 * torch.randn((env.num_actions, env.num_obs), generator=generator, device=env.torch_device)

    epsilons = args.eps or list(DEFAULT_GRAD_CHECK_EPS)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Running the tape backwards may produce incorrect gradients.*",
            category=UserWarning,
        )
        action0_checks = _check_action0(
            env=env,
            q0=q0,
            qd0=qd0,
            action0=action0,
            action1=action1,
            q_weight=q_weight,
            qd_weight=qd_weight,
            obs_weight=obs_weight,
            simple_policy_w=simple_policy_w,
            epsilons=epsilons,
            directions=args.directions,
            seed=args.seed + 100,
        )
        state0_checks = _check_state0(
            env=env,
            q0=q0,
            qd0=qd0,
            action0=action0,
            action1=action1,
            q_weight=q_weight,
            qd_weight=qd_weight,
            obs_weight=obs_weight,
            simple_policy_w=simple_policy_w,
            epsilons=epsilons,
            directions=args.directions,
            seed=args.seed + 200,
        )
        with torch.no_grad():
            q1_base, qd1_base = env.step(q0, qd0, env.action_to_joint_f(action0))
        state0_reward1 = _check_one_step_state(
            env=env,
            q_base=q0,
            qd_base=qd0,
            action=action0,
            epsilons=epsilons,
            directions=args.directions,
            seed=args.seed + 300,
        )
        state1_reward2_fixed_action = _check_one_step_state(
            env=env,
            q_base=q1_base,
            qd_base=qd1_base,
            action=action1,
            epsilons=epsilons,
            directions=args.directions,
            seed=args.seed + 400,
        )

    result = {
        "mode": "ant_state_jacobian_diagnostics",
        "timestamp_pacific": pacific_now_iso(),
        "contact_backend": args.contact_backend,
        "num_envs": args.num_envs,
        "dt": args.dt,
        "force_scale": args.force_scale,
        "epsilon_values": epsilons,
        "directions": args.directions,
        "action0_checks": action0_checks,
        "state0_reward1": state0_reward1,
        "state1_reward2_fixed_action": state1_reward2_fixed_action,
        "state0_reward2_simple_policy": state0_checks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, result)
    print(f"wrote Ant state-Jacobian diagnostics to {out_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent
            / "assets"
            / "ant_state_jacobian_diagnostics"
            / "ant_state_jacobian_diagnostics.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-backend", choices=["none", "newton", "mujoco"], default="none")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--force-scale", type=float, default=100.0)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--eps", type=float, nargs="*")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
