#!/usr/bin/env python3
"""Multistep Ant gradient diagnostics around a saved policy."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp

from run_newton_shac import (
    ANT_DEFAULT_TERMINATION_PENALTY,
    DEFAULT_GRAD_CHECK_EPS,
    AntRewardWeights,
    CheetahRewardWeights,
    ContactTargetRewardWeights,
    HopperRewardWeights,
    NewtonMuJoCoTorchEnv,
    assign_flat_parameters,
    central_difference_rows,
    deterministic_policy_action,
    finite_float,
    flatten_gradients,
    flatten_parameters,
    load_actor_checkpoint,
    load_obs_rms,
    make_actor,
    masked_random_directions,
    normalize_obs,
    one_step_action_loss,
    pacific_now_iso,
    parse_float_list,
    parse_int_list,
    query_gpu,
    trainable_parameters,
    write_json,
    finalize_terminal_reward,
    git_commit_for_imported_module,
)

import mujoco_warp
import newton


def _best(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: row["mean_relative_error"])


def _directions(count: int, width: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    dirs = torch.randn((count, width), generator=generator, dtype=dtype, device=device)
    return dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1.0e-12)


def _step_reward(env: NewtonMuJoCoTorchEnv, q: torch.Tensor, qd: torch.Tensor, action: torch.Tensor, args):
    q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
    invalid = env.invalid_state(q_next, qd_next)
    fell = torch.logical_and(env.fallen_state(q_next), ~invalid)
    obs_next = env.observe(q_next, qd_next, action)
    reward = env.reward(q_next, qd_next, action, obs=obs_next)
    reward = finalize_terminal_reward(
        reward,
        invalid=invalid,
        fell=fell,
        termination_penalty=args.termination_penalty,
    )
    return q_next, qd_next, reward, invalid, fell


def _rollout_fixed_actions(env: NewtonMuJoCoTorchEnv, q0, qd0, actions, args):
    q = q0
    qd = qd0
    gamma_vec = torch.ones(env.num_envs, dtype=torch.float32, device=env.torch_device)
    loss = torch.zeros((), dtype=torch.float32, device=env.torch_device)
    fall_count = 0
    invalid_count = 0
    rewards = []
    for action in actions:
        q, qd, reward, invalid, fell = _step_reward(env, q, qd, action, args)
        rewards.append(reward.detach().mean())
        loss = loss - (gamma_vec * reward * args.rew_scale).sum()
        done = torch.logical_or(invalid, fell)
        fall_count += int(fell.detach().sum().cpu())
        invalid_count += int(invalid.detach().sum().cpu())
        gamma_vec = gamma_vec * args.gamma * (~done).to(torch.float32)
    return loss / max(1, len(actions) * env.num_envs), {
        "mean_reward": float(torch.stack(rewards).mean().detach().cpu()) if rewards else None,
        "fall_count": fall_count,
        "invalid_count": invalid_count,
    }


def _rollout_policy(env: NewtonMuJoCoTorchEnv, actor, q0, qd0, prev0, obs_stats, args):
    q = q0
    qd = qd0
    prev_action = prev0
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    actions = []
    gamma_vec = torch.ones(env.num_envs, dtype=torch.float32, device=env.torch_device)
    loss = torch.zeros((), dtype=torch.float32, device=env.torch_device)
    fall_count = 0
    invalid_count = 0
    rewards = []
    for _ in range(args.horizon):
        obs = normalize_obs(env.observe(q, qd, prev_action, phase=progress), obs_stats)
        action = deterministic_policy_action(actor, obs)
        actions.append(action)
        q, qd, reward, invalid, fell = _step_reward(env, q, qd, action, args)
        rewards.append(reward.detach().mean())
        loss = loss - (gamma_vec * reward * args.rew_scale).sum()
        done = torch.logical_or(invalid, fell)
        fall_count += int(fell.detach().sum().cpu())
        invalid_count += int(invalid.detach().sum().cpu())
        gamma_vec = gamma_vec * args.gamma * (~done).to(torch.float32)
        prev_action = action
        progress = torch.where(done, torch.zeros_like(progress), progress + 1)
    return loss / max(1, args.horizon * env.num_envs), actions, {
        "mean_reward": float(torch.stack(rewards).mean().detach().cpu()) if rewards else None,
        "fall_count": fall_count,
        "invalid_count": invalid_count,
    }


def _check_actions(env, q0, qd0, actions_base, args, epsilons):
    actions_req = actions_base.detach().clone().requires_grad_(True)
    actions = [actions_req[i] for i in range(actions_req.shape[0])]
    loss, metrics = _rollout_fixed_actions(env, q0, qd0, actions, args)
    loss.backward()
    analytic = torch.nan_to_num(actions_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    base = actions_base.detach().reshape(-1)
    mutable = base.clone()

    def assign(values: torch.Tensor) -> None:
        mutable.copy_(values)

    def evaluate() -> float:
        with torch.no_grad():
            action_seq = list(mutable.view_as(actions_base))
            value, _ = _rollout_fixed_actions(env, q0, qd0, action_seq, args)
        return float(value.detach().cpu())

    rows = central_difference_rows(
        base_values=base,
        analytic_grad=analytic,
        directions=_directions(args.directions, base.numel(), args.seed + 3000, env.torch_device, base.dtype),
        epsilons=epsilons,
        evaluate=evaluate,
        assign=assign,
    )
    return {
        "loss": float(loss.detach().cpu()),
        "metrics": metrics,
        "analytic_grad_norm": finite_float(float(analytic.to(torch.float64).norm().detach().cpu())),
        "best": _best(rows),
        "epsilon_sweep": rows,
    }


def _check_initial_state(env, q0, qd0, actions_base, args, epsilons):
    q_req = q0.detach().clone().requires_grad_(True)
    qd_req = qd0.detach().clone().requires_grad_(True)
    actions = [action.detach() for action in actions_base]
    loss, metrics = _rollout_fixed_actions(env, q_req, qd_req, actions, args)
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

    def assign(values: torch.Tensor) -> None:
        mutable.copy_(values)

    def evaluate() -> float:
        with torch.no_grad():
            q_eval = mutable[:q_count].view_as(q0)
            qd_eval = mutable[q_count:].view_as(qd0)
            value, _ = _rollout_fixed_actions(env, q_eval, qd_eval, actions, args)
        return float(value.detach().cpu())

    rows = central_difference_rows(
        base_values=base,
        analytic_grad=analytic,
        directions=_directions(args.directions, base.numel(), args.seed + 4000, env.torch_device, base.dtype),
        epsilons=epsilons,
        evaluate=evaluate,
        assign=assign,
    )
    return {
        "loss": float(loss.detach().cpu()),
        "metrics": metrics,
        "analytic_grad_norm": finite_float(float(analytic.to(torch.float64).norm().detach().cpu())),
        "best": _best(rows),
        "epsilon_sweep": rows,
    }


def _check_one_step_at_state(env, q_base, qd_base, action_base, args, epsilons, seed_offset: int) -> dict:
    action_req = action_base.detach().clone().requires_grad_(True)
    loss, metrics = one_step_action_loss(
        env,
        q0=q_base.detach(),
        qd0=qd_base.detach(),
        action=action_req,
        termination_penalty=args.termination_penalty,
    )
    loss.backward()
    action_grad = torch.nan_to_num(action_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    action_flat = action_base.detach().reshape(-1)
    action_mutable = action_flat.clone()

    def assign_action(values: torch.Tensor) -> None:
        action_mutable.copy_(values)

    def evaluate_action() -> float:
        with torch.no_grad():
            value, _ = one_step_action_loss(
                env,
                q0=q_base,
                qd0=qd_base,
                action=action_mutable.view_as(action_base),
                termination_penalty=args.termination_penalty,
            )
        return float(value.detach().cpu())

    action_rows = central_difference_rows(
        base_values=action_flat,
        analytic_grad=action_grad,
        directions=_directions(args.directions, action_flat.numel(), args.seed + seed_offset, env.torch_device, action_flat.dtype),
        epsilons=epsilons,
        evaluate=evaluate_action,
        assign=assign_action,
    )

    q_req = q_base.detach().clone().requires_grad_(True)
    qd_req = qd_base.detach().clone().requires_grad_(True)
    action_detached = action_base.detach()
    state_loss, state_metrics = one_step_action_loss(
        env,
        q0=q_req,
        qd0=qd_req,
        action=action_detached,
        termination_penalty=args.termination_penalty,
    )
    state_loss.backward()
    state_grad = torch.cat(
        [
            torch.nan_to_num(q_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(qd_req.grad.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
        ]
    )
    state_flat = torch.cat([q_base.detach().reshape(-1), qd_base.detach().reshape(-1)])
    state_mutable = state_flat.clone()
    q_count = q_base.numel()

    def assign_state(values: torch.Tensor) -> None:
        state_mutable.copy_(values)

    def evaluate_state() -> float:
        with torch.no_grad():
            value, _ = one_step_action_loss(
                env,
                q0=state_mutable[:q_count].view_as(q_base),
                qd0=state_mutable[q_count:].view_as(qd_base),
                action=action_detached,
                termination_penalty=args.termination_penalty,
            )
        return float(value.detach().cpu())

    state_rows = central_difference_rows(
        base_values=state_flat,
        analytic_grad=state_grad,
        directions=_directions(args.directions, state_flat.numel(), args.seed + seed_offset + 10000, env.torch_device, state_flat.dtype),
        epsilons=epsilons,
        evaluate=evaluate_state,
        assign=assign_state,
    )
    component_masks = {
        "root_pos_q": torch.zeros_like(state_flat),
        "root_quat_q": torch.zeros_like(state_flat),
        "joint_q": torch.zeros_like(state_flat),
        "root_qd": torch.zeros_like(state_flat),
        "joint_qd": torch.zeros_like(state_flat),
    }
    for env_id in range(env.num_envs):
        q_offset = env_id * env.q_dim
        qd_offset = q_count + env_id * env.qd_dim
        component_masks["root_pos_q"][q_offset : q_offset + 3] = 1.0
        component_masks["root_quat_q"][q_offset + 3 : q_offset + 7] = 1.0
        component_masks["joint_q"][q_offset + 7 : q_offset + env.q_dim] = 1.0
        component_masks["root_qd"][qd_offset : qd_offset + 6] = 1.0
        component_masks["joint_qd"][qd_offset + 6 : qd_offset + env.qd_dim] = 1.0
    component_rows = {}
    for idx, (name, mask) in enumerate(component_masks.items()):
        component_rows[name] = central_difference_rows(
            base_values=state_flat,
            analytic_grad=state_grad,
            directions=masked_random_directions(
                count=args.directions,
                width=state_flat.numel(),
                mask=mask,
                seed=args.seed + seed_offset + 20000 + idx,
                device=env.torch_device,
                dtype=state_flat.dtype,
            ),
            epsilons=epsilons,
            evaluate=evaluate_state,
            assign=assign_state,
        )

    return {
        "action": {
            "loss": float(loss.detach().cpu()),
            "metrics": metrics,
            "analytic_grad_norm": finite_float(float(action_grad.to(torch.float64).norm().detach().cpu())),
            "best": _best(action_rows),
            "epsilon_sweep": action_rows,
        },
        "state": {
            "loss": float(state_loss.detach().cpu()),
            "metrics": state_metrics,
            "analytic_grad_norm": finite_float(float(state_grad.to(torch.float64).norm().detach().cpu())),
            "best": _best(state_rows),
            "epsilon_sweep": state_rows,
            "components": component_rows,
        },
    }


def _local_step_checks(env, q0, qd0, actions_base, args, epsilons) -> list[dict]:
    q = q0.detach()
    qd = qd0.detach()
    rows = []
    for step, action in enumerate(actions_base):
        checks = _check_one_step_at_state(env, q, qd, action.detach(), args, epsilons, seed_offset=5000 + step * 31)
        rows.append(
            {
                "step": step,
                "action_best": checks["action"]["best"],
                "state_best": checks["state"]["best"],
                "state_component_best": {
                    name: _best(rows) for name, rows in checks["state"]["components"].items()
                },
                "action_metrics": checks["action"]["metrics"],
                "state_metrics": checks["state"]["metrics"],
            }
        )
        with torch.no_grad():
            q, qd, _, _, _ = _step_reward(env, q, qd, action.detach(), args)
            q = q.detach()
            qd = qd.detach()
    return rows


def _check_policy(env, actor, q0, qd0, prev0, obs_stats, args, epsilons):
    params = trainable_parameters(actor)
    base = flatten_parameters(params)
    actor.zero_grad(set_to_none=True)
    loss, _, metrics = _rollout_policy(env, actor, q0, qd0, prev0, obs_stats, args)
    loss.backward()
    analytic = flatten_gradients(params)
    mutable = base.clone()

    def assign(values: torch.Tensor) -> None:
        mutable.copy_(values)
        assign_flat_parameters(params, mutable)

    def evaluate() -> float:
        with torch.no_grad():
            value, _, _ = _rollout_policy(env, actor, q0, qd0, prev0, obs_stats, args)
        return float(value.detach().cpu())

    rows = central_difference_rows(
        base_values=base,
        analytic_grad=analytic,
        directions=_directions(args.directions, base.numel(), args.seed + 2000, env.torch_device, base.dtype),
        epsilons=epsilons,
        evaluate=evaluate,
        assign=assign,
    )
    return {
        "loss": float(loss.detach().cpu()),
        "metrics": metrics,
        "analytic_grad_norm": finite_float(float(analytic.to(torch.float64).norm().detach().cpu())),
        "best": _best(rows),
        "epsilon_sweep": rows,
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
        sim_substeps=args.sim_substeps,
        mujoco_integrator=args.mujoco_integrator,
        ant_asset=args.ant_asset,
        ant_contact_mu=args.ant_contact_mu,
        ant_contact_margin=args.ant_contact_margin,
        ant_contact_gap=args.ant_contact_gap,
        ant_disable_joint_limits=args.ant_disable_joint_limits,
        ant_density_override=args.ant_density_override,
        ant_joint_damping=args.ant_joint_damping,
        ant_armature=args.ant_armature,
        ant_start_height=args.ant_start_height,
        ant_start_joint_q=args.ant_start_joint_q,
        ant_reset_position_scale=args.ant_reset_position_scale,
        ant_reset_angle_scale=args.ant_reset_angle_scale,
        ant_reset_joint_scale=args.ant_reset_joint_scale,
        ant_reset_velocity_scale=args.ant_reset_velocity_scale,
        ant_termination_height=args.ant_termination_height,
        ant_max_healthy_height=args.ant_max_healthy_height,
        ant_observation_style=args.ant_observation_style,
        ant_reward_style=args.ant_reward_style,
        ant_dof_limit_mode=args.ant_dof_limit_mode,
        ant_action_order=args.ant_action_order,
        mujoco_smooth_adjoint=args.mujoco_smooth_adjoint,
        mujoco_smooth_friction_viscosity=args.mujoco_smooth_friction_viscosity,
        mujoco_smooth_friction_scale=args.mujoco_smooth_friction_scale,
        mujoco_smooth_friction_bypass_kf=args.mujoco_smooth_friction_bypass_kf,
        mujoco_smooth_penalty_damping_alpha=args.mujoco_smooth_penalty_damping_alpha,
        mujoco_smooth_friction_surrogate_alpha=args.mujoco_smooth_friction_surrogate_alpha,
        ant_reward=AntRewardWeights(
            progress=args.ant_progress_weight,
            heading=args.ant_heading_weight,
            up=args.ant_up_weight,
            height=args.ant_height_weight,
            alive=args.ant_alive_reward,
            actions_cost=args.ant_actions_cost,
            energy_cost=args.ant_energy_cost,
            dof_limit_cost=args.ant_dof_limit_cost,
            dof_vel_scale=args.ant_dof_vel_scale,
        ),
        hopper_reward=HopperRewardWeights(),
        cheetah_reward=CheetahRewardWeights(),
        contact_reward=ContactTargetRewardWeights(),
    )
    actor = make_actor(
        env,
        stochastic=False,
        hidden_dims=args.actor_hidden_dims,
        actor_logstd_init=args.actor_logstd_init,
        actor_layer_norm=args.actor_layer_norm,
        action_squash=args.action_squash,
    )
    load_actor_checkpoint(actor, args.actor_path, env.torch_device)
    obs_stats = load_obs_rms(args.obs_rms_path, env.torch_device) if args.obs_rms_path is not None else None
    q0, qd0 = env.reset(noise=0.0, stochastic_init=False)
    prev0 = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    epsilons = args.eps or list(DEFAULT_GRAD_CHECK_EPS)
    with torch.no_grad():
        _, actions, _ = _rollout_policy(env, actor, q0, qd0, prev0, obs_stats, args)
        actions_base = torch.stack([action.detach() for action in actions], dim=0)
    result = {
        "mode": "ant_multistep_gradient_diagnostics",
        "timestamp_pacific": pacific_now_iso(),
        "newton_commit": git_commit_for_imported_module(newton),
        "mujoco_warp_commit": git_commit_for_imported_module(mujoco_warp),
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "contact_backend": args.contact_backend,
        "dt": args.dt,
        "sim_substeps": args.sim_substeps,
        "mujoco_integrator": args.mujoco_integrator,
        "force_scale": args.force_scale,
        "ant_asset": args.ant_asset,
        "ant_contact_margin": args.ant_contact_margin,
        "ant_contact_gap": args.ant_contact_gap,
        "ant_disable_joint_limits": args.ant_disable_joint_limits,
        "ant_density_override": args.ant_density_override,
        "ant_joint_damping": args.ant_joint_damping,
        "ant_armature": args.ant_armature,
        "ant_max_healthy_height": args.ant_max_healthy_height,
        "ant_dof_limit_mode": args.ant_dof_limit_mode,
        "ant_observation_style": args.ant_observation_style,
        "ant_reward_style": args.ant_reward_style,
        "ant_action_order": args.ant_action_order,
        "ant_reset_position_scale": args.ant_reset_position_scale,
        "ant_reset_angle_scale": args.ant_reset_angle_scale,
        "ant_reset_joint_scale": args.ant_reset_joint_scale,
        "ant_reset_velocity_scale": args.ant_reset_velocity_scale,
        "mujoco_smooth_adjoint": args.mujoco_smooth_adjoint,
        "mujoco_smooth_friction_viscosity": args.mujoco_smooth_friction_viscosity,
        "mujoco_smooth_friction_scale": args.mujoco_smooth_friction_scale,
        "mujoco_smooth_friction_bypass_kf": args.mujoco_smooth_friction_bypass_kf,
        "mujoco_smooth_penalty_damping_alpha": args.mujoco_smooth_penalty_damping_alpha,
        "mujoco_smooth_friction_surrogate_alpha": args.mujoco_smooth_friction_surrogate_alpha,
        "termination_penalty": args.termination_penalty,
        "ant_termination_height": args.ant_termination_height,
        "epsilon_values": epsilons,
        "directions": args.directions,
        "policy": _check_policy(env, actor, q0, qd0, prev0, obs_stats, args, epsilons),
        "fixed_action_sequence": _check_actions(env, q0, qd0, actions_base, args, epsilons),
        "initial_state_fixed_actions": _check_initial_state(env, q0, qd0, actions_base, args, epsilons),
        "local_one_step_checks": _local_step_checks(env, q0, qd0, actions_base, args, epsilons),
        "gpu": query_gpu(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, result)
    print(f"wrote multistep diagnostics to {out_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--eps", type=parse_float_list, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--actor-path", type=Path, required=True)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--actor-hidden-dims", type=parse_int_list, default=[256, 128, 64])
    parser.add_argument("--actor-logstd-init", type=float, default=0.0)
    parser.add_argument("--action-squash", choices=["tanh", "none"], default="tanh")
    parser.add_argument("--actor-layer-norm", dest="actor_layer_norm", action="store_true", default=True)
    parser.add_argument("--no-actor-layer-norm", dest="actor_layer_norm", action="store_false")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--sim-substeps", type=int, default=2)
    parser.add_argument("--mujoco-integrator", choices=["euler", "rk4", "implicitfast", "implicit"], default="implicitfast")
    parser.add_argument("--force-scale", type=float, default=10.0)
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default="mujoco")
    parser.add_argument("--mujoco-smooth-adjoint", choices=["off", "smooth", "free_body", "surrogate"], default="off")
    parser.add_argument("--mujoco-smooth-friction-viscosity", type=float, default=10.0)
    parser.add_argument("--mujoco-smooth-friction-scale", type=float, default=0.01)
    parser.add_argument("--mujoco-smooth-friction-bypass-kf", type=float, default=0.0)
    parser.add_argument("--mujoco-smooth-penalty-damping-alpha", type=float, default=0.0)
    parser.add_argument("--mujoco-smooth-friction-surrogate-alpha", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rew-scale", type=float, default=1.0)
    parser.add_argument("--termination-penalty", type=float, default=ANT_DEFAULT_TERMINATION_PENALTY)
    parser.add_argument("--ant-progress-weight", type=float, default=2.0)
    parser.add_argument("--ant-heading-weight", type=float, default=0.5)
    parser.add_argument("--ant-up-weight", type=float, default=0.1)
    parser.add_argument("--ant-height-weight", type=float, default=1.0)
    parser.add_argument("--ant-alive-reward", type=float, default=0.5)
    parser.add_argument("--ant-actions-cost", type=float, default=0.005)
    parser.add_argument("--ant-energy-cost", type=float, default=0.05)
    parser.add_argument("--ant-dof-limit-cost", type=float, default=1.0)
    parser.add_argument("--ant-dof-vel-scale", type=float, default=0.2)
    parser.add_argument("--ant-asset", choices=["diffrl", "nv"], default="diffrl")
    parser.add_argument("--ant-contact-mu", type=float, default=1.0)
    parser.add_argument("--ant-contact-margin", type=float, default=0.0)
    parser.add_argument("--ant-contact-gap", type=float, default=None)
    parser.add_argument("--ant-disable-joint-limits", action="store_true")
    parser.add_argument("--ant-density-override", type=float, default=None)
    parser.add_argument("--ant-joint-damping", type=float, default=0.1)
    parser.add_argument("--ant-armature", type=float, default=None)
    parser.add_argument("--ant-start-height", type=float, default=0.5)
    parser.add_argument("--ant-start-joint-q", type=parse_float_list, default=(0, 0.785398, 0, -0.785398, 0, -0.785398, 0, 0.785398))
    parser.add_argument("--ant-reset-position-scale", type=float, default=0.1)
    parser.add_argument("--ant-reset-angle-scale", type=float, default=math.pi / 24.0)
    parser.add_argument("--ant-reset-joint-scale", type=float, default=0.2)
    parser.add_argument("--ant-reset-velocity-scale", type=float, default=0.25)
    parser.add_argument("--ant-termination-height", type=float, default=0.31)
    parser.add_argument("--ant-max-healthy-height", type=float, default=1.5)
    parser.add_argument("--ant-observation-style", choices=["isaac", "diffrl"], default="isaac")
    parser.add_argument(
        "--ant-reward-style",
        choices=["isaac", "isaaclab", "isaac_heading_gated", "diffrl"],
        default="isaac",
    )
    parser.add_argument("--ant-dof-limit-mode", choices=["abs", "upper"], default="abs")
    parser.add_argument("--ant-action-order", choices=["joint", "actuator"], default="joint")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
