#!/usr/bin/env python3
"""Trace deterministic no-reset Ant rollouts from saved report policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from render_policy_rollout import (
    build_env,
    default_actor_path,
    default_obs_rms_path,
    detect_algo,
    load_obs_rms,
    load_run_json,
    ppo_actor_from_result,
)
from run_newton_shac import deterministic_policy_action, load_actor_checkpoint, make_actor, normalize_obs, pacific_now_iso, write_json


def _load_actor(result: dict, result_path: Path, env, algo: str, actor_path: Path | None):
    run_dir = result_path.parent
    path = actor_path or default_actor_path(run_dir, "ant", algo)
    if path is None:
        raise FileNotFoundError(f"no saved actor found in {run_dir}")
    if algo == "ppo":
        actor = ppo_actor_from_result(result, env)
        actor.load_state_dict(torch.load(path, map_location=env.torch_device))
    else:
        actor = make_actor(
            env,
            stochastic=bool(result.get("stochastic_actor") or False),
            hidden_dims=result.get("actor_hidden_dims"),
            actor_logstd_init=float(result.get("actor_logstd_init") or -1.0),
            actor_layer_norm=bool(result.get("actor_layer_norm", True)),
            action_squash=result.get("action_squash") or "tanh",
        )
        load_actor_checkpoint(actor, path, env.torch_device)
    actor.eval()
    return actor, path


def contact_stats(env) -> dict:
    try:
        data = env.solver.mjw_data
        nacon = int(data.nacon.numpy()[0])
    except Exception:
        return {}

    naconmax = int(getattr(data, "naconmax", 0))
    stats = {
        "nacon": nacon,
        "naconmax": naconmax,
    }
    if naconmax <= 0:
        return stats

    usable = min(nacon, naconmax)
    try:
        nefc = np.asarray(data.nefc.numpy(), dtype=np.int64)
        stats.update(
            {
                "nefc_min": int(nefc.min()) if nefc.size else 0,
                "nefc_mean": float(nefc.mean()) if nefc.size else 0.0,
                "nefc_max": int(nefc.max()) if nefc.size else 0,
            }
        )
    except Exception:
        pass

    if usable <= 0:
        stats.update(
            {
                "contacts_per_world_min": 0,
                "contacts_per_world_mean": 0.0,
                "contacts_per_world_max": 0,
                "active_contacts_per_world_min": 0,
                "active_contacts_per_world_mean": 0.0,
                "active_contacts_per_world_max": 0,
            }
        )
        return stats

    try:
        worldid = np.asarray(data.contact.worldid.numpy()[:usable], dtype=np.int64)
        dist = np.asarray(data.contact.dist.numpy()[:usable], dtype=np.float64)
        includemargin = np.asarray(data.contact.includemargin.numpy()[:usable], dtype=np.float64)
        contact_type = np.asarray(data.contact.type.numpy()[:usable], dtype=np.int64)
    except Exception:
        return stats

    valid_world = (worldid >= 0) & (worldid < env.num_envs)
    constraint = (contact_type & 1) != 0
    active = valid_world & constraint & (dist < includemargin)
    detected = valid_world & constraint
    detected_counts = np.bincount(worldid[detected], minlength=env.num_envs)
    active_counts = np.bincount(worldid[active], minlength=env.num_envs)
    stats.update(
        {
            "contacts_per_world_min": int(detected_counts.min()) if detected_counts.size else 0,
            "contacts_per_world_mean": float(detected_counts.mean()) if detected_counts.size else 0.0,
            "contacts_per_world_max": int(detected_counts.max()) if detected_counts.size else 0,
            "active_contacts_per_world_min": int(active_counts.min()) if active_counts.size else 0,
            "active_contacts_per_world_mean": float(active_counts.mean()) if active_counts.size else 0.0,
            "active_contacts_per_world_max": int(active_counts.max()) if active_counts.size else 0,
        }
    )
    return stats


@torch.no_grad()
def trace_rollout(env, actor, obs_rms, horizon: int, *, contact_stats_interval: int | None = 1) -> dict:
    q, qd = env.reset(noise=0.0, stochastic_init=False)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.torch_device)
    terminal_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.torch_device)
    terminal_height = torch.full((env.num_envs,), float("nan"), dtype=torch.float32, device=env.torch_device)
    terminal_up = torch.full((env.num_envs,), float("nan"), dtype=torch.float32, device=env.torch_device)
    terminal_x = torch.full((env.num_envs,), float("nan"), dtype=torch.float32, device=env.torch_device)
    terminal_reason = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    forward_displacement = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
    step_rows = []

    for step_idx in range(horizon):
        obs = normalize_obs(env.observe(q, qd, prev_action, phase=progress), obs_rms)
        action = deterministic_policy_action(actor, obs)
        action = torch.where(active.unsqueeze(-1), action, torch.zeros_like(action))
        root_x_before = q[:, 0].clone()
        q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
        finite = torch.logical_and(torch.isfinite(q_next).all(dim=-1), torch.isfinite(qd_next).all(dim=-1))
        invalid = torch.logical_or(env.invalid_state(q_next, qd_next), ~finite)
        fell = torch.logical_and(env.fallen_state(q_next), ~invalid)
        new_terminal = torch.logical_and(active, torch.logical_or(fell, invalid))

        torso_pos, _, _, _, up_vec, heading = env.ant_pose_terms(q_next, qd_next)
        forward_displacement = forward_displacement + torch.where(
            active & finite,
            q_next[:, 0] - root_x_before,
            torch.zeros_like(root_x_before),
        )
        active_finite = active & finite
        if active_finite.any():
            heights = torso_pos[active_finite, 1]
            ups = up_vec[active_finite, 1]
            headings = heading[active_finite]
            qd_active = qd_next[active_finite]
            action_active = action[active_finite]
            row = {
                "step": step_idx + 1,
                "active_count": int(active.sum().cpu()),
                "new_falls": int(torch.logical_and(active, fell).sum().cpu()),
                "new_invalid": int(torch.logical_and(active, invalid).sum().cpu()),
                "mean_height": float(heights.mean().cpu()),
                "min_height": float(heights.min().cpu()),
                "mean_up": float(ups.mean().cpu()),
                "min_up": float(ups.min().cpu()),
                "mean_heading": float(headings.mean().cpu()),
                "min_heading": float(headings.min().cpu()),
                "mean_x": float(q_next[active_finite, 0].mean().cpu()),
                "mean_dx": float(forward_displacement[active_finite].mean().cpu()),
                "mean_abs_action": float(action_active.abs().mean().cpu()),
                "max_abs_action": float(action_active.abs().max().cpu()),
                "max_abs_qd": float(qd_active.abs().max().cpu()),
            }
        else:
            row = {
                "step": step_idx + 1,
                "active_count": 0,
                "new_falls": int(torch.logical_and(active, fell).sum().cpu()),
                "new_invalid": int(torch.logical_and(active, invalid).sum().cpu()),
            }
        if contact_stats_interval is not None and contact_stats_interval > 0 and (step_idx + 1) % contact_stats_interval == 0:
            row.update(contact_stats(env))
        step_rows.append(row)

        terminal_step = torch.where(new_terminal, torch.full_like(terminal_step, step_idx + 1), terminal_step)
        terminal_height = torch.where(new_terminal, torso_pos[:, 1], terminal_height)
        terminal_up = torch.where(new_terminal, up_vec[:, 1], terminal_up)
        terminal_x = torch.where(new_terminal, q_next[:, 0], terminal_x)
        terminal_reason = torch.where(torch.logical_and(new_terminal, fell), torch.ones_like(terminal_reason), terminal_reason)
        terminal_reason = torch.where(
            torch.logical_and(new_terminal, invalid),
            torch.full_like(terminal_reason, 2),
            terminal_reason,
        )
        active = torch.logical_and(active, ~torch.logical_or(fell, invalid))
        freeze = torch.logical_or(invalid, ~finite)
        q = torch.where(freeze.unsqueeze(-1), q, q_next)
        qd = torch.where(freeze.unsqueeze(-1), torch.zeros_like(qd), qd_next)
        prev_action = action
        progress = torch.where(active, progress + 1, progress)

    terminal_mask = terminal_step >= 0
    terminal_indices = terminal_mask.nonzero(as_tuple=False).squeeze(-1)
    terminal_events = []
    for idx in terminal_indices.detach().cpu().tolist():
        reason = "fall" if int(terminal_reason[idx].cpu()) == 1 else "invalid"
        terminal_events.append(
            {
                "env": int(idx),
                "step": int(terminal_step[idx].cpu()),
                "reason": reason,
                "height": float(terminal_height[idx].cpu()),
                "up": float(terminal_up[idx].cpu()),
                "x": float(terminal_x[idx].cpu()),
                "forward_displacement": float(forward_displacement[idx].cpu()),
            }
        )

    terminal_steps = terminal_step[terminal_mask]
    contact_rows = [row for row in step_rows if "nacon" in row]
    max_contact_summary = {}
    if contact_rows:
        for key in (
            "nacon",
            "contacts_per_world_max",
            "active_contacts_per_world_max",
            "nefc_max",
        ):
            values = [row[key] for row in contact_rows if key in row]
            if values:
                max_contact_summary[f"max_{key}"] = max(values)

    return {
        "timestamp_pacific": pacific_now_iso(),
        "num_envs": env.num_envs,
        "horizon": horizon,
        "terminal_count": int(terminal_mask.sum().cpu()),
        "fall_count": int((terminal_reason == 1).sum().cpu()),
        "invalid_count": int((terminal_reason == 2).sum().cpu()),
        "first_terminal_step": int(terminal_steps.min().cpu()) if terminal_steps.numel() else None,
        "mean_terminal_step": float(terminal_steps.to(torch.float32).mean().cpu()) if terminal_steps.numel() else None,
        "mean_forward_displacement": float(forward_displacement.mean().cpu()),
        "min_forward_displacement": float(forward_displacement.min().cpu()),
        "max_forward_displacement": float(forward_displacement.max().cpu()),
        "terminal_events": terminal_events,
        "contact_stats_interval": contact_stats_interval,
        "contact_summary": max_contact_summary,
        "steps": step_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algo", choices=["auto", "shac", "ppo"], default="auto")
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--mujoco-integrator", choices=["euler", "rk4", "implicitfast", "implicit"], default=None)
    parser.add_argument("--mujoco-world-spacing-z", type=float, default=None)
    parser.add_argument("--contact-stats-interval", type=int, default=1)
    args = parser.parse_args()

    wp.init()
    result_path = args.result_json.resolve()
    result = load_run_json(result_path)
    if result.get("env") != "ant":
        raise ValueError("run_ant_rollout_trace.py only supports Ant result JSONs")
    if args.contact_backend is not None:
        result["contact_backend"] = args.contact_backend
        result.pop("eval_contact_backend", None)
    if args.sim_substeps is not None:
        result["sim_substeps"] = args.sim_substeps
    if args.mujoco_integrator is not None:
        result["mujoco_integrator"] = args.mujoco_integrator
    if args.mujoco_world_spacing_z is not None:
        result["mujoco_world_spacing_z"] = args.mujoco_world_spacing_z
    algo = detect_algo(result, result_path) if args.algo == "auto" else args.algo
    env = build_env(result, SimpleNamespace(device=args.device, video_num_envs=args.num_envs))
    actor, actor_path = _load_actor(result, result_path, env, algo, args.actor_path)
    obs_rms_path = args.obs_rms_path or default_obs_rms_path(result_path.parent, "ant", algo)
    obs_rms = load_obs_rms(obs_rms_path, env.torch_device, env.num_obs) if obs_rms_path and obs_rms_path.exists() else None
    horizon = args.horizon or int(result.get("eval_horizon") or result.get("selection_horizon") or 480)
    stats_interval = args.contact_stats_interval if args.contact_stats_interval > 0 else None
    trace = trace_rollout(env, actor, obs_rms, horizon, contact_stats_interval=stats_interval)
    trace.update(
        {
            "source_result": str(result_path),
            "algo": algo,
            "actor_path": str(actor_path),
            "obs_rms_path": str(obs_rms_path) if obs_rms_path else None,
            "contact_backend": result.get("contact_backend"),
            "mujoco_integrator": result.get("mujoco_integrator"),
            "sim_substeps": result.get("sim_substeps"),
            "mujoco_world_spacing_z": result.get("mujoco_world_spacing_z"),
            "world_spacing": list(env.world_spacing) if env.world_spacing is not None else None,
            "force_scale": result.get("force_scale"),
            "ant_reward": result.get("ant_reward"),
            "ant_reward_style": result.get("ant_reward_style"),
            "ant_observation_style": result.get("ant_observation_style"),
            "ant_termination_height": result.get("ant_termination_height"),
            "ant_min_up": result.get("ant_min_up"),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, trace)
    print(
        f"wrote {args.out} terminals={trace['terminal_count']} "
        f"falls={trace['fall_count']} invalid={trace['invalid_count']} "
        f"dx={trace['mean_forward_displacement']:.3f}"
    )


if __name__ == "__main__":
    main()
