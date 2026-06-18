#!/usr/bin/env python3
"""Compare one saved Ant policy under two MJWarp world counts."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

from render_policy_rollout import build_env, default_obs_rms_path, detect_algo, load_obs_rms, load_run_json
from run_ant_rollout_trace import _load_actor, contact_stats
from run_newton_shac import deterministic_policy_action, normalize_obs, pacific_now_iso, write_json


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def apply_overrides(result: dict, args: argparse.Namespace) -> dict:
    result = dict(result)
    if args.contact_backend is not None:
        result["contact_backend"] = args.contact_backend
        result.pop("eval_contact_backend", None)
    if args.sim_substeps is not None:
        result["sim_substeps"] = args.sim_substeps
    if args.mujoco_integrator is not None:
        result["mujoco_integrator"] = args.mujoco_integrator
    if args.mujoco_world_spacing_z is not None:
        result["mujoco_world_spacing_z"] = args.mujoco_world_spacing_z
    return result


@torch.no_grad()
def compare_rollouts(env_a, env_b, actor, obs_rms, horizon: int, thresholds: list[float]) -> dict:
    q_a, qd_a = env_a.reset(noise=0.0, stochastic_init=False)
    q_b, qd_b = env_b.reset(noise=0.0, stochastic_init=False)
    prev_a = torch.zeros((env_a.num_envs, env_a.num_actions), dtype=torch.float32, device=env_a.torch_device)
    prev_b = torch.zeros((env_b.num_envs, env_b.num_actions), dtype=torch.float32, device=env_b.torch_device)
    progress_a = torch.zeros(env_a.num_envs, dtype=torch.long, device=env_a.torch_device)
    progress_b = torch.zeros(env_b.num_envs, dtype=torch.long, device=env_b.torch_device)
    active_a = torch.ones(env_a.num_envs, dtype=torch.bool, device=env_a.torch_device)
    active_b = torch.ones(env_b.num_envs, dtype=torch.bool, device=env_b.torch_device)
    threshold_steps: dict[str, int | None] = {f"{threshold:g}": None for threshold in thresholds}
    rows = []

    for step_idx in range(horizon):
        obs_a = normalize_obs(env_a.observe(q_a, qd_a, prev_a, phase=progress_a), obs_rms)
        obs_b = normalize_obs(env_b.observe(q_b, qd_b, prev_b, phase=progress_b), obs_rms)
        act_a = deterministic_policy_action(actor, obs_a)
        act_b = deterministic_policy_action(actor, obs_b)
        act_a = torch.where(active_a.unsqueeze(-1), act_a, torch.zeros_like(act_a))
        act_b = torch.where(active_b.unsqueeze(-1), act_b, torch.zeros_like(act_b))
        q_next_a, qd_next_a = env_a.step(q_a, qd_a, env_a.action_to_joint_f(act_a))
        q_next_b, qd_next_b = env_b.step(q_b, qd_b, env_b.action_to_joint_f(act_b))

        q_diff = (q_next_a[0] - q_next_b[0]).abs()
        qd_diff = (qd_next_a[0] - qd_next_b[0]).abs()
        action_diff = (act_a[0] - act_b[0]).abs()
        max_diff = max(float(q_diff.max().cpu()), float(qd_diff.max().cpu()), float(action_diff.max().cpu()))
        for threshold in thresholds:
            key = f"{threshold:g}"
            if threshold_steps[key] is None and max_diff > threshold:
                threshold_steps[key] = step_idx + 1

        torso_a, _, _, _, up_a, heading_a = env_a.ant_pose_terms(q_next_a[:1], qd_next_a[:1])
        torso_b, _, _, _, up_b, heading_b = env_b.ant_pose_terms(q_next_b[:1], qd_next_b[:1])
        stats_a = contact_stats(env_a)
        stats_b = contact_stats(env_b)
        rows.append(
            {
                "step": step_idx + 1,
                "max_abs_q0_diff": float(q_diff.max().cpu()),
                "max_abs_qd0_diff": float(qd_diff.max().cpu()),
                "max_abs_action0_diff": float(action_diff.max().cpu()),
                "height_a": float(torso_a[0, 1].cpu()),
                "height_b": float(torso_b[0, 1].cpu()),
                "up_a": float(up_a[0, 1].cpu()),
                "up_b": float(up_b[0, 1].cpu()),
                "heading_a": float(heading_a[0].cpu()),
                "heading_b": float(heading_b[0].cpu()),
                "nacon_a": stats_a.get("nacon"),
                "nacon_b": stats_b.get("nacon"),
                "nefc_max_a": stats_a.get("nefc_max"),
                "nefc_max_b": stats_b.get("nefc_max"),
                "active_contacts_per_world_max_a": stats_a.get("active_contacts_per_world_max"),
                "active_contacts_per_world_max_b": stats_b.get("active_contacts_per_world_max"),
            }
        )

        finite_a = torch.isfinite(q_next_a).all(dim=-1) & torch.isfinite(qd_next_a).all(dim=-1)
        finite_b = torch.isfinite(q_next_b).all(dim=-1) & torch.isfinite(qd_next_b).all(dim=-1)
        invalid_a = env_a.invalid_state(q_next_a, qd_next_a) | ~finite_a
        invalid_b = env_b.invalid_state(q_next_b, qd_next_b) | ~finite_b
        fell_a = env_a.fallen_state(q_next_a) & ~invalid_a
        fell_b = env_b.fallen_state(q_next_b) & ~invalid_b
        active_a = active_a & ~(fell_a | invalid_a)
        active_b = active_b & ~(fell_b | invalid_b)
        q_a = torch.where(invalid_a.unsqueeze(-1), q_a, q_next_a)
        q_b = torch.where(invalid_b.unsqueeze(-1), q_b, q_next_b)
        qd_a = torch.where(invalid_a.unsqueeze(-1), torch.zeros_like(qd_a), qd_next_a)
        qd_b = torch.where(invalid_b.unsqueeze(-1), torch.zeros_like(qd_b), qd_next_b)
        prev_a = act_a
        prev_b = act_b
        progress_a = torch.where(active_a, progress_a + 1, progress_a)
        progress_b = torch.where(active_b, progress_b + 1, progress_b)

    return {
        "timestamp_pacific": pacific_now_iso(),
        "num_envs_a": env_a.num_envs,
        "num_envs_b": env_b.num_envs,
        "horizon": horizon,
        "threshold_steps": threshold_steps,
        "final": rows[-1] if rows else None,
        "steps": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--actor-path", type=Path, required=True)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algo", choices=["auto", "shac", "ppo"], default="auto")
    parser.add_argument("--num-envs-a", type=int, default=1023)
    parser.add_argument("--num-envs-b", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--mujoco-integrator", choices=["euler", "rk4", "implicitfast", "implicit"], default=None)
    parser.add_argument("--mujoco-world-spacing-z", type=float, default=None)
    parser.add_argument("--thresholds", type=parse_thresholds, default=[1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    args = parser.parse_args()

    wp.init()
    result_path = args.result_json.resolve()
    result = apply_overrides(load_run_json(result_path), args)
    if result.get("env") != "ant":
        raise ValueError("run_ant_nworld_compare.py only supports Ant result JSONs")
    algo = detect_algo(result, result_path) if args.algo == "auto" else args.algo
    env_a = build_env(result, SimpleNamespace(device=args.device, video_num_envs=args.num_envs_a))
    env_b = build_env(result, SimpleNamespace(device=args.device, video_num_envs=args.num_envs_b))
    actor, actor_path = _load_actor(result, result_path, env_a, algo, args.actor_path)
    obs_rms_path = args.obs_rms_path or default_obs_rms_path(result_path.parent, "ant", algo)
    obs_rms = load_obs_rms(obs_rms_path, env_a.torch_device, env_a.num_obs) if obs_rms_path and obs_rms_path.exists() else None
    out = compare_rollouts(env_a, env_b, actor, obs_rms, args.horizon, args.thresholds)
    out.update(
        {
            "source_result": str(result_path),
            "algo": algo,
            "actor_path": str(actor_path),
            "obs_rms_path": str(obs_rms_path) if obs_rms_path else None,
            "contact_backend": result.get("contact_backend"),
            "mujoco_integrator": result.get("mujoco_integrator"),
            "sim_substeps": result.get("sim_substeps"),
            "mujoco_world_spacing_z": result.get("mujoco_world_spacing_z"),
            "world_spacing_a": list(env_a.world_spacing) if env_a.world_spacing is not None else None,
            "world_spacing_b": list(env_b.world_spacing) if env_b.world_spacing is not None else None,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, out)
    print(f"wrote {args.out} thresholds={out['threshold_steps']} final={out['final']}")


if __name__ == "__main__":
    main()
