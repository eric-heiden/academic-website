from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from run_newton_shac import (
    AntRewardWeights,
    NewtonMuJoCoTorchEnv,
    ant_defaults_for_asset,
    deterministic_policy_action,
    normalize_obs,
    pacific_now_iso,
    write_json,
)


class ZeroActor(torch.nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return torch.zeros((obs.shape[0], self.action_dim), dtype=obs.dtype, device=obs.device)


class SineActor(torch.nn.Module):
    def __init__(self, action_dim: int, amplitude: float, period: int):
        super().__init__()
        self.action_dim = action_dim
        self.amplitude = amplitude
        self.period = max(1, period)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        phase = 2.0 * np.pi * float(int(self.step.item()) % self.period) / float(self.period)
        signs = torch.ones(self.action_dim, dtype=obs.dtype, device=obs.device)
        signs[1::2] = -1.0
        action = self.amplitude * torch.sin(torch.tensor(phase, dtype=obs.dtype, device=obs.device)) * signs
        self.step += 1
        return action.view(1, -1).repeat(obs.shape[0], 1)


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach()
    return {
        "mean": float(values.mean().cpu()),
        "min": float(values.min().cpu()),
        "max": float(values.max().cpu()),
    }


def safe_numpy(value: Any) -> Any:
    if value is None:
        return None
    try:
        array = value.numpy()
    except Exception:
        return None
    return np.asarray(array).copy()


def contact_snapshot(env: NewtonMuJoCoTorchEnv) -> dict[str, Any]:
    data = getattr(env.solver, "mjw_data", None)
    if data is None:
        return {}
    out: dict[str, Any] = {}
    active_contact_count = 0
    for name in ("ncon", "nacon", "nefc"):
        array = safe_numpy(getattr(data, name, None))
        if array is not None:
            flat_values = [int(x) for x in array.reshape(-1).tolist()]
            if name == "nacon":
                active_contact_count = int(np.sum(array))
                out[name] = active_contact_count
            elif name == "nefc":
                out[name] = {
                    "mean": float(np.mean(array)),
                    "min": int(np.min(array)),
                    "max": int(np.max(array)),
                }
            else:
                out[name] = int(np.sum(array)) if len(flat_values) > 1 else flat_values[0]
    contact = getattr(data, "contact", None)
    if contact is not None and active_contact_count > 0:
        worldid = safe_numpy(getattr(contact, "worldid", None))
        if worldid is not None:
            flat = worldid.reshape(-1)[:active_contact_count]
            valid = flat[flat >= 0]
            if valid.size:
                out["contact_world_counts"] = {
                    str(int(world)): int((valid == world).sum()) for world in np.unique(valid)
                }
    return out


def make_result_env(result: dict[str, Any], args: argparse.Namespace) -> NewtonMuJoCoTorchEnv:
    env_args = argparse.Namespace(video_num_envs=args.num_envs, device=args.device)
    result = dict(result)
    if args.contact_backend is not None:
        result["eval_contact_backend"] = args.contact_backend
    if args.ant_dof_limit_mode is not None:
        result["eval_ant_dof_limit_mode"] = args.ant_dof_limit_mode
    if args.mujoco_integrator is not None:
        result["mujoco_integrator"] = args.mujoco_integrator
    if args.sim_substeps is not None:
        result["sim_substeps"] = args.sim_substeps
    return build_env(result, env_args)


def make_default_ant_env(args: argparse.Namespace) -> NewtonMuJoCoTorchEnv:
    defaults = ant_defaults_for_asset(args.ant_asset)
    contact_backend = args.contact_backend or "mujoco"
    dof_limit_mode = args.ant_dof_limit_mode or defaults["dof_limit_mode"]
    return NewtonMuJoCoTorchEnv(
        env_name="ant",
        num_envs=args.num_envs,
        device=args.device,
        dt=args.dt,
        sim_substeps=args.sim_substeps or defaults["sim_substeps"],
        mujoco_integrator=args.mujoco_integrator or "euler",
        force_scale=args.force_scale if args.force_scale is not None else defaults["force_scale"],
        contact_backend=contact_backend,
        ant_asset=args.ant_asset,
        ant_contact_mu=defaults["contact_mu"],
        ant_joint_damping=defaults["joint_damping"],
        ant_armature=defaults["armature"],
        ant_start_height=defaults["start_height"],
        ant_start_joint_q=defaults["start_joint_q"],
        ant_termination_height=defaults["termination_height"],
        ant_observation_style=defaults["observation_style"],
        ant_reward_style=defaults["reward_style"],
        ant_dof_limit_mode=dof_limit_mode,
        ant_action_order=defaults["action_order"],
        ant_reward=AntRewardWeights(
            heading=defaults["heading_weight"],
            dof_limit_cost=defaults["dof_limit_cost"],
        ),
    )


def load_actor_and_obs(result: dict[str, Any], env: NewtonMuJoCoTorchEnv, args: argparse.Namespace) -> tuple[torch.nn.Module, Any]:
    run_dir = args.result_json.parent
    algo = detect_algo(result, args.result_json) if args.algo == "auto" else args.algo
    actor_path = args.actor_path or default_actor_path(run_dir, result["env"], algo)
    obs_rms_path = args.obs_rms_path or default_obs_rms_path(run_dir, result["env"], algo)
    if actor_path is None:
        raise FileNotFoundError(f"no actor checkpoint found in {run_dir}")
    if algo != "ppo":
        from run_newton_shac import load_actor_checkpoint, make_actor

        actor = make_actor(
            env,
            stochastic=bool(result.get("stochastic_actor") or False),
            hidden_dims=result.get("actor_hidden_dims"),
            actor_logstd_init=float(result.get("actor_logstd_init") or -1.0),
            actor_layer_norm=bool(result.get("actor_layer_norm", True)),
            action_squash=result.get("action_squash") or "tanh",
        )
        load_actor_checkpoint(actor, actor_path, env.torch_device)
    else:
        actor = ppo_actor_from_result(result, env)
        actor.load_state_dict(torch.load(actor_path, map_location=env.torch_device))
    actor.eval()
    obs_rms = load_obs_rms(obs_rms_path, env.torch_device, env.num_obs) if result.get("obs_rms") else None
    return actor, obs_rms


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure identical-clone divergence in Newton/MJWarp batched Ant.")
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--algo", choices=["auto", "ppo", "shac"], default="auto")
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=["checkpoint", "zero", "sine"], default="checkpoint")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--ant-asset", choices=["diffrl", "nv"], default="nv")
    parser.add_argument("--ant-dof-limit-mode", choices=["abs", "upper"], default=None)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--mujoco-integrator", choices=["euler", "rk4", "implicitfast", "implicit"], default=None)
    parser.add_argument("--sine-amplitude", type=float, default=0.5)
    parser.add_argument("--sine-period", type=int, default=60)
    parser.add_argument("--stochastic-init", action="store_true")
    args = parser.parse_args()

    wp.init()
    torch.manual_seed(0)
    np.random.seed(0)

    result = load_run_json(args.result_json) if args.result_json else None
    if result is not None:
        env = make_result_env(result, args)
    else:
        env = make_default_ant_env(args)

    if args.policy == "checkpoint":
        if result is None:
            raise ValueError("--policy checkpoint requires --result-json")
        actor, obs_rms = load_actor_and_obs(result, env, args)
    elif args.policy == "sine":
        actor, obs_rms = SineActor(env.num_actions, args.sine_amplitude, args.sine_period).to(env.torch_device), None
    else:
        actor, obs_rms = ZeroActor(env.num_actions).to(env.torch_device), None

    q, qd = env.reset(noise=0.0, stochastic_init=args.stochastic_init)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    q_reference = env.start_q[: env.num_envs].clone()

    samples = []
    first_threshold_step = {str(threshold): None for threshold in (1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)}
    start = time.perf_counter()
    peak_allocated = 0
    peak_reserved = 0
    if env.torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(env.torch_device)

    for step_idx in range(args.horizon):
        obs_raw = env.observe(q, qd, prev_action, phase=progress)
        obs = normalize_obs(obs_raw, obs_rms)
        action = deterministic_policy_action(actor, obs)
        q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
        progress = progress + 1
        next_obs_raw = env.observe(q_next, qd_next, action, phase=progress)
        invalid = env.invalid_state(q_next, qd_next)
        fallen = env.fallen_state(q_next)

        q_rel = q_next - q_reference
        q_ref = q_rel[0:1]
        qd_ref = qd_next[0:1]
        obs_ref = next_obs_raw[0:1]
        action_ref = action[0:1]
        q_error = (q_rel - q_ref).abs().amax()
        qd_error = (qd_next - qd_ref).abs().amax()
        obs_error = (next_obs_raw - obs_ref).abs().amax()
        action_error = (action - action_ref).abs().amax()
        max_state_error = torch.maximum(q_error, qd_error)
        max_state_error_float = float(max_state_error.detach().cpu())
        for threshold, value in first_threshold_step.items():
            if value is None and max_state_error_float > float(threshold):
                first_threshold_step[threshold] = step_idx + 1

        if env.torch_device.type == "cuda":
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(env.torch_device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(env.torch_device))

        if step_idx < 10 or (step_idx + 1) % 10 == 0 or step_idx == args.horizon - 1:
            torso_pos, _, _, _, up_vec, heading = env.ant_pose_terms(q_next, qd_next)
            sample = {
                "step": step_idx + 1,
                "max_q_delta_error": float(q_error.detach().cpu()),
                "max_qd_error": float(qd_error.detach().cpu()),
                "max_obs_error": float(obs_error.detach().cpu()),
                "max_action_error": float(action_error.detach().cpu()),
                "fallen_count": int(fallen.sum().detach().cpu()),
                "invalid_count": int(invalid.sum().detach().cpu()),
                "height": tensor_stats(torso_pos[:, 1]),
                "up": tensor_stats(up_vec[:, 1]),
                "heading": tensor_stats(heading[:, 0]),
            }
            sample.update(contact_snapshot(env))
            samples.append(sample)

        q = q_next
        qd = qd_next
        prev_action = action

    elapsed = time.perf_counter() - start
    q_rel_final = q - q_reference
    final_q_error_by_env = (q_rel_final - q_rel_final[0:1]).abs().amax(dim=1)
    final_qd_error_by_env = (qd - qd[0:1]).abs().amax(dim=1)
    worst_env = int(torch.maximum(final_q_error_by_env, final_qd_error_by_env).argmax().detach().cpu())

    out = {
        "generated_at": pacific_now_iso(),
        "source_result": str(args.result_json) if args.result_json else None,
        "policy": args.policy,
        "num_envs": env.num_envs,
        "horizon": args.horizon,
        "device": args.device,
        "env": env.env_name,
        "contact_backend": env.contact_backend,
        "world_spacing": list(env.world_spacing) if env.world_spacing is not None else None,
        "mjw_nworld": int(env.solver.mjw_data.nworld) if getattr(env.solver, "mjw_data", None) is not None else None,
        "ant_asset": env.ant_asset,
        "ant_dof_limit_mode": env.ant_dof_limit_mode,
        "ant_reward_style": env.ant_reward_style,
        "ant_action_order": env.ant_action_order,
        "sim_substeps": env.sim_substeps,
        "dt": env.dt,
        "force_scale": env.force_scale,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.horizon * env.num_envs / max(elapsed, 1.0e-9),
        "cuda_peak_allocated_mb": peak_allocated / (1024.0 * 1024.0),
        "cuda_peak_reserved_mb": peak_reserved / (1024.0 * 1024.0),
        "first_threshold_step": first_threshold_step,
        "final_max_q_delta_error": float(final_q_error_by_env.max().detach().cpu()),
        "final_max_qd_error": float(final_qd_error_by_env.max().detach().cpu()),
        "final_worst_env": worst_env,
        "final_fallen_count": int(env.fallen_state(q).sum().detach().cpu()),
        "final_invalid_count": int(env.invalid_state(q, qd).sum().detach().cpu()),
        "samples": samples,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "clone_consistency.json", out)
    print(json.dumps({k: out[k] for k in ("contact_backend", "policy", "final_max_q_delta_error", "final_max_qd_error", "first_threshold_step")}, indent=2))


if __name__ == "__main__":
    main()
