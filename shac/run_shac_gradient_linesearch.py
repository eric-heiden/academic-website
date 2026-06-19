from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from render_policy_rollout import build_env, load_obs_rms
from run_newton_shac import (
    assign_flat_parameters,
    clone_module_state,
    evaluate_policy_chunked,
    evaluate_policy_uninterrupted,
    flatten_gradients,
    flatten_parameters,
    load_actor_checkpoint,
    make_actor,
    pacific_now_iso,
    rollout_constraint_shortfalls,
    rollout_selection_score,
    shac_rollout_loss,
    summarize_rollout_repeats,
    trainable_parameters,
    write_json,
)


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return [float(item) for item in items]


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return [int(item) for item in items]


def load_result(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def actor_from_result(result: dict, env, actor_path: Path):
    actor = make_actor(
        env,
        stochastic=bool(result.get("stochastic_actor") or False),
        hidden_dims=result.get("actor_hidden_dims"),
        actor_logstd_init=float(result.get("actor_logstd_init") or -1.0),
        actor_layer_norm=bool(result.get("actor_layer_norm", True)),
        action_squash=result.get("action_squash") or "tanh",
    )
    load_actor_checkpoint(actor, actor_path, env.torch_device)
    actor.eval()
    return actor


def evaluate_candidate(
    result: dict,
    args: argparse.Namespace,
    actor,
    obs_rms,
    *,
    eval_num_envs: int | None = None,
) -> dict:
    if args.eval_total_envs is not None and args.eval_chunk_size is not None:
        return evaluate_policy_chunked(
            lambda n: build_env(result, argparse.Namespace(video_num_envs=n, device=args.device)),
            actor,
            args.eval_horizon,
            total_envs=args.eval_total_envs,
            chunk_size=args.eval_chunk_size,
            obs_rms=obs_rms,
            termination_penalty=float(result.get("termination_penalty") or 0.0),
            stochastic_init=args.eval_stochastic_init,
            uninterrupted=True,
        )

    eval_env = build_env(
        result,
        argparse.Namespace(video_num_envs=int(eval_num_envs or args.eval_num_envs), device=args.device),
    )
    return evaluate_policy_uninterrupted(
        eval_env,
        actor,
        args.eval_horizon,
        obs_rms=obs_rms,
        termination_penalty=float(result.get("termination_penalty") or 0.0),
        stochastic_init=args.eval_stochastic_init,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Line-search a SHAC policy-gradient direction with no-reset evals.")
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--actor-path", type=Path, required=True)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--rew-scale", type=float, default=None)
    parser.add_argument("--loss-objective", choices=["reward", "displacement"], default="reward")
    parser.add_argument("--displacement-objective-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=parse_float_list, default=[0.0, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3])
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--grad-num-envs", type=int, default=1)
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--eval-env-counts", type=parse_int_list, default=None)
    parser.add_argument("--eval-total-envs", type=int, default=None)
    parser.add_argument("--eval-chunk-size", type=int, default=None)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--eval-stochastic-init", action="store_true")
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--mujoco-smooth-adjoint", choices=["off", "smooth", "free_body", "surrogate"], default=None)
    parser.add_argument("--ant-dof-limit-mode", choices=["abs", "upper"], default=None)
    parser.add_argument("--eval-contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--eval-mujoco-smooth-adjoint", choices=["off", "smooth", "free_body", "surrogate"], default=None)
    parser.add_argument("--eval-ant-dof-limit-mode", choices=["abs", "upper"], default=None)
    parser.add_argument("--selection-fall-penalty", type=float, default=None)
    parser.add_argument("--selection-invalid-penalty", type=float, default=None)
    parser.add_argument("--selection-displacement-weight", type=float, default=None)
    parser.add_argument("--selection-min-height", type=float, default=None)
    parser.add_argument("--selection-min-up", type=float, default=None)
    parser.add_argument("--selection-min-heading", type=float, default=None)
    parser.add_argument("--selection-max-abs-joint", type=float, default=None)
    parser.add_argument("--selection-posture-penalty", type=float, default=None)
    parser.add_argument("--save-best-actor", action="store_true")
    parser.add_argument("--save-prefix", default=None)
    args = parser.parse_args()

    wp.init()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_result = load_result(args.result_json)
    grad_result = copy.deepcopy(base_result)
    eval_result = copy.deepcopy(base_result)
    gradient_overrides = {}
    eval_overrides = {}
    if args.contact_backend is not None:
        grad_result["contact_backend"] = args.contact_backend
        gradient_overrides["contact_backend"] = args.contact_backend
    if args.mujoco_smooth_adjoint is not None:
        grad_result["mujoco_smooth_adjoint"] = args.mujoco_smooth_adjoint
        gradient_overrides["mujoco_smooth_adjoint"] = args.mujoco_smooth_adjoint
    if args.ant_dof_limit_mode is not None:
        grad_result["ant_dof_limit_mode"] = args.ant_dof_limit_mode
        gradient_overrides["ant_dof_limit_mode"] = args.ant_dof_limit_mode
    if args.eval_contact_backend is not None:
        eval_result["contact_backend"] = args.eval_contact_backend
        eval_overrides["contact_backend"] = args.eval_contact_backend
    if args.eval_mujoco_smooth_adjoint is not None:
        eval_result["mujoco_smooth_adjoint"] = args.eval_mujoco_smooth_adjoint
        eval_overrides["mujoco_smooth_adjoint"] = args.eval_mujoco_smooth_adjoint
    if args.eval_ant_dof_limit_mode is not None:
        eval_result["ant_dof_limit_mode"] = args.eval_ant_dof_limit_mode
        eval_overrides["ant_dof_limit_mode"] = args.eval_ant_dof_limit_mode
    if args.eval_horizon is None:
        args.eval_horizon = int(base_result.get("selection_horizon") or base_result.get("eval_horizon") or 480)

    env = build_env(grad_result, argparse.Namespace(video_num_envs=args.grad_num_envs, device=args.device))
    actor = actor_from_result(grad_result, env, args.actor_path)
    obs_rms = load_obs_rms(args.obs_rms_path, env.torch_device, env.num_obs) if args.obs_rms_path else None
    obs_stats = None if obs_rms is None else (obs_rms.mean.detach().clone(), obs_rms.var.detach().clone())

    q0, qd0 = env.reset(noise=0.0, stochastic_init=False)
    prev0 = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    params = trainable_parameters(actor)
    base_params = flatten_parameters(params)

    actor.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed + 1000)
    loss, metrics = shac_rollout_loss(
        env,
        actor,
        horizon=args.horizon,
        gamma=float(args.gamma if args.gamma is not None else grad_result.get("gamma", 0.99)),
        rew_scale=float(args.rew_scale if args.rew_scale is not None else grad_result.get("rew_scale", 1.0)),
        termination_penalty=float(grad_result.get("termination_penalty") or 0.0),
        obs_stats=obs_stats,
        q0=q0,
        qd0=qd0,
        prev_action0=prev0,
        stochastic_actor=bool(grad_result.get("stochastic_actor") or False),
        loss_objective=args.loss_objective,
        displacement_objective_weight=args.displacement_objective_weight,
    )
    loss.backward()
    grad = flatten_gradients(params)
    grad_norm = float(grad.to(torch.float64).norm().detach().cpu())

    fall_penalty = float(args.selection_fall_penalty if args.selection_fall_penalty is not None else base_result.get("selection_fall_penalty", 500000.0))
    invalid_penalty = float(args.selection_invalid_penalty if args.selection_invalid_penalty is not None else base_result.get("selection_invalid_penalty", 500000.0))
    displacement_weight = float(args.selection_displacement_weight if args.selection_displacement_weight is not None else base_result.get("selection_displacement_weight", 0.0))
    posture_penalty = float(args.selection_posture_penalty if args.selection_posture_penalty is not None else base_result.get("selection_posture_penalty", 0.0))
    min_height = args.selection_min_height if args.selection_min_height is not None else base_result.get("selection_min_height")
    min_up = args.selection_min_up if args.selection_min_up is not None else base_result.get("selection_min_up")
    min_heading = args.selection_min_heading if args.selection_min_heading is not None else base_result.get("selection_min_heading")
    max_abs_joint = args.selection_max_abs_joint if args.selection_max_abs_joint is not None else base_result.get("selection_max_abs_joint")

    def score_rollout(rollout: dict) -> float:
        return rollout_selection_score(
            rollout,
            num_envs=int(rollout.get("num_envs") or args.eval_num_envs),
            fall_penalty=fall_penalty,
            invalid_penalty=invalid_penalty,
            displacement_weight=displacement_weight,
            min_height=min_height,
            min_up=min_up,
            min_heading=min_heading,
            max_abs_joint=max_abs_joint,
            posture_penalty=posture_penalty,
        )

    base_state = clone_module_state(actor)
    candidates = []
    t0 = time.perf_counter()
    eval_env_counts = args.eval_env_counts or [args.eval_num_envs]
    for step in args.steps:
        actor.load_state_dict(base_state)
        assign_flat_parameters(params, base_params - float(step) * grad)
        rollouts = []
        scores = []
        eval_entries = []
        for eval_count in eval_env_counts:
            current_rollouts = [evaluate_candidate(eval_result, args, actor, obs_rms, eval_num_envs=eval_count)]
            current_scores = [score_rollout(current_rollouts[0])]
            for _ in range(max(0, int(args.eval_repeats) - 1)):
                current_rollouts.append(
                    evaluate_candidate(eval_result, args, actor, obs_rms, eval_num_envs=eval_count)
                )
                current_scores.append(score_rollout(current_rollouts[-1]))
            rollouts.extend(current_rollouts)
            scores.extend(current_scores)
            representative = current_rollouts[0]
            eval_entries.append(
                {
                    "num_envs": int(eval_count),
                    "score": min(current_scores),
                    "scores": current_scores,
                    "dx": representative.get("mean_forward_displacement"),
                    "ret": representative.get("return"),
                    "falls": representative.get("fall_count"),
                    "invalid": representative.get("invalid_count"),
                    "terminal": representative.get("terminal_count"),
                    "min_h": representative.get("min_height"),
                    "min_up": representative.get("min_up"),
                    "rollout": representative,
                    "rollout_repeats": summarize_rollout_repeats(current_rollouts, current_scores)
                    if len(current_rollouts) > 1
                    else None,
                }
            )
        worst_idx = int(np.argmin(scores))
        rollout = rollouts[worst_idx]
        score = min(scores)
        repeat_summary = summarize_rollout_repeats(rollouts, scores) if len(rollouts) > 1 else None
        candidates.append(
            {
                "step": float(step),
                "score": score,
                "scores": scores,
                "eval_env_counts": eval_env_counts,
                "evals": eval_entries,
                "shortfalls": rollout_constraint_shortfalls(
                    rollout,
                    min_height=min_height,
                    min_up=min_up,
                    min_heading=min_heading,
                    max_abs_joint=max_abs_joint,
                ),
                "rollout": rollout,
                "rollout_repeats": repeat_summary,
            }
        )
        print(
            f"step={step:g} score={score:.3f} dx={rollout.get('mean_forward_displacement')} "
            f"term={rollout.get('terminal_count')} invalid={rollout.get('invalid_count')}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    best = max(candidates, key=lambda item: item["score"]) if candidates else None
    if args.save_best_actor and best is not None:
        actor.load_state_dict(base_state)
        assign_flat_parameters(params, base_params - float(best["step"]) * grad)
        prefix = args.save_prefix or str(base_result.get("env") or "policy")
        torch.save(actor.state_dict(), args.out.parent / f"{prefix}_best_actor.pt")
        if args.obs_rms_path is not None:
            obs_state = torch.load(args.obs_rms_path, map_location=env.torch_device)
            torch.save(obs_state, args.out.parent / f"{prefix}_best_obs_rms.pt")
    actor.load_state_dict(base_state)
    output = {
        "mode": "shac_gradient_linesearch",
        "timestamp_pacific": pacific_now_iso(),
        "source_result": str(args.result_json),
        "overrides": gradient_overrides,
        "gradient_overrides": gradient_overrides,
        "eval_overrides": eval_overrides,
        "actor_path": str(args.actor_path),
        "obs_rms_path": str(args.obs_rms_path) if args.obs_rms_path else None,
        "env": base_result.get("env"),
        "gradient_contact_backend": grad_result.get("contact_backend"),
        "eval_contact_backend": eval_result.get("contact_backend"),
        "gradient_mujoco_smooth_adjoint": grad_result.get("mujoco_smooth_adjoint"),
        "eval_mujoco_smooth_adjoint": eval_result.get("mujoco_smooth_adjoint"),
        "gradient_ant_dof_limit_mode": grad_result.get("ant_dof_limit_mode"),
        "eval_ant_dof_limit_mode": eval_result.get("ant_dof_limit_mode"),
        "horizon": args.horizon,
        "loss_objective": args.loss_objective,
        "displacement_objective_weight": args.displacement_objective_weight
        if args.loss_objective == "displacement"
        else None,
        "loss": float(loss.detach().cpu()),
        "loss_metrics": metrics,
        "grad_norm": grad_norm,
        "steps": [float(item) for item in args.steps],
        "eval_horizon": args.eval_horizon,
        "eval_num_envs": args.eval_num_envs,
        "eval_env_counts": eval_env_counts,
        "eval_total_envs": args.eval_total_envs,
        "eval_chunk_size": args.eval_chunk_size,
        "eval_repeats": args.eval_repeats,
        "eval_repeat_score_mode": "min",
        "selection": {
            "fall_penalty": fall_penalty,
            "invalid_penalty": invalid_penalty,
            "displacement_weight": displacement_weight,
            "min_height": min_height,
            "min_up": min_up,
            "min_heading": min_heading,
            "max_abs_joint": max_abs_joint,
            "posture_penalty": posture_penalty,
        },
        "best_step": best["step"] if best else None,
        "best_score": best["score"] if best else None,
        "candidates": candidates,
        "total_seconds": time.perf_counter() - t0,
    }
    write_json(args.out, output)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
