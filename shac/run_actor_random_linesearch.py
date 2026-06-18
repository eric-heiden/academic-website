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
    evaluate_policy_chunked,
    evaluate_policy_uninterrupted,
    flatten_parameters,
    load_actor_checkpoint,
    make_actor,
    pacific_now_iso,
    rollout_constraint_shortfalls,
    rollout_selection_score,
    summarize_rollout_repeats,
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


def named_trainable_parameters(actor: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in actor.named_parameters() if param.requires_grad]


def _last_indexed_prefix(named_params: list[tuple[str, torch.nn.Parameter]], root: str) -> str | None:
    best_idx = None
    best_prefix = None
    prefix = f"{root}."
    for name, param in named_params:
        if not name.startswith(prefix) or not name.endswith(".weight") or param.ndim != 2:
            continue
        parts = name.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        if best_idx is None or idx > best_idx:
            best_idx = idx
            best_prefix = f"{root}.{idx}."
    return best_prefix


def flat_parameter_mask(named_params: list[tuple[str, torch.nn.Parameter]], mode: str) -> torch.Tensor:
    if mode == "all":
        return torch.cat([torch.ones_like(param.detach()).reshape(-1) for _, param in named_params])
    if mode != "head":
        raise ValueError(f"unknown parameter mask: {mode}")

    head_prefixes = ["mean."]
    for root in ("actor", "mu_net", "backbone"):
        prefix = _last_indexed_prefix(named_params, root)
        if prefix is not None:
            head_prefixes.append(prefix)
    chunks = []
    active = 0
    for name, param in named_params:
        is_head = (
            name.startswith(tuple(head_prefixes))
            or name in {"log_std", "logstd"}
            or name.endswith(".log_std")
            or name.endswith(".logstd")
        )
        value = torch.ones_like(param.detach()) if is_head else torch.zeros_like(param.detach())
        active += int(is_head) * param.numel()
        chunks.append(value.reshape(-1))
    if active == 0:
        raise RuntimeError("could not identify any head parameters to perturb")
    return torch.cat(chunks)


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
    parser = argparse.ArgumentParser(description="Random local policy perturbation line-search under guarded evals.")
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--actor-path", type=Path, required=True)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--scales", type=parse_float_list, default=[1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5])
    parser.add_argument("--mask", choices=["head", "all"], default="head")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--eval-env-counts", type=parse_int_list, default=None)
    parser.add_argument("--eval-total-envs", type=int, default=None)
    parser.add_argument("--eval-chunk-size", type=int, default=None)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--eval-stochastic-init", action="store_true")
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
    eval_result = copy.deepcopy(base_result)
    eval_overrides = {}
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

    env = build_env(eval_result, argparse.Namespace(video_num_envs=1, device=args.device))
    actor = actor_from_result(eval_result, env, args.actor_path)
    obs_rms = load_obs_rms(args.obs_rms_path, env.torch_device, env.num_obs) if args.obs_rms_path else None
    named_params = named_trainable_parameters(actor)
    params = [param for _, param in named_params]
    base_params = flatten_parameters(params)
    mask = flat_parameter_mask(named_params, args.mask).to(base_params.device)
    active_param_count = int(mask.detach().sum().cpu())
    active_param_norm = float((base_params * mask).to(torch.float64).norm().detach().cpu())
    total_param_norm = float(base_params.to(torch.float64).norm().detach().cpu())

    fall_penalty = float(
        args.selection_fall_penalty
        if args.selection_fall_penalty is not None
        else base_result.get("selection_fall_penalty", 500000.0)
    )
    invalid_penalty = float(
        args.selection_invalid_penalty
        if args.selection_invalid_penalty is not None
        else base_result.get("selection_invalid_penalty", 500000.0)
    )
    displacement_weight = float(
        args.selection_displacement_weight
        if args.selection_displacement_weight is not None
        else base_result.get("selection_displacement_weight", 0.0)
    )
    posture_penalty = float(
        args.selection_posture_penalty
        if args.selection_posture_penalty is not None
        else base_result.get("selection_posture_penalty", 0.0)
    )
    min_height = args.selection_min_height if args.selection_min_height is not None else base_result.get("selection_min_height")
    min_up = args.selection_min_up if args.selection_min_up is not None else base_result.get("selection_min_up")
    min_heading = (
        args.selection_min_heading if args.selection_min_heading is not None else base_result.get("selection_min_heading")
    )
    max_abs_joint = (
        args.selection_max_abs_joint
        if args.selection_max_abs_joint is not None
        else base_result.get("selection_max_abs_joint")
    )

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

    def evaluate_current(direction_index: int | None, scale: float, sign: int) -> dict:
        rollouts = []
        scores = []
        eval_entries = []
        for eval_count in eval_env_counts:
            current_rollouts = [evaluate_candidate(eval_result, args, actor, obs_rms, eval_num_envs=eval_count)]
            current_scores = [score_rollout(current_rollouts[0])]
            for _ in range(max(0, int(args.eval_repeats) - 1)):
                current_rollouts.append(evaluate_candidate(eval_result, args, actor, obs_rms, eval_num_envs=eval_count))
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
        return {
            "direction_index": direction_index,
            "scale": float(scale),
            "sign": int(sign),
            "delta_norm": abs(float(scale)),
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
            "rollout_repeats": summarize_rollout_repeats(rollouts, scores) if len(rollouts) > 1 else None,
        }

    eval_env_counts = args.eval_env_counts or [args.eval_num_envs]
    candidates = []
    directions: list[torch.Tensor] = []
    t0 = time.perf_counter()

    assign_flat_parameters(params, base_params)
    baseline = evaluate_current(None, 0.0, 0)
    candidates.append(baseline)
    print(
        f"baseline score={baseline['score']:.3f} dx={baseline['rollout'].get('mean_forward_displacement')} "
        f"term={baseline['rollout'].get('terminal_count')} invalid={baseline['rollout'].get('invalid_count')}",
        flush=True,
    )

    signs = (-1, 1) if args.bidirectional else (1,)
    for direction_index in range(args.num_directions):
        direction = torch.randn_like(base_params) * mask
        direction_norm = direction.to(torch.float64).norm()
        if float(direction_norm.detach().cpu()) == 0.0:
            raise RuntimeError("sampled zero perturbation direction")
        direction = direction / direction_norm.to(direction.dtype)
        directions.append(direction.detach().clone())
        for scale in args.scales:
            for sign in signs:
                assign_flat_parameters(params, base_params + float(sign) * float(scale) * direction)
                candidate = evaluate_current(direction_index, float(scale), int(sign))
                candidates.append(candidate)
                print(
                    f"dir={direction_index} sign={sign:+d} scale={scale:g} score={candidate['score']:.3f} "
                    f"dx={candidate['rollout'].get('mean_forward_displacement')} "
                    f"term={candidate['rollout'].get('terminal_count')} invalid={candidate['rollout'].get('invalid_count')}",
                    flush=True,
                )

    best = max(candidates, key=lambda item: item["score"]) if candidates else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_best_actor and best is not None:
        if best["direction_index"] is None:
            assign_flat_parameters(params, base_params)
        else:
            direction = directions[int(best["direction_index"])]
            assign_flat_parameters(params, base_params + float(best["sign"]) * float(best["scale"]) * direction)
        prefix = args.save_prefix or str(base_result.get("env") or "policy")
        torch.save(actor.state_dict(), args.out.parent / f"{prefix}_best_actor.pt")
        if args.obs_rms_path is not None:
            obs_state = torch.load(args.obs_rms_path, map_location=env.torch_device)
            torch.save(obs_state, args.out.parent / f"{prefix}_best_obs_rms.pt")
    assign_flat_parameters(params, base_params)

    output = {
        "mode": "actor_random_linesearch",
        "timestamp_pacific": pacific_now_iso(),
        "source_result": str(args.result_json),
        "actor_path": str(args.actor_path),
        "obs_rms_path": str(args.obs_rms_path) if args.obs_rms_path else None,
        "env": base_result.get("env"),
        "eval_contact_backend": eval_result.get("contact_backend"),
        "eval_overrides": eval_overrides,
        "mask": args.mask,
        "active_param_count": active_param_count,
        "active_param_norm": active_param_norm,
        "total_param_norm": total_param_norm,
        "num_directions": args.num_directions,
        "scales": [float(item) for item in args.scales],
        "bidirectional": bool(args.bidirectional),
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
        "best_direction_index": best["direction_index"] if best else None,
        "best_sign": best["sign"] if best else None,
        "best_scale": best["scale"] if best else None,
        "best_score": best["score"] if best else None,
        "baseline_score": baseline["score"],
        "candidates": candidates,
        "total_seconds": time.perf_counter() - t0,
    }
    write_json(args.out, output)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
