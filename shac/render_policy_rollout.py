from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import torch
import warp as wp

from run_newton_ppo import PPOActor
from run_newton_shac import (
    AcrobotRewardWeights,
    AntRewardWeights,
    CartpoleRewardWeights,
    CheetahRewardWeights,
    ContactTargetRewardWeights,
    HOPPER_TERMINATION_ANGLE,
    HOPPER_TERMINATION_HEIGHT,
    HOPPER_TERMINATION_HEIGHT_TOLERANCE,
    HopperRewardWeights,
    NewtonMuJoCoTorchEnv,
    load_actor_checkpoint,
    render_rollout,
    write_json,
)
from utils.running_mean_std import RunningMeanStd


T = TypeVar("T")


def dataclass_from_json(cls: type[T], value: dict[str, Any] | None) -> T:
    allowed = {field.name for field in fields(cls)}
    if not value:
        return cls()
    return cls(**{key: item for key, item in value.items() if key in allowed})


def json_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def load_run_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def default_actor_path(run_dir: Path, env_name: str, algo: str) -> Path | None:
    if algo == "ppo":
        return first_existing([run_dir / f"{env_name}_ppo_actor.pt", run_dir / f"{env_name}_actor.pt"])
    return first_existing([run_dir / f"{env_name}_actor.pt", run_dir / f"{env_name}_ppo_actor.pt"])


def default_obs_rms_path(run_dir: Path, env_name: str, algo: str) -> Path | None:
    if algo == "ppo":
        return first_existing([run_dir / f"{env_name}_ppo_obs_rms.pt", run_dir / f"{env_name}_obs_rms.pt"])
    return first_existing([run_dir / f"{env_name}_obs_rms.pt", run_dir / f"{env_name}_ppo_obs_rms.pt"])


def detect_algo(result: dict[str, Any], path: Path) -> str:
    if result.get("algo") == "ppo" or path.name.endswith("_ppo_results.json"):
        return "ppo"
    return "shac"


def load_obs_rms(path: Path | None, device: torch.device, obs_dim: int) -> RunningMeanStd | None:
    if path is None:
        return None
    data = torch.load(path, map_location=device)
    obs_rms = RunningMeanStd(shape=(obs_dim,), device=device)
    obs_rms.mean = data["mean"].to(device)
    obs_rms.var = data["var"].to(device)
    obs_rms.count = data["count"]
    return obs_rms


def build_env(result: dict[str, Any], args: argparse.Namespace) -> NewtonMuJoCoTorchEnv:
    env_name = result["env"]
    return NewtonMuJoCoTorchEnv(
        env_name=env_name,
        num_envs=args.video_num_envs,
        device=args.device,
        dt=float(result.get("dt", 1.0 / 60.0)),
        sim_substeps=int(result.get("sim_substeps") or (16 if env_name in {"ant", "hopper", "cheetah"} else 1)),
        mujoco_integrator=result.get("mujoco_integrator") or "euler",
        force_scale=float(result.get("force_scale") or 200.0),
        contact_backend=result.get("contact_backend") or ("mujoco" if env_name in {"ant", "hopper", "cheetah"} else "none"),
        acrobot_actuation=result.get("acrobot_actuation") or "elbow",
        ant_asset=result.get("ant_asset") or "diffrl",
        ant_disable_joint_limits=bool(result.get("ant_disable_joint_limits") or False),
        ant_density_override=result.get("ant_density_override"),
        ant_contact_margin=float(result.get("ant_contact_margin") or 0.0),
        ant_contact_gap=result.get("ant_contact_gap"),
        ant_contact_mu=float(result.get("ant_contact_mu") or 0.75),
        ant_joint_damping=result.get("ant_joint_damping"),
        ant_min_up=result.get("ant_min_up"),
        ant_start_height=result.get("ant_start_height"),
        ant_start_joint_q=result.get("ant_start_joint_q"),
        ant_reset_position_scale=json_float(result.get("ant_reset_position_scale"), 0.1),
        ant_reset_angle_scale=json_float(result.get("ant_reset_angle_scale"), 0.1308996938995747),
        ant_reset_joint_scale=json_float(result.get("ant_reset_joint_scale"), 0.2),
        ant_reset_velocity_scale=json_float(result.get("ant_reset_velocity_scale"), 0.25),
        ant_termination_height=float(result.get("ant_termination_height") or 0.27),
        ant_max_healthy_height=float(result.get("ant_max_healthy_height") or 1.5),
        ant_observation_style=result.get("ant_observation_style") or "diffrl",
        ant_reward_style=result.get("ant_reward_style") or "diffrl",
        ant_action_order=result.get("ant_action_order") or "joint",
        hopper_reward_style=result.get("hopper_reward_style") or "diffrl",
        hopper_start_joint_q=result.get("hopper_start_joint_q"),
        hopper_contact_mu=json_float(result.get("hopper_contact_mu"), 0.9),
        hopper_joint_damping=json_float(result.get("hopper_joint_damping"), 2.0),
        hopper_armature=json_float(result.get("hopper_armature"), 1.0),
        hopper_termination_height=json_float(result.get("hopper_termination_height"), HOPPER_TERMINATION_HEIGHT),
        hopper_termination_angle=json_float(result.get("hopper_termination_angle"), HOPPER_TERMINATION_ANGLE),
        hopper_termination_height_tolerance=json_float(
            result.get("hopper_termination_height_tolerance"), HOPPER_TERMINATION_HEIGHT_TOLERANCE
        ),
        hopper_reset_position_scale=json_float(result.get("hopper_reset_position_scale"), 0.05),
        hopper_reset_angle_scale=json_float(result.get("hopper_reset_angle_scale"), 0.1),
        hopper_reset_joint_scale=json_float(result.get("hopper_reset_joint_scale"), 0.05),
        hopper_reset_velocity_scale=json_float(result.get("hopper_reset_velocity_scale"), 0.05),
        phase_observation=bool(result.get("phase_observation") or False),
        phase_period=int(result.get("phase_period") or 60),
        hopper_terminate_angle=bool(result.get("hopper_terminate_angle") or False),
        locomotion_disable_joint_limits=bool(result.get("locomotion_disable_joint_limits") or False),
        cartpole_reward=dataclass_from_json(CartpoleRewardWeights, result.get("cartpole_reward")),
        ant_reward=dataclass_from_json(AntRewardWeights, result.get("ant_reward")),
        hopper_reward=dataclass_from_json(HopperRewardWeights, result.get("hopper_reward")),
        cheetah_reward=dataclass_from_json(CheetahRewardWeights, result.get("cheetah_reward")),
        acrobot_reward=dataclass_from_json(AcrobotRewardWeights, result.get("acrobot_reward")),
        contact_reward=dataclass_from_json(ContactTargetRewardWeights, result.get("contact_reward")),
    )


def ppo_actor_from_result(result: dict[str, Any], env: NewtonMuJoCoTorchEnv) -> PPOActor:
    hidden_dims = result.get("actor_hidden_dims") or [128, 64, 32]
    layer_norm = result.get("layer_norm")
    initial_log_std = result.get("initial_log_std")
    return PPOActor(
        env.num_obs,
        env.num_actions,
        [int(width) for width in hidden_dims],
        layer_norm=True if layer_norm is None else bool(layer_norm),
        initial_log_std=-0.5 if initial_log_std is None else float(initial_log_std),
        action_squash=result.get("action_squash") or "tanh",
    ).to(env.torch_device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh a report policy video with the shared smoothed follow camera.")
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--algo", choices=["auto", "shac", "ppo"], default="auto")
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--video-num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metadata", action="store_true", help="Write a small render metadata JSON next to the video.")
    args = parser.parse_args()

    wp.init()
    result_path = args.result_json.resolve()
    result = load_run_json(result_path)
    run_dir = result_path.parent
    env_name = result["env"]
    algo = detect_algo(result, result_path) if args.algo == "auto" else args.algo
    actor_path = args.actor_path or default_actor_path(run_dir, env_name, algo)
    obs_rms_path = args.obs_rms_path or default_obs_rms_path(run_dir, env_name, algo)
    if actor_path is None:
        raise FileNotFoundError(f"no saved actor found in {run_dir}")

    env = build_env(result, args)
    if algo == "ppo":
        actor = ppo_actor_from_result(result, env)
        actor.load_state_dict(torch.load(actor_path, map_location=env.torch_device))
    else:
        from run_newton_shac import make_actor

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

    obs_rms = load_obs_rms(obs_rms_path, env.torch_device, env.num_obs) if result.get("obs_rms") else None
    horizon = args.horizon or int(result.get("eval_horizon") or result.get("selection_horizon") or 480)
    render_name = f"{env_name}_ppo" if algo == "ppo" else env_name
    video_path, poster_path = render_rollout(env, actor, run_dir, horizon, render_name, obs_rms=obs_rms)
    print(f"wrote {video_path}")
    print(f"wrote {poster_path}")
    if args.metadata:
        write_json(
            run_dir / f"{env_name}_render_metadata.json",
            {
                "source_result": str(result_path),
                "algo": algo,
                "actor_path": str(actor_path),
                "obs_rms_path": str(obs_rms_path) if obs_rms_path else None,
                "horizon": horizon,
                "video": video_path.name,
                "poster": poster_path.name,
                "camera": "SmoothedFollowCamera",
                "source": "ViewerGL.get_frame()",
                "overlays": False,
            },
        )


if __name__ == "__main__":
    main()
