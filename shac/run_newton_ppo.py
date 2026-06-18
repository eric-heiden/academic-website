from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import warp as wp
import mujoco_warp
import newton

from run_newton_shac import (
    ANT_DEFAULT_SELECTION_FALL_PENALTY,
    ANT_DEFAULT_SELECTION_INVALID_PENALTY,
    ANT_DEFAULT_TERMINATION_PENALTY,
    ANT_INVALID_PENALTY,
    ANT_ISAACLAB_START_JOINT_Q,
    ANT_ISAACLAB_TERMINATION_HEIGHT,
    ANT_START_HEIGHT,
    HOPPER_TERMINATION_ANGLE,
    HOPPER_TERMINATION_HEIGHT,
    HOPPER_TERMINATION_HEIGHT_TOLERANCE,
    AntRewardWeights,
    CheetahRewardWeights,
    HopperRewardWeights,
    NewtonMuJoCoTorchEnv,
    clone_module_state,
    clone_obs_rms_state,
    clone_optimizer_state,
    evaluate_policy,
    evaluate_policy_chunked,
    evaluate_policy_uninterrupted,
    finalize_terminal_reward,
    git_commit_for_imported_module,
    is_locomotion_env,
    is_planar_locomotion_env,
    normalize_obs,
    obs_rms_snapshot,
    pacific_now_iso,
    parse_float_list,
    render_rollout,
    resolve_ant_defaults,
    rollout_constraint_shortfalls,
    rollout_selection_score,
    summarize_rollout_repeats,
    write_json,
)
from utils.running_mean_std import RunningMeanStd


def parse_int_list(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integer list")
    return [int(item) for item in values]


def make_mlp(input_dim: int, hidden_dims: list[int], *, layer_norm: bool) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ELU())
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        *,
        layer_norm: bool,
        initial_log_std: float,
        action_squash: str = "tanh",
    ):
        super().__init__()
        if action_squash not in {"tanh", "none"}:
            raise ValueError(f"unsupported action_squash: {action_squash}")
        self.action_squash = action_squash
        self.backbone, last_dim = make_mlp(obs_dim, hidden_dims, layer_norm=layer_norm)
        self.mean = nn.Linear(last_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def raw_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.backbone(obs))

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean = self.raw_mean(obs)
        if self.action_squash == "tanh":
            return torch.tanh(mean)
        return torch.clamp(mean, -1.0, 1.0)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.raw_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        if self.action_squash == "tanh":
            action = torch.tanh(raw_action)
            log_prob = tanh_normal_log_prob(dist, raw_action, action)
        else:
            action = torch.clamp(raw_action, -1.0, 1.0)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return raw_action, action, log_prob

    def evaluate_raw(self, obs: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        if self.action_squash == "tanh":
            action = torch.tanh(raw_action)
            log_prob = tanh_normal_log_prob(dist, raw_action, action)
        else:
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class PPOValue(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int], *, layer_norm: bool):
        super().__init__()
        backbone, last_dim = make_mlp(obs_dim, hidden_dims, layer_norm=layer_norm)
        self.net = nn.Sequential(backbone, nn.Linear(last_dim, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def load_actor_checkpoint(actor: PPOActor, path: Path, device: torch.device | str) -> None:
    state = torch.load(path, map_location=device)
    try:
        actor.load_state_dict(state)
        return
    except RuntimeError:
        pass

    mapped = {}
    source_prefix = None
    for prefix in ("actor.", "mu_net."):
        if any(key.startswith(prefix) for key in state):
            source_prefix = prefix
            break
    if source_prefix is None:
        raise RuntimeError(f"unsupported actor checkpoint format: {path}")

    linear_keys = sorted(
        {
            key.removeprefix(source_prefix).split(".", 1)[0]
            for key in state
            if key.startswith(source_prefix) and key.endswith(".weight")
        },
        key=int,
    )
    if not linear_keys:
        raise RuntimeError(f"actor checkpoint has no linear layers: {path}")
    output_idx = linear_keys[-1]
    for key, value in state.items():
        if key in {"logstd", "log_std"}:
            mapped["log_std"] = value
            continue
        if not key.startswith(source_prefix):
            continue
        suffix = key.removeprefix(source_prefix)
        layer_idx, param_name = suffix.split(".", 1)
        if layer_idx == output_idx:
            mapped[f"mean.{param_name}"] = value
        else:
            mapped[f"backbone.{suffix}"] = value
    missing, unexpected = actor.load_state_dict(mapped, strict=False)
    unexpected = [key for key in unexpected if key != "log_std"]
    missing = [key for key in missing if key != "log_std"]
    if missing or unexpected:
        raise RuntimeError(f"actor checkpoint mapping failed for {path}: missing={missing}, unexpected={unexpected}")


def load_critic_checkpoint(critic: PPOValue, path: Path, device: torch.device | str) -> None:
    state = torch.load(path, map_location=device)
    critic.load_state_dict(state)


def tanh_normal_log_prob(
    dist: torch.distributions.Normal, raw_action: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    correction = torch.log(torch.clamp(1.0 - action.square(), min=1.0e-6)).sum(dim=-1)
    return log_prob - correction


def train_ppo(args: argparse.Namespace) -> dict:
    if args.contact_backend is None:
        args.contact_backend = "mujoco"
    resolve_ant_defaults(args)
    if args.sim_substeps is None:
        args.sim_substeps = 2 if args.env == "ant" else (16 if is_locomotion_env(args.env) else 1)
    if args.eval_horizon is None:
        args.eval_horizon = 480
    if args.selection_horizon is None and is_locomotion_env(args.env):
        args.selection_horizon = args.eval_horizon
    if args.episode_length is None:
        args.episode_length = 1000
    if args.force_scale is None:
        args.force_scale = 7.5 if args.env == "ant" else 200.0
    if args.termination_penalty is None:
        args.termination_penalty = ANT_DEFAULT_TERMINATION_PENALTY if args.env in {"ant", "hopper"} else 0.0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wp.init()

    env = NewtonMuJoCoTorchEnv(
        env_name=args.env,
        num_envs=args.num_envs,
        device=args.device,
        dt=args.dt,
        sim_substeps=args.sim_substeps,
        mujoco_integrator=args.mujoco_integrator,
        force_scale=args.force_scale,
        contact_backend=args.contact_backend,
        ant_asset=args.ant_asset,
        ant_disable_joint_limits=args.ant_disable_joint_limits,
        ant_density_override=args.ant_density_override,
        ant_contact_margin=args.ant_contact_margin,
        ant_contact_gap=args.ant_contact_gap,
        ant_contact_mu=args.ant_contact_mu,
        ant_joint_damping=args.ant_joint_damping,
        ant_armature=args.ant_armature,
        ant_min_up=args.ant_min_up,
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
        ant_reward_min_up=args.ant_reward_min_up,
        ant_reward_min_height=args.ant_reward_min_height,
        hopper_reward_style=args.hopper_reward_style,
        hopper_start_joint_q=args.hopper_start_joint_q,
        hopper_contact_mu=args.hopper_contact_mu,
        hopper_joint_damping=args.hopper_joint_damping,
        hopper_armature=args.hopper_armature,
        hopper_termination_height=args.hopper_termination_height,
        hopper_termination_angle=args.hopper_termination_angle,
        hopper_termination_height_tolerance=args.hopper_termination_height_tolerance,
        hopper_reset_position_scale=args.hopper_reset_position_scale,
        hopper_reset_angle_scale=args.hopper_reset_angle_scale,
        hopper_reset_joint_scale=args.hopper_reset_joint_scale,
        hopper_reset_velocity_scale=args.hopper_reset_velocity_scale,
        phase_observation=args.phase_observation,
        phase_period=args.phase_period,
        hopper_terminate_angle=args.hopper_terminate_angle,
        locomotion_disable_joint_limits=args.locomotion_disable_joint_limits,
        ant_reward=AntRewardWeights(
            progress=args.ant_progress_weight,
            heading=args.ant_heading_weight,
            up=args.ant_up_weight,
            height=args.ant_height_weight,
            action=args.ant_action_penalty,
            alive=args.ant_alive_reward,
            actions_cost=args.ant_actions_cost,
            energy_cost=args.ant_energy_cost,
            dof_limit_cost=args.ant_dof_limit_cost,
            dof_vel_scale=args.ant_dof_vel_scale,
            up_margin=args.ant_up_margin_penalty,
            height_margin=args.ant_height_margin_penalty,
        ),
        hopper_reward=HopperRewardWeights(
            progress=args.hopper_progress_weight,
            height=args.hopper_height_weight,
            angle=args.hopper_angle_weight,
            action=args.hopper_action_penalty,
            alive=args.hopper_alive_reward,
        ),
        cheetah_reward=CheetahRewardWeights(action=args.cheetah_action_penalty),
    )
    if args.selection_env_counts is not None:
        selection_env_counts = list(dict.fromkeys(int(item) for item in args.selection_env_counts))
    else:
        selection_env_counts = [int(args.selection_num_envs or args.num_envs)]
    if not selection_env_counts or any(item <= 0 for item in selection_env_counts):
        raise ValueError("--selection-env-counts must contain positive integers")
    selection_env_cache: dict[int, NewtonMuJoCoTorchEnv] = {int(args.num_envs): env}

    def get_selection_env(num_envs: int) -> NewtonMuJoCoTorchEnv:
        num_envs = int(num_envs)
        if num_envs not in selection_env_cache:
            selection_env_cache[num_envs] = NewtonMuJoCoTorchEnv(
                env_name=args.env,
                num_envs=num_envs,
                device=args.device,
                dt=args.dt,
                sim_substeps=args.sim_substeps,
                mujoco_integrator=args.mujoco_integrator,
                force_scale=args.force_scale,
                contact_backend=args.contact_backend,
                ant_asset=args.ant_asset,
                ant_disable_joint_limits=args.ant_disable_joint_limits,
                ant_density_override=args.ant_density_override,
                ant_contact_margin=args.ant_contact_margin,
                ant_contact_gap=args.ant_contact_gap,
                ant_contact_mu=args.ant_contact_mu,
                ant_joint_damping=args.ant_joint_damping,
                ant_armature=args.ant_armature,
                ant_min_up=args.ant_min_up,
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
                ant_reward_min_up=args.ant_reward_min_up,
                ant_reward_min_height=args.ant_reward_min_height,
                hopper_reward_style=args.hopper_reward_style,
                hopper_start_joint_q=args.hopper_start_joint_q,
                hopper_contact_mu=args.hopper_contact_mu,
                hopper_joint_damping=args.hopper_joint_damping,
                hopper_armature=args.hopper_armature,
                hopper_termination_height=args.hopper_termination_height,
                hopper_termination_angle=args.hopper_termination_angle,
                hopper_termination_height_tolerance=args.hopper_termination_height_tolerance,
                hopper_reset_position_scale=args.hopper_reset_position_scale,
                hopper_reset_angle_scale=args.hopper_reset_angle_scale,
                hopper_reset_joint_scale=args.hopper_reset_joint_scale,
                hopper_reset_velocity_scale=args.hopper_reset_velocity_scale,
                phase_observation=args.phase_observation,
                phase_period=args.phase_period,
                hopper_terminate_angle=args.hopper_terminate_angle,
                locomotion_disable_joint_limits=args.locomotion_disable_joint_limits,
                ant_reward=env.ant_reward,
                hopper_reward=env.hopper_reward,
                cheetah_reward=env.cheetah_reward,
            )
        return selection_env_cache[num_envs]

    selection_env = get_selection_env(selection_env_counts[0])

    def compact_selection_rollout(candidate: dict) -> dict:
        return {
            "num_envs": candidate.get("num_envs"),
            "return": candidate.get("return"),
            "mean_reward": candidate.get("mean_reward"),
            "alive_fraction": candidate.get("alive_fraction"),
            "mean_forward_displacement": candidate.get("mean_forward_displacement"),
            "min_forward_displacement": candidate.get("min_forward_displacement"),
            "max_forward_displacement": candidate.get("max_forward_displacement"),
            "mean_height": candidate.get("mean_height"),
            "min_height": candidate.get("min_height"),
            "mean_up": candidate.get("mean_up"),
            "min_up": candidate.get("min_up"),
            "mean_heading": candidate.get("mean_heading"),
            "min_heading": candidate.get("min_heading"),
            "terminal_count": candidate.get("terminal_count"),
            "fall_count": candidate.get("fall_count"),
            "invalid_count": candidate.get("invalid_count"),
        }

    def make_eval_env(num_envs: int) -> NewtonMuJoCoTorchEnv:
        return NewtonMuJoCoTorchEnv(
            env_name=args.env,
            num_envs=num_envs,
            device=args.device,
            dt=args.dt,
            sim_substeps=args.sim_substeps,
            mujoco_integrator=args.mujoco_integrator,
            force_scale=args.force_scale,
            contact_backend=args.contact_backend,
            ant_asset=args.ant_asset,
            ant_disable_joint_limits=args.ant_disable_joint_limits,
            ant_density_override=args.ant_density_override,
            ant_contact_margin=args.ant_contact_margin,
            ant_contact_gap=args.ant_contact_gap,
            ant_contact_mu=args.ant_contact_mu,
            ant_joint_damping=args.ant_joint_damping,
            ant_armature=args.ant_armature,
            ant_min_up=args.ant_min_up,
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
            ant_reward_min_up=args.ant_reward_min_up,
            ant_reward_min_height=args.ant_reward_min_height,
            hopper_reward_style=args.hopper_reward_style,
            hopper_start_joint_q=args.hopper_start_joint_q,
            hopper_contact_mu=args.hopper_contact_mu,
            hopper_joint_damping=args.hopper_joint_damping,
            hopper_armature=args.hopper_armature,
            hopper_termination_height=args.hopper_termination_height,
            hopper_termination_angle=args.hopper_termination_angle,
            hopper_termination_height_tolerance=args.hopper_termination_height_tolerance,
            hopper_reset_position_scale=args.hopper_reset_position_scale,
            hopper_reset_angle_scale=args.hopper_reset_angle_scale,
            hopper_reset_joint_scale=args.hopper_reset_joint_scale,
            hopper_reset_velocity_scale=args.hopper_reset_velocity_scale,
            phase_observation=args.phase_observation,
            phase_period=args.phase_period,
            hopper_terminate_angle=args.hopper_terminate_angle,
            locomotion_disable_joint_limits=args.locomotion_disable_joint_limits,
            ant_reward=env.ant_reward,
            hopper_reward=env.hopper_reward,
            cheetah_reward=env.cheetah_reward,
        )

    obs_dim = env.num_obs
    actor = PPOActor(
        obs_dim,
        env.num_actions,
        args.actor_hidden_dims,
        layer_norm=args.layer_norm,
        initial_log_std=args.initial_log_std,
        action_squash=args.action_squash,
    ).to(env.torch_device)
    critic = PPOValue(obs_dim, args.critic_hidden_dims, layer_norm=args.layer_norm).to(env.torch_device)
    if args.actor_path is not None:
        load_actor_checkpoint(actor, args.actor_path, env.torch_device)
    if args.actor_logstd_override is not None:
        with torch.no_grad():
            actor.log_std.fill_(args.actor_logstd_override)
    if args.critic_path is not None:
        load_critic_checkpoint(critic, args.critic_path, env.torch_device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1.0e-5,
    )
    obs_rms = RunningMeanStd(shape=(obs_dim,), device=env.torch_device) if args.obs_rms else None
    if obs_rms is not None and args.obs_rms_path is not None:
        obs_data = torch.load(args.obs_rms_path, map_location=env.torch_device)
        obs_rms.mean = obs_data["mean"].to(env.torch_device)
        obs_rms.var = obs_data["var"].to(env.torch_device)
        obs_rms.count = obs_data["count"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    live_history_path = out_dir / f"{args.env}_ppo_history_live.json"
    q, qd = env.reset(noise=0.0, stochastic_init=args.stochastic_init)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    best_score = -float("inf")
    best_state = None
    best_critic_state = None
    best_optimizer_state = None
    best_obs_rms = None
    best_update = 0
    history = []
    initial_selection = None
    initial_selection_score = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(env.torch_device)
        torch.cuda.synchronize(env.torch_device)

    t0 = time.perf_counter()

    def selection_evaluation() -> tuple[dict, float, dict]:
        worst_selection = None
        worst_score = float("inf")
        worst_shortfalls = None
        worst_env_count = None
        robust_entries = []
        repeat_count = max(1, int(args.selection_repeats))
        selection_horizon = args.selection_horizon or args.eval_horizon
        for env_count in selection_env_counts:
            current_selection_env = get_selection_env(env_count)
            for repeat_idx in range(repeat_count):
                if args.selection_uninterrupted:
                    candidate = evaluate_policy_uninterrupted(
                        current_selection_env,
                        actor,
                        selection_horizon,
                        obs_rms=obs_rms,
                        termination_penalty=args.termination_penalty,
                        stochastic_init=args.eval_stochastic_init,
                    )
                else:
                    candidate = evaluate_policy(
                        current_selection_env,
                        actor,
                        selection_horizon,
                        obs_rms=obs_rms,
                        termination_penalty=args.termination_penalty,
                        stochastic_init=args.eval_stochastic_init,
                    )
                candidate_score = rollout_selection_score(
                    candidate,
                    num_envs=current_selection_env.num_envs,
                    fall_penalty=args.selection_fall_penalty,
                    invalid_penalty=args.selection_invalid_penalty,
                    displacement_weight=args.selection_displacement_weight,
                    height_weight=args.selection_height_weight,
                    up_weight=args.selection_up_weight,
                    heading_weight=args.selection_heading_weight,
                    min_height=args.selection_min_height,
                    min_up=args.selection_min_up,
                    min_heading=args.selection_min_heading,
                    max_abs_joint=args.selection_max_abs_joint,
                    posture_penalty=args.selection_posture_penalty,
                )
                candidate_shortfalls = rollout_constraint_shortfalls(
                    candidate,
                    min_height=args.selection_min_height,
                    min_up=args.selection_min_up,
                    min_heading=args.selection_min_heading,
                    max_abs_joint=args.selection_max_abs_joint,
                )
                robust_entries.append(
                    {
                        "env_count": env_count,
                        "repeat": repeat_idx + 1,
                        "score": candidate_score,
                        "shortfalls": candidate_shortfalls,
                        "rollout": compact_selection_rollout(candidate),
                    }
                )
                if candidate_score < worst_score or worst_selection is None:
                    worst_selection = candidate
                    worst_score = candidate_score
                    worst_shortfalls = candidate_shortfalls
                    worst_env_count = env_count
        assert worst_selection is not None
        assert worst_shortfalls is not None
        worst_selection = copy.deepcopy(worst_selection)
        worst_selection["selection_env_count"] = worst_env_count
        if len(selection_env_counts) > 1 or repeat_count > 1:
            worst_selection["robust_selection"] = {
                "mode": "worst_score",
                "env_counts": selection_env_counts,
                "repeats": repeat_count,
                "worst_env_count": worst_env_count,
                "worst_score": worst_score,
                "entries": robust_entries,
            }
        return worst_selection, worst_score, worst_shortfalls

    if args.actor_path is not None or args.updates == 0:
        initial_selection, initial_selection_score, _ = selection_evaluation()
        best_score = initial_selection_score
        best_state = clone_module_state(actor)
        best_critic_state = clone_module_state(critic)
        best_optimizer_state = clone_optimizer_state(optimizer)
        torch.save(best_state, out_dir / f"{args.env}_ppo_best_actor.pt")
        torch.save(critic.state_dict(), out_dir / f"{args.env}_ppo_best_critic.pt")
        if obs_rms is not None:
            best_obs_rms = clone_obs_rms_state(obs_rms)
            torch.save(best_obs_rms, out_dir / f"{args.env}_ppo_best_obs_rms.pt")
        print(
            f"{args.env} ppo initial: sel={initial_selection_score: .1f} "
            f"ret={initial_selection['return']: .1f} "
            f"dx={initial_selection['mean_forward_displacement']: .2f} "
            f"falls={initial_selection['fall_count']} "
            f"invalid={initial_selection['invalid_count']}",
            flush=True,
        )

    for update in range(args.updates):
        update_t0 = time.perf_counter()
        obs_buf = []
        raw_action_buf = []
        log_prob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        mean_reward_steps = []
        invalid_count = 0
        fall_count = 0
        timeout_count = 0

        for _ in range(args.rollout_steps):
            with torch.no_grad():
                obs_raw = env.observe(q, qd, prev_action, phase=progress)
                if obs_rms is not None and not args.freeze_obs_rms:
                    obs_rms.update(obs_raw)
                obs = normalize_obs(obs_raw, obs_rms_snapshot(obs_rms))
                raw_action, action, log_prob = actor.sample(obs)
                value = critic(obs)
                q_next, qd_next = env.step(q, qd, env.action_to_joint_f(action))
                invalid = env.invalid_state(q_next, qd_next)
                fell = torch.logical_and(env.fallen_state(q_next), ~invalid)
                q_next, qd_next, action = env.sanitize_state(
                    q_next, qd_next, action, invalid, stochastic_init=args.stochastic_init
                )
                next_obs = env.observe(q_next, qd_next, action, phase=progress + 1)
                reward = env.transition_reward(q, qd, q_next, qd_next, action, obs=next_obs)
                reward = finalize_terminal_reward(
                    reward,
                    invalid=invalid,
                    fell=fell,
                    termination_penalty=args.termination_penalty,
                )
                progress = progress + 1
                timeout = progress >= args.episode_length
                done = torch.logical_or(torch.logical_or(timeout, fell), invalid)

                invalid_count += int(invalid.sum().cpu())
                fall_count += int(fell.sum().cpu())
                timeout_count += int(timeout.sum().cpu())
                mean_reward_steps.append(reward.mean())

                obs_buf.append(obs)
                raw_action_buf.append(raw_action)
                log_prob_buf.append(log_prob)
                value_buf.append(value)
                reward_buf.append(reward)
                done_buf.append(done.to(torch.float32))

                if done.any():
                    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    q_next, qd_next = env.reset_done(q_next, qd_next, done_ids, stochastic_init=args.stochastic_init)
                    action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                    progress = torch.where(done, torch.zeros_like(progress), progress)
                q, qd, prev_action = q_next.detach(), qd_next.detach(), action.detach()

        with torch.no_grad():
            last_obs_raw = env.observe(q, qd, prev_action, phase=progress)
            last_obs = normalize_obs(last_obs_raw, obs_rms_snapshot(obs_rms))
            last_value = critic(last_obs)
            rewards = torch.stack(reward_buf)
            dones = torch.stack(done_buf)
            values = torch.stack(value_buf)
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(env.num_envs, dtype=torch.float32, device=env.torch_device)
            for step in reversed(range(args.rollout_steps)):
                next_nonterminal = 1.0 - dones[step]
                next_value = last_value if step == args.rollout_steps - 1 else values[step + 1]
                delta = rewards[step] + args.gamma * next_value * next_nonterminal - values[step]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[step] = last_gae
            returns = advantages + values

        obs_flat = torch.stack(obs_buf).reshape(-1, obs_dim)
        raw_action_flat = torch.stack(raw_action_buf).reshape(-1, env.num_actions)
        old_log_prob_flat = torch.stack(log_prob_buf).reshape(-1)
        advantage_flat = advantages.reshape(-1)
        return_flat = returns.reshape(-1)
        value_flat = values.reshape(-1)
        advantage_flat = (advantage_flat - advantage_flat.mean()) / (advantage_flat.std().clamp(min=1.0e-6))

        sample_count = obs_flat.shape[0]
        batch_size = min(args.minibatch_size, sample_count)
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        approx_kl_value = 0.0
        kl_values = []
        for _ in range(args.ppo_epochs):
            order = torch.randperm(sample_count, device=env.torch_device)
            for start in range(0, sample_count, batch_size):
                idx = order[start : start + batch_size]
                new_log_prob, entropy = actor.evaluate_raw(obs_flat[idx], raw_action_flat[idx])
                new_value = critic(obs_flat[idx])
                log_ratio = new_log_prob - old_log_prob_flat[idx]
                ratio = log_ratio.exp()
                unclipped = ratio * advantage_flat[idx]
                clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * advantage_flat[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_pred_clipped = value_flat[idx] + (new_value - value_flat[idx]).clamp(
                    -args.value_clip_coef, args.value_clip_coef
                )
                value_loss = 0.5 * torch.max(
                    (new_value - return_flat[idx]).square(),
                    (value_pred_clipped - return_flat[idx]).square(),
                ).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
                optimizer.step()
                policy_loss_value = float(policy_loss.detach().cpu())
                value_loss_value = float(value_loss.detach().cpu())
                entropy_value = float(entropy_loss.detach().cpu())
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                approx_kl_value = float(approx_kl.detach().cpu())
                kl_values.append(approx_kl_value)

        if kl_values:
            approx_kl_value = float(sum(kl_values) / len(kl_values))
        if args.adaptive_kl and approx_kl_value > 0.0:
            current_lr = optimizer.param_groups[0]["lr"]
            new_lr = current_lr
            if approx_kl_value > 2.0 * args.desired_kl:
                new_lr = max(args.min_lr, current_lr / 1.5)
            elif approx_kl_value < 0.5 * args.desired_kl:
                new_lr = min(args.max_lr, current_lr * 1.5)
            if new_lr != current_lr:
                for group in optimizer.param_groups:
                    group["lr"] = new_lr

        if torch.cuda.is_available():
            torch.cuda.synchronize(env.torch_device)

        should_eval = (
            update == 0
            or update == args.updates - 1
            or ((update + 1) % max(1, args.eval_interval) == 0)
        )
        selection = None
        selection_score = None
        selection_shortfalls = {}
        selection_update_rolled_back = False
        guard_reference_score = best_score
        guard_threshold = best_score - args.selection_guard_max_score_drop
        if should_eval:
            selection, selection_score, selection_shortfalls = selection_evaluation()
            if selection_score > best_score:
                best_score = selection_score
                best_update = update + 1
                best_state = clone_module_state(actor)
                best_critic_state = clone_module_state(critic)
                best_optimizer_state = clone_optimizer_state(optimizer)
                torch.save(best_state, out_dir / f"{args.env}_ppo_best_actor.pt")
                torch.save(critic.state_dict(), out_dir / f"{args.env}_ppo_best_critic.pt")
                if obs_rms is not None:
                    best_obs_rms = clone_obs_rms_state(obs_rms)
                    torch.save(best_obs_rms, out_dir / f"{args.env}_ppo_best_obs_rms.pt")
            elif (
                args.selection_guard_updates
                and best_state is not None
                and selection_score < guard_threshold
            ):
                if args.save_rejected_updates:
                    rejected_prefix = out_dir / f"{args.env}_ppo_rejected_update_{update + 1:03d}"
                    torch.save(actor.state_dict(), rejected_prefix.with_name(f"{rejected_prefix.name}_actor.pt"))
                    torch.save(critic.state_dict(), rejected_prefix.with_name(f"{rejected_prefix.name}_critic.pt"))
                    if obs_rms is not None:
                        torch.save(
                            {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
                            rejected_prefix.with_name(f"{rejected_prefix.name}_obs_rms.pt"),
                        )
                actor.load_state_dict(best_state)
                if best_critic_state is not None:
                    critic.load_state_dict(best_critic_state)
                if best_optimizer_state is not None:
                    optimizer.load_state_dict(best_optimizer_state)
                if obs_rms is not None and best_obs_rms is not None:
                    obs_rms.mean = best_obs_rms["mean"].detach().clone()
                    obs_rms.var = best_obs_rms["var"].detach().clone()
                    obs_rms.count = best_obs_rms["count"]
                q, qd = env.reset(noise=0.0, stochastic_init=args.stochastic_init)
                prev_action = torch.zeros(
                    (env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device
                )
                progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
                selection_update_rolled_back = True

        update_s = time.perf_counter() - update_t0
        mean_reward = float(torch.stack(mean_reward_steps).mean().cpu())
        row = {
            "update": update + 1,
            "mean_reward": mean_reward,
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "approx_kl": approx_kl_value,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_return": selection["return"] if selection is not None else None,
            "selection_score": selection_score,
            "selection_mean_reward": selection["mean_reward"] if selection is not None else None,
            "selection_forward_displacement": (
                selection["mean_forward_displacement"] if selection is not None else None
            ),
            "selection_env_count": selection.get("selection_env_count") if selection is not None else None,
            "selection_alive_fraction": selection["alive_fraction"] if selection is not None else None,
            "selection_mean_height": selection["mean_height"] if selection is not None else None,
            "selection_min_height": selection.get("min_height") if selection is not None else None,
            "selection_mean_up": selection["mean_up"] if selection is not None else None,
            "selection_min_up": selection.get("min_up") if selection is not None else None,
            "selection_mean_heading": selection["mean_heading"] if selection is not None else None,
            "selection_min_heading": selection.get("min_heading") if selection is not None else None,
            "selection_height_shortfall": selection_shortfalls.get("height_shortfall"),
            "selection_up_shortfall": selection_shortfalls.get("up_shortfall"),
            "selection_heading_shortfall": selection_shortfalls.get("heading_shortfall"),
            "selection_joint_shortfall": selection_shortfalls.get("joint_shortfall"),
            "selection_posture_shortfall": selection_shortfalls.get("posture_shortfall"),
            "selection_terminal_count": selection.get("terminal_count") if selection is not None else None,
            "selection_fall_count": selection["fall_count"] if selection is not None else None,
            "selection_invalid_count": selection["invalid_count"] if selection is not None else None,
            "selection_guard_active": args.selection_guard_updates,
            "selection_guard_reference_score": guard_reference_score if args.selection_guard_updates else None,
            "selection_guard_acceptance_threshold": guard_threshold if args.selection_guard_updates else None,
            "selection_update_rolled_back": selection_update_rolled_back,
            "invalid_resets": invalid_count,
            "fall_resets": fall_count,
            "timeout_resets": timeout_count,
            "update_seconds": update_s,
            "fps": args.num_envs * args.rollout_steps / update_s,
        }
        history.append(row)
        write_json(
            live_history_path,
            {
                "env": args.env,
                "algo": "ppo",
                "timestamp_pacific": pacific_now_iso(),
                "history": history,
                "selection_env_counts": selection_env_counts,
                "best_update": best_update,
                "best_eval_score": best_score,
            },
        )
        sel_text = f"{selection_score: .1f}" if selection_score is not None else " skipped"
        print(
            f"{args.env} ppo update {update + 1:03d}: reward={mean_reward: .4f} "
            f"sel={sel_text} fps={row['fps']: .1f}",
            flush=True,
        )

    if best_state is not None:
        actor.load_state_dict(best_state)
    if best_critic_state is not None:
        critic.load_state_dict(best_critic_state)
    if obs_rms is not None and best_obs_rms is not None:
        obs_rms.mean = best_obs_rms["mean"]
        obs_rms.var = best_obs_rms["var"]
        obs_rms.count = best_obs_rms["count"]
    torch.save(actor.state_dict(), out_dir / f"{args.env}_ppo_actor.pt")
    torch.save(critic.state_dict(), out_dir / f"{args.env}_ppo_critic.pt")
    if obs_rms is not None:
        torch.save(
            {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
            out_dir / f"{args.env}_ppo_obs_rms.pt",
        )

    final_eval_num_envs = args.final_eval_num_envs or selection_env.num_envs
    eval_chunk_size = args.eval_chunk_size or final_eval_num_envs
    use_chunked_final_eval = eval_chunk_size < final_eval_num_envs

    def final_eval_once(*, uninterrupted: bool) -> dict:
        if use_chunked_final_eval:
            return evaluate_policy_chunked(
                make_eval_env,
                actor,
                args.eval_horizon,
                total_envs=final_eval_num_envs,
                chunk_size=eval_chunk_size,
                obs_rms=obs_rms,
                termination_penalty=args.termination_penalty,
                stochastic_init=args.eval_stochastic_init,
                uninterrupted=uninterrupted,
            )
        if final_eval_num_envs != selection_env.num_envs:
            eval_env = make_eval_env(final_eval_num_envs)
        else:
            eval_env = selection_env
        evaluator = evaluate_policy_uninterrupted if uninterrupted else evaluate_policy
        return evaluator(
            eval_env,
            actor,
            args.eval_horizon,
            obs_rms=obs_rms,
            termination_penalty=args.termination_penalty,
            stochastic_init=args.eval_stochastic_init,
        )

    rollout = final_eval_once(uninterrupted=False)
    rollout_uninterrupted = final_eval_once(uninterrupted=True)
    def score_rollout(candidate: dict) -> float:
        return rollout_selection_score(
            candidate,
            num_envs=int(candidate.get("num_envs") or selection_env.num_envs),
            fall_penalty=args.selection_fall_penalty,
            invalid_penalty=args.selection_invalid_penalty,
            displacement_weight=args.selection_displacement_weight,
            height_weight=args.selection_height_weight,
            up_weight=args.selection_up_weight,
            heading_weight=args.selection_heading_weight,
            min_height=args.selection_min_height,
            min_up=args.selection_min_up,
            min_heading=args.selection_min_heading,
            max_abs_joint=args.selection_max_abs_joint,
            posture_penalty=args.selection_posture_penalty,
        )

    eval_score = score_rollout(rollout)
    eval_repeats = None
    eval_uninterrupted_repeats = None
    if args.final_eval_repeats > 1:
        repeated_rollouts = [rollout]
        repeated_uninterrupted = [rollout_uninterrupted]
        for _ in range(args.final_eval_repeats - 1):
            repeated_rollouts.append(final_eval_once(uninterrupted=False))
            repeated_uninterrupted.append(final_eval_once(uninterrupted=True))
        eval_repeats = summarize_rollout_repeats(repeated_rollouts, [score_rollout(item) for item in repeated_rollouts])
        eval_uninterrupted_repeats = summarize_rollout_repeats(
            repeated_uninterrupted,
            [score_rollout(item) for item in repeated_uninterrupted],
        )
    video_path = None
    poster_path = None
    if args.render_video:
        render_env = env if args.video_num_envs == args.num_envs else make_eval_env(args.video_num_envs)
        video_horizon = args.video_horizon or args.eval_horizon
        video_path, poster_path = render_rollout(
            render_env,
            actor,
            out_dir,
            video_horizon,
            f"{args.env}_ppo",
            obs_rms=obs_rms,
        )
    total_s = time.perf_counter() - t0

    result = {
        "env": args.env,
        "algo": "ppo",
        "title": "SHAC with MuJoCo Warp",
        "timestamp_pacific": pacific_now_iso(),
        "newton_commit": git_commit_for_imported_module(newton),
        "newton_path": str(Path(newton.__path__[0]).resolve()) if hasattr(newton, "__path__") else None,
        "mujoco_warp_commit": git_commit_for_imported_module(mujoco_warp),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "contact_backend": args.contact_backend,
        "rollout_steps": args.rollout_steps,
        "updates": args.updates,
        "eval_horizon": args.eval_horizon,
        "dt": args.dt,
        "sim_substeps": env.sim_substeps,
        "mujoco_integrator": env.mujoco_integrator,
        "nconmax": env.nconmax,
        "njmax": env.njmax,
        "world_spacing": list(env.world_spacing) if env.world_spacing is not None else None,
        "force_scale": args.force_scale,
        "episode_length": args.episode_length,
        "stochastic_init": args.stochastic_init,
        "eval_stochastic_init": args.eval_stochastic_init,
        "obs_rms": args.obs_rms,
        "freeze_obs_rms": args.freeze_obs_rms,
        "actor_path": str(args.actor_path) if args.actor_path is not None else None,
        "critic_path": str(args.critic_path) if args.critic_path is not None else None,
        "obs_rms_path": str(args.obs_rms_path) if args.obs_rms_path is not None else None,
        "termination_penalty": args.termination_penalty,
        "selection_fall_penalty": args.selection_fall_penalty,
        "selection_invalid_penalty": args.selection_invalid_penalty,
        "selection_displacement_weight": args.selection_displacement_weight,
        "selection_height_weight": args.selection_height_weight,
        "selection_up_weight": args.selection_up_weight,
        "selection_heading_weight": args.selection_heading_weight,
        "selection_min_height": args.selection_min_height,
        "selection_min_up": args.selection_min_up,
        "selection_min_heading": args.selection_min_heading,
        "selection_max_abs_joint": args.selection_max_abs_joint,
        "selection_posture_penalty": args.selection_posture_penalty,
        "selection_repeats": args.selection_repeats,
        "selection_uninterrupted": args.selection_uninterrupted,
        "selection_guard_updates": args.selection_guard_updates,
        "selection_guard_max_score_drop": args.selection_guard_max_score_drop,
        "save_rejected_updates": args.save_rejected_updates,
        "final_eval_repeats": args.final_eval_repeats,
        "final_eval_num_envs": final_eval_num_envs,
        "eval_chunk_size": eval_chunk_size if use_chunked_final_eval else None,
        "lr": args.lr,
        "adaptive_kl": args.adaptive_kl,
        "desired_kl": args.desired_kl,
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "clip_coef": args.clip_coef,
        "value_clip_coef": args.value_clip_coef,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "actor_hidden_dims": args.actor_hidden_dims,
        "critic_hidden_dims": args.critic_hidden_dims,
        "layer_norm": args.layer_norm,
        "initial_log_std": args.initial_log_std,
        "actor_logstd_override": args.actor_logstd_override,
        "action_squash": args.action_squash,
        "hopper_reward": env.hopper_reward.__dict__ if args.env == "hopper" else None,
        "hopper_terminate_angle": env.hopper_terminate_angle if args.env == "hopper" else None,
        "hopper_termination_angle": env.hopper_termination_angle if args.env == "hopper" else None,
        "hopper_termination_height": env.hopper_termination_height if args.env == "hopper" else None,
        "hopper_termination_height_tolerance": env.hopper_termination_height_tolerance if args.env == "hopper" else None,
        "hopper_reward_style": env.hopper_reward_style if args.env == "hopper" else None,
        "hopper_start_joint_q": env.hopper_start_joint_q if args.env == "hopper" else None,
        "hopper_contact_mu": env.hopper_contact_mu if args.env == "hopper" else None,
        "hopper_joint_damping": env.hopper_joint_damping if args.env == "hopper" else None,
        "hopper_armature": env.hopper_armature if args.env == "hopper" else None,
        "hopper_reset_position_scale": env.hopper_reset_position_scale if args.env == "hopper" else None,
        "hopper_reset_angle_scale": env.hopper_reset_angle_scale if args.env == "hopper" else None,
        "hopper_reset_joint_scale": env.hopper_reset_joint_scale if args.env == "hopper" else None,
        "hopper_reset_velocity_scale": env.hopper_reset_velocity_scale if args.env == "hopper" else None,
        "cheetah_reward": env.cheetah_reward.__dict__ if args.env == "cheetah" else None,
        "ant_reward": env.ant_reward.__dict__ if args.env == "ant" else None,
        "ant_asset": env.ant_asset if args.env == "ant" else None,
        "ant_disable_joint_limits": env.ant_disable_joint_limits if args.env == "ant" else None,
        "ant_density_override": env.ant_density_override if args.env == "ant" else None,
        "ant_contact_margin": env.ant_contact_margin if args.env == "ant" else None,
        "ant_contact_gap": env.ant_contact_gap if args.env == "ant" else None,
        "ant_contact_mu": env.ant_contact_mu if args.env == "ant" else None,
        "ant_joint_damping": env.ant_joint_damping if args.env == "ant" else None,
        "ant_armature": env.ant_armature if args.env == "ant" else None,
        "ant_min_up": env.ant_min_up if args.env == "ant" else None,
        "ant_start_height": env.ant_start_height if args.env == "ant" else None,
        "ant_start_joint_q": env.ant_start_joint_q if args.env == "ant" else None,
        "ant_reset_position_scale": env.ant_reset_position_scale if args.env == "ant" else None,
        "ant_reset_angle_scale": env.ant_reset_angle_scale if args.env == "ant" else None,
        "ant_reset_joint_scale": env.ant_reset_joint_scale if args.env == "ant" else None,
        "ant_reset_velocity_scale": env.ant_reset_velocity_scale if args.env == "ant" else None,
        "ant_termination_height": env.ant_termination_height if args.env == "ant" else None,
        "ant_max_healthy_height": env.ant_max_healthy_height if args.env == "ant" else None,
        "ant_observation_style": env.ant_observation_style if args.env == "ant" else None,
        "ant_reward_style": env.ant_reward_style if args.env == "ant" else None,
        "ant_dof_limit_mode": env.ant_dof_limit_mode if args.env == "ant" else None,
        "ant_action_order": env.ant_action_order if args.env == "ant" else None,
        "ant_reward_min_up": env.ant_reward_min_up if args.env == "ant" else None,
        "ant_reward_min_height": env.ant_reward_min_height if args.env == "ant" else None,
        "phase_observation": env.phase_observation if args.env == "ant" else None,
        "phase_period": env.phase_period if args.env == "ant" else None,
        "locomotion_disable_joint_limits": env.locomotion_disable_joint_limits if is_planar_locomotion_env(args.env) else None,
        "selection_num_envs": selection_env.num_envs,
        "selection_env_counts": selection_env_counts,
        "total_seconds": total_s,
        "mean_update_seconds": float(np.mean([h["update_seconds"] for h in history])) if history else None,
        "mean_fps": float(np.mean([h["fps"] for h in history])) if history else None,
        "initial_selection": initial_selection,
        "initial_selection_score": initial_selection_score,
        "best_update": best_update,
        "best_eval_score": best_score,
        "eval_score": eval_score,
        "max_cuda_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(env.torch_device) / (1024**2)) if torch.cuda.is_available() else None
        ),
        "max_cuda_memory_reserved_mb": (
            float(torch.cuda.max_memory_reserved(env.torch_device) / (1024**2)) if torch.cuda.is_available() else None
        ),
        "history": history,
        "eval": rollout,
        "eval_uninterrupted": rollout_uninterrupted,
        "eval_repeats": eval_repeats,
        "eval_uninterrupted_repeats": eval_uninterrupted_repeats,
        "video": str(video_path) if video_path is not None else None,
        "poster": str(poster_path) if poster_path is not None else None,
        "video_horizon": args.video_horizon or args.eval_horizon if video_path is not None else None,
        "gpu": torch.cuda.get_device_name(env.torch_device) if torch.cuda.is_available() else "cpu",
    }
    write_json(out_dir / f"{args.env}_ppo_results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["hopper", "cheetah", "ant"], default="hopper")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "assets"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--actor-hidden-dims", type=parse_int_list, default=[128, 64, 32])
    parser.add_argument("--critic-hidden-dims", type=parse_int_list, default=[128, 64])
    parser.add_argument("--layer-norm", dest="layer_norm", action="store_true", default=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    parser.add_argument("--action-squash", choices=["tanh", "none"], default="tanh")
    parser.add_argument("--initial-log-std", type=float, default=-0.5)
    parser.add_argument("--actor-logstd-override", type=float, default=None)
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--mujoco-integrator", choices=["euler", "rk4", "implicitfast", "implicit"], default="euler")
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--adaptive-kl", action="store_true")
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--max-lr", type=float, default=1.0e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--termination-penalty", type=float, default=None)
    parser.add_argument("--stochastic-init", dest="stochastic_init", action="store_true", default=True)
    parser.add_argument("--deterministic-init", dest="stochastic_init", action="store_false")
    parser.add_argument("--eval-stochastic-init", dest="eval_stochastic_init", action="store_true")
    parser.add_argument("--eval-deterministic-init", dest="eval_stochastic_init", action="store_false")
    parser.set_defaults(eval_stochastic_init=False)
    parser.add_argument("--obs-rms", dest="obs_rms", action="store_true", default=True)
    parser.add_argument("--no-obs-rms", dest="obs_rms", action="store_false")
    parser.add_argument("--freeze-obs-rms", action="store_true")
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--critic-path", type=Path, default=None)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--ant-progress-weight", type=float, default=1.0)
    parser.add_argument("--ant-heading-weight", type=float, default=None)
    parser.add_argument("--ant-up-weight", type=float, default=0.1)
    parser.add_argument("--ant-height-weight", type=float, default=1.0)
    parser.add_argument("--ant-action-penalty", type=float, default=0.0)
    parser.add_argument("--ant-alive-reward", type=float, default=0.5)
    parser.add_argument("--ant-actions-cost", type=float, default=0.005)
    parser.add_argument("--ant-energy-cost", type=float, default=0.05)
    parser.add_argument("--ant-dof-limit-cost", type=float, default=None)
    parser.add_argument("--ant-dof-vel-scale", type=float, default=0.2)
    parser.add_argument("--ant-asset", choices=["diffrl", "nv"], default="diffrl")
    parser.add_argument("--ant-disable-joint-limits", action="store_true")
    parser.add_argument("--ant-density-override", type=float, default=None)
    parser.add_argument("--ant-contact-margin", type=float, default=0.0)
    parser.add_argument("--ant-contact-gap", type=float, default=None)
    parser.add_argument("--ant-contact-mu", type=float, default=None)
    parser.add_argument("--ant-joint-damping", type=float, default=None)
    parser.add_argument("--ant-armature", type=float, default=None)
    parser.add_argument("--ant-min-up", type=float, default=None)
    parser.add_argument("--ant-start-height", type=float, default=None)
    parser.add_argument("--ant-start-joint-q", type=parse_float_list, default=None)
    parser.add_argument("--ant-reset-position-scale", type=float, default=0.1)
    parser.add_argument("--ant-reset-angle-scale", type=float, default=math.pi / 24.0)
    parser.add_argument("--ant-reset-joint-scale", type=float, default=0.2)
    parser.add_argument("--ant-reset-velocity-scale", type=float, default=0.25)
    parser.add_argument("--ant-termination-height", type=float, default=None)
    parser.add_argument("--ant-max-healthy-height", type=float, default=1.5)
    parser.add_argument("--ant-observation-style", choices=["diffrl", "isaac"], default=None)
    parser.add_argument(
        "--ant-reward-style",
        choices=["diffrl", "isaac", "isaaclab", "isaaclab_potential", "isaaclab_potential_height", "isaac_heading_gated"],
        default=None,
    )
    parser.add_argument("--ant-dof-limit-mode", choices=["abs", "upper"], default=None)
    parser.add_argument("--ant-action-order", choices=["joint", "actuator"], default=None)
    parser.add_argument("--ant-reward-min-up", type=float, default=None)
    parser.add_argument("--ant-reward-min-height", type=float, default=None)
    parser.add_argument("--ant-up-margin-penalty", type=float, default=0.0)
    parser.add_argument("--ant-height-margin-penalty", type=float, default=0.0)
    parser.add_argument("--phase-observation", action="store_true")
    parser.add_argument("--phase-period", type=int, default=60)
    parser.add_argument("--hopper-height-weight", type=float, default=1.0)
    parser.add_argument("--hopper-progress-weight", type=float, default=1.0)
    parser.add_argument("--hopper-angle-weight", type=float, default=1.0)
    parser.add_argument("--hopper-action-penalty", type=float, default=-0.1)
    parser.add_argument("--hopper-alive-reward", type=float, default=1.0)
    parser.add_argument("--hopper-reward-style", choices=["diffrl", "gym"], default="diffrl")
    parser.add_argument("--hopper-start-joint-q", type=parse_float_list, default=None)
    parser.add_argument("--hopper-contact-mu", type=float, default=0.9)
    parser.add_argument("--hopper-joint-damping", type=float, default=2.0)
    parser.add_argument("--hopper-armature", type=float, default=1.0)
    parser.add_argument("--hopper-termination-height", type=float, default=HOPPER_TERMINATION_HEIGHT)
    parser.add_argument("--hopper-termination-angle", type=float, default=HOPPER_TERMINATION_ANGLE)
    parser.add_argument("--hopper-termination-height-tolerance", type=float, default=HOPPER_TERMINATION_HEIGHT_TOLERANCE)
    parser.add_argument("--hopper-reset-position-scale", type=float, default=0.05)
    parser.add_argument("--hopper-reset-angle-scale", type=float, default=0.1)
    parser.add_argument("--hopper-reset-joint-scale", type=float, default=0.05)
    parser.add_argument("--hopper-reset-velocity-scale", type=float, default=0.05)
    parser.add_argument("--hopper-terminate-angle", action="store_true")
    parser.add_argument("--cheetah-action-penalty", type=float, default=-0.1)
    parser.add_argument("--locomotion-disable-joint-limits", action="store_true")
    parser.add_argument("--selection-num-envs", type=int, default=None)
    parser.add_argument("--selection-env-counts", type=parse_int_list, default=None)
    parser.add_argument("--selection-horizon", type=int, default=None)
    parser.add_argument("--selection-fall-penalty", type=float, default=ANT_DEFAULT_SELECTION_FALL_PENALTY)
    parser.add_argument("--selection-invalid-penalty", type=float, default=ANT_DEFAULT_SELECTION_INVALID_PENALTY)
    parser.add_argument("--selection-displacement-weight", type=float, default=0.0)
    parser.add_argument("--selection-height-weight", type=float, default=0.0)
    parser.add_argument("--selection-up-weight", type=float, default=0.0)
    parser.add_argument("--selection-heading-weight", type=float, default=0.0)
    parser.add_argument("--selection-min-height", type=float, default=None)
    parser.add_argument("--selection-min-up", type=float, default=None)
    parser.add_argument("--selection-min-heading", type=float, default=None)
    parser.add_argument("--selection-max-abs-joint", type=float, default=None)
    parser.add_argument("--selection-posture-penalty", type=float, default=0.0)
    parser.add_argument("--selection-repeats", type=int, default=1)
    parser.add_argument("--selection-uninterrupted", action="store_true")
    parser.add_argument("--selection-guard-updates", action="store_true")
    parser.add_argument("--selection-guard-max-score-drop", type=float, default=0.0)
    parser.add_argument("--save-rejected-updates", action="store_true")
    parser.add_argument("--final-eval-repeats", type=int, default=1)
    parser.add_argument("--final-eval-num-envs", type=int, default=None)
    parser.add_argument("--eval-chunk-size", type=int, default=None)
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-num-envs", type=int, default=1)
    parser.add_argument("--video-horizon", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train_ppo(parse_args())
