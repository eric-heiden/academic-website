from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import warp as wp
import mujoco_warp

from run_newton_shac import (
    ANT_DEFAULT_SELECTION_FALL_PENALTY,
    ANT_DEFAULT_SELECTION_INVALID_PENALTY,
    ANT_DEFAULT_TERMINATION_PENALTY,
    ANT_INVALID_PENALTY,
    AntRewardWeights,
    CheetahRewardWeights,
    HopperRewardWeights,
    NewtonMuJoCoTorchEnv,
    evaluate_policy,
    finalize_terminal_reward,
    git_commit_for_imported_module,
    is_locomotion_env,
    is_planar_locomotion_env,
    normalize_obs,
    obs_rms_snapshot,
    pacific_now_iso,
    render_rollout,
    rollout_selection_score,
    write_json,
)
from utils.running_mean_std import RunningMeanStd


class PPOActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.LayerNorm(32),
        )
        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.mean(self.backbone(obs))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        log_prob = tanh_normal_log_prob(dist, raw_action, action)
        return raw_action, action, log_prob

    def evaluate_raw(self, obs: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = torch.tanh(raw_action)
        log_prob = tanh_normal_log_prob(dist, raw_action, action)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class PPOValue(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def tanh_normal_log_prob(
    dist: torch.distributions.Normal, raw_action: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    correction = torch.log(torch.clamp(1.0 - action.square(), min=1.0e-6)).sum(dim=-1)
    return log_prob - correction


def train_ppo(args: argparse.Namespace) -> dict:
    if args.contact_backend is None:
        args.contact_backend = "mujoco"
    if args.sim_substeps is None:
        args.sim_substeps = 16 if is_locomotion_env(args.env) else 1
    if args.eval_horizon is None:
        args.eval_horizon = 480
    if args.selection_horizon is None and is_locomotion_env(args.env):
        args.selection_horizon = min(args.eval_horizon, 96)
    if args.episode_length is None:
        args.episode_length = 1000
    if args.force_scale is None:
        args.force_scale = 200.0
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
        ant_disable_joint_limits=args.ant_disable_joint_limits,
        ant_contact_margin=args.ant_contact_margin,
        ant_contact_gap=args.ant_contact_gap,
        ant_min_up=args.ant_min_up,
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
        ),
        hopper_reward=HopperRewardWeights(
            progress=args.hopper_progress_weight,
            height=args.hopper_height_weight,
            angle=args.hopper_angle_weight,
            action=args.hopper_action_penalty,
        ),
        cheetah_reward=CheetahRewardWeights(action=args.cheetah_action_penalty),
    )

    obs_dim = env.num_obs
    actor = PPOActor(obs_dim, env.num_actions).to(env.torch_device)
    critic = PPOValue(obs_dim).to(env.torch_device)
    if args.actor_path is not None:
        actor.load_state_dict(torch.load(args.actor_path, map_location=env.torch_device))
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

    q, qd = env.reset(noise=0.0, stochastic_init=args.stochastic_init)
    prev_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.torch_device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.torch_device)
    best_score = -float("inf")
    best_state = None
    best_obs_rms = None
    best_update = 0
    history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(env.torch_device)
        torch.cuda.synchronize(env.torch_device)

    t0 = time.perf_counter()
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
                if obs_rms is not None:
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
                reward = env.reward(q_next, qd_next, action, obs=next_obs)
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
                approx_kl_value = float(((ratio - 1.0) - log_ratio).mean().detach().cpu())

        if torch.cuda.is_available():
            torch.cuda.synchronize(env.torch_device)

        should_eval = (
            update == 0
            or update == args.updates - 1
            or ((update + 1) % max(1, args.eval_interval) == 0)
        )
        selection = None
        selection_score = None
        if should_eval:
            selection_horizon = args.selection_horizon or args.eval_horizon
            selection = evaluate_policy(env, actor, selection_horizon, obs_rms=obs_rms, termination_penalty=args.termination_penalty)
            selection_score = rollout_selection_score(
                selection,
                num_envs=args.num_envs,
                fall_penalty=args.selection_fall_penalty,
                invalid_penalty=args.selection_invalid_penalty,
            )
            if selection_score > best_score:
                best_score = selection_score
                best_update = update + 1
                best_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
                if obs_rms is not None:
                    best_obs_rms = {
                        "mean": obs_rms.mean.detach().clone(),
                        "var": obs_rms.var.detach().clone(),
                        "count": obs_rms.count,
                    }

        update_s = time.perf_counter() - update_t0
        mean_reward = float(torch.stack(mean_reward_steps).mean().cpu())
        row = {
            "update": update + 1,
            "mean_reward": mean_reward,
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "approx_kl": approx_kl_value,
            "selection_return": selection["return"] if selection is not None else None,
            "selection_score": selection_score,
            "selection_mean_reward": selection["mean_reward"] if selection is not None else None,
            "selection_fall_count": selection["fall_count"] if selection is not None else None,
            "selection_invalid_count": selection["invalid_count"] if selection is not None else None,
            "invalid_resets": invalid_count,
            "fall_resets": fall_count,
            "timeout_resets": timeout_count,
            "update_seconds": update_s,
            "fps": args.num_envs * args.rollout_steps / update_s,
        }
        history.append(row)
        sel_text = f"{selection_score: .1f}" if selection_score is not None else " skipped"
        print(
            f"{args.env} ppo update {update + 1:03d}: reward={mean_reward: .4f} "
            f"sel={sel_text} fps={row['fps']: .1f}",
            flush=True,
        )

    total_s = time.perf_counter() - t0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        actor.load_state_dict(best_state)
    if obs_rms is not None and best_obs_rms is not None:
        obs_rms.mean = best_obs_rms["mean"]
        obs_rms.var = best_obs_rms["var"]
        obs_rms.count = best_obs_rms["count"]
    torch.save(actor.state_dict(), out_dir / f"{args.env}_ppo_actor.pt")
    if obs_rms is not None:
        torch.save(
            {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
            out_dir / f"{args.env}_ppo_obs_rms.pt",
        )

    rollout = evaluate_policy(env, actor, args.eval_horizon, obs_rms=obs_rms, termination_penalty=args.termination_penalty)
    eval_score = rollout_selection_score(
        rollout,
        num_envs=args.num_envs,
        fall_penalty=args.selection_fall_penalty,
        invalid_penalty=args.selection_invalid_penalty,
    )
    video_path = None
    poster_path = None
    if args.render_video:
        render_env = env
        if args.video_num_envs != args.num_envs:
            render_env = NewtonMuJoCoTorchEnv(
                env_name=args.env,
                num_envs=args.video_num_envs,
                device=args.device,
                dt=args.dt,
                sim_substeps=args.sim_substeps,
                mujoco_integrator=args.mujoco_integrator,
                force_scale=args.force_scale,
                contact_backend=args.contact_backend,
                ant_disable_joint_limits=args.ant_disable_joint_limits,
                ant_contact_margin=args.ant_contact_margin,
                ant_contact_gap=args.ant_contact_gap,
                ant_min_up=args.ant_min_up,
                phase_observation=args.phase_observation,
                phase_period=args.phase_period,
                hopper_terminate_angle=args.hopper_terminate_angle,
                locomotion_disable_joint_limits=args.locomotion_disable_joint_limits,
                ant_reward=env.ant_reward,
                hopper_reward=env.hopper_reward,
                cheetah_reward=env.cheetah_reward,
            )
        video_path, poster_path = render_rollout(
            render_env,
            actor,
            out_dir,
            args.eval_horizon,
            f"{args.env}_ppo",
            obs_rms=obs_rms,
        )

    result = {
        "env": args.env,
        "algo": "ppo",
        "title": "SHAC with MuJoCo Warp",
        "timestamp_pacific": pacific_now_iso(),
        "newton_commit": git_commit_for_imported_module(newton),
        "newton_path": str(Path(newton.__path__[0]).resolve()) if hasattr(newton, "__path__") else None,
        "mujoco_warp_commit": git_commit_for_imported_module(mujoco_warp),
        "num_envs": args.num_envs,
        "contact_backend": args.contact_backend,
        "rollout_steps": args.rollout_steps,
        "updates": args.updates,
        "dt": args.dt,
        "sim_substeps": env.sim_substeps,
        "mujoco_integrator": env.mujoco_integrator,
        "nconmax": env.nconmax,
        "njmax": env.njmax,
        "world_spacing": list(env.world_spacing) if env.world_spacing is not None else None,
        "force_scale": args.force_scale,
        "episode_length": args.episode_length,
        "stochastic_init": args.stochastic_init,
        "obs_rms": args.obs_rms,
        "actor_path": str(args.actor_path) if args.actor_path is not None else None,
        "obs_rms_path": str(args.obs_rms_path) if args.obs_rms_path is not None else None,
        "termination_penalty": args.termination_penalty,
        "selection_fall_penalty": args.selection_fall_penalty,
        "selection_invalid_penalty": args.selection_invalid_penalty,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "clip_coef": args.clip_coef,
        "value_clip_coef": args.value_clip_coef,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "hopper_reward": env.hopper_reward.__dict__ if args.env == "hopper" else None,
        "hopper_terminate_angle": env.hopper_terminate_angle if args.env == "hopper" else None,
        "cheetah_reward": env.cheetah_reward.__dict__ if args.env == "cheetah" else None,
        "ant_reward": env.ant_reward.__dict__ if args.env == "ant" else None,
        "ant_disable_joint_limits": env.ant_disable_joint_limits if args.env == "ant" else None,
        "ant_contact_margin": env.ant_contact_margin if args.env == "ant" else None,
        "ant_contact_gap": env.ant_contact_gap if args.env == "ant" else None,
        "ant_min_up": env.ant_min_up if args.env == "ant" else None,
        "phase_observation": env.phase_observation if args.env == "ant" else None,
        "phase_period": env.phase_period if args.env == "ant" else None,
        "locomotion_disable_joint_limits": env.locomotion_disable_joint_limits if is_planar_locomotion_env(args.env) else None,
        "total_seconds": total_s,
        "mean_update_seconds": float(np.mean([h["update_seconds"] for h in history])),
        "mean_fps": float(np.mean([h["fps"] for h in history])),
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
        "video": str(video_path) if video_path is not None else None,
        "poster": str(poster_path) if poster_path is not None else None,
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
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--mujoco-integrator", choices=["euler", "rk4", "implicitfast", "implicit"], default="euler")
    parser.add_argument("--lr", type=float, default=3.0e-4)
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
    parser.add_argument("--obs-rms", dest="obs_rms", action="store_true", default=True)
    parser.add_argument("--no-obs-rms", dest="obs_rms", action="store_false")
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--obs-rms-path", type=Path, default=None)
    parser.add_argument("--ant-progress-weight", type=float, default=1.0)
    parser.add_argument("--ant-heading-weight", type=float, default=1.0)
    parser.add_argument("--ant-up-weight", type=float, default=0.1)
    parser.add_argument("--ant-height-weight", type=float, default=1.0)
    parser.add_argument("--ant-action-penalty", type=float, default=0.0)
    parser.add_argument("--ant-disable-joint-limits", action="store_true")
    parser.add_argument("--ant-contact-margin", type=float, default=0.0)
    parser.add_argument("--ant-contact-gap", type=float, default=None)
    parser.add_argument("--ant-min-up", type=float, default=None)
    parser.add_argument("--phase-observation", action="store_true")
    parser.add_argument("--phase-period", type=int, default=60)
    parser.add_argument("--hopper-height-weight", type=float, default=1.0)
    parser.add_argument("--hopper-progress-weight", type=float, default=1.0)
    parser.add_argument("--hopper-angle-weight", type=float, default=1.0)
    parser.add_argument("--hopper-action-penalty", type=float, default=-0.1)
    parser.add_argument("--hopper-terminate-angle", action="store_true")
    parser.add_argument("--cheetah-action-penalty", type=float, default=-0.1)
    parser.add_argument("--locomotion-disable-joint-limits", action="store_true")
    parser.add_argument("--selection-horizon", type=int, default=None)
    parser.add_argument("--selection-fall-penalty", type=float, default=ANT_DEFAULT_SELECTION_FALL_PENALTY)
    parser.add_argument("--selection-invalid-penalty", type=float, default=ANT_DEFAULT_SELECTION_INVALID_PENALTY)
    parser.add_argument("--contact-backend", choices=["mujoco", "newton", "none"], default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-num-envs", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train_ppo(parse_args())
