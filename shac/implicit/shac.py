"""Short-Horizon Actor-Critic (SHAC) on MJWarp's analytic adjoint.

The update rules mirror the reference implementation (Xu et al., ICLR 2022,
``algorithms/shac.py``): a stochastic tanh-squashed Gaussian actor trained by
back-propagating the discounted short-horizon return plus a bootstrapped
terminal value through the simulator; a TD(lambda) critic with a Polyak target;
frozen observation-normalisation statistics during each rollout; linear
learning-rate decay; and gradient-norm clipping at 1.0.
"""

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

from diffsim import SimConfig
from locomotion import LocomotionEnv
from robots import ROBOTS

# ----------------------------------------------------------------------------- networks


def mlp(sizes, out, layer_norm=True):
    layers, d = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(d, h), nn.ELU()]
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, units, logstd_init=-1.0):
        super().__init__()
        self.mu = mlp([obs_dim, *units], act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), logstd_init))

    def forward(self, obs, deterministic=False):
        mu = self.mu(obs)
        if deterministic:
            return mu
        return mu + torch.exp(self.logstd) * torch.randn_like(mu)


class Critic(nn.Module):
    def __init__(self, obs_dim, units):
        super().__init__()
        self.v = mlp([obs_dim, *units], 1)
        for m in self.v.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x.detach()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta.square() * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-5)

    def snapshot(self):
        s = RunningMeanStd.__new__(RunningMeanStd)
        s.mean, s.var, s.count = self.mean.clone(), self.var.clone(), self.count
        return s

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}


# ----------------------------------------------------------------------------- defaults

PRESETS = {
    "ant": dict(actor_units=[128, 64, 32], critic_units=[64, 64], actor_lr=2e-3, critic_lr=2e-3, target_alpha=0.2),
    "humanoid": dict(actor_units=[256, 128], critic_units=[128, 128], actor_lr=2e-3, critic_lr=5e-4,
                     target_alpha=0.995),
    "go1": dict(actor_units=[256, 128], critic_units=[128, 128], actor_lr=2e-3, critic_lr=2e-3, target_alpha=0.2),
    "humanoid_ref": dict(actor_units=[256, 128], critic_units=[128, 128], actor_lr=2e-3, critic_lr=5e-4,
                         target_alpha=0.995),
    "h1_pd": dict(actor_units=[256, 128], critic_units=[128, 128], actor_lr=2e-3, critic_lr=5e-4, target_alpha=0.995),
    "h1": dict(actor_units=[256, 128], critic_units=[128, 128], actor_lr=2e-3, critic_lr=5e-4, target_alpha=0.995),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    p.add_argument("--integrator", default="implicitfast")
    p.add_argument("--dt", type=float, default=0.01, help="physics timestep (s)")
    p.add_argument("--control-dt", type=float, default=0.02)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--critic-iterations", type=int, default=16)
    p.add_argument("--critic-batches", type=int, default=4)
    p.add_argument("--grad-norm", type=float, default=1.0)
    p.add_argument("--betas", type=float, nargs=2, default=(0.7, 0.95))
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--actor-lr", type=float, default=None)
    p.add_argument("--critic-lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--target-alpha", type=float, default=None)
    p.add_argument("--cone", default="elliptic", choices=("elliptic", "pyramidal"))
    p.add_argument("--state-grad-clip", type=float, default=None, help="cap each world state-adjoint norm at this multiple of the median")
    p.add_argument("--solref-timeconst", type=float, default=None, help="override contact/limit time constant (s)")
    p.add_argument("--init-noise-scale", type=float, default=1.0)
    p.add_argument("--logstd-init", type=float, default=-1.0)
    p.add_argument("--min-height", type=float, default=None, help="override the termination height (m)")
    p.add_argument("--task", nargs="*", default=[], help="task overrides, e.g. velocity_weight=0")
    p.add_argument("--time-budget", type=float, default=None, help="stop after this many wall-clock seconds")
    return p.parse_args()


# ----------------------------------------------------------------------------- training


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    preset = PRESETS[args.robot]
    actor_lr = args.actor_lr or preset["actor_lr"]
    critic_lr = args.critic_lr or preset["critic_lr"]
    substeps = round(args.control_dt / args.dt)
    assert abs(substeps * args.dt - args.control_dt) < 1e-9
    r = ROBOTS[args.robot]
    sim_cfg = SimConfig(xml=r.xml, integrator=args.integrator, timestep=args.dt, substeps=substeps,
                        keyframe=r.keyframe, nconmax=r.nconmax, njmax=r.njmax, solref_timeconst=args.solref_timeconst, cone=args.cone)
    overrides = {k: float(v) for k, v in (kv.split("=") for kv in args.task)}
    env = LocomotionEnv(args.robot, sim_cfg, args.num_envs, args.episode_length, seed=args.seed,
                        task_overrides=overrides, init_noise_scale=args.init_noise_scale)
    env.sim.state_grad_clip = args.state_grad_clip
    if args.min_height is not None:
        import dataclasses as _dc
        env.robot = _dc.replace(env.robot, min_height=args.min_height)
    dev = env.dev
    obs_dim, act_dim = env.num_obs, env.nu
    actor = Actor(obs_dim, act_dim, preset["actor_units"], logstd_init=args.logstd_init).to(dev)
    critic = Critic(obs_dim, preset["critic_units"]).to(dev)
    target = copy.deepcopy(critic)
    for prm in target.parameters():
        prm.requires_grad_(False)
    a_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr, betas=tuple(args.betas), fused=True)
    c_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr, betas=tuple(args.betas), fused=True)
    obs_rms = RunningMeanStd(obs_dim, dev)

    T, N, gamma, lam = args.horizon, args.num_envs, args.gamma, args.lam
    obs_buf = torch.zeros(T, N, obs_dim, device=dev)
    rew_buf = torch.zeros(T, N, device=dev)
    done_buf = torch.zeros(T, N, device=dev)
    nv_buf = torch.zeros(T, N, device=dev)

    ep_ret = torch.zeros(N, device=dev)
    ep_len = torch.zeros(N, device=dev)
    ep_x0 = env.q[:, 0].clone()
    finished = []  # (return, length, speed) for completed episodes
    history = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    sim_time = 0.0
    skipped_updates = 0
    torch.cuda.synchronize()

    for epoch in range(args.epochs):
        t_epoch = time.time()
        frac = epoch / args.epochs
        for g in a_opt.param_groups:
            g["lr"] = (1e-5 - actor_lr) * frac + actor_lr
        for g in c_opt.param_groups:
            g["lr"] = (1e-5 - critic_lr) * frac + critic_lr

        # ---------------------------------------------------------- actor rollout
        env.detach()
        rms = obs_rms.snapshot()
        obs = env.obs()
        obs_rms.update(obs)
        rew_acc = torch.zeros(N, device=dev)
        gamma_k = torch.ones(N, device=dev)
        loss = torch.zeros((), device=dev)
        for i in range(T):
            nobs = rms.normalize(obs)
            obs_buf[i] = nobs.detach()
            action = torch.tanh(actor(nobs))
            obs, rew, done, info = env.step(action)
            obs_rms.update(obs)
            next_v = target(rms.normalize(obs))
            term, trunc, bad = info["terminated"], info["truncated"], info["nonfinite"]
            if trunc.any():
                v_trunc = target(rms.normalize(info["obs_before_reset"]))
                next_v = torch.where(trunc, v_trunc, next_v)
            next_v = torch.where(term | bad, 0.0, next_v)
            rew_acc = rew_acc + gamma_k * rew
            last = i == T - 1
            contrib = -(rew_acc + gamma * gamma_k * next_v)
            if last:
                loss = loss + contrib.sum()
            elif done.any():
                loss = loss + torch.where(done, contrib, 0.0).sum()
            gamma_k = gamma_k * gamma
            gamma_k = torch.where(done, 1.0, gamma_k)
            rew_acc = torch.where(done, 0.0, rew_acc)
            rew_buf[i] = rew.detach()
            done_buf[i] = 1.0 if last else done.float()
            nv_buf[i] = next_v.detach()

            # episode bookkeeping
            ep_ret += rew.detach()
            ep_len += 1
            if done.any():
                idx = done.nonzero().squeeze(-1)
                dur = ep_len[idx] * args.control_dt
                spd = (info["x"][idx] - ep_x0[idx]) / dur
                finished.extend(zip(ep_ret[idx].tolist(), ep_len[idx].tolist(), spd.tolist()))
                ep_ret[idx] = 0.0
                ep_len[idx] = 0.0
                ep_x0[idx] = env.q[idx, 0]
        loss = loss / (T * N)
        sim_time += T * N * args.control_dt
        torch.cuda.synchronize()
        t_rollout = time.time() - t_epoch

        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        t_backward = time.time() - t_epoch - t_rollout
        gnorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_norm)
        gnorm_v = float(gnorm)
        if not math.isfinite(gnorm_v) or gnorm_v > 1e6:
            skipped_updates += 1
            a_opt.zero_grad(set_to_none=True)
        else:
            a_opt.step()

        # ---------------------------------------------------------- critic
        with torch.no_grad():
            target_vals = torch.zeros(T, N, device=dev)
            Ai = torch.zeros(N, device=dev)
            Bi = torch.zeros(N, device=dev)
            lam_i = torch.ones(N, device=dev)
            for i in reversed(range(T)):
                d = done_buf[i]
                lam_i = lam_i * lam * (1.0 - d) + d
                Ai = (1.0 - d) * (lam * gamma * Ai + gamma * nv_buf[i] + (1.0 - lam_i) / (1.0 - lam) * rew_buf[i])
                Bi = gamma * (nv_buf[i] * d + Bi * (1.0 - d)) + rew_buf[i]
                target_vals[i] = (1.0 - lam) * Ai + lam_i * Bi
        flat_obs = obs_buf.reshape(T * N, obs_dim)
        flat_tgt = target_vals.reshape(T * N)
        bs = T * N // args.critic_batches
        for _ in range(args.critic_iterations):
            for b in range(args.critic_batches):
                sl = slice(b * bs, (b + 1) * bs)
                c_loss = (critic(flat_obs[sl]) - flat_tgt[sl]).square().mean()
                c_opt.zero_grad(set_to_none=True)
                c_loss.backward()
                for prm in critic.parameters():
                    if prm.grad is not None:
                        prm.grad.nan_to_num_(0.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_norm)
                c_opt.step()
        with torch.no_grad():
            alpha = args.target_alpha if args.target_alpha is not None else preset["target_alpha"]
            for pt, pc in zip(target.parameters(), critic.parameters()):
                pt.mul_(alpha).add_((1.0 - alpha) * pc)

        torch.cuda.synchronize()
        epoch_time = time.time() - t_epoch
        t_critic = epoch_time - t_rollout - t_backward
        wall = time.time() - t_start
        recent = finished[-200:]
        row = {
            "epoch": epoch,
            "wall_s": wall,
            "epoch_s": epoch_time,
            "rollout_s": t_rollout,
            "backward_s": t_backward,
            "critic_s": t_critic,
            "env_steps": (epoch + 1) * T * N,
            "sim_seconds": sim_time,
            "actor_loss": float(loss.detach()),
            "grad_norm": gnorm_v,
            "critic_loss": float(c_loss),
            "episodes": len(finished),
            "recent_return": float(np.mean([e[0] for e in recent])) if recent else None,
            "recent_length": float(np.mean([e[1] for e in recent])) if recent else None,
            "recent_speed": float(np.mean([e[2] for e in recent])) if recent else None,
            "mean_reward": float(rew_buf.mean()),
            "skipped_updates": skipped_updates,
            "nonfinite_grad_events": env.sim.nonfinite_grad_events,
            "nonfinite_resets": env.nonfinite_resets,
            "logstd": float(actor.logstd.mean()),
        }
        history.append(row)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"ep {epoch:5d} t={wall:7.1f}s ({epoch_time*1e3:5.0f}ms) R={row['recent_return']} "
                  f"len={row['recent_length']} v={row['recent_speed']} r/step={row['mean_reward']:.3f} "
                  f"|g|={gnorm_v:.3g} skip={skipped_updates} nanG={env.sim.nonfinite_grad_events}", flush=True)
        stop = args.time_budget is not None and wall > args.time_budget
        if epoch % 50 == 0 or epoch == args.epochs - 1 or stop:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "obs_rms": obs_rms.state_dict(),
                        "args": vars(args), "preset": preset, "obs_dim": obs_dim, "act_dim": act_dim, "epoch": epoch},
                       out.with_suffix(".pt"))
            out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "preset": preset, "history": history}))
        if stop:
            break

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "obs_rms": obs_rms.state_dict(),
                "args": vars(args), "preset": preset, "obs_dim": obs_dim, "act_dim": act_dim, "epoch": epoch},
               out.with_suffix(".pt"))
    out.with_suffix(".json").write_text(json.dumps({"args": vars(args), "preset": preset, "history": history,
                                                    "total_wall_s": time.time() - t_start}))


if __name__ == "__main__":
    main()
