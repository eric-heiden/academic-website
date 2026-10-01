"""PPO baseline (rsl_rl / Isaac Lab settings) on the same MJWarp environments, forward simulation only.

Gaussian policy with a state-independent std (init 1.0), unbounded actions, clipped surrogate and
clipped value loss, GAE, adaptive learning rate on the KL (desired 0.01), 5 epochs x 4 minibatches,
entropy bonus. Time-outs bootstrap with gamma * V(s_timeout). Checkpoints are compatible with
eval2.py (they store the mean network under the "actor" key of a shac2.Actor).

Usage: python ppo.py --task h1_vel --out runs/loop/ppo_h1 [--set key=value ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shac2 import Config as ShacConfig
from shac2 import build_env, mlp
from tasks import TASKS


@dataclasses.dataclass
class PPOConfig:
    task: str = "h1_vel"
    dt: float = 0.005
    control_dt: float = 0.02
    integrator: str = "implicitfast"
    num_envs: int = 4096
    steps: int = 24
    iterations: int = 1000
    epochs: int = 5
    minibatches: int = 4
    lr: float = 1e-3
    desired_kl: float = 0.01
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    init_std: float = 1.0
    units: tuple = (128, 128, 128)
    obs_norm: bool = False
    seed: int = 0
    time_budget: float | None = None
    save_every: int = 50
    task_overrides: dict = dataclasses.field(default_factory=dict)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, units, init_std):
        super().__init__()
        self.mu = mlp([obs_dim, *units], act_dim, layer_norm=False)
        self.v = mlp([obs_dim, *units], 1, layer_norm=False)
        self.logstd = nn.Parameter(torch.full((act_dim,), math.log(init_std)))

    def dist(self, obs):
        return torch.distributions.Normal(self.mu(obs), torch.exp(self.logstd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tset", nargs="*", default=[])
    a = ap.parse_args()
    cfg = PPOConfig(task=a.task)
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert hasattr(cfg, k), k
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    for kv in a.tset:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg.task_overrides[k] = v
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    scfg = ShacConfig(task=cfg.task, dt=cfg.dt, control_dt=cfg.control_dt, integrator=cfg.integrator,
                      num_envs=cfg.num_envs, seed=cfg.seed, task_overrides=cfg.task_overrides,
                      actor_units=cfg.units, layer_norm=False)
    env = build_env(scfg, backward=False)
    dev = env.dev
    N, S = cfg.num_envs, cfg.steps
    obs_dim, act_dim = env.num_obs, env.nu
    ac = ActorCritic(obs_dim, act_dim, cfg.units, cfg.init_std).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    lr = cfg.lr

    obs_b = torch.zeros(S, N, obs_dim, device=dev)
    act_b = torch.zeros(S, N, act_dim, device=dev)
    logp_b = torch.zeros(S, N, device=dev)
    val_b = torch.zeros(S, N, device=dev)
    rew_b = torch.zeros(S, N, device=dev)
    done_b = torch.zeros(S, N, device=dev)
    mu_b = torch.zeros(S, N, act_dim, device=dev)

    meta = {"config": dataclasses.asdict(scfg), "ppo": dataclasses.asdict(cfg), "task": dataclasses.asdict(env.task),
            "obs_dim": obs_dim, "act_dim": act_dim, "algo": "ppo"}
    history = []
    t_start = time.time()
    obs = env.obs()
    identity_rms = {"mean": torch.zeros(obs_dim, device=dev), "var": torch.ones(obs_dim, device=dev) - 1e-5,
                    "count": 1.0}

    def save(it):
        # Store the mean network as a shac2 Actor (mu + logstd) for eval2.py; tanh is not applied by PPO,
        # so eval uses action_squash=False recorded in meta.
        sd = {"mu." + k: v for k, v in ac.mu.state_dict().items()}
        sd["logstd"] = ac.logstd.detach()
        torch.save({"actor": sd, "obs_rms": identity_rms, "meta": {**meta, "action_squash": False}, "epoch": it,
                    "wall_s": time.time() - t_start}, out.with_suffix(".pt"))
        out.with_suffix(".json").write_text(json.dumps({**meta, "history": history,
                                                        "total_wall_s": time.time() - t_start}))

    for it in range(cfg.iterations):
        t0 = time.time()
        with torch.no_grad():
            for i in range(S):
                d = ac.dist(obs)
                act = d.sample()
                obs_b[i] = obs
                act_b[i] = act
                mu_b[i] = d.mean
                logp_b[i] = d.log_prob(act).sum(-1)
                val_b[i] = ac.v(obs).squeeze(-1)
                obs, rew, done, info = env.step(act, differentiable=False)
                trunc = info["truncated"]
                v_to = ac.v(info["obs_before_reset"]).squeeze(-1)
                rew_b[i] = rew + cfg.gamma * torch.where(trunc, v_to, 0.0)
                done_b[i] = done.float()
            last_v = ac.v(obs).squeeze(-1)
            adv = torch.zeros(S, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for i in reversed(range(S)):
                nv = last_v if i == S - 1 else val_b[i + 1]
                nonterm = 1.0 - done_b[i]
                delta = rew_b[i] + cfg.gamma * nv * nonterm - val_b[i]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[i] = gae
            ret = adv + val_b
        t_roll = time.time()

        B = S * N
        f_obs, f_act, f_logp = obs_b.reshape(B, -1), act_b.reshape(B, -1), logp_b.reshape(B)
        f_adv, f_ret, f_val, f_mu = adv.reshape(B), ret.reshape(B), val_b.reshape(B), mu_b.reshape(B, -1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        old_std = torch.exp(ac.logstd.detach())
        mb = B // cfg.minibatches
        stats = torch.zeros(4, device=dev)
        for _ in range(cfg.epochs):
            perm = torch.randperm(B, device=dev)
            for k in range(cfg.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                d = ac.dist(f_obs[idx])
                logp = d.log_prob(f_act[idx]).sum(-1)
                ent = d.entropy().sum(-1).mean()
                # KL(old || new) for the adaptive learning rate (rsl_rl).
                with torch.no_grad():
                    std = torch.exp(ac.logstd)
                    kl = (torch.log(std / old_std) + (old_std ** 2 + (f_mu[idx] - d.mean) ** 2) / (2 * std ** 2)
                          - 0.5).sum(-1).mean()
                    kl_v = float(kl)
                    if kl_v > 2 * cfg.desired_kl:
                        lr = max(1e-5, lr / 1.5)
                    elif 0 < kl_v < cfg.desired_kl / 2:
                        lr = min(1e-2, lr * 1.5)
                    for g in opt.param_groups:
                        g["lr"] = lr
                ratio = torch.exp(logp - f_logp[idx])
                s1 = ratio * f_adv[idx]
                s2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * f_adv[idx]
                pol_loss = -torch.min(s1, s2).mean()
                v = ac.v(f_obs[idx]).squeeze(-1)
                v_clip = f_val[idx] + (v - f_val[idx]).clamp(-cfg.clip, cfg.clip)
                v_loss = torch.max((v - f_ret[idx]).square(), (v_clip - f_ret[idx]).square()).mean()
                loss = pol_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()
                stats += torch.stack((pol_loss.detach(), v_loss.detach(), ent.detach(), kl.detach()))
            # (old_mu/old_std for KL are those of the rollout policy, as in rsl_rl)
        stats = (stats / (cfg.epochs * cfg.minibatches)).cpu().tolist()
        st = env.pop_stats()
        wall = time.time() - t_start
        row = {"epoch": it, "wall_s": wall, "epoch_s": time.time() - t0, "rollout_s": t_roll - t0,
               "env_steps": (it + 1) * B, "pol_loss": stats[0], "v_loss": stats[1], "entropy": stats[2],
               "kl": stats[3], "lr": lr, "std": float(torch.exp(ac.logstd).mean()), "mean_reward": float(rew_b.mean()),
               **{"ep_" + k: v for k, v in st.items()}}
        history.append(row)
        if it % 20 == 0 or it == cfg.iterations - 1:
            print(f"it {it:5d} t={wall:7.1f}s ({row['epoch_s']*1e3:5.0f}ms) R={st.get('return')} "
                  f"len={st.get('length')} trk={st.get('track_err')} v={st.get('speed_x')} lr={lr:.2e} "
                  f"std={row['std']:.2f}", flush=True)
        stop = cfg.time_budget is not None and wall > cfg.time_budget
        if it % cfg.save_every == 0 or it == cfg.iterations - 1 or stop:
            save(it)
            if it % cfg.save_every == 0:
                sd = {"mu." + k: v for k, v in ac.mu.state_dict().items()}
                sd["logstd"] = ac.logstd.detach()
                torch.save({"actor": sd, "obs_rms": identity_rms, "meta": {**meta, "action_squash": False},
                            "epoch": it, "wall_s": wall}, out.parent / f"{out.name}_ep{it:05d}.pt")
        if stop:
            break
    save(it)


if __name__ == "__main__":
    main()
