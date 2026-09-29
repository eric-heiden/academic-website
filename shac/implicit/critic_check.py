"""Checks whether a trained SHAC critic predicts the remaining episode lifetime.

Rolls out the deterministic policy from the reset distribution, records the
critic value of every visited state and the number of control steps until
termination (censored at the rollout length), and reports the Spearman rank
correlation between the two. A correlation near zero means the terminal value
term in the actor loss carries little information about falling.
"""

import argparse
import json

import torch

from diffsim import SimConfig
from locomotion import LocomotionEnv
from robots import ROBOTS
from shac import Critic
from evaluate import load_policy

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=150)
p.add_argument("--num-envs", type=int, default=128)
args = p.parse_args()
dev = "cuda:0"
actor, rms, a = load_policy(args.checkpoint, dev)
ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
critic = Critic(ck["obs_dim"], ck["preset"]["critic_units"]).to(dev)
critic.load_state_dict(ck["critic"])
r = ROBOTS[a["robot"]]
cfg = SimConfig(xml=r.xml, integrator=a["integrator"], timestep=a["dt"], substeps=round(a["control_dt"] / a["dt"]),
                keyframe=r.keyframe, nconmax=r.nconmax, njmax=r.njmax)
env = LocomotionEnv(a["robot"], cfg, args.num_envs, episode_length=10**9, seed=7)
values, term_step = [], torch.full((args.num_envs,), float(args.steps), device=dev)
alive = torch.ones(args.num_envs, dtype=torch.bool, device=dev)
obs = env.obs()
with torch.no_grad():
    for t in range(args.steps):
        o = rms.normalize(obs)
        values.append(torch.where(alive, critic(o), torch.nan))
        act = torch.tanh(actor(o, deterministic=True))
        obs, _, _, info = env.step(act, differentiable=False)
        newly = alive & info["terminated"]
        term_step = torch.where(newly, torch.full_like(term_step, t + 1.0), term_step)
        alive &= ~info["terminated"]
V = torch.stack(values)  # (steps, envs)
remaining = term_step[None, :] - torch.arange(args.steps, device=dev)[:, None]
mask = ~torch.isnan(V)
v, rem = V[mask], remaining[mask]
rank = lambda x: torch.argsort(torch.argsort(x)).float()  # noqa: E731
rv, rr = rank(v), rank(rem)
rho = float(((rv - rv.mean()) * (rr - rr.mean())).mean() / (rv.std() * rr.std()))
print(json.dumps({"checkpoint": args.checkpoint, "samples": int(mask.sum()), "spearman_value_vs_remaining": rho,
                  "value_mean": float(v.mean()), "value_sd": float(v.std()),
                  "remaining_mean_steps": float(rem.mean())}))
