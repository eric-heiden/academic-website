"""Open-loop trajectory optimisation of Humanoid standing with analytic gradients.

Each of ``nworld`` worlds starts from a slightly perturbed standing pose and
optimises its own action sequence a_{0..H-1} (held for one 20 ms control step
each) with Adam to maximise the mean over steps of the standing terms of the
SHAC reward (uprightness, heading, and the height reward). Gradients are
back-propagated through the full horizon, optionally truncated every ``tbptt``
steps. The survival time of the optimised open-loop plan is reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diffsim import SimConfig
from locomotion import LocomotionEnv
from robots import ROBOTS

p = argparse.ArgumentParser()
p.add_argument("--robot", default="humanoid")
p.add_argument("--integrator", default="implicitfast")
p.add_argument("--dt", type=float, default=0.002)
p.add_argument("--nworld", type=int, default=64)
p.add_argument("--horizon", type=int, default=100)
p.add_argument("--tbptt", type=int, default=0)
p.add_argument("--iterations", type=int, default=200)
p.add_argument("--lr", type=float, default=0.03)
p.add_argument("--out", required=True)
args = p.parse_args()

r = ROBOTS[args.robot]
K = round(0.02 / args.dt)
cfg = SimConfig(xml=r.xml, integrator=args.integrator, timestep=args.dt, substeps=K, keyframe=r.keyframe,
                nconmax=r.nconmax, njmax=r.njmax)
env = LocomotionEnv(args.robot, cfg, args.nworld, episode_length=10**9, seed=0, init_noise_scale=0.25)
dev = env.dev
q0, v0 = env.q.clone(), env.v.clone()
u = torch.zeros(args.horizon, args.nworld, env.nu, device=dev, requires_grad=True)
opt = torch.optim.Adam([u], lr=args.lr, betas=(0.7, 0.95))


def rollout(actions, grad=True):
    q, v, w = q0.clone(), v0.clone(), torch.zeros_like(v0)
    alive = torch.ones(args.nworld, device=dev)
    total = torch.zeros(args.nworld, device=dev)
    steps_alive = torch.zeros(args.nworld, device=dev)
    for t in range(args.horizon):
        if grad and args.tbptt and t % args.tbptt == 0:
            q, v = q.detach(), v.detach()
        a = torch.tanh(actions[t])
        ctrl = env.ctrl(a)
        if grad:
            q, v, w = env.sim.step(q, v, ctrl, w)
        else:
            q, v, w = env.sim.step_nograd(q, v, ctrl.detach(), w)
        h = q[:, 2]
        x = (h - 0.84).clamp(-1.0, 0.1)
        rew = 0.1 * env.up_z(q) + env.heading_x(q) + torch.where(x < 0, -200 * x * x, 10 * x)
        total = total + alive * rew
        fell, _ = env.terminated(q, v)
        alive = alive * (~fell).float()
        steps_alive = steps_alive + alive
    return total / args.horizon, steps_alive


hist = []
t0 = time.time()
for it in range(args.iterations):
    obj, _ = rollout(u)
    loss = -obj.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([u], 10.0)
    opt.step()
    if it % 10 == 0 or it == args.iterations - 1:
        with torch.no_grad():
            _, alive_steps = rollout(u, grad=False)
        row = {"iteration": it, "objective": float(obj.mean()), "mean_survival_s": float(alive_steps.mean()) * 0.02,
               "full_horizon_fraction": float((alive_steps >= args.horizon).float().mean()), "wall_s": time.time() - t0}
        hist.append(row)
        print(json.dumps(row), flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps({"args": vars(args), "history": hist}, indent=1))
