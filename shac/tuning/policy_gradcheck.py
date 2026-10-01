"""Policy-level gradient diagnostics through the simulator.

For a fixed, randomly initialised deterministic actor and a fixed batch of start
states, define the T-step objective

    J(theta) = (1/N) sum_n sum_{t<T} gamma^t r_t^n * alive_t^n,

where alive_t^n stops the sum after a termination. The script reports, for each
horizon T:

* the analytic directional derivative along the normalised gradient compared
  with central finite differences at several step sizes (common random numbers);
* the gradient norm; and
* the cosine similarity between gradients computed on two disjoint halves of
  the batch, a measure of how consistent the gradient direction is.
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
from shac import PRESETS, Actor, RunningMeanStd

p = argparse.ArgumentParser()
p.add_argument("--robot", required=True)
p.add_argument("--integrator", default="implicitfast")
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--control-dt", type=float, default=0.02)
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 32, 64])
p.add_argument("--warmup", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--checkpoint", default=None, help="optional trained SHAC checkpoint")
p.add_argument("--no-contact", action="store_true")
p.add_argument("--no-gravity", action="store_true")
p.add_argument("--no-limit", action="store_true")
p.add_argument("--solref-timeconst", type=float, default=None)
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--cone", default="elliptic")
p.add_argument("--state-grad-clip", type=float, default=None)
p.add_argument("--ls-iterations", type=int, default=50)
p.add_argument("--no-damper", action="store_true")
p.add_argument("--solimp", type=float, nargs=5, default=None, help="solimp override for all geoms")
p.add_argument("--margin", type=float, default=None, help="margin override for all geoms (m)")
p.add_argument("--grad-solimp", type=float, nargs=5, default=None, help="backward-only solimp")
p.add_argument("--grad-margin", type=float, default=None, help="backward-only margin (m)")
p.add_argument("--model-edit", default=None, help="'+'-separated robots.MODEL_EDITS applied to the model")
p.add_argument("--nconmax", type=int, default=None)
p.add_argument("--njmax", type=int, default=None)
p.add_argument("--out", required=True)
args = p.parse_args()

r = ROBOTS[args.robot]
K = round(args.control_dt / args.dt)
cfg = SimConfig(xml=r.xml, integrator=args.integrator, timestep=args.dt, substeps=K, keyframe=r.keyframe,
                nconmax=args.nconmax or r.nconmax, njmax=args.njmax or r.njmax, solref_timeconst=args.solref_timeconst, cone=args.cone,
                model_edit=args.model_edit,
                iterations=args.iterations, ls_iterations=args.ls_iterations,
                geom_solimp=tuple(args.solimp) if args.solimp else None, geom_margin=args.margin,
                grad_solimp=tuple(args.grad_solimp) if args.grad_solimp else None, grad_margin=args.grad_margin,
                extra_disableflags=(int(__import__("mujoco").mjtDisableBit.mjDSBL_CONTACT) if args.no_contact else 0)
                | (int(__import__("mujoco").mjtDisableBit.mjDSBL_GRAVITY) if args.no_gravity else 0)
                | (int(__import__("mujoco").mjtDisableBit.mjDSBL_LIMIT) if args.no_limit else 0)
                | (int(__import__("mujoco").mjtDisableBit.mjDSBL_DAMPER) if args.no_damper else 0))
env = LocomotionEnv(args.robot, cfg, args.num_envs, episode_length=10**9, seed=args.seed)
dev = env.dev
env.sim.state_grad_clip = args.state_grad_clip
torch.manual_seed(args.seed)
preset = PRESETS[args.robot]
actor = Actor(env.num_obs, env.nu, preset["actor_units"]).to(dev)
rms = RunningMeanStd(env.num_obs, dev)
if args.checkpoint:
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    actor.load_state_dict(ck["actor"])
    rms.mean, rms.var = ck["obs_rms"]["mean"], ck["obs_rms"]["var"]
params = [p_ for p_ in actor.parameters()]

# Warm up with the actor to reach typical states, recording obs statistics.
with torch.no_grad():
    for _ in range(args.warmup):
        o = env.obs()
        if not args.checkpoint:
            rms.update(o)
        a = torch.tanh(actor(rms.normalize(o), deterministic=True))
        env.step(a, differentiable=False)
start = (env.q.clone(), env.v.clone(), env.w.clone(), env.prev_action.clone(), env.progress.clone())
gamma = 0.99


def objective(T, differentiable=True):
    env.q, env.v, env.w, env.prev_action, env.progress = (x.clone() for x in start)
    alive = torch.ones(args.num_envs, device=dev)
    per_env = torch.zeros(args.num_envs, device=dev)
    for t in range(T):
        a = torch.tanh(actor(rms.normalize(env.obs()), deterministic=True))
        _, rew, done, info = env.step(a, differentiable=differentiable)
        per_env = per_env + (gamma ** t) * alive * rew
        alive = alive * (~info["terminated"]).float()
    return per_env


def flat_grad(loss):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p_)).reshape(-1) for x, p_ in zip(g, params)])


def set_params(vec):
    i = 0
    with torch.no_grad():
        for p_ in params:
            n = p_.numel()
            p_.copy_(vec[i:i + n].view_as(p_))
            i += n


theta0 = torch.cat([p_.detach().reshape(-1) for p_ in params]).clone()
rows = []
for T in args.horizons:
    t0 = time.time()
    per_env = objective(T)
    J = per_env.mean()
    half = args.num_envs // 2
    g = flat_grad(J)
    g1 = flat_grad(per_env[:half].mean())
    g2 = flat_grad(per_env[half:].mean())
    torch.cuda.synchronize()
    grad_time = time.time() - t0
    gnorm = float(g.norm())
    d = g / (g.norm() + 1e-12)
    analytic = float((g * d).sum())
    fd = {}
    for eps in (1e-2, 1e-3, 1e-4):
        with torch.no_grad():
            set_params(theta0 + eps * d)
            jp = float(objective(T, differentiable=False).mean())
            set_params(theta0 - eps * d)
            jm = float(objective(T, differentiable=False).mean())
            set_params(theta0)
        fd[str(eps)] = (jp - jm) / (2 * eps)
    row = {
        "horizon": T,
        "objective": float(J),
        "grad_norm": gnorm,
        "analytic_slope": analytic,
        "fd_slope": fd,
        "half_batch_cosine": float(torch.nn.functional.cosine_similarity(g1, g2, dim=0)),
        "grad_seconds": grad_time,
        "nonfinite_grad_events": env.sim.nonfinite_grad_events,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
