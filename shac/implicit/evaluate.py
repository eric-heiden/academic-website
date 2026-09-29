"""Evaluates a SHAC checkpoint in a chosen simulator configuration.

Each of ``num_envs`` episodes starts from the stochastic reset distribution and
runs the deterministic policy (tanh of the Gaussian mean) for ``steps`` control
steps without auto-reset. Reported statistics:

* survival: fraction of episodes without termination;
* forward speed: x displacement divided by the time survived (m/s);
* lateral drift: |y displacement| at the end of the survived interval (m);
* return: undiscounted training reward summed over the survived interval;
* action rate: RMS of per-step action changes (normalised action units).

Optionally records the root-and-joint trajectory of selected episodes for rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffsim import SimConfig
from locomotion import LocomotionEnv
from robots import ROBOTS
from shac import PRESETS, Actor, RunningMeanStd


def load_policy(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    actor = Actor(ck["obs_dim"], ck["act_dim"], ck["preset"]["actor_units"]).to(dev)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    rms = RunningMeanStd(ck["obs_dim"], dev)
    rms.mean, rms.var = ck["obs_rms"]["mean"].to(dev), ck["obs_rms"]["var"].to(dev)
    return actor, rms, a


@torch.no_grad()
def evaluate(env: LocomotionEnv, actor, rms, steps, record=0):
    env.reset_all()
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.dev)
    ret = torch.zeros(n, device=env.dev)
    length = torch.zeros(n, device=env.dev)
    x0, y0 = env.q[:, 0].clone(), env.q[:, 1].clone()
    xl, yl = x0.clone(), y0.clone()
    rate_sq = torch.zeros(n, device=env.dev)
    prev = torch.zeros(n, env.nu, device=env.dev)
    traj = []
    obs = env.obs()
    for t in range(steps):
        a = torch.tanh(actor(rms.normalize(obs), deterministic=True))
        if record:
            traj.append(env.q[:record].detach().cpu().numpy().copy())
        obs, r, done, info = env.step(a, differentiable=False)
        ret += torch.where(alive, r, 0.0)
        rate_sq += torch.where(alive, (a - prev).square().mean(-1), 0.0)
        prev = a
        length += alive.float()
        # info holds pre-reset positions, so the last survived position is kept.
        xl = torch.where(alive, info["x"], xl)
        yl = torch.where(alive, info["y"], yl)
        alive = alive & ~info["terminated"]
    dur = (length * env.sim.cfg.control_dt).clamp(min=1e-6)
    speed = (xl - x0) / dur
    out = {
        "episodes": n,
        "steps": steps,
        "survival_fraction": float(alive.float().mean()),
        "mean_survived_seconds": float(dur.mean()),
        "mean_forward_speed": float(speed.mean()),
        "median_forward_speed": float(speed.median()),
        "p10_forward_speed": float(speed.quantile(0.1)),
        "mean_abs_lateral_m": float((yl - y0).abs().mean()),
        "mean_return": float(ret.mean()),
        "mean_return_per_step": float((ret / length.clamp(min=1)).mean()),
        "action_rate_rms": float(torch.sqrt(rate_sq / length.clamp(min=1)).mean()),
        "nonfinite_resets": env.nonfinite_resets,
    }
    if record:
        out["trajectory"] = np.stack(traj, 1)  # (record, steps, nq)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--integrator", default=None, help="default: training integrator")
    p.add_argument("--dt", type=float, default=None, help="default: training timestep")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--record", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = "cuda:0"
    actor, rms, targs = load_policy(args.checkpoint, dev)
    robot = ROBOTS[targs["robot"]]
    integ = args.integrator or targs["integrator"]
    dt = args.dt or targs["dt"]
    k = round(targs["control_dt"] / dt)
    cfg = SimConfig(xml=robot.xml, integrator=integ, timestep=dt, substeps=k, keyframe=robot.keyframe,
                    nconmax=robot.nconmax, njmax=robot.njmax)
    env = LocomotionEnv(targs["robot"], cfg, args.num_envs, episode_length=10**9, seed=args.seed)
    env.sim.clear_overflow()
    res = evaluate(env, actor.to(env.dev), rms, args.steps, record=args.record)
    res["overflow_world_fraction"] = env.sim.overflow_stats()
    res["sim"] = {"integrator": integ, "timestep": dt, "substeps": k, "control_dt": targs["control_dt"]}
    res["checkpoint"] = args.checkpoint
    traj = res.pop("trajectory", None)
    if traj is not None:
        np.savez_compressed(Path(args.out).with_suffix(".npz"), qpos=traj, control_dt=targs["control_dt"],
                            xml=robot.xml)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
