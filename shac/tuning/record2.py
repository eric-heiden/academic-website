"""Records deterministic rollouts of a shac2/ppo checkpoint for rendering (render.py).

Runs the policy in the reference simulator (implicitfast, 1 ms; training-time contact
softening removed) from --envs randomised starts, optionally with a fixed velocity command,
and saves the per-control-step qpos of every environment plus survival/speed/tracking
summaries. render.py then draws one episode with native MuJoCo.

Usage: python record2.py --ckpt runs/loop/x.pt --out media/x.npz [--cmd 0.8 0 0] [--steps 500]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from eval2 import load
from shac2 import Actor, build_env, make_actor


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--cmd", type=float, nargs=3, default=None)
    # Command schedule: segments "seconds vx vy wz", applied in order (overrides --cmd).
    ap.add_argument("--cmd-schedule", nargs="+", default=None)
    ap.add_argument("--sim", default="ref", choices=("ref", "train"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    ck, cfg = load(a.ckpt)
    over = dict(integrator="implicitfast", timestep=0.001, substeps=round(cfg.control_dt / 0.001), iterations=100,
                ls_iterations=50, geom_solimp=None, geom_margin=None, solref_timeconst=None,
                cone="elliptic") if a.sim == "ref" else {}
    env = build_env(cfg, num_envs=a.envs, seed=a.seed, backward=False, **over)
    meta = ck["meta"]
    actor = make_actor(cfg, meta["obs_dim"], meta["act_dim"]).cuda()
    actor.load_state_dict(ck["actor"])
    mean, var = ck["obs_rms"]["mean"].cuda(), ck["obs_rms"]["var"].cuda()
    squash = meta.get("action_squash", True)
    env.reset_all()
    sched = None
    if a.cmd_schedule:
        sched = []
        for seg in a.cmd_schedule:
            sec, *c = (float(x) for x in seg.split())
            sched += [c] * round(sec / env.dt)
        a.steps = len(sched)
        a.cmd = sched[0]
    if a.cmd is not None:
        env.cmd[:] = torch.tensor(a.cmd, device=env.dev)
        env.cmd_resample = 10 ** 9
    qs, alive = [env.q[:, :].cpu().numpy()], torch.ones(env.N, dtype=torch.bool, device=env.dev)
    life = torch.zeros(env.N, device=env.dev)
    obs = env.obs()
    for t in range(a.steps):
        if sched is not None:
            a.cmd = sched[t]
            env.cmd[:] = torch.tensor(a.cmd, device=env.dev)
        u = actor((obs - mean) / torch.sqrt(var + 1e-5), deterministic=True)
        obs, rew, done, info = env.step(torch.tanh(u) if squash else u, differentiable=False)
        if a.cmd is not None:
            env.cmd[:] = torch.tensor(a.cmd, device=env.dev)
        alive &= ~info["terminated"]
        life += alive.float()
        qs.append(env.q.cpu().numpy())
    q = np.stack(qs, 1)  # (envs, steps + 1, nq)
    summary = {"survival": float((life >= a.steps).float().mean()), "mean_alive_s": float(life.mean()) * env.dt}
    np.savez_compressed(a.out, qpos=q, control_dt=env.dt, xml=env.robot.xml, alive_steps=life.cpu().numpy(),
                        summary=json.dumps(summary))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
