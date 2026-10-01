"""Tracking accuracy of a velocity-command policy for fixed commands, in the reference simulator.

For each command (vx, vy, wz), runs --envs environments for --steps control steps with the
command held fixed and reports survival, the mean body-frame planar velocity after a 2 s
transient, and the mean tracking errors.

Usage: python cmd_sweep.py --ckpt runs/loop/x.pt --out results/loop/cmd_x.json [--cmds "0 0 0" "0.5 0 0" ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval2 import load
from shac2 import Actor, build_env


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--cmds", nargs="+", default=["0 0 0", "0.25 0 0", "0.5 0 0", "0.75 0 0", "1.0 0 0",
                                                    "0.5 0 0.5", "0.5 0 -0.5", "0 0 1.0"])
    a = ap.parse_args()
    ck, cfg = load(a.ckpt)
    over = dict(integrator="implicitfast", timestep=0.001, substeps=round(cfg.control_dt / 0.001), iterations=100,
                ls_iterations=50, geom_solimp=None, geom_margin=None, solref_timeconst=None, cone="elliptic")
    env = build_env(cfg, num_envs=a.envs, seed=99, backward=False, **over)
    meta = ck["meta"]
    actor = Actor(meta["obs_dim"], meta["act_dim"], cfg.actor_units, cfg.logstd_init, cfg.layer_norm).cuda()
    actor.load_state_dict(ck["actor"])
    mean, var = ck["obs_rms"]["mean"].cuda(), ck["obs_rms"]["var"].cuda()
    squash = meta.get("action_squash", True)
    rows = []
    for cs in a.cmds:
        c = torch.tensor([float(x) for x in cs.split()], device=env.dev)
        env.gen.manual_seed(99)
        env.reset_all()
        env.cmd_resample = 10 ** 9
        env.cmd[:] = c
        alive = torch.ones(env.N, dtype=torch.bool, device=env.dev)
        vsum = torch.zeros(env.N, 3, device=env.dev)
        n = torch.zeros(env.N, device=env.dev)
        obs = env.obs()
        for t in range(a.steps):
            u = actor((obs - mean) / torch.sqrt(var + 1e-5), deterministic=True)
            obs, rew, done, info = env.step(torch.tanh(u) if squash else u, differentiable=False)
            env.cmd[:] = c
            alive &= ~info["terminated"]
            if t >= 100:
                f = env.F(env, env.q, env.v)
                vb = torch.cat((f.lin_vel_yaw[:, 0:2], f.ang_vel_b[:, 2:3]), -1)
                w = alive.float()
                vsum += w[:, None] * vb
                n += w
        ok = n > 0
        vm = (vsum[ok] / n[ok, None]).mean(0).tolist() if ok.any() else [float("nan")] * 3
        row = {"cmd": c.tolist(), "survival": float(alive.float().mean()), "v_mean": vm,
               "err_xy": float(((vsum[ok] / n[ok, None])[:, 0:2] - c[0:2]).norm(dim=-1).mean()) if ok.any() else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
