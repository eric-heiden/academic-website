"""Deterministic evaluation of shac2 checkpoints in a chosen simulator.

For each checkpoint, runs the deterministic policy from ``--envs`` randomised
starts for ``--steps`` control steps. Each environment counts only its first
episode: survival is the fraction that did not terminate, speed is the mean
forward (x) velocity over the time alive, lateral drift is |y| at the end of the
first episode, and tracking error is the mean norm of (commanded - measured)
body-frame planar velocity over the time alive. ``--sim ref`` evaluates in the
reference simulator (implicitfast, 1 ms); ``--sim train`` in the training one.

Usage: python eval2.py --ckpt runs/loop/x.pt [more.pt ...] --sim ref --out results/loop/eval_x.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shac2 import Actor, Config, build_env, make_actor


def load(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    meta = ck["meta"]
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    return ck, cfg


@torch.no_grad()
def evaluate(ck, cfg, env, steps):
    meta = ck["meta"]
    actor = make_actor(cfg, meta["obs_dim"], meta["act_dim"]).cuda()
    actor.load_state_dict(ck["actor"])
    rms = ck["obs_rms"]
    mean, var = rms["mean"].cuda(), rms["var"].cuda()
    env.reset_all()
    N = env.N
    alive = torch.ones(N, dtype=torch.bool, device=env.dev)
    t_alive = torch.zeros(N, device=env.dev)
    x0 = env.q[:, 0:2].clone()
    disp = torch.zeros(N, 2, device=env.dev)
    trk = torch.zeros(N, device=env.dev)
    ret = torch.zeros(N, device=env.dev)
    obs = env.obs()
    squash = meta.get("action_squash", True)
    # Gait-quality statistics over alive steps: base vertical speed (bouncing), roll/pitch rate (rocking), yaw-rate
    # error (yaw oscillation), action change (smoothness) and joint deviation from the default pose.
    gq = {k: torch.zeros(N, device=env.dev) for k in ("vz", "w_rp", "w_yaw", "act_rate", "pose_dev", "vx")}
    prev_a = None
    for _ in range(steps):
        a = actor((obs - mean) / torch.sqrt(var + 1e-5), deterministic=True)
        a = torch.tanh(a) if squash else a
        q_before = env.q[:, 0:2].clone()
        fb = env.F(env, env.q, env.v)
        vbf, wb = fb.lin_vel_b, fb.ang_vel_b
        live = alive.float()
        gq["vz"] += live * vbf[:, 2].abs()
        gq["vx"] += live * vbf[:, 0]
        gq["w_rp"] += live * wb[:, 0:2].norm(dim=-1)
        gq["w_yaw"] += live * (wb[:, 2] - (env.cmd[:, 2] if env.has_cmd else 0.0)).square()
        gq["pose_dev"] += live * (fb.joint_pos - env.home).square().mean(-1).sqrt()
        if prev_a is not None:
            gq["act_rate"] += live * (a - prev_a).square().sum(-1)
        prev_a = a
        obs, rew, done, info = env.step(a, differentiable=False)
        term = info["terminated"]
        # Position before a possible reset: use the pre-reset observation path via q_before + velocity.
        pos = torch.where(done[:, None], q_before, env.q[:, 0:2])
        still = alive & ~term
        disp = torch.where(alive[:, None], pos - x0, disp)
        t_alive += alive.float() * env.dt
        ret += alive.float() * rew
        if env.has_cmd:
            vb = env.F(env, env.q, env.v).lin_vel_b[:, 0:2]
            trk += alive.float() * (env.cmd[:, 0:2] - vb).norm(dim=-1)
        alive = still & ~done  # a truncation also ends the first episode
    t = t_alive.clamp(min=env.dt)
    out = {
        "survival": float((t_alive >= steps * env.dt - 1e-6).float().mean()),
        "alive_s": float(t_alive.mean()),
        "speed": float((disp[:, 0] / t).mean()),
        "lateral": float(disp[:, 1].abs().mean()),
        "return": float(ret.mean()),
    }
    if env.has_cmd:
        out["track_err"] = float((trk / (t / env.dt)).mean())
    n_alive = (t / env.dt)
    out["gait"] = {"vz_abs": float((gq["vz"] / n_alive).mean()), "w_rollpitch": float((gq["w_rp"] / n_alive).mean()),
                   "yaw_rate_rms": float((gq["w_yaw"] / n_alive).sqrt().mean()),
                   "action_rate": float((gq["act_rate"] / n_alive).mean()),
                   "pose_rms": float((gq["pose_dev"] / n_alive).mean()), "vx_body": float((gq["vx"] / n_alive).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--sim", default="ref", choices=("ref", "train"))
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, env, key = [], None, None
    for path in a.ckpt:
        ck, cfg = load(path)
        # The reference simulator uses the unmodified contact model (training-time contact
        # softening is removed) but keeps robot edits such as actuator gains.
        over = dict(integrator="implicitfast", timestep=0.001, substeps=round(cfg.control_dt / 0.001),
                    iterations=100, ls_iterations=50, geom_solimp=None, geom_margin=None,
                    solref_timeconst=None, cone="elliptic", grad_solimp=None, grad_margin=None) if a.sim == "ref" else {}
        k = (cfg.task, json.dumps(cfg.task_overrides, sort_keys=True), json.dumps(over, sort_keys=True),
             cfg.geom_solimp, cfg.geom_margin, cfg.model_edit)
        if k != key:
            env = build_env(cfg, num_envs=a.envs, seed=a.seed, backward=False, **over)
            key = k
        # Same start states for every checkpoint.
        env.gen.manual_seed(a.seed)
        r = evaluate(ck, cfg, env, a.steps)
        r.update(ckpt=str(path), epoch=ck.get("epoch"), wall_s=ck.get("wall_s"), sim=a.sim)
        rows.append(r)
        print(json.dumps(r), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
