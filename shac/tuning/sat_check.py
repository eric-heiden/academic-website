"""Action saturation of stochastic policies along training (training simulator, from resets).

For checkpoints runs/loop/<run>_epNNNNN.pt, rolls out the sampled policy (as in training) for --steps control steps
and reports the fraction of action components with |tanh(u)| > 0.95, the mean squash derivative 1 - tanh(u)^2, and
the mean |mu| of the pre-tanh policy mean.
Usage: python sat_check.py --runs a b --epochs 0 50 100 --out results/loop/sat_check.json
"""

import argparse
import json
from pathlib import Path

import torch

from eval2 import load
from shac2 import build_env, make_actor


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", required=True)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, env, key = [], None, None
    for run in a.runs:
        for ep in a.epochs:
            path = Path(f"runs/loop/{run}_ep{ep:05d}.pt")
            if not path.exists():
                continue
            ck, cfg = load(path)
            k = (cfg.task, cfg.geom_solimp, cfg.model_edit, json.dumps(cfg.task_overrides, sort_keys=True))
            if k != key:
                env = build_env(cfg, num_envs=a.envs, seed=5, backward=False)
                key = k
            actor = make_actor(cfg, ck["meta"]["obs_dim"], ck["meta"]["act_dim"]).cuda()
            actor.load_state_dict(ck["actor"])
            mean, var = ck["obs_rms"]["mean"].cuda(), ck["obs_rms"]["var"].cuda()
            env.gen.manual_seed(5)
            env.reset_all()
            obs = env.obs()
            sat = dsq = mu_abs = 0.0
            for _ in range(a.steps):
                mu, logstd = actor.dist((obs - mean) / torch.sqrt(var + 1e-5))
                u = mu + torch.exp(logstd) * torch.randn_like(mu)
                act = torch.tanh(u)
                sat += float((act.abs() > 0.95).float().mean())
                dsq += float((1 - act.square()).mean())
                mu_abs += float(mu.abs().mean())
                obs, _, _, _ = env.step(act, differentiable=False)
            row = dict(run=run, epoch=ep, sat_frac=sat / a.steps, squash_deriv=dsq / a.steps, mu_abs=mu_abs / a.steps)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
