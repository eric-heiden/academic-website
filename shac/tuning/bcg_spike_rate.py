"""Per-environment gradient spike rate with and without bundled contact gradients.

Same protocol as grad_spike_rate.py (go1_vel, 64 envs, 4-step window, seed 3, same initial state each pass): a pass
counts as spiked when some environment's action-gradient norm exceeds 10x the median over environments.
Usage: python bcg_spike_rate.py PASSES CONFIG...   with CONFIG = name:B:trigger:sigma_q:clip  (clip 0 = off)
"""
import json
import sys

import torch

import test_bcg as tb
from shac2 import Config, build_env

T, N = 4, 64


def make(B, trig, sq, clip, solimp=None):
    cfg = Config(task="go1_vel", num_envs=N, dt=0.005, horizon=T, seed=3, bcg_branches=B, bcg_trigger=trig,
                 bcg_sigma_q=sq, bcg_sigma_qd=2.5 * sq, ctrl_grad_clip=clip or None, state_grad_clip=clip or None,
                 geom_solimp=solimp)
    return build_env(cfg, bundle=B > 1)


if __name__ == "__main__":
    passes = int(sys.argv[1])
    base = make(1, "never", 0.0, 0)
    snap = tb.snapshot(base)
    g = torch.Generator(device="cuda").manual_seed(0)
    acts = 0.5 * torch.randn(T, N, base.nu, device="cuda", generator=g)
    tb.T, tb.N = T, N
    out = {}
    for spec in sys.argv[2:]:
        name, B, trig, sq, clip, *rest = spec.split(":")
        solimp = [float(x) for x in rest[0].split(",")] if rest else None
        env = base if int(B) == 1 and float(clip) == 0 and solimp is None else make(int(B), trig, float(sq), float(clip), solimp)
        tb.restore(env, snap)
        s = tb.snapshot(env)
        ratios, norms = [], []
        for i in range(passes):
            _, gr, _ = tb.run(env, acts, s, window=True, noise_seed=100 + i)
            pe = gr.norm(dim=(0, 2))
            ratios.append(float(pe.max() / pe.median()))
            norms.append(float(gr.norm()))
        spiked = sum(r > 10 for r in ratios)
        out[name] = dict(B=int(B), trigger=trig, sigma_q=float(sq), clip=float(clip), solimp=solimp, passes=passes, spiked=spiked,
                         max_over_median=ratios, total_norm=norms)
        print(f"{name}: spiked {spiked}/{passes}; max/median ratio median {sorted(ratios)[len(ratios)//2]:.1f} "
              f"max {max(ratios):.1f}; |g| median {sorted(norms)[len(norms)//2]:.2f}", flush=True)
        if env is not base:
            del env
    json.dump(out, open(sys.argv[0].replace(".py", "") and "results/loop/" + (__import__("os").environ.get("SPIKE_OUT") or "bcg_spike_rate.json"), "w"))
