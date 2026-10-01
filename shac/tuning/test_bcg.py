"""Consistency tests for bundled contact gradients (bcg.py) on a 4-step window.

1. never:  B branches without perturbation reproduce the unbundled states and gradients (up to the run-to-run
           variation of the GPU's contact ordering).
2. window: with perturbations every step (trigger always) and common noise, the stored-forward and the recomputing
           backward give the same gradient.
3. fd:     with common noise and trigger always, central finite differences of the bundled objective along a random
           action direction match the analytic directional derivative.
Usage: python test_bcg.py [task] [B]
"""

import sys

import torch

from shac2 import Config, build_env

TASK, B = "go1_vel", 4
T, N = 4, 32


def make(bundle_mode):
    cfg = Config(task=TASK, num_envs=N, dt=0.005, horizon=T, seed=3, bcg_branches=B if bundle_mode else 1,
                 bcg_trigger=bundle_mode or "never")
    return build_env(cfg, bundle=bool(bundle_mode))


def snapshot(env):
    return {k: v.clone() for k, v in vars(env).items() if torch.is_tensor(v)}


def restore(env, snap):
    for k, v in snap.items():
        setattr(env, k, v.clone())


def run(env, acts, snap, window, noise_seed=0, grad=True):
    restore(env, snap)
    sim = env.sim
    if getattr(sim, "bundled", False):
        sim.gen_b.manual_seed(noise_seed)
        sim.trigger = torch.ones_like(sim.trigger) if sim.bcg.trigger == "always" else torch.zeros_like(sim.trigger)
    if window:
        if sim.window is None:
            sim.enable_window(T)
        sim.begin_window()
    else:
        sim.window = None
    a = acts.clone().requires_grad_(grad)
    loss = 0.0
    for t in range(T):
        obs, r, d, info = env.step(torch.tanh(a[t]))
        loss = loss + r.sum() + env.q[:, 0].sum() + env.v[:, 0].sum()
    if not grad:
        return float(loss), None, env.q.detach().clone()
    loss.backward()
    return float(loss), a.grad.clone(), env.q.detach().clone()


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


if __name__ == "__main__":
    TASK = sys.argv[1] if len(sys.argv) > 1 else TASK
    B = int(sys.argv[2]) if len(sys.argv) > 2 else B
    torch.manual_seed(0)
    u = make(None)
    snap_u = snapshot(u)
    g = torch.Generator(device="cuda").manual_seed(0)
    acts = 0.5 * torch.randn(T, N, u.nu, device="cuda", generator=g)

    # 1. never vs unbundled (recompute backward), plus unbundled run-to-run variation.
    bn = make("never")
    lu, gu, qu = run(u, acts, snap_u, window=False)
    lu2, gu2, _ = run(u, acts, snap_u, window=False)
    restore(bn, snap_u)
    snap_b = snapshot(bn)
    lb, gb, qb = run(bn, acts, snap_b, window=False)
    print(f"[never] loss {lu:.4f} vs {lb:.4f}; state max diff {float((qu - qb).abs().max()):.2e}; "
          f"grad rel err {rel(gb, gu):.2e} (unbundled run-to-run {rel(gu2, gu):.2e})", flush=True)
    del bn

    # 2. trigger always: recompute vs stored-forward window with common noise.
    ba = make("always")
    restore(ba, snap_u)
    snap_a = snapshot(ba)
    l1, g1, q1 = run(ba, acts, snap_a, window=False, noise_seed=7)
    l1b, g1b, _ = run(ba, acts, snap_a, window=False, noise_seed=7)
    l2, g2, q2 = run(ba, acts, snap_a, window=True, noise_seed=7)
    print(f"[always] recompute vs window: loss {l1:.4f} vs {l2:.4f}; state max diff "
          f"{float((q1 - q2).abs().max()):.2e}; grad rel err {rel(g2, g1):.2e} "
          f"(recompute run-to-run {rel(g1b, g1):.2e}); |g| bundled {float(g1.norm()):.3f} vs unbundled "
          f"{float(gu.norm()):.3f}", flush=True)

    # 3. finite differences of the bundled objective (common noise, trigger always) along a random direction.
    gd = torch.Generator(device="cuda").manual_seed(1)
    d = torch.randn(acts.shape, device="cuda", generator=gd)
    d = d / d.norm()
    slope = float((g1 * d).sum())
    for eps in (1e-2, 1e-3):
        lp, _, _ = run(ba, acts + eps * d, snap_a, window=False, noise_seed=7, grad=False)
        lm, _, _ = run(ba, acts - eps * d, snap_a, window=False, noise_seed=7, grad=False)
        print(f"[fd] eps {eps:g}: analytic slope {slope:.4f}, central FD {(lp - lm) / (2 * eps):.4f}", flush=True)
    # Same check for the unbundled objective, for reference.
    slope_u = float((gu * d).sum())
    lp, _, _ = run(u, acts + 1e-2 * d, snap_u, window=False, grad=False)
    lm, _, _ = run(u, acts - 1e-2 * d, snap_u, window=False, grad=False)
    print(f"[fd unbundled] eps 0.01: analytic slope {slope_u:.4f}, central FD {(lp - lm) / 2e-2:.4f}", flush=True)
