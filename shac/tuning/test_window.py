"""Checks that the stored-forward window backward matches the recompute backward."""
import sys
import torch
from shac2 import Config, build_env

task = sys.argv[1] if len(sys.argv) > 1 else "go1_vel"
T = 4
res = {}
for mode in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("recompute", "window")):
    cfg = Config(task=task, num_envs=64, dt=0.005, horizon=T, seed=3)
    env = build_env(cfg)
    if mode.startswith("window"):
        env.sim.enable_window(T)
        env.sim.begin_window()
    g = torch.Generator(device="cuda").manual_seed(0)
    acts = (0.5 * torch.randn(T, 64, env.nu, device="cuda", generator=g)).requires_grad_(True)
    q0 = env.q.clone().requires_grad_(True)
    env.q = q0
    loss = 0
    for t in range(T):
        obs, r, d, info = env.step(torch.tanh(acts[t]))
        loss = loss + r.sum() + env.q[:, 0].sum() + env.v[:, 0].sum()
    loss.backward()
    res[mode + str(len(res))] = (float(loss), acts.grad.clone(), q0.grad.clone(), env.q.detach().clone())
a, b = list(res.values())[0], list(res.values())[-1]
print(task, "loss", a[0], b[0])
print("  state match", float((a[3] - b[3]).abs().max()))
print("  action grad rel err", float((a[1] - b[1]).norm() / a[1].norm()), "norm", float(a[1].norm()))
print("  q0 grad rel err", float((a[2] - b[2]).norm() / a[2].norm()))
