"""Smoke test: capture, roll out, and FD-check one robot configuration."""

import argparse
import time

import torch

from diffsim import DiffSim, SimConfig
from robots import ROBOTS

p = argparse.ArgumentParser()
p.add_argument("--robot", default="ant")
p.add_argument("--integrator", default="implicitfast")
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--substeps", type=int, default=2)
p.add_argument("--nworld", type=int, default=64)
p.add_argument("--steps", type=int, default=5)
args = p.parse_args()

r = ROBOTS[args.robot]
cfg = SimConfig(xml=r.xml, integrator=args.integrator, timestep=args.dt, substeps=args.substeps,
                keyframe=r.keyframe, nconmax=r.nconmax, njmax=r.njmax)
t0 = time.time()
sim = DiffSim(cfg, args.nworld)
torch.cuda.synchronize()
print(f"build+capture {time.time()-t0:.1f}s nq={sim.nq} nv={sim.nv} nu={sim.nu}")
dev = sim.torch_device
g = torch.Generator(device=dev).manual_seed(0)

qpos0 = sim.init_qpos.expand(args.nworld, -1).clone()
qvel0 = torch.zeros(args.nworld, sim.nv, device=dev)
warm0 = torch.zeros_like(qvel0)
mid = 0.5 * (sim.ctrl_low + sim.ctrl_high) if r.position_control else torch.zeros(sim.nu, device=dev)
half = 0.5 * (sim.ctrl_high - sim.ctrl_low)
if r.position_control:
    mid = torch.tensor(sim.mjd.ctrl if sim.mjd.ctrl.any() else sim.mjd.qpos[7:], dtype=torch.float32, device=dev)
    half = 0.3 * half
ctrls = mid + 0.5 * half * (2 * torch.rand(args.steps, args.nworld, sim.nu, device=dev, generator=g) - 1)
ctrls = sim.clamp_ctrl(ctrls)

# settle 0.5 s so contacts are active
q, v, w = qpos0, qvel0, warm0
for _ in range(int(0.5 / cfg.control_dt)):
    q, v, w = sim.step_nograd(q, v, mid.expand(args.nworld, -1), w)
print("settled height", q[:, 2].mean().item(), "finite", torch.isfinite(q).all().item())


def rollout(c, grad=False):
    qq, vv, ww = q.clone(), v.clone(), w.clone()
    for t in range(args.steps):
        if grad:
            qq, vv, ww = sim.step(qq, vv, c[t], ww)
        else:
            qq, vv, ww = sim.step_nograd(qq, vv, c[t], ww)
    return qq[:, 0] + 0.1 * vv[:, 0] + 0.5 * qq[:, 2]


c = ctrls.clone().requires_grad_(True)
loss = rollout(c, grad=True).sum()
loss.backward()
direction = torch.randn_like(ctrls) * 0.1 * half
analytic = (c.grad * direction).sum(dim=(0, 2))
for eps in (1e-1, 3e-2, 1e-2):
    with torch.no_grad():
        fd = (rollout(ctrls + eps * direction) - rollout(ctrls - eps * direction)) / (2 * eps)
    rel = (analytic - fd).abs() / (fd.abs() + 1e-3 * fd.abs().mean())
    cos = torch.nn.functional.cosine_similarity(analytic, fd, dim=0).item()
    print(f"eps={eps:g} median rel={rel.median().item():.3g} p90={rel.quantile(0.9).item():.3g} frac<5%={(rel<0.05).float().mean().item():.2f} cos(world vec)={cos:.4f}")

# timing
torch.cuda.synchronize()
n = 50
t0 = time.time()
for _ in range(n):
    q2, v2, w2 = sim.step_nograd(q, v, ctrls[0], w)
torch.cuda.synchronize()
tf = (time.time() - t0) / n
c = ctrls.clone().requires_grad_(True)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    loss = rollout(c, grad=True).sum()
    loss.backward()
torch.cuda.synchronize()
tb = (time.time() - t0) / (10 * args.steps)
print(f"control step fwd {tf*1e3:.2f} ms, fwd+bwd {tb*1e3:.2f} ms (nworld={args.nworld}, K={args.substeps})")
