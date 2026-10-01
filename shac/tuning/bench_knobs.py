"""Simulator cost versus solver settings (iterations, ls_iterations, tolerance).

For each setting, settles a batch of worlds under zero action, then times one
control step forward and forward+backward, and reports the mean/max Newton
iterations used and the state error after one control step against the
default setting (iterations 100, ls_iterations 50, tolerance 1e-8).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diffsim import DiffSim, SimConfig
from locomotion import LocomotionEnv
from robots import ROBOTS

p = argparse.ArgumentParser()
p.add_argument("--robot", default="go1")
p.add_argument("--dt", type=float, default=0.002)
p.add_argument("--nworld", type=int, default=1024)
p.add_argument("--settings", nargs="+", default=["100,50,1e-8", "20,10,1e-8", "10,10,1e-8", "5,5,1e-8",
                                                  "3,5,1e-8", "100,50,1e-6", "10,10,1e-6"])
p.add_argument("--reps", type=int, default=20)
p.add_argument("--out", required=True)
args = p.parse_args()
robot = ROBOTS[args.robot]
substeps = round(0.02 / args.dt)


def time_it(fn, reps):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


# Common start states: random-action rollout in the default simulator.
cfg0 = SimConfig(xml=robot.xml, timestep=args.dt, substeps=substeps, keyframe=robot.keyframe,
                 nconmax=robot.nconmax, njmax=robot.njmax)
env = LocomotionEnv(args.robot, cfg0, args.nworld, seed=0)
gen = torch.Generator(device=env.dev).manual_seed(1)
for _ in range(20):
    a = 2 * torch.rand(args.nworld, env.nu, device=env.dev, generator=gen) - 1
    env.step(a, differentiable=False)
q0, v0, w0 = env.q.clone(), env.v.clone(), env.w.clone()
a = 2 * torch.rand(args.nworld, env.nu, device=env.dev, generator=gen) - 1
ctrl = env.ctrl(a)
del env

rows, ref = [], None
for s in args.settings:
    it, ls, tol = s.split(",")
    cfg = SimConfig(xml=robot.xml, timestep=args.dt, substeps=substeps, keyframe=robot.keyframe,
                    nconmax=robot.nconmax, njmax=robot.njmax, iterations=int(it), ls_iterations=int(ls),
                    tolerance=float(tol))
    sim = DiffSim(cfg, args.nworld)
    niter = []
    t_niter = __import__("warp").to_torch(sim.d.solver_niter)

    def fwd():
        return sim.step_nograd(q0, v0, ctrl, w0)

    q1, v1, _ = fwd()
    niter = t_niter.float()
    qg = q0.clone().requires_grad_(True)
    cg = ctrl.clone().requires_grad_(True)

    def fwdbwd():
        qa, va, _ = sim.step(qg, v0, cg, w0)
        (qa[:, 0].sum() + va[:, 0].sum()).backward()

    tf = time_it(fwd, args.reps)
    tb = time_it(fwdbwd, args.reps)
    qg.grad = None
    cg.grad = None
    qa, va, _ = sim.step(qg, v0, cg, w0)
    (va[:, 0].sum()).backward()
    g = cg.grad.clone()
    if ref is None:
        ref = (q1, v1, g)
    err_q = (q1 - ref[0]).abs().amax(1)
    err_v = (v1 - ref[1]).abs().amax(1)
    cos = torch.nn.functional.cosine_similarity(g, ref[2], dim=1)
    row = {"robot": args.robot, "dt": args.dt, "nworld": args.nworld, "iterations": int(it), "ls_iterations": int(ls),
           "tolerance": float(tol), "fwd_ms_per_control_step": 1e3 * tf, "fwdbwd_ms_per_control_step": 1e3 * tb,
           "niter_last_mean": float(niter.mean()), "niter_last_max": float(niter.max()),
           "qerr_median": float(err_q.median()), "qerr_p99": float(err_q.quantile(0.99)),
           "verr_median": float(err_v.median()), "verr_p99": float(err_v.quantile(0.99)),
           "ctrl_grad_cos_median": float(cos.median()), "ctrl_grad_cos_p05": float(cos.quantile(0.05))}
    rows.append(row)
    print(json.dumps(row), flush=True)
    del sim
    Path(args.out).write_text(json.dumps(rows, indent=1))
