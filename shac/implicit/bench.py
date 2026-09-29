"""Throughput of forward and forward+backward physics steps.

Times one control step (``substeps`` physics steps) with and without the analytic
backward, for CUDA-graph replay and for eager (Python-launched) execution.
Reported per physics step and per simulated second. Run with the GPU otherwise idle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diffsim import DiffSim, SimConfig
from locomotion import nominal_qpos
from robots import ROBOTS

p = argparse.ArgumentParser()
p.add_argument("--robots", nargs="+", default=["ant", "go1", "go1_kv", "humanoid", "h1"])
p.add_argument("--integrators", nargs="+", default=["euler", "implicitfast"])
p.add_argument("--nworlds", type=int, nargs="+", default=[256, 1024, 4096])
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--substeps", type=int, default=2)
p.add_argument("--reps", type=int, default=30)
p.add_argument("--eager", action="store_true", help="also time eager (non-graph) execution at 1024 worlds")
p.add_argument("--out", required=True)
args = p.parse_args()


def settle(sim, robot, nworld):
    q = nominal_qpos(robot, sim).expand(nworld, -1).clone()
    v = torch.zeros(nworld, sim.nv, device=sim.torch_device)
    w = torch.zeros_like(v)
    home = q[0, 7:]
    ctrl = sim.action_to_ctrl(torch.zeros(nworld, sim.nu, device=sim.torch_device), robot.position_control, home)
    for _ in range(25):  # 0.5 s with contacts established
        q, v, w = sim.step_nograd(q, v, ctrl, w)
    return q, v, w, ctrl


def time_it(fn, reps):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


rows = []
for name in args.robots:
    robot = ROBOTS[name]
    for integ in args.integrators:
        for n in args.nworlds:
            for graph in ([True, False] if args.eager and n == 1024 else [True]):
                cfg = SimConfig(xml=robot.xml, integrator=integ, timestep=args.dt, substeps=args.substeps,
                                keyframe=robot.keyframe, nconmax=robot.nconmax, njmax=robot.njmax)
                sim = DiffSim(cfg, n, graph=graph)
                q, v, w, ctrl = settle(sim, robot, n)

                def fwd():
                    sim.step_nograd(q, v, ctrl, w)

                qg = q.clone().requires_grad_(True)
                cg = ctrl.clone().requires_grad_(True)

                def fwdbwd():
                    q1, v1, _ = sim.step(qg, v, cg, w)
                    (q1[:, 0].sum() + v1[:, 0].sum()).backward()

                tf = time_it(fwd, args.reps)
                tb = time_it(fwdbwd, args.reps)
                k = args.substeps
                row = {"robot": name, "integrator": integ, "nworld": n, "graph": graph, "timestep": args.dt,
                       "forward_ms_per_physics_step": 1e3 * tf / k,
                       "forward_backward_ms_per_physics_step": 1e3 * tb / k,
                       "forward_world_steps_per_s": n * k / tf,
                       "forward_backward_world_steps_per_s": n * k / tb}
                rows.append(row)
                print(json.dumps(row), flush=True)
                del sim
                Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
