"""Performance benchmarks for SolverMACFluid across grid resolutions.

Run from the newton worktree:
    uv run --extra sim python <this script> --output <json>
"""

from __future__ import annotations

import argparse
import json
import time

import warp as wp

import newton
from newton.solvers import SolverMACFluid


def make_solver(device, res, iters=120, timers=False):
    builder = newton.ModelBuilder()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(b, radius=0.15, cfg=newton.ModelBuilder.ShapeConfig(density=1500.0))
    model = builder.finalize(device=device)
    cfg = SolverMACFluid.Config(
        resolution=(res, res, res),
        cell_size=1.0 / res,
        origin=(-0.5, -0.5, 0.0),
        pressure_iterations=iters,
        kinematic_viscosity=1.0e-4,
        enable_timers=timers,
    )
    return model, SolverMACFluid(model, cfg)


def bench(device, res, steps=60, capture=False, timers=False):
    model, solver = make_solver(device, res, timers=timers)
    s0, s1 = model.state(), model.state()
    dt = 1.0 / 60.0

    # warm up (module load, allocations)
    for _ in range(3):
        solver.step(s0, s1, None, None, dt)
    wp.synchronize_device(model.device)

    graph = None
    if capture and model.device.is_cuda:
        with wp.ScopedDevice(model.device):
            with wp.ScopedCapture() as cap:
                solver.step(s0, s1, None, None, dt)
        graph = cap.graph

    t0 = time.perf_counter()
    for _ in range(steps):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            solver.step(s0, s1, None, None, dt)
    wp.synchronize_device(model.device)
    elapsed = time.perf_counter() - t0

    result = {
        "device": str(model.device),
        "resolution": res,
        "cells": res**3,
        "steps": steps,
        "capture": bool(graph is not None),
        "ms_per_step": 1000.0 * elapsed / steps,
    }
    if timers:
        result["stage_ms"] = {k: sum(v) / len(v) for k, v in solver.timings.items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    results = []
    for res in (32, 48, 64, 96):
        r = bench("cuda:0", res, capture=False)
        print(r)
        results.append(r)
        r = bench("cuda:0", res, capture=True)
        print(r)
        results.append(r)

    # per-stage timings (synchronizing timers, uncaptured)
    r = bench("cuda:0", 64, steps=30, timers=True)
    print(r)
    results.append(r)

    # CPU reference point
    r = bench("cpu", 32, steps=10)
    print(r)
    results.append(r)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
