"""Reproduce the headline validation numbers of the SolverMACFluid test suite.

Run from the newton worktree:
    uv run --extra dev python <this script> --output <json>
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMACFluid


def make_tank(device, gravity=-9.81, sphere=None):
    builder = newton.ModelBuilder(gravity=gravity)
    if sphere is not None:
        pos, radius, mass = sphere
        b = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(b, radius=radius)
    return builder.finalize(device=device)


def make_solver(model, res, iters=150, viscosity=0.0):
    cfg = SolverMACFluid.Config(
        resolution=(res, res, res),
        cell_size=1.0 / res,
        origin=(0.0, 0.0, 0.0),
        pressure_iterations=iters,
        kinematic_viscosity=viscosity,
    )
    return SolverMACFluid(model, cfg)


def projection(device):
    model = make_tank(device, gravity=0.0)
    solver = make_solver(model, 16, iters=120)
    g = solver.grid
    rng = np.random.default_rng(7)
    g.u.assign(rng.uniform(-1, 1, size=g.u.shape).astype(np.float32))
    g.v.assign(rng.uniform(-1, 1, size=g.v.shape).astype(np.float32))
    g.w.assign(rng.uniform(-1, 1, size=g.w.shape).astype(np.float32))
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 1e-3)
    d = solver.read_diagnostics()
    return {
        "div_l2_pre": d["div_l2_pre"],
        "div_l2_post": d["div_l2_post"],
        "reduction_factor": d["div_l2_pre"] / max(d["div_l2_post"], 1e-30),
        "pressure_residual": d["pressure_residual"],
    }


def hydrostatics(device):
    model = make_tank(device)
    solver = make_solver(model, 16, iters=200)
    s0, s1 = model.state(), model.state()
    for _ in range(5):
        solver.step(s0, s1, None, None, 1 / 60)
    u = solver.velocity_u.numpy()
    v = solver.velocity_v.numpy()
    w = solver.velocity_w.numpy()
    p = solver.pressure.numpy()
    dp = (p[:, :, 1:] - p[:, :, :-1]).mean()
    expected = -1000.0 * 9.81 * solver.dx
    return {
        "max_velocity": float(max(np.abs(u).max(), np.abs(v).max(), np.abs(w).max())),
        "pressure_gradient": float(dp),
        "expected_gradient": expected,
        "gradient_rel_error": float(abs(dp - expected) / abs(expected)),
    }


def buoyancy(device):
    out = {}
    radius = 0.2
    analytic = 1000.0 * 9.81 * 4 / 3 * math.pi * radius**3
    for res in (16, 24, 32, 48):
        model = make_tank(device, sphere=((0.5, 0.5, 0.5), radius, 10.0))
        solver = make_solver(model, res, iters=300)
        s0, s1 = model.state(), model.state()
        for _ in range(3):
            solver.step(s0, s1, None, None, 1 / 60)
        d = solver.read_diagnostics()
        labels = solver.cell_label.numpy()
        v_vox = int((labels == 0).sum()) * solver.dx**3
        wrench_z = float(d["body_wrench"][0][2])
        out[str(res)] = {
            "wrench_z": wrench_z,
            "voxel_expected": 1000.0 * 9.81 * v_vox,
            "analytic": analytic,
            "rel_error_analytic": wrench_z / analytic - 1.0,
            "noslip_max": d["noslip_max"],
        }
    return out


def momentum_balance(device):
    model = make_tank(device, sphere=((0.5, 0.5, 0.6), 0.15, 10.0))
    solver = make_solver(model, 16, iters=120, viscosity=1e-4)
    s0, s1 = model.state(), model.state()
    qd = np.zeros((1, 6), dtype=np.float32)
    qd[0, 2] = -0.4
    s0.body_qd.assign(qd)
    solver.step(s0, s1, None, None, 1 / 120)
    d = solver.read_diagnostics()
    err = float(np.abs(np.array(d["momentum_balance_error"])).max())
    imp = float(np.abs(np.array(d["boundary_impulse_pressure"])).max())
    return {"error_Ns": err, "impulse_scale_Ns": imp, "relative": err / imp}


def restore_bitexact(device):
    model = make_tank(device, sphere=((0.5, 0.5, 0.5), 0.15, 10.0))
    solver = make_solver(model, 16, iters=60, viscosity=1e-4)
    s0, s1 = model.state(), model.state()
    rng = np.random.default_rng(5)
    solver.grid.u.assign(rng.uniform(-1, 1, size=solver.grid.u.shape).astype(np.float32))
    solver.step(s0, s1, None, None, 1 / 60)
    u1 = solver.velocity_u.numpy().copy()
    solver.coupling_notify_input_state_update(s0, 0, iteration_restart=True, dt=1 / 60)
    solver.step(s0, s1, None, None, 1 / 60)
    u2 = solver.velocity_u.numpy().copy()
    return {"bit_exact": bool(np.array_equal(u1, u2)), "max_diff": float(np.abs(u1 - u2).max())}


def cpu_gpu_consistency():
    def run(dev):
        model = make_tank(dev, sphere=((0.5, 0.5, 0.6), 0.15, 10.0))
        solver = make_solver(model, 16, iters=100, viscosity=1e-4)
        s0, s1 = model.state(), model.state()
        qd = np.zeros((1, 6), dtype=np.float32)
        qd[0, 2] = -0.3
        s0.body_qd.assign(qd)
        for _ in range(3):
            solver.step(s0, s1, None, None, 1 / 120)
        return solver.velocity_u.numpy(), solver.body_impulse.numpy()

    u_gpu, imp_gpu = run("cuda:0")
    u_cpu, imp_cpu = run("cpu")
    return {
        "max_velocity_diff": float(np.abs(u_gpu - u_cpu).max()),
        "max_impulse_rel_diff": float(np.abs(imp_gpu - imp_cpu).max() / max(np.abs(imp_gpu).max(), 1e-12)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    device = "cuda:0"
    results = {
        "projection": projection(device),
        "hydrostatics": hydrostatics(device),
        "buoyancy_convergence": buoyancy(device),
        "momentum_balance": momentum_balance(device),
        "coupled_restart_restore": restore_bitexact(device),
        "cpu_gpu_consistency": cpu_gpu_consistency(),
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
