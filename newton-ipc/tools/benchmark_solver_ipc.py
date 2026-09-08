"""Run SolverIPC correctness, stability, and CUDA performance experiments."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton._src.solvers.ipc import kernels
from newton.solvers import style3d


@wp.kernel
def array_inner_atomic(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    output: wp.array[float],
):
    """Reference reduction with one contended atomic per vector."""
    index = wp.tid()
    wp.atomic_add(output, 0, wp.dot(a[index], b[index]))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def time_graph(graph, *, batches: int, launches_per_batch: int) -> dict[str, float]:
    for _ in range(20):
        wp.capture_launch(graph)
    wp.synchronize()
    samples = []
    for _ in range(batches):
        started = time.perf_counter()
        for _ in range(launches_per_batch):
            wp.capture_launch(graph)
        wp.synchronize()
        samples.append((time.perf_counter() - started) * 1.0e6 / launches_per_batch)
    return {
        "median_us": statistics.median(samples),
        "p95_us": percentile(samples, 95.0),
        "minimum_us": min(samples),
        "batches": batches,
        "launches_per_batch": launches_per_batch,
    }


def particle_model(count: int, device) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverIPC.register_custom_attributes(builder)
    builder.add_particles(
        [(float(i % 128) * 0.01, float(i // 128) * 0.01, 0.3) for i in range(count)],
        [(0.0, 0.0, 0.0)] * count,
        [1.0] * count,
    )
    return builder.finalize(device=device)


def graph_mode_benchmark(device) -> list[dict]:
    rows = []
    for mode in ("conditional", "unrolled"):
        model = particle_model(4096, device)
        solver = newton.solvers.SolverIPC(
            model,
            config=newton.solvers.SolverIPC.Config(
                graph_mode=mode,
                max_newton_iterations=16,
                max_pcg_iterations=8,
                max_line_search_iterations=8,
            ),
        )
        state_in, state_out = model.state(), model.state()
        started = time.perf_counter()
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        wp.synchronize()
        capture_ms = (time.perf_counter() - started) * 1.0e3
        timing = time_graph(capture.graph, batches=30, launches_per_batch=100)
        wp.capture_launch(capture.graph)
        row = {
            "mode": mode,
            "particle_count": model.particle_count,
            "capture_ms": capture_ms,
            "status": solver.Status(int(solver.diagnostics.status.numpy()[0])).name,
            **timing,
        }
        rows.append(row)
    return rows


def factorization_benchmark(device) -> list[dict]:
    count = 1_000_000
    static_diagonal = wp.full(count, 100.0, dtype=float, device=device)
    contact_hessian = wp.full(count, 1000.0, dtype=float, device=device)
    flags = wp.full(count, int(newton.ParticleFlags.ACTIVE), dtype=int, device=device)
    active = wp.ones(1, dtype=int, device=device)
    output = wp.zeros(count, dtype=wp.mat33, device=device)
    normal = wp.normalize(wp.vec3(0.3, -0.2, 1.0))
    rows = []
    for name, kernel in (
        ("rank_one", kernels.prepare_rank_one_preconditioner),
        ("dense_inverse", kernels.prepare_dense_preconditioner),
    ):
        started = time.perf_counter()
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                kernel,
                dim=count,
                inputs=[static_diagonal, contact_hessian, normal, flags, active],
                outputs=[output],
                device=device,
            )
        wp.synchronize()
        capture_ms = (time.perf_counter() - started) * 1.0e3
        rows.append(
            {
                "factorization": name,
                "block_count": count,
                "capture_ms": capture_ms,
                **time_graph(capture.graph, batches=40, launches_per_batch=100),
            }
        )
    return rows


def reduction_benchmark(device) -> list[dict]:
    count = 1_000_000
    block_dim = 256
    block_count = (count + block_dim - 1) // block_dim
    a = wp.full(count, wp.vec3(1.0, 2.0, 3.0), dtype=wp.vec3, device=device)
    b = wp.full(count, wp.vec3(0.5, 0.25, 0.125), dtype=wp.vec3, device=device)
    output = wp.zeros(1, dtype=float, device=device)
    rows = []
    for name in ("per_element_atomic", "tile_sum"):
        with wp.ScopedCapture(device=device) as capture:
            output.zero_()
            if name == "per_element_atomic":
                wp.launch(
                    array_inner_atomic,
                    dim=count,
                    inputs=[a, b],
                    outputs=[output],
                    device=device,
                )
            else:
                wp.launch(
                    kernels.array_inner_tiled,
                    dim=(block_count, block_dim),
                    block_dim=block_dim,
                    inputs=[a, b, count, 0],
                    outputs=[output],
                    device=device,
                )
        timing = time_graph(capture.graph, batches=20, launches_per_batch=20)
        wp.capture_launch(capture.graph)
        rows.append(
            {
                "reduction": name,
                "vector_count": count,
                "atomic_adds": count if name == "per_element_atomic" else block_count,
                "result": float(output.numpy()[0]),
                "expected": 1.375 * count,
                **timing,
            }
        )
    return rows


def cloth_model(
    resolution: int, device, *, height: float = 1.2, tilted: bool = True
) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    newton.solvers.SolverIPC.register_custom_attributes(builder)
    rotation = wp.quat_rpy(0.12, -0.18, 0.2) if tilted else wp.quat_identity()
    style3d.add_cloth_grid(
        builder,
        pos=(-0.5 * resolution * 0.04, -0.5 * resolution * 0.04, height),
        rot=rotation,
        vel=(0.0, 0.0, 0.0),
        dim_x=resolution,
        dim_y=resolution,
        cell_x=0.04,
        cell_y=0.04,
        mass=0.005,
        particle_radius=0.01,
        tri_aniso_ke=wp.vec3(5.0e2, 5.0e2, 5.0e1),
        edge_aniso_ke=wp.vec3(2.0e-5),
    )
    return builder.finalize(device=device)


def cloth_config(**overrides) -> newton.solvers.SolverIPC.Config:
    values = {
        "minimum_separation": 0.01,
        "contact_distance": 0.05,
        "barrier_stiffness": 0.005,
        "max_newton_iterations": 64,
        "max_pcg_iterations": 24,
        "max_line_search_iterations": 16,
        "absolute_tolerance": 0.01,
        "relative_tolerance": 0.002,
        "energy_tolerance": 1.0e-5,
        "initial_step_size": 1.0,
    }
    values.update(overrides)
    return newton.solvers.SolverIPC.Config(**values)


def cloth_scaling_benchmark(device) -> list[dict]:
    rows = []
    for resolution in (8, 16, 32):
        model = cloth_model(resolution, device, height=0.04, tilted=False)
        solver = newton.solvers.SolverIPC(model, config=cloth_config())
        state_in, state_out = model.state(), model.state()
        started = time.perf_counter()
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        wp.synchronize()
        capture_ms = (time.perf_counter() - started) * 1.0e3
        timing = time_graph(capture.graph, batches=30, launches_per_batch=50)
        wp.capture_launch(capture.graph)
        rows.append(
            {
                "resolution": resolution,
                "particle_count": model.particle_count,
                "triangle_count": model.tri_count,
                "capture_ms": capture_ms,
                "status": solver.Status(int(solver.diagnostics.status.numpy()[0])).name,
                "newton_iterations": int(
                    solver.diagnostics.newton_iterations.numpy()[0]
                ),
                "line_search_iterations": int(
                    solver.diagnostics.line_search_iterations.numpy()[0]
                ),
                "residual_n": float(solver.diagnostics.residual.numpy()[0]),
                "minimum_gap_m": float(solver.diagnostics.minimum_gap.numpy()[0]),
                **timing,
            }
        )
    return rows


def step_size_sweep(device) -> list[dict]:
    model = cloth_model(8, device)
    loose_solver = newton.solvers.SolverIPC(
        model,
        config=cloth_config(max_newton_iterations=40, absolute_tolerance=0.035),
    )
    state_a, state_b = model.state(), model.state()
    for _ in range(58):
        loose_solver.step(state_a, state_b, None, None, 1.0 / 120.0)
        state_a, state_b = state_b, state_a
    q = state_a.particle_q.numpy().copy()
    qd = state_a.particle_qd.numpy().copy()

    rows = []
    for initial_step_size in (1.0, 2.0, 4.0, 8.0):
        state_in, state_out = model.state(), model.state()
        state_in.particle_q.assign(q)
        state_in.particle_qd.assign(qd)
        solver = newton.solvers.SolverIPC(
            model,
            config=cloth_config(
                max_newton_iterations=100,
                absolute_tolerance=0.01,
                initial_step_size=initial_step_size,
            ),
        )
        started = time.perf_counter()
        solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        wp.synchronize()
        rows.append(
            {
                "initial_step_size": initial_step_size,
                "status": solver.Status(int(solver.diagnostics.status.numpy()[0])).name,
                "newton_iterations": int(
                    solver.diagnostics.newton_iterations.numpy()[0]
                ),
                "line_search_iterations": int(
                    solver.diagnostics.line_search_iterations.numpy()[0]
                ),
                "residual_n": float(solver.diagnostics.residual.numpy()[0]),
                "wall_ms": (time.perf_counter() - started) * 1.0e3,
            }
        )
    return rows


def stability_run(device, *, frame_count: int = 1_000, **config_overrides) -> dict:
    model = cloth_model(16, device)
    solver = newton.solvers.SolverIPC(model, config=cloth_config(**config_overrides))
    state_a, state_b = model.state(), model.state()
    with wp.ScopedCapture(device=device) as capture_ab:
        solver.step(state_a, state_b, None, None, 1.0 / 120.0)
    with wp.ScopedCapture(device=device) as capture_ba:
        solver.step(state_b, state_a, None, None, 1.0 / 120.0)

    minimum_gap = math.inf
    maximum_residual = 0.0
    maximum_newton_iterations = 0
    failures = []
    started = time.perf_counter()
    for frame in range(frame_count):
        for substep, graph in enumerate((capture_ab.graph, capture_ba.graph)):
            wp.capture_launch(graph)
            status = solver.Status(int(solver.diagnostics.status.numpy()[0]))
            minimum_gap = min(
                minimum_gap, float(solver.diagnostics.minimum_gap.numpy()[0])
            )
            maximum_residual = max(
                maximum_residual, float(solver.diagnostics.residual.numpy()[0])
            )
            maximum_newton_iterations = max(
                maximum_newton_iterations,
                int(solver.diagnostics.newton_iterations.numpy()[0]),
            )
            if status != solver.Status.CONVERGED:
                failures.append(
                    {"frame": frame, "substep": substep, "status": status.name}
                )
                break
        if failures:
            break
    wp.synchronize()
    q = state_a.particle_q.numpy()
    qd = state_a.particle_qd.numpy()
    return {
        "requested_frames": frame_count,
        "completed_frames": frame + 1,
        "simulated_steps": 2 * frame + substep + 1,
        "simulated_seconds": (2 * frame + substep + 1) / 120.0,
        "wall_seconds": time.perf_counter() - started,
        "particle_count": model.particle_count,
        "triangle_count": model.tri_count,
        "minimum_gap_m": minimum_gap,
        "maximum_residual_n": maximum_residual,
        "maximum_newton_iterations": maximum_newton_iterations,
        "failure_count": len(failures),
        "failures": failures,
        "finite_positions": bool(np.isfinite(q).all()),
        "finite_velocities": bool(np.isfinite(qd).all()),
        "final_speed_max_m_s": float(np.linalg.norm(qd, axis=1).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("the benchmark requires CUDA conditional-graph support")
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu_name = str(device)

    result = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "newton_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(newton.__file__).resolve().parents[1],
                text=True,
            ).strip(),
            "warp": wp.__version__,
            "device": str(device),
            "gpu": gpu_name,
        },
        "graph_modes": graph_mode_benchmark(device),
        "factorizations": factorization_benchmark(device),
        "reductions": reduction_benchmark(device),
        "cloth_scaling": cloth_scaling_benchmark(device),
        "step_size_sweep": step_size_sweep(device),
        "stability_tuning": [
            {
                "name": "strict_absolute",
                **stability_run(
                    device,
                    frame_count=250,
                    absolute_tolerance=0.005,
                    relative_tolerance=1.0e-5,
                    energy_tolerance=1.0e-8,
                ),
            },
            {"name": "scale_aware", **stability_run(device, frame_count=250)},
        ],
        "stability": stability_run(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
