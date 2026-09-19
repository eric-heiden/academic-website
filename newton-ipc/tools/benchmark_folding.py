"""Compare fold closure against a fixed lower cloth panel on the IPC branch.

The planar lower panel permits a solver-independent crossing oracle. A flap
vertex that moves through its interior certifies a cloth intersection. Positive
vertex gaps alone do not certify all triangle/edge pairs as intersection-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverIPC, SolverVBD, style3d

LENGTH = 0.8
WIDTH = 0.4
BASE_HEIGHT = 0.12
HINGE_START = 0.35
FOLD_RADIUS = 0.04
THICKNESS = 0.008


def make_case(method, nx, stiffness, dt, device):
    """Use identical topology, lumped masses, flat rest shape and folded state."""
    ny = nx // 2
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverIPC.register_custom_attributes(builder)
    builder.default_tri_ke = stiffness / 2.0
    builder.default_tri_ka = 0.0
    builder.add_ground_plane()
    style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, BASE_HEIGHT),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=nx,
        dim_y=ny,
        cell_x=LENGTH / nx,
        cell_y=WIDTH / ny,
        mass=0.3 / ((nx + 1) * (ny + 1)),
        particle_radius=THICKNESS / 2.0,
        tri_aniso_ke=wp.vec3(stiffness, stiffness, stiffness / 2.0),
        tri_ka=0.0,
        tri_kd=0.0,
        edge_aniso_ke=wp.vec3(2.0e-5),
        edge_kd=0.0,
    )
    rest = np.asarray(builder.particle_q, dtype=np.float32)
    pinned = rest[:, 0] <= HINGE_START + 1.0e-6
    for index in np.flatnonzero(pinned):
        builder.particle_mass[index] = 0.0
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    if method == "ipc":
        solver = SolverIPC(
            model,
            config=SolverIPC.Config(
                minimum_separation=THICKNESS / 2.0,
                contact_distance=0.05,
                barrier_stiffness=0.005,
                max_newton_iterations=128,
                max_pcg_iterations=32,
                max_line_search_iterations=24,
                absolute_tolerance=1.0e-2,
                relative_tolerance=2.0e-3,
                energy_tolerance=1.0e-5,
                velocity_damping=1.0,
            ),
        )
        pipeline, contacts = None, None
    else:
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        solver = SolverVBD(
            model,
            iterations=20,
            particle_enable_self_contact=method == "vbd",
            particle_self_contact_margin=THICKNESS,
            particle_self_contact_gap=THICKNESS,
            particle_vertex_contact_buffer_size=128,
            particle_edge_contact_buffer_size=256,
        )
    # The flat rest state stays authoritative for both solvers. Only the
    # dynamic state is folded, preserving topology-based exclusion semantics.
    folded = rest.copy()
    distance = np.maximum(rest[:, 0] - HINGE_START, 0.0)
    angle = np.minimum(distance / FOLD_RADIUS, np.pi)
    beyond = np.maximum(distance - np.pi * FOLD_RADIUS, 0.0)
    movable = ~pinned
    folded[movable, 0] = (
        HINGE_START + FOLD_RADIUS * np.sin(angle[movable]) - beyond[movable]
    )
    folded[movable, 2] += FOLD_RADIUS * (1.0 - np.cos(angle[movable]))
    state_a, state_b = model.state(), model.state()
    for state in (state_a, state_b):
        state.particle_q.assign(folded)
        state.particle_qd.zero_()
        state.clear_forces()
    graphs = []
    for source, target in ((state_a, state_b), (state_b, state_a)):
        with wp.ScopedCapture(device=device) as capture:
            if pipeline is not None:
                pipeline.collide(source, contacts)
            solver.step(source, target, None, contacts, dt)
        graphs.append(capture.graph)
    triangles = model.tri_indices.numpy()
    edges = np.unique(
        np.sort(
            np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, ::2])),
            axis=1,
        ),
        axis=0,
    )
    return model, solver, (state_a, state_b), graphs, rest, folded, pinned, edges


def crossed_panel(previous, current, free):
    """Count exact straight-segment crossings of the fixed panel interior."""
    before, after = previous[:, 2] - BASE_HEIGHT, current[:, 2] - BASE_HEIGHT
    crossing = free & (before > 0.0) & (after <= 0.0)
    ids = np.flatnonzero(crossing)
    if not len(ids):
        return 0
    t = before[ids] / (before[ids] - after[ids])
    impact = previous[ids] + t[:, None] * (current[ids] - previous[ids])
    inside = (
        (impact[:, 0] > 1.0e-5)
        & (impact[:, 0] < HINGE_START - 1.0e-5)
        & (impact[:, 1] > 1.0e-5)
        & (impact[:, 1] < WIDTH - 1.0e-5)
    )
    return int(inside.sum())


def intersected_panel(q, free_triangles):
    """Count free triangles cutting the interior of the fixed planar panel.

    Intersect each triangle with z=BASE_HEIGHT, then clip the resulting line
    segment to the rectangular lower panel. Shared pinned triangles are excluded
    by the caller; tolerances omit tangency and shared boundary contact.
    """
    count = 0
    for tri in q[free_triangles].astype(np.float64):
        z = tri[:, 2] - BASE_HEIGHT
        if z.min() >= -1.0e-7 or z.max() <= 1.0e-7:
            continue
        points = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if z[a] * z[b] < 0.0:
                t = z[a] / (z[a] - z[b])
                points.append(tri[a, :2] + t * (tri[b, :2] - tri[a, :2]))
            elif abs(z[a]) < 1.0e-12:
                points.append(tri[a, :2])
        if len(points) < 2:
            continue
        p, end = points[0], points[1]
        direction = end - p
        t_low, t_high = 0.0, 1.0
        for axis, upper in ((0, HINGE_START), (1, WIDTH)):
            if abs(direction[axis]) < 1.0e-12:
                if not 1.0e-5 < p[axis] < upper - 1.0e-5:
                    t_high = -1.0
                    break
            else:
                bounds = sorted(
                    (
                        (1.0e-5 - p[axis]) / direction[axis],
                        (upper - 1.0e-5 - p[axis]) / direction[axis],
                    )
                )
                t_low, t_high = max(t_low, bounds[0]), min(t_high, bounds[1])
        count += int(t_low < t_high)
    return count


def run_case(method, nx, stiffness, dt, duration, repeat, device, output):
    model, solver, states, graphs, rest, q, pinned, edges = make_case(
        method, nx, stiffness, dt, device
    )
    initial = q.copy()
    triangles = model.tri_indices.numpy()
    free_triangles = triangles[~pinned[triangles].any(axis=1)]
    edge_lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    trajectory, samples, times = [q.copy()], [], []
    failures, crossings, overflows = [], 0, 0
    first_crossing = None
    first_intersection = None
    frame_stride = max(1, round(1.0 / (60.0 * dt)))
    steps = round(duration / dt)
    free = ~pinned
    for step in range(steps):
        started = time.perf_counter()
        wp.capture_launch(graphs[step % 2])
        wp.synchronize_device(device)
        times.append((time.perf_counter() - started) * 1.0e3)
        state = states[1 - step % 2]
        previous, q = q, state.particle_q.numpy()
        qd = state.particle_qd.numpy()
        finite = bool(np.isfinite(q).all() and np.isfinite(qd).all())
        if not finite:
            failures.append({"step": step, "reason": "nonfinite"})
            break
        # A zero-mass lower panel must not drift; otherwise this oracle would
        # no longer describe the actual contact surface.
        if not np.array_equal(q[pinned], initial[pinned]):
            raise AssertionError("fixed lower cloth panel moved")
        count = crossed_panel(previous, q, free)
        crossings += count
        if count and first_crossing is None:
            first_crossing = (step + 1) * dt
        intersections = intersected_panel(q, free_triangles)
        if intersections and first_intersection is None:
            first_intersection = (step + 1) * dt
        inside = (
            free
            & (q[:, 0] > 1.0e-5)
            & (q[:, 0] < HINGE_START - 1.0e-5)
            & (q[:, 1] > 1.0e-5)
            & (q[:, 1] < WIDTH - 1.0e-5)
        )
        gap = float(np.min(q[inside, 2] - BASE_HEIGHT)) if inside.any() else None
        status, residual = "not_exposed", None
        if method == "ipc":
            status = solver.Status(int(solver.diagnostics.status.numpy()[0])).name
            residual = float(solver.diagnostics.residual.numpy()[0])
            if status != "CONVERGED":
                failures.append(
                    {"step": step, "reason": status, "residual_n": residual}
                )
        elif method == "vbd":
            info = solver.trimesh_collision_detector.collision_info
            for kind in ("vertex_colliding_triangles", "edge_colliding_edges"):
                counts = getattr(info, kind + "_count").numpy()
                capacity = getattr(info, kind + "_buffer_sizes").numpy()
                overflows += int((counts > capacity).sum())
        strain = np.abs(
            np.linalg.norm(q[edges[:, 1]] - q[edges[:, 0]], axis=1) / edge_lengths - 1.0
        )
        samples.append(
            {
                "time_s": (step + 1) * dt,
                "minimum_layer_gap_m": gap,
                "ground_gap_m": float(q[:, 2].min() - THICKNESS / 2.0),
                "vertices_below_panel": int(
                    (q[inside, 2] < BASE_HEIGHT - 1.0e-6).sum()
                ),
                "triangles_cutting_panel": intersections,
                "speed_max_m_s": float(np.linalg.norm(qd[free], axis=1).max()),
                "edge_strain_max": float(strain.max()),
                "status": status,
                "residual_n": residual,
            }
        )
        if (step + 1) % frame_stride == 0:
            trajectory.append(q.copy())
        # Stop at solver failure; time has not advanced on an IPC rollback.
        if failures:
            break
    run_id = f"{method}-n{nx}-k{stiffness:g}-hz{round(1 / dt)}-r{repeat}"
    if repeat == 0:
        np.savez_compressed(
            output / f"{run_id}.npz",
            positions=np.asarray(trajectory),
            triangles=model.tri_indices.numpy(),
            rest=rest,
            pinned=pinned,
            frame_dt=frame_stride * dt,
        )
    gaps = [
        sample["minimum_layer_gap_m"]
        for sample in samples
        if sample["minimum_layer_gap_m"] is not None
    ]
    row = {
        "id": run_id,
        "method": method,
        "resolution": [nx, nx // 2],
        "stiffness_n_m": stiffness,
        "dt_s": dt,
        "repeat": repeat,
        "particles": model.particle_count,
        "triangles": model.tri_count,
        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "requested_steps": steps,
        "completed_steps": sum(
            s["status"] in ("CONVERGED", "not_exposed") for s in samples
        ),
        "failed_steps": len(failures),
        "failures": failures,
        "first_crossing_s": first_crossing,
        "crossing_events": crossings,
        "first_surface_intersection_s": first_intersection,
        "max_triangles_cutting_panel": max(
            (s["triangles_cutting_panel"] for s in samples), default=0
        ),
        "minimum_layer_gap_m": min(gaps) if gaps else None,
        "maximum_edge_strain": max(
            (s["edge_strain_max"] for s in samples), default=None
        ),
        "maximum_speed_m_s": max((s["speed_max_m_s"] for s in samples), default=None),
        "minimum_ground_gap_m": min((s["ground_gap_m"] for s in samples), default=None),
        "contact_overflow_rows": overflows,
        "median_step_ms": float(np.median(times[1:])) if len(times) > 1 else None,
        "p95_step_ms": float(np.percentile(times[1:], 95)) if len(times) > 1 else None,
        "samples": samples,
    }
    print(
        json.dumps({key: value for key, value in row.items() if key != "samples"}),
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--hz", type=int, nargs="+", default=[60, 120, 240])
    parser.add_argument("--stiffness", type=float, nargs="+", default=[500.0])
    parser.add_argument("--methods", nargs="+", default=["ipc", "vbd", "vbd_no_self"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    wp.init()
    device = wp.get_device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "environment": {
            "newton_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(newton.__file__).resolve().parents[1],
                text=True,
            ).strip(),
            "warp": wp.__version__,
            "python": platform.python_version(),
            "device": device.name,
        },
        "scene": {
            "length_m": LENGTH,
            "width_m": WIDTH,
            "base_height_m": BASE_HEIGHT,
            "hinge_start_m": HINGE_START,
            "fold_radius_m": FOLD_RADIUS,
            "interaction_distance_m": THICKNESS,
            "total_mass_before_pinning_kg": 0.3,
            "ipc_budgets": [128, 32, 24],
            "vbd_iterations": 20,
            "material_note": "Equal small-strain membrane tangent (lambda=0, mu=k/2); finite-strain and bending energies differ.",
            "timing_note": "Captured one-step wall latency plus device synchronization; excludes host diagnostics; first step omitted.",
            "oracle_note": "Linear vertex trajectory crossings and float64 triangle-plane cuts clipped to the fixed panel interior, checked every substep. Excludes shared pinned triangles, coplanar/tangent contact, and a 1e-5 m boundary margin; triangle cut z tolerance 1e-7 m. Not a general mesh CCD certificate.",
        },
        "runs": [],
    }
    for nx in args.resolution:
        for stiffness in args.stiffness:
            for hz in args.hz:
                for repeat in range(args.repeats):
                    for method in args.methods:
                        result["runs"].append(
                            run_case(
                                method,
                                nx,
                                stiffness,
                                1.0 / hz,
                                args.duration,
                                repeat,
                                device,
                                args.output,
                            )
                        )
                        (args.output / "results.json").write_text(
                            json.dumps(result, indent=2, allow_nan=False) + "\n"
                        )


if __name__ == "__main__":
    main()
