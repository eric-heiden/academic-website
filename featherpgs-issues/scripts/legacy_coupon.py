"""The earlier report's minimal static grasp, re-run and instrumented.

The scene is byte-for-byte the one that report used: a cube pinched between two
pads that are welded to the world with a 0.5 mm geometric overlap, shaken by an
oscillating body force along y while gravity pulls along -z. Only the recording
is added, so the sliding motion can be plotted rather than reduced to a single
end-of-run number.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton

CUBE_HALF = 0.02
START_Z = 0.065
OVERLAP = 5.0e-4
PAD_HALF_X = 0.005
PAD_HALF_YZ = 0.06
AMPLITUDE = 0.03
FREQUENCY = 2.0


@wp.kernel
def apply_shake(body_f: wp.array[wp.spatial_vector], body: int, force_y: float):
    body_f[body] = wp.spatial_vector(wp.vec3(0.0, force_y, 0.0), wp.vec3(0.0))


def cube_mesh(half: float) -> newton.Mesh:
    vertices = np.array(
        [
            [-half, -half, -half], [half, -half, -half],
            [half, half, -half], [-half, half, -half],
            [-half, -half, half], [half, -half, half],
            [half, half, half], [-half, half, half],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
            0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices, indices)


def build(geom="mesh", overlap=OVERLAP, mu=1.0):
    pad_x = CUBE_HALF + PAD_HALF_X - 0.5 * overlap
    cfg = newton.ModelBuilder.ShapeConfig(
        density=1000.0, mu=mu, ke=2.5e5, kd=1.0e3, restitution=0.0
    )
    builder = newton.ModelBuilder()
    cube = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, START_Z), wp.quat_identity())
    )
    if geom == "mesh":
        builder.add_shape_mesh(cube, mesh=cube_mesh(CUBE_HALF), cfg=cfg)
    else:
        builder.add_shape_box(cube, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF, cfg=cfg)
    for x in (-pad_x, pad_x):
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(x, 0.0, START_Z), wp.quat_identity()),
            hx=PAD_HALF_X,
            hy=PAD_HALF_YZ,
            hz=PAD_HALF_YZ,
            cfg=cfg,
        )
    return builder, cube


def make(geom="mesh", solver="fpgs", solver_kw=None, overlap=OVERLAP, mu=1.0):
    builder, cube = build(geom, overlap, mu)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    if solver == "fpgs":
        cfg = dict(pgs_mode="matrix_free", pgs_iterations=12, pgs_beta=0.2,
                   pgs_cfm=1.0e-6, double_buffer=False)
        cfg.update(solver_kw or {})
        solv = newton.solvers.SolverFeatherPGS(model, **cfg)
        control._use_coord_layout_targets = False
    else:
        cfg = dict(solver="newton", integrator="implicitfast", iterations=15,
                   ls_iterations=50, nconmax=4096, njmax=8192,
                   cone="elliptic", impratio=50.0, use_mujoco_contacts=False)
        cfg.update(solver_kw or {})
        solv = newton.solvers.SolverMuJoCo(model, **cfg)
    return dict(model=model, pipeline=pipeline, contacts=contacts, state=[s0, s1],
                control=control, solver=solv, cube=cube, cfg=cfg)


def run(env, frames=600, substeps=16, record_poses=False):
    model, pipeline, contacts = env["model"], env["pipeline"], env["contacts"]
    state, control, solver = env["state"], env["control"], env["solver"]
    cube = env["cube"]
    dt = 1.0 / (60.0 * substeps)
    mass = float(model.body_mass.numpy()[cube])
    omega = 2.0 * math.pi * FREQUENCY

    z, y, x, nc, t = [], [], [], [], []
    poses = []
    first_lost = None
    failure = ""
    try:
        for frame in range(frames):
            for k in range(substeps):
                tt = (frame * substeps + k) * dt
                state[0].clear_forces()
                fy = -mass * AMPLITUDE * omega * omega * math.sin(omega * tt)
                wp.launch(apply_shake, dim=1, inputs=[state[0].body_f, cube, fy])
                pipeline.collide(state[0], contacts)
                solver.step(state[0], state[1], control, contacts, dt)
                state[0], state[1] = state[1], state[0]

            pipeline.collide(state[0], contacts)
            n = int(contacts.rigid_contact_count.numpy()[0])
            if first_lost is None and n == 0:
                first_lost = frame
            bq = state[0].body_q.numpy()
            t.append(frame / 60.0)
            x.append(float(bq[cube, 0]))
            y.append(float(bq[cube, 1]))
            z.append(float(bq[cube, 2]))
            nc.append(n)
            if record_poses:
                poses.append(bq.copy())
            if not np.isfinite(bq[cube, 2]):
                failure = f"non-finite pose at frame {frame}"
                break
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"

    z = np.asarray(z)
    out = {"t": np.asarray(t), "x": np.asarray(x), "y": np.asarray(y), "z": z,
           "n_contacts": np.asarray(nc), "failure": failure,
           "first_lost_contact_frame": first_lost}
    if record_poses:
        out["poses"] = np.stack(poses) if poses else np.zeros((0, 1, 7))
    if len(z):
        out["final_slip_mm"] = float((START_Z - z[-1]) * 1000.0)
        out["worst_downward_mm"] = float((START_Z - z.min()) * 1000.0)
        out["worst_upward_mm"] = float((z.max() - START_Z) * 1000.0)
        # phase-independent: the furthest it ever gets from where it started
        out["worst_excursion_mm"] = float(np.abs(z - START_Z).max() * 1000.0)
        held = bool(np.isfinite(z[-1]) and abs(z[-1] - START_Z) < 0.01)
        out["held"] = held
        # slide rate over the second half of the run, while still held
        half = len(z) // 2
        if held and len(z) > half + 10:
            out["slide_rate_mm_per_s"] = float(
                (z[half] - z[-1]) * 1000.0 / (out["t"][-1] - out["t"][half]))
    return out
