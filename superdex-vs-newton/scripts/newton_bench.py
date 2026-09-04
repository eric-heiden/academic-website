"""Matched Newton benchmark: N rigid cubes settling on a ground plane.

Mirrors /tmp/sdx_bench.py's rigid_pile scene so the two engines can be compared
on the same geometry, mass and time step.
"""

import json
import math
import statistics
import sys
import time

import warp as wp

import newton


def build(n_bodies, n_envs=1, up_axis="Z"):
    b = newton.ModelBuilder(up_axis=up_axis)
    side = int(math.ceil(n_bodies ** (1 / 3)))
    env = newton.ModelBuilder(up_axis=up_axis)
    k = 0
    for i in range(side):
        for j in range(side):
            for l in range(side):
                if k >= n_bodies:
                    break
                body = env.add_body(
                    xform=wp.transform(
                        wp.vec3(i * 0.115, j * 0.115, 0.06 + l * 0.115),
                        wp.quat_identity()),
                    label=f"c{k}")
                env.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05,
                                  cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))
                env.add_joint_free(body)
                k += 1
    for e in range(n_envs):
        b.add_builder(env, xform=wp.transform(wp.vec3(e * 4.0, 0.0, 0.0),
                                              wp.quat_identity()))
    b.add_ground_plane()
    return b.finalize()


def run(solver_name, n_bodies, n_envs, dt=1 / 60, substeps=1, steps=200, warm=30,
        graph=True):
    model = build(n_bodies, n_envs)
    if solver_name == "mujoco":
        solver = newton.solvers.SolverMuJoCo(model)
    elif solver_name == "xpbd":
        solver = newton.solvers.SolverXPBD(model, iterations=10)
    elif solver_name == "kamino":
        solver = newton.solvers.SolverKamino(model)
    elif solver_name == "featherstone":
        solver = newton.solvers.SolverFeatherstone(model)
    elif solver_name == "semi_implicit":
        solver = newton.solvers.SolverSemiImplicit(model)
    else:
        raise ValueError(solver_name)

    state0, state1 = model.state(), model.state()
    control = model.control()
    contacts = None
    collide = getattr(solver, "requires_contacts", True)

    def one_step():
        nonlocal state0, state1, contacts
        for _ in range(substeps):
            if collide:
                contacts = model.collide(state0)
            state0.clear_forces()
            solver.step(state0, state1, control, contacts, dt / substeps)
            state0, state1 = state1, state0

    # warmup (also triggers kernel compilation)
    for _ in range(warm):
        one_step()
    wp.synchronize()

    cg = None
    if graph and wp.get_device().is_cuda:
        try:
            with wp.ScopedCapture() as cap:
                one_step()
            cg = cap.graph
        except Exception as e:
            print(f"# graph capture failed for {solver_name}: {e}", file=sys.stderr)

    ts = []
    for _ in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        if cg is not None:
            wp.capture_launch(cg)
        else:
            one_step()
        wp.synchronize()
        ts.append(time.perf_counter() - t0)

    mean = statistics.mean(ts)
    tot = n_bodies * n_envs
    return {
        "engine": "newton",
        "solver": solver_name,
        "n_bodies": n_bodies,
        "n_envs": n_envs,
        "total_bodies": tot,
        "dt": dt,
        "substeps": substeps,
        "graph": cg is not None,
        "mean_ms": 1e3 * mean,
        "median_ms": 1e3 * statistics.median(ts),
        "min_ms": 1e3 * min(ts),
        "steps_per_s": 1.0 / mean,
        "env_steps_per_s": n_envs / mean,
        "rt_factor": dt / mean,
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    wp.init()
    if mode == "single":
        for solver in ["mujoco", "xpbd", "kamino", "featherstone", "semi_implicit"]:
            for n in [1, 8, 27, 64, 125, 216]:
                try:
                    print(json.dumps(run(solver, n, 1)), flush=True)
                except Exception as e:
                    print(json.dumps({"solver": solver, "n_bodies": n,
                                      "error": repr(e)[:400]}), flush=True)
    elif mode == "scale":
        for solver in ["xpbd", "mujoco"]:
            for envs in [1, 16, 64, 256, 1024, 4096]:
                try:
                    print(json.dumps(run(solver, 8, envs, steps=50, warm=15)), flush=True)
                except Exception as e:
                    print(json.dumps({"solver": solver, "n_envs": envs,
                                      "error": repr(e)[:400]}), flush=True)
