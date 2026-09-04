"""SuperDex accuracy / stability probes: penetration, timestep robustness, determinism."""
import json, math, sys
import numpy as np
import superdex.physics as physics

physics.initialize(num_worker_threads=0)


def cube_shape(size):
    h = size / 2
    v = np.array([[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
                  [-h,-h,h],[h,-h,h],[h,h,h],[-h,h,h]], dtype=np.float32).flatten()
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,4,7],[0,7,3],
                  [1,2,6],[1,6,5],[0,1,5],[0,5,4],[2,3,7],[2,7,6]], dtype=np.int32).flatten()
    return physics.create_tri_mesh_shape(coordinates=v, connectivity=f)


def drop(dt, density=1000.0, penalty=None, h0=1.0, sim_time=3.0, size=0.1,
         nl_iter=None):
    sc = physics.create_scene("d")
    sc.set_gravity([0, -9.8, 0])
    if penalty is not None or nl_iter is not None:
        p = sc.get_solver_params()
        if nl_iter is not None:
            p.non_linear_solver.max_iter = nl_iter
        sc.set_solver_params(p)
    sc.create_rigid_actor(name="g",
        shape=physics.create_plane_shape(normal=[0,1,0], distance=0.0), is_static=True)
    a = sc.create_rigid_actor(name="c", shape=cube_shape(size), density=density,
        world_from_local=physics.TransformRT(translation=[0, h0, 0]),
        collider_type=physics.ColliderType.BOX)
    if penalty is not None:
        cp = a.get_contact_params()
        cp.penalty_coefficient = penalty
        a.set_contact_params(cp)
    n = int(round(sim_time / dt))
    diverged = False
    for _ in range(n):
        sc.step(dt)
        y = a.get_root_transform().translation[1]
        if not math.isfinite(y) or abs(y) > 100:
            diverged = True
            break
    T = a.get_root_transform()
    y = T.translation[1]
    v = a.get_linear_velocity()
    st = sc.get_solver_stats()
    cp = a.get_contact_params()
    out = {"dt": dt, "density": density,
           "penalty": cp.penalty_coefficient,
           "nl_iter": sc.get_solver_params().non_linear_solver.max_iter,
           "final_y": float(y), "ideal_y": size/2,
           "penetration_mm": 1e3 * (size/2 - float(y)) if math.isfinite(y) else None,
           "speed": float(np.linalg.norm([v[0], v[1], v[2]])),
           "diverged": diverged, "status": str(st.convergence_status)}
    physics.destroy_scene(sc)
    return out


mode = sys.argv[1]
if mode == "dt":
    for dt in [1/1000, 1/500, 1/240, 1/120, 1/60, 1/30, 1/20, 1/10, 1/5, 1/2]:
        print(json.dumps({"probe": "dt", **drop(dt)}), flush=True)
elif mode == "penalty":
    for pen in [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]:
        print(json.dumps({"probe": "penalty", **drop(1/60, penalty=pen)}), flush=True)
elif mode == "mass":
    for d in [10, 100, 1000, 5000, 20000]:
        print(json.dumps({"probe": "mass", **drop(1/60, density=d)}), flush=True)
elif mode == "iters":
    for it in [1, 2, 4, 8, 16]:
        print(json.dumps({"probe": "iters", **drop(1/60, nl_iter=it)}), flush=True)
elif mode == "determinism":
    ys = [drop(1/60)["final_y"] for _ in range(5)]
    print(json.dumps({"probe": "determinism", "ys": ys,
                      "bit_identical": len(set(ys)) == 1}), flush=True)
physics.shutdown()
