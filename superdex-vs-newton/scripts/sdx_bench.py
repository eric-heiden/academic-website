"""Benchmark harness for SuperDex Physics 1.0.0 (headless)."""

import json
import math
import statistics
import sys
import time

import numpy as np
import superdex.physics as physics
from superdex.physics.paths import resolve_asset, resolve_asset_root

RESULTS = []


def timeit(scene, dt, n_warm, n_steps):
    for _ in range(n_warm):
        scene.step(dt)
    ts = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        scene.step(dt)
        ts.append(time.perf_counter() - t0)
    ts_sorted = sorted(ts)
    return {
        "mean_ms": 1e3 * statistics.mean(ts),
        "median_ms": 1e3 * statistics.median(ts),
        "p95_ms": 1e3 * ts_sorted[int(0.95 * len(ts_sorted)) - 1],
        "min_ms": 1e3 * min(ts),
        "steps_per_s": 1.0 / statistics.mean(ts),
        "rt_factor": dt / statistics.mean(ts),
    }


def cube_shape(size):
    h = size / 2
    v = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ], dtype=np.float32).flatten()
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    ], dtype=np.int32).flatten()
    return physics.create_tri_mesh_shape(coordinates=v, connectivity=f)


# ---------------------------------------------------------------- rigid pile
def bench_rigid_pile(n_bodies, dt=1 / 60, steps=200):
    scene = physics.create_scene(f"pile{n_bodies}")
    scene.set_gravity([0, -9.8, 0])
    plane = physics.create_plane_shape(normal=[0, 1, 0], distance=0.0)
    scene.create_rigid_actor(name="ground", shape=plane, is_static=True)
    cs = cube_shape(0.1)
    side = int(math.ceil(n_bodies ** (1 / 3)))
    k = 0
    for i in range(side):
        for j in range(side):
            for l in range(side):
                if k >= n_bodies:
                    break
                scene.create_rigid_actor(
                    name=f"c{k}", shape=cs, density=1000.0,
                    world_from_local=physics.TransformRT(
                        translation=[i * 0.115, 0.06 + l * 0.115, j * 0.115]),
                    collider_type=physics.ColliderType.BOX)
                k += 1
    r = timeit(scene, dt, 20, steps)
    r.update(scene="rigid_pile", n_bodies=n_bodies, dt=dt,
             n_actors=scene.get_num_actors())
    st = scene.get_solver_stats()
    r["nl_iters"] = st.max_non_linear_iters
    r["converged"] = str(st.convergence_status)
    physics.destroy_scene(scene)
    return r


# ---------------------------------------------------------------- soft body
def bench_soft(asset, dt=1 / 60, steps=100, n_copies=1):
    scene = physics.create_scene(f"soft-{asset}-{n_copies}")
    scene.set_gravity([0, -9.8, 0])
    plane = physics.create_plane_shape(normal=[0, 1, 0], distance=-0.5)
    scene.create_rigid_actor(name="ground", shape=plane, is_static=True)
    shp = physics.load_shape_from_file(file_path=str(resolve_asset(asset)))
    for i in range(n_copies):
        scene.create_soft_actor(
            name=f"duck{i}", shape=shp,
            world_from_local=physics.TransformRT(
                translation=[-0.5 + 0.6 * (i % 4), 0.5 + 0.6 * (i // 4), -1.0]))
    r = timeit(scene, dt, 10, steps)
    r.update(scene="soft", asset=asset, n_copies=n_copies, dt=dt)
    st = scene.get_solver_stats()
    r["nl_iters"] = st.max_non_linear_iters
    r["converged"] = str(st.convergence_status)
    physics.destroy_scene(scene)
    return r


# ---------------------------------------------------------------- allegro hand
def bench_allegro(dt=1 / 60, steps=100, with_cube=True):
    scene = physics.create_scene("allegro")
    scene.set_gravity([0, -9.8, 0])
    p = str(resolve_asset("allegro/allegro.mochi_prefab"))
    physics.prefab.add_to_scene(
        prefab_path=p, root_path=str(resolve_asset_root(p)), scene=scene,
        params=physics.prefab.PrefabParams(name="hand"))
    if with_cube:
        cs = cube_shape(0.05)
        scene.create_rigid_actor(
            name="cube", shape=cs, density=500.0,
            world_from_local=physics.TransformRT(translation=[0, 0.1, 0]),
            collider_type=physics.ColliderType.BOX)
    plane = physics.create_plane_shape(normal=[0, 1, 0], distance=-0.5)
    scene.create_rigid_actor(name="ground", shape=plane, is_static=True)
    r = timeit(scene, dt, 10, steps)
    r.update(scene="allegro", with_cube=with_cube, dt=dt,
             n_actors=scene.get_num_actors())
    physics.destroy_scene(scene)
    return r


# ---------------------------------------------------------------- timestep sweep
def bench_dt_sweep(dts, steps=60):
    out = []
    for dt in dts:
        try:
            r = bench_rigid_pile(27, dt=dt, steps=steps)
            out.append(r)
        except Exception as e:  # noqa
            out.append({"scene": "rigid_pile", "dt": dt, "error": str(e)})
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    nthreads = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    physics.initialize(num_worker_threads=nthreads)
    print(f"# threads={physics.get_num_threads()} "
          f"single={physics.is_single_threaded()} "
          f"precision={physics.PRECISION_NAME}", file=sys.stderr)

    res = []
    if mode in ("all", "rigid"):
        for n in [1, 8, 27, 64, 125, 216]:
            r = bench_rigid_pile(n)
            r["threads"] = nthreads
            res.append(r)
            print(json.dumps(r), flush=True)
    if mode in ("all", "soft"):
        for asset, copies in [("duck/duck_coarse.mochi.h5", 1),
                              ("duck/duck_730.mochi.h5", 1),
                              ("duck/duck_1899.mochi.h5", 1),
                              ("duck/duck_1899.mochi.h5", 4),
                              ("duck/duck_1899.mochi.h5", 8)]:
            try:
                r = bench_soft(asset, n_copies=copies)
                r["threads"] = nthreads
                res.append(r)
                print(json.dumps(r), flush=True)
            except Exception as e:
                print(json.dumps({"scene": "soft", "asset": asset,
                                  "n_copies": copies, "error": str(e)}), flush=True)
    if mode in ("all", "allegro"):
        for wc in [False, True]:
            try:
                r = bench_allegro(with_cube=wc)
                r["threads"] = nthreads
                res.append(r)
                print(json.dumps(r), flush=True)
            except Exception as e:
                print(json.dumps({"scene": "allegro", "error": str(e)}), flush=True)
    if mode in ("all", "dt"):
        for r in bench_dt_sweep([1 / 1000, 1 / 500, 1 / 240, 1 / 120, 1 / 60,
                                 1 / 30, 1 / 20, 1 / 10]):
            r["threads"] = nthreads
            res.append(r)
            print(json.dumps(r), flush=True)

    physics.shutdown()


if __name__ == "__main__":
    main()
