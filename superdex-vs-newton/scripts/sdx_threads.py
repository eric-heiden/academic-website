"""SuperDex thread-scaling + accuracy probes."""
import json, statistics, sys, time
import numpy as np
import superdex.physics as physics
from superdex.physics.paths import resolve_asset

NT = int(sys.argv[1])
physics.initialize(num_worker_threads=NT)


def timeit(scene, dt, warm, n):
    for _ in range(warm):
        scene.step(dt)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); scene.step(dt); ts.append(time.perf_counter() - t0)
    return statistics.mean(ts), statistics.median(ts)


def soft(asset, copies, dt=1/60, warm=10, n=60):
    sc = physics.create_scene(f"s{copies}")
    sc.set_gravity([0, -9.8, 0])
    sc.create_rigid_actor(name="g",
        shape=physics.create_plane_shape(normal=[0,1,0], distance=-0.5), is_static=True)
    shp = physics.load_shape_from_file(file_path=str(resolve_asset(asset)))
    for i in range(copies):
        sc.create_soft_actor(name=f"d{i}", shape=shp,
            world_from_local=physics.TransformRT(
                translation=[-0.5+0.6*(i%4), 0.5+0.6*(i//4), -1.0]))
    m, md = timeit(sc, dt, warm, n)
    physics.destroy_scene(sc)
    return {"probe":"soft_threads","asset":asset,"copies":copies,"threads":NT,
            "mean_ms":1e3*m,"median_ms":1e3*md,"steps_per_s":1/m,"rt_factor":dt/m}


for copies in [1, 4, 8]:
    print(json.dumps(soft("duck/duck_1899.mochi.h5", copies)), flush=True)
physics.shutdown()
