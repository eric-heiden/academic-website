"""How large a time step can each solver take and still hold the cube, and how
fast does it run there?

The controller runs at 60 Hz in every case; only the number of physics substeps
per control frame changes. Speed is measured separately with a captured CUDA
graph so the number reflects the solver, not Python overhead.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402
from newtontests.franka_cube_shake import Phase  # noqa: E402

SUBSTEPS = [16, 8, 4, 2, 1]


def timed_frames(env, n=200):
    """Wall time per control frame with the physics loop in a CUDA graph."""
    try:
        with wp.ScopedCapture() as cap:
            env._simulate()
        graph = cap.graph
    except Exception as exc:  # noqa: BLE001
        return {"graph": f"{type(exc).__name__}: {exc}"[:160], "ms_per_frame": None}
    for _ in range(20):
        wp.capture_launch(graph)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        wp.capture_launch(graph)
    wp.synchronize()
    dt = (time.perf_counter() - t0) / n
    return {"graph": "captured", "ms_per_frame": dt * 1000.0,
            "physics_rtf": (1.0 / 60.0) / dt}


def main():
    wp.init()
    wp.set_device("cuda:0")
    results = {}
    for solver in (["mujoco"] if len(sys.argv)>1 and sys.argv[1]=="mujoco" else ["fpgs","mujoco"]):
        for ss in SUBSTEPS:
            tag = f"{solver}_ss{ss}"
            kw = {}
            if solver == "fpgs":
                kw = {"solver_kw": {"double_buffer": False}}
            try:
                env = make_env(solver, substeps=ss, **kw)
            except Exception as exc:  # noqa: BLE001
                results[tag] = {"failure": f"construct: {exc}"[:200]}
                print(tag, results[tag], flush=True)
                continue
            res = run_gc(env, 900, gc=(solver == "fpgs"))
            s = summarize(res)
            s["substeps"] = ss
            s["dt_ms"] = 1000.0 / 60.0 / ss
            s["max_phase"] = Phase(int(res["phase"].max())).name if len(res["phase"]) else "?"
            results[tag] = s
            print(tag, json.dumps(s), flush=True)
            np.savez_compressed(f"data/{tag}.npz",
                                **{k: v for k, v in res.items() if k != "failure"},
                                failure=np.asarray(res["failure"]))
            del env

            # Fresh env for the timing measurement so state is comparable.
            env = make_env(solver, substeps=ss, capture_sim_graph=False, **kw)
            results[tag].update(timed_frames(env))
            print(tag, "timing", results[tag].get("ms_per_frame"), flush=True)
            del env
    print("===JSON===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
