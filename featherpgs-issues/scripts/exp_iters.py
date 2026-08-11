"""Grasp quality and cost against the number of solver iterations."""

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


def timed(env, n=200):
    with wp.ScopedCapture() as cap:
        env._simulate()
    g = cap.graph
    for _ in range(20):
        wp.capture_launch(g)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        wp.capture_launch(g)
    wp.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    out = {}
    for it in (4, 8, 12, 20, 32, 48):
        kw = {"pgs_iterations": it}
        env = make_env("fpgs", solver_kw=kw)
        r = run_gc(env, 700)
        s = summarize(r)
        np.savez_compressed(f"data/iters{it}.npz",
                            **{k: v for k, v in r.items() if k != "failure"},
                            failure=np.asarray(r["failure"]))
        del env
        env = make_env("fpgs", solver_kw=kw, capture_sim_graph=False)
        s["ms_per_frame"] = timed(env)
        s["physics_rtf"] = (1000.0 / 60.0) / s["ms_per_frame"]
        s["pgs_iterations"] = it
        del env
        out[f"it{it}"] = s
        print(f"it{it}", json.dumps(s, default=str)[:300], flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
