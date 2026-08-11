"""Timing only, for the iteration sweep, measured with nothing else on the GPU."""

from __future__ import annotations

import json
import sys
import time

import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env  # noqa: E402


def timed(env, n=300):
    with wp.ScopedCapture() as cap:
        env._simulate()
    g = cap.graph
    for _ in range(30):
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
        env = make_env("fpgs", solver_kw={"pgs_iterations": it},
                       capture_sim_graph=False)
        ms = timed(env)
        out[f"it{it}"] = {"pgs_iterations": it, "ms_per_frame": ms,
                          "physics_rtf": (1000.0 / 60.0) / ms}
        print(f"it{it}", json.dumps(out[f"it{it}"]), flush=True)
        del env
    # re-measure the substep sweep the same way for a like-for-like comparison
    for ss in (16, 8, 4, 2, 1):
        env = make_env("fpgs", substeps=ss, capture_sim_graph=False)
        ms = timed(env)
        out[f"ss{ss}"] = {"substeps": ss, "ms_per_frame": ms,
                          "physics_rtf": (1000.0 / 60.0) / ms}
        print(f"ss{ss}", json.dumps(out[f"ss{ss}"]), flush=True)
        del env
    print("===JSON===")
    print(json.dumps(out, indent=2))
