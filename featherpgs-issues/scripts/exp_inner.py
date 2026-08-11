"""Frozen-basis inner substeps: accuracy and speed at a large solver step.

An unmerged FeatherPGS branch adds `pgs_inner_substeps`, which runs several
small position substeps per solver step while reusing the mass-matrix
factorisation and the contact Jacobians. That is the cheapest way to buy back
accuracy at a large outer step, so it is measured against plain substepping.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/horde/repos/fpgs-frozen")
sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402


def timed(env, n=200):
    try:
        with wp.ScopedCapture() as cap:
            env._simulate()
        g = cap.graph
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"[:150]
    for _ in range(20):
        wp.capture_launch(g)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        wp.capture_launch(g)
    wp.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0, "captured"


CASES = [
    ("outer2_inner1", 2, 1),
    ("outer2_inner4", 2, 4),
    ("outer2_inner8", 2, 8),
    ("outer1_inner1", 1, 1),
    ("outer1_inner4", 1, 4),
    ("outer1_inner8", 1, 8),
    ("outer1_inner16", 1, 16),
    ("outer4_inner4", 4, 4),
]

if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    print("newton from", newton.__file__, flush=True)
    out = {}
    for tag, outer, inner in CASES:
        kw = {"double_buffer": False}
        if inner > 1:
            kw["pgs_inner_substeps"] = inner
        try:
            env = make_env("fpgs", substeps=outer, solver_kw=kw)
            r = run_gc(env, 900)
            s = summarize(r)
            del env
            env = make_env("fpgs", substeps=outer, solver_kw=kw, capture_sim_graph=False)
            ms, note = timed(env)
            s.update(ms_per_frame=ms, graph=note, outer=outer, inner=inner,
                     effective_dt_ms=1000.0 / 60.0 / (outer * inner))
            s["physics_rtf"] = (1000.0 / 60.0) / ms if ms else None
            del env
        except Exception as exc:  # noqa: BLE001
            s = {"error": f"{type(exc).__name__}: {exc}"[:220], "outer": outer, "inner": inner}
        out[tag] = s
        print(tag, json.dumps(s, default=str), flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
