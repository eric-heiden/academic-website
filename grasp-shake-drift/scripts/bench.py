# SPDX-License-Identifier: Apache-2.0
"""Time the shake test per frame for a set of variants.

Checks whether the recommended fix costs anything at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull

from newtontests.franka_cube_shake import create_parser
from newtontests.experiments import Variant, VARIANTS


def bench(variant, warmup=240, measure=600):
    argv_backup = sys.argv
    sys.argv = ["bench"]
    args = newton.examples.default_args(create_parser())
    sys.argv = argv_backup

    viewer = ViewerNull(num_frames=warmup + measure)
    ex = Variant(viewer, args, VARIANTS[variant])

    for _ in range(warmup):
        ex.step()
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(measure):
        ex.step()
    wp.synchronize()
    dt = time.perf_counter() - t0

    ms = dt / measure * 1000.0
    print(f"[bench] {variant:20s} {ms:7.3f} ms/frame   {measure / dt:8.1f} frames/s   "
          f"({measure / dt / 60.0:5.2f}x realtime)", flush=True)
    return {"variant": variant, "ms_per_frame": ms, "fps": measure / dt, "realtime_factor": measure / dt / 60.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = [bench(v) for v in a.variants]
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
