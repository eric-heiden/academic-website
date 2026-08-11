"""What happens with SolverFeatherPGS's own default settings.

Every earlier measurement used the configuration the previous report settled on
(matrix-free mode, 256 dense rows, double buffering off). This checks the
library defaults, and the row capacity the default dense solve can actually
build.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
import harness  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402
from newtontests.franka_cube_shake import Phase  # noqa: E402

CASES = [
    ("library_defaults", {}),
    ("defaults_no_double_buffer", {"double_buffer": False}),
    ("split_rows150", {"double_buffer": False, "dense_max_constraints": 150}),
    ("split_rows201", {"double_buffer": False, "dense_max_constraints": 201}),
    ("split_rows256", {"double_buffer": False, "dense_max_constraints": 256}),
    ("matrixfree_rows256", {"double_buffer": False, "pgs_mode": "matrix_free",
                            "dense_max_constraints": 256}),
]

if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    # Start from the library's own defaults rather than the report's overrides.
    harness.FPGS_DEFAULTS = {}
    out = {}
    for tag, kw in CASES:
        try:
            env = make_env("fpgs", solver_kw=kw)
            r = run_gc(env, 700)
            s = summarize(r)
            s["max_phase"] = Phase(int(r["phase"].max())).name if len(r["phase"]) else "?"
            s["pgs_mode"] = env.solver.pgs_mode
            s["dense_max_constraints"] = env.solver.dense_max_constraints
            s["pgs_kernel"] = env.solver.pgs_kernel
            np.savez_compressed(f"data/def_{tag}.npz",
                                **{k: v for k, v in r.items() if k != "failure"},
                                failure=np.asarray(r["failure"]))
            del env
        except Exception as exc:  # noqa: BLE001
            s = {"construct_or_run_error": f"{type(exc).__name__}: {str(exc)[:260]}"}
        s["cfg"] = kw
        out[tag] = s
        print(tag, json.dumps(s, default=str)[:400], flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
