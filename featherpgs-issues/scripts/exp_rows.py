"""Constraint-row budget and run-to-run reproducibility.

Two questions:
  1. How many constraint rows does this scene actually need, and what does the
     solver do when the configured budget is smaller than that?
  2. Does the same configuration produce the same trajectory twice?
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402
from newtontests.franka_cube_shake import Phase  # noqa: E402


def watermarks():
    """Row high-water marks over a full grasp+shake with a generous budget."""
    env = make_env("fpgs", solver_kw={"row_watermark": True, "dense_max_constraints": 512})
    res = run_gc(env, 500)
    wm = env.solver.constraint_row_watermarks()
    out = {k: int(v) for k, v in wm.items()}
    out["peak_contacts_reported"] = int(res["n_contacts"].max())
    out["median_contacts"] = float(np.median(res["n_contacts"]))
    del env
    return out


def budget_sweep():
    """Grasp quality vs. the dense row cap, including the library default."""
    out = {}
    for cap in (32, 48, 64, 96, 128, 256):
        env = make_env("fpgs", solver_kw={"dense_max_constraints": cap, "row_watermark": True})
        res = run_gc(env, 700)
        s = summarize(res)
        s["dense_max_constraints"] = cap
        s["max_phase"] = Phase(int(res["phase"].max())).name if len(res["phase"]) else "?"
        try:
            s["watermarks"] = {k: int(v) for k, v in env.solver.constraint_row_watermarks().items()}
        except Exception:  # noqa: BLE001
            pass
        out[f"cap{cap}"] = s
        print("cap", cap, json.dumps(s), flush=True)
        np.savez_compressed(f"data/cap{cap}.npz",
                            **{k: v for k, v in res.items() if k != "failure"},
                            failure=np.asarray(res["failure"]))
        del env
    return out


def repeatability(n=3, frames=500, deterministic=False):
    """Run the identical configuration n times and compare cube trajectories."""
    traj = []
    for i in range(n):
        env = make_env("fpgs")
        if deterministic:
            contact_max = 4096
            env.collision_pipeline = newton.CollisionPipeline(
                env.model, reduce_contacts=True, rigid_contact_max=contact_max,
                broad_phase="nxn", deterministic=True)
            env.contacts = env.collision_pipeline.contacts()
        res = run_gc(env, frames)
        traj.append(res["cube_pos"])
        del env
    n_min = min(len(t) for t in traj)
    traj = [t[:n_min] for t in traj]
    np.savez_compressed(
        "data/repeat_traces.npz" if not deterministic else "data/repeat_traces_det.npz",
        traj=np.stack(traj), frame=np.arange(n_min))
    ref = traj[0]
    devs = [float(np.abs(t - ref).max() * 1000.0) for t in traj[1:]]
    first_div = []
    for t in traj[1:]:
        d = np.abs(t - ref).max(axis=1)
        nz = np.nonzero(d > 0)[0]
        first_div.append(int(nz[0]) if len(nz) else -1)
    return {"max_deviation_mm": devs, "first_diverging_frame": first_div,
            "frames": n_min, "deterministic_pipeline": deterministic}


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = {}
    if which in ("all", "wm"):
        r["watermarks"] = watermarks()
        print("watermarks", json.dumps(r["watermarks"]), flush=True)
    if which in ("all", "budget"):
        r["budget"] = budget_sweep()
    if which in ("all", "repeat"):
        r["repeat_default"] = repeatability()
        print("repeat_default", json.dumps(r["repeat_default"]), flush=True)
        r["repeat_deterministic"] = repeatability(deterministic=True)
        print("repeat_det", json.dumps(r["repeat_deterministic"]), flush=True)
    print("===JSON===")
    print(json.dumps(r, indent=2, default=str))
