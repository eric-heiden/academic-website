"""The static two-pad coupon, across geometry, solver and friction options."""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")

import newton  # noqa: E402
import static_coupon as L  # noqa: E402


def keep(r):
    out = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
           for k, v in r.items() if not hasattr(v, "shape")}
    out["contacts_median"] = float(np.median(r["n_contacts"]))
    return out


def save(tag, r):
    np.savez_compressed(
        f"data/legacy_{tag}.npz",
        **{k: v for k, v in r.items()
           if hasattr(v, "shape") and k != "poses"})


CASES = [
    ("box_fpgs", dict(geom="box", solver="fpgs")),
    ("mesh_fpgs", dict(geom="mesh", solver="fpgs")),
    ("box_mujoco", dict(geom="box", solver="mujoco")),
    ("mesh_mujoco", dict(geom="mesh", solver="mujoco")),
    ("box_anchors", dict(geom="box", solver="fpgs",
                         solver_kw={"contact_friction_anchor_limit": 2})),
    ("mesh_anchors", dict(geom="mesh", solver="fpgs",
                          solver_kw={"contact_friction_anchor_limit": 2})),
    ("box_shared", dict(geom="box", solver="fpgs",
                        solver_kw={"contact_friction_shared_anchor": True})),
    ("mesh_shared", dict(geom="mesh", solver="fpgs",
                         solver_kw={"contact_friction_shared_anchor": True})),
]


def grid():
    out = {}
    for tag, kw in CASES:
        try:
            e = L.make(**kw)
            r = L.run(e, frames=600)
            save(tag, r)
            s = keep(r)
            del e
        except Exception as exc:  # noqa: BLE001
            s = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        s.update(case=tag)
        out[tag] = s
        print(tag, json.dumps(s, default=str), flush=True)
    return out


def iteration_probe(frames=600):
    """If the squeeze is indeterminate, creep should move with iteration count."""
    out = {}
    for geom in ("box", "mesh"):
        for it in (4, 12, 32, 64):
            k = f"{geom}_it{it}"
            e = L.make(geom=geom, solver_kw={"pgs_iterations": it})
            r = L.run(e, frames=frames)
            out[k] = dict(keep(r), geom=geom, pgs_iterations=it)
            print(k, json.dumps(out[k], default=str), flush=True)
            del e
    return out


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("all", "grid"):
        res["grid"] = grid()
    if which in ("all", "iters"):
        res["iters"] = iteration_probe()
    print("===JSON===")
    print(json.dumps(res, indent=2, default=str))
