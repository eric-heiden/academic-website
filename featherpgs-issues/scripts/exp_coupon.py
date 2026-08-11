"""Minimal parallel-gripper coupons.

Two pads and one object, with nothing else in the scene:

  static  - pads welded to the world, object squeezed by a fixed geometric
            overlap. This is the shape the earlier report's minimal
            reproduction used.
  driven  - pads on a position-driven two-axis carrier that closes, lifts the
            object off a support, and shakes it. Grip force is set by the
            finger drive rather than by geometry.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")

import newton  # noqa: E402
import coupon  # noqa: E402


def detail(r, start):
    """When the grasp was lost and how fast the object was moving."""
    pos = r["obj_pos"]
    if len(pos) < 3:
        return {}
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 60.0
    out = {"peak_speed_m_s": float(v.max())}
    pad = r["pad_contacts"]
    lost = np.nonzero(pad[start:] == 0)[0]
    out["first_zero_contact_frame"] = int(lost[0] + start) if len(lost) else None
    out["frames"] = int(len(pos))
    return out


def summarize(r, start=0):
    keep = ("held", "slip_mm", "axial_slip_mm", "lateral_mm", "final_slip_mm",
            "contacts_median", "zero_contact_frames", "shake_frames",
            "finger_overshoot_mm", "failure")
    s = {k: (float(r[k]) if isinstance(r[k], (np.floating, float)) else r[k])
         for k in keep if k in r}
    s.update(detail(r, start))
    return s


def static_grid(frames=300):
    """Overlap sweep: does squeezing harder help or hurt?"""
    out = {}
    for ov in (2.5e-4, 5.0e-4, 1.0e-3, 2.0e-3):
        for geom in ("box", "mesh"):
            for solv in ("fpgs", "mujoco"):
                k = f"ov{ov * 1000:g}_{geom}_{solv}"
                try:
                    e = coupon.make("static", geom, solver=solv, overlap=ov)
                    r = coupon.run(e, frames=frames)
                    s = summarize(r, start=90)
                    del e
                except Exception as exc:  # noqa: BLE001
                    s = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
                s.update(overlap_mm=ov * 1000.0, geom=geom, solver=solv)
                out[k] = s
                print(k, json.dumps(s, default=str), flush=True)
    return out


def driven_grid(frames=420):
    """The same object held by a real position-driven gripper."""
    out = {}
    for sq in (0.5e-3, 1.0e-3, 2.0e-3, 4.0e-3):
        for geom in ("box", "mesh"):
            for solv in ("fpgs", "mujoco"):
                k = f"sq{sq * 1000:g}_{geom}_{solv}"
                try:
                    e = coupon.make("driven", geom, solver=solv, squeeze=sq)
                    r = coupon.run(e, frames=frames)
                    s = summarize(r, start=int(r["start_shake"]))
                    del e
                except Exception as exc:  # noqa: BLE001
                    s = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
                s.update(squeeze_mm=sq * 1000.0, geom=geom, solver=solv)
                out[k] = s
                print(k, json.dumps(s, default=str), flush=True)
    return out


def driven_knobs(frames=420):
    """Iterations, time step and friction on the driven coupon."""
    out = {}
    for it in (4, 8, 12, 20, 32):
        e = coupon.make("driven", "box", solver_kw={"pgs_iterations": it})
        r = coupon.run(e, frames=frames)
        out[f"it{it}"] = dict(summarize(r, int(r["start_shake"])), pgs_iterations=it)
        print(f"it{it}", json.dumps(out[f"it{it}"], default=str), flush=True)
        del e
    for ss in (16, 8, 4, 2, 1):
        for solv in ("fpgs", "mujoco"):
            k = f"ss{ss}_{solv}"
            e = coupon.make("driven", "box", solver=solv)
            r = coupon.run(e, frames=frames, substeps=ss)
            out[k] = dict(summarize(r, int(r["start_shake"])), substeps=ss, solver=solv)
            print(k, json.dumps(out[k], default=str), flush=True)
            del e
    for mu in (0.2, 0.4, 0.6, 1.0, 1.5):
        for solv in ("fpgs", "mujoco"):
            k = f"mu{mu}_{solv}"
            e = coupon.make("driven", "box", solver=solv, mu=mu)
            r = coupon.run(e, frames=frames)
            out[k] = dict(summarize(r, int(r["start_shake"])), mu=mu, solver=solv)
            print(k, json.dumps(out[k], default=str), flush=True)
            del e
    return out


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("all", "static"):
        res["static"] = static_grid()
    if which in ("all", "driven"):
        res["driven"] = driven_grid()
    if which in ("all", "knobs"):
        res["knobs"] = driven_knobs()
    print("===JSON===")
    print(json.dumps(res, indent=2, default=str))
