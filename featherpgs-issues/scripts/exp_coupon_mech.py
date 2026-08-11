"""Where does the ejection energy in the static coupon come from?

No gravity, no shake: the object is simply placed between two immovable pads
with a fixed overlap and left alone. Under a hard contact model the two opposed
normal constraints have no force balance to converge to, so whatever velocity
the object leaves with is a property of the solver, not of the model.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")

import newton  # noqa: E402
import coupon  # noqa: E402


def squeeze_only(overlap, geom="box", solver="fpgs", beta=0.2, substeps=16,
                 frames=60, solver_kw=None):
    kw = dict(solver_kw or {})
    if solver == "fpgs":
        kw.setdefault("pgs_beta", beta)
    e = coupon.make("static", geom, solver=solver, solver_kw=kw,
                    overlap=overlap, gravity=False)
    model, pipeline, contacts = e["model"], e["pipeline"], e["contacts"]
    state, control, solv = e["state"], e["control"], e["solver"]
    obj = e["obj"]
    dt = 1.0 / 60.0 / substeps
    speeds, pos = [], []
    for _ in range(frames):
        pipeline.collide(state[0], contacts)
        for _k in range(substeps):
            state[0].clear_forces()
            solv.step(state[0], state[1], control, contacts, dt)
            state[0], state[1] = state[1], state[0]
        v = state[0].body_qd.numpy()[obj]
        speeds.append(float(np.linalg.norm(v[:3])))
        pos.append(state[0].body_q.numpy()[obj, :3].copy())
    pos = np.asarray(pos)
    disp = pos - pos[0]
    del e
    return {"overlap_mm": overlap * 1000.0, "geom": geom, "solver": solver,
            "beta": beta if solver == "fpgs" else None,
            "peak_speed_m_s": float(np.max(speeds)),
            "final_speed_m_s": float(speeds[-1]),
            "displacement_mm": float(np.abs(disp).max() * 1000.0),
            "baumgarte_prediction_m_s": beta * (0.5 * overlap) / dt
            if solver == "fpgs" else None}


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    out = {}
    for ov in (2.5e-4, 5.0e-4, 1.0e-3, 2.0e-3):
        for solv in ("fpgs", "mujoco"):
            k = f"ov{ov * 1000:g}_{solv}"
            try:
                out[k] = squeeze_only(ov, solver=solv)
            except Exception as exc:  # noqa: BLE001
                out[k] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
            print(k, json.dumps(out[k], default=str), flush=True)
    # the same overlap under three position-correction factors
    for beta in (0.0, 0.1, 0.2, 0.5):
        k = f"beta{beta}"
        try:
            out[k] = squeeze_only(1.0e-3, beta=beta)
        except Exception as exc:  # noqa: BLE001
            out[k] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
        print(k, json.dumps(out[k], default=str), flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
