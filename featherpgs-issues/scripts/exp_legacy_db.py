"""Does the double-buffer flag change the answer, or only the schedule?

It is documented as an execution option, so the trajectory should not depend on
it. The earlier report ran with it on (the default); everything measured here
runs with it off, so the two need to be compared directly.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")

import newton  # noqa: E402
import legacy_coupon as L  # noqa: E402

if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    out = {}
    traces = {}
    for geom in ("box", "mesh"):
        for db in (False, True):
            for streams in (False, True):
                k = f"{geom}_db{int(db)}_st{int(streams)}"
                try:
                    e = L.make(geom=geom, solver_kw={"double_buffer": db,
                                                     "use_parallel_streams": streams})
                    r = L.run(e, frames=600)
                    traces[k] = r["z"]
                    s = {kk: (float(v) if isinstance(v, (float, np.floating)) else v)
                         for kk, v in r.items() if not hasattr(v, "shape")}
                    del e
                except Exception as exc:  # noqa: BLE001
                    s = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
                s.update(geom=geom, double_buffer=db, parallel_streams=streams)
                out[k] = s
                print(k, json.dumps(s, default=str), flush=True)
    # exact agreement between the flag settings, per geometry
    for geom in ("box", "mesh"):
        base = traces.get(f"{geom}_db0_st0")
        if base is None:
            continue
        for k, z in traces.items():
            if not k.startswith(geom) or k.endswith("db0_st0"):
                continue
            n = min(len(base), len(z))
            d = float(np.abs(base[:n] - z[:n]).max() * 1000.0)
            out[f"diff_{k}"] = {"max_z_difference_mm": d, "identical": d == 0.0}
            print(f"diff_{k}", json.dumps(out[f"diff_{k}"]), flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
