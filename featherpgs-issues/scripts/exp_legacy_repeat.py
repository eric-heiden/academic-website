"""Is the coupon reproducible run to run, with everything held fixed?

Same process, same settings, same machine, three times each. Any difference is
the solver, not the configuration.
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
    for geom in ("box", "mesh"):
        traces = []
        for i in range(3):
            e = L.make(geom=geom)
            r = L.run(e, frames=600)
            traces.append(r["z"])
            print(f"{geom} run {i}: down={r['worst_downward_mm']:.4f} mm "
                  f"up={r['worst_upward_mm']:.4f} mm", flush=True)
            del e
        n = min(len(t) for t in traces)
        devs = [float(np.abs(traces[0][:n] - t[:n]).max() * 1000.0) for t in traces[1:]]
        first = []
        for t in traces[1:]:
            d = np.abs(traces[0][:n] - t[:n])
            nz = np.nonzero(d > 0)[0]
            first.append(int(nz[0]) if len(nz) else -1)
        out[geom] = {"max_deviation_mm": devs, "first_diverging_frame": first,
                     "identical": all(d == 0.0 for d in devs),
                     "worst_downward_mm": [float((0.065 - t.min()) * 1000.0)
                                           for t in traces]}
        print(geom, json.dumps(out[geom]), flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
