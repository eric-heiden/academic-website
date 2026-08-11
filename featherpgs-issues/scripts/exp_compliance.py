"""Does the contact compliance knob rescue the wedged coupon and help the grasp?"""

from __future__ import annotations

import json
import sys

import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")

import newton  # noqa: E402
import coupon  # noqa: E402
from exp_coupon import summarize  # noqa: E402

if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    out = {}
    for comp in (0.0, 1.0e-8, 1.0e-6, 1.0e-4):
        for kind, extra in (("static", {"overlap": 2.0e-3}), ("driven", {})):
            k = f"{kind}_c{comp:g}"
            try:
                e = coupon.make(kind, "box",
                                solver_kw={"dense_contact_compliance": comp}, **extra)
                r = coupon.run(e, frames=300 if kind == "static" else 420)
                s = summarize(r, int(r.get("start_shake", 90)))
                del e
            except Exception as exc:  # noqa: BLE001
                s = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
            s.update(compliance=comp, kind=kind)
            out[k] = s
            print(k, json.dumps(s, default=str), flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
