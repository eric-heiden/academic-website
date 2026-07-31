# SPDX-License-Identifier: Apache-2.0
"""Map how the grasp drift scales with shake amplitude and frequency."""

from __future__ import annotations

import argparse
import json

from newtontests.experiments import run_variant, VARIANTS


def main(variant, cases, frames, out):
    results = []
    for amp, freq in cases:
        name = f"{variant}_a{amp:g}_f{freq:g}"
        s, rows = run_variant(name, VARIANTS[variant], frames, amp, freq, verbose=False)
        s["amplitude"] = amp
        s["frequency"] = freq
        s["variant"] = variant
        # peak commanded lateral acceleration of the shake path
        s["peak_accel"] = amp * (2 * 3.141592653589793 * freq) ** 2
        results.append(s)
        print(f"  -> a={amp:g} f={freq:g}: drift {s['drift_x_mm_per_s']:.3f} mm/s, "
              f"slip {s['final_slip_mm']:.2f} mm, cube_z {s['final_cube_z']:.3f}", flush=True)
        if out:
            json.dump(results, open(out, "w"), indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--frames", type=int, default=660)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="both", choices=["freq", "amp", "both"])
    a = ap.parse_args()
    cases = []
    if a.mode in ("freq", "both"):
        cases += [(0.03, f) for f in (0.5, 1.0, 2.0, 3.0)]
    if a.mode in ("amp", "both"):
        cases += [(amp, 1.0) for amp in (0.01, 0.06, 0.10)]
    main(a.variant, cases, a.frames, a.out)
