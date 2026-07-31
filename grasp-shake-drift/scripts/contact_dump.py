# SPDX-License-Identifier: Apache-2.0
"""Dump the live MuJoCo-Warp contact rows and constraint forces for the grasp.

Answers: how many contacts hold the cube, what condim / friction they carry, how
much normal force they apply, and how much of the friction cone is in use.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import newton
from newton.viewer import ViewerNull

from newtontests.franka_cube_shake import Phase, create_parser
from newtontests.experiments import Variant, VARIANTS
from newtontests.measure import rel_pose


def arr(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def main(variant="baseline", frames=900, dump_frames=(300, 500, 700)):
    argv_backup = sys.argv
    sys.argv = ["dump"]
    args = newton.examples.default_args(create_parser())
    sys.argv = argv_backup

    viewer = ViewerNull(num_frames=frames)
    ex = Variant(viewer, args, VARIANTS[variant])
    # keep CUDA graph; MuJoCo data arrays are readable after replay

    d = ex.solver.mjw_data
    c = d.contact
    g2s = arr(ex.solver.mjc_geom_to_newton_shape)[0]
    shape_body = arr(ex.model.shape_body)
    cube_geoms = {i for i in range(len(g2s))
                  if 0 <= int(g2s[i]) < len(shape_body) and int(shape_body[int(g2s[i])]) == ex.cube_index}
    finger_geoms = {i for i in range(len(g2s))
                    if 0 <= int(g2s[i]) < len(shape_body) and int(shape_body[int(g2s[i])]) in (12, 13)}
    print("cube geoms:", sorted(cube_geoms), " finger geoms:", sorted(finger_geoms), flush=True)

    print("contact field shapes:", {k: arr(getattr(c, k)).shape
                                    for k in ["geom", "dist", "friction", "dim", "efc_address", "solimp", "includemargin"]
                                    if hasattr(c, k)}, flush=True)
    efc = d.efc.force if hasattr(d, "efc") else d.efc_force
    print("efc force shape:", arr(efc).shape, flush=True)

    records = []
    for frame in range(frames):
        ex.step()
        if frame not in dump_frames:
            continue
        phase = int(ex.phase_index.numpy()[0])
        nacon = int(arr(d.nacon)[0]) if hasattr(d, "nacon") else int(arr(d.ncon)[0])
        geom = arr(c.geom)[:nacon]
        dist = arr(c.dist)[:nacon]
        fric = arr(c.friction)[:nacon]
        dim = arr(c.dim)[:nacon]
        addr = arr(c.efc_address)[:nacon]
        incm = arr(c.includemargin)[:nacon] if hasattr(c, "includemargin") else None
        f = arr(d.efc.force if hasattr(d, "efc") else d.efc_force)
        f = f[0] if f.ndim == 2 else f

        rows = []
        for k in range(nacon):
            g0, g1 = int(geom[k][0]), int(geom[k][1])
            pair_is_grasp = (g0 in cube_geoms and g1 in finger_geoms) or (g1 in cube_geoms and g0 in finger_geoms)
            if not pair_is_grasp:
                continue
            a0 = int(addr[k][0]) if np.ndim(addr[k]) else int(addr[k])
            nd = int(dim[k])
            fn = float(f[a0]) if 0 <= a0 < len(f) else float("nan")
            ft = float(np.linalg.norm(f[a0 + 1: a0 + nd])) if 0 <= a0 and nd > 1 else 0.0
            mu = float(fric[k][0])
            rows.append({
                "k": k, "geoms": [g0, g1], "dist": float(dist[k]), "dim": nd,
                "mu": mu, "fn": fn, "ft": ft,
                "cone_util": (ft / (mu * fn)) if (mu > 0 and fn > 1e-9) else 0.0,
                "includemargin": float(incm[k]) if incm is not None else None,
            })
        active = [r for r in rows if r["fn"] > 1e-6]
        fn_tot = sum(r["fn"] for r in active)
        ft_tot = sum(r["ft"] for r in active)
        util = max((r["cone_util"] for r in active), default=0.0)
        print(f"\n=== frame {frame} phase={Phase(phase).name} nacon={nacon} grasp_rows={len(rows)} "
              f"force-carrying={len(active)} ===", flush=True)
        print(f"    total normal force  {fn_tot:9.4f} N", flush=True)
        print(f"    total tangential    {ft_tot:9.4f} N", flush=True)
        print(f"    max cone utilization {util:8.4f}   (1.0 = sliding)", flush=True)
        print(f"    condim values: {sorted({r['dim'] for r in rows})}   mu values: {sorted({round(r['mu'],3) for r in rows})}", flush=True)
        print(f"    includemargin: {sorted({round(r['includemargin'],5) for r in rows if r['includemargin'] is not None})}", flush=True)
        for r in sorted(rows, key=lambda r: -r["fn"])[:8]:
            print(f"      geoms={r['geoms']} dist={r['dist']:+.6f} dim={r['dim']} mu={r['mu']:.2f} "
                  f"Fn={r['fn']:8.4f} Ft={r['ft']:7.4f} util={r['cone_util']:.4f}", flush=True)
        records.append({"frame": frame, "phase": phase, "nacon": nacon,
                        "fn_total": fn_tot, "ft_total": ft_tot, "max_util": util,
                        "n_force_carrying": len(active), "rows": rows})
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--frames", type=int, default=760)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = main(a.variant, a.frames)
    if a.out:
        json.dump(recs, open(a.out, "w"))
