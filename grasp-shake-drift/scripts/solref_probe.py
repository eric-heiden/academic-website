# SPDX-License-Identifier: Apache-2.0
"""Print the solref MuJoCo ends up using for the loaded grasp contacts.

Used to check whether raising ``ke`` past ~2.5e5 still changes the reference
dynamics, or whether MuJoCo's refsafe clamp (timeconst >= 2*dt) has taken over.
"""

from __future__ import annotations

import sys

import numpy as np

import newton
from newton.viewer import ViewerNull

from newtontests.franka_cube_shake import Phase, create_parser
from newtontests.experiments import Variant, VARIANTS


def arr(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def probe(variant, frames=560):
    argv_backup = sys.argv
    sys.argv = ["solref"]
    args = newton.examples.default_args(create_parser())
    sys.argv = argv_backup

    viewer = ViewerNull(num_frames=frames)
    ex = Variant(viewer, args, VARIANTS[variant])
    d, c = ex.solver.mjw_data, ex.solver.mjw_data.contact
    g2s = arr(ex.solver.mjc_geom_to_newton_shape)[0]
    shape_body = arr(ex.model.shape_body)
    cube_geoms = {i for i in range(len(g2s))
                  if 0 <= int(g2s[i]) < len(shape_body) and int(shape_body[int(g2s[i])]) == ex.cube_index}

    for _ in range(frames):
        ex.step()

    nacon = int(arr(d.nacon)[0]) if hasattr(d, "nacon") else int(arr(d.ncon)[0])
    geom, solref, dist = arr(c.geom)[:nacon], arr(c.solref)[:nacon], arr(c.dist)[:nacon]
    addr, dim = arr(c.efc_address)[:nacon], arr(c.dim)[:nacon]
    force = arr(d.efc.force)
    force = force[0] if force.ndim == 2 else force

    print(f"\n--- {variant} ---   sim_dt = {ex.sim_dt:.6f} s   (2*dt = {2 * ex.sim_dt:.6f})")
    seen = set()
    for k in range(nacon):
        if not (int(geom[k][0]) in cube_geoms or int(geom[k][1]) in cube_geoms):
            continue
        a0 = int(addr[k][0]) if np.ndim(addr[k]) else int(addr[k])
        if a0 < 0 or a0 >= len(force) or abs(force[a0]) < 1e-6:
            continue
        key = tuple(np.round(solref[k], 8))
        if key in seen:
            continue
        seen.add(key)
        print(f"   loaded contact solref = {solref[k]}   penetration = {dist[k]:+.6f} m   Fn = {force[a0]:.4f} N")
    if not seen:
        print("   (no loaded grasp contacts found)")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["baseline"]:
        probe(v)
