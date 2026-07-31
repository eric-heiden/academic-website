# SPDX-License-Identifier: Apache-2.0
"""Check whether the grasp's friction rows carry a position residual.

MuJoCo friction constraints are expected to be velocity-level: the normal row has
a position residual (penetration) but the two tangential rows should have
efc_pos = 0, meaning nothing pulls the object back to where it was.
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


def main(variant="baseline", frames=560):
    argv_backup = sys.argv
    sys.argv = ["efc"]
    args = newton.examples.default_args(create_parser())
    sys.argv = argv_backup

    viewer = ViewerNull(num_frames=frames)
    ex = Variant(viewer, args, VARIANTS[variant])
    d = ex.solver.mjw_data
    c = d.contact
    g2s = arr(ex.solver.mjc_geom_to_newton_shape)[0]
    shape_body = arr(ex.model.shape_body)
    cube_geoms = {i for i in range(len(g2s))
                  if 0 <= int(g2s[i]) < len(shape_body) and int(shape_body[int(g2s[i])]) == ex.cube_index}

    for f in range(frames):
        ex.step()

    m = ex.solver.mjw_model
    opt = m.opt
    def show(name):
        v = getattr(opt, name, None)
        if v is None:
            return "n/a"
        v = arr(v)
        return str(v.reshape(-1)[:3])
    print("\nresolved solver options:",
          {k: show(k) for k in ("impratio", "cone", "tolerance", "ls_tolerance", "iterations", "timestep")},
          flush=True)

    efc = d.efc
    fields = {}
    for name in ("pos", "vel", "force", "D", "aref", "J", "frictionloss"):
        if hasattr(efc, name):
            fields[name] = arr(getattr(efc, name))
    print("efc fields:", {k: v.shape for k, v in fields.items()}, flush=True)

    pos = fields["pos"][0] if fields["pos"].ndim == 2 else fields["pos"]
    vel = fields["vel"][0] if "vel" in fields and fields["vel"].ndim == 2 else fields.get("vel")
    force = fields["force"][0] if fields["force"].ndim == 2 else fields["force"]

    nacon = int(arr(d.nacon)[0]) if hasattr(d, "nacon") else int(arr(d.ncon)[0])
    geom = arr(c.geom)[:nacon]
    dim = arr(c.dim)[:nacon]
    addr = arr(c.efc_address)[:nacon]

    print(f"\nphase={Phase(int(ex.phase_index.numpy()[0])).name}  nacon={nacon}")
    print("grasp contacts carrying force — per-row breakdown:")
    shown = 0
    for k in range(nacon):
        if not (int(geom[k][0]) in cube_geoms or int(geom[k][1]) in cube_geoms):
            continue
        a0 = int(addr[k][0]) if np.ndim(addr[k]) else int(addr[k])
        nd = int(dim[k])
        if a0 < 0 or a0 + nd > len(force) or abs(force[a0]) < 1e-6:
            continue
        print(f"  contact geoms=({int(geom[k][0])},{int(geom[k][1])}) condim={nd}")
        for j in range(nd):
            kind = "normal " if j == 0 else f"friction{j}"
            v = f"{vel[a0 + j]:+.3e}" if vel is not None else "n/a"
            print(f"    row {j} [{kind}]  efc_pos={pos[a0 + j]:+.6e}  efc_vel={v}  force={force[a0 + j]:+.5f}")
        shown += 1
        if shown >= 3:
            break


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
