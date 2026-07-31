# SPDX-License-Identifier: Apache-2.0
"""Report the friction and solimp values MuJoCo actually applies to the grasp.

The example raises ``mujoco:geom_priority`` on the finger geoms. In MuJoCo a
higher-priority geom supplies friction/solimp/solref for the pair outright, so
the value authored on the cube may never be used.
"""

from __future__ import annotations

import sys

import numpy as np

import newton
from newton.viewer import ViewerNull

from newtontests.franka_cube_shake import Phase, create_parser
from newtontests.experiments import Variant, VARIANTS


def main(variant="baseline", frames=260):
    argv_backup = sys.argv
    sys.argv = ["probe"]
    args = newton.examples.default_args(create_parser())
    sys.argv = argv_backup

    viewer = ViewerNull(num_frames=frames)
    ex = Variant(viewer, args, VARIANTS[variant])

    m = ex.solver.mjw_model
    g2s = ex.solver.mjc_geom_to_newton_shape.numpy()[0]
    shape_body = ex.model.shape_body.numpy()
    shape_mu = ex.model.shape_material_mu.numpy()
    labels = list(ex.model.shape_label) if hasattr(ex.model, "shape_label") else None

    print("=== Newton shape_material_mu ===")
    for i in range(len(shape_body)):
        b = int(shape_body[i])
        if b in (12, 13, ex.cube_index):
            nm = labels[i] if labels else f"shape{i}"
            print(f"  shape {i:3d} body {b:3d} mu={shape_mu[i]:.3f}  {nm}")

    gf = m.geom_friction.numpy()
    gp = m.geom_priority.numpy()
    gsolimp = m.geom_solimp.numpy()
    gsolmix = m.geom_solmix.numpy()
    ggap = m.geom_gap.numpy()
    gmargin = m.geom_margin.numpy()
    print("\n=== MuJoCo geom properties (world 0) ===")
    gf0 = gf[0] if gf.ndim == 3 else gf
    gp0 = gp[0] if gp.ndim == 2 else gp
    gi0 = gsolimp[0] if gsolimp.ndim == 3 else gsolimp
    gm0 = gsolmix[0] if gsolmix.ndim == 2 else gsolmix
    gg0 = ggap[0] if ggap.ndim == 2 else ggap
    gmg0 = gmargin[0] if gmargin.ndim == 2 else gmargin
    print("  (mjc geom -> newton shape mapping in use)")
    for i in range(len(gf0)):
        s_idx = int(g2s[i]) if i < len(g2s) else -1
        b = int(shape_body[s_idx]) if 0 <= s_idx < len(shape_body) else -99
        if b in (12, 13, ex.cube_index):
            print(f"  geom {i:3d} shape {s_idx:3d} body {b:3d} priority={int(gp0[i])} friction={gf0[i]} "
                  f"solimp={gi0[i]} solmix={gm0[i]:.3f} gap={gg0[i]:.4f} margin={gmg0[i]:.4f}")

    # Step into the grasp, then read the live contacts.
    for f in range(frames):
        ex.step()
        if int(ex.phase_index.numpy()[0]) >= Phase.LIFT.value:
            break
    for _ in range(30):
        ex.step()

    d = ex.solver.mjw_data
    ncon = int(d.nacon.numpy()[0]) if hasattr(d, 'nacon') else int(d.ncon.numpy()[0])
    c = d.contact
    geom = c.geom.numpy()[0][:ncon] if c.geom.numpy().ndim == 3 else c.geom.numpy()[:ncon]
    fric = c.friction.numpy()[0][:ncon] if c.friction.numpy().ndim == 3 else c.friction.numpy()[:ncon]
    dist = c.dist.numpy()[0][:ncon] if c.dist.numpy().ndim == 2 else c.dist.numpy()[:ncon]
    solimp = c.solimp.numpy()[0][:ncon] if c.solimp.numpy().ndim == 3 else c.solimp.numpy()[:ncon]
    solref = c.solref.numpy()[0][:ncon] if c.solref.numpy().ndim == 3 else c.solref.numpy()[:ncon]
    includemargin = c.includemargin.numpy()[0][:ncon] if hasattr(c, "includemargin") else None

    cube_geoms = {i for i in range(len(g2s)) if 0 <= int(g2s[i]) < len(shape_body) and int(shape_body[int(g2s[i])]) == ex.cube_index}
    print(f"\n=== live contacts during grasp (ncon={ncon}), cube geoms={cube_geoms} ===")
    shown = 0
    for k in range(ncon):
        g = geom[k]
        if not (int(g[0]) in cube_geoms or int(g[1]) in cube_geoms):
            continue
        im = float(includemargin[k]) if includemargin is not None else float("nan")
        print(f"  con{k:3d} geoms=({int(g[0])},{int(g[1])}) dist={float(dist[k]):+.6f} "
              f"friction={fric[k]} solref={solref[k]} solimp={solimp[k]} includemargin={im:.5f}")
        shown += 1
        if shown >= 12:
            print("  ...")
            break
    if shown == 0:
        print("  (no cube contacts in MuJoCo's list — Newton contacts are forwarded directly)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
