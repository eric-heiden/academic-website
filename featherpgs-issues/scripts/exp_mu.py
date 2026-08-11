"""Grasp quality as the grasped object's friction coefficient changes.

The coefficient is set when the cube is built, before either solver is
constructed, because SolverMuJoCo converts materials into its own model at
construction and does not see later edits.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
import newtontests.franka_cube_shake as fcs  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402

ORIGINAL = fcs.FrankaCubeShake._add_cube
MU = [1.2]


def add_cube_mu(self, scene) -> int:
    cfg = newton.ModelBuilder.ShapeConfig(density=fcs.CUBE_DENSITY, ke=8.0e4,
                                          kd=8.0e2, mu=MU[0])
    idx = scene.add_body(xform=wp.transform(self.cube_start_pos, wp.quat_identity()),
                         label="shake_cube")
    scene.add_shape_box(body=idx, hx=0.5 * self.cube_size, hy=0.5 * self.cube_size,
                        hz=0.5 * self.cube_size, cfg=cfg, color=(0.15, 0.45, 0.9))
    return idx


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    fcs.FrankaCubeShake._add_cube = add_cube_mu
    out = {}
    for mu in (0.2, 0.3, 0.5, 0.8, 1.2, 2.0):
        for solv in ("fpgs", "mujoco"):
            MU[0] = mu
            k = f"mu{mu}_{solv}"
            try:
                env = make_env(solv)
                eff = float(env.model.shape_material_mu.numpy()[
                    [i for i, b in enumerate(env.model.shape_body.numpy())
                     if b == env.cube_index][0]])
                r = run_gc(env, 700, gc=(solv == "fpgs"))
                s = summarize(r)
                z = r["cube_pos"][:, 2]
                m = r["phase"] == 4
                s["held"] = bool(m.any() and z[m][0] > 0.10 and z[-1] > 0.10
                                 and np.linalg.norm(r["rel_pos"][m], axis=1).max() < 0.08)
                s["mu_cube"] = eff
                s["solver"] = solv
                del env
            except Exception as exc:  # noqa: BLE001
                s = {"error": f"{type(exc).__name__}: {str(exc)[:180]}",
                     "mu_cube": mu, "solver": solv, "held": False}
            out[k] = s
            print(k, json.dumps(s, default=str)[:210], flush=True)
    fcs.FrankaCubeShake._add_cube = ORIGINAL
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
