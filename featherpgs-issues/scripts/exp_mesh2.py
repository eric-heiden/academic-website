"""Does the collision representation of the grasped object change the outcome?

The scene is unchanged except that the cube is built from a triangle mesh
instead of a primitive box, with identical dimensions, density and material.
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

ORIGINAL_ADD_CUBE = fcs.FrankaCubeShake._add_cube


def cube_mesh(h):
    v = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                  [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], dtype=np.float32)
    i = np.array([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
                  0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
                  0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5], dtype=np.int32)
    return newton.Mesh(v, i)


def mesh_add_cube(self, scene) -> int:
    cfg = newton.ModelBuilder.ShapeConfig(density=fcs.CUBE_DENSITY, ke=8.0e4,
                                          kd=8.0e2, mu=1.2)
    idx = scene.add_body(xform=wp.transform(self.cube_start_pos, wp.quat_identity()),
                         label="shake_cube")
    scene.add_shape_mesh(body=idx, mesh=cube_mesh(0.5 * self.cube_size),
                         cfg=cfg, color=(0.15, 0.45, 0.9))
    return idx


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    out = {}
    for tag, use_mesh, solver in (("box_fpgs", False, "fpgs"),
                                  ("mesh_fpgs", True, "fpgs"),
                                  ("box_mujoco", False, "mujoco"),
                                  ("mesh_mujoco", True, "mujoco")):
        fcs.FrankaCubeShake._add_cube = mesh_add_cube if use_mesh else ORIGINAL_ADD_CUBE
        try:
            env = make_env(solver)
            r = run_gc(env, 900, gc=(solver == "fpgs"))
            s = summarize(r)
            z = r["cube_pos"][:, 2]
            m = r["phase"] == 4
            s["z_at_shake"] = float(z[m][0]) if m.any() else None
            s["z_end"] = float(z[-1])
            s["contacts_median"] = float(np.median(r["n_contacts"]))
            np.savez_compressed(f"data/meshscene_{tag}.npz",
                                **{k: v for k, v in r.items() if k != "failure"},
                                failure=np.asarray(r["failure"]))
            del env
        except Exception as exc:  # noqa: BLE001
            s = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        out[tag] = s
        print(tag, json.dumps(s, default=str)[:340], flush=True)
    fcs.FrankaCubeShake._add_cube = ORIGINAL_ADD_CUBE
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
