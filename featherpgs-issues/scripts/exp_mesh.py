"""Does the collision representation change the outcome?

A cube is pinched between two static pads with a fixed overlap and driven by an
oscillating body force. The only difference between the two runs is whether the
cube is a primitive box or the same box as a triangle mesh.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import warp as wp

import newton


@wp.kernel
def shake(body_f: wp.array[wp.spatial_vector], body: int, fy: float):
    body_f[body] = wp.spatial_vector(wp.vec3(0.0, fy, 0.0), wp.vec3(0.0))


def cube_mesh(h):
    v = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                  [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], dtype=np.float32)
    i = np.array([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
                  0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
                  0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5], dtype=np.int32)
    return newton.Mesh(v, i)


def coupon(kind, frames=600, substeps=16, solver_kw=None):
    half, z0, overlap = 0.02, 0.065, 0.0005
    pad_hx, pad_h = 0.005, 0.06
    pad_x = half + pad_hx - 0.5 * overlap
    dt = 1.0 / (60.0 * substeps)

    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=1.0, ke=2.5e5, kd=1.0e3,
                                          restitution=0.0)
    b = newton.ModelBuilder()
    cube = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()))
    if kind == "mesh":
        b.add_shape_mesh(cube, mesh=cube_mesh(half), cfg=cfg)
    else:
        b.add_shape_box(cube, hx=half, hy=half, hz=half, cfg=cfg)
    b.add_joint_free(cube)
    for x in (-pad_x, pad_x):
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(x, 0.0, z0), wp.quat_identity()),
                        hx=pad_hx, hy=pad_h, hz=pad_h, cfg=cfg)
    model = b.finalize()
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    kw = dict(pgs_mode="matrix_free", pgs_iterations=12, pgs_beta=0.2, pgs_cfm=1.0e-6,
              double_buffer=False)
    kw.update(solver_kw or {})
    solver = newton.solvers.SolverFeatherPGS(model, **kw)

    mass = float(model.body_mass.numpy()[cube])
    amp, freq = 0.03, 2.0
    om = 2.0 * math.pi * freq
    first_zero = None
    zs = []
    for f in range(frames):
        for k in range(substeps):
            t = (f * substeps + k) * dt
            s0.clear_forces()
            wp.launch(shake, dim=1, inputs=[s0.body_f, cube, -mass * amp * om * om * math.sin(om * t)])
            pipe.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        pipe.collide(s0, contacts)
        if first_zero is None and int(contacts.rigid_contact_count.numpy()[0]) == 0:
            first_zero = f
        zs.append(float(s0.body_q.numpy()[cube, 2]))
    zs = np.asarray(zs)
    return {"kind": kind, "first_zero_contact_frame": first_zero,
            "final_slip_mm": float((z0 - zs[-1]) * 1000.0),
            "worst_slip_mm": float((z0 - zs.min()) * 1000.0),
            "cfg": {k: v for k, v in kw.items() if k != "double_buffer"}}


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    out = {}
    for kind in ("box", "mesh"):
        out[kind] = coupon(kind)
        print(kind, json.dumps(out[kind]), flush=True)
    for kind in ("box", "mesh"):
        out[kind + "_it32"] = coupon(kind, solver_kw={"pgs_iterations": 32})
        print(kind + "_it32", json.dumps(out[kind + "_it32"]), flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))
