"""Assorted capability checks: row capacity, control layout, friction modes."""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402
from harness import summarize  # noqa: E402


def simple_model():
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.49), wp.quat_identity()))
    b.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    b.add_joint_free(body)
    b.add_ground_plane()
    m = b.finalize()
    pipe = newton.CollisionPipeline(m)
    c = pipe.contacts()
    s0, s1 = m.state(), m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    return m, pipe, c, s0, s1, m.control()


def dense_row_capacity():
    """Does the tiled dense solve build at large row counts on this GPU?"""
    out = {}
    for rows in (32, 64, 128, 256, 512):
        m, pipe, c, s0, s1, ctl = simple_model()
        try:
            solver = newton.solvers.SolverFeatherPGS(
                m, pgs_mode="split", pgs_iterations=12, dense_max_constraints=rows)
            s0.clear_forces()
            pipe.collide(s0, c)
            solver.step(s0, s1, ctl, c, 1.0 / 960.0)
            wp.synchronize()
            out[rows] = "ok"
        except Exception as exc:  # noqa: BLE001
            out[rows] = f"{type(exc).__name__}: {str(exc)[:160]}"
        print("dense rows", rows, "->", out[rows], flush=True)
    return out


def coord_layout_targets():
    """Can FeatherPGS consume Newton's current coordinate-layout drive targets?"""
    prev = newton.use_coord_layout_targets
    result = {}
    for flag in (False, True):
        newton.use_coord_layout_targets = flag
        try:
            b = newton.ModelBuilder()
            body = b.add_body()
            b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            b.add_joint_revolute(-1, body, axis=wp.vec3(0.0, 0.0, 1.0),
                                 target_ke=100.0, target_kd=10.0)
            m = b.finalize()
            pipe = newton.CollisionPipeline(m)
            c = pipe.contacts()
            s0, s1 = m.state(), m.state()
            ctl = m.control()
            newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
            solver = newton.solvers.SolverFeatherPGS(m, pgs_mode="matrix_free",
                                                     double_buffer=False)
            s0.clear_forces()
            pipe.collide(s0, c)
            solver.step(s0, s1, ctl, c, 1.0 / 240.0)
            wp.synchronize()
            result[f"use_coord_layout_targets={flag}"] = "ok"
        except Exception as exc:  # noqa: BLE001
            result[f"use_coord_layout_targets={flag}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print("coord layout", flag, "->", result[f"use_coord_layout_targets={flag}"], flush=True)
    newton.use_coord_layout_targets = prev
    return result


def long_horizon():
    """Long-horizon stability: does a 2000-frame run stay healthy?"""
    out = {}
    for tag, gain, gc, frames in (("gain100_2000f", 100.0, False, 2000),
                                  ("gravity_2000f", 1.0, True, 2000)):
        env = make_env("fpgs", gain_scale=gain)
        r = run_gc(env, frames, gc=gc, stride=10)
        s = summarize(r)
        s["frames_requested"] = frames
        s["frames_completed"] = int(r["steps"])
        out[tag] = s
        print(tag, json.dumps(s), flush=True)
        del env
    return out


def friction_options():
    """Alternative friction formulations on the articulated contact path."""
    out = {}
    for mode in ("current", "bisection", "bisection_desaxce", "coulomb_newton"):
        try:
            env = make_env("fpgs", solver_kw={"friction_mode": mode})
            r = run_gc(env, 700)
            s = summarize(r)
            out[mode] = s
        except Exception as exc:  # noqa: BLE001
            out[mode] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        print("friction_mode", mode, json.dumps(out[mode], default=str)[:260], flush=True)
    return out


def mu_combination():
    """How is the pair friction coefficient formed from two materials?"""
    env = make_env("fpgs")
    mu = env.model.shape_material_mu.numpy()
    sb = env.model.shape_body.numpy()
    finger = [i for i, b in enumerate(sb) if b in (12, 13)]
    cube = [i for i, b in enumerate(sb) if b == env.cube_index]
    out = {"finger_mu": float(mu[finger[0]]) if finger else None,
           "cube_mu": float(mu[cube[0]]) if cube else None}
    if out["finger_mu"] is not None and out["cube_mu"] is not None:
        out["arithmetic_mean"] = 0.5 * (out["finger_mu"] + out["cube_mu"])
        out["minimum"] = min(out["finger_mu"], out["cube_mu"])
    del env
    return out


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = {}
    try:
        if which in ("all", "dense"):
            r["dense_rows"] = dense_row_capacity()
        if which in ("all", "coord"):
            r["coord_layout"] = coord_layout_targets()
        if which in ("all", "mu"):
            r["mu"] = mu_combination()
        if which in ("all", "friction"):
            r["friction_modes"] = friction_options()
        if which in ("all", "long"):
            r["long_horizon"] = long_horizon()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    print("===JSON===")
    print(json.dumps(r, indent=2, default=str))
