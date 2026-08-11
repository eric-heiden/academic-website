"""Two independent checks.

A. Does the default FeatherPGS execution configuration survive CUDA graph
   capture of a 16-substep frame?
B. Does the Franka controller fail because the drive has no gravity term?
"""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env, run, summarize, FPGS_DEFAULTS  # noqa: E402
from newtontests.franka_cube_shake import Phase  # noqa: E402

RESULTS = {}


def graph_capture_probe():
    """Capture a 16-substep frame under each stream/buffer combination."""
    out = {}
    for streams, dbuf in ((True, True), (True, False), (False, True), (False, False)):
        key = f"streams={streams},double_buffer={dbuf}"
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.49), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_joint_free(body)
        builder.add_ground_plane()
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        solver = newton.solvers.SolverFeatherPGS(
            model, pgs_mode="matrix_free", pgs_iterations=12,
            use_parallel_streams=streams, double_buffer=dbuf,
        )
        state = [s0, s1]

        def frame():
            for _ in range(16):
                state[0].clear_forces()
                pipeline.collide(state[0], contacts)
                solver.step(state[0], state[1], control, contacts, 1.0 / 960.0)
                state[0], state[1] = state[1], state[0]

        try:
            for _ in range(4):
                frame()
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                frame()
            wp.capture_launch(cap.graph)
            wp.synchronize()
            out[key] = "captured"
        except Exception as exc:  # noqa: BLE001
            out[key] = f"{type(exc).__name__}: {exc}"[:220]
        print(key, "->", out[key], flush=True)
    return out


def tracking_probe():
    """Hold the arm at its start pose and measure the steady-state joint error."""
    out = {}
    for tag, solver, gain, zero_g in (
        ("mujoco_g", "mujoco", 1.0, False),
        ("fpgs_g_1x", "fpgs", 1.0, False),
        ("fpgs_g_10x", "fpgs", 10.0, False),
        ("fpgs_g_100x", "fpgs", 100.0, False),
        ("fpgs_nog_1x", "fpgs", 1.0, True),
    ):
        env = make_env(solver, gain_scale=gain)
        if zero_g:
            env.model.gravity.assign(np.zeros_like(env.model.gravity.numpy()))
        # Freeze the controller: hold the initial joint targets for 2 s.
        q0 = env.model.joint_q.numpy()[:9].copy()
        env.control.joint_target_q.assign(
            np.concatenate([q0, env.control.joint_target_q.numpy()[9:]]).astype(np.float32))
        err = []
        for _ in range(120):
            if env.sim_graph is not None:
                wp.capture_launch(env.sim_graph)
            else:
                env._simulate()
            err.append(env.state_0.joint_q.numpy()[:9] - q0)
        err = np.asarray(err)
        out[tag] = {
            "final_err_rad": err[-1].tolist(),
            "final_err_max_deg": float(np.degrees(np.abs(err[-1][:7])).max()),
            "gain_scale": gain,
            "gravity": not zero_g,
        }
        print(tag, "max arm joint error", round(out[tag]["final_err_max_deg"], 4), "deg", flush=True)
        del env
    return out


def phase_probe():
    """Does the scripted task reach SHAKE, and after how many frames?"""
    out = {}
    for tag, gain, zero_g in (
        ("fpgs_1x", 1.0, False),
        ("fpgs_1x_nograv", 1.0, True),
        ("fpgs_3x", 3.0, False),
        ("fpgs_10x", 10.0, False),
        ("fpgs_30x", 30.0, False),
        ("fpgs_100x", 100.0, False),
    ):
        env = make_env("fpgs", gain_scale=gain)
        if zero_g:
            env.model.gravity.assign(np.zeros_like(env.model.gravity.numpy()))
        res = run(env, 420)
        s = summarize(res)
        s["max_phase"] = int(res["phase"].max())
        s["max_phase_name"] = Phase(int(res["phase"].max())).name
        out[tag] = s
        print(tag, s["max_phase_name"], "shake:", s["reached_shake"], flush=True)
        del env
    return out


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "graph"):
        RESULTS["graph_capture"] = graph_capture_probe()
    if which in ("all", "track"):
        RESULTS["tracking"] = tracking_probe()
    if which in ("all", "phase"):
        RESULTS["phase"] = phase_probe()
    print("===JSON===")
    print(json.dumps(RESULTS, indent=2, default=str))
