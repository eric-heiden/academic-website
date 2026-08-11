"""Franka grasp-and-shake harness with per-frame instrumentation.

Runs the StoneT2000/newton-tests FrankaCubeShake scene under a selectable
solver and records the quantities that decide whether a grasp is stable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
import newton.examples  # noqa: E402
from newtontests.franka_cube_shake import FrankaCubeShake, Phase, create_parser  # noqa: E402


def build_args(**over):
    args = newton.examples.default_args(create_parser())
    args.device = "cuda:0"
    args.viewer = "null"
    args.headless = True
    args.quiet = True
    args.shake_amplitude = 0.03
    args.shake_frequency = 1.0
    for k, v in over.items():
        setattr(args, k, v)
    return args


FPGS_DEFAULTS = dict(
    pgs_mode="matrix_free",
    pgs_iterations=12,
    pgs_beta=0.2,
    pgs_cfm=1.0e-6,
    dense_max_constraints=256,
    mf_max_constraints=4096,
    use_parallel_streams=False,
    double_buffer=False,
)


class _NoSimGraph:
    """Capture only the IK graph; the physics loop is replayed eagerly."""

    def _capture_graphs(self):
        self.sim_graph = None
        self.ik_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)
            self.ik_graph = capture.graph


def make_env(solver_name, substeps=16, solver_kw=None, gain_scale=1.0, kd_scale=None,
             mu_cube=None, mu_finger=None, capture_sim_graph=None):
    if capture_sim_graph is None:
        capture_sim_graph = solver_name == "mujoco"
    args = build_args()
    # The scene fixes its substep count and captures its physics graph inside
    # __init__, so the count has to be changed before that capture happens.
    original_capture = FrankaCubeShake._capture_graphs

    def _pre_capture(self):
        self.sim_substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        original_capture(self)

    FrankaCubeShake._capture_graphs = _pre_capture
    try:
        env = FrankaCubeShake(newton.viewer.ViewerNull(num_frames=100000), args)
    finally:
        FrankaCubeShake._capture_graphs = original_capture
    if not capture_sim_graph:
        env._capture_graphs = _NoSimGraph._capture_graphs.__get__(env)
        env.sim_graph = None

    if gain_scale != 1.0:
        ke = env.model.joint_target_ke.numpy()
        kd = env.model.joint_target_kd.numpy()
        ke[:9] *= gain_scale
        kd[:9] *= gain_scale if kd_scale is None else kd_scale
        env.model.joint_target_ke.assign(ke)
        env.model.joint_target_kd.assign(kd)

    if mu_cube is not None or mu_finger is not None:
        mu = env.model.shape_material_mu.numpy()
        sb = env.model.shape_body.numpy()
        for s, b in enumerate(sb):
            if b == env.cube_index and mu_cube is not None:
                mu[s] = mu_cube
            if b in (12, 13) and mu_finger is not None:
                mu[s] = mu_finger
        env.model.shape_material_mu.assign(mu)

    if substeps != env.sim_substeps:
        env.sim_substeps = substeps
        env.sim_dt = env.frame_dt / substeps

    kw = dict(solver_kw or {})
    if solver_name == "fpgs":
        cfg = dict(FPGS_DEFAULTS)
        cfg.update(kw)
        env.solver = newton.solvers.SolverFeatherPGS(env.model, **cfg)
        env.control._use_coord_layout_targets = False
        env.solver_cfg = cfg
    elif solver_name == "mujoco":
        env.solver_cfg = "scene default (SolverMuJoCo)"
        if kw:
            cfg = dict(
                solver="newton", integrator="implicitfast", iterations=15,
                ls_iterations=100, nconmax=4096, njmax=8192,
                cone="elliptic", impratio=50.0, use_mujoco_contacts=False,
            )
            cfg.update(kw)
            env.solver = newton.solvers.SolverMuJoCo(env.model, **cfg)
            env.solver_cfg = cfg
    else:
        raise ValueError(solver_name)

    if solver_name != "mujoco":
        env.reset()
    return env


def finger_indices(env):
    labels = list(env.model.joint_key) if hasattr(env.model, "joint_key") else []
    return labels


def run(env, frames, stride=1, record_poses=False, capture_graph=False):
    """Advance the environment and log grasp-stability signals per frame."""
    cube = env.cube_index
    ee = env.ee_index

    if capture_graph and wp.get_device().is_cuda:
        try:
            with wp.ScopedCapture() as cap:
                env._simulate()
            env.sim_graph = cap.graph
        except Exception as exc:  # noqa: BLE001
            env.sim_graph = None
            env.graph_error = str(exc)

    log = {
        "frame": [], "t": [], "phase": [],
        "cube_pos": [], "ee_pos": [], "cube_quat": [],
        "rel_pos": [], "rel_pos_local": [],
        "n_contacts": [], "joint_q": [], "joint_qd": [], "target_q": [],
    }
    poses = []
    failure = ""
    ref = None
    t0 = time.perf_counter()
    steps_done = 0
    try:
        for frame in range(frames):
            env.step()
            steps_done += 1
            if frame % stride:
                continue
            bq = env.state_0.body_q.numpy()
            phase = int(env.phase_index.numpy()[0])
            cq = bq[cube]
            eq = bq[ee]
            # cube position expressed in the end-effector frame
            q = eq[3:7]
            x, y, z, w = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            rel = cq[:3] - eq[:3]
            rel_local = R.T @ rel
            if phase == Phase.SHAKE.value and ref is None:
                ref = rel_local.copy()
            log["frame"].append(frame)
            log["t"].append(env.sim_time)
            log["phase"].append(phase)
            log["cube_pos"].append(cq[:3].tolist())
            log["cube_quat"].append(cq[3:7].tolist())
            log["ee_pos"].append(eq[:3].tolist())
            log["rel_pos"].append(rel.tolist())
            log["rel_pos_local"].append(rel_local.tolist())
            log["n_contacts"].append(int(env.contacts.rigid_contact_count.numpy()[0]))
            log["joint_q"].append(env.state_0.joint_q.numpy()[:9].tolist())
            log["joint_qd"].append(env.state_0.joint_qd.numpy()[:9].tolist())
            log["target_q"].append(env.control.joint_target_q.numpy()[:9].tolist())
            if record_poses:
                poses.append(bq.copy())
            if not np.all(np.isfinite(cq[:3])):
                failure = f"non-finite cube pose at frame {frame}"
                break
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"

    wall = time.perf_counter() - t0
    out = {k: np.asarray(v) for k, v in log.items()}
    out["failure"] = failure
    out["wall_s"] = wall
    out["steps"] = steps_done
    out["sim_s"] = steps_done * env.frame_dt
    out["rtf"] = (steps_done * env.frame_dt / wall) if wall > 0 else 0.0
    out["ref_local"] = ref if ref is not None else np.zeros(3)
    if record_poses:
        out["poses"] = np.stack(poses) if poses else np.zeros((0, 1, 7))
    return out


def summarize(res):
    """Reduce a run to the numbers that describe grasp quality."""
    ph = res["phase"]
    shake = ph == Phase.SHAKE.value
    s = {
        "failure": res["failure"],
        "reached_shake": bool(shake.any()),
        "frames": int(len(ph)),
        "rtf": float(res["rtf"]),
        "wall_s": float(res["wall_s"]),
    }
    if shake.any():
        rl = res["rel_pos_local"][shake]
        ref = rl[0]
        drift = rl - ref
        s["shake_frames"] = int(shake.sum())
        s["drift_max_mm"] = float(np.abs(drift).max() * 1000.0)
        s["drift_final_mm"] = float(np.linalg.norm(drift[-1]) * 1000.0)
        s["slip_axial_mm"] = float(drift[-1][2] * 1000.0)
        s["drift_rms_mm"] = float(np.sqrt((drift**2).sum(axis=1).mean()) * 1000.0)
        sep = np.linalg.norm(res["rel_pos"][shake], axis=1)
        s["max_separation_m"] = float(sep.max())
        s["dropped"] = bool(sep.max() > 0.08)
        s["first_shake_frame"] = int(res["frame"][shake][0])
    else:
        s["last_phase"] = int(ph[-1]) if len(ph) else -1
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", default="fpgs")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--substeps", type=int, default=16)
    p.add_argument("--gain-scale", type=float, default=1.0)
    p.add_argument("--kd-scale", type=float, default=None)
    p.add_argument("--solver-kw", default="{}")
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default="run")
    cli = p.parse_args()

    wp.init()
    wp.set_device("cuda:0")
    env = make_env(
        cli.solver,
        substeps=cli.substeps,
        solver_kw=json.loads(cli.solver_kw),
        gain_scale=cli.gain_scale,
        kd_scale=cli.kd_scale,
    )
    res = run(env, cli.frames)
    s = summarize(res)
    s["tag"] = cli.tag
    print(json.dumps(s, indent=2))
    if cli.out:
        Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cli.out, **{k: v for k, v in res.items() if k != "failure"},
                            failure=np.asarray(res["failure"]))


if __name__ == "__main__":
    main()
