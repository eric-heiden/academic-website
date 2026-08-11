"""Grasp-and-shake with the missing gravity term supplied as feed-forward torque."""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from gravcomp import GravityCompensator  # noqa: E402
from newtontests.franka_cube_shake import Phase  # noqa: E402


def run_gc(env, frames, gc=True, stride=1, record_poses=False):
    comp = GravityCompensator(env) if gc else None
    cube, ee = env.cube_index, env.ee_index
    log = {k: [] for k in ("frame", "t", "phase", "cube_pos", "ee_pos", "cube_quat",
                           "rel_pos", "rel_pos_local", "n_contacts", "joint_q", "target_q",
                           "grip_q")}
    poses = []
    failure = ""
    t0 = time.perf_counter()
    done = 0
    try:
        for frame in range(frames):
            if comp is not None:
                comp.apply()
            env.step()
            done += 1
            if frame % stride:
                continue
            bq = env.state_0.body_q.numpy()
            cq, eq = bq[cube], bq[ee]
            x, y, z, w = eq[3:7]
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            rel = cq[:3] - eq[:3]
            log["frame"].append(frame)
            log["t"].append(env.sim_time)
            log["phase"].append(int(env.phase_index.numpy()[0]))
            log["cube_pos"].append(cq[:3].tolist())
            log["cube_quat"].append(cq[3:7].tolist())
            log["ee_pos"].append(eq[:3].tolist())
            log["rel_pos"].append(rel.tolist())
            log["rel_pos_local"].append((R.T @ rel).tolist())
            log["n_contacts"].append(int(env.contacts.rigid_contact_count.numpy()[0]))
            jq = env.state_0.joint_q.numpy()[:9]
            log["joint_q"].append(jq.tolist())
            log["grip_q"].append(jq[7:9].tolist())
            log["target_q"].append(env.control.joint_target_q.numpy()[:9].tolist())
            if record_poses:
                poses.append(bq.copy())
            if not np.all(np.isfinite(cq[:3])):
                failure = f"non-finite pose at frame {frame}"
                break
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    out = {k: np.asarray(v) for k, v in log.items()}
    out["failure"] = failure
    out["wall_s"] = wall
    out["steps"] = done
    out["rtf"] = done * env.frame_dt / wall if wall > 0 else 0.0
    if record_poses:
        out["poses"] = np.stack(poses) if poses else np.zeros((0, 1, 7))
    return out


CASES = {
    "fpgs_gc_default": dict(solver_kw={}),
    "fpgs_gc_sharedanchor": dict(solver_kw={"contact_friction_shared_anchor": True}),
    "fpgs_gc_velit4": dict(solver_kw={"pgs_velocity_iterations": 4}),
    "fpgs_gc_anchor_velit4": dict(
        solver_kw={"contact_friction_shared_anchor": True, "pgs_velocity_iterations": 4}),
    "fpgs_gc_it32": dict(solver_kw={"pgs_iterations": 32}),
    "fpgs_gc_best": dict(solver_kw={
        "contact_friction_shared_anchor": True,
        "contact_shared_anchor": True,
        "pgs_velocity_iterations": 4,
        "pgs_iterations": 20,
    }),
}

if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    only = sys.argv[1] if len(sys.argv) > 1 else None
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    results = {}
    for tag, cfg in CASES.items():
        if only and only != "all" and tag != only:
            continue
        env = make_env("fpgs", **cfg)
        res = run_gc(env, frames)
        s = summarize(res)
        s["max_phase"] = Phase(int(res["phase"].max())).name
        s["cfg"] = cfg["solver_kw"]
        results[tag] = s
        np.savez_compressed(f"data/{tag}.npz",
                            **{k: v for k, v in res.items() if k != "failure"},
                            failure=np.asarray(res["failure"]))
        print(tag, json.dumps(s), flush=True)
        del env
    print("===JSON===")
    print(json.dumps(results, indent=2, default=str))
