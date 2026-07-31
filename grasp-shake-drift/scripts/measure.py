# SPDX-License-Identifier: Apache-2.0
"""Headless instrumented runner for the Franka cube shake test.

Measures cube slip relative to the gripper TCP frame, contact statistics, and
finger joint state so the grasp can be diagnosed without a viewer.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull

from newtontests.franka_cube_shake import FrankaCubeShake, Phase, create_parser


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_rot(q, v):
    qv = np.array(q[:3])
    w = q[3]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def rel_pose(parent_q, child_q):
    """Return child pose expressed in the parent frame (pos, quat)."""
    pp, pq = parent_q[:3], parent_q[3:7]
    cp, cq = child_q[:3], child_q[3:7]
    inv = quat_conj(pq)
    return quat_rot(inv, cp - pp), quat_mul(inv, cq)


def run(overrides=None, num_frames=900, tag="baseline", verbose=True):
    overrides = overrides or {}
    argv_backup = sys.argv
    sys.argv = ["measure"]
    parser = create_parser()
    args = newton.examples.default_args(parser)
    sys.argv = argv_backup
    args.shake_amplitude = overrides.pop("shake_amplitude", 0.03)
    args.shake_frequency = overrides.pop("shake_frequency", 1.0)

    viewer = ViewerNull(num_frames=num_frames)
    ex = FrankaCubeShake(viewer, args)

    # Apply post-construction overrides (patched variants set these before use).
    for key, value in overrides.items():
        setattr(ex, key, value)

    ref_rel_pos = None
    ref_rel_quat = None
    rows = []
    for frame in range(num_frames):
        ex.step()
        body_q = ex.state_0.body_q.numpy()
        cube = body_q[ex.cube_index]
        tcp = body_q[ex.ee_index]
        rp, rq = rel_pose(tcp, cube)
        phase = int(ex.phase_index.numpy()[0])
        ptime = float(ex.phase_time.numpy()[0])

        if phase == Phase.SHAKE.value and ref_rel_pos is None:
            ref_rel_pos = rp.copy()
            ref_rel_quat = rq.copy()

        slip = float(np.linalg.norm(rp - ref_rel_pos)) if ref_rel_pos is not None else 0.0
        if ref_rel_quat is not None:
            dq = quat_mul(rq, quat_conj(ref_rel_quat))
            ang = float(np.degrees(2.0 * np.arccos(np.clip(abs(dq[3]), 0.0, 1.0))))
        else:
            ang = 0.0

        joint_q = ex.state_0.joint_q.numpy()
        contacts = ex.contacts
        ncon = int(contacts.rigid_contact_count.numpy()[0])

        rows.append(
            {
                "frame": frame,
                "t": frame / 60.0,
                "phase": phase,
                "phase_time": ptime,
                "cube_z": float(cube[2]),
                "tcp_z": float(tcp[2]),
                "rel_x": float(rp[0]),
                "rel_y": float(rp[1]),
                "rel_z": float(rp[2]),
                "slip": slip,
                "slip_deg": ang,
                "finger0": float(joint_q[7]),
                "finger1": float(joint_q[8]),
                "ncon": ncon,
            }
        )
        if verbose and frame % 60 == 0:
            r = rows[-1]
            print(
                f"[{tag}] f={frame:4d} t={r['t']:5.2f} phase={Phase(phase).name:8s} "
                f"cube_z={r['cube_z']:.4f} slip={slip * 1000:6.2f}mm rot={ang:6.2f}deg "
                f"fingers=({r['finger0']:.4f},{r['finger1']:.4f}) ncon={ncon}",
                flush=True,
            )

    return rows


def summarize(rows, tag):
    shake = [r for r in rows if r["phase"] == Phase.SHAKE.value]
    final = rows[-1]
    out = {
        "tag": tag,
        "reached_shake": bool(shake),
        "shake_frames": len(shake),
        "final_slip_mm": final["slip"] * 1000.0,
        "final_slip_deg": final["slip_deg"],
        "max_slip_mm": max((r["slip"] for r in shake), default=0.0) * 1000.0,
        "max_slip_deg": max((r["slip_deg"] for r in shake), default=0.0),
        "final_cube_z": final["cube_z"],
        "final_rel": [final["rel_x"], final["rel_y"], final["rel_z"]],
        "mean_ncon": float(np.mean([r["ncon"] for r in shake])) if shake else 0.0,
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--amplitude", type=float, default=0.03)
    ap.add_argument("--frequency", type=float, default=1.0)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = run(
        overrides={"shake_amplitude": a.amplitude, "shake_frequency": a.frequency},
        num_frames=a.frames,
        tag=a.tag,
    )
    s = summarize(rows, a.tag)
    print(json.dumps(s, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": s, "rows": rows}, f)
