"""What the grasp actually looks like at the contact level.

Records finger joint positions, commanded finger targets, pad penetration depth
and the number of pad contacts, so squeeze force and penetration can be compared
between the gravity-compensated run and the 100x-gain workaround.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env  # noqa: E402
from gravcomp import GravityCompensator  # noqa: E402
from newtontests.franka_cube_shake import Phase  # noqa: E402

FINGER_BODIES = (12, 13)


def pad_stats(env):
    """Penetration depth and contact count for finger/cube pairs."""
    c = env.contacts
    n = int(c.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0, 0.0, 0.0
    s0 = c.rigid_contact_shape0.numpy()[:n]
    s1 = c.rigid_contact_shape1.numpy()[:n]
    shape_body = env.model.shape_body.numpy()
    b0 = np.where(s0 >= 0, shape_body[np.clip(s0, 0, None)], -1)
    b1 = np.where(s1 >= 0, shape_body[np.clip(s1, 0, None)], -1)
    is_pad = (np.isin(b0, FINGER_BODIES) & (b1 == env.cube_index)) | \
             (np.isin(b1, FINGER_BODIES) & (b0 == env.cube_index))
    if not is_pad.any():
        return 0, 0.0, 0.0
    bq = env.state_0.body_q.numpy()
    st = env.model.shape_transform.numpy()
    p0 = c.rigid_contact_point0.numpy()[:n][is_pad]
    p1 = c.rigid_contact_point1.numpy()[:n][is_pad]
    nn = c.rigid_contact_normal.numpy()[:n][is_pad]
    m0 = c.rigid_contact_margin0.numpy()[:n][is_pad]
    m1 = c.rigid_contact_margin1.numpy()[:n][is_pad]
    bb0, bb1 = b0[is_pad], b1[is_pad]

    def to_world(pt, body):
        out = np.zeros_like(pt)
        for i, (p, b) in enumerate(zip(pt, body)):
            if b < 0:
                out[i] = p
                continue
            x, y, z, qx, qy, qz, qw = bq[b]
            t = 2.0 * np.cross([qx, qy, qz], p)
            out[i] = np.array([x, y, z]) + p + qw * t + np.cross([qx, qy, qz], t)
        return out

    w0, w1 = to_world(p0, bb0), to_world(p1, bb1)
    sep = np.einsum("ij,ij->i", nn, w1 - w0) - (m0 + m1)
    return int(is_pad.sum()), float(sep.min()), float(np.mean(sep))


def probe(tag, gain, gc, frames=700, solver="fpgs", solver_kw=None):
    env = make_env(solver, gain_scale=gain, solver_kw=solver_kw)
    comp = GravityCompensator(env) if gc else None
    rec = {k: [] for k in ("frame", "phase", "grip_q", "grip_target", "pad_n",
                           "pen_min", "pen_mean", "cube_z", "cube_quat")}
    for f in range(frames):
        if comp is not None:
            comp.apply()
        env.step()
        n, pmin, pmean = pad_stats(env)
        jq = env.state_0.joint_q.numpy()[:9]
        rec["frame"].append(f)
        rec["phase"].append(int(env.phase_index.numpy()[0]))
        rec["grip_q"].append(jq[7:9].tolist())
        rec["grip_target"].append(env.control.joint_target_q.numpy()[7:9].tolist())
        rec["pad_n"].append(n)
        rec["pen_min"].append(pmin)
        rec["pen_mean"].append(pmean)
        bq = env.state_0.body_q.numpy()
        rec["cube_z"].append(float(bq[env.cube_index, 2]))
        rec["cube_quat"].append(bq[env.cube_index, 3:7].tolist())
    out = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(f"data/grip_{tag}.npz", **out)
    shake = out["phase"] == Phase.SHAKE.value
    summary = {
        "tag": tag, "gain": gain, "gravcomp": gc,
        "reached_shake": bool(shake.any()),
        "pad_contacts_median": float(np.median(out["pad_n"][shake])) if shake.any() else 0,
        "pad_contacts_max": int(out["pad_n"].max()),
        "penetration_worst_mm": float(out["pen_min"].min() * 1000.0),
        "penetration_shake_mean_mm": float(out["pen_mean"][shake].mean() * 1000.0) if shake.any() else 0,
        "finger_gap_mm": float(np.mean(out["grip_q"][shake]) * 1000.0) if shake.any() else 0,
        "finger_target_mm": float(np.mean(out["grip_target"][shake]) * 1000.0) if shake.any() else 0,
    }
    summary["finger_overshoot_mm"] = summary["finger_gap_mm"] - summary["finger_target_mm"]
    del env
    return summary


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    res = {}
    for tag, gain, gc, solver in (
        ("mujoco", 1.0, False, "mujoco"),
        ("fpgs_gc", 1.0, True, "fpgs"),
        ("fpgs_100x", 100.0, False, "fpgs"),
        ("fpgs_30x", 30.0, False, "fpgs"),
    ):
        try:
            res[tag] = probe(tag, gain, gc, solver=solver)
        except Exception as exc:  # noqa: BLE001
            res[tag] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(tag, json.dumps(res[tag]), flush=True)
    print("===JSON===")
    print(json.dumps(res, indent=2, default=str))
