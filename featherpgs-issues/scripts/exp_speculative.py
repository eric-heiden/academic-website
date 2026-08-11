"""Why do extra velocity iterations make the grasp worse?

The velocity-only pass rebuilds every contact right-hand side with the
speculative term switched off, so a contact that is still a few millimetres
away stops being "you may close this gap" and becomes "you may not approach at
all". This measures how many contacts are in that state and what happens when
the collision gap is shrunk so that almost none of them exist.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402


def count_open_contacts(env):
    """Split live contacts into touching (phi<=0) and speculative (phi>0)."""
    c = env.contacts
    n = int(c.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0, 0
    bq = env.state_0.body_q.numpy()
    shape_body = env.model.shape_body.numpy()
    s0 = c.rigid_contact_shape0.numpy()[:n]
    s1 = c.rigid_contact_shape1.numpy()[:n]
    p0 = c.rigid_contact_point0.numpy()[:n]
    p1 = c.rigid_contact_point1.numpy()[:n]
    nn = c.rigid_contact_normal.numpy()[:n]
    m0 = c.rigid_contact_margin0.numpy()[:n]
    m1 = c.rigid_contact_margin1.numpy()[:n]

    def world(pt, sh):
        out = np.zeros_like(pt)
        for i, (p, s) in enumerate(zip(pt, sh)):
            b = shape_body[s] if s >= 0 else -1
            if b < 0:
                out[i] = p
                continue
            x, y, z, qx, qy, qz, qw = bq[b]
            t = 2.0 * np.cross([qx, qy, qz], p)
            out[i] = np.array([x, y, z]) + p + qw * t + np.cross([qx, qy, qz], t)
        return out

    phi = np.einsum("ij,ij->i", nn, world(p1, s1) - world(p0, s0)) - (m0 + m1)
    return int((phi <= 0).sum()), int((phi > 0).sum())


def gap_scale(env, factor):
    g = env.model.shape_gap.numpy()
    env.model.shape_gap.assign((g * factor).astype(np.float32))


CASES = [
    ("base_gap1", {}, 1.0),
    ("velit4_gap1", {"pgs_velocity_iterations": 4}, 1.0),
    ("velit4_gap0.02", {"pgs_velocity_iterations": 4}, 0.02),
    ("base_gap0.02", {}, 0.02),
    ("velit8_gap0.02", {"pgs_velocity_iterations": 8}, 0.02),
]

if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    res = {}
    for tag, kw, gf in CASES:
        env = make_env("fpgs", solver_kw=kw)
        if gf != 1.0:
            gap_scale(env, gf)
        r = run_gc(env, 700)
        s = summarize(r)
        s["cfg"] = kw
        s["gap_factor"] = gf
        # sample the touching / speculative split at the end of the run
        touch, spec = count_open_contacts(env)
        s["touching_contacts"] = touch
        s["speculative_contacts"] = spec
        res[tag] = s
        print(tag, json.dumps(s), flush=True)
        np.savez_compressed(f"data/spec_{tag}.npz",
                            **{k: v for k, v in r.items() if k != "failure"},
                            failure=np.asarray(r["failure"]))
        del env
    print("===JSON===")
    print(json.dumps(res, indent=2, default=str))
