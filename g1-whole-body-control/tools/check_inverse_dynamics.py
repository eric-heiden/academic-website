"""Diagnostic only: compare Newton inverse dynamics to MuJoCo; not a supported correction."""

import argparse
import json, time
import numpy as np
import warp as wp
import mujoco
import newton
import newton.viewer
from newton.examples.robot.example_robot_g1_wbc import Example
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("motion")
parser.add_argument("output")
args = parser.parse_args()
ex = Example(
    newton.viewer.ViewerNull(),
    Example.create_parser().parse_args(["--motion", args.motion, "--viewer", "null"]),
)
n = ex.model
m = ex.mj
s = n.state()
d = mujoco.MjData(m)
M = wp.zeros((1, 35, 35), device="cpu")
c = wp.zeros(35, device="cpu")
g = wp.zeros(35, device="cpu")
force = wp.zeros(35, device="cpu")
acc = wp.zeros(35, device="cpu")
rng = np.random.default_rng(123)
rows = []
for i in range(75):
    q, v, _ = ex.motion.sample((i % 25) * 0.19)
    v = rng.normal(size=35) * 2
    mode = ("stationary", "moving_joints", "moving_base_and_joints")[i // 25]
    if i < 25:
        v[:] = 0
    elif i < 50:
        v[:6] = 0
    d.qpos[:] = q
    d.qvel[:] = v
    mujoco.mj_forward(m, d)
    R = d.xmat[1].reshape(3, 3)
    r = R @ m.body_ipos[1]
    w = R @ v[3:6]
    skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    B = np.eye(35)
    B[:3, 3:6] = -skew @ R
    B[3:6, 3:6] = R
    qn = q.copy()
    qn[3:7] = q[[4, 5, 6, 3]]
    s.joint_q.assign(qn.astype(np.float32))
    s.joint_qd.assign((B @ v).astype(np.float32))
    newton.eval_fk(n, s.joint_q, s.joint_qd, s)
    start = time.perf_counter()
    newton.eval_inverse_dynamics_passive(
        n, s, mass_matrix=M, coriolis_force=c, gravity_force=g
    )
    Mn = M.numpy()[0].astype(float)
    bn = (c.numpy() + g.numpy()).astype(float)
    raw_bn = bn.copy()
    svel = np.cross(w, (B @ v)[:3])
    correction = -Mn[:, :3] @ svel
    correction[:3] += m.body_mass[1] * svel
    bn += correction
    bdv = np.r_[np.cross(w, np.cross(w, r)), np.zeros(32)]
    Mm = B.T @ Mn @ B + np.diag(m.dof_armature)
    bm = B.T @ (bn + Mn @ bdv)
    native_ms = (time.perf_counter() - start) * 1e3
    Mr = np.zeros((35, 35))
    mujoco.mj_fullM(m, d, Mr)
    a = rng.normal(size=35)
    acc.assign((B @ a + bdv).astype(np.float32))
    newton.eval_inverse_dynamics_force(
        n,
        s,
        mass_matrix=M,
        coriolis_force=c,
        gravity_force=g,
        joint_qdd=acc,
        joint_f=force,
    )
    fm = B.T @ force.numpy() + m.dof_armature * a
    raw_bm = B.T @ (raw_bn + Mn @ bdv)
    rows.append(
        dict(
            mode=mode,
            raw_bias_error=float(np.max(np.abs(raw_bm - d.qfrc_bias))),
            raw_joint_bias_error=float(np.max(np.abs(raw_bm[6:] - d.qfrc_bias[6:]))),
            corrected_force_error=float(
                np.max(np.abs(fm + B.T @ correction - (Mr @ a + d.qfrc_bias)))
            ),
            M_error=float(np.max(np.abs(Mm - Mr))),
            bias_error=float(np.max(np.abs(bm - d.qfrc_bias))),
            force_error=float(np.max(np.abs(fm - (Mr @ a + d.qfrc_bias)))),
            native_ms=native_ms,
        )
    )
open(args.output, "w").write(json.dumps(rows, indent=2))
for mode in ("stationary", "moving_joints", "moving_base_and_joints"):
    print(
        mode,
        {
            k: max(r[k] for r in rows if r["mode"] == mode)
            for k in rows[0]
            if k != "mode"
        },
    )
