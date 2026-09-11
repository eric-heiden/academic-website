"""Run with the pinned PR #1535 on PYTHONPATH in the Newton WBC environment."""

import argparse
import json
from pathlib import Path

import newton.viewer
import numpy as np
import warp as wp
from newton.examples.robot.example_robot_g1_wbc import Example
from newton.examples.robot.wbc_mpc_adjoint import _projection_seed

parser = argparse.ArgumentParser(
    description="Compare one local analytic trajectory derivative with central differences."
)
parser.add_argument("--motion", required=True)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--output", type=Path, required=True)
a = parser.parse_args()
steps = a.steps
if steps < 1:
    parser.error("--steps must be positive")
args = Example.create_parser().parse_args(
    [
        "--viewer",
        "null",
        "--controller",
        "mpc-adjoint",
        "--adjoint-sketch",
        "8",
        "--nonfoot-weight",
        "0",
        "--horizon",
        str(steps * 0.005),
        "--motion",
        a.motion,
    ]
)
e = Example(newton.viewer.ViewerNull(), args)
m = e.mpc
m.record_traces = False
q, v, _ = e.motion.sample(0.43)
q = wp.array(q[None], dtype=float)
v = wp.array(v[None], dtype=float)
clock = wp.array([0.43], dtype=float)
rng = np.random.default_rng(57)
z = rng.normal(0, 0.015, m.plan.shape).astype(np.float32)
direction = rng.normal(size=z.shape).astype(np.float32)
direction /= np.linalg.norm(direction)
m.gradient_proposals.assign(np.broadcast_to(z, m.gradient_proposals.shape).copy())
m.differentiate(q, v, clock)
g = m.gradient.numpy()
values = []
seed = np.empty((steps, m.residual_width))
for t in range(steps):
    wp.launch(
        _projection_seed,
        m.residual.shape,
        inputs=[m.sketch, t, m.seed, m.residual],
        outputs=[m.residual.grad, m.stage_costs.grad],
    )
    seed[t] = m.residual.grad.numpy()[0]
for eps in [0.001, 0.003, 0.01]:
    plans = np.broadcast_to(z, m.line_proposals.shape).copy()
    plans[1] += eps * direction
    plans[2] -= eps * direction
    m.data, m.proposals, m.costs = m.line_data, m.line_proposals, m.line_costs
    m.samples = 8
    m.residual = wp.zeros((8, steps * m.residual_width))
    m.save_residuals = True
    m.line_proposals.assign(plans)
    m.rollout(q, v, clock)
    costs = m.costs.numpy()
    res = m.residual.numpy().reshape(8, steps, m.residual_width)
    fd = (costs[1] - costs[2]) / (2 * eps)
    ad = float((g[-1] * direction).sum())
    fds = float((seed * (res[1] - res[2]) / (2 * eps)).sum())
    ads = float((g[0] * direction).sum())
    values.append(
        {
            "epsilon": eps,
            "cost_fd": float(fd),
            "cost_ad": ad,
            "cost_relative": float(abs(fd - ad) / max(abs(fd), abs(ad), 1e-6)),
            "sketch_fd": fds,
            "sketch_ad": ads,
            "sketch_relative": abs(fds - ads) / max(abs(fds), abs(ads), 1e-6),
        }
    )
print(values, flush=True)
a.output.write_text(json.dumps(values, indent=2) + "\n")
