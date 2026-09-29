# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Recording a second CUDA graph around the same SolverMuJoCo instance crashes.

Newton examples capture a CUDA graph right after constructing SolverMuJoCo, so
the solver's first step (which allocates internal buffers) runs inside the
capture and those buffers belong to that graph. Capturing another graph around
the same solver then segfaults or corrupts the simulation. One eager step
before the first capture avoids it.

Usage: python mujoco_recapture_repro.py {once|recapture|warm_recapture}
Observed on Newton main @ 6354432c and eric-heiden/newton-live-mcp @ de507713 (Warp 1.17.0,
mujoco_warp 3.12.0, RTX PRO 6000 Blackwell MIG slice):
  once            -> box rests at z = 0.0499
  recapture       -> segmentation fault
  warm_recapture  -> box rests at z = 0.0499
"""

import sys

import warp as wp

import newton

mode = sys.argv[1] if len(sys.argv) > 1 else "recapture"
builder = newton.ModelBuilder()
builder.add_ground_plane()
body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()))
builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
model = builder.finalize()
solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
state_0, state_1, control = model.state(), model.state(), model.control()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)


def simulate():
    global state_0, state_1
    for _ in range(2):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0


if mode == "warm_recapture":
    simulate()  # eager warm-up: the solver allocates its buffers outside any graph
with wp.ScopedCapture() as capture:
    simulate()
graph = capture.graph
if mode in ("recapture", "warm_recapture"):
    with wp.ScopedCapture() as capture:
        simulate()
    graph = capture.graph
for _ in range(240):
    wp.capture_launch(graph)
print(mode, state_0.body_q.numpy()[0, :3].round(4))
