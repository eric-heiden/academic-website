"""Public-API composition prototype for PR #3850, without RSS or a new core API.

GUI: uv run --extra examples python /absolute/path/newton_public_ui.py
Probe: uv run --extra examples python /absolute/path/newton_public_ui.py --check
A callback queues intent; only the main simulation thread changes state.
"""

from __future__ import annotations

import argparse
import threading
import time
from queue import Empty, SimpleQueue

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverXPBD
from newton.viewer import ViewerNull, ViewerViser


@wp.kernel
def add_velocity(qd: wp.array(dtype=wp.spatial_vector), delta_v: float):
    qd[0] = qd[0] + wp.spatial_vector(0.0, 0.0, delta_v, 0.0, 0.0, 0.0)


class Example:
    def __init__(self, viewer):
        self.viewer = viewer
        self.commands = SimpleQueue()
        self.generation = 0
        self.sim_time = 0.0
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()), label="study/box")
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.solver = SolverXPBD(self.model, iterations=5)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.control = self.model.control()
        self.reset()
        viewer.set_model(self.model)
        self.status = None
        if isinstance(viewer, ViewerViser):
            viewer.set_camera(wp.vec3(2, -3, 1.8), pitch=-20.0, yaw=124.0)
            # The only Viser-specific boundary is the PR's public server property.
            with viewer.server.gui.add_folder("Partner panel prototype"):
                kick = viewer.server.gui.add_button("Add upward velocity (+2 m/s)")
                reset = viewer.server.gui.add_button("Reset this scene")
                self.status = viewer.server.gui.add_markdown("Commands are applied between steps.")
            kick.on_click(lambda _: self.commands.put((self.generation, "kick", 2.0)))
            reset.on_click(lambda _: self.commands.put((self.generation, "reset", 0.0)))

    def reset(self):
        self.generation += 1
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.sim_time = 0.0

    def drain(self):
        while True:
            try:
                generation, action, value = self.commands.get_nowait()
            except Empty:
                break
            if generation != self.generation:
                continue
            if action == "reset":
                self.reset()
            elif action == "kick":
                wp.launch(add_velocity, dim=1, inputs=[self.state_0.body_qd, value], device=self.model.device)

    def step(self):
        self.drain()
        for _ in range(4):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, 1.0 / 240.0)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += 1.0 / 60.0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pose = self.state_0.body_q.numpy()
        assert np.isfinite(pose).all()
        assert pose[0, 2] > 0.05, pose


def check(device):
    with wp.ScopedDevice(device):
        example = Example(ViewerNull())
        for _ in range(180):
            example.step()
        z0 = float(example.state_0.body_q.numpy()[0, 2])
        worker = threading.Thread(target=lambda: example.commands.put((example.generation, "kick", 2.0)))
        worker.start()
        worker.join()
        heights = []
        for _ in range(30):
            example.step()
            heights.append(float(example.state_0.body_q.numpy()[0, 2]))
        assert max(heights) > z0 + 0.1, (z0, max(heights))
        old = example.generation
        example.commands.put((old, "reset", 0.0))
        example.commands.put((old, "kick", 50.0))
        example.drain()
        assert example.generation == old + 1
        assert np.allclose(example.state_0.body_qd.numpy(), 0.0), "stale command crossed reset"
        example.test_final()
        print(f"PASS {device}: worker command raised box {max(heights) - z0:.4f} m; reset discarded stale command")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    if args.check:
        check(args.device)
    else:
        with wp.ScopedDevice(args.device):
            viewer = ViewerViser(port=args.port)
            example = Example(viewer)
            try:
                while viewer.is_running():
                    start = time.perf_counter()
                    example.drain()
                    if not viewer.is_paused():
                        example.step()
                    example.render()
                    time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - start)))
            finally:
                viewer.close()
