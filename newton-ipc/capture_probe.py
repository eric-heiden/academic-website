# SPDX-FileCopyrightText: Copyright (c) 2026 Eric Heiden
# SPDX-License-Identifier: Apache-2.0
"""Bounded scalar IPC control-flow experiment; not a mesh contact solver.

Run with: uv run --with warp-lang==1.17.0 --with numpy python capture_probe.py
"""

import json
import math
import platform
import unittest
from pathlib import Path

import numpy as np
import warp as wp


@wp.func
def barrier(x: wp.float64) -> wp.float64:
    t = x * x / wp.float64(0.04)
    result = wp.float64(0.0)
    if x <= wp.float64(0.0):
        result = wp.float64(1.0e100)
    elif t < wp.float64(1.0):
        u = t - wp.float64(1.0)
        result = -u * u * wp.log(t)
    return result


@wp.func
def energy(x: wp.float64, target: wp.float64) -> wp.float64:
    delta = x - target
    return wp.float64(50.0) * delta * delta + wp.float64(0.1) * barrier(x)


@wp.func
def derivatives(x: wp.float64, target: wp.float64):
    g = wp.float64(100.0) * (x - target)
    h = wp.float64(100.0)
    t = x * x / wp.float64(0.04)
    if t < wp.float64(1.0):
        u = t - wp.float64(1.0)
        bt = -wp.float64(2.0) * u * wp.log(t) - u * u / t
        btt = -wp.float64(2.0) * wp.log(t) - wp.float64(4.0) * u / t + u * u / (t * t)
        tx = wp.float64(50.0) * x
        g += wp.float64(0.1) * bt * tx
        h += wp.float64(0.1) * (btt * tx * tx + bt * wp.float64(50.0))
    return g, h


@wp.kernel
def initialize(
    start: wp.array[wp.float64],
    x: wp.array[wp.float64],
    status: wp.array[wp.int32],
    iterations: wp.array[wp.int32],
    total_trials: wp.array[wp.int32],
):
    i = wp.tid()
    x[i] = start[i]
    status[i] = 0
    if start[i] <= wp.float64(0.0):
        status[i] = 4
    iterations[i] = 0
    total_trials[i] = 0


@wp.kernel
def prepare(
    x: wp.array[wp.float64],
    target: wp.array[wp.float64],
    status: wp.array[wp.int32],
    iterations: wp.array[wp.int32],
    p: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    g: wp.array[wp.float64],
    e: wp.array[wp.float64],
    search: wp.array[wp.int32],
    trials: wp.array[wp.int32],
    max_newton: int,
):
    i = wp.tid()
    search[i] = 0
    if status[i] == 0:
        grad, hess = derivatives(x[i], target[i])
        if wp.abs(grad) <= wp.float64(1.0e-7):
            status[i] = 1
        elif iterations[i] >= max_newton:
            status[i] = 2
        elif not wp.isfinite(grad) or not wp.isfinite(hess) or hess <= wp.float64(0.0):
            status[i] = 5
        else:
            g[i] = grad
            p[i] = -grad / hess
            e[i] = energy(x[i], target[i])
            alpha[i] = wp.float64(1.0)
            if p[i] < wp.float64(0.0):
                alpha[i] = wp.min(wp.float64(1.0), -wp.float64(0.9) * x[i] / p[i])
            trials[i] = 0
            search[i] = 1


@wp.kernel
def search_trial(
    x: wp.array[wp.float64],
    target: wp.array[wp.float64],
    status: wp.array[wp.int32],
    iterations: wp.array[wp.int32],
    p: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    g: wp.array[wp.float64],
    e: wp.array[wp.float64],
    search: wp.array[wp.int32],
    trials: wp.array[wp.int32],
    total_trials: wp.array[wp.int32],
    max_search: int,
):
    i = wp.tid()
    if search[i] != 0:
        if trials[i] >= max_search:
            status[i] = 3
            search[i] = 0
        else:
            candidate = x[i] + alpha[i] * p[i]
            trial_energy = energy(candidate, target[i])
            trials[i] += 1
            total_trials[i] += 1
            if (
                candidate > wp.float64(0.0)
                and wp.isfinite(trial_energy)
                and trial_energy <= e[i] + wp.float64(1.0e-4) * alpha[i] * g[i] * p[i]
            ):
                x[i] = candidate
                iterations[i] += 1
                search[i] = 0
            else:
                alpha[i] *= wp.float64(0.5)


@wp.kernel
def reduce_condition(values: wp.array[wp.int32], expected: int, condition: wp.array[wp.int32]):
    i = wp.tid()
    if values[i] == expected:
        wp.atomic_max(condition, 0, 1)


@wp.kernel
def commit(
    start: wp.array[wp.float64], x: wp.array[wp.float64], status: wp.array[wp.int32], output: wp.array[wp.float64]
):
    i = wp.tid()
    output[i] = start[i]
    if status[i] == 1:
        output[i] = x[i]


@wp.kernel
def derivative_samples(x: wp.array[wp.float64], output: wp.array2d[wp.float64]):
    i = wp.tid()
    g, h = derivatives(x[i], wp.float64(0.0))
    output[i, 0] = energy(x[i], wp.float64(0.0))
    output[i, 1] = g
    output[i, 2] = h


@wp.kernel
def append_bounded(count: wp.array[wp.int32], buffer: wp.array[wp.int32], overflow: wp.array[wp.int32]):
    i = wp.tid()
    slot = wp.atomic_add(count, 0, 1)
    if slot < buffer.shape[0]:
        buffer[slot] = i
    else:
        wp.atomic_max(overflow, 0, 1)


class Probe:
    """Four independent one-dimensional particle/plane problems."""

    def __init__(self, max_newton=40, max_search=24, device="cpu"):
        self.device = wp.get_device(device)
        self.n = 4
        self.max_newton = max_newton
        self.max_search = max_search
        self.start = wp.array([0.1] * self.n, dtype=wp.float64, device=device)
        self.target = wp.array([-0.2, -0.05, 0.12, 0.3], dtype=wp.float64, device=device)
        for name in ("x", "p", "alpha", "g", "e", "output"):
            setattr(self, name, wp.zeros(self.n, dtype=wp.float64, device=device))
        for name in ("status", "iterations", "search", "trials", "total_trials"):
            setattr(self, name, wp.zeros(self.n, dtype=wp.int32, device=device))
        self.outer = wp.zeros(1, dtype=wp.int32, device=device)
        self.inner = wp.zeros(1, dtype=wp.int32, device=device)
        self.callbacks = 0

    def launch(self, kernel, inputs):
        wp.launch(kernel, dim=self.n, inputs=inputs, device=self.device)

    def condition(self, values, expected, out):
        out.zero_()
        self.launch(reduce_condition, [values, expected, out])

    def step(self, conditional=True):
        self.launch(initialize, [self.start, self.x, self.status, self.iterations, self.total_trials])
        self.condition(self.status, 0, self.outer)

        def trial():
            self.callbacks += 1
            self.launch(
                search_trial,
                [
                    self.x,
                    self.target,
                    self.status,
                    self.iterations,
                    self.p,
                    self.alpha,
                    self.g,
                    self.e,
                    self.search,
                    self.trials,
                    self.total_trials,
                    self.max_search,
                ],
            )
            self.condition(self.search, 1, self.inner)

        def iteration():
            self.callbacks += 1
            self.launch(
                prepare,
                [
                    self.x,
                    self.target,
                    self.status,
                    self.iterations,
                    self.p,
                    self.alpha,
                    self.g,
                    self.e,
                    self.search,
                    self.trials,
                    self.max_newton,
                ],
            )
            self.condition(self.search, 1, self.inner)
            if conditional:
                wp.capture_while(self.inner, trial)
            else:
                for _ in range(self.max_search + 1):
                    trial()
            self.condition(self.status, 0, self.outer)

        if conditional:
            wp.capture_while(self.outer, iteration)
        else:
            for _ in range(self.max_newton + 1):
                iteration()
        self.launch(commit, [self.start, self.x, self.status, self.output])

    def capture(self):
        wp.load_module(device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self.step()
        return capture.graph

    def result(self):
        return {
            name: getattr(self, name).numpy().tolist()
            for name in ("target", "output", "status", "iterations", "total_trials")
        }


class TestProbe(unittest.TestCase):
    def test_eager_feasibility_and_convergence(self):
        p = Probe()
        p.step()
        self.assertTrue(np.all(p.output.numpy() > 0))
        np.testing.assert_array_equal(p.status.numpy(), 1)

    def test_nested_capture_matches_eager_and_replays_new_targets(self):
        p = Probe()
        graph = p.capture()
        calls = p.callbacks
        for targets in ([-0.2, -0.05, 0.12, 0.3], [0.3, -0.2, 0.12, -0.05]):
            p.target.assign(np.array(targets, dtype=np.float64))
            wp.capture_launch(graph)
            ref = Probe()
            ref.target.assign(np.array(targets, dtype=np.float64))
            ref.step()
            np.testing.assert_allclose(p.output.numpy(), ref.output.numpy(), rtol=0, atol=1e-12)
            np.testing.assert_array_equal(p.iterations.numpy(), ref.iterations.numpy())
            np.testing.assert_array_equal(p.status.numpy(), 1)
        self.assertEqual(calls, p.callbacks)

    def test_fixed_schedule_matches_conditional(self):
        a, b = Probe(), Probe()
        a.step()
        b.step(conditional=False)
        np.testing.assert_allclose(a.output.numpy(), b.output.numpy(), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a.status.numpy(), b.status.numpy())

    def test_iteration_limit_rolls_back_during_capture(self):
        p = Probe(max_newton=0)
        graph = p.capture()
        wp.capture_launch(graph)
        np.testing.assert_array_equal(p.status.numpy(), 2)
        np.testing.assert_array_equal(p.output.numpy(), p.start.numpy())

    def test_line_search_limit_rolls_back_during_capture(self):
        p = Probe(max_search=0)
        graph = p.capture()
        wp.capture_launch(graph)
        np.testing.assert_array_equal(p.status.numpy(), 3)
        np.testing.assert_array_equal(p.output.numpy(), p.start.numpy())

    def test_infeasible_start_rejected(self):
        p = Probe()
        p.start.assign(np.array([0, -0.1, 0.1, 0.1], dtype=np.float64))
        wp.capture_launch(p.capture())
        np.testing.assert_array_equal(p.status.numpy(), [4, 4, 1, 1])
        np.testing.assert_array_equal(p.output.numpy()[:2], p.start.numpy()[:2])

    def test_analytic_derivatives(self):
        def f(x):
            t = x * x / 0.04
            return 50 * x * x + (0.1 * -((t - 1) ** 2) * math.log(t) if t < 1 else 0)

        points = np.array([0.025, 0.07, 0.13, 0.19, 0.21], dtype=np.float64)
        x = wp.array(points, dtype=wp.float64, device="cpu")
        out = wp.zeros((len(points), 3), dtype=wp.float64, device="cpu")
        wp.launch(derivative_samples, len(points), inputs=[x, out], device="cpu")
        samples = out.numpy()
        eps = 1e-6
        for i, value in enumerate(points):
            self.assertAlmostEqual(samples[i, 0], f(value), places=12)
            self.assertAlmostEqual(samples[i, 1], (f(value + eps) - f(value - eps)) / (2 * eps), delta=2e-7)
            self.assertAlmostEqual(samples[i, 2], (f(value + eps) - 2 * f(value) + f(value - eps)) / eps**2, delta=3e-3)

    def test_capacity_overflow_counts_all_candidates(self):
        count = wp.zeros(1, dtype=wp.int32, device="cpu")
        buffer = wp.full(3, -1, dtype=wp.int32, device="cpu")
        overflow = wp.zeros(1, dtype=wp.int32, device="cpu")
        with wp.ScopedCapture(device="cpu") as capture:
            count.zero_()
            overflow.zero_()
            wp.launch(append_bounded, 7, inputs=[count, buffer, overflow], device="cpu")
        wp.capture_launch(capture.graph)
        self.assertEqual(count.numpy()[0], 7)
        self.assertEqual(overflow.numpy()[0], 1)
        self.assertTrue(np.all(buffer.numpy() >= 0))


if __name__ == "__main__":
    wp.init()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestProbe)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    p = Probe()
    graph = p.capture()
    wp.capture_launch(graph)
    data = {
        "warp": wp.__version__,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "device": "cpu",
        "cuda_available": wp.is_cuda_available(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "scalar_cases": p.result(),
        "scope": "Scalar point-plane barrier and bounded control-flow probes; no mesh IPC, PCG, or CUDA validation.",
    }
    Path(__file__).with_name("probe_results.json").write_text(json.dumps(data, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
