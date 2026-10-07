"""Behavioral probes for the RSS integration report, outside either project.

Run: uv run --project /path/to/RoboSimStudio python this_file.py -v
These are characterization tests, not claims of solver accuracy.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import robosimstudio as rss
import warp as wp


class TestSceneContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = rss.Scene(cell="bench_075", name="study-contract")
        cls.cube = cls.scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
        cls.sim = cls.scene.build()

    def test_work_surface_frame_and_world_readback(self):
        """Distinguish authored bench-relative coordinates from returned world coordinates."""
        self.sim.reset()
        np.testing.assert_allclose(self.cube.position, [0.6, 0, 0.95], atol=1e-5)

    def test_settle_and_reset(self):
        """Settle a free box onto the bench and restore its initial state."""
        self.sim.reset()
        initial = np.array(self.cube.position)
        self.sim.run(180).sync()
        self.assertTrue(self.sim.state_is_finite())
        self.assertAlmostEqual(float(self.cube.position[2]), 0.775, delta=0.01)
        self.sim.reset()
        np.testing.assert_allclose(self.cube.position, initial, atol=1e-5)

    def test_duplicate_object_is_rejected(self):
        """Reject a second rigid object with the same stable name."""
        with self.assertRaises(ValueError):
            self.scene.add(rss.sphere(name="cube", radius=0.02))

    def test_one_robot_slot_is_replaced(self):
        """Characterize the single-robot slot without claiming multi-robot support."""
        scene = rss.Scene(cell="void")
        scene.add(rss.robot("panda"))
        second = rss.robot("yam_i2rt")
        scene.add(second)
        self.assertIs(scene._robot, second)  # Explicit inspection of current implementation.

    def test_benchmark_tier_is_unavailable(self):
        """Distinguish shipped studio functionality from the withheld benchmark tier."""
        from robosimstudio_studio.seams import NotInThisRelease, benchmark

        with self.assertRaises(NotInThisRelease):
            benchmark("study task loading")


class TestForeignForceContract(unittest.TestCase):
    def test_feedback_cap_changes_transferred_impulse(self):
        """Measure force shaping on a 63 g rigid body receiving a 100 N reaction."""
        from robosimstudio.engines.feedback import WrenchAccumulator

        model = SimpleNamespace(
            body_count=1,
            body_com=wp.array([[0, 0, 0]], dtype=wp.vec3, device="cpu"),
            body_mass=wp.array([0.063], dtype=float, device="cpu"),
        )
        acc = WrenchAccumulator(model)
        dt = 1 / 60
        acc.begin()
        acc.add(0, np.array([100, 0, 0]), np.zeros(3), np.array([0, 0, 0, 0, 0, 0, 1]))
        delivered = acc.shape(dt).copy()
        print(f"\nFeedback probe: requested impulse={100 * dt:.6f} N s; delivered={delivered[0, 0] * dt:.6f} N s")
        self.assertLess(float(delivered[0, 0]), 2.0)
        acc.begin()
        np.testing.assert_array_equal(acc.shape(dt), np.zeros((1, 6)))


if __name__ == "__main__":
    unittest.main()
