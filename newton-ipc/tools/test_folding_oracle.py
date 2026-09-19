"""Independent geometric tests for the folding comparison's fixed-panel oracle."""

import unittest

import numpy as np
from benchmark_folding import BASE_HEIGHT, crossed_panel, intersected_panel


class TestFoldingOracle(unittest.TestCase):
    def test_crossing_inside_panel(self):
        """Detect a vertex crossing the panel interior in one linear step."""
        before = np.array([[0.1, 0.2, BASE_HEIGHT + 0.1]])
        after = before - [0.0, 0.0, 0.2]
        self.assertEqual(crossed_panel(before, after, np.array([True])), 1)
        self.assertEqual(crossed_panel(before, after, np.array([False])), 0)

    def test_crossing_outside_panel(self):
        """Allow downward motion outside the finite panel boundary."""
        before = np.array([[-0.1, 0.2, BASE_HEIGHT + 0.1]])
        after = before - [0.0, 0.0, 0.2]
        self.assertEqual(crossed_panel(before, after, np.array([True])), 0)

    def test_intersection_without_interior_vertices(self):
        """Detect a triangle slicing the panel when all its vertices are outside."""
        q = np.array(
            [
                [-0.2, 0.2, BASE_HEIGHT - 0.1],
                [0.5, -0.2, BASE_HEIGHT + 0.1],
                [0.5, 0.6, BASE_HEIGHT + 0.1],
            ]
        )
        self.assertEqual(intersected_panel(q, np.array([[0, 1, 2]])), 1)

    def test_no_cut_for_parallel_triangle(self):
        """Do not report separated or coplanar triangles as transverse cuts."""
        q = np.array(
            [[0.1, 0.1, BASE_HEIGHT], [0.2, 0.1, BASE_HEIGHT], [0.1, 0.2, BASE_HEIGHT]]
        )
        for dz in (-0.1, 0.0, 0.1):
            self.assertEqual(
                intersected_panel(q + [0, 0, dz], np.array([[0, 1, 2]])), 0
            )

    def test_triangle_plane_cut_outside_rectangle(self):
        """Reject plane intersections beyond the finite cloth panel."""
        q = np.array(
            [
                [-0.5, 0.2, BASE_HEIGHT - 0.1],
                [-0.2, 0.1, BASE_HEIGHT + 0.1],
                [-0.2, 0.3, BASE_HEIGHT + 0.1],
            ]
        )
        self.assertEqual(intersected_panel(q, np.array([[0, 1, 2]])), 0)


if __name__ == "__main__":
    unittest.main()
