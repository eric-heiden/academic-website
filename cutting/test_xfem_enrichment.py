#!/usr/bin/env python3
"""Small self-tests for the report-local X-FEM exploration."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xfem_enrichment_exploration import (
    clipped_tet_volume_fraction,
    make_tet_grid,
    summarize_xfem_enrichment,
)


class XFEMEnrichmentExplorationTest(unittest.TestCase):
    def test_clipped_tet_volume_fraction_splits_unit_tet_in_half(self) -> None:
        tet = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        signed = tet[:, 0] + tet[:, 1] + tet[:, 2] - (0.5 ** (1.0 / 3.0))

        fraction = clipped_tet_volume_fraction(tet, signed)

        self.assertAlmostEqual(fraction, 0.5, places=12)

    def test_xfem_summary_allocates_local_enrichment_dofs_only_on_cut_support(self) -> None:
        vertices, tets = make_tet_grid(4, 4, 2, spacing=0.01)
        normal = np.array([0.9, 0.25, 0.35], dtype=np.float64)
        offset = float(vertices.mean(axis=0) @ (normal / np.linalg.norm(normal)))

        result = summarize_xfem_enrichment(vertices, tets, normal, offset)

        self.assertGreater(result["cut_tets"], 0)
        self.assertGreater(result["enriched_nodes"], 0)
        self.assertLess(result["enriched_nodes"], result["vertices"])
        self.assertEqual(result["enriched_vector_dofs"], 3 * result["enriched_nodes"])
        self.assertGreater(result["quadrature_subcells"], result["cut_tets"])


if __name__ == "__main__":
    unittest.main()
