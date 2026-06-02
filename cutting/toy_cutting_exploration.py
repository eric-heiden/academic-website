#!/usr/bin/env python3
"""Small NumPy explorations for Newton soft-object cutting.

This is not a solver. It is a report-local sanity check for the data structures
Newton would need to update when a knife plane crosses a tetrahedral soft body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_tet_grid(nx: int, ny: int, nz: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and five-tet-per-cell connectivity for a box grid."""
    xs = np.arange(nx + 1, dtype=np.float64) * spacing
    ys = np.arange(ny + 1, dtype=np.float64) * spacing
    zs = np.arange(nz + 1, dtype=np.float64) * spacing
    vertices = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    def vid(x: int, y: int, z: int) -> int:
        return (x * (ny + 1) + y) * (nz + 1) + z

    tets: list[tuple[int, int, int, int]] = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                v0 = vid(x, y, z)
                v1 = vid(x + 1, y, z)
                v2 = vid(x + 1, y, z + 1)
                v3 = vid(x, y, z + 1)
                v4 = vid(x, y + 1, z)
                v5 = vid(x + 1, y + 1, z)
                v6 = vid(x + 1, y + 1, z + 1)
                v7 = vid(x, y + 1, z + 1)
                if (x & 1) ^ (y & 1) ^ (z & 1):
                    tets.extend([(v0, v1, v4, v3), (v2, v3, v6, v1), (v5, v4, v1, v6), (v7, v6, v3, v4), (v4, v1, v6, v3)])
                else:
                    tets.extend([(v1, v2, v5, v0), (v3, v0, v7, v2), (v4, v7, v0, v5), (v6, v5, v2, v7), (v5, v2, v7, v0)])
    return vertices, np.asarray(tets, dtype=np.int32)


def classify_plane_cut(vertices: np.ndarray, tets: np.ndarray, normal: np.ndarray, offset: float) -> dict[str, float | int]:
    """Classify tetrahedra crossed by a plane n dot x = offset."""
    normal = normal / np.linalg.norm(normal)
    signed = vertices @ normal - offset
    tet_signed = signed[tets]
    cut_mask = (tet_signed.min(axis=1) < 0.0) & (tet_signed.max(axis=1) > 0.0)
    cut_tets = tets[cut_mask]

    tet_edges = np.asarray([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int32)
    cut_edges: set[tuple[int, int]] = set()
    cut_triangles = 0
    for tet, vals in zip(cut_tets, tet_signed[cut_mask], strict=True):
        local_hits = 0
        for a, b in tet_edges:
            sa = vals[a]
            sb = vals[b]
            if sa == 0.0 or sb == 0.0 or sa * sb < 0.0:
                i = int(tet[a])
                j = int(tet[b])
                cut_edges.add((min(i, j), max(i, j)))
                local_hits += 1
        if local_hits == 3:
            cut_triangles += 1
        elif local_hits == 4:
            cut_triangles += 2
        elif local_hits > 0:
            cut_triangles += max(1, local_hits - 2)

    near_band = np.abs(signed) <= 1.5 * np.min(np.linalg.norm(vertices[tets[:, 1]] - vertices[tets[:, 0]], axis=1))
    return {
        "vertices": int(vertices.shape[0]),
        "tets": int(tets.shape[0]),
        "cut_tets": int(cut_mask.sum()),
        "cut_tet_fraction": round(float(cut_mask.mean()), 4),
        "unique_cut_edges": int(len(cut_edges)),
        "estimated_cut_surface_triangles": int(cut_triangles),
        "damage_band_vertices": int(near_band.sum()),
    }


def force_scale(width_m: float, fracture_energy_j_m2: float, shear_strength_pa: float, contact_area_m2: float) -> dict[str, float]:
    """A first-order cutting-force scale: fracture plus interfacial shear."""
    fracture_force = fracture_energy_j_m2 * width_m
    shear_force = shear_strength_pa * contact_area_m2
    return {
        "width_m": width_m,
        "fracture_energy_j_m2": fracture_energy_j_m2,
        "shear_strength_pa": shear_strength_pa,
        "contact_area_m2": contact_area_m2,
        "fracture_force_n": round(fracture_force, 4),
        "interface_force_n": round(shear_force, 4),
        "combined_force_n": round(fracture_force + shear_force, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cutting/toy_cutting_results.json"))
    args = parser.parse_args()

    cases = []
    for cells in [(4, 4, 2), (8, 8, 4), (16, 16, 8)]:
        vertices, tets = make_tet_grid(*cells, spacing=0.01)
        box_center = vertices.mean(axis=0)
        normal = np.asarray([0.9, 0.25, 0.35], dtype=np.float64)
        offset = float(box_center @ (normal / np.linalg.norm(normal)))
        result = classify_plane_cut(vertices, tets, normal, offset)
        result["cells"] = list(cells)
        result["spacing_m"] = 0.01
        cases.append(result)

    outputs = {
        "plane_cut_cases": cases,
        "force_scales": [
            force_scale(width_m=0.03, fracture_energy_j_m2=25.0, shear_strength_pa=1_000.0, contact_area_m2=1.0e-4),
            force_scale(width_m=0.05, fracture_energy_j_m2=100.0, shear_strength_pa=4_000.0, contact_area_m2=2.5e-4),
            force_scale(width_m=0.08, fracture_energy_j_m2=250.0, shear_strength_pa=10_000.0, contact_area_m2=5.0e-4),
        ],
    }
    args.output.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
