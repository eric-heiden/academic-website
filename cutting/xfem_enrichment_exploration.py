#!/usr/bin/env python3
"""Report-local X-FEM enrichment exploration for Newton soft-body cutting.

This is not a production Newton solver. It probes the arrays a Newton/Warp
solver would need for shifted-Heaviside X-FEM cutting on tetrahedral meshes:
cut-element classification, side volume fractions, enriched node allocation,
cut-cell quadrature work, and near-null enrichment DOFs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


TET_EDGES = np.asarray([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int32)
TET_FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))


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
                    tets.extend(
                        [
                            (v0, v1, v4, v3),
                            (v2, v3, v6, v1),
                            (v5, v4, v1, v6),
                            (v7, v6, v3, v4),
                            (v4, v1, v6, v3),
                        ]
                    )
                else:
                    tets.extend(
                        [
                            (v1, v2, v5, v0),
                            (v3, v0, v7, v2),
                            (v4, v7, v0, v5),
                            (v6, v5, v2, v7),
                            (v5, v2, v7, v0),
                        ]
                    )
    return vertices, np.asarray(tets, dtype=np.int32)


def tet_volume(tet_vertices: np.ndarray) -> float:
    """Return the absolute volume of one tetrahedron."""
    a, b, c, d = tet_vertices
    return float(abs(np.linalg.det(np.column_stack((b - a, c - a, d - a)))) / 6.0)


def tet_shape_gradients(tet_vertices: np.ndarray) -> np.ndarray:
    """Return gradients of the four linear tetrahedral shape functions."""
    a, b, c, d = tet_vertices
    dm = np.column_stack((b - a, c - a, d - a))
    inv_dm = np.linalg.inv(dm)
    grads = np.empty((4, 3), dtype=np.float64)
    grads[1] = inv_dm[0]
    grads[2] = inv_dm[1]
    grads[3] = inv_dm[2]
    grads[0] = -(grads[1] + grads[2] + grads[3])
    return grads


def _unique_points(points: Iterable[np.ndarray], tol: float = 1.0e-10) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - other) <= tol for other in unique):
            unique.append(point)
    return unique


def _clip_polygon_to_positive(poly: list[np.ndarray], values: list[float], eps: float = 1.0e-12) -> list[np.ndarray]:
    clipped: list[np.ndarray] = []
    if not poly:
        return clipped

    prev_point = poly[-1]
    prev_value = values[-1]
    prev_inside = prev_value >= -eps

    for point, value in zip(poly, values, strict=True):
        inside = value >= -eps
        if inside != prev_inside:
            denom = prev_value - value
            alpha = 0.0 if abs(denom) <= eps else prev_value / denom
            clipped.append(prev_point + alpha * (point - prev_point))
        if inside:
            clipped.append(point)
        prev_point = point
        prev_value = value
        prev_inside = inside

    return _unique_points(clipped)


def _cap_face(tet_vertices: np.ndarray, signed: np.ndarray, normal: np.ndarray, eps: float = 1.0e-12) -> list[np.ndarray]:
    intersections: list[np.ndarray] = []
    for a, b in TET_EDGES:
        sa = signed[a]
        sb = signed[b]
        if abs(sa) <= eps and abs(sb) <= eps:
            intersections.extend([tet_vertices[a], tet_vertices[b]])
        elif abs(sa) <= eps:
            intersections.append(tet_vertices[a])
        elif abs(sb) <= eps:
            intersections.append(tet_vertices[b])
        elif sa * sb < 0.0:
            alpha = sa / (sa - sb)
            intersections.append(tet_vertices[a] + alpha * (tet_vertices[b] - tet_vertices[a]))

    points = _unique_points(intersections)
    if len(points) < 3:
        return []

    center = np.mean(points, axis=0)
    normal = normal / np.linalg.norm(normal)
    basis_u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(basis_u) < 1.0e-8:
        basis_u = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    basis_u = basis_u / np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)

    return sorted(
        points,
        key=lambda p: math.atan2(float(np.dot(p - center, basis_v)), float(np.dot(p - center, basis_u))),
    )


def _polyhedron_volume(faces: list[list[np.ndarray]]) -> float:
    points = [point for face in faces for point in face]
    if len(points) < 4:
        return 0.0
    center = np.mean(points, axis=0)
    volume = 0.0
    for face in faces:
        if len(face) < 3:
            continue
        first = face[0]
        for i in range(1, len(face) - 1):
            volume += abs(np.linalg.det(np.column_stack((first - center, face[i] - center, face[i + 1] - center)))) / 6.0
    return float(volume)


def clipped_tet_volume_fraction(tet_vertices: np.ndarray, signed: np.ndarray) -> float:
    """Return the fraction of a tetrahedron on the positive side of a plane."""
    total_volume = tet_volume(tet_vertices)
    if total_volume <= 0.0:
        return 0.0
    if np.min(signed) >= 0.0:
        return 1.0
    if np.max(signed) <= 0.0:
        return 0.0

    normal = _signed_plane_normal(tet_vertices, signed)
    faces: list[list[np.ndarray]] = []
    for face_ids in TET_FACES:
        face = [tet_vertices[i] for i in face_ids]
        values = [float(signed[i]) for i in face_ids]
        clipped = _clip_polygon_to_positive(face, values)
        if len(clipped) >= 3:
            faces.append(clipped)

    cap = _cap_face(tet_vertices, signed, normal)
    if len(cap) >= 3:
        faces.append(cap)

    return max(0.0, min(1.0, _polyhedron_volume(faces) / total_volume))


def _signed_plane_normal(tet_vertices: np.ndarray, signed: np.ndarray) -> np.ndarray:
    """Recover a plane normal from nodal signed distances in least-squares form."""
    a = np.column_stack((tet_vertices, np.ones(4)))
    coeffs, *_ = np.linalg.lstsq(a, signed, rcond=None)
    normal = coeffs[:3]
    norm = np.linalg.norm(normal)
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / norm


def summarize_xfem_enrichment(
    vertices: np.ndarray,
    tets: np.ndarray,
    normal: np.ndarray,
    offset: float,
    sliver_fraction: float = 0.02,
    weak_dof_ratio: float = 0.05,
) -> dict[str, float | int]:
    """Summarize shifted-Heaviside X-FEM data needed for one planar cut."""
    normal = normal / np.linalg.norm(normal)
    signed = vertices @ normal - offset
    tet_signed = signed[tets]
    cut_mask = (tet_signed.min(axis=1) < 0.0) & (tet_signed.max(axis=1) > 0.0)
    cut_tets = tets[cut_mask]
    enriched_nodes = np.unique(cut_tets) if len(cut_tets) else np.asarray([], dtype=np.int32)

    enriched_diag = np.zeros(vertices.shape[0], dtype=np.float64)
    standard_diag = np.zeros(vertices.shape[0], dtype=np.float64)
    material_fractions: list[float] = []
    quadrature_subcells = 0
    cut_surface_triangles = 0
    one_three_splits = 0
    two_two_splits = 0

    for tet, values in zip(cut_tets, tet_signed[cut_mask], strict=True):
        tet_vertices = vertices[tet]
        pos_fraction = clipped_tet_volume_fraction(tet_vertices, values)
        min_fraction = min(pos_fraction, 1.0 - pos_fraction)
        material_fractions.append(min_fraction)

        positive_count = int(np.sum(values > 0.0))
        if positive_count in (1, 3):
            quadrature_subcells += 4
            cut_surface_triangles += 1
            one_three_splits += 1
        else:
            quadrature_subcells += 6
            cut_surface_triangles += 2
            two_two_splits += 1

        volume = tet_volume(tet_vertices)
        grads = tet_shape_gradients(tet_vertices)
        node_signs = np.where(values >= 0.0, 1.0, -1.0)
        for local, node in enumerate(tet):
            grad_norm_sqr = float(np.dot(grads[local], grads[local]))
            standard_diag[node] += volume * grad_norm_sqr
            opposite_fraction = (1.0 - pos_fraction) if node_signs[local] > 0.0 else pos_fraction
            enriched_diag[node] += 4.0 * volume * opposite_fraction * grad_norm_sqr

    active_mask = standard_diag[enriched_nodes] > 0.0
    ratios = np.divide(
        enriched_diag[enriched_nodes],
        standard_diag[enriched_nodes],
        out=np.zeros(enriched_nodes.shape[0], dtype=np.float64),
        where=active_mask,
    )
    weak_enriched_nodes = int(np.sum(ratios < weak_dof_ratio)) if len(ratios) else 0
    constrained_vector_dofs = 3 * weak_enriched_nodes
    material_fractions_arr = np.asarray(material_fractions, dtype=np.float64)

    standard_vector_dofs = 3 * vertices.shape[0]
    enriched_vector_dofs = 3 * int(enriched_nodes.shape[0])
    unconstrained_enriched_vector_dofs = enriched_vector_dofs - constrained_vector_dofs
    total_unconstrained_vector_dofs = standard_vector_dofs + unconstrained_enriched_vector_dofs

    return {
        "vertices": int(vertices.shape[0]),
        "tets": int(tets.shape[0]),
        "cut_tets": int(cut_tets.shape[0]),
        "cut_tet_fraction": round(float(cut_tets.shape[0] / max(1, tets.shape[0])), 4),
        "enriched_nodes": int(enriched_nodes.shape[0]),
        "standard_vector_dofs": int(standard_vector_dofs),
        "enriched_vector_dofs": int(enriched_vector_dofs),
        "constrained_enriched_vector_dofs": int(constrained_vector_dofs),
        "unconstrained_enriched_vector_dofs": int(unconstrained_enriched_vector_dofs),
        "total_unconstrained_vector_dofs": int(total_unconstrained_vector_dofs),
        "dof_overhead_ratio": round(float(total_unconstrained_vector_dofs / max(1, standard_vector_dofs)), 4),
        "quadrature_subcells": int(quadrature_subcells),
        "estimated_cut_surface_triangles": int(cut_surface_triangles),
        "one_three_split_tets": int(one_three_splits),
        "two_two_split_tets": int(two_two_splits),
        "min_material_fraction": round(float(material_fractions_arr.min(initial=1.0)), 6),
        "median_material_fraction": round(float(np.median(material_fractions_arr)) if len(material_fractions_arr) else 1.0, 6),
        "sliver_tets": int(np.sum(material_fractions_arr < sliver_fraction)) if len(material_fractions_arr) else 0,
        "weak_enriched_nodes": int(weak_enriched_nodes),
        "weak_dof_ratio": float(weak_dof_ratio),
    }


def generate_results() -> dict[str, object]:
    """Generate reproducible data for the X-FEM section."""
    normal = np.asarray([0.9, 0.25, 0.35], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    cases = []
    for cells in [(4, 4, 2), (8, 8, 4), (16, 16, 8)]:
        vertices, tets = make_tet_grid(*cells, spacing=0.01)
        offset = float(vertices.mean(axis=0) @ normal + 0.0013)
        result = summarize_xfem_enrichment(vertices, tets, normal, offset)
        result["cells"] = list(cells)
        result["spacing_m"] = 0.01
        result["cut_type"] = "center_plane"
        cases.append(result)

    vertices, tets = make_tet_grid(8, 8, 4, spacing=0.01)
    sliver_offset = float(np.min(vertices @ normal) + 0.003)
    sliver_probe = summarize_xfem_enrichment(vertices, tets, normal, sliver_offset)
    sliver_probe["cells"] = [8, 8, 4]
    sliver_probe["spacing_m"] = 0.01
    sliver_probe["cut_type"] = "near_corner_sliver_probe"

    return {
        "method": "shifted Heaviside X-FEM exploration",
        "normal": [round(float(x), 6) for x in normal],
        "center_plane_cases": cases,
        "sliver_probe": sliver_probe,
        "newton_solver_arrays": {
            "xfem_node_indices": "int32[enriched_node_count]",
            "xfem_node_offsets": "int32[particle_count + 1] or dense particle->enrichment map",
            "xfem_q": "vec3[enriched_node_count] enriched displacement DOFs",
            "xfem_qd": "vec3[enriched_node_count] enriched velocity DOFs",
            "xfem_tet_flags": "int32[tet_count] active, cut, tip, constrained flags",
            "xfem_tet_side_fraction": "float32[tet_count, 2] positive/negative material volumes",
            "xfem_tet_quad": "fixed-capacity quadrature descriptors for cut tets",
            "xfem_cut_surface": "triangle mesh or halfspace/boolean cut geometry for contact and rendering",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cutting/xfem_enrichment_results.json"))
    args = parser.parse_args()

    outputs = generate_results()
    args.output.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
