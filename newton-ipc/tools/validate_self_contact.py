"""Independent NumPy endpoint-distance audit of recorded IPC trajectories.

This exhaustive primitive-pair oracle does not use the solver's BVH, contact
buffers, distance functions, or diagnostics. It is not a continuous mesh CCD
certificate; the separate per-substep panel oracle detects fold crossings.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def point_segment(p, a, b):
    edge = b - a
    denom = dot(edge, edge)
    t = np.divide(dot(p - a, edge), denom, out=np.zeros(len(p)), where=denom > 0)
    delta = p - a - np.clip(t, 0, 1)[:, None] * edge
    return dot(delta, delta)


def point_triangle(p, a, b, c):
    result = np.minimum.reduce(
        [point_segment(p, a, b), point_segment(p, a, c), point_segment(p, b, c)]
    )
    ab, ac, ap = b - a, c - a, p - a
    aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    ad, cd = dot(ab, ap), dot(ac, ap)
    denom = aa * cc - bb * bb
    u = np.divide(cc * ad - bb * cd, denom, out=np.full(len(p), -1.0), where=denom > 0)
    v = np.divide(aa * cd - bb * ad, denom, out=np.full(len(p), -1.0), where=denom > 0)
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    delta = ap - u[:, None] * ab - v[:, None] * ac
    return np.where(inside, np.minimum(result, dot(delta, delta)), result)


def edge_edge(a, b, c, d):
    result = np.minimum.reduce(
        [
            point_segment(a, c, d),
            point_segment(b, c, d),
            point_segment(c, a, b),
            point_segment(d, a, b),
        ]
    )
    e, f, r = b - a, d - c, a - c
    aa, bb, cc, dd, ee = dot(e, e), dot(e, f), dot(f, f), dot(e, r), dot(f, r)
    denom = aa * cc - bb * bb
    s = np.divide(bb * ee - cc * dd, denom, out=np.full(len(a), -1.0), where=denom > 0)
    t = np.divide(aa * ee - bb * dd, denom, out=np.full(len(a), -1.0), where=denom > 0)
    inside = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    delta = r + s[:, None] * e - t[:, None] * f
    return np.where(inside, np.minimum(result, dot(delta, delta)), result)


def audit(path, thickness=0.008, cutoff=0.018):
    data = np.load(path)
    tri, pinned = data["triangles"], data["pinned"]
    edges = np.unique(
        np.sort(np.concatenate((tri[:, :2], tri[:, 1:], tri[:, ::2])), axis=1), axis=0
    )
    pi, ti = np.indices((len(pinned), len(tri))).reshape(2, -1)
    keep = ~(tri[ti] == pi[:, None]).any(axis=1)
    keep &= ~pinned[pi] | ~pinned[tri[ti]].all(axis=1)
    pi, ti = pi[keep], ti[keep]
    ei, ej = np.triu_indices(len(edges), k=1)
    keep = ~(edges[ei, :, None] == edges[ej, None, :]).any(axis=(1, 2))
    keep &= ~pinned[edges[ei]].all(axis=1) | ~pinned[edges[ej]].all(axis=1)
    ei, ej = ei[keep], ej[keep]
    rows = []
    for frame, raw in enumerate(data["positions"]):
        q = raw.astype(np.float64)
        tq = q[tri]
        lo, hi = tq.min(axis=1), tq.max(axis=1)
        near = ((q[pi] >= lo[ti] - cutoff) & (q[pi] <= hi[ti] + cutoff)).all(axis=1)
        a, b = pi[near], ti[near]
        distances = point_triangle(q[a], tq[b, 0], tq[b, 1], tq[b, 2])
        pt = float(np.sqrt(distances.min())) if len(distances) else cutoff
        eq = q[edges]
        lo, hi = eq.min(axis=1), eq.max(axis=1)
        near = ((lo[ei] <= hi[ej] + cutoff) & (lo[ej] <= hi[ei] + cutoff)).all(axis=1)
        a, b = ei[near], ej[near]
        distances = edge_edge(eq[a, 0], eq[a, 1], eq[b, 0], eq[b, 1])
        ee = float(np.sqrt(distances.min())) if len(distances) else cutoff
        minimum = min(pt, ee, cutoff)
        rows.append(
            {
                "time_s": frame * float(data["frame_dt"]),
                "minimum_distance_capped_m": minimum,
            }
        )
        if minimum <= thickness:
            raise AssertionError(f"{path.name} frame {frame}: {minimum} <= {thickness}")
    return {
        "trajectory": path.name,
        "frames": len(rows),
        "minimum_distance_m": min(r["minimum_distance_capped_m"] for r in rows),
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    # Known distances check the independent formulas before auditing data.
    np.testing.assert_allclose(
        point_triangle(
            np.array([[0.2, 0.2, 0.1]]),
            np.array([[0, 0, 0]]),
            np.array([[1, 0, 0]]),
            np.array([[0, 1, 0]]),
        ),
        [0.01],
    )
    np.testing.assert_allclose(
        edge_edge(
            np.array([[0, 0, 0]]),
            np.array([[1, 0, 0]]),
            np.array([[0.2, 0.1, 0]]),
            np.array([[0.8, 0.1, 0]]),
        ),
        [0.01],
    )
    result = {
        "thickness_m": 0.008,
        "distance_cap_m": 0.018,
        "note": "Repeat-0 recorded endpoints at 60 Hz; exhaustive nonincident primitive pairs with at least one movable vertex. Positive endpoint distances alone do not certify full mesh CCD.",
        "runs": [],
    }
    for path in sorted(args.input_dir.glob("ipc-*.npz")):
        row = audit(path)
        result["runs"].append(row)
        print({k: v for k, v in row.items() if k != "samples"}, flush=True)
    (args.input_dir / "distance_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
