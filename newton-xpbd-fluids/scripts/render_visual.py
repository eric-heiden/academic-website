"""Render the captured dam-break snapshots into side-by-side filmstrip PNGs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path(__file__).parent / "results"

# blue sequential ramp from the design system (100 -> 700), used as a
# speed ramp: slow water is pale, fast water is deep blue.
WATER = LinearSegmentedColormap.from_list(
    "water",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"],
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"


def panel(ax, q, qd, bounds, vmax, *, point_size):
    lo, hi = bounds
    # view down -y: x horizontal, z vertical; draw far particles first
    order = np.argsort(-q[:, 1])
    x, z = q[order, 0], q[order, 2]
    speed = np.linalg.norm(qd[order], axis=1)
    # subtle depth cue: particles further from camera are slightly lighter
    depth = (q[order, 1] - lo[1]) / max(hi[1] - lo[1], 1e-9)
    alpha = 0.55 + 0.45 * (1.0 - depth)

    ax.scatter(
        x,
        z,
        c=speed,
        cmap=WATER,
        vmin=0.0,
        vmax=vmax,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(0.0, hi[2] * 0.92)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # tank floor + back wall as hairlines
    ax.axhline(0.0, color="#c3c2b7", lw=1.0, zorder=0)
    ax.axvline(lo[0], color="#c3c2b7", lw=1.0, zorder=0)
    ax.axvline(hi[0], color="#c3c2b7", lw=1.0, zorder=0)


def main() -> int:
    meta = json.loads((OUT / "visual_meta.json").read_text())
    data = np.load(OUT / "visual.npz")
    specs = list(meta)
    frames = meta[specs[0]]["frames"]

    # shared speed scale across every panel so color means the same thing
    vmax = float(
        np.percentile(
            np.concatenate(
                [np.linalg.norm(data[f"{s}|{f}|qd"], axis=1) for s in specs for f in frames]
            ),
            99.0,
        )
    )

    nrow, ncol = len(specs), len(frames)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(2.35 * ncol, 2.0 * nrow + 0.5),
        facecolor=SURFACE,
        squeeze=False,
    )
    n = meta[specs[0]]["n"]
    point_size = max(0.35, 900.0 / np.sqrt(n))

    for r, spec in enumerate(specs):
        bounds = meta[spec]["bounds"]
        lo = np.array(bounds[0], dtype=float)
        hi = np.array(bounds[1], dtype=float)
        for c, f in enumerate(frames):
            ax = axes[r][c]
            panel(ax, data[f"{spec}|{f}|q"], data[f"{spec}|{f}|qd"], (lo, hi), vmax, point_size=point_size)
            if r == 0:
                ax.set_title(
                    f"frame {f}   t = {f / 60.0:.2f}s",
                    fontsize=9,
                    color=MUTED,
                    pad=6,
                    fontfamily="DejaVu Sans",
                )
            if c == 0:
                ax.set_ylabel(
                    meta[spec]["label"].replace(" [", "\n["),
                    fontsize=9.5,
                    color=INK,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=14,
                    fontfamily="DejaVu Sans",
                )

    fig.suptitle(
        f"Dam break, {n:,} fluid particles, 8 substeps x 3 iterations - identical scene, identical parameters",
        fontsize=10.5,
        color=INK,
        y=0.995,
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.012,
        f"color = particle speed (pale 0 m/s  ->  deep {vmax:.2f} m/s), shared scale across all panels",
        ha="center",
        fontsize=8.5,
        color=MUTED,
        fontfamily="DejaVu Sans",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    fig.savefig(OUT / "visual_filmstrip.png", dpi=150, facecolor=SURFACE)
    print("wrote", OUT / "visual_filmstrip.png")

    # ---- quantitative divergence: bulk statistics per frame ----
    stats = {}
    for spec in specs:
        rows = []
        for f in frames:
            q = data[f"{spec}|{f}|q"]
            qd = data[f"{spec}|{f}|qd"]
            rows.append(
                {
                    "frame": int(f),
                    "com_x": float(q[:, 0].mean()),
                    "com_z": float(q[:, 2].mean()),
                    "front_x": float(np.percentile(q[:, 0], 99.5)),
                    "height_z": float(np.percentile(q[:, 2], 99.5)),
                    "speed_mean": float(np.linalg.norm(qd, axis=1).mean()),
                }
            )
        stats[spec] = rows
    (OUT / "visual_stats.json").write_text(json.dumps({"stats": stats, "meta": meta}, indent=2))
    print("wrote", OUT / "visual_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
